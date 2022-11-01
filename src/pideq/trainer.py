from contextlib import nullcontext
import logging
from multiprocessing import Pool
import random

from abc import ABC, abstractmethod
from pathlib import Path
from time import time

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import wandb

from torchdiffeq import odeint
from pyDOE import lhs
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import grad
from pideq.deq.model import DEQ

from pideq.four_tanks import four_tanks
from pideq.net import PIDEQ, PINC, PINN


def f(y, u, mu=1.):
    return torch.stack((
        y[...,1],
        mu * (1 - y[...,0]**2) * y[...,1] - y[...,0] + u[...,0]
    ),dim=-1)

def timeit(fun):
    def fun_(*args, **kwargs):
        start_time = time()
        f_ret = fun(*args, **kwargs)
        end_time = time()

        return end_time - start_time, f_ret

    return fun_

def _run_get_a(args):
    prob, A0, i = args

    prob.parameters()[0].value = A0[i]  # a0
    try:
        prob.solve(verbose=False, abstol=1e-6, feastol=1e-6)
        # prob.solve(verbose=False, solver='SCS')

        return prob.variables()[0].value  # a
    except cp.SolverError:
        return A0[i]

class Trainer(ABC):
    """Generic trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, epochs=5, lr= 0.1,
                 optimizer: str = 'Adam', optimizer_params: dict = dict(),
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, mixed_precision=True,
                 device=None, wandb_project=None, wandb_group=None,
                 logger=None, checkpoint_every=50, random_seed=42,
                 max_loss=None) -> None:
        self._is_initalized = False

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self._dtype = next(self.net.parameters()).dtype

        self.mixed_precision = mixed_precision

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.checkpoint_every = checkpoint_every

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

        self._log_to_wandb = False if wandb_project is None else True
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group

        self.max_loss = max_loss

    def _load_optim(self, state_dict=None):
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )

        if state_dict is not None:
            self._optim.load_state_dict(state_dict)

    @classmethod
    def load_trainer(cls, net: nn.Module, run_id: str, wandb_project=None,
                     logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and create the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project=wandb_project,
            entity="brunompac",
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            device=wandb.config['device'],
            logger=logger,
            wandb_project=wandb_project,
            random_seed=wandb.config['random_seed'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        self._load_optim(checkpoint['optimizer_state_dict'])

        # load scheduler
        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self._loss_func = eval(f"nn.{self.loss_func}()")

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.prepare_data()

        self._is_initalized = True

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        self._load_optim()

        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        if self._log_to_wandb:
            self._add_to_wandb_config({
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "mixed_precision": self.mixed_precision,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            })

            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self._loss_func = eval(f"nn.{self.loss_func}()")

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def _add_to_wandb_config(self, d: dict):
        if not hasattr(self, '_wandb_config'):
            self._wandb_config = dict()

        for k, v in d.items():
            self._wandb_config[k] = v

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    @abstractmethod
    def prepare_data(self):
        """Must populate `self.data` and `self.val_data`.
        """

    @staticmethod
    def _add_data_to_log(data: dict, prefix: str, data_to_log=dict()):
        for k, v in data.items():
            if k != 'all':
                data_to_log[prefix+k] = v
        
        return data_to_log

    def _run_epoch(self):
        # train
        train_time, (train_losses, train_times) = timeit(self.train_pass)()

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_losses['all']}")

        # validation
        val_time, (val_losses, val_times) = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"Validation loss = {val_losses['all']}")

        data_to_log = {
            "train_loss": train_losses['all'],
            "val_loss": val_losses['all'],
            "train_time": train_time,
            "val_time": val_time,
        }
        self._add_data_to_log(train_losses, 'train_loss_', data_to_log)
        self._add_data_to_log(val_losses, 'val_loss_', data_to_log)
        self._add_data_to_log(train_times, 'train_time_', data_to_log)
        self._add_data_to_log(val_times, 'val_time_', data_to_log)

        val_score = val_losses['all']  # defines best model

        return data_to_log, val_score

    def run(self) -> nn.Module:
        if not self._is_initalized:
            self.setup_training()

        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            if val_score < self.best_val:
                if self._log_to_wandb:
                    self.l.info(f"Saving best model")
                    self.save_model(name='model_best')

                self.best_val = val_score

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            if self.max_loss is not None:
                if val_score > self.max_loss:
                    break

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('Training finished!')

        return self.net

    def train_pass(self):
        train_loss = 0

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in self.data:
                X = X.to(self.device)
                y = y.to(self.device)

                self._optim.zero_grad()

                with self.autocast_if_mp():
                    forward_time_, y_hat = timeit(self.net)(X)
                    forward_time += forward_time_

                    loss_time_, loss = timeit(self._loss_func)(y_hat, y)
                    loss_time += loss_time_

                if self.mixed_precision:
                    backward_time_, _  = timeit(self._scaler.scale(loss).backward)()
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    backward_time_, _  = timeit(loss.backwad)()
                    self._optim.step()
                backward_time += backward_time_

                train_loss += loss.item() * len(y)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # scale to data size
        train_loss = train_loss / len(self.data)

        losses = {
            'all': train_loss,
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times

    def validation_pass(self):
        val_loss = 0

        forward_time = 0
        loss_time = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self.val_data:
                X = X.to(self.device)
                y = y.to(self.device)

                with self.autocast_if_mp():
                    forward_time_, y_hat = timeit(self.net)(X)
                    forward_time += forward_time_

                    loss_time_, loss = timeit(self._loss_func)(y_hat, y)
                    loss_time += loss_time_

                val_loss += loss.item() * len(y)  # scales to data size

        # scale to data size
        len_data = len(self.val_data)
        val_loss = val_loss / len_data
        losses = {
            'all': val_loss
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
        }

        return losses, times

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath

class DEQTrainer(Trainer):
    """Trainer for synthetic data using a DEQ."""
    def __init__(self, net: DEQ, N=200, epochs=5000, lr=1e-3, optimizer: str = 'Adam',
                 optimizer_params: dict = dict(), A_oo_lambda=0., A_oo_n=4,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None, kappa=-1,
                 lr_scheduler_params: dict = None, project_grad=False, eps=1e-3,
                 device=None, wandb_project="pideq-ghaoui", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None, jac_lambda=0,
                 max_loss=None, bcd_every=-1):
        self.N = int(N)    # number of points
        self.A_oo_lambda = A_oo_lambda
        self.jac_lambda = jac_lambda
        self.A_oo_n = A_oo_n
        self.kappa = kappa
        self.project_grad = project_grad
        self.eps = eps
        self.bcd_every = bcd_every

        if self.jac_lambda > 0:
            assert net.compute_jac_loss

        self._add_to_wandb_config({
            'N': self.N,
            'A_oo_lambda': self.A_oo_lambda,
            'jac_lambda': self.jac_lambda,
            'A_oo_n': self.A_oo_n,
            'kappa': self.kappa,
            'project_grad': self.project_grad,
            'eps': self.eps,
            'bcd_every': self.bcd_every,
        })

        # initial state
        super().__init__(net, epochs, lr, optimizer, optimizer_params,
                         loss_func, lr_scheduler, lr_scheduler_params,
                         False, device, wandb_project,
                         wandb_group, logger, checkpoint_every, random_seed,
                         max_loss)

        self.data = None
        self.val_data = None

        if self.kappa > 0:
            # initialize gradient projection optimization problem
            # decomposed for each row of A (including bias, assuming x = (x,1))
            a0 = cp.Parameter(self.net.A.weight.shape[0] + 1, complex=False)
            a = cp.Variable(self.net.A.weight.shape[0] + 1, complex=False)
            constraint = [cp.norm(a, p=1) <= self.kappa - 1e-3,]

            self._pgd_prob = cp.Problem(cp.Minimize(cp.norm(a - a0, p=2)), constraint)

            # multiprocessing pool for parallelization of gradient projection
            self._pgd_pool = Pool()

    def _load_optim(self, state_dict=None):
        if self.bcd_every > 0:
            Optimizer = eval(f"torch.optim.{self.optimizer}")
            self._optim = Optimizer(
                [self.net.A.weight, self.net.A.bias, self.net.B.weight, self.net.B.bias],
                lr=self.lr,
                **self.optimizer_params
            )
            if state_dict is not None:
                self._optim.load_state_dict(state_dict)

            self._optim_CD = torch.optim.LBFGS(
                [self.net.C.weight, self.net.C.bias, self.net.D.weight, self.net.D.bias],
            )
        else:
            return super()._load_optim(state_dict)

    def _run_epoch(self):
        if self._e % self.bcd_every == 1:
            self.train_pass_CD()

        return super()._run_epoch()

    def prepare_data(self):
        self.f_true = lambda u: 5 * np.cos(np.pi * u) * np.exp(-np.abs(u) / 2)
        def f(u):
            w = 2 * np.random.rand(*u.shape) - 1
            return self.f_true(u) + w
        self.f = f

        X = 10 * torch.rand(self.N, 1) - 5
        X = X.float()
        Y = torch.from_numpy(f(X.numpy())).float()
        X.requires_grad_()

        X = X.to(self.device)
        Y = Y.to(self.device)

        self.data = X, Y

        X = 10 * torch.rand(self.N, 1) - 5
        X = X.float()
        Y = torch.from_numpy(f(X.numpy())).float()

        X = X.to(self.device)
        Y = Y.to(self.device)

        self.val_data = X, Y

        # import matplotlib.pyplot as plt
        # plt.scatter(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), s=2)
        # plt.grid()
        # plt.savefig('test.png')

    def project_matrix(self, A0):
        return_tensor = isinstance(A0, torch.Tensor)

        if return_tensor:
            A0_ = A0.detach().cpu().numpy()
        else:
            A0_ = A0

        A_rows = self._pgd_pool.map(
            _run_get_a,
            [(self._pgd_prob,A0_,i) for i in range(A0_.shape[0])]
        )

        A = np.vstack(A_rows)

        if return_tensor:
            A = torch.from_numpy(A).to(A0)

        return A

    def project_A(self, A):
        A0 = np.hstack([  # includes bias for x = (x,1)
            A.weight.data.detach().cpu().numpy(),
            A.bias.data.detach().cpu().numpy()[...,None],
        ])

        A_ = self.project_matrix(A0)

        A_weight = torch.from_numpy(A_[:,:-1]).to(A.weight.data)
        A_bias = torch.from_numpy(A_[:,-1]).to(A.bias.data)

        A.weight.data.copy_(A_weight)
        A.bias.data.copy_(A_bias)

    def train_pass_CD(self):
        X, y = self.data

        self.net.train()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim_CD.zero_grad()

                y_hat = 6 * self.net(X)

                return self._loss_func(y, y_hat)

            self._optim_CD.step(closure)

    def train_pass(self):
        X, y = self.data

        self.net.train()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()

            if self.jac_lambda > 0:
                forward_time, (y_hat, jac_loss) = timeit(self.net)(X)
            else:
                forward_time, y_hat = timeit(self.net)(X)

            y_hat = 6 * y_hat

            loss_time, loss = timeit(self._loss_func)(y, y_hat)

            try:
                A_oo = torch.linalg.norm(self.net.A.weight, ord=torch.inf)
            except AttributeError:
                A_oo = torch.Tensor([0])

            if self.A_oo_lambda > 0:
                loss += self.A_oo_lambda * (A_oo ** self.A_oo_n - 1)

            if self.jac_lambda > 0:
                loss += self.jac_lambda * jac_loss

            backward_time, _  = timeit(loss.backward)()

            grad_proj_time = 0
            if self.project_grad:  # project gradient of A
                with torch.no_grad():
                    # project A gradients
                    A = torch.hstack([self.net.A.weight.data, self.net.A.bias.data.unsqueeze(-1)])
                    A_grad = torch.hstack([self.net.A.weight.grad, self.net.A.bias.grad.unsqueeze(-1)])
                    
                    # project gradient to be tangent to the constrained space
                    A0 = A - self.eps * A_grad
                    if torch.linalg.norm(A0, ord=torch.inf) > self.kappa:
                        start_time = time()
                        A_grad = (A - self.project_matrix(A0)) / self.eps

                        self.net.A.weight.grad.copy_(A_grad[...,:-1].to(self.net.A.weight.grad))
                        self.net.A.bias.grad.copy_(A_grad[...,-1].to(self.net.A.bias.grad))
                        grad_proj_time = time() - start_time

            self._optim.step()

            train_loss = loss.item()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # project A matrix
        A_proj_time = 0
        if A_oo.item() > self.kappa and self.kappa > 0:
            if self.net.A.weight.requires_grad:
                A_proj_time, _ = timeit(self.project_A)(self.net.A)
                A_oo = torch.linalg.norm(self.net.A.weight, ord=torch.inf)

        # scale to data size
        losses = {
            'all': train_loss,
            'A_oo': A_oo.item(),
        }
        if self.jac_lambda > 0:
            losses['jac_loss'] = jac_loss.item()
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
            'grad_proj': grad_proj_time,
            'A_proj': A_proj_time,
        }

        return losses, times

    def validation_pass(self):
        X, y = self.data

        self.net.eval()
        with torch.set_grad_enabled(False):
            forward_time, y_hat = timeit(self.net)(X)
            y_hat = 6 * y_hat

            loss_time, loss = timeit(self._loss_func)(y, y_hat)

            val_loss = loss.item()

        # scale to data size
        len_data = len(self.val_data)
        val_loss = val_loss / len_data
        losses = {
            'all': val_loss
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
        }

        return losses, times
