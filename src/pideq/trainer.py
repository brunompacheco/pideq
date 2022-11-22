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
                 optimizer: str = 'Adam', optimizer_params: dict = None,
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

        # load optimizer
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])

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

        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )

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

class PINNTrainer(Trainer):
    """Trainer for the sine ODE using a Physics-Informed NN."""
    def __init__(self, net: PINN, Nf=1e2, val_dt=1e-3 * np.pi/2,
                 epochs=5, lr=1e-3, optimizer: str = 'Adam',
                 optimizer_params: dict = None, lamb=0.1,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 mixed_precision=False, lr_scheduler_params: dict = None,
                 device=None, wandb_project="pideq-nls", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None,
                 max_loss=None):
        # initial state
        self.h0_func = lambda x: 4 / (np.exp(x) + np.exp(-x))
        self.h_bounds = (-5, 5)

        self._add_to_wandb_config({
            'T': net.T,
        })

        super().__init__(net, epochs, lr, optimizer, optimizer_params,
                         loss_func, lr_scheduler, lr_scheduler_params,
                         mixed_precision, device, wandb_project,
                         wandb_group, logger, checkpoint_every, random_seed,
                         max_loss)

        # self.N0 = int(N0)    # number of initial condition points
        # self.Nb = int(Nb)    # number of boundary condition points
        self.Nf = int(Nf)    # number of collocation points

        self.val_dt = val_dt
        self.T = net.T

        if lamb is None:
            self.lamb = 1 / self.Nf
        else:
            self.lamb = lamb

        self.data = None
        self.val_data = None

    def prepare_data(self):
        # 
        t = np.arange(0, self.T, .1)
        y_val = np.vstack((np.sin(t), np.cos(t))).T

        X_val = torch.Tensor(t[:,None]).to(self.device).type(self._dtype)
        Y_val = torch.Tensor(y_val).to(self.device).type(self._dtype)

        self.val_data = (X_val, Y_val)

        X0 = torch.zeros(1,1).to(self.device).type(self._dtype)
        Y0 = torch.zeros(1,2).to(self.device).type(self._dtype)
        Y0[0,1] = 1.

        Xf = torch.rand(self.Nf,1) * self.T
        Xf = Xf.to(self.device).type(self._dtype)

        self.data = ((X0,Y0), Xf)

    def get_jacobian(self, Y, x_):
        dys = list()
        for i in range(Y.shape[-1]):
            dys.append(grad(Y[:,i].sum(), x_, create_graph=True)[0])
        return torch.stack(dys, dim=-1).squeeze(1)

    def get_loss_f(self, y, t):
        y_t_pred = self.get_jacobian(y, t)

        y_t = torch.stack((y[:,1], -y[:,0]), -1)

        ode = y_t_pred - y_t

        return self._loss_func(ode, torch.zeros_like(ode))

    def train_pass(self):
        self.net.train()

        # training data
        (X0, Y0), Xf = self.data
        Xf.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    Y0_pred = self.net(X0)

                    global loss_0
                    loss_0 = self._loss_func(Y0_pred, Y0.to(Y0_pred))

                    global forward_time
                    forward_time, Yf_pred = timeit(self.net)(Xf)

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(Yf_pred, Xf)

                    global loss
                    loss = loss_0 + loss_f

                    if loss.requires_grad:
                        global backward_time

                        if self.mixed_precision:
                            backward_time, _ = timeit(self._scaler.scale(loss).backward)()
                        else:
                            backward_time, _ = timeit(loss.backward)()

                return loss

            if self.optimizer == 'LBFGS':
                self._optim.step(closure)
            else:
                closure()

                if self.mixed_precision:
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    self._optim.step()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        losses = {
            'all': loss.item(),
            '0': loss_0.item(),
            'f': loss_f.item(),
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }
        return losses, times

    def validation_pass(self):
        self.net.eval()

        X, Y = self.val_data

        with torch.set_grad_enabled(False):
            self._optim.zero_grad()
            forward_time, y_pred = timeit(self.net)(X)

        val_loss = self._loss_func(y_pred, Y)

        losses = {
            'all': val_loss.item(),
        }
        times = {
            'forward': forward_time,
        }

        return losses, times

class PIDEQTrainer(PINNTrainer):
    def __init__(self, net: PIDEQ, Nf=1e2, jac_lambda=0., A_oo_lambda=0., A_oo_n=4,
                 val_dt=0.001 * np.pi / 2, epochs=5, lr=0.001, kappa=-1,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 lamb=0.1, loss_func: str = 'MSELoss', eps=1e-3,
                 lr_scheduler: str = None, mixed_precision=False,
                 lr_scheduler_params: dict = None, device=None, project_grad=False,
                 wandb_project="pideq-nls", wandb_group=None, logger=None,
                 checkpoint_every=1000, random_seed=None, max_loss=None):
        self.A_oo_lambda = A_oo_lambda
        self.jac_lambda = jac_lambda
        self.A_oo_n = A_oo_n
        self.kappa = kappa
        self.project_grad = project_grad
        self.eps = eps

        if self.jac_lambda > 0:
            assert net.implicit.compute_jac_loss
        else:
            assert not net.implicit.compute_jac_loss

        self._add_to_wandb_config({
            'n_states': net.n_states,
            'solver_max_nfe': net.implicit.solver_kwargs['threshold'],
            'solver_eps': net.implicit.solver_kwargs['eps'],
            'solver': net.implicit.solver.__name__,
            'A_oo_lambda': self.A_oo_lambda,
            'jac_lambda': self.jac_lambda,
            'A_oo_n': self.A_oo_n,
            'kappa': self.kappa,
            'project_grad': self.project_grad,
            'eps': self.eps,
        })

        super().__init__(net, Nf, val_dt, epochs, lr, optimizer,
                         optimizer_params, lamb, loss_func, lr_scheduler,
                         mixed_precision, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed, max_loss)

        if self.kappa > 0:
            # initialize gradient projection optimization problem
            # decomposed for each row of A (including bias, assuming x = (x,1))
            a0 = cp.Parameter(self.net.implicit.A.weight.shape[0] + 1, complex=False)
            a = cp.Variable(self.net.implicit.A.weight.shape[0] + 1, complex=False)
            constraint = [cp.norm(a, p=1) <= self.kappa - 1e-3,]

            self._pgd_prob = cp.Problem(cp.Minimize(cp.norm(a - a0, p=2)), constraint)

            # multiprocessing pool for parallelization of gradient projection
            self._pgd_pool = Pool()

    def _load_optim(self, state_dict=None):
        if self.bcd_every > 0:
            Optimizer = eval(f"torch.optim.{self.optimizer}")
            self._optim = Optimizer(
                [self.net.implicit.A.weight, self.net.implicit.A.bias, self.net.implicit.B.weight, self.net.implicit.B.bias],
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

    def train_pass(self):
        self.net.train()

        # training data
        (X0, Y0), Xf = self.data
        Xf.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    if self.jac_lambda > 0:
                        Y0_pred, jac_loss_0 = self.net(X0)
                    else:
                        Y0_pred = self.net(X0)

                    global loss_0
                    loss_0 = self._loss_func(Y0_pred, Y0.to(Y0_pred))

                    global forward_time
                    if self.jac_lambda > 0:
                        forward_time, (Yf_pred, jac_loss_f) = timeit(self.net)(Xf)
                        forward_time -= self.net.implicit.jac_loss_time
                    else:
                        forward_time, Yf_pred = timeit(self.net)(Xf)

                    global forward_nfe
                    forward_nfe = self.net.implicit.latest_nfe

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(Yf_pred, Xf)

                    global backward_nfe
                    backward_nfe = self.net.implicit.latest_backward_nfe

                    global loss
                    loss = (loss_0 + loss_f) / (1 + self.jac_lambda)

                    if self.jac_lambda > 0:
                        global jac_loss
                        jac_loss = (jac_loss_0 + self.Nf * jac_loss_f) \
                                    / (self.Nf)
                        loss += self.jac_lambda * jac_loss

                    global A_oo
                    try:
                        A_oo = torch.linalg.norm(self.net.implicit.A.weight, ord=torch.inf)
                    except AttributeError:
                        A_oo = torch.Tensor([0])

                    if self.A_oo_lambda > 0:
                        loss += self.A_oo_lambda * (A_oo ** self.A_oo_n - 1)

                    if loss.requires_grad:
                        global backward_time

                        if self.mixed_precision:
                            backward_time, _ = timeit(self._scaler.scale(loss).backward)()
                        else:
                            backward_time, _ = timeit(loss.backward)()

                global grad_proj_time
                grad_proj_time = 0
                if self.project_grad:  # project gradient of A
                    with torch.no_grad():
                        # project A gradients
                        A = torch.hstack([self.net.implicit.A.weight.data, self.net.implicit.A.bias.data.unsqueeze(-1)])
                        A_grad = torch.hstack([self.net.implicit.A.weight.grad, self.net.implicit.A.bias.grad.unsqueeze(-1)])
                        
                        # project gradient to be tangent to the constrained space
                        A0 = A - self.eps * A_grad
                        if torch.linalg.norm(A0, ord=torch.inf) > self.kappa:
                            start_time = time()
                            A_grad = (A - self.project_matrix(A0)) / self.eps

                            self.net.implicit.A.weight.grad.copy_(A_grad[...,:-1].to(self.net.implicit.A.weight.grad))
                            self.net.implicit.A.bias.grad.copy_(A_grad[...,-1].to(self.net.implicit.A.bias.grad))
                            grad_proj_time = time() - start_time

                return loss

            if self.optimizer == 'LBFGS':
                self._optim.step(closure)
            else:
                closure()

                if self.mixed_precision:
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    self._optim.step()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # project A matrix
        A_proj_time = 0
        global A_oo
        if A_oo.item() > self.kappa and self.kappa > 0:
            if self.net.implicit.A.weight.requires_grad:
                A_proj_time, _ = timeit(self.project_A)(self.net.implicit.A)
                A_oo = torch.linalg.norm(self.net.implicit.A.weight, ord=torch.inf)

        losses = {
            'all': loss.item(),
            '0': loss_0.item(),
            'f': loss_f.item(),
            'fwd_nfe': forward_nfe,
            'bwd_nfe': backward_nfe,
            'A_oo': A_oo.item(),
        }
        if self.jac_lambda > 0:
            losses['jac'] = jac_loss.item()
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
            'grad_proj': grad_proj_time,
            'A_proj': A_proj_time,
        }

        return losses, times

    def validation_pass(self):
        losses, times = super().validation_pass()

        losses['fwd_nfe'] = self.net.implicit.latest_nfe
        losses['bwd_nfe'] = self.net.implicit.latest_backward_nfe

        return losses, times

class PIDEQTrainerV2(PIDEQTrainer):
    def __init__(self, net: PIDEQ, N0=50, Nb=50, Nf=20000, kappa=1.0,
                 val_dt=0.001 * np.pi / 2, epochs=5, lr=0.001,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 lamb=0.1, loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 mixed_precision=False, lr_scheduler_params: dict = None,
                 device=None, wandb_project="pideq-nls", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None,
                 max_loss=5):
        self.kappa = kappa

        self._add_to_wandb_config({
            'kappa': self.kappa,
        })

        super().__init__(net, N0, Nb, Nf, 0, val_dt, epochs, lr, optimizer,
                         optimizer_params, lamb, loss_func, lr_scheduler,
                         mixed_precision, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed, max_loss)

        self.net.implicit.compute_jac_loss = False

        # initialize gradient projection optimization problem
        # decomposed for each row of A (including bias, assuming x = (x,1))
        a0 = cp.Parameter(self.net.implicit.A.weight.shape[0] + 1, complex=False)
        a = cp.Variable(self.net.implicit.A.weight.shape[0] + 1, complex=False)
        constraint = [cp.norm(a, p=1) <= self.kappa - 1e-3,]

        self._pgd_prob = cp.Problem(cp.Minimize(cp.norm(a - a0, p=2)), constraint)

        # multiprocessing pool for parallelization of gradient projection
        self._pgd_pool = Pool()

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

    def project_gradient_update(self, A):
        A0 = np.hstack([  # includes bias for x = (x,1)
            A.weight.data.detach().cpu().numpy(),
            A.bias.data.detach().cpu().numpy()[...,None],
        ])

        A_ = self.project_matrix(A0)

        A_weight = torch.from_numpy(A_[:,:-1]).to(A.weight.data)
        A_bias = torch.from_numpy(A_[:,-1]).to(A.bias.data)

        A.weight.data.copy_(A_weight)
        A.bias.data.copy_(A_bias)

    def train_pass(self):
        self.net.train()

        # training data
        (X0, Y0), Xb, Xf = self.data
        x0, t0 = X0
        (xb_low, xb_high), tb = Xb
        xf, tf = Xf

        xb_low.requires_grad_()
        xb_high.requires_grad_()
        tb.requires_grad_()
        xf.requires_grad_()
        tf.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    Y0_pred = self.net(t0, x0)

                    global loss_0
                    loss_0 = self._loss_func(Y0_pred, Y0.to(Y0_pred))

                    Yb_low_pred = self.net(tb, xb_low)
                    Yb_high_pred = self.net(tb, xb_high)

                    global loss_b
                    loss_b = self.get_loss_b(Yb_low_pred, Yb_high_pred, xb_low, xb_high)

                    global forward_time
                    forward_time, Yf_pred = timeit(self.net)(tf, xf)

                    global forward_nfe
                    forward_nfe = self.net.implicit.latest_nfe

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(Yf_pred, xf, tf)

                    global backward_nfe
                    backward_nfe = self.net.implicit.latest_backward_nfe

                    global loss
                    loss = loss_0 + loss_b + loss_f

                    if loss.requires_grad:
                        global backward_time

                        if self.mixed_precision:
                            backward_time, _ = timeit(self._scaler.scale(loss).backward)()
                        else:
                            backward_time, _ = timeit(loss.backward)()

                return loss

            if self.optimizer == 'LBFGS':
                self._optim.step(closure)
            else:
                closure()

                if self.mixed_precision:
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    self._optim.step()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        with torch.no_grad():
            A_oo = torch.linalg.norm(self.net.implicit.A.weight, ord=torch.inf)
            if A_oo.item() > self.kappa and self.net.implicit.A.requires_grad:
                grad_proj_time, _ = timeit(self.project_gradient_update)(self.net.implicit.A)
                A_oo = torch.linalg.norm(self.net.implicit.A.weight, ord=torch.inf)
            else:
                grad_proj_time = 0

        losses = {
            'all': loss.item(),
            '0': loss_0.item(),
            'b': loss_b.item(),
            'f': loss_f.item(),
            'fwd_nfe': forward_nfe,
            'bwd_nfe': backward_nfe,
            'A_oo': A_oo.item(),
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
            'grad_proj': grad_proj_time,
        }

        return losses, times

class PIDEQTrainerV3(PIDEQTrainerV2):
    def __init__(self, net: PIDEQ, N0=50, Nb=50, Nf=20000, kappa=1,
                 val_dt=0.001 * np.pi / 2, epochs=5, lr=0.001,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 lamb=0.1, loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 mixed_precision=False, lr_scheduler_params: dict = None,
                 device=None, wandb_project="pideq-nls", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None,
                 max_loss=None, eps=1e-3):
        self.eps = eps

        self._add_to_wandb_config({
            'eps': self.eps,
        })

        super().__init__(net, N0, Nb, Nf, kappa, val_dt, epochs, lr, optimizer,
                         optimizer_params, lamb, loss_func, lr_scheduler,
                         mixed_precision, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed, max_loss)

    def train_pass(self):
        self.net.train()

        # training data
        (X0, Y0), Xb, Xf = self.data
        x0, t0 = X0
        (xb_low, xb_high), tb = Xb
        xf, tf = Xf

        xb_low.requires_grad_()
        xb_high.requires_grad_()
        tb.requires_grad_()
        xf.requires_grad_()
        tf.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    Y0_pred = self.net(t0, x0)

                    global loss_0
                    loss_0 = self._loss_func(Y0_pred, Y0.to(Y0_pred))

                    Yb_low_pred = self.net(tb, xb_low)
                    Yb_high_pred = self.net(tb, xb_high)

                    global loss_b
                    loss_b = self.get_loss_b(Yb_low_pred, Yb_high_pred, xb_low, xb_high)

                    global forward_time
                    forward_time, Yf_pred = timeit(self.net)(tf, xf)

                    global forward_nfe
                    forward_nfe = self.net.implicit.latest_nfe

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(Yf_pred, xf, tf)

                    global backward_nfe
                    backward_nfe = self.net.implicit.latest_backward_nfe

                    global loss
                    loss = loss_0 + loss_b + loss_f

                    if loss.requires_grad:
                        global backward_time

                        if self.mixed_precision:
                            backward_time, _ = timeit(self._scaler.scale(loss).backward)()
                        else:
                            backward_time, _ = timeit(loss.backward)()

                return loss

            if self.optimizer == 'LBFGS':
                self._optim.step(closure)
            else:
                closure()

                with torch.no_grad():
                    # project A gradients
                    A = torch.hstack([self.net.implicit.A.weight.data, self.net.implicit.A.bias.data.unsqueeze(-1)])
                    A_grad = torch.hstack([self.net.implicit.A.weight.grad, self.net.implicit.A.bias.grad.unsqueeze(-1)])
                    
                    # project gradient to be tangent to the constrained space
                    A0 = A - self.eps * A_grad
                    if torch.linalg.norm(A0, ord=torch.inf) > self.kappa:
                        start_time = time()
                        A_grad = (A - self.project_matrix(A0)) / self.eps

                        self.net.implicit.A.weight.grad.copy_(A_grad[...,:-1].to(self.net.implicit.A.weight.grad))
                        self.net.implicit.A.bias.grad.copy_(A_grad[...,-1].to(self.net.implicit.A.bias.grad))
                        grad_proj_time = time() - start_time
                    else:
                        grad_proj_time = 0

                if self.mixed_precision:
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    self._optim.step()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        with torch.no_grad():
            A_oo = torch.linalg.norm(self.net.implicit.A.weight, ord=torch.inf)
            # if A_oo.item() > self.kappa:
            #     grad_proj_time, _ = timeit(self.project_gradient_update)(self.net.A)
            #     A_oo = torch.linalg.norm(self.net.A.weight, ord=torch.inf)
            # else:
            #     grad_proj_time = 0

        losses = {
            'all': loss.item(),
            '0': loss_0.item(),
            'b': loss_b.item(),
            'f': loss_f.item(),
            'fwd_nfe': forward_nfe,
            'bwd_nfe': backward_nfe,
            'A_oo': A_oo.item(),
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
            'grad_proj': grad_proj_time,
        }

        return losses, times

class PINCTrainer(PINNTrainer):
    def __init__(self, net: PINC, u0=np.array([0]), u_bounds=np.array([-1, 1]),
                 Nf=100000, Nt=1000, T=1, val_T=10, val_y0=np.array([0, .1]),
                 val_dt=0.1, epochs=5, lr=0.001,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 lamb=0.1, loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 mixed_precision=False, lr_scheduler_params: dict = None,
                 device=None, wandb_project="pideqc-vdp", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None):
        self.y_bounds = net.y_bounds
        self.u_bounds = u_bounds

        self.Nt = Nt

        super().__init__(net, u0, Nf, T, val_dt, epochs, lr, optimizer,
                         optimizer_params, lamb, loss_func, lr_scheduler,
                         mixed_precision, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed)

        self.val_T = val_T
        self.val_y0 = torch.Tensor(val_y0).view(1,val_y0.shape[0]).to(self.device)

    def prepare_data(self):
        y_range = self.y_bounds[:,1] - self.y_bounds[:,0]

        # data points
        y0_t = torch.rand(self.Nt, 2) * y_range + self.y_bounds[:,0]
        t_t = torch.zeros(self.Nt, 1)

        y0_t = y0_t.to(self.device).type(self._dtype)
        t_t = t_t.to(self.device).type(self._dtype)

        # collocation points
        y0_f = torch.rand(self.Nf, 2) * y_range + self.y_bounds[0]
        t_f = torch.rand(self.Nf, 1) * self.T

        y0_f = y0_f.to(self.device).type(self._dtype)
        t_f = t_f.to(self.device).type(self._dtype)

        self.data = (y0_t, t_t), (y0_f, t_f)

        if self.val_data is None:
            K = int(0.5 + self.val_T / self.val_dt)

            t_val = torch.Tensor([i * self.val_dt for i in range(K+1)])

            y_val = odeint(lambda t, y: self.f(y,self.u0), self.val_y0, t_val,
                           method='rk4')

            self.val_data = (
                t_val.to(self.device).type(self._dtype).unsqueeze(-1),
                y_val.to(self.device).type(self._dtype).squeeze()
            )

    def train_pass(self):
        self.net.train()

        (y0_t, t_t), (y0_f, t_f) = self.data

        t_t.requires_grad_()
        t_f.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    y_t_pred = self.net(y0_t, t_t)

                    global loss_y
                    loss_y = self._loss_func(y_t_pred, y0_t.to(y_t_pred))

                    global forward_time
                    forward_time, y_pred = timeit(self.net)(y0_f, t_f)

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(y_pred, t_f)

                    global loss
                    loss = loss_y + self.lamb * loss_f

                    if loss.requires_grad:
                        global backward_time

                        if self.mixed_precision:
                            backward_time, _ = timeit(self._scaler.scale(loss).backward)()
                        else:
                            backward_time, _ = timeit(loss.backward)()

                return loss

            if self.optimizer == 'LBFGS':
                self._optim.step(closure)
            else:
                closure()

                if self.mixed_precision:
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    self._optim.step()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        losses = {
            'all': loss.item(),
            'y': loss_y.item(),
            'f': loss_f.item(),
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }
        return losses, times

    def validation_pass(self):
        self.net.eval()

        _, y = self.val_data

        t = torch.rand(y.shape[0] - 1, 1) * self.val_dt
        t = t.to(y)

        with torch.set_grad_enabled(False):
            self._optim.zero_grad()
            forward_time, y_pred = timeit(self.net)(y[:-1], t)

            y_preds = list()
            for y_single in y[:-1]:
                y_preds.append(self.net(y_single, t[0]))
            
        y_pred_selfloop = torch.stack(y_preds)
        iae_selfloop = (y[1:] - y_pred_selfloop).abs().sum().item() * self.val_dt

        iae = (y[1:] - y_pred).abs().sum().item() * self.val_dt
        mae = (y[1:] - y_pred).abs().mean().item()

        losses = {
            'all': iae,
            'iae': iae,
            'iae_selfloop': iae_selfloop,
            'mae': mae,
        }
        times = {
            'forward': forward_time,
        }

        return losses, times
