from contextlib import nullcontext
import logging
import random

from abc import ABC, abstractmethod
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from torchdiffeq import odeint
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import grad
from pideq.deq.model import DEQ

from pideq.four_tanks import four_tanks
from pideq.net import PINN


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
                 logger=None, checkpoint_every=50, random_seed=42) -> None:
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

    def run(self):
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

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('Training finished!')

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
    """Trainer for the Van der Pol oscillator using a Physics-Informed NN."""
    def __init__(self, net: PINN, y0=np.array([0., .1]),
                 u0=np.array([0.]), Nf=1e5, T=2, val_dt=.002, epochs=5,
                 lr=1e-3, optimizer: str = 'Adam', optimizer_params: dict = None,
                 lamb=0.1, loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 mixed_precision=False, lr_scheduler_params: dict = None,
                 device=None, wandb_project="pideq-vdp", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None):
        self._add_to_wandb_config({
            'T': T,
            'y0': y0,
            'u0': u0,
        })

        super().__init__(net, epochs, lr, optimizer, optimizer_params,
                         loss_func, lr_scheduler, lr_scheduler_params,
                         mixed_precision, device, wandb_project,
                         wandb_group, logger, checkpoint_every, random_seed)

        self.Nf = int(Nf)    # number of collocation points
        self.val_dt = val_dt

        # initial state
        self.y0 = torch.from_numpy(y0).view(1,y0.shape[0]).to(self.device)

        # initial control
        self.u0 = torch.from_numpy(u0).view(1,u0.shape[0]).to(self.device)

        self.T = T

        if lamb is None:
            self.lamb = 1 / self.Nf
        else:
            self.lamb = lamb

        self.data = None
        self.val_data = None

        self.f = f

    def prepare_data(self):
        X = torch.rand(self.Nf,1) * self.T

        self.data = X.to(self.device).type(self._dtype)

        if self.val_data is None:
            K = int(0.5 + self.T / self.val_dt)

            X_val = torch.Tensor([i * self.val_dt for i in range(K+1)])

            Y_val = odeint(lambda t, y: self.f(y,self.u0), self.y0, X_val,
                           method='rk4')

            self.val_data = (
                X_val.to(self.device).type(self._dtype).unsqueeze(-1),
                Y_val.to(self.device).type(self._dtype).squeeze()
            )

    def _run_epoch(self):
        self.prepare_data()

        return super()._run_epoch()

    def get_loss_f(self, y_pred, x):
        dy_i_preds = list()
        for i in range(y_pred.shape[-1]):
            dy_i_preds.append(grad(y_pred[:,i].sum(), x, create_graph=True)[0])
        # dy_i_preds = [
        #     y_pred[:,1:],
        #     grad(y_pred[:,1].sum(), x, create_graph=True)[0],
        # ]

        Jy_pred = torch.stack(dy_i_preds, dim=-1).squeeze(1)

        u0_ = self.u0.to(y_pred).repeat(y_pred.shape[0],1)
        Jy = self.f(y_pred, u0_)

        ode = Jy_pred - Jy

        return self._loss_func(ode, torch.zeros_like(ode))

    def train_pass(self):
        self.net.train()

        # boundary
        X_t = torch.zeros(1,1).to(self.device).type(self._dtype)
        Y_t = self.y0.clone().to(self.device).type(self._dtype)

        # collocation
        X_f = self.data

        X_t.requires_grad_()
        X_f.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    y_t_pred = self.net(X_t)

                    global loss_y
                    loss_y = self._loss_func(y_t_pred, Y_t.to(y_t_pred))

                    global forward_time
                    forward_time, y_pred = timeit(self.net)(X_f)

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(y_pred, X_f)

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

        X, Y = self.val_data

        with torch.set_grad_enabled(False):
            self._optim.zero_grad()
            forward_time, y_pred = timeit(self.net)(X)

        iae = (Y - y_pred).abs().sum().item() * self.val_dt
        mae = (Y - y_pred).abs().mean().item()

        losses = {
            'all': iae,
            'iae': iae,
            'mae': mae,
        }
        times = {
            'forward': forward_time,
        }

        return losses, times

class PIDEQTrainer(PINNTrainer):
    def __init__(self, net: DEQ, y0=np.array([0., .1]),
                 u0=np.array([0.]), Nf=100000, T=2, val_dt=.002,
                 epochs=5, lr=1e-3, optimizer: str = 'Adam',
                 optimizer_params: dict = None, lamb=0.1, jac_lamb=1.,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 mixed_precision=False, lr_scheduler_params: dict = None,
                 device=None, wandb_project="pideq-vdp", wandb_group=None,
                 logger=None, checkpoint_every=1000, random_seed=None):
        self._add_to_wandb_config({
            'n_states': net.n_states,
            'n_hidden': net.n_hidden,
            'solver_max_nfe': net.solver_kwargs['threshold'],
            'solver_eps': net.solver_kwargs['eps'],
            'solver': net.solver.__name__,
        })

        super().__init__(net, y0, u0, Nf, T, val_dt, epochs, lr, optimizer,
                         optimizer_params, lamb, loss_func, lr_scheduler,
                         mixed_precision, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed)

        self.jac_loss_lamb = jac_lamb

    def train_pass(self):
        self.net.train()

        # boundary
        X_t = torch.zeros(1,1).to(self.device).type(self._dtype)
        Y_t = self.y0.clone().to(self.device).type(self._dtype)

        # collocation
        X_f = self.data

        X_t.requires_grad_()
        X_f.requires_grad_()
        with torch.set_grad_enabled(True):
            def closure():
                if torch.is_grad_enabled():
                    self._optim.zero_grad()

                with self.autocast_if_mp():
                    Y_t_pred, jac_loss_t = self.net(X_t)

                    global loss_y
                    loss_y = self._loss_func(Y_t_pred, Y_t.to(Y_t_pred))

                    global forward_time
                    forward_time, (y_pred, jac_loss_f) = timeit(self.net)(X_f)
                    forward_time -= self.net.jac_loss_time

                    global forward_nfe
                    forward_nfe = self.net.latest_nfe

                    # dy_pred = torch.autograd.grad(
                    #     y_pred.sum(),
                    #     X_f,
                    #     create_graph=True,
                    # )[0]

                    # ddy_pred = torch.autograd.grad(
                    #     dy_pred.sum(),
                    #     X_f,
                    #     create_graph=True,
                    # )[0]

                    # mu = 1.
                    # ddy = + mu * (1 - y_pred ** 2) * dy_pred - y_pred
                    # ode = ddy_pred - ddy

                    global loss_f, loss_time
                    loss_time, loss_f = timeit(self.get_loss_f)(y_pred, X_f)
                    loss_time += self.net.jac_loss_time
                    # loss_f = self._loss_func(ode, torch.zeros_like(ode))

                    global backward_nfe
                    backward_nfe = self.net.latest_backward_nfe

                    global jac_loss
                    # jac_loss = (jac_loss_t + jac_loss_f) / 2
                    jac_loss = jac_loss_t + jac_loss_f

                    global loss
                    loss = loss_y + self.lamb * loss_f + self.jac_loss_lamb * jac_loss

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
            'jac': jac_loss.item(),
            'fwd_nfe': forward_nfe,
            'bwd_nfe': backward_nfe,
        }
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }
        return losses, times

    def validation_pass(self):
        losses, times = super().validation_pass()

        losses['fwd_nfe'] = self.net.latest_nfe
        losses['bwd_nfe'] = self.net.latest_backward_nfe

        return losses, times
