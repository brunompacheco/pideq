from tkinter import N
import numpy as np
import torch
import torch.nn as nn
from pideq.deq.solvers import broyden, forward_iteration

from pideq.net import PIDEQ, PINN
from pideq.trainer import PIDEQTrainer, PINNTrainer
from pideq.utils import load_from_wandb, debugger_is_active

# import wandb
# from wandb import AlertLevel


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        import random
        seed = 33
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        wandb_project = None
    else:
        seed = None
        wandb_project = 'pideq-4t'

    T = 20.

    torch.autograd.set_detect_anomaly(True)
    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    PIDEQTrainer(
        PIDEQ(T, n_states=40, solver=broyden, solver_kwargs={'threshold': 300, 'eps': 1e-6}).to(device),
        lr=1e-3,
        epochs=100000,
        T=T,
        val_dt=.1,
        wandb_group='test',
        mixed_precision=False,
        # lr_scheduler='MultiStepLR',
        # lr_scheduler_params={'milestones': [(i+1) * 1000 for i in range (10)], 'gamma': .1},
        # lr_scheduler='MultiplicativeLR',
        # lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/100000 - 1))},
        wandb_project=wandb_project,
        random_seed=seed,
    ).run()

    # PINNTrainer(
    #     PINN(T,),
    #     lr=1e-3,
    #     epochs=10000,
    #     T=T,
    #     val_dt=.1,
    #     # optimizer='LBFGS',
    #     # optimizer_params={'max_iter': 2000,},
    #     # wandb_project=None,
    #     wandb_group='test',
    #     # lr_scheduler='MultiStepLR',
    #     # lr_scheduler_params={'milestones': [60000, 80000]}
    #     # lr_scheduler='CosineAnnealingLR',
    #     # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-3},
    #     mixed_precision=False,
    #     # loss_func='L1Loss',
    #     wandb_project=wandb_project,
    # ).run()

    # wandb.alert(
    #     title='Training finished',
    #     level=AlertLevel.INFO,
    # )
