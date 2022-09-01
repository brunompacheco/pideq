from tkinter import N
import numpy as np
import torch
import torch.nn as nn

from pideq.net import PIDEQ, PINC, PINN
from pideq.trainer import PIDEQTrainer, PINCTrainer, PINNTrainer
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
        wandb_project = 'pideqc-vdp'

    T = 1.

    torch.autograd.set_detect_anomaly(True)
    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    for _ in range(1):
        # PIDEQTrainer(
        #     PIDEQ(T, n_states=20, n_out=2, n_hidden=1, solver_kwargs={'threshold': 300, 'eps': 1e-6}).to(device),
        #     lr=1e-3,
        #     epochs=1000,
        #     T=T,
        #     val_dt=.1,
        #     jac_lamb=.1,
        #     wandb_project=wandb_project,
        #     wandb_group='test-deq',
        #     mixed_precision=False,
        #     # lr_scheduler='MultiStepLR',
        #     # lr_scheduler_params={'milestones': [30000, 40000], 'gamma': .1},
        #     # lr_scheduler='MultiplicativeLR',
        #     # lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/100000 - 1))},
        #     random_seed=seed,
        # ).run()

        PINCTrainer(
            PINC(T,),
            lr=1e-3,
            epochs=5000,
            T=T,
            # optimizer='LBFGS',
            # optimizer_params={'max_iter': 2000,},
            wandb_project=wandb_project,
            wandb_group='test',
            lr_scheduler='MultiStepLR',
            lr_scheduler_params={'milestones': [500, 1000, 1500, 2000, 2500, 3000]},
            # lr_scheduler='CosineAnnealingLR',
            # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-3},
            mixed_precision=False,
            # loss_func='L1Loss',
        ).run()

        # wandb.alert(
        #     title='Training finished',
        #     level=AlertLevel.INFO,
        # )
