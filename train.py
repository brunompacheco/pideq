import numpy as np
import torch
import torch.nn as nn

from pideq.net import PIDEQ, PINN
from pideq.trainer import DEQTrainer4T, Trainer4T
from pideq.utils import load_from_wandb


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = 2.

    # torch.autograd.set_detect_anomaly(True)
    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    for _ in range(1):
        try:
            DEQTrainer4T(
                PIDEQ(T, n_states=80).to(device),
                lr=1e-3,
                epochs=50000,
                jac_lamb=.01,
                T=T,
                val_dt=.1,
                # wandb_project=None,
                wandb_group='test-deq',
                mixed_precision=False,
            ).run()
        except Exception as e:
            print(e)
        try:
            DEQTrainer4T(
                PIDEQ(T).to(device),
                lr=1e-4,
                epochs=50000,
                jac_lamb=.1,
                T=T,
                val_dt=.1,
                # wandb_project=None,
                wandb_group='test-deq',
                mixed_precision=False,
            ).run()
        except Exception as e:
            print(e)
        try:
            DEQTrainer4T(
                PIDEQ(T, n_states=80).to(device),
                lr=1e-4,
                epochs=50000,
                jac_lamb=.01,
                T=T,
                val_dt=.1,
                # wandb_project=None,
                wandb_group='test-deq',
                mixed_precision=False,
            ).run()
        except Exception as e:
            print(e)
        # Trainer4T(
        #     net,
        #     lr=1e-3,
        #     epochs=100000,
        #     T=T,
        #     val_dt=.1,
        #     # optimizer='LBFGS',
        #     # optimizer_params={'max_iter': 2000,},
        #     # wandb_project=None,
        #     wandb_group='test',
        #     lr_scheduler='MultiStepLR',
        #     lr_scheduler_params={'milestones': [60000, 80000]}
        #     # lr_scheduler='CosineAnnealingLR',
        #     # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-3},
        #     mixed_precision=False,
        #     # loss_func='L1Loss',
        # ).run()
