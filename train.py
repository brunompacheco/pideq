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
    for _ in range(1):
        net = PINN(T, n_out=2, n_hidden=4, n_nodes=20).to(device)
        net = load_from_wandb(net, 'o76oyh72')
        Trainer4T(
            net,
            # lr=1e-3,
            lr=1,
            epochs=1,
            T=T,
            val_dt=.1,
            optimizer='LBFGS',
            optimizer_params={'max_iter': 2000,},
            # wandb_project=None,
            wandb_group='test-vdp',
            # lr_scheduler='CosineAnnealingLR',
            # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-3},
            mixed_precision=False,
            # loss_func='L1Loss',
        ).run()
