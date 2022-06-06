import torch

from pideq.net import PINN
from pideq.trainer import Trainer4T
from pideq.utils import load_from_wandb


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = 2.

    torch.autograd.set_detect_anomaly(True)
    for _ in range(1):
        net = PINN(T, n_nodes=100).to(device)
        # net = load_from_wandb(net, '2f2dmyf0')
        Trainer4T(
            net,
            lr=1e-3,
            epochs=1000,
            T=T,
            val_dt=.1,
            # optimizer='LBFGS',
            # wandb_project=None,
            wandb_group='test-kurz',
            # lr_scheduler='CosineAnnealingLR',
            # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-3},
            mixed_precision=False,
            # loss_func='L1Loss',
        ).run()
