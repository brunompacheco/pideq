import torch
import torch.nn as nn

from pideq.net import PIDEQ, PINN
from pideq.trainer import DEQTrainer4T, Trainer4T
from pideq.utils import load_from_wandb


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = 10.

    torch.autograd.set_detect_anomaly(True)
    # for run_id in ['1o89emea', '14az3g9t', '105xvwm9', '3bz20l11', '28a8g3jd']:
    for _ in range(1):
        # net = PINN(T, n_nodes=20).to(device)
        # net = load_from_wandb(net, 'wwiuyqiw', model_fname='model_last')
        # net = net.double()
        # torch.backends.cuda.matmul.allow_tf32 = False
        DEQTrainer4T(
            PIDEQ(T, n_out=4, n_states=20),
            lr=1e-3,
            epochs=1000,
            T=T,
            val_dt=.01,
            # optimizer='LBFGS',
            # optimizer_params={'max_iter': 1, 'max_eval': 2000},
            # wandb_project=None,
            wandb_group=f'test-deq',
            # lr_scheduler='ExponentialLR',
            # lr_scheduler_params={'gamma': 0.99},
            # lr_scheduler='CosineAnnealingLR',
            # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-4},
            mixed_precision=False,
            # loss_func='L1Loss',
        ).run()
