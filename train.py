import torch

from pideq.net import PINN
from pideq.trainer import Trainer4T


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # torch.autograd.set_detect_anomaly(True)
    for _ in range(5):
        net = PINN().to(device)
        Trainer4T(
            net,
            lr=1e-2,
            epochs=1000,
            T=200,
            val_dt=1.,
            # wandb_project=None,
            wandb_group='test',
            mixed_precision=False,
        ).run()
