import torch

from pideq.net import PINN
from pideq.trainer import Trainer4T


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Experiment 1
    # Baseline FCN for different time horizons
    for T, epochs in [(2, 1e4), (5, 1e5), (10, 3e5)]:
        for _ in range(5):
            Trainer4T(
                PINN(T, n_nodes=20).to(device),
                lr=1e-3 if T <= 5 else 1e-2,
                epochs=epochs,
                T=T,
                val_dt=T/1000,
                wandb_group=f'FCN-baseline-T={T}',
                mixed_precision=False,
            ).run()
