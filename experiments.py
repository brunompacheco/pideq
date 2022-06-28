import torch

from pideq.net import PIDEQ, PINN
from pideq.trainer import PIDEQTrainer, PINNTrainer


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1: Baseline PINN
    for _ in range(5):
        PINNTrainer(
            PINN(2., n_nodes=20).to(device),
            epochs=5e4,
            wandb_group=f'PINN-baseline',
        ).run()

    # 2: PIDEQ Baseline
    for _ in range(5):
        PIDEQTrainer(
            PIDEQ(2., n_states=80),
            epochs=5e4,
            wandb_group=f'PIDEQ-baseline',
        )

    # 3: PIDEQ #States
    for n_states in [40, 20, 10, 5]:
        for _ in range(5):
            PIDEQTrainer(
                PIDEQ(2., n_states=n_states),
                epochs=5e4,
                wandb_group=f'PIDEQ-#z={n_states}',
            )
