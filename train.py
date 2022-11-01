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
        wandb_project = 'pideq-nls'

    T = np.pi / 2

    torch.autograd.set_detect_anomaly(True)
    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    for _ in range(5):
        try:
            PIDEQTrainer(
                PIDEQ(T),
                epochs=20000,
                wandb_project=wandb_project,
                wandb_group='No-Reg',
                random_seed=seed,
            ).run()
        except RuntimeError:
            pass

        PIDEQTrainer(
            PIDEQ(T, compute_jac_loss=True),
            epochs=20000,
            jac_lambda=0.1,
            wandb_project=wandb_project,
            wandb_group='Jac-Loss-small',
            random_seed=seed,
        ).run()

        PIDEQTrainer(
            PIDEQ(T, compute_jac_loss=True),
            epochs=20000,
            jac_lambda=1.,
            wandb_project=wandb_project,
            wandb_group='Jac-Loss',
            random_seed=seed,
        ).run()

        PIDEQTrainer(
            PIDEQ(T),
            epochs=20000,
            A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='Aoo-Reg',
            random_seed=seed,
        ).run()

        PIDEQTrainer(
            PIDEQ(T),
            epochs=20000,
            kappa=.9,
            wandb_project=wandb_project,
            wandb_group='Aoo-Proj',
            random_seed=seed,
        ).run()

        PIDEQTrainer(
            PIDEQ(T),
            epochs=20000,
            kappa=.9,
            project_grad=True,
            wandb_project=wandb_project,
            wandb_group='Aoo-Grad-Proj',
            random_seed=seed,
        ).run()

        # wandb.alert(
        #     title='Training finished',
        #     level=AlertLevel.INFO,
        # )
