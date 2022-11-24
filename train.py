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
        wandb_project = 'pideq-sine'

    T = np.pi

    torch.autograd.set_detect_anomaly(True)
    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    PIDEQTrainer(
        PIDEQ(T, n_in=1, n_states=80, compute_jac_loss=True),
        jac_lambda=10,
        epochs=10000,
        wandb_project=wandb_project,
        wandb_group='test',
        random_seed=seed,
        device='cpu',
    ).run()
