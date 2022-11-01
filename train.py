from tkinter import N
import numpy as np
import torch
import torch.nn as nn
from pideq.deq.model import DEQ

from pideq.deq.solvers import broyden
from pideq.net import PIDEQ, PINC, PINN
from pideq.trainer import DEQTrainer
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
        wandb_project = 'pideq-ghaoui'

    T = np.pi / 2

    torch.autograd.set_detect_anomaly(True)

    for _ in range(5):
        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=True),
            jac_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ-jac-reg',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=.1, compute_jac_loss=True),
            jac_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ-jac-reg-small-init',
            random_seed=seed,
        ).run()

    # for _ in range(10):
    #     DEQTrainer(
    #         # DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         nn.Sequential(
    #             nn.Linear(1,25),
    #             nn.Tanh(),
    #             nn.Linear(25,25),
    #             nn.Tanh(),
    #             nn.Linear(25,25),
    #             nn.Tanh(),
    #             nn.Linear(25,1),
    #         ).to(device),
    #         lr=1e-2,
    #         # kappa=.9,
    #         # project_grad=True,
    #         # A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='NN',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         # DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         nn.Sequential(
    #             nn.Linear(1,25),
    #             nn.Tanh(),
    #             nn.Linear(25,25),
    #             nn.Tanh(),
    #             nn.Linear(25,25),
    #             nn.Tanh(),
    #             nn.Linear(25,1),
    #         ).to(device),
    #         # kappa=.9,
    #         # project_grad=True,
    #         # A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='NN',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         # kappa=.9,
    #         # project_grad=True,
    #         # A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='DEQ',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         lr=1e-2,
    #         # kappa=.9,
    #         # project_grad=True,
    #         # A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='DEQ',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         # kappa=.9,
    #         # project_grad=True,
    #         A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='DEQ-reg',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         lr=1e-2,
    #         # kappa=.9,
    #         # project_grad=True,
    #         A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='DEQ-reg',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         kappa=.9,
    #         # project_grad=True,
    #         # A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='DEQ-proj',
    #         random_seed=seed,
    #     ).run()

    #     DEQTrainer(
    #         DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
    #         kappa=.9,
    #         project_grad=True,
    #         # A_oo_lambda=1,
    #         wandb_project=wandb_project,
    #         wandb_group='DEQ-grad-proj',
    #         random_seed=seed,
    #     ).run()
