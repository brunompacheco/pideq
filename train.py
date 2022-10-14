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

    for _ in range(10):
        DEQTrainer(
            # DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            nn.Sequential(
                nn.Linear(1,25),
                nn.Tanh(),
                nn.Linear(25,25),
                nn.Tanh(),
                nn.Linear(25,25),
                nn.Tanh(),
                nn.Linear(25,1),
            ).to(device),
            lr=1e-2,
            # kappa=.9,
            # project_grad=True,
            # A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='NN',
            random_seed=seed,
        ).run()

        DEQTrainer(
            # DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            nn.Sequential(
                nn.Linear(1,25),
                nn.Tanh(),
                nn.Linear(25,25),
                nn.Tanh(),
                nn.Linear(25,25),
                nn.Tanh(),
                nn.Linear(25,1),
            ).to(device),
            # kappa=.9,
            # project_grad=True,
            # A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='NN',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            # kappa=.9,
            # project_grad=True,
            # A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            lr=1e-2,
            # kappa=.9,
            # project_grad=True,
            # A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            # kappa=.9,
            # project_grad=True,
            A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ-reg',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            lr=1e-2,
            # kappa=.9,
            # project_grad=True,
            A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ-reg',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            kappa=.9,
            # project_grad=True,
            # A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ-proj',
            random_seed=seed,
        ).run()

        DEQTrainer(
            DEQ(n_states=75, weight_initialization_factor=1, compute_jac_loss=False),
            kappa=.9,
            project_grad=True,
            # A_oo_lambda=1,
            wandb_project=wandb_project,
            wandb_group='DEQ-grad-proj',
            random_seed=seed,
        ).run()

    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    # pinn = load_from_wandb(PINN(np.pi / 2, n_hidden=4, n_nodes=100), '2n1n4ml0', project='pideq-nls', model_fname='model_last')
    # for _ in range(5):
    #     PIDEQTrainer(
    #         PIDEQ.from_pinn(pinn),
    #         jac_lambda=10,
    #         lr=1e-5,
    #         epochs=5000,
    #         wandb_project=wandb_project,
    #         wandb_group='test',
    #         random_seed=seed,
    #     ).run()

    #     PIDEQTrainer(
    #         PIDEQ(T, n_states=200),
    #         epochs=5000,
    #         wandb_project=wandb_project,
    #         wandb_group='test',
    #         random_seed=seed,
    #     ).run()

    #     PIDEQTrainer(
    #         PIDEQ(T, n_states=200),
    #         jac_lambda=10,
    #         epochs=5000,
    #         wandb_project=wandb_project,
    #         wandb_group='test',
    #         random_seed=seed,
    #     ).run()

        # wandb.alert(
        #     title='Training finished',
        #     level=AlertLevel.INFO,
        # )
