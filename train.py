import numpy as np
import torch
import torch.nn as nn

from pideq.net import PIDEQ, PINN
from pideq.trainer import DEQTrainer4T, Trainer4T
from pideq.utils import load_from_wandb, debugger_is_active

# import wandb
# from wandb import AlertLevel


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        import random
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        wandb_project = None
    else:
        seed = None
        wandb_project = 'pideq-vdp'

    T = 2.

    # torch.autograd.set_detect_anomaly(True)
    # TODO: check if jac_loss_t is overwhelming the gradients in comparison to jac_loss_f
    for _ in range(1):
        DEQTrainer4T(
            PIDEQ(T, n_states=80, n_out=1, solver_kwargs={'threshold': 200, 'eps': 1e-4}).to(device),
            lr=1e-3,
            epochs=100000,
            T=T,
            val_dt=.1,
            wandb_project=wandb_project,
            wandb_group='test-deq',
            mixed_precision=False,
            random_seed=seed,
        ).run()

        # Trainer4T(
        #     net,
        #     lr=1e-3,
        #     epochs=100000,
        #     T=T,
        #     val_dt=.1,
        #     # optimizer='LBFGS',
        #     # optimizer_params={'max_iter': 2000,},
        #     # wandb_project=None,
        #     wandb_group='test',
        #     lr_scheduler='MultiStepLR',
        #     lr_scheduler_params={'milestones': [60000, 80000]}
        #     # lr_scheduler='CosineAnnealingLR',
        #     # lr_scheduler_params={'T_max': 200, 'eta_min': 1e-3},
        #     mixed_precision=False,
        #     # loss_func='L1Loss',
        # ).run()

        # wandb.alert(
        #     title='Training finished',
        #     level=AlertLevel.INFO,
        # )
