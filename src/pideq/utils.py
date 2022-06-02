import wandb

import torch
import torch.nn as nn


def load_from_wandb(net: nn.Module, run_id: str,
                    project='pideq-4t', model_fname='model_last'):
    best_model_file = wandb.restore(
        f'{model_fname}.pth',
        run_path=f"brunompac/{project}/{run_id}",
        replace=True,
    )
    net.load_state_dict(torch.load(best_model_file.name))

    return net
