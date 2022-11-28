import sys

import wandb

import torch
import torch.nn as nn

from torch.autograd import grad


def load_from_wandb(net: nn.Module, run_id: str,
                    project='pideq-sine', model_fname='model_last'):
    best_model_file = wandb.restore(
        f'{model_fname}.pth',
        run_path=f"brunompac/{project}/{run_id}",
        replace=True,
    )
    net.load_state_dict(torch.load(best_model_file.name))

    return net

def debugger_is_active() -> bool:
    """Return if the debugger is currently active.

    From https://stackoverflow.com/a/67065084
    """
    gettrace = getattr(sys, 'gettrace', lambda : None) 
    return gettrace() is not None

def get_jacobian(Y, x_):
    """Compute the jacobian of a vector->vector function using `autograd.grad`.
    """
    grads_y = list()
    for i in range(Y.shape[1]):
        grad_y = grad(Y[:,i].sum(), x_, create_graph=True)[0]
        grad_y = grad_y.unsqueeze(1)  # effectively transposing each vector of the batch

        grads_y.append(grad_y)

    J = torch.cat(grads_y, dim=1)

    return J
