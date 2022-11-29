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

def get_jacobian(Y, x):
    """Compute the Jacobian of a vector->tensor function using `autograd.grad`.

    Computing the Jacobian of a vector->matrix product is useful for computing
    the Hessian of a given function. In this case, Y can be the Jacobian, which
    can also be computed through this function.

    Args:
        Y: Function batched output. Must have shape (b x m1 x ... x mk), in which b
        is the batch dimension and k >= 1.
        x: Input with respect to which to differentiate. Has shape (b x n).
    
    Returns:
        J: Jacobian of Y with respect to x. Has shape (b x m1 x ... x mk x n).
    """
    y_shape = Y.shape[1:]
    Y_ = Y.flatten(start_dim=1)  # encode matrix output as a vector
    grads_y = list()
    for i in range(Y_.shape[1]):
        grad_y = grad(Y_[:,i].sum(), x, create_graph=True)[0]
        grad_y = grad_y.unsqueeze(1)  # effectively transposing each vector of the batch

        grads_y.append(grad_y)

    J = torch.cat(grads_y, dim=1)
    J = J.unflatten(dim=1, sizes=y_shape)  # recover original output's shape

    return J
