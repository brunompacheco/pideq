import sys

import scipy as sp
import torch
import torch.nn as nn
import wandb
import tensorly as tl
tl.set_backend('pytorch')

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

def numerical_jacobian(f, x0, batched=False):
    x_shape = list(x0.shape)
    _y = f(x0)
    y_shape = list(_y.shape)

    def f_(x, i):
        x_ = torch.from_numpy(x).to(x0)
        x_ = x_.unflatten(dim=0, sizes=x_shape)
        y_ = f(x_)
        y = y_.flatten().detach().cpu().numpy()

        return y[i]

    x0_ = x0.flatten().detach().cpu().numpy()
    jacs = list()
    for i in range(_y.flatten().shape[0]):
        jac = sp.optimize.approx_fprime(x0_, f_, 1e-8, i)
        jac = torch.from_numpy(jac).to(x0).unflatten(dim=0, sizes=x_shape)
        jacs.append(jac)

    J = torch.stack(jacs)
    J = J.unflatten(dim=0, sizes=y_shape)

    if batched:
        J = torch.diagonal(J, dim1=0, dim2=len(y_shape))
        ax_J = torch.arange(len(J.shape))
        J = J.permute((ax_J - 1).tolist())

    return J

def batched_nmode_product(A, U, n:int):
    """Performs n-mode product between batches of tensors and matrices.

    TODO: make efficient, not iterative.
    """
    assert A.shape[0] == U.shape[0], "different batch sizes"
    Bs = list()
    for b in range(A.shape[0]):
        Bs.append(tl.tenalg.mode_dot(A[b], U[b], n-1))
    
    return torch.stack(Bs)

def nmode_product(A, U, n: int):
    """Performs n-mode product between tensor A and matrix U.
    """
    B = torch.tensordot(A, U.T, ([n,], [0,]))

    # torch.tensordot contracts the n-th mode and stacks it at the last
    # mode of B, so we must perform a permute operation to get the expected
    # shape

    dims = torch.arange(len(A.shape))
    dims[n] = -1
    try:
        # only if n is not the last dimension
        dims[n+1:] -= 1
    except IndexError:
        pass

    return B.permute(list(dims))
