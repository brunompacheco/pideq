"""Test Jacobians of implicit functions with dummy data.
"""
import numpy as np
import scipy as sp
import torch
import torch.nn as nn

from torch.autograd import grad, Function

from pideq.deq.model import get_implicit
from pideq.utils import get_jacobian


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
        Bs.append(nmode_product(A[b], U[b], n-1))
    
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

if __name__ == '__main__':
    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    z0 = torch.zeros(batch_size, n_states).double()

    implicit = get_implicit(forward_max_steps=500, forward_eps=1e-6,
                            backward_max_steps=500, backward_eps=1e-6)

    ### SYNTHETIC TESTS

    ### Jacobians

    x = torch.ones(batch_size, n_in).double()
    x.requires_grad_()

    A = nn.Linear(n_states, n_states).double()
    A_ = torch.zeros_like(A.weight)
    A_[-2:,:2] = torch.eye(2).to(A.weight)
    A.weight.data.copy_(A_)  # A = [[0, 0], [I, 0]]
    A.bias.data.copy_(torch.zeros_like(A.bias))

    B = nn.Linear(n_in, n_states).double()
    B_ = torch.zeros_like(B.weight)
    B_[:2,:2] = torch.eye(2).to(B.weight)
    B.weight.data.copy_(B_)  # B = [[I,], [0,]]
    B.bias.data.copy_(torch.zeros_like(B.bias))

    z = implicit(x, z0, A.weight, A.bias, B.weight, B.bias)

    # explicit Jacobian
    # WARNING: do not backpropagate through the explicit Jacobians
    J_tanh = torch.diag_embed(1-z**2)
    J_fx = J_tanh @ B.weight
    J_fz = J_tanh @ A.weight
    implicit_inv = torch.linalg.inv(J_fz - torch.eye(n_states))
    J_zx = -implicit_inv @ J_fx

    # automatic Jacobians
    autograd_Jzx = get_jacobian(z, x)
    functional_Jzx = torch.autograd.functional.jacobian(lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias), x, create_graph=True)
    functional_Jzx = torch.diagonal(functional_Jzx, dim1=0, dim2=2).transpose(2,1).transpose(1,0)

    numerical_Jzx = numerical_jacobian(lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias), x, batched=True)

    assert torch.isclose(autograd_Jzx, J_zx).all(), "Jacobian generated with autograd is different from the analytical one"
    assert torch.isclose(functional_Jzx, J_zx).all(), "Jacobian generated with functional.Jacobian is different from the analytical one"
    assert torch.isclose(numerical_Jzx, J_zx).all(), "Jacobian generated with sp.approx_fprime is different from the analytical one"

    ### Hessians

    H_tanh = torch.diag_embed(J_tanh * torch.diag_embed(-2*z))

    H_fxx = nmode_product(nmode_product(H_tanh, B.weight.T, 3), B.weight.T, 2)
    H_fzx = nmode_product(nmode_product(H_tanh, B.weight.T, 3), A.weight.T, 2)

    H_zxx = - batched_nmode_product(
        H_fxx + batched_nmode_product(H_fzx, J_zx.transpose(1,2), 2),
        implicit_inv,
        1
    )

    f_ = lambda x, z: torch.tanh(z @ A.weight.T + A.bias + x @ B.weight.T + B.bias)

    def get_Jfx(x):
        x.requires_grad_()
        return get_jacobian(f_(x, z), x)
    numerical_Hfxx = numerical_jacobian(get_Jfx, x, batched=True)
    assert torch.isclose(numerical_Hfxx, H_fxx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    def get_Jfz(x):
        z.requires_grad_()
        return get_jacobian(f_(x, z), z)
    numerical_Hfzx = numerical_jacobian(get_Jfz, x, batched=True)
    assert torch.isclose(numerical_Hfzx, H_fzx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    def get_Jzx(x):
        x.requires_grad_()
        z = implicit(x, z0, A.weight, A.bias, B.weight, B.bias)
        return get_jacobian(z, x)
    numerical_Hzxx = numerical_jacobian(get_Jzx, x, batched=True)
    assert torch.isclose(numerical_Hzxx, H_zxx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    print('DONE')
