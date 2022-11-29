"""Test Jacobians of implicit functions with dummy data.
"""
import torch
import torch.nn as nn

from torch.autograd import grad, Function

from pideq.deq.model import get_implicit
from pideq.utils import get_jacobian


def nmode_product(A, U, n):
    """Performs n-mode product between tensor A and matrix U.
    """
    B = torch.tensordot(A, U, ([n,], [0,]))

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

    return B.permute(torch.Size(dims))

if __name__ == '__main__':
    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    z0 = torch.zeros(batch_size, n_states).double()

    implicit = get_implicit(forward_max_steps=500, forward_eps=1e-6,
                            backward_max_steps=500, backward_eps=1e-6)

    ### SYNTHETIC TESTS

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

    # analytical Jacobian
    J_tanh = torch.diag_embed(1-z**2)
    J_fx = J_tanh @ B.weight
    J_fz = J_tanh @ A.weight
    J_zx = -torch.linalg.inv(J_fz - torch.eye(n_states)) @ J_fx

    # make_dot(z, {'z': z})
    # grad_z = grad(z, x, torch.ones_like(z), create_graph=True)[0]
    autograd_Jzx = get_jacobian(z, x)
    functional_Jzx = torch.autograd.functional.jacobian(lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias), x, create_graph=True)
    functional_Jzx = torch.diagonal(functional_Jzx, dim1=0, dim2=2).transpose(2,1).transpose(1,0)

    assert torch.isclose(autograd_Jzx, J_zx).all(), "Jacobian generated with autograd is different from the analytical one"
    assert torch.isclose(functional_Jzx, J_zx).all(), "Jacobian generated with functional.Jacobian is different from the analytical one"

    # TODO: implement analytical Hessian

    H_tanh = torch.diag_embed(J_tanh * torch.diag_embed(-2*z))

    print(nmode_product(H_tanh, B.weight, 2).shape)
    print(nmode_product(H_tanh, B.weight, 1).shape)

    autograd_Hzxx = get_jacobian(J_zx, x)

    print('DONE')
