"""Test Jacobians of implicit functions with dummy data.
"""
import torch
import torch.nn as nn
import tensorly as tl
tl.set_backend('pytorch')

from pideq.deq.model import get_implicit
from pideq.utils import get_jacobian, numerical_jacobian, batched_nmode_product


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

    f_ = lambda x, z: torch.tanh(z @ A.weight.T + A.bias + x @ B.weight.T + B.bias)

    H_tanh = torch.diag_embed(J_tanh * torch.diag_embed(-2*z))

    H_fxx = tl.tenalg.mode_dot(tl.tenalg.mode_dot(H_tanh, B.weight.T, 3), B.weight.T, 2)
    def get_Jfx(x):
        x.requires_grad_()
        return get_jacobian(f_(x, z), x)
    numerical_Hfxx = numerical_jacobian(get_Jfx, x, batched=True)
    assert torch.isclose(numerical_Hfxx, H_fxx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    H_fzx = tl.tenalg.mode_dot(tl.tenalg.mode_dot(H_tanh, B.weight.T, 3), A.weight.T, 2)
    def get_Jfz(x):
        z.requires_grad_()
        return get_jacobian(f_(x, z), z)
    numerical_Hfzx = numerical_jacobian(get_Jfz, x, batched=True)
    assert torch.isclose(numerical_Hfzx, H_fzx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    H_zxx = - batched_nmode_product(
        H_fxx + batched_nmode_product(H_fzx, J_zx.transpose(1,2), 2),
        implicit_inv,
        1
    )
    def get_Jzx(x):
        x.requires_grad_()
        z = implicit(x, z0[:1], A.weight, A.bias, B.weight, B.bias)
        return get_jacobian(z, x)
    numerical_Hzxx = numerical_jacobian(get_Jzx, x[:1], batched=True)
    # numerical_Hzxx = numerical_Hzxx[0]
    assert torch.isclose(numerical_Hzxx, H_zxx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    # ignore batch
    J_zx = J_zx[0]
    implicit_inv = implicit_inv[0]

    H_fxx = H_fxx[0]
    H_fzx = H_fzx[0]

    H_zxx = torch.zeros(4,2,2)
    import numpy as np
    for (i, j, k), _ in np.ndenumerate(H_zxx.numpy()):
        # euclidian column vectors
        ei = torch.eye(4)[i].unsqueeze(-1).to(J_zx)
        ej = torch.eye(2)[j].unsqueeze(-1).to(J_zx)
        ek = torch.eye(2)[k].unsqueeze(-1).to(J_zx)
        ek = torch.eye(2)[k].to(J_zx)
        
        H_zxx[i,j,k] = ei.T @ (
            - implicit_inv @ tl.tenalg.mode_dot(H_fxx, ek, 2)
            - implicit_inv @ tl.tenalg.mode_dot(H_fzx, ek, 2) @ J_zx
        ) @ ej

    def get_Jzx(x):
        x.requires_grad_()
        z = implicit(x, z0[:1], A.weight, A.bias, B.weight, B.bias)
        return get_jacobian(z, x)
    numerical_Hzxx = numerical_jacobian(get_Jzx, x[:1], batched=True)
    numerical_Hzxx = numerical_Hzxx[0]
    assert torch.isclose(numerical_Hzxx, H_zxx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    print('DONE')
