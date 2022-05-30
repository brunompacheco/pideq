import numpy as np
import torch
import torch.nn as nn


class GetEq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z0, A_weight, A_bias, B_weight, B_bias, nonlin=torch.tanh, solver=forward_iteration):
        f = lambda x,z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

        with torch.no_grad():
            z_star = solver(
                # lambda z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias),
                lambda z: f(x,z),
                z0,
                threshold=199, eps=1e-5,
            )['result']

        ctx.save_for_backward(z_star.detach(), x, A_weight, A_bias, B_weight, B_bias)
        ctx.solver = solver
        ctx.nonlin = nonlin

        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        z, x, A_weight, A_bias, B_weight, B_bias, = ctx.saved_tensors

        f = lambda x,z: ctx.nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

        z.requires_grad_()
        with torch.enable_grad():
            f_ = f(x,z)

            grad_z = ctx.solver(
                lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[-1] + grad_output,
                torch.zeros_like(grad_output),
                threshold=199, eps=1e-5,
            )['result']

        new_grad_x = torch.autograd.grad(f_, x, grad_z, retain_graph=True)[-1]
        new_grad_A_weight = torch.autograd.grad(f_, A_weight, grad_z, retain_graph=True)[-1]
        new_grad_A_bias = torch.autograd.grad(f_, A_bias, grad_z, retain_graph=True)[-1]
        new_grad_B_weight = torch.autograd.grad(f_, B_weight, grad_z, retain_graph=True)[-1]
        new_grad_B_bias = torch.autograd.grad(f_, B_bias, grad_z)[-1]

        return new_grad_x, None, new_grad_A_weight, new_grad_A_bias, new_grad_B_weight, new_grad_B_bias, None

class DEQ(nn.Module):
    def __init__(self, n_in=1, n_states=2) -> None:
        super().__init__()

        A = torch.rand(n_in,n_states) * 2 * np.sqrt(n_in) - np.sqrt(n_in)
        self.A = nn.Parameter(A, requires_grad=True)

        B = torch.rand(n_states,n_states) * 2 * np.sqrt(n_states) - np.sqrt(n_states)
        self.B = nn.Parameter(B, requires_grad=True)