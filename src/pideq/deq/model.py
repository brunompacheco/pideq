from turtle import forward
import numpy as np
import torch
import torch.nn as nn

from pideq.deq.jacobian import jac_loss_estimate
from pideq.deq.solvers import forward_iteration, anderson


class DEQ(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_states=2, nonlin=torch.tanh, solver=forward_iteration, solver_kwargs={'threshold': 200, 'eps':1e-3}) -> None:
        super().__init__()

        A = .1 * (torch.rand(n_states,n_in) * 2 * np.sqrt(n_in) - np.sqrt(n_in))
        self.A = nn.Parameter(A, requires_grad=True)

        B = .1 * (torch.rand(n_states,n_states) * 2 * np.sqrt(n_states) - np.sqrt(n_states))
        self.B = nn.Parameter(B, requires_grad=True)

        b = .1 * (torch.rand(n_states) * 2 * np.sqrt(n_states) - np.sqrt(n_states))
        self.b = nn.Parameter(b, requires_grad=True)

        self.h = nn.Linear(n_states, n_out)

        self.solver = solver

        self.n_in = n_in
        self.n_out = n_out
        self.n_states = n_states

        self.nonlin = nonlin

        class GetEq(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, z0, A, B, bias):
                f = lambda x,z: nonlin(x @ A.T + z @ B.T + bias)

                with torch.no_grad():
                    z_star = anderson(
                        lambda z: f(x,z),
                        z0,
                        **solver_kwargs,
                    )['result']

                ctx.save_for_backward(z_star.detach(), x, A, B, bias)

                return z_star

            @staticmethod
            def backward(ctx, grad_output):
                z, x, A, B, bias, = ctx.saved_tensors

                f = lambda x,z: nonlin(x @ A.T + z @ B.T + bias)

                z.requires_grad_()
                with torch.enable_grad():
                    f_ = f(x,z)

                    grad_z = solver(
                        lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[0] + grad_output,
                        torch.zeros_like(grad_output),
                        **solver_kwargs,
                    )['result']

                new_grad_x = torch.autograd.grad(f_, x, grad_z, retain_graph=True)[0]
                new_grad_A = torch.autograd.grad(f_, A, grad_z, retain_graph=True)[0]
                new_grad_B = torch.autograd.grad(f_, B, grad_z, retain_graph=True)[0]
                new_grad_bias = torch.autograd.grad(f_, bias, grad_z)[0]

                return new_grad_x, None, new_grad_A, new_grad_B, new_grad_bias

        self.get_eq = GetEq()


    def forward(self, x):
        z0 = torch.zeros(x.shape[0], self.n_states).to(x)
        z_star = self.get_eq.apply(x, z0, self.A, self.B, self.b)

        y = self.h(z_star)

        f = lambda x,z: torch.tanh(x @ self.A.T + z @ self.B.T + self.b)

        return y, jac_loss_estimate(f(x,z_star), z_star)


if __name__ == '__main__':
    # test gradients of DEQ model

    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    x = torch.rand(batch_size,n_in).double()
    x.requires_grad_()

    z0 = torch.zeros(x.shape[0], n_states).double()
    z0.requires_grad_()

    deq = DEQ(n_in, n_out, n_states, solver_kwargs={'threshold': 1000, 'eps': 1e-7})
    deq = deq.double()

    torch.autograd.gradcheck(
    lambda x: deq.get_eq.apply(x, z0, deq.A, deq.B, deq.b),
        x,
        eps=1e-4,
        atol=1e-5,
    )

    torch.autograd.gradcheck(
        lambda x: deq(x)[0],
        x,
        eps=1e-4,
        atol=1e-5,
    )

    # TODO: solve poor jacobian gradient
    # torch.autograd.gradcheck(
    #     lambda x: deq(x)[1],
    #     x,
    #     eps=1e-1,
    #     atol=1e-4,
    # )