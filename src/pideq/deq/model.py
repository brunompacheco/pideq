from time import time
from turtle import forward
import numpy as np
import torch
import torch.nn as nn

from pideq.deq.jacobian import jac_loss_estimate
from pideq.deq.solvers import forward_iteration, anderson


# class DEQ(nn.Module):
#     def __init__(self, n_in=1, n_out=1, n_states=2, nonlin=torch.tanh,
#                  solver=forward_iteration,
#                  solver_kwargs={'threshold': 200, 'eps':1e-3}) -> None:
#         super().__init__()

#         A = .01 * (torch.rand(n_states,n_in) * 2 * np.sqrt(n_in) - np.sqrt(n_in))
#         self.A = nn.Parameter(A, requires_grad=True)

#         B = .01 * (torch.rand(n_states,n_states) * 2 * np.sqrt(n_states) - np.sqrt(n_states))
#         self.B = nn.Parameter(B, requires_grad=True)

#         b = .01 * (torch.rand(n_states) * 2 * np.sqrt(n_states) - np.sqrt(n_states))
#         self.b = nn.Parameter(b, requires_grad=True)

#         self.h = nn.Linear(n_states, n_out)

#         self.solver = solver

#         self.n_in = n_in
#         self.n_out = n_out
#         self.n_states = n_states

#         self.nonlin = nonlin

#         class GetEq(torch.autograd.Function):
#             @staticmethod
#             def forward(ctx, x, z0, A, B, bias):
#                 f = lambda x,z: nonlin(x @ A.T + z @ B.T + bias)

#                 with torch.no_grad():
#                     z_star = anderson(
#                         lambda z: f(x,z),
#                         z0,
#                         **solver_kwargs,
#                     )['result']

#                 ctx.save_for_backward(z_star.detach(), x, A, B, bias)

#                 return z_star

#             @staticmethod
#             def backward(ctx, grad_output):
#                 z, x, A, B, bias, = ctx.saved_tensors

#                 f = lambda x,z: nonlin(x @ A.T + z @ B.T + bias)

#                 z.requires_grad_()
#                 with torch.enable_grad():
#                     f_ = f(x,z)

#                     grad_z = solver(
#                         lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[0] + grad_output,
#                         torch.zeros_like(grad_output),
#                         **solver_kwargs,
#                     )['result']

#                 new_grad_x = torch.autograd.grad(f_, x, grad_z, retain_graph=True)[0]
#                 new_grad_A = torch.autograd.grad(f_, A, grad_z, retain_graph=True)[0]
#                 new_grad_B = torch.autograd.grad(f_, B, grad_z, retain_graph=True)[0]
#                 new_grad_bias = torch.autograd.grad(f_, bias, grad_z)[0]

#                 return new_grad_x, None, new_grad_A, new_grad_B, new_grad_bias

#         self.get_eq = GetEq()

#     def forward(self, x):
#         z0 = torch.zeros(x.shape[0], self.n_states).to(x)
#         z_star = self.get_eq.apply(x, z0, self.A, self.B, self.b)

#         y = self.h(z_star)

#         f = lambda x,z: torch.tanh(x @ self.A.T + z @ self.B.T + self.b)

#         return y, jac_loss_estimate(f(x,z_star), z_star)

class DEQ(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_states=20, solver=forward_iteration,
                 phi=torch.tanh, always_compute_grad=False, compute_jac_loss=True,
                 solver_kwargs={'threshold': 200, 'eps':1e-3},
                 weight_initialization_factor=.1) -> None:
        super().__init__()

        self.n_states = n_states

        self.A = nn.Linear(n_states,n_states)
        self.B = nn.Linear(n_in,n_states)
        self.C = nn.Linear(n_states,n_out)
        self.D = nn.Linear(n_in,n_out)

        self.phi = phi

        # decreasing initial weights to increase stability
        self.A.weight = nn.Parameter(weight_initialization_factor * self.A.weight)
        self.B.weight = nn.Parameter(weight_initialization_factor * self.B.weight)
        self.C.weight = nn.Parameter(weight_initialization_factor * self.C.weight)
        self.D.weight = nn.Parameter(weight_initialization_factor * self.D.weight)

        self.solver = solver
        self.solver_kwargs = solver_kwargs

        self.always_compute_grad = always_compute_grad
        self.compute_jac_loss = compute_jac_loss

    def f(self, u, x):
        return self.phi(self.A(x) + self.B(u))

    def forward(self, u: torch.Tensor):
        x0 = torch.zeros(u.shape[0], self.n_states).type(u.dtype).to(u.device)

        # compute forward pass
        with torch.no_grad():
            solver_out = self.solver(
                lambda x : self.f(u, x),
                x0,
                **self.solver_kwargs
            )
            x_star_ = solver_out['result']
            self.latest_nfe = solver_out['nstep']
        x_star = self.f(u, x_star_)

        # (Prepare for) Backward pass, see step 3 above
        if self.training or self.always_compute_grad:
            x_ = x_star.clone().detach().requires_grad_()
            # x_star.requires_grad_()
            # re-engage autograd. this is necessary to add the df/d(*) hook
            f_ = self.f(u, x_)

            # Jacobian-related computations, see additional step above. For instance:
            if self.training and self.compute_jac_loss:
                start_time = time()
                jac_loss = jac_loss_estimate(f_, x_, vecs=1)
                self.jac_loss_time = time() - start_time

            # new_x_start already has the df/d(*) hook, but the J_g^-1 must be added mannually
            def backward_hook(grad):
                # the following is necessary to add breakpoints here
                # import pydevd
                # pydevd.settrace(suspend=False, trace_only_current_thread=True)

                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion

                # this fixes a bug that happens every now and then, that the
                # backwards graph of f_ disappears in the second call of this
                # hook during loss.backward()
                with torch.enable_grad():
                    f_ = self.f(u, x_)

                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                # forward iteration is the only solver through which I could backprop (tested with gradgradcheck)
                backward_solver_out = forward_iteration(
                    lambda y: torch.autograd.grad(f_, x_, y, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad),
                    **self.solver_kwargs
                )
                new_grad = backward_solver_out['result']
                self.latest_backward_nfe = backward_solver_out['nstep']

                return new_grad

            self.hook = x_star.register_hook(backward_hook)

        y = self.C(x_star) + self.D(u)

        if self.training and self.compute_jac_loss:
            return y, jac_loss
        else:
            return y

if __name__ == '__main__':
    # test gradients of DEQ model

    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    u = torch.rand(batch_size,n_in).double()
    u.requires_grad_()

    x0 = torch.zeros(u.shape[0], n_states).double()
    x0.requires_grad_()

    deq = DEQ(n_in, n_out, n_states, always_compute_grad=True,
              solver_kwargs={'threshold': 1000, 'eps': 1e-7})
    deq = deq.double()

    # torch.autograd.gradcheck(
    # lambda x: deq.get_eq.apply(u, x0, deq.A, deq.B, deq.b),
    #     u,
    #     eps=1e-4,
    #     atol=1e-5,
    # )

    torch.autograd.gradcheck(
        lambda u: deq(u)[0],
        u,
        eps=1e-4,
        atol=1e-5,
        check_undefined_grad=False,
    )

    # TODO: solve poor jacobian gradient
    # torch.autograd.gradcheck(
    #     lambda x: deq(x)[1],
    #     x,
    #     eps=1e-1,
    #     atol=1e-4,
    # )