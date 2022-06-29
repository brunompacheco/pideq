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
    def __init__(self, n_in=1, n_out=1, n_states=20, n_hidden=0,
                 nonlin=torch.tanh, always_compute_grad=False, solver=anderson,
                 solver_kwargs={'threshold': 200, 'eps':1e-3}) -> None:
        super().__init__()

        self.n_states = n_states
        self.n_hidden = n_hidden

        self.B = nn.Linear(n_states,n_states)
        self.A = nn.Linear(n_in,n_states)

        self.nonlin = nonlin

        if self.n_hidden > 0:
            hidden_layers = list()
            for _ in range(n_hidden):
                linear = nn.Linear(n_states, n_states)
                linear.weight = nn.Parameter(0.1 * linear.weight)
                hidden_layers += [
                    linear,
                    nn.Tanh(),  # TODO refactor nonlin
                ]
            self.hidden = nn.Sequential(*hidden_layers)
        else:
            self.hidden = lambda x: x

        # decreasing initial weights to increase stability
        self.A.weight = nn.Parameter(0.1 * self.A.weight)
        self.B.weight = nn.Parameter(0.1 * self.B.weight)

        self.solver = solver
        self.solver_kwargs = solver_kwargs

        self.always_compute_grad = always_compute_grad

        self.h = nn.Linear(n_states, n_out)

    def f(self, x, z):
        y = self.nonlin(self.A(x) + self.B(z))

        return self.hidden(y)


    def forward(self, x: torch.Tensor):
        z0 = torch.zeros(x.shape[0], self.n_states).type(x.dtype).to(x.device)

        # compute forward pass
        with torch.no_grad():
            solver_out = self.solver(
                lambda z : self.f(x, z),
                z0,
                **self.solver_kwargs
            )
            z_star_ = solver_out['result']
            self.latest_nfe = solver_out['nstep']
        z_star = self.f(x, z_star_)

        # (Prepare for) Backward pass, see step 3 above
        if self.training or self.always_compute_grad:
            z_ = z_star.clone().detach().requires_grad_()
            # z_star.requires_grad_()
            # re-engage autograd. this is necessary to add the df/d(*) hook
            f_ = self.f(x, z_)

            # Jacobian-related computations, see additional step above. For instance:
            if self.training:
                jac_loss = jac_loss_estimate(f_, z_, vecs=1)

            # new_z_start already has the df/d(*) hook, but the J_g^-1 must be added mannually
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
                    f_ = self.f(x, z_)

                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                # forward iteration is the only solver through which I could backprop (tested with gradgradcheck)
                backward_solver_out = forward_iteration(
                    lambda y: torch.autograd.grad(f_, z_, y, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad),
                    **self.solver_kwargs
                )
                new_grad = backward_solver_out['result']
                self.latest_backward_nfe = backward_solver_out['nstep']

                return new_grad

            self.hook = z_star.register_hook(backward_hook)

        y = self.h(z_star)

        if self.training:
            return y, jac_loss
        else:
            return y

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

    deq = DEQ(n_in, n_out, n_states, always_compute_grad=True,
              solver_kwargs={'threshold': 1000, 'eps': 1e-7})
    deq = deq.double()

    # torch.autograd.gradcheck(
    # lambda x: deq.get_eq.apply(x, z0, deq.A, deq.B, deq.b),
    #     x,
    #     eps=1e-4,
    #     atol=1e-5,
    # )

    torch.autograd.gradcheck(
        lambda x: deq(x)[0],
        x,
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