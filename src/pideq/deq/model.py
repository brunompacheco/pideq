from time import time
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Function, grad

from pideq.deq.jacobian import jac_loss_estimate
from pideq.deq.solvers import forward_iteration


def get_implicit(nonlin=torch.tanh, solver=forward_iteration,
                 forward_max_steps=200, forward_eps=1e-3,
                 backward_max_steps=200, backward_eps=1e-3):
    class Implicit(Function):
        @staticmethod
        def forward(ctx, x, z0, A_weight, A_bias, B_weight, B_bias):
            f = lambda x,z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

            with torch.no_grad():
                # find equilibrium point for f
                z_star = solver(
                    lambda z: f(x,z),
                    z0,
                    threshold=forward_max_steps,
                    eps=forward_eps,
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
                    lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[0] + grad_output,
                    torch.zeros_like(grad_output),
                    threshold=backward_max_steps,
                    eps=backward_eps,
                )['result']

            new_grad_x = torch.autograd.grad(f_, x, grad_z, retain_graph=True)[0]
            new_grad_A_weight = torch.autograd.grad(f_, A_weight, grad_z, retain_graph=True)[0]
            new_grad_A_bias = torch.autograd.grad(f_, A_bias, grad_z, retain_graph=True)[0]
            new_grad_B_weight = torch.autograd.grad(f_, B_weight, grad_z, retain_graph=True)[0]
            new_grad_B_bias = torch.autograd.grad(f_, B_bias, grad_z, retain_graph=True)[0]

            return new_grad_x, None, new_grad_A_weight, new_grad_A_bias, new_grad_B_weight, new_grad_B_bias, None, None

    return Implicit.apply

class ImplicitLayer(nn.Module):
    """ Implicit layer inspired in Ghaoui's formulation.

    Given an input `u`, returns the vector `z*` that solves
        z* = phi(Az* + Bu)
    where A and B are linear transformations (with bias).
    """
    def __init__(self, in_features=1, out_features=1, phi=torch.tanh,
                 solver=forward_iteration, always_compute_grad=False,
                 compute_jac_loss=True,
                 solver_kwargs={'threshold': 200, 'eps':1e-3},
                 weight_initialization_factor=.1) -> None:
        super().__init__()

        self.out_features = out_features
        self.in_features = in_features

        self.A = nn.Linear(self.out_features, self.out_features)
        self.B = nn.Linear(self.in_features, self.out_features)

        self.phi = phi

        # decreasing initial weights to increase stability
        self.A.weight = nn.Parameter(weight_initialization_factor * self.A.weight)
        self.B.weight = nn.Parameter(weight_initialization_factor * self.B.weight)

        self.solver = solver
        self.solver_kwargs = solver_kwargs

        self.always_compute_grad = always_compute_grad
        self.compute_jac_loss = compute_jac_loss

    def f(self, u, z):
        return self.phi(self.A(z) + self.B(u))

    def forward(self, u: torch.Tensor):
        """Based on Locus Lab's implementation.
        """
        z0 = torch.zeros(u.shape[0], self.out_features).type(u.dtype).to(u.device)

        # compute forward pass
        with torch.no_grad():
            solver_out = self.solver(
                lambda z : self.f(u, z),
                z0,
                **self.solver_kwargs
            )
            z_star_ = solver_out['result']
            self.latest_nfe = solver_out['nstep']
        z_star = self.f(u, z_star_)

        # (Prepare for) Backward pass
        if self.training or self.always_compute_grad:
            z_ = z_star.clone().detach().requires_grad_()

            # re-engage autograd. this is necessary to add the df/d(*) hook
            f_ = self.f(u, z_)

            # Jacobian-related computations
            if self.training and self.compute_jac_loss:
                start_time = time()
                jac_loss = jac_loss_estimate(f_, z_, vecs=1)
                self.jac_loss_time = time() - start_time

            # new_z_start already has the df/d(*) hook, but the J_g^-1 must be added mannually
            def backward_hook(grad):
                # the following is necessary to add breakpoints here:
                # import pydevd
                # pydevd.settrace(suspend=False, trace_only_current_thread=True)

                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion

                # this fixes a bug that happens every now and then, that the
                # backwards graph of f_ disappears in the second call of this
                # hook during loss.backward()
                with torch.enable_grad():
                    f_ = self.f(u, z_)

                # Compute the fixed point of yJ + grad, where J=J_f is the
                # Jacobian of f at z_star. `forward iteration` is the only
                # solver through which I could backprop (tested with
                # gradgradcheck).
                backward_solver_out = forward_iteration(
                    lambda y: torch.autograd.grad(f_, z_, y, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad),
                    **self.solver_kwargs
                )
                new_grad = backward_solver_out['result']
                self.latest_backward_nfe = backward_solver_out['nstep']

                return new_grad

            self.hook = z_star.register_hook(backward_hook)

        if self.training and self.compute_jac_loss:
            return z_star, jac_loss
        else:
            return z_star

class DEQ(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_states=20, solver=forward_iteration,
                 phi=torch.tanh, always_compute_grad=False, compute_jac_loss=True,
                 solver_kwargs={'threshold': 200, 'eps':1e-3},
                 weight_initialization_factor=.1) -> None:
        super().__init__()

        self.n_states = n_states

        self.implicit = ImplicitLayer(n_in, n_states, phi, solver,
                                      always_compute_grad, compute_jac_loss,
                                      solver_kwargs,
                                      weight_initialization_factor)

        self.C = nn.Linear(n_states,n_out)
        self.D = nn.Linear(n_in,n_out)

    def forward(self, u: torch.Tensor):
        if self.training and self.implicit.compute_jac_loss:
            z_star, jac_loss = self.implicit(u)
        else:
            z_star = self.implicit(u)

        y = self.C(z_star) + self.D(u)

        if self.training and self.implicit.compute_jac_loss:
            return y, jac_loss
        else:
            return y

if __name__ == '__main__':
    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    # test functional implicit model implementation
    implicit = get_implicit(forward_max_steps=500, forward_eps=1e-6,
                            backward_max_steps=500, backward_eps=1e-6)

    x = torch.rand(batch_size, n_in).double()
    x.requires_grad_()
    z0 = torch.zeros(batch_size, n_states).double()

    A = nn.Linear(n_states, n_states).double()
    B = nn.Linear(n_in, n_states).double()

    z = implicit(x, z0, A.weight, A.bias, B.weight, B.bias)
    # make_dot(z, {'z': z})
    grad_z = grad(z, x, torch.ones_like(z))[0]

    torch.autograd.gradcheck(
        lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias),
        x,
    )
    torch.autograd.gradcheck(
        lambda A_weight: implicit(x, z0, A_weight, A.bias, B.weight, B.bias),
        A.weight,
    )
    torch.autograd.gradcheck(
        lambda A_bias: implicit(x, z0, A.weight, A_bias, B.weight, B.bias),
        A.bias,
    )
    torch.autograd.gradcheck(
        lambda B_weight: implicit(x, z0, A.weight, A.bias, B_weight, B.bias),
        B.weight,
    )
    torch.autograd.gradcheck(
        lambda B_bias: implicit(x, z0, A.weight, A.bias, B.weight, B_bias),
        B.bias,
    )

    torch.autograd.gradgradcheck(
        lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias),
        x,
        fast_mode=True,
        # torch.eye(n_states)[0].double().repeat(5,1),
        # is_grads_batched=True,
    )
    # torch.autograd.gradgradcheck(
    #     lambda A_weight: implicit(x, z0, A_weight, A.bias, B.weight, B.bias),
    #     A.weight,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda A_bias: implicit(x, z0, A.weight, A_bias, B.weight, B.bias),
    #     A.bias,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda B_weight: implicit(x, z0, A.weight, A.bias, B_weight, B.bias),
    #     B.weight,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda B_bias: implicit(x, z0, A.weight, A.bias, B.weight, B_bias),
    #     B.bias,
    # )

    # test gradients of DEQ model
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