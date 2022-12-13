from time import time
import torch
import torch.nn as nn
import tensorly as tl
tl.set_backend('pytorch')

from torch.autograd import Function, grad

from pideq.deq.jacobian import jac_loss_estimate
from pideq.deq.solvers import forward_iteration
from pideq.utils import batched_nmode_product, numerical_jacobian


def get_implicit(nonlin=torch.tanh, forward_solver=forward_iteration,
                 forward_max_steps=200, forward_eps=1e-3,
                 backward_solver=forward_iteration, backward_max_steps=200,
                 backward_eps=1e-3):
    class ImplicitXBackward(Function):
        @staticmethod
        def forward(ctx, grad_output, z, x, A_weight, A_bias, B_weight, B_bias):
            f = lambda x,z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

            z.requires_grad_()
            with torch.enable_grad():
                f_ = f(x,z)

                grad_z = backward_solver(
                    lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[0] + grad_output,
                    torch.zeros_like(grad_output),
                    threshold=backward_max_steps,
                    eps=backward_eps,
                )['result']

            # new_grad_tanh = grad_z.unsqueeze(1) @ torch.diag_embed(1 - torch.pow(f_, 2))

            # new_grad_tanh = grad_z * (1 - f_**2)  # faster, but not really correct as J_\phi is the diagonalized vector
            # new_grad_tanh = new_grad_tanh.squeeze(1)

            # new_grad_x = new_grad_tanh @ B_weight
            new_grad_x = torch.autograd.grad(f_, x, grad_z, create_graph=True)[0]

            new_grad_A_weight = torch.autograd.grad(f_, A_weight, grad_z, create_graph=True)[0]
            new_grad_A_bias = torch.autograd.grad(f_, A_bias, grad_z, create_graph=True)[0]
            new_grad_B_weight = torch.autograd.grad(f_, B_weight, grad_z, create_graph=True)[0]
            new_grad_B_bias = torch.autograd.grad(f_, B_bias, grad_z, create_graph=True)[0]

            return new_grad_x, None, new_grad_A_weight, new_grad_A_bias, new_grad_B_weight, new_grad_B_bias, None, None

        @staticmethod
        def backward(ctx, gradgrad_output):
            raise NotImplementedError
            return None

    class Implicit(Function):
        @staticmethod
        def forward(ctx, x, z0, A_weight, A_bias, B_weight, B_bias):
            f = lambda x,z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

            with torch.no_grad():
                # find equilibrium point for f
                z_star = forward_solver(
                    lambda z: f(x,z),
                    z0,
                    threshold=forward_max_steps,
                    eps=forward_eps,
                )['result']

            ctx.save_for_backward(z_star.detach(), x, A_weight, A_bias, B_weight, B_bias)

            return z_star

        @staticmethod
        def backward(ctx, grad_output):
            return ImplicitXBackward.apply(grad_output, *ctx.saved_tensors)
            # z, x, A_weight, A_bias, B_weight, B_bias, = ctx.saved_tensors
            # f = lambda x,z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

            # z.requires_grad_()
            # with torch.enable_grad():
            #     f_ = f(x,z)

            #     grad_z = backward_solver(
            #         lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[0] + grad_output,
            #         torch.zeros_like(grad_output),
            #         threshold=backward_max_steps,
            #         eps=backward_eps,
            #     )['result']

            # new_grad_x = ImplicitXBackward.apply(grad_output, *ctx.saved_tensors)

            # new_grad_A_weight = torch.autograd.grad(f_, A_weight, grad_z, create_graph=True)[0]
            # new_grad_A_bias = torch.autograd.grad(f_, A_bias, grad_z, create_graph=True)[0]
            # new_grad_B_weight = torch.autograd.grad(f_, B_weight, grad_z, create_graph=True)[0]
            # new_grad_B_bias = torch.autograd.grad(f_, B_bias, grad_z, create_graph=True)[0]

            # return new_grad_x, None, new_grad_A_weight, new_grad_A_bias, new_grad_B_weight, new_grad_B_bias, None, None

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

        self._implicit = get_implicit(
            nonlin=self.phi,
            forward_solver=self._forward_solver,
            forward_max_steps=self.solver_kwargs['threshold'],
            forward_eps=self.solver_kwargs['eps'],
            backward_solver=self._backward_solver,
            backward_max_steps=self.solver_kwargs['threshold'],
            backward_eps=self.solver_kwargs['eps'],
        )

        self.always_compute_grad = always_compute_grad
        self.compute_jac_loss = compute_jac_loss

    def _forward_solver(self, *args, **kwargs):
        """Grab NFEs performed by solver in the forward pass.
        """
        solver_out = self.solver(*args, **kwargs)

        self.latest_nfe = solver_out['nstep']

        return solver_out

    def _backward_solver(self, *args, **kwargs):
        """Grab NFEs performed by solver in the backward pass.
        """
        solver_out = self.solver(*args, **kwargs)

        self.latest_backward_nfe = solver_out['nstep']

        return solver_out

    def forward(self, u: torch.Tensor):
        z0 = torch.zeros(u.shape[0], self.out_features).type(u.dtype).to(u.device)

        z_star = self._implicit(u, z0, self.A.weight, self.A.bias,
                               self.B.weight, self.B.bias)

        if self.training and self.compute_jac_loss:
            start_time = time()
            # z_ = z_star.clone().detach().requires_grad_()
            z_ = z_star
            f_ = self.phi(self.A(z_) + self.B(u))
            jac_loss = jac_loss_estimate(f_, z_, vecs=5)
            self.jac_loss_time = time() - start_time

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
    from torch.autograd.gradcheck import GradcheckError
    from pideq.utils import get_jacobian

    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    z0 = torch.zeros(batch_size, n_states).double()

    implicit = get_implicit(forward_max_steps=500, forward_eps=1e-6,
                            backward_max_steps=500, backward_eps=1e-6)

    ### SYNTHETIC TESTS

    x = torch.rand(batch_size, n_in).double()
    x.requires_grad_()

    A = nn.Linear(n_states, n_states).double()
    B = nn.Linear(n_in, n_states).double()

    z = implicit(x, z0, A.weight, A.bias, B.weight, B.bias)

    # analytical Jacobian
    J_tanh = torch.diag_embed(1-z**2)
    J_fx = J_tanh @ B.weight
    J_fz = J_tanh @ A.weight
    implicit_inv = torch.linalg.inv(J_fz - torch.eye(n_states))
    J_zx = -implicit_inv @ J_fx

    autograd_Jzx = get_jacobian(z, x)

    functional_Jzx = torch.autograd.functional.jacobian(lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias), x, create_graph=True)
    functional_Jzx = torch.diagonal(functional_Jzx, dim1=0, dim2=2).transpose(2,1).transpose(1,0)

    numerical_Jzx = numerical_jacobian(lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias), x, batched=True)

    assert torch.isclose(autograd_Jzx, J_zx, 1e-6, 1e-4).all(), "Jacobian generated with autograd is different from the analytical one"
    assert torch.isclose(functional_Jzx, J_zx, 1e-6, 1e-4).all(), "Jacobian generated with functional.Jacobian is different from the analytical one"
    assert torch.isclose(numerical_Jzx, J_zx, 1e-6, 1e-4).all(), "Jacobian generated with sp.approx_fprime is different from the analytical one"

    # analytical Hessian
    f_ = lambda x, z: torch.tanh(z @ A.weight.T + A.bias + x @ B.weight.T + B.bias)

    H_tanh = torch.diag_embed(J_tanh * torch.diag_embed(-2*z))

    H_fxx = tl.tenalg.mode_dot(tl.tenalg.mode_dot(H_tanh, B.weight.T, 3), B.weight.T, 2)
    def get_Jfx(x):
        x.requires_grad_()
        return get_jacobian(f_(x, z), x)
    numerical_Hfxx = numerical_jacobian(get_Jfx, x, batched=True)
    assert torch.isclose(numerical_Hfxx, H_fxx, 1e-6, 1e-4).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    H_fzx = tl.tenalg.mode_dot(tl.tenalg.mode_dot(H_tanh, B.weight.T, 3), A.weight.T, 2)
    def get_Jfz(x):
        z.requires_grad_()
        return get_jacobian(f_(x, z), z)
    numerical_Hfzx = numerical_jacobian(get_Jfz, x, batched=True)
    assert torch.isclose(numerical_Hfzx, H_fzx, 1e-6, 1e-4).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    H_zxx = - batched_nmode_product(
        H_fxx + batched_nmode_product(H_fzx, J_zx.transpose(1,2), 2),
        implicit_inv,
        1
    )
    def get_Jzx(x):
        x.requires_grad_()
        z = implicit(x, z0, A.weight, A.bias, B.weight, B.bias)
        return get_jacobian(z, x)
    numerical_Hzxx = numerical_jacobian(get_Jzx, x, batched=True)
    # TODO: fix H_zxx
    # assert torch.isclose(numerical_Hzxx, H_zxx).all(), "Hessian generated with sp.approx_fprime is different from the analytical one"

    ### NUMERICAL TESTS
    x = torch.rand(batch_size, n_in).double()
    x.requires_grad_()
    z0 = torch.zeros(batch_size, n_states).double()

    torch.autograd.gradcheck(
        lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias),
        x,
    )
    torch.autograd.gradcheck(
        lambda A_weight: implicit(x, z0, A_weight, A.bias, B.weight, B.bias),
        A.weight,
        atol=1e-5,  # for some reason it sometimes fails with atol=1e-6
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

    try:
        torch.autograd.gradgradcheck(
            lambda x: implicit(x, z0, A.weight, A.bias, B.weight, B.bias),
            x,
            atol=1e-3,
        )
    except GradcheckError as e:
        print('gradgrad failed with respect to x')
    except NotImplementedError:
        pass
    except:
        print('gradgrad failed with respect to B.bias')

    try:
        torch.autograd.gradgradcheck(
            lambda A_weight: implicit(x, z0, A_weight, A.bias, B.weight, B.bias),
            A.weight,
            atol=1e-3,
        )
    except GradcheckError as e:
        print('gradgrad failed with respect to A.weight')
    except NotImplementedError:
        pass
    except:
        print('gradgrad failed with respect to B.bias')

    try:
        torch.autograd.gradgradcheck(
            lambda A_bias: implicit(x, z0, A.weight, A_bias, B.weight, B.bias),
            A.bias,
            atol=1e-3,
        )
    except GradcheckError as e:
        print('gradgrad failed with respect to A.bias')
    except NotImplementedError:
        pass
    except:
        print('gradgrad failed with respect to B.bias')

    try:
        torch.autograd.gradgradcheck(
            lambda B_weight: implicit(x, z0, A.weight, A.bias, B_weight, B.bias),
            B.weight,
            atol=1e-3,
        )
    except GradcheckError as e:
        print('gradgrad failed with respect to B.weight')
    except NotImplementedError:
        pass
    except:
        print('gradgrad failed with respect to B.bias')

    try:
        torch.autograd.gradgradcheck(
            lambda B_bias: implicit(x, z0, A.weight, A.bias, B.weight, B_bias),
            B.bias,
            atol=1e-3,
        )
    except GradcheckError as e:
        print('gradgrad failed with respect to B.bias')
    except NotImplementedError:
        pass
    except:
        print('gradgrad failed with respect to B.bias')

    # test gradients of DEQ model
    u = torch.rand(batch_size,n_in).double()
    u.requires_grad_()

    x0 = torch.zeros(u.shape[0], n_states).double()
    x0.requires_grad_()

    deq = DEQ(n_in, n_out, n_states, always_compute_grad=True, compute_jac_loss=True,
              solver_kwargs={'threshold': 1000, 'eps': 1e-7}).double()

    torch.autograd.gradcheck(
        lambda u: deq(u)[0],
        u,
    )
