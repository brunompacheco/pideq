from abc import ABC, abstractmethod
from turtle import forward

import numpy as np
import torch
import torch.nn as nn

from pideq.deq.model import DEQ
from pideq.deq.solvers import anderson, forward_iteration


class PhysicsInformedModel(nn.Module,ABC):
    def __init__(self, T: float) -> None:
        super().__init__()

        self.T = T
    
    @abstractmethod
    def _forward(self, x):
        """Run inner model on (already stacked and normalized) input.
        """
    
    def forward(self, t, x):
        # rescaling the input => better convergence
        # y = self.fcn(t / self.T) + self.eps
        x_ = torch.hstack((
            2 * t / self.T - 1,
            2 * (x - self.xb[0]) / (self.xb[1] - self.xb[0]) - 1
        ))

        return self._forward(x_)

class PINN(PhysicsInformedModel):
    def __init__(self, T: float, n_in=2, n_out=2, Nonlin=nn.Tanh,
                 n_hidden=4, n_nodes=100, xb=[-5, 5]) -> None:
        super().__init__(T)

        self.xb = np.array(xb)

        l = list()
        l.append(nn.Linear(n_in, n_nodes))
        l.append(Nonlin())

        for _ in range(n_hidden-1):
            l.append(nn.Linear(n_nodes, n_nodes))
            l.append(Nonlin())

        l.append(nn.Linear(n_nodes, n_out))

        # l.append(nn.ReLU())  # avoids negative outputs, which are out-of-bounds

        self.fcn = nn.Sequential(*l)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.fcn.apply(init_weights)

    def _forward(self, x_):
        return self.fcn(x_)

class PINC(PINN):
    def __init__(self, T: float, n_out=2, y_bounds=np.array([[-5, 5], [-5, 5]]),
                 Nonlin=nn.Tanh, n_hidden=4, n_nodes=20) -> None:
        self.y_bounds = y_bounds

        super().__init__(T, self.y_bounds.shape[0] + 1, n_out,
                         y_bounds.mean(axis=-1), Nonlin, n_hidden, n_nodes)
    
    def forward(self, y0, t):
        # rescaling the input => better convergence
        t_ = t / self.T

        x = torch.cat([t_, y0], axis=-1)

        y = self.fcn(x)

        y_range = self.y_bounds[:,1] - self.y_bounds[:,0]
        y_range = torch.from_numpy(y_range)

        return y * y_range.to(y) + self.y0.to(y)

class PIDEQ(PhysicsInformedModel,DEQ):
    def __init__(self, T: float, n_in=2, n_out=2, n_states=200, compute_jac_loss=False,
                 nonlin=torch.tanh, always_compute_grad=False, solver=forward_iteration,
                 solver_kwargs={'threshold': 200, 'eps':1e-4}, xb=[-5, 5],
                 weight_initialization_factor=.1) -> None:
        PhysicsInformedModel.__init__(self,T)
        DEQ.__init__(
            self,
            n_in=n_in,
            n_out=n_out,
            n_states=n_states,
            phi=nonlin,
            always_compute_grad=always_compute_grad,
            compute_jac_loss=compute_jac_loss,
            solver=solver,
            solver_kwargs=solver_kwargs,
            weight_initialization_factor=weight_initialization_factor,
        )

        self.xb = np.array(xb)

    def _forward(self, x_):
        return DEQ.forward(self, x_)

    @classmethod
    def from_pinn(cls, pinn: PINN):
        # compute number of states necessary to simulate the ANN
        n_states = 0
        for l in pinn.fcn[:-2]:  # exclude ouptut layer
            if isinstance(l, nn.Linear):
                n_states += l.out_features

        self = cls(
            T=pinn.T,
            n_in=pinn.fcn[0].in_features,
            n_out=pinn.fcn[-1].out_features,
            n_states=n_states,
            solver=forward_iteration,
        )

        # build B matrix from first layer
        l1 = pinn.fcn[0]  # first layer

        B_w = torch.zeros_like(self.B.weight)
        B_w[:l1.weight.shape[0]] = l1.weight
        self.B.weight = nn.Parameter(B_w)

        B_b = torch.zeros_like(self.B.bias)
        B_b[:l1.bias.shape[0]] = l1.bias
        self.B.bias = nn.Parameter(B_b)

        # build A matrix from hidden layers
        A_w = torch.zeros_like(self.A.weight)
        A_b = torch.zeros_like(self.A.bias)

        l0 = pinn.fcn[0].out_features  # end of the last hidden layer's output
        for l in pinn.fcn[1:-2]:  # skip first and last layers
            if isinstance(l, nn.Linear):
                A_w[l0:l0 + l.out_features,l0 - l.in_features:l0] = l.weight
                A_b[l0:l0 + l.out_features] = l.bias
                l0 = l0 + l.out_features

        self.A.weight = nn.Parameter(A_w)
        self.A.bias = nn.Parameter(A_b)

        # build C matrix from last layer
        ll = pinn.fcn[-1]  # last layer

        C_w = torch.zeros_like(self.C.weight)
        C_w[:,-ll.weight.shape[-1]:] = ll.weight
        self.C.weight = nn.Parameter(C_w)

        C_b = torch.zeros_like(self.C.bias)
        C_b[-ll.bias.shape[0]:] = ll.bias
        self.C.bias = nn.Parameter(C_b)

        # build (erase) D matrix
        self.D.weight = nn.Parameter(torch.zeros_like(self.D.weight))
        self.D.bias = nn.Parameter(torch.zeros_like(self.D.bias))

        return self
