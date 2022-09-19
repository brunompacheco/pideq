import numpy as np
import torch
import torch.nn as nn

from pideq.deq.model import DEQ
from pideq.deq.solvers import anderson, forward_iteration


class PINN(nn.Module):
    def __init__(self, T: float, n_in=2, n_out=2, Nonlin=nn.Tanh,
                 n_hidden=4, n_nodes=100, xb=[-5, 5]) -> None:
        super().__init__()

        self.T = T

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

        # self.eps = 1e-3  # added to the output, avoids sqrt gradient at 0

        # assert y0.shape[0] == n_out

        # self.y0 = torch.Tensor(y0)

    def forward(self, t, x):
        # rescaling the input => better convergence
        # y = self.fcn(t / self.T) + self.eps
        x_ = torch.hstack((
            2 * t / self.T - 1,
            2 * (x - self.xb[0]) / (self.xb[1] - self.xb[0]) - 1
        ))

        h = self.fcn(x_)

        return h

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

class PIDEQ(DEQ):
    def __init__(self, T: float, n_in=2, n_out=2, n_states=100, n_hidden=1,
                 nonlin=torch.tanh, always_compute_grad=False, solver=anderson,
                 solver_kwargs={'threshold': 200, 'eps':1e-4}, xb=[-5, 5],
                ) -> None:
        super().__init__(n_in, n_out, n_states, n_hidden, nonlin, always_compute_grad, solver, solver_kwargs)

        self.T = T

        self.xb = np.array(xb)

    @classmethod
    def from_pinn(cls, pinn: PINN):
        # compute number of states necessary to simulate the ANN
        n_states = 0
        for l in pinn.fcn[:-2]:  # exclude ouptut layer
            if isinstance(l, nn.Linear):
                n_states += l.out_features

        deq = cls(
            pinn.T,
            n_in=pinn.fcn[0].in_features,
            n_out=pinn.fcn[-1].out_features,
            n_states=n_states,
            n_hidden=0,
            solver=forward_iteration,
        )

        # build A matrix from first layer

        l1 = pinn.fcn[0]  # first layer

        A_w = torch.zeros_like(deq.A.weight)
        A_w[:l1.weight.shape[0]] = l1.weight
        deq.A.weight = nn.Parameter(A_w)

        A_b = torch.zeros_like(deq.A.bias)
        A_b[:l1.bias.shape[0]] = l1.bias
        deq.A.bias = nn.Parameter(A_b)

        # build B matrix from hidden layers
        B_w = torch.zeros_like(deq.B.weight)
        B_b = torch.zeros_like(deq.B.bias)

        l0 = pinn.fcn[0].out_features  # end of the last hidden layer's output
        for l in pinn.fcn[1:-2]:  # skip first and last layers
            if isinstance(l, nn.Linear):
                B_w[l0:l0 + l.out_features,l0 - l.in_features:l0] = l.weight
                B_b[l0:l0 + l.out_features] = l.bias
                l0 = l0 + l.out_features

        deq.B.weight = nn.Parameter(B_w)
        deq.B.bias = nn.Parameter(B_b)

        # build h function from last layer
        ll = pinn.fcn[-1]  # last layer

        h_w = torch.zeros_like(deq.h.weight)
        h_w[:,-ll.weight.shape[-1]:] = ll.weight
        deq.h.weight = nn.Parameter(h_w)

        h_b = torch.zeros_like(deq.h.bias)
        h_b[-ll.bias.shape[0]:] = ll.bias
        deq.h.bias = nn.Parameter(h_b)

        return deq

    def forward(self, t, x):
        x_ = torch.hstack((
            2 * t / self.T - 1,
            2 * (x - self.xb[0]) / (self.xb[1] - self.xb[0]) - 1
        ))

        return super().forward(x_)
