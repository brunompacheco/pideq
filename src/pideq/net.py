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

    def forward(self, t, x):
        x_ = torch.hstack((
            2 * t / self.T - 1,
            2 * (x - self.xb[0]) / (self.xb[1] - self.xb[0]) - 1
        ))

        return super().forward(x_)
