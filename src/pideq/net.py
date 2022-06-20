import numpy as np
import torch
import torch.nn as nn

from pideq.deq.model import DEQ
from pideq.deq.solvers import anderson, forward_iteration


class PINN(nn.Module):
    def __init__(self, T: float, n_in=1, n_out=2,
                 y0=np.array([0., .1]), Nonlin=nn.Tanh,
                 n_hidden=4, n_nodes=20) -> None:
        super().__init__()

        self.T = T

        l = list()
        l.append(nn.Linear(n_in, n_nodes))
        l.append(Nonlin())

        for _ in range(n_hidden-1):
            l.append(nn.Linear(n_nodes, n_nodes))
            l.append(Nonlin())

        l.append(nn.Linear(n_nodes, n_out))

        # l.append(nn.ReLU())  # avoids negative outputs, which are out-of-bounds

        self.fcn = nn.Sequential(*l)

        # self.eps = 1e-3  # added to the output, avoids sqrt gradient at 0

        # assert y0.shape[0] == n_out

        self.y0 = torch.Tensor(y0)

    def forward(self, t):
        # rescaling the input => better convergence
        # y = self.fcn(t / self.T) + self.eps
        y = self.fcn(t / self.T)

        return (y / 100.) + self.y0.to(y)

class PIDEQ(DEQ):
    def __init__(self, T: float, y0=np.array([0., .1]), n_in=1, n_out=2,
                 n_states=20, nonlin=torch.tanh, always_compute_grad=False,
                 solver=anderson, solver_kwargs={'threshold': 200, 'eps':1e-3}
                ) -> None:
        super().__init__(n_in, n_out, n_states, nonlin, always_compute_grad, solver, solver_kwargs)

        self.T = T

        # assert y0.shape[0] == n_out

        self.y0 = torch.Tensor(y0)

    def forward(self, t):
        # rescaling the input => better convergence
        # t_ = t / self.T
        t_ = t

        if self.training:
            y_, jac_loss = super().forward(t_)
        else:
            y_ = super().forward(t_)

        y = y_ + self.y0.to(y_)[0]

        if self.training:
            return y, jac_loss
        else:
            return y
