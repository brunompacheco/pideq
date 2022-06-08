import numpy as np
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, T: float, n_in=1, n_out=4,
                 y0=np.array([12.6, 13.0, 4.8, 4.9]), Nonlin=nn.Tanh,
                 n_hidden=5, n_nodes=20) -> None:
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

        assert y0.shape[0] == n_out

        self.y0 = torch.Tensor(y0)

    def forward(self, t):
        # rescaling the input => better convergence
        # y = self.fcn(t / self.T) + self.eps
        y = self.fcn(t / self.T)

        return (y / 100.) + self.y0.to(y)
