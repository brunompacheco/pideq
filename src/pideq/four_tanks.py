import torch


def four_tanks(y, u):
    assert y.device == u.device

    g = [.43, .34]
    k = [3.14, 3.29]
    K = torch.Tensor([
        [g[0]*k[0], 0],
        [0, g[1]*k[1]],
        [0, (1-g[1])*k[1]],
        [(1-g[0])*k[0], 0],
    ]).to(y)

    a = [.071, .057, .071, .057]
    g = 981
    W = torch.Tensor([
        [-a[0],     0,  a[2],     0],
        [    0, -a[1],     0,  a[3]],
        [    0,     0, -a[2],     0],
        [    0,     0,     0, -a[3]],
    ]).to(y)

    A = torch.Tensor([1/28, 1/32, 1/28, 1/32]).unsqueeze(0).to(y)

    return A * (u @ K.T + torch.sqrt(2*g*y) @ W.T)
