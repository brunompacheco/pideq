# Based on https://github.com/locuslab/deq/blob/1fb7059d6d89bb26d16da80ab9489dcc73fc5472/lib/jacobian.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def jac_loss_estimate(f0, z0, vecs=2, create_graph=True):
    """Estimating tr(J^TJ)=tr(JJ^T) via Hutchinson estimator

    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        vecs (int, optional): Number of random Gaussian vectors to use. Defaults to 2.
        create_graph (bool, optional): Whether to create backward graph (e.g., to train on this loss). 
                                       Defaults to True.

    Returns:
        torch.Tensor: A 1x1 torch tensor that encodes the (shape-normalized) jacobian loss
    """
    vecs = vecs
    result = 0
    for i in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.norm()**2
    return result / vecs / np.prod(z0.shape)

def power_method(f0, z0, n_iters=200):
    """Estimating the spectral radius of J using power method

    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        n_iters (int, optional): Number of power method iterations. Defaults to 200.

    Returns:
        tuple: (largest eigenvector, largest (abs.) eigenvalue)
    """
    evector = torch.randn_like(z0)
    bsz = evector.shape[0]
    for i in range(n_iters):
        vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
        evalue = (vTJ * evector).reshape(bsz, -1).sum(1, keepdim=True) / (evector * evector).reshape(bsz, -1).sum(1, keepdim=True)
        evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
    return (evector, torch.abs(evalue))


if __name__ == '__main__':
    from pideq.deq.model import get_implicit
    from torch.autograd.gradcheck import GradcheckError

    batch_size = 5

    n_in = 2
    n_out = 3
    n_states = 4

    x = torch.rand(batch_size, n_in).double()
    x.requires_grad_()
    z0 = torch.zeros(batch_size, n_states).double()

    A = nn.Linear(n_states, n_states).double()
    B = nn.Linear(n_in, n_states).double()

    implicit = get_implicit(forward_max_steps=500, forward_eps=1e-6,
                            backward_max_steps=500, backward_eps=1e-6)
    
    def get_jac_loss(x, A_weight, A_bias, B_weight, B_bias):
        z0 = torch.zeros(batch_size, n_states).double()

        z = implicit(x, z0, A_weight, A_bias, B_weight, B_bias)

        # Also failed with power method
        # return power_method(
        #     torch.tanh(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias),
        #     z,
        #     n_iters=500,
        # )[1]

        return jac_loss_estimate(
            torch.tanh(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias),
            z,
            vecs=2
        )

    # TODO: check why gradcheck of jac loss fails
    try:
        torch.autograd.gradcheck(
            lambda x: get_jac_loss(x, A.weight, A.bias, B.weight, B.bias),
            x,
        )
    except GradcheckError as e:
        print('gradcheck failed with respect to x')

    try:
        torch.autograd.gradcheck(
            lambda A_weight: get_jac_loss(x, A_weight, A.bias, B.weight, B.bias),
            A.weight,
        )
    except GradcheckError as e:
        print('gradcheck failed with respect to A.weight')

    try:
        torch.autograd.gradcheck(
            lambda A_bias: get_jac_loss(x, A.weight, A_bias, B.weight, B.bias),
            A.bias,
        )
    except GradcheckError as e:
        print('gradcheck failed with respect to A.bias')

    try:
        torch.autograd.gradcheck(
            lambda B_weight: get_jac_loss(x, A.weight, A.bias, B_weight, B.bias),
            B.weight,
        )
    except GradcheckError as e:
        print('gradcheck failed with respect to A.weight')

    try:
        torch.autograd.gradcheck(
            lambda B_bias: get_jac_loss(x, A.weight, A.bias, B.weight, B_bias),
            B.bias,
        )
    except GradcheckError as e:
        print('gradcheck failed with respect to A.bias')
