from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def finite_difference_gradient(fun, x: "Tensor", h: float = 1e-5) -> "Tensor":
    """Approximate the gradient of fun at x using central finite differences.

    Parameters
    ----------
    fun : callable
        Scalar function, fun(x: Tensor) -> Tensor. Must return a scalar Tensor.
    x : Tensor
        1-D tensor. Detached and cloned internally before perturbation.
    h : float
        Step size. Default 1e-5.

    Returns
    -------
    Tensor
        Gradient approximation, same shape and dtype as x.
        Costs 2 * x.numel() function evaluations.
    """
    import torch  # lazy import — torch is optional for pure HTTP SDK usage

    x_base = x.detach().clone().double()
    grad = torch.zeros_like(x_base)

    for i in range(x_base.numel()):
        x_plus = x_base.clone()
        x_minus = x_base.clone()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (fun(x_plus) - fun(x_minus)).item() / (2 * h)

    return grad.to(dtype=x.dtype)
