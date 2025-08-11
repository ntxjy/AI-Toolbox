"""Utility functions for evaluation and analysis."""

from __future__ import annotations

import math
from typing import Tuple

import torch

__all__ = ["five_param_logistic", "fit_logistic"]


def five_param_logistic(x: torch.Tensor, b1, b2, b3, b4, b5):
    """Five parameter logistic function used in IQA evaluation."""
    return b2 + (b1 - b2) / (1 + torch.exp((b3 - x) / torch.abs(b4))) + b5 * x


def fit_logistic(x: torch.Tensor, y: torch.Tensor, iters: int = 2000, lr: float = 1e-3) -> Tuple[float, float, float, float, float]:
    """Fit the parameters of the five parameter logistic using gradient descent.

    The function avoids external dependencies (SciPy).  Parameters are
    optimised to minimise mean squared error between ``five_param_logistic(x)``
    and ``y``.
    """

    b = torch.zeros(5, requires_grad=True, dtype=x.dtype)
    optimiser = torch.optim.Adam([b], lr=lr)
    for _ in range(iters):
        optimiser.zero_grad()
        pred = five_param_logistic(x, *b)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimiser.step()
    return tuple(b.detach().cpu().tolist())
