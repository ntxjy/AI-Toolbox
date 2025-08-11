"""Loss functions for multi-view MOS prediction.

This module collects the different loss components described in the
specification.  They are implemented as lightweight PyTorch functions so
that they can be composed in a training loop.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "smooth_l1",
    "pairwise_rank_loss",
    "zscore_mse",
    "anchor_consistency_loss",
    "anchor_relative_loss",
]


def smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Wrapper around :func:`torch.nn.functional.smooth_l1_loss`.

    Parameters
    ----------
    pred, target: torch.Tensor
        Predicted and ground truth MOS scores.
    """

    return F.smooth_l1_loss(pred, target)


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple pairwise ranking loss.

    A hinge loss is applied to all pairs ``(i, j)`` where ``target[i] >
    target[j]`` such that ``pred[i]`` should be greater than ``pred[j]``.
    The implementation follows common practice in IQA literature.
    """

    N = pred.size(0)
    if N < 2:
        return pred.new_tensor(0.0)
    loss = pred.new_tensor(0.0)
    count = 0
    for i in range(N):
        for j in range(N):
            if target[i] > target[j]:
                loss = loss + F.relu(1.0 - (pred[i] - pred[j]))
                count += 1
    if count > 0:
        loss = loss / count
    return loss


def zscore_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error computed on z-score normalised values."""

    def _z(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / (x.std() + 1e-8)

    return F.mse_loss(_z(pred), _z(target))


def anchor_consistency_loss(
    pred: torch.Tensor, target: torch.Tensor, max_val: float, min_val: float
) -> torch.Tensor:
    """Anchor consistency loss.

    Parameters
    ----------
    pred, target: torch.Tensor
        Batch predictions and ground truth scores.
    max_val, min_val: float
        Ground truth scores corresponding to the highest and lowest anchor
        images respectively.
    """

    loss_max = F.mse_loss(pred.max(), pred.new_tensor(max_val))
    loss_min = F.mse_loss(pred.min(), pred.new_tensor(min_val))
    return loss_max + loss_min


def anchor_relative_loss(
    pred: torch.Tensor, target: torch.Tensor, max_val: float, min_val: float
) -> torch.Tensor:
    """Encourages relative distances to anchors to match the ground truth."""

    anchor_max = pred.new_tensor(max_val)
    anchor_min = pred.new_tensor(min_val)
    diff_pred = torch.stack([anchor_max - pred, anchor_min - pred], dim=0)
    diff_gt = torch.stack([anchor_max - target, anchor_min - target], dim=0)
    return F.mse_loss(diff_pred, diff_gt)
