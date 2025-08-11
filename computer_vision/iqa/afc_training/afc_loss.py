# -*- coding: utf-8 -*-
"""Implementation of 2AFC loss for image quality assessment.

The loss supports two different modes:

``pair``
    Pair-wise 2AFC loss.  The network predicts individual quality scores
    for two images and the loss encourages the difference of the scores to
    agree with the ground-truth preference.
``list``
    List-wise 2AFC loss.  Given a list of predictions and ground-truth
    MOS values for the same scene, the loss sums pair-wise comparisons
    over the entire list.

During training the MOS head usually outputs unbounded values.  To obtain
absolute scores we provide :func:`calibrate_mos` which maps predictions to
the desired MOS range before computing evaluation metrics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def calibrate_mos(
    scores: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 100.0,
) -> torch.Tensor:
    """Calibrate raw MOS predictions to an absolute range.

    The MOS head typically produces values on the real line.  For
    evaluation the predictions are mapped to ``[min_val, max_val]`` using
    a sigmoid function.

    Parameters
    ----------
    scores: Tensor
        Raw predictions from the MOS head.
    min_val: float, optional
        Lower bound of the target MOS range.
    max_val: float, optional
        Upper bound of the target MOS range.

    Returns
    -------
    Tensor
        Calibrated scores residing in the specified range.
    """

    return torch.sigmoid(scores) * (max_val - min_val) + min_val


class AFCLoss(nn.Module):
    """Two-Alternative Forced Choice loss.

    Parameters
    ----------
    mode: str, optional
        ``"pair"`` for pair-wise loss or ``"list"`` for list-wise loss.
    reduction: str, optional
        Reduction method to apply, ``"mean"`` or ``"sum"``.
    """

    def __init__(self, mode: str = "pair", reduction: str = "mean") -> None:
        super().__init__()
        if mode not in {"pair", "list"}:
            raise ValueError("mode must be 'pair' or 'list'")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.mode = mode
        self.reduction = reduction

    def forward(
        self,
        pred_a: torch.Tensor,
        pred_b: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the 2AFC loss.

        Usage depends on ``mode``:

        ``mode="pair"``
            ``pred_a`` and ``pred_b`` are the predicted scores of two images
            in a pair while ``target`` is ``1`` if image ``a`` is better than
            image ``b`` and ``0`` otherwise.

        ``mode="list"``
            ``pred_a`` is a tensor of shape ``(B, N)`` containing the
            predictions for ``N`` images from the same scene and ``pred_b``
            holds the corresponding ground-truth MOS values.  ``target`` is
            ignored.  The loss internally forms pair-wise comparisons over
            the ``N`` images.
        """

        if self.mode == "pair":
            if pred_b is None or target is None:
                raise ValueError("pred_b and target must be provided for pair-wise mode")
            diff = pred_a - pred_b
            loss = F.binary_cross_entropy_with_logits(diff, target.float(), reduction=self.reduction)
            return loss

        # list-wise mode
        if pred_b is None:
            raise ValueError("Ground truth MOS must be provided via pred_b for list-wise mode")

        scores = pred_a
        mos = pred_b
        batch, n = scores.shape
        loss_accum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = scores[:, i] - scores[:, j]
                target_ij = (mos[:, i] > mos[:, j]).float()
                pair_loss = F.binary_cross_entropy_with_logits(diff, target_ij, reduction="sum")
                loss_accum += pair_loss
                count += diff.numel()
        if self.reduction == "mean" and count > 0:
            loss_accum = loss_accum / count
        return loss_accum


__all__ = ["AFCLoss", "calibrate_mos"]
