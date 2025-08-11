"""List-wise ranking network for image quality assessment.

The module defines a small neural network that learns to order images by
quality using a list-wise ranking objective.  In addition to predicting a
relative ranking score for each item, the network also outputs a MOS
(mean-opinion score) regression estimate.  The regression branch is trained
with a ``1 - PLCC`` loss encouraging high correlation with ground-truth MOS
values.
"""
from __future__ import annotations

from typing import Tuple

try:  # PyTorch is optional at import time.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - dependency missing
    torch = None
    nn = object  # type: ignore


class RankNetwork(nn.Module):
    """Simple feed-forward network with ranking and MOS heads."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        if torch is None:  # pragma: no cover - defensive
            raise ImportError("PyTorch is required to use RankNetwork")
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Ranking head outputs a score per item
        self.rank_head = nn.Linear(hidden_dim, 1)
        # MOS regression head
        self.mos_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return ranking scores and MOS estimates for the input list."""
        feat = self.backbone(x)
        rank_scores = self.rank_head(feat).squeeze(-1)
        mos = self.mos_head(feat).squeeze(-1)
        return rank_scores, mos


def listwise_ranking_loss(
    pred_scores: "torch.Tensor", target_scores: "torch.Tensor"
) -> "torch.Tensor":
    """ListNet-style cross entropy ranking loss."""
    pred_prob = torch.softmax(pred_scores, dim=-1)
    target_prob = torch.softmax(target_scores, dim=-1)
    loss = -(target_prob * torch.log(pred_prob + 1e-12)).sum(dim=-1)
    return loss.mean()


def plcc_loss(
    pred_mos: "torch.Tensor", target_mos: "torch.Tensor", eps: float = 1e-8
) -> "torch.Tensor":
    """Loss based on ``1 -`` Pearson Linear Correlation Coefficient."""
    pred_mean = pred_mos.mean()
    target_mean = target_mos.mean()
    numerator = ((pred_mos - pred_mean) * (target_mos - target_mean)).mean()
    denom = pred_mos.std() * target_mos.std() + eps
    plcc = numerator / denom
    return 1 - plcc


def combined_loss(
    pred_scores: "torch.Tensor",
    target_scores: "torch.Tensor",
    pred_mos: "torch.Tensor",
    target_mos: "torch.Tensor",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> "torch.Tensor":
    """Combined ranking and MOS regression loss."""
    return alpha * listwise_ranking_loss(pred_scores, target_scores) + beta * plcc_loss(
        pred_mos, target_mos
    )
