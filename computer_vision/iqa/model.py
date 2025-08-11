"""Simple IQA model wrapper with configurable backbone."""
from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn

from .backbones import build_backbone


class IQAModel(nn.Module):
    """Image quality assessment model with pluggable backbone.

    Args:
        backbone_cfg: Configuration passed to :func:`build_backbone`.
    """

    def __init__(self, backbone_cfg: Dict[str, Any]):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass delegating to the selected backbone."""
        return self.backbone(features)
