"""FastITPN backbone placeholder.

This module defines a minimal backbone exposing a unified ``forward``
interface. The implementation is intentionally lightweight as a placeholder
for the real network.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FastITPN(nn.Module):
    """A minimal FastITPN backbone.

    Args:
        pretrained: Optional path to a state dict for initialization.
        freeze: If ``True``, parameters are frozen after loading weights.
    """

    def __init__(self, pretrained: Optional[str] = None, freeze: bool = False):
        super().__init__()
        # Placeholder feature processor
        self.net = nn.Identity()

        if pretrained:
            state = torch.load(pretrained, map_location="cpu")
            self.load_state_dict(state, strict=False)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Process input features.

        Args:
            features: Input tensor of features.
        Returns:
            Tensor processed by the backbone.
        """
        return self.net(features)
