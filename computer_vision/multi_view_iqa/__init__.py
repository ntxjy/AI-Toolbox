"""Multi-view stereoscopic MOS prediction package.

This subpackage implements the core components described in the
multi-view MOS prediction system:

- :class:`DFEModule` for distortion feature extraction
- :class:`MIAModule` for multi-view interaction aggregation
- :class:`MultiViewIQAModel` combining the two modules

The code is designed to be self contained so that users can easily
plug it into their own training or evaluation pipelines.
"""

from .model import DFEModule, MIAModule, MultiViewIQAModel
from .dataset import MVSDataset
from .losses import (
    smooth_l1,
    pairwise_rank_loss,
    zscore_mse,
    anchor_consistency_loss,
    anchor_relative_loss,
)

__all__ = [
    "DFEModule",
    "MIAModule",
    "MultiViewIQAModel",
    "MVSDataset",
    "smooth_l1",
    "pairwise_rank_loss",
    "zscore_mse",
    "anchor_consistency_loss",
    "anchor_relative_loss",
]
