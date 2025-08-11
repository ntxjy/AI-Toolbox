"""Core model components for multi-view MOS prediction.

This module implements the distortion feature encoder (DFE) and the
multi-view interaction aggregation (MIA) transformer.  The design is
inspired by the LoDa IQA model [Xu et al., CVPR 2024] and stereo image
quality assessment research.

The code is intentionally lightweight.  Only the high level logic is
implemented; users can swap the backbone or extend the modules for
research purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional import
    from torchvision import models
except Exception:  # pragma: no cover - torchvision is optional
    models = None


class DFEModule(nn.Module):
    """Distortion Feature Encoding module.

    Parameters
    ----------
    backbone: nn.Module
        Convolutional backbone used to extract features.  If ``None`` a
        torchvision ResNet-18 backbone is created.  Only the feature
        extraction part is used and its parameters are frozen.
    feat_dim: int, default=512
        Dimension of the output feature vector for each image.
    use_depth: bool, default=False
        Whether an additional depth map is provided for each image.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        feat_dim: int = 512,
        use_depth: bool = False,
    ) -> None:
        super().__init__()
        if backbone is None:
            if models is None:  # fallback minimal CNN if torchvision unavailable
                backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                backbone.out_channels = 64
            else:
                base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                backbone = nn.Sequential(*list(base.children())[:-2])
                for p in backbone.parameters():
                    p.requires_grad = False
                backbone.out_channels = base.fc.in_features
        self.backbone = backbone
        self.use_depth = use_depth

        if use_depth:
            # simple depth encoder, parameters are trainable
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.depth_fc = nn.Linear(32, feat_dim)

        self.reduce_conv = nn.Conv2d(self.backbone.out_channels, feat_dim, 1)
        self.output_fc = nn.Linear(feat_dim, feat_dim)

    def forward(self, img: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a single image (and optional depth map) into a feature vector.

        Parameters
        ----------
        img: torch.Tensor
            Tensor of shape ``(B, 3, H, W)``.
        depth: torch.Tensor, optional
            Tensor of shape ``(B, 1, H, W)`` representing depth.

        Returns
        -------
        feat: torch.Tensor
            Tensor of shape ``(B, feat_dim)`` containing the distortion
            related representation of the input image.
        """

        conv_feat = self.backbone(img)  # (B, C, h, w)
        local_feat = self.reduce_conv(conv_feat)
        local_feat = F.adaptive_avg_pool2d(local_feat, (1, 1)).flatten(1)

        if self.use_depth and depth is not None:
            d_feat = self.depth_encoder(depth).flatten(1)
            d_feat = self.depth_fc(d_feat)
            local_feat = local_feat + d_feat

        feat = self.output_fc(local_feat)
        return feat


class MIAModule(nn.Module):
    """Multi-view Interaction Aggregation module.

    A stack of Transformer encoder layers is used to model relations
    among features from different viewpoints.
    """

    def __init__(
        self,
        feat_dim: int,
        num_views: int = 6,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads, dim_feedforward=feat_dim * 2
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.view_pos_embedding = nn.Parameter(torch.randn(num_views, feat_dim))

    def forward(self, feat_seq: torch.Tensor) -> torch.Tensor:
        """Fuse a sequence of view features.

        Parameters
        ----------
        feat_seq: torch.Tensor
            Sequence tensor of shape ``(V, B, D)`` where ``V`` is the
            number of views, ``B`` the batch size and ``D`` the feature
            dimension.

        Returns
        -------
        torch.Tensor
            Fused feature of shape ``(B, D)``.
        """

        seq = feat_seq + self.view_pos_embedding.unsqueeze(1)
        fused = self.transformer(seq)
        fused_feat = fused.mean(dim=0)
        return fused_feat


class MultiViewIQAModel(nn.Module):
    """Complete multi-view MOS prediction network."""

    def __init__(
        self,
        dfe: DFEModule,
        feat_dim: int = 512,
        use_distortion_cls: bool = False,
        num_distortion_types: int = 0,
    ) -> None:
        super().__init__()
        self.dfe = dfe
        self.mia = MIAModule(feat_dim=feat_dim, num_views=6)
        self.score_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
        )
        self.use_distortion_cls = use_distortion_cls
        if use_distortion_cls and num_distortion_types > 0:
            self.distortion_cls_head = nn.Linear(feat_dim, num_distortion_types)
        else:
            self.distortion_cls_head = None

    def forward(
        self,
        images: Iterable[torch.Tensor],
        depths: Optional[Iterable[torch.Tensor]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats: List[torch.Tensor] = []
        depths = list(depths) if depths is not None else [None] * len(images)
        for img, dep in zip(images, depths):
            feats.append(self.dfe(img, dep).unsqueeze(0))
        feat_seq = torch.cat(feats, dim=0)
        fused = self.mia(feat_seq)
        score = self.score_head(fused)
        if self.distortion_cls_head is not None:
            logits = self.distortion_cls_head(fused)
            return score, logits
        return score


__all__ = ["DFEModule", "MIAModule", "MultiViewIQAModel"]
