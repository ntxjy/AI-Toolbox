import torch
from torch import nn

from .backbone import ViewBackbone
from .multi_view_cross_attention import MultiViewCrossAttention


class IQAModel(nn.Module):
    """Integrates the backbone and cross-attention for multi-view IQA."""

    def __init__(
        self,
        backbone: str = "fast_itpn",
        embed_dim: int = 64,
        num_heads: int = 4,
        num_views: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = ViewBackbone(
            name=backbone, embed_dim=embed_dim, num_views=num_views
        )
        self.cross_attn = MultiViewCrossAttention(dim=embed_dim, num_heads=num_heads)
        self.delta_head = nn.Linear(embed_dim, 1)
        self.mos_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        imgs: torch.Tensor,
        view_indices: torch.Tensor,
        is_ref: torch.Tensor,
    ):
        """Forward pass.

        Args:
            imgs: Tensor of shape ``(B, V, C, H, W)``.
            view_indices: Tensor of shape ``(B, V)`` with viewpoint ids.
            is_ref: Bool tensor of shape ``(B, V)`` marking reference views.

        Returns:
            A tuple ``(pred_mos, base_mos)`` where ``pred_mos`` is the final
            MOS prediction ``(ΔMOS + MOS)`` and ``base_mos`` is the base MOS
            estimate used during training.
        """

        feats = self.backbone(imgs, view_indices)  # (B, V, 1, C)
        feats = self.cross_attn(feats, is_ref).squeeze(2)  # (B, V, C)
        delta = self.delta_head(feats).squeeze(-1)
        mos = self.mos_head(feats).squeeze(-1)
        pred = mos + delta
        return pred, mos
