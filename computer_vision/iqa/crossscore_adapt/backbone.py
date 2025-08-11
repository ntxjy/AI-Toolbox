import torch
from torch import nn


class FastITPN(nn.Module):
    """A lightweight placeholder for the Fast-iTPN backbone."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return feat.flatten(1)


class VMamba(nn.Module):
    """A lightweight placeholder for the VMamba backbone."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return feat.flatten(1)


class ViewBackbone(nn.Module):
    """Wrapper that adds viewpoint positional encoding and selects the backbone."""

    def __init__(
        self,
        name: str = "fast_itpn",
        in_ch: int = 3,
        embed_dim: int = 64,
        num_views: int = 4,
    ) -> None:
        super().__init__()
        name = name.lower()
        if name == "fast_itpn":
            self.backbone = FastITPN(in_ch, embed_dim)
        elif name == "vmamba":
            self.backbone = VMamba(in_ch, embed_dim)
        else:
            raise ValueError(f"Unknown backbone: {name}")

        self.pos_embed = nn.Embedding(num_views, embed_dim)

    def forward(self, x: torch.Tensor, view_indices: torch.Tensor) -> torch.Tensor:
        """Extract per-view features with positional encoding.

        Args:
            x: Tensor of shape ``(B, V, C, H, W)`` containing input images.
            view_indices: Long tensor of shape ``(B, V)`` specifying the
                viewpoint index of each image.

        Returns:
            Tensor of shape ``(B, V, 1, C)`` ready for cross-attention.
        """

        B, V = x.shape[:2]
        feats = []
        for v in range(V):
            feat = self.backbone(x[:, v])  # (B, C)
            feat = feat + self.pos_embed(view_indices[:, v])
            feats.append(feat.unsqueeze(1))
        feats = torch.cat(feats, dim=1)  # (B, V, C)
        return feats.unsqueeze(2)
