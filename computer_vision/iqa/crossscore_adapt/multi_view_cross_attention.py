import torch
from torch import nn


class MultiViewCrossAttention(nn.Module):
    """Cross-attention across multiple views.

    Each view attends to a set of other views in order to find a relative
    reference.  Views flagged with ``is_ref`` are treated as references while
    the others are predicted images.  For every view ``v`` the module gathers
    features from the opposite type (reference vs. non-reference) and performs
    multi-head cross-attention.

    Args:
        dim: Feature dimension of the tokens.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, feats: torch.Tensor, is_ref: torch.Tensor) -> torch.Tensor:
        """Apply cross attention.

        Args:
            feats: Tensor of shape ``(B, V, N, C)`` where ``V`` is the number of
                views and ``N`` is the number of tokens per view.
            is_ref: Bool tensor of shape ``(B, V)`` indicating whether the view
                is a reference.

        Returns:
            Tensor of the same shape as ``feats`` containing the updated
            features after cross-attention.
        """

        B, V, N, C = feats.shape
        output = torch.zeros_like(feats)

        for b in range(B):
            for v in range(V):
                query = feats[b, v].unsqueeze(0)  # (1, N, C)

                # Select context views based on ``is_ref`` flag.
                if is_ref[b, v]:
                    mask = ~is_ref[b]
                else:
                    mask = is_ref[b]

                context = feats[b, mask]
                if context.numel() == 0:
                    # No available context views; copy the query features.
                    output[b, v] = feats[b, v]
                    continue

                key = context.reshape(1, -1, C)  # (1, ctx*N, C)
                attn_out, _ = self.attn(query, key, key)
                output[b, v] = attn_out.squeeze(0)

        return output
