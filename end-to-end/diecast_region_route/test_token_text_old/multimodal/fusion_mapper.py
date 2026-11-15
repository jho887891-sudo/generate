from __future__ import annotations

import torch
from torch import nn


class PrefixFusionMapper(nn.Module):
    """Fuse visual, coordinate-region and process-parameter tokens before LLM input."""

    def __init__(self, embed_dim: int, token_count: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.token_count = token_count
        self.query_tokens = nn.Parameter(torch.randn(token_count, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if prefix_tokens.numel() == 0:
            return prefix_tokens.new_zeros((0, self.query_tokens.size(-1)))

        src = prefix_tokens.float().unsqueeze(0)
        query = self.query_tokens.unsqueeze(0).to(device=prefix_tokens.device, dtype=torch.float32)
        fused, _ = self.attn(query=query, key=src, value=src, need_weights=False)
        fused = self.norm1(fused + query)
        fused = self.norm2(fused + self.ffn(fused))
        return fused.squeeze(0)
