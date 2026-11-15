from __future__ import annotations

import torch
from torch import nn


class NumericParamEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, token_count: int = 2):
        super().__init__()
        self.token_count = token_count
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * token_count),
            nn.GELU(),
            nn.Linear(embed_dim * token_count, embed_dim * token_count),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(x.size(0), self.token_count, self.embed_dim)
