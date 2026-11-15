from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import nn


class RegionPriorEncoder(nn.Module):
    """把区域划分结果作为先验 token 注入，而不是要求网络重新学习区域边界。"""

    def __init__(
        self,
        region_order: Sequence[str],
        embed_dim: int,
        hidden_dim: int = 1024,
        view_vocab_size: int = 16,
        use_external_summary_tokens: bool = True,
        use_coordinate_region_summary: bool = True,
    ):
        super().__init__()
        self.region_order = list(region_order)
        self.use_external_summary_tokens = use_external_summary_tokens
        self.use_coordinate_region_summary = use_coordinate_region_summary
        self.region_to_idx = {name: i for i, name in enumerate(self.region_order)}
        self.unk_idx = len(self.region_order)
        self.region_embed = nn.Embedding(len(self.region_order) + 1, embed_dim)
        self.view_embed = nn.Embedding(view_vocab_size, embed_dim)
        self.defect_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.region_summary_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.coord_region_classifier = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.region_order)),
        )
        self.coord_summary_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _region_idx(self, region_name: str) -> int:
        if not region_name:
            return self.unk_idx
        if region_name in self.region_to_idx:
            return self.region_to_idx[region_name]
        if str(region_name).startswith("未知区域"):
            region_name = "未知区域"
        return self.region_to_idx.get(region_name, self.unk_idx)

    def encode_region_summary(self, summary: Dict[str, Any], device: str) -> torch.Tensor:
        region_count = summary.get("region_count", {}) or {}
        region_max_area = summary.get("region_max_area_mm2", {}) or {}
        region_max_diam = summary.get("region_max_eq_diameter_mm", {}) or {}
        tokens = []
        for region in self.region_order:
            vals = torch.tensor([
                float(region_count.get(region, 0.0) or 0.0),
                float(region_max_area.get(region, 0.0) or 0.0),
                float(region_max_diam.get(region, 0.0) or 0.0),
            ], dtype=torch.float32, device=device)
            base = self.region_embed(torch.tensor(self._region_idx(region), device=device))
            tokens.append(base + self.region_summary_proj(vals))
        return torch.stack(tokens, dim=0)

    def encode_defect_priors(self, defect_meta_by_view: Sequence[Sequence[Dict[str, Any]]], image_size: int, max_defect_tokens: int, device: str) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for view_idx, metas in enumerate(defect_meta_by_view, start=1):
            for meta in metas[:max_defect_tokens]:
                region_name = meta.get("region_type", "未知区域")
                base = self.region_embed(torch.tensor(self._region_idx(region_name), device=device))
                view = self.view_embed(torch.tensor(min(view_idx, self.view_embed.num_embeddings - 1), device=device))
                x = float(meta.get("x", 0.0) or 0.0) / max(1.0, float(image_size))
                y = float(meta.get("y", 0.0) or 0.0) / max(1.0, float(image_size))
                area = float(meta.get("area_mm2", 0.0) or 0.0)
                diam = float(meta.get("diameter_mm", 0.0) or 0.0)
                is_pore = 1.0 if bool(meta.get("is_pore", False)) else 0.0
                scalars = torch.tensor([x, y, area, diam, is_pore], dtype=torch.float32, device=device)
                tokens.append(base + view + self.defect_proj(scalars))
        if not tokens:
            return torch.zeros((0, self.region_embed.embedding_dim), dtype=torch.float32, device=device)
        return torch.stack(tokens, dim=0)

    def encode_coordinate_region_summary(
        self,
        defect_meta_by_view: Sequence[Sequence[Dict[str, Any]]],
        image_size: int,
        max_defect_tokens: int,
        device: str,
    ) -> torch.Tensor:
        if not self.use_coordinate_region_summary:
            return torch.zeros((0, self.region_embed.embedding_dim), dtype=torch.float32, device=device)

        scalars = []
        for metas in defect_meta_by_view:
            for meta in metas[:max_defect_tokens]:
                x = float(meta.get("x", 0.0) or 0.0) / max(1.0, float(image_size))
                y = float(meta.get("y", 0.0) or 0.0) / max(1.0, float(image_size))
                area = float(meta.get("area_mm2", 0.0) or 0.0)
                diam = float(meta.get("diameter_mm", 0.0) or 0.0)
                is_pore = 1.0 if bool(meta.get("is_pore", False)) else 0.0
                scalars.append([x, y, area, diam, is_pore])

        if not scalars:
            return torch.zeros((0, self.region_embed.embedding_dim), dtype=torch.float32, device=device)

        x = torch.tensor(scalars, dtype=torch.float32, device=device)
        probs = torch.softmax(self.coord_region_classifier(x), dim=-1)
        count = probs.sum(dim=0)
        area_sum = probs.transpose(0, 1).matmul(x[:, 2])
        diam_sum = probs.transpose(0, 1).matmul(x[:, 3])
        pore_sum = probs.transpose(0, 1).matmul(x[:, 4])
        stats = torch.stack([count, area_sum, diam_sum, pore_sum], dim=-1)

        tokens = []
        for idx, region in enumerate(self.region_order):
            base = self.region_embed(torch.tensor(self._region_idx(region), device=device))
            tokens.append(base + self.coord_summary_proj(stats[idx]))
        return torch.stack(tokens, dim=0)

    def forward(self, summary: Dict[str, Any], defect_meta_by_view: Sequence[Sequence[Dict[str, Any]]], image_size: int, max_defect_tokens: int, device: str) -> torch.Tensor:
        summary_tokens = (
            self.encode_region_summary(summary, device)
            if self.use_external_summary_tokens
            else torch.zeros((0, self.region_embed.embedding_dim), dtype=torch.float32, device=device)
        )
        coord_summary_tokens = self.encode_coordinate_region_summary(
            defect_meta_by_view,
            image_size=image_size,
            max_defect_tokens=max_defect_tokens,
            device=device,
        )
        defect_tokens = self.encode_defect_priors(defect_meta_by_view, image_size=image_size, max_defect_tokens=max_defect_tokens, device=device)
        parts = []
        if summary_tokens.numel() > 0:
            parts.append(summary_tokens)
        if coord_summary_tokens.numel() > 0:
            parts.append(coord_summary_tokens)
        if defect_tokens.numel() == 0:
            if parts:
                return torch.cat(parts, dim=0)
            return torch.zeros((0, self.region_embed.embedding_dim), dtype=torch.float32, device=device)
        parts.append(defect_tokens)
        return torch.cat(parts, dim=0)
