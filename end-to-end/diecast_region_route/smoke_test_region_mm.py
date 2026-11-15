from __future__ import annotations

import torch

from multimodal.param_encoder import NumericParamEncoder
from multimodal.projector import DefectProjector
from multimodal.region_prior import RegionPriorEncoder


def main() -> None:
    embed_dim = 128
    projector = DefectProjector(in_dim=256, hidden_dim=256, out_dim=embed_dim)
    param_encoder = NumericParamEncoder(input_dim=13, embed_dim=embed_dim, token_count=2)
    region_encoder = RegionPriorEncoder(
        region_order=["一般区域", "牙孔区域", "焊接区域", "重点区域", "厚大区域", "背景", "未知区域"],
        embed_dim=embed_dim,
        hidden_dim=256,
        view_vocab_size=16,
    )

    fake_visual = torch.randn(10, 256)
    visual_tokens = projector(fake_visual)
    print("visual_tokens:", tuple(visual_tokens.shape))

    fake_params = torch.randn(1, 13)
    numeric_tokens = param_encoder(fake_params)
    print("numeric_tokens:", tuple(numeric_tokens.shape))

    summary = {
        "region_count": {"一般区域": 12, "牙孔区域": 4},
        "region_max_area_mm2": {"一般区域": 0.24, "牙孔区域": 0.18},
        "region_max_eq_diameter_mm": {"一般区域": 0.81, "牙孔区域": 0.62},
    }
    defect_meta_by_view = [
        [
            {"region_type": "一般区域", "x": 120, "y": 300, "area_mm2": 0.11, "diameter_mm": 0.42, "is_pore": False},
            {"region_type": "牙孔区域", "x": 460, "y": 280, "area_mm2": 0.09, "diameter_mm": 0.38, "is_pore": True},
        ]
    ]
    region_tokens = region_encoder(summary, defect_meta_by_view, image_size=960, max_defect_tokens=64, device="cpu")
    print("region_tokens:", tuple(region_tokens.shape))

    prefix = torch.cat([visual_tokens, region_tokens, numeric_tokens.squeeze(0)], dim=0)
    print("prefix_tokens:", tuple(prefix.shape))
    print("smoke test passed")


if __name__ == "__main__":
    main()
