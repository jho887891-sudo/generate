from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal.config_region_mm import RegionAwareMMConfig
from multimodal.dataset_region_mm import RegionMMJsonlDataset
from multimodal.graph_guard import get_graph_evidence
from multimodal.model_region_mm import RegionAwareMultiModalModel


def main() -> None:
    parser = argparse.ArgumentParser(description="区域感知多模态模型推理 demo。")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--stage1-adapter", type=str, default=None)
    args = parser.parse_args()

    cfg = RegionAwareMMConfig(train_jsonl=Path(args.jsonl))
    if args.base_model:
        cfg.llm_base_path = args.base_model
    if args.stage1_adapter:
        cfg.llm_stage1_adapter_path = args.stage1_adapter
    ds = RegionMMJsonlDataset(args.jsonl, cfg)
    sample = ds[args.index]
    model = RegionAwareMultiModalModel(cfg)
    model.eval()

    graph_evidence = []
    if args.use_graph:
        graph_evidence = get_graph_evidence(sample.get("info_text", ""), sample["question"]) 

    text = model.generate_one(
        workpiece_id=sample["workpiece_id"],
        instruction=sample["instruction"],
        images=sample["images"],
        boxes_by_view=sample["boxes_by_view"],
        defect_meta_by_view=sample["defect_meta_by_view"],
        summary=sample["summary"],
        params=sample["params"],
        question=sample["question"],
        graph_evidence=graph_evidence,
    )
    print(text)


if __name__ == "__main__":
    main()
