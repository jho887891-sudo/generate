#!/usr/bin/env python3
"""Train YOLOv8n-seg or YOLOv11n-seg with local weights only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "tools" / "vision_compare"))
from common import get_prepared_paths, load_config, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    parser.add_argument("--model-name", required=True, choices=["yolov8n_seg", "yolov11n_seg"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["models"][args.model_name]
    do_resume = args.resume or bool(cfg["training"].get("resume", False))
    logger = setup_logger(f"train_{args.model_name}", Path(cfg["logs_dir"]) / f"train_{args.model_name}.log")
    paths = get_prepared_paths(cfg)

    weights = Path(model_cfg["weights"])
    if not weights.exists():
        raise FileNotFoundError(
            f"Local YOLO weights not found: {weights}. This script will not auto-download. "
            "Please download manually on the server and place the file at the configured path."
        )
    data_yaml = paths["yolo_data_yaml"]
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}. Run prepare_data.sh first.")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is not importable in the configured environment. Do not upgrade automatically; inspect environment first with 00_check_env.py.") from exc

    project = Path(cfg["runs_dir"])
    name = model_cfg["run_subdir"]
    target_dir = project / name
    if target_dir.exists() and any(target_dir.iterdir()) and not do_resume:
        raise FileExistsError(
            f"YOLO output directory already exists and is not empty: {target_dir}. "
            "Use --resume to continue, or move/rename the directory to avoid overwriting."
        )

    model = YOLO(str(weights))
    logger.info("Starting %s training with weights=%s data=%s", args.model_name, weights, data_yaml)
    model.train(
        data=str(data_yaml),
        epochs=int(cfg["training"]["epochs"]),
        imgsz=int(cfg["image"]["img_size"]),
        batch=int(model_cfg["batch_size"]),
        workers=int(cfg["training"]["num_workers"]),
        device=str(cfg["training"].get("device", "cuda:0")).replace("cuda:", ""),
        project=str(project),
        name=name,
        exist_ok=bool(do_resume),
        resume=bool(do_resume),
        pretrained=False,
        verbose=True,
    )
    logger.info("Done. Expected best weight: %s", target_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
