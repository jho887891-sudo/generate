#!/usr/bin/env python3
"""Check Python environment, key packages, CUDA, and local pretrained weights.
This script never installs packages and never downloads weights.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from common import ensure_dir, load_config, setup_logger, write_json


def package_version(name: str):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return {"missing": True, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["logs_dir"])
    logger = setup_logger("check_env", Path(cfg["logs_dir"]) / "00_check_env.log")

    report = {
        "python_executable": sys.executable,
        "configured_python_bin": cfg.get("python_bin"),
        "packages": {},
        "cuda": {},
        "pretrained_weights": {},
        "notes": [],
    }

    logger.info("Checking Python executable: %s", sys.executable)
    if str(sys.executable) != str(cfg.get("python_bin")):
        logger.warning("Current Python is not the configured python_bin. Prefer running with: %s", cfg.get("python_bin"))

    for pkg in ["torch", "torchvision", "ultralytics", "cv2", "PIL", "numpy", "yaml"]:
        v = package_version(pkg)
        report["packages"][pkg] = v
        logger.info("Package %-12s: %s", pkg, v)

    try:
        import torch
        report["cuda"] = {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count(),
            "device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["cuda"] = {"error": str(exc)}
    logger.info("CUDA report: %s", json.dumps(report["cuda"], ensure_ascii=False))

    weight_paths = []
    for key, model_cfg in (cfg.get("models") or {}).items():
        for field in ["weights", "pretrained_checkpoint", "pretrained_resnet50", "pretrained_encoder_safetensors", "pretrained_encoder_config"]:
            if field in model_cfg:
                weight_paths.append((f"{key}.{field}", Path(model_cfg[field])))

    for name, path in weight_paths:
        exists = path.exists()
        size = path.stat().st_size if exists else None
        report["pretrained_weights"][name] = {"path": str(path), "exists": exists, "size": size}
        if exists:
            logger.info("Found weight %-45s %s (%s bytes)", name, path, size)
        else:
            logger.warning("Missing weight %-45s %s", name, path)
            if "yolov11n_seg.weights" in name:
                logger.warning("YOLOv11n-seg weight is missing. Download manually on the server if needed, for example: yolo predict model=yolo11n-seg.pt source=/path/to/test.jpg, then move it to %s", path)
            if "yolov8n_seg.weights" in name:
                logger.warning("YOLOv8n-seg weight is missing. Download manually on the server if needed, then move it to %s", path)

    out = Path(cfg["logs_dir"]) / "00_check_env_report.json"
    write_json(out, report)
    logger.info("Wrote environment report: %s", out)
    logger.info("No packages were installed and no weights were downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
