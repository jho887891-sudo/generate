#!/usr/bin/env python3
"""Common utilities for the vision_compare training package."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency PyYAML. Install minimally with: pip install pyyaml --break-system-packages") from exc

IMAGE_EXTENSIONS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, log_file: Optional[str | os.PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def read_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_jsonl(path: str | os.PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def list_images(raw_dir: str | os.PathLike, valid_exts: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    root = Path(raw_dir)
    exts = {e.lower() for e in (valid_exts or IMAGE_EXTENSIONS_DEFAULT)}
    images: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in images:
                raise ValueError(f"Duplicate image stem found: {p.stem}; files: {images[p.stem]} and {p}")
            images[p.stem] = p
    return images


def list_jsons(raw_dir: str | os.PathLike) -> Dict[str, Path]:
    root = Path(raw_dir)
    jsons: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".json":
            if p.stem in jsons:
                raise ValueError(f"Duplicate JSON stem found: {p.stem}; files: {jsons[p.stem]} and {p}")
            jsons[p.stem] = p
    return jsons


def open_image_size(path: str | os.PathLike) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        return img.size  # width, height


def normalize_label(label: str, cfg: Dict[str, Any]) -> str:
    mapping = cfg.get("label_mapping", {}) or {}
    if label in mapping:
        return mapping[label]
    classes = cfg.get("classes", ["background", "defect"])
    if label in classes:
        return label
    default_fg = mapping.get("default_foreground", "defect")
    return default_fg


def class_to_id(label: str, cfg: Dict[str, Any]) -> int:
    classes = cfg.get("classes", ["background", "defect"])
    if label not in classes:
        raise ValueError(f"Label '{label}' is not in classes {classes}. Add it to configs/vision_compare.yaml or label_mapping.")
    return classes.index(label)


def parse_labelme(json_obj: Dict[str, Any], json_path: str | os.PathLike, width: int, height: int, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(json_obj, dict) or "shapes" not in json_obj:
        raise ValueError(f"{json_path} is not valid LabelMe format: missing top-level 'shapes'.")
    shapes = json_obj.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{json_path} is not valid LabelMe format: 'shapes' must be a list.")
    instances: List[Dict[str, Any]] = []
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"{json_path} shape[{idx}] is not an object; expected LabelMe shape dict.")
        shape_type = shape.get("shape_type", "polygon")
        if shape_type not in (None, "polygon"):
            continue
        points = shape.get("points")
        if not points or not isinstance(points, list):
            continue
        if len(points) < 3:
            continue
        clean_points: List[List[float]] = []
        for pt_idx, pt in enumerate(points):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise ValueError(f"{json_path} shape[{idx}] point[{pt_idx}] must be [x, y].")
            x, y = float(pt[0]), float(pt[1])
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError(
                    f"{json_path} shape[{idx}] point[{pt_idx}] out of image bounds: "
                    f"({x}, {y}) not within width={width}, height={height}."
                )
            clean_points.append([x, y])
        raw_label = str(shape.get("label", "defect"))
        label = normalize_label(raw_label, cfg)
        class_id = class_to_id(label, cfg)
        if class_id == 0:
            continue
        instances.append({"label": label, "class_id": class_id, "points": clean_points})
    return instances


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def make_semantic_mask(width: int, height: int, instances: Sequence[Dict[str, Any]]) -> np.ndarray:
    from PIL import Image, ImageDraw
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for inst in instances:
        pts = [(float(x), float(y)) for x, y in inst["points"]]
        draw.polygon(pts, fill=int(inst["class_id"]))
    return np.array(mask, dtype=np.uint8)


def make_instance_masks(width: int, height: int, instances: Sequence[Dict[str, Any]]) -> np.ndarray:
    from PIL import Image, ImageDraw
    masks = []
    for inst in instances:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        pts = [(float(x), float(y)) for x, y in inst["points"]]
        draw.polygon(pts, fill=1)
        masks.append(np.array(mask, dtype=np.uint8))
    if not masks:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.stack(masks, axis=0)


def yolo_seg_line(inst: Dict[str, Any], width: int, height: int) -> str:
    # YOLO class ids do not include background, so foreground class_id 1 becomes YOLO id 0.
    yolo_class = int(inst["class_id"]) - 1
    coords: List[str] = []
    for x, y in inst["points"]:
        coords.append(f"{max(0.0, min(1.0, float(x) / width)):.8f}")
        coords.append(f"{max(0.0, min(1.0, float(y) / height)):.8f}")
    return " ".join([str(yolo_class)] + coords)


def copy_or_link(src: str | os.PathLike, dst: str | os.PathLike, prefer_symlink: bool = True) -> None:
    import shutil
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if dst_p.exists() or dst_p.is_symlink():
        dst_p.unlink()
    if prefer_symlink:
        try:
            os.symlink(src_p, dst_p)
            return
        except OSError:
            pass
    shutil.copy2(src_p, dst_p)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def torch_load(path: str | os.PathLike, map_location: str = "cpu"):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_run_dir(cfg: Dict[str, Any], model_key: str, allow_existing_for_resume: bool = False) -> Path:
    base = Path(cfg["runs_dir"]) / cfg["models"][model_key]["run_subdir"]
    if allow_existing_for_resume:
        base.mkdir(parents=True, exist_ok=True)
        return base
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    # Avoid overwriting previous complete/partial outputs. Keep expected canonical path if empty.
    if not any(base.iterdir()):
        return base
    raise FileExistsError(
        f"Run directory already exists and is not empty: {base}. "
        "Refusing to create a timestamped alternative because expected_weights points to the canonical path. "
        "Use --resume to continue, or move/rename the existing directory before starting a new run."
    )


def get_prepared_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    work = Path(cfg["work_dir"])
    return {
        "work": work,
        "manifest": work / "clean_manifest.jsonl",
        "report": work / "clean_report.json",
        "splits": work / "splits.json",
        "semantic_root": work / "semantic",
        "mask_rcnn_root": work / "mask_rcnn",
        "yolo_root": work / "yolo_seg",
        "yolo_data_yaml": work / "yolo_seg" / "data.yaml",
    }
