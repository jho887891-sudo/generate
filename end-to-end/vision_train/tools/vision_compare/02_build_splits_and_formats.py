#!/usr/bin/env python3
"""Build one fixed train/val/test split and export formats for all five models."""
from __future__ import annotations

import argparse
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from PIL import Image

from common import (
    copy_or_link,
    ensure_dir,
    get_prepared_paths,
    load_config,
    make_instance_masks,
    make_semantic_mask,
    read_jsonl,
    setup_logger,
    write_json,
    yolo_seg_line,
)


def split_rows(rows: List[Dict], seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, List[Dict]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }


def save_semantic(split_name: str, rows: List[Dict], root: Path) -> None:
    img_dir = root / "images" / split_name
    mask_dir = root / "masks" / split_name
    ensure_dir(img_dir)
    ensure_dir(mask_dir)
    for row in rows:
        src = Path(row["image_path"])
        dst_img = img_dir / src.name
        copy_or_link(src, dst_img)
        mask = make_semantic_mask(row["width"], row["height"], row["instances"])
        Image.fromarray(mask).save(mask_dir / f"{row['stem']}.png")


def save_mask_rcnn(split_name: str, rows: List[Dict], root: Path) -> None:
    img_dir = root / "images" / split_name
    ann_dir = root / "annotations" / split_name
    mask_dir = root / "instance_masks" / split_name
    ensure_dir(img_dir)
    ensure_dir(ann_dir)
    ensure_dir(mask_dir)
    for row in rows:
        src = Path(row["image_path"])
        dst_img = img_dir / src.name
        copy_or_link(src, dst_img)
        masks = make_instance_masks(row["width"], row["height"], row["instances"])
        inst_records = []
        for idx, inst in enumerate(row["instances"]):
            mask_path = mask_dir / f"{row['stem']}_{idx:03d}.png"
            Image.fromarray((masks[idx] * 255).astype("uint8")).save(mask_path)
            ys, xs = masks[idx].nonzero()
            if len(xs) == 0 or len(ys) == 0:
                continue
            inst_records.append({
                "label": inst["label"],
                "class_id": inst["class_id"],
                "polygon": inst["points"],
                "mask_path": str(mask_path),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area": float(masks[idx].sum()),
            })
        write_json(ann_dir / f"{row['stem']}.json", {
            "stem": row["stem"],
            "image_path": str(dst_img),
            "width": row["width"],
            "height": row["height"],
            "instances": inst_records,
        })


def save_yolo(split_name: str, rows: List[Dict], root: Path) -> None:
    img_dir = root / "images" / split_name
    label_dir = root / "labels" / split_name
    ensure_dir(img_dir)
    ensure_dir(label_dir)
    for row in rows:
        src = Path(row["image_path"])
        dst_img = img_dir / src.name
        copy_or_link(src, dst_img)
        lines = [yolo_seg_line(inst, row["width"], row["height"]) for inst in row["instances"]]
        (label_dir / f"{row['stem']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_one_row(args):
    split_name, row, semantic_root, mask_rcnn_root, yolo_root = args
    save_semantic(split_name, [row], Path(semantic_root))
    save_mask_rcnn(split_name, [row], Path(mask_rcnn_root))
    save_yolo(split_name, [row], Path(yolo_root))
    return row["stem"], len(row.get("instances", []))


def export_split(split_name: str, rows: List[Dict], paths: Dict[str, Path], num_workers: int, progress_interval: int, logger) -> None:
    total = len(rows)
    if total == 0:
        return
    jobs = [
        (split_name, row, str(paths["semantic_root"]), str(paths["mask_rcnn_root"]), str(paths["yolo_root"]))
        for row in rows
    ]
    if num_workers <= 1:
        instance_count = 0
        for done, job in enumerate(jobs, start=1):
            _, n_inst = export_one_row(job)
            instance_count += n_inst
            if done == total or done % progress_interval == 0:
                logger.info("Exporting split=%s progress=%d/%d instances_seen=%d", split_name, done, total, instance_count)
        return

    instance_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(export_one_row, job) for job in jobs]
        for done, fut in enumerate(as_completed(futures), start=1):
            _, n_inst = fut.result()
            instance_count += n_inst
            if done == total or done % progress_interval == 0:
                logger.info("Exporting split=%s progress=%d/%d instances_seen=%d", split_name, done, total, instance_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_prepared_paths(cfg)
    ensure_dir(cfg["logs_dir"])
    logger = setup_logger("build_formats", Path(cfg["logs_dir"]) / "02_build_splits_and_formats.log")
    export_cfg = cfg.get("data_export", {}) or {}
    default_workers = min(12, max(1, (os.cpu_count() or 1) // 2))
    num_workers = int(export_cfg.get("num_workers", default_workers))
    progress_interval = max(1, int(export_cfg.get("progress_interval", 100)))

    if not paths["manifest"].exists():
        raise FileNotFoundError(f"Clean manifest not found: {paths['manifest']}. Run 01_clean_dataset.py first.")

    rows = read_jsonl(paths["manifest"])
    split_cfg = cfg["split"]
    splits = split_rows(rows, int(split_cfg["seed"]), float(split_cfg["train_ratio"]), float(split_cfg["val_ratio"]), float(split_cfg["test_ratio"]))

    # Rebuild only derived data under work_dir. This never touches raw_dataset_dir.
    work_root = paths["work"].resolve()
    for key in ["semantic_root", "mask_rcnn_root", "yolo_root"]:
        target = paths[key].resolve()
        if work_root not in target.parents:
            raise RuntimeError(f"Refusing to remove derived directory outside work_dir: {target}")
        if paths[key].exists():
            shutil.rmtree(paths[key])

    split_index = {k: [r["stem"] for r in v] for k, v in splits.items()}
    write_json(paths["splits"], {
        "seed": split_cfg["seed"],
        "ratios": {"train": split_cfg["train_ratio"], "val": split_cfg["val_ratio"], "test": split_cfg["test_ratio"]},
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": split_index,
    })

    logger.info("Using data export workers: %d", num_workers)
    for split_name, split_rows_ in splits.items():
        logger.info("Exporting split=%s count=%d", split_name, len(split_rows_))
        export_split(split_name, split_rows_, paths, num_workers, progress_interval, logger)

    names = cfg.get("classes", ["background", "defect"])[1:]
    data_yaml = (
        f"path: {paths['yolo_root']}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
    )
    for idx, name in enumerate(names):
        data_yaml += f"  {idx}: {name}\n"
    paths["yolo_data_yaml"].write_text(data_yaml, encoding="utf-8")

    logger.info("Wrote splits: %s", paths["splits"])
    logger.info("Wrote semantic data: %s", paths["semantic_root"])
    logger.info("Wrote Mask R-CNN data: %s", paths["mask_rcnn_root"])
    logger.info("Wrote YOLO data.yaml: %s", paths["yolo_data_yaml"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
