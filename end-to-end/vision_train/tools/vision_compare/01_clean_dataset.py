#!/usr/bin/env python3
"""Clean a raw folder containing images and same-stem LabelMe JSON files."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from common import (
    ensure_dir,
    list_images,
    list_jsons,
    load_config,
    open_image_size,
    parse_labelme,
    polygon_area,
    read_json,
    setup_logger,
    write_json,
    write_jsonl,
    get_prepared_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_prepared_paths(cfg)
    ensure_dir(paths["work"])
    ensure_dir(cfg["logs_dir"])
    logger = setup_logger("clean_dataset", Path(cfg["logs_dir"]) / "01_clean_dataset.log")

    raw_dir = Path(cfg["raw_dataset_dir"])
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(
            f"raw_dataset_dir does not exist or is not a directory: {raw_dir}. "
            "Edit configs/vision_compare.yaml and set raw_dataset_dir to your dataset folder."
        )

    valid_exts = cfg.get("image", {}).get("valid_extensions")
    images = list_images(raw_dir, valid_exts)
    jsons = list_jsons(raw_dir)
    image_stems = set(images)
    json_stems = set(jsons)

    missing_json = sorted(image_stems - json_stems)
    missing_image = sorted(json_stems - image_stems)
    common = sorted(image_stems & json_stems)

    logger.info("Raw images: %d, JSON files: %d, matched pairs: %d", len(images), len(jsons), len(common))
    if missing_json:
        logger.warning("Images without same-stem JSON: %d", len(missing_json))
    if missing_image:
        logger.warning("JSON files without same-stem image: %d", len(missing_image))

    clean_rows: List[Dict] = []
    rejected: List[Dict] = []
    label_counter: Counter = Counter()
    instance_counter = 0

    for stem in common:
        image_path = images[stem]
        json_path = jsons[stem]
        try:
            width, height = open_image_size(image_path)
        except Exception as exc:
            rejected.append({"stem": stem, "image": str(image_path), "json": str(json_path), "reason": f"image_open_failed: {exc}"})
            continue

        try:
            obj = read_json(json_path)
        except Exception as exc:
            rejected.append({"stem": stem, "image": str(image_path), "json": str(json_path), "reason": f"json_parse_failed: {exc}"})
            continue

        try:
            instances = parse_labelme(obj, json_path, width, height, cfg)
        except Exception as exc:
            rejected.append({
                "stem": stem,
                "image": str(image_path),
                "json": str(json_path),
                "reason": f"labelme_invalid_or_polygon_out_of_bounds: {exc}. Expected LabelMe JSON with top-level shapes and polygon points.",
            })
            continue

        valid_instances = []
        for inst_idx, inst in enumerate(instances):
            area = polygon_area(inst["points"])
            if area <= 0:
                rejected.append({"stem": stem, "image": str(image_path), "json": str(json_path), "reason": f"empty_or_zero_area_polygon_instance_{inst_idx}"})
                continue
            inst = dict(inst)
            inst["area"] = area
            valid_instances.append(inst)

        if not valid_instances:
            rejected.append({"stem": stem, "image": str(image_path), "json": str(json_path), "reason": "empty_annotation_sample_no_valid_polygon"})
            continue

        for inst in valid_instances:
            label_counter[inst["label"]] += 1
        instance_counter += len(valid_instances)
        clean_rows.append({
            "stem": stem,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "width": width,
            "height": height,
            "instances": valid_instances,
        })

    report = {
        "raw_dataset_dir": str(raw_dir),
        "num_images": len(images),
        "num_jsons": len(jsons),
        "num_matched_pairs": len(common),
        "num_missing_json": len(missing_json),
        "num_missing_image": len(missing_image),
        "missing_json_stems": missing_json[:200],
        "missing_image_stems": missing_image[:200],
        "num_clean_samples": len(clean_rows),
        "num_rejected_samples_or_instances": len(rejected),
        "num_instances": instance_counter,
        "labels": dict(label_counter),
        "rejected": rejected,
        "manifest": str(paths["manifest"]),
    }

    write_jsonl(paths["manifest"], clean_rows)
    write_json(paths["report"], report)
    logger.info("Clean samples: %d; rejected records: %d; instances: %d", len(clean_rows), len(rejected), instance_counter)
    logger.info("Wrote clean manifest: %s", paths["manifest"])
    logger.info("Wrote clean report: %s", paths["report"])
    if not clean_rows:
        raise RuntimeError("No valid samples after cleaning. Check clean_report.json for reasons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
