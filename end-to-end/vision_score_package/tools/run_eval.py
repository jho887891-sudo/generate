#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from common_coco_eval import evaluate_coco_multiple_iou_types, print_brief_results


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CATEGORY_ID = 1


def load_cfg(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def polygon_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=np.float32)
    return float(abs(cv2.contourArea(pts)))


def polygon_bbox(points: List[List[float]]) -> List[float]:
    pts = np.asarray(points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    return [float(x), float(y), float(w), float(h)]


def flatten_polygon(points: Iterable[Iterable[float]]) -> List[float]:
    flat = []
    for x, y in points:
        flat.extend([float(x), float(y)])
    return flat


def collect_labelme_pairs(labelme_dir: Path) -> List[Tuple[Path, Path]]:
    images = {}
    for p in labelme_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p
    pairs = []
    for stem, img in sorted(images.items()):
        js = img.with_suffix(".json")
        if js.exists():
            pairs.append((img, js))
    if not pairs:
        raise RuntimeError(f"No image/json pairs found under {labelme_dir}")
    return pairs


def labelme_to_coco(labelme_dir: Path, out_json: Path) -> Tuple[Path, List[Dict]]:
    pairs = collect_labelme_pairs(labelme_dir)
    images = []
    annotations = []
    image_records = []
    ann_id = 1

    for image_id, (img_path, json_path) in enumerate(pairs, start=1):
        with Image.open(img_path) as img:
            width, height = img.size
        images.append({"id": image_id, "file_name": img_path.name, "width": width, "height": height})
        image_records.append({"id": image_id, "path": str(img_path), "width": width, "height": height})

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for shape in obj.get("shapes", []):
            if shape.get("shape_type", "polygon") not in (None, "polygon"):
                continue
            pts = shape.get("points") or []
            if len(pts) < 3:
                continue
            clean = []
            for x, y in pts:
                x = max(0.0, min(float(width - 1), float(x)))
                y = max(0.0, min(float(height - 1), float(y)))
                clean.append([x, y])
            area = polygon_area(clean)
            if area <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "segmentation": [flatten_polygon(clean)],
                "bbox": polygon_bbox(clean),
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": "vk3 LabelMe converted to COCO"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": "defect", "supercategory": "defect"}],
    }
    ensure_dir(out_json.parent)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"[gt] wrote {out_json} images={len(images)} annotations={len(annotations)}")
    return out_json, image_records


def image_to_tensor(img_path: Path, img_size: int, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, (orig_w, orig_h)


def mask_to_coco_results(mask: np.ndarray, score_map: np.ndarray, image_id: int, min_area: float) -> List[Dict]:
    results = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label_id in range(1, num_labels):
        area = float(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == label_id).astype(np.uint8)
        score = float(score_map[labels == label_id].mean()) if np.any(labels == label_id) else 0.0
        results.extend(binary_mask_to_results(comp, image_id, score, min_area))
    return results


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_dedup_predictions(preds: List[Dict], iou_thr: float) -> List[Dict]:
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    kept = []
    suppressed = np.zeros(len(preds), dtype=bool)
    for i, pred in enumerate(preds):
        if suppressed[i]:
            continue
        kept.append(pred)
        for j in range(i + 1, len(preds)):
            if suppressed[j]:
                continue
            if bbox_iou(pred["bbox"], preds[j]["bbox"]) >= iou_thr:
                suppressed[j] = True
    return kept


def to_float_score(score_obj, default: float = 1.0) -> float:
    if score_obj is None:
        return float(default)
    if isinstance(score_obj, (float, int, np.floating, np.integer)):
        return float(score_obj)
    for attr in ("value", "score", "confidence"):
        if hasattr(score_obj, attr):
            try:
                return float(getattr(score_obj, attr))
            except Exception:
                pass
    if isinstance(score_obj, dict):
        for key in ("value", "score", "confidence"):
            if key in score_obj:
                try:
                    return float(score_obj[key])
                except Exception:
                    pass
    try:
        return float(str(score_obj))
    except Exception:
        return float(default)


def binary_mask_to_results(mask: np.ndarray, image_id: int, score: float, min_area: float) -> List[Dict]:
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rle = binary_mask_to_rle(mask)
        out.append({
            "image_id": int(image_id),
            "category_id": CATEGORY_ID,
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": float(score),
            "segmentation": rle,
        })
    return out


def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    mask = np.asfortranarray((mask > 0).astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def save_semantic_visualization(model_name: str, rec: Dict, prob_np: np.ndarray, mask: np.ndarray, cfg: Dict) -> None:
    vis_cfg = cfg.get("visualization", {}) or {}
    if not bool(vis_cfg.get("save_semantic", True)):
        return
    img_path = Path(rec["path"])
    image = cv2.imread(str(img_path))
    if image is None:
        return
    alpha = float(vis_cfg.get("alpha", 0.45))
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[mask > 0] = (0, 220, 80)
    overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept_id = 1
    min_area = float(cfg["thresholds"]["min_area"])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 1)
        cv2.putText(overlay, str(kept_id), (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        kept_id += 1

    prob_vis = np.clip(prob_np * 255, 0, 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    combined = np.concatenate([overlay, prob_color], axis=1)
    cv2.putText(combined, f"{model_name} mask overlay", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(combined, "foreground probability", (image.shape[1] + 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    out_dir = Path(cfg["output_dir"]) / "visualizations" / "semantic" / model_name
    ensure_dir(out_dir)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_semantic.png"), combined)


def add_train_paths(vision_train_root: Path) -> None:
    sys.path.insert(0, str(vision_train_root / "tools" / "vision_compare"))
    sys.path.insert(0, str(vision_train_root / "train" / "vision_compare"))


def load_torch_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def predict_semantic(model_name: str, cfg: Dict, images: List[Dict]) -> List[Dict]:
    add_train_paths(Path(cfg["vision_train_root"]))
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    img_size = int(cfg["img_size"])
    num_classes = int(cfg["num_classes"])
    threshold = float(cfg["thresholds"]["semantic_prob"])
    min_area = float(cfg["thresholds"]["min_area"])
    weight_path = Path(cfg["weights"][model_name])

    if model_name == "unet":
        from train_unet import ResNetUNet
        model = ResNetUNet(num_classes, None)
    elif model_name == "deeplabv3plus":
        from train_deeplabv3plus import DeepLabV3PlusResNet50
        model = DeepLabV3PlusResNet50(num_classes, None)
    else:
        raise ValueError(model_name)

    ckpt = load_torch_checkpoint(weight_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.to(device).eval()

    results = []
    with torch.no_grad():
        for idx, rec in enumerate(images, start=1):
            tensor, (orig_w, orig_h) = image_to_tensor(Path(rec["path"]), img_size, device)
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[:, 1:2]
            prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0]
            prob_np = prob.detach().cpu().numpy()
            mask = (prob_np >= threshold).astype(np.uint8)
            save_semantic_visualization(model_name, rec, prob_np, mask, cfg)
            results.extend(mask_to_coco_results(mask, prob_np, int(rec["id"]), min_area))
            if idx % 10 == 0 or idx == len(images):
                print(f"[{model_name}] predicted {idx}/{len(images)}")
    return results


def predict_mask_rcnn(cfg: Dict, images: List[Dict]) -> List[Dict]:
    add_train_paths(Path(cfg["vision_train_root"]))
    from train_mask_rcnn import build_model

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    img_size = int(cfg["img_size"])
    score_thr = float(cfg["thresholds"]["mask_rcnn_score"])
    min_area = float(cfg["thresholds"]["min_area"])
    weight_path = Path(cfg["weights"]["mask_rcnn"])

    model = build_model(int(cfg["num_classes"]), None)
    ckpt = load_torch_checkpoint(weight_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.to(device).eval()

    results = []
    with torch.no_grad():
        for idx, rec in enumerate(images, start=1):
            tensor, (orig_w, orig_h) = image_to_tensor(Path(rec["path"]), img_size, device)
            out = model([tensor[0]])[0]
            masks = out.get("masks", torch.empty(0))
            scores = out.get("scores", torch.empty(0)).detach().cpu().numpy()
            for j, score in enumerate(scores):
                if float(score) < score_thr:
                    continue
                mask = masks[j, 0].detach().cpu().numpy()
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                results.extend(binary_mask_to_results(mask >= 0.5, int(rec["id"]), float(score), min_area))
            if idx % 10 == 0 or idx == len(images):
                print(f"[mask_rcnn] predicted {idx}/{len(images)}")
    return results


def predict_yolo(model_name: str, cfg: Dict, images: List[Dict]) -> List[Dict]:
    from ultralytics import YOLO

    weight_path = Path(cfg["weights"][model_name])
    model = YOLO(str(weight_path))
    device = str(cfg.get("device", "cuda:0")).replace("cuda:", "")
    conf = float(cfg["thresholds"]["yolo_conf"])
    img_size = int(cfg["img_size"])
    min_area = float(cfg["thresholds"]["min_area"])
    results = []

    for idx, rec in enumerate(images, start=1):
        pred = model.predict(str(rec["path"]), imgsz=img_size, conf=conf, device=device, verbose=False)
        if pred:
            r = pred[0]
            scores = r.boxes.conf.detach().cpu().numpy() if r.boxes is not None else []
            if r.masks is not None:
                for poly, score in zip(r.masks.xy, scores):
                    pts = np.asarray(poly, dtype=np.float32)
                    if len(pts) < 3:
                        continue
                    area = float(cv2.contourArea(pts))
                    if area < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
                    poly_mask = np.zeros((int(rec["height"]), int(rec["width"])), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [pts.astype(np.int32)], 1)
                    rle = binary_mask_to_rle(poly_mask)
                    results.append({
                        "image_id": int(rec["id"]),
                        "category_id": CATEGORY_ID,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                        "segmentation": rle,
                    })
        if idx % 10 == 0 or idx == len(images):
            print(f"[{model_name}] predicted {idx}/{len(images)}")
    return results


def predict_yolo_sahi(model_name: str, cfg: Dict, images: List[Dict]) -> List[Dict]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    sahi_cfg = cfg.get("sahi", {}) or {}
    weight_path = Path(cfg["weights"][model_name])
    device = str(cfg.get("device", "cuda:0"))
    model_type = str(sahi_cfg.get("model_type", "yolov11"))
    conf = float(sahi_cfg.get("confidence_threshold", cfg["thresholds"].get("yolo_conf", 0.05)))
    image_size = int(sahi_cfg.get("image_size", 1280))
    slice_height = int(sahi_cfg.get("slice_height", 640))
    slice_width = int(sahi_cfg.get("slice_width", 640))
    overlap_h = float(sahi_cfg.get("overlap_height_ratio", 0.3))
    overlap_w = float(sahi_cfg.get("overlap_width_ratio", 0.3))
    dedup_iou = float(sahi_cfg.get("dedup_iou_threshold", 0.55))
    min_area = float(cfg["thresholds"]["min_area"])

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=str(weight_path),
        confidence_threshold=conf,
        device=device,
        image_size=image_size,
    )

    results = []
    for idx, rec in enumerate(images, start=1):
        image = cv2.imread(str(rec["path"]))
        if image is None:
            continue
        sliced = get_sliced_prediction(
            image,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_h,
            overlap_width_ratio=overlap_w,
        )
        raw_preds = []
        for obj in sliced.object_prediction_list:
            if obj.mask is None:
                continue
            mask_bool = obj.mask.bool_mask
            if mask_bool is None:
                continue
            bb = obj.bbox
            try:
                x1, y1, x2, y2 = int(bb.minx), int(bb.miny), int(bb.maxx), int(bb.maxy)
            except Exception:
                x1, y1, x2, y2 = map(int, bb.to_xyxy())
            score = to_float_score(getattr(obj, "score", None), default=1.0)
            raw_preds.append({"bbox": (x1, y1, x2, y2), "score": score, "mask_bool": mask_bool})

        for pred in nms_dedup_predictions(raw_preds, dedup_iou):
            mask = np.asarray(pred["mask_bool"], dtype=np.uint8)
            score = float(pred["score"])
            results.extend(binary_mask_to_results(mask, int(rec["id"]), score, min_area))
        if idx % 10 == 0 or idx == len(images):
            print(f"[{model_name}/sahi] predicted {idx}/{len(images)} raw={len(raw_preds)}")
    return results


def clean_metrics(results: Dict) -> Dict:
    return {k: v for k, v in results.items() if k != "raw"}


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_model(model_name: str, cfg: Dict, gt_json: Path, images: List[Dict]) -> Dict:
    weight_path = Path(cfg["weights"][model_name])
    if not weight_path.exists() or weight_path.stat().st_size == 0:
        print(f"[skip] {model_name}: missing weight {weight_path}")
        return {"model": model_name, "status": "missing_weight", "weight": str(weight_path)}

    sahi_models = set((cfg.get("sahi", {}) or {}).get("models", []))
    if model_name in {"unet", "deeplabv3plus"}:
        preds = predict_semantic(model_name, cfg, images)
    elif model_name == "mask_rcnn":
        preds = predict_mask_rcnn(cfg, images)
    elif model_name in sahi_models:
        preds = predict_yolo_sahi(model_name, cfg, images)
    else:
        preds = predict_yolo(model_name, cfg, images)

    out_dir = Path(cfg["output_dir"])
    pred_path = out_dir / "predictions" / f"{model_name}_predictions.json"
    save_json(pred_path, preds)
    print(f"[pred] wrote {pred_path} detections={len(preds)}")

    metrics = evaluate_coco_multiple_iou_types(
        gt_json=str(gt_json),
        pred=str(pred_path),
        iou_types=("bbox", "segm"),
        class_ids=[CATEGORY_ID],
        class_names=[cfg.get("class_name", "defect")],
        print_summary=True,
    )
    print_brief_results(metrics)
    cleaned = clean_metrics(metrics)
    metrics_path = out_dir / "metrics" / f"{model_name}_metrics.json"
    save_json(metrics_path, cleaned)
    return {"model": model_name, "status": "ok", "weight": str(weight_path), "predictions": str(pred_path), "metrics": cleaned}


def write_summary(out_dir: Path, rows: List[Dict]) -> None:
    csv_path = out_dir / "summary.csv"
    fields = ["model", "status", "bbox_AP", "bbox_AP50", "bbox_AP75", "segm_AP", "segm_AP50", "segm_AP75", "weight"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            writer.writerow({
                "model": row.get("model"),
                "status": row.get("status"),
                "bbox_AP": metrics.get("bbox", {}).get("AP", ""),
                "bbox_AP50": metrics.get("bbox", {}).get("AP50", ""),
                "bbox_AP75": metrics.get("bbox", {}).get("AP75", ""),
                "segm_AP": metrics.get("segm", {}).get("AP", ""),
                "segm_AP50": metrics.get("segm", {}).get("AP50", ""),
                "segm_AP75": metrics.get("segm", {}).get("AP75", ""),
                "weight": row.get("weight", ""),
            })
    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| model | status | bbox AP | bbox AP50 | segm AP | segm AP50 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            metrics = row.get("metrics", {})
            bbox = metrics.get("bbox", {})
            segm = metrics.get("segm", {})
            f.write(
                f"| {row.get('model')} | {row.get('status')} | "
                f"{bbox.get('AP', '')} | {bbox.get('AP50', '')} | "
                f"{segm.get('AP', '')} | {segm.get('AP50', '')} |\n"
            )
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names to evaluate.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = ensure_dir(cfg["output_dir"])
    gt_json, images = labelme_to_coco(Path(cfg["test_labelme_dir"]), out_dir / "gt_vk3_coco.json")

    all_models = ["unet", "deeplabv3plus", "mask_rcnn", "yolov8n_seg", "yolov11n_seg", "yolo_x_best_lr_75"]
    models = args.models or all_models
    rows = []
    for model_name in models:
        print(f"\n========== evaluating {model_name} ==========")
        try:
            rows.append(evaluate_model(model_name, cfg, gt_json, images))
        except Exception as exc:
            print(f"[error] {model_name}: {exc}")
            rows.append({"model": model_name, "status": f"error: {exc}", "weight": str(cfg.get("weights", {}).get(model_name, ""))})
    write_summary(out_dir, rows)
    save_json(out_dir / "summary_full.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
