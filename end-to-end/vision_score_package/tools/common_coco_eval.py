"""COCO evaluation helpers copied from the project scoring script, with demo code removed."""
from __future__ import annotations

import copy
import json
import os
from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _check_iou_type(iou_type: str):
    valid = {"bbox", "segm", "keypoints"}
    if iou_type not in valid:
        raise ValueError(f"iou_type must be one of {valid}, but got: {iou_type}")


def _metric_names_for_iou_type(iou_type: str):
    if iou_type in {"bbox", "segm"}:
        return ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    if iou_type == "keypoints":
        return ["AP", "AP50", "AP75", "APm", "APl"]
    raise ValueError(f"Unsupported iou_type: {iou_type}")


def _summarize_coco_stats(coco_eval: COCOeval, iou_type: str):
    metrics = _metric_names_for_iou_type(iou_type)
    return {
        metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else float("nan"))
        for idx, metric in enumerate(metrics)
    }


def _per_category_ap(coco_eval: COCOeval, class_names=None):
    if class_names is None:
        return OrderedDict(), OrderedDict()
    precisions = coco_eval.eval.get("precision", None)
    if precisions is None:
        return OrderedDict(), OrderedDict()
    num_classes = precisions.shape[2]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names length ({len(class_names)}) != precision.shape[2] ({num_classes}).")

    results_per_category_ap = []
    results_per_category_ap50 = []
    for idx, name in enumerate(class_names):
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category_ap.append((name, float(ap * 100)))

        precision50 = precisions[0, :, idx, 0, -1]
        precision50 = precision50[precision50 > -1]
        ap50 = np.mean(precision50) if precision50.size else float("nan")
        results_per_category_ap50.append((name, float(ap50 * 100)))

    ap_dict = OrderedDict((f"AP-{name}", ap) for name, ap in results_per_category_ap)
    ap50_dict = OrderedDict((f"AP50-{name}", ap50) for name, ap50 in results_per_category_ap50)
    return ap_dict, ap50_dict


def save_coco_results(coco_results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(coco_results, f, ensure_ascii=False)
    return save_path


def load_coco_results(pred):
    if isinstance(pred, str):
        with open(pred, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(pred, list):
        return pred
    raise TypeError("pred must be either a path to json file or a list of coco result dicts.")


def evaluate_coco(
    gt_json,
    pred,
    iou_type="bbox",
    class_ids=None,
    class_names=None,
    img_ids=None,
    kpt_oks_sigmas=None,
    remove_bbox_for_segm=True,
    print_summary=True,
):
    _check_iou_type(iou_type)
    class_ids = class_ids or []
    coco_results = load_coco_results(pred)

    if len(coco_results) == 0:
        metrics = _metric_names_for_iou_type(iou_type)
        return {iou_type: {m: float("nan") for m in metrics}, "raw": {"coco_eval": None, "coco_results": coco_results}}

    if iou_type == "segm" and remove_bbox_for_segm:
        coco_results = copy.deepcopy(coco_results)
        for x in coco_results:
            x.pop("bbox", None)

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if class_ids:
        coco_eval.params.catIds = class_ids
    if img_ids:
        coco_eval.params.imgIds = img_ids
    if kpt_oks_sigmas is not None:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    coco_eval.evaluate()
    coco_eval.accumulate()
    if print_summary:
        coco_eval.summarize()

    summary = _summarize_coco_stats(coco_eval, iou_type)
    if class_names is not None:
        ap_dict, ap50_dict = _per_category_ap(coco_eval, class_names=class_names)
        summary.update(ap_dict)
        summary.update(ap50_dict)

    return {iou_type: summary, "raw": {"coco_eval": coco_eval, "coco_results": coco_results}}


def evaluate_coco_multiple_iou_types(
    gt_json,
    pred,
    iou_types=("bbox",),
    class_ids=None,
    class_names=None,
    img_ids=None,
    kpt_oks_sigmas=None,
    remove_bbox_for_segm=True,
    print_summary=True,
):
    results = OrderedDict()
    raw = {}
    for iou_type in iou_types:
        out = evaluate_coco(
            gt_json=gt_json,
            pred=pred,
            iou_type=iou_type,
            class_ids=class_ids,
            class_names=class_names,
            img_ids=img_ids,
            kpt_oks_sigmas=kpt_oks_sigmas,
            remove_bbox_for_segm=remove_bbox_for_segm,
            print_summary=print_summary,
        )
        results[iou_type] = out[iou_type]
        raw[iou_type] = out["raw"]
    results["raw"] = raw
    return results


def print_brief_results(results):
    for task, res in results.items():
        if task == "raw":
            continue
        print(f"\n===== {task} =====")
        for k in [k for k in ["AP", "AP50", "AP75", "APs", "APm", "APl"] if k in res]:
            print(f"{k}: {res[k]:.3f}")
