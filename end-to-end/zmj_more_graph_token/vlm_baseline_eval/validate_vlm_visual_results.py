#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
官方 VLM 视觉层结果验证脚本
============================================================

用途：
1. 读取逐视角五模型测试输出 all_modes_results.json / visual_summaries_by_model.json；
2. 对每个模型视觉总结做结构一致性检查；
3. 如果提供 reference-json（你的视觉层/人工标注结果），计算视觉统计误差；
4. 输出 visual_validation_results.json、visual_validation_summary.csv、visual_validation_report.md。

reference-json 支持以下字段名：
- total_defect_count / 总气孔条目 / 总缺陷条目
- region_count / 区域缺陷统计 / 区域气孔统计
- region_max_area_mm2 / 区域最大面积(mm²)
- region_max_eq_diameter_mm / 区域最大等效直径(mm)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_REGIONS = ["一般区域", "牙孔区域", "焊接区域", "重点区域"]
HIGH_REGIONS = ["牙孔区域", "焊接区域", "重点区域"]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(round(float(x)))
    except Exception:
        return default


def normalize_region_dict(raw: Any, as_int: bool = False) -> Dict[str, float]:
    out = {r: 0 for r in ALLOWED_REGIONS}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        ks = str(k)
        if ks not in ALLOWED_REGIONS:
            continue
        out[ks] = to_int(v, 0) if as_int else to_float(v, 0.0)
    return out


def get_first(obj: Dict[str, Any], names: List[str], default: Any = None) -> Any:
    for n in names:
        if n in obj:
            return obj[n]
    return default


def normalize_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj or {}
    region_count = normalize_region_dict(get_first(obj, ["region_count", "区域缺陷统计", "区域气孔统计"]), as_int=True)
    region_area = normalize_region_dict(get_first(obj, ["region_max_area_mm2", "区域最大面积(mm²)", "区域最大面积(mm2)", "区域最大面积"]), as_int=False)
    region_dia = normalize_region_dict(get_first(obj, ["region_max_eq_diameter_mm", "区域最大等效直径(mm)", "区域最大等效直径"]), as_int=False)
    total = to_int(get_first(obj, ["total_defect_count", "总气孔条目", "总缺陷条目", "total_count"]), sum(int(v) for v in region_count.values()))
    if total <= 0:
        total = sum(int(v) for v in region_count.values())
    qlead = max_regions(region_count)
    risk = sorted(set(max_regions(region_area) + max_regions(region_dia) + [r for r in qlead if r in HIGH_REGIONS]))
    return {
        "total_defect_count": total,
        "region_count": region_count,
        "region_max_area_mm2": region_area,
        "region_max_eq_diameter_mm": region_dia,
        "quantity_leading_regions": qlead,
        "risk_focus_regions": risk,
        "view_summaries": obj.get("view_summaries", []) if isinstance(obj.get("view_summaries"), list) else [],
        "raw": obj,
    }


def max_regions(d: Dict[str, Any]) -> List[str]:
    vals = {k: to_float(v, 0.0) for k, v in d.items() if to_float(v, 0.0) > 0}
    if not vals:
        return []
    mx = max(vals.values())
    return [k for k, v in vals.items() if abs(v - mx) < 1e-9]


def load_visual_summaries(input_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = load_json(input_path)
    if isinstance(data, dict) and "visual_summaries_by_model" in data:
        return data.get("visual_summaries_by_model", {}) or {}, data
    if isinstance(data, dict):
        # assume itself is visual_summaries_by_model
        return data, {"visual_summaries_by_model": data}
    raise ValueError(f"无法识别输入结构: {input_path}")


def view_text_contains_defect(view: Dict[str, Any]) -> bool:
    text = json.dumps(view, ensure_ascii=False)
    return any(k in text for k in ["明显", "白色", "亮斑", "候选", "气孔", "缺陷"])


def self_consistency_checks(summary: Dict[str, Any]) -> Dict[str, Any]:
    notes: List[str] = []
    total = int(summary.get("total_defect_count", 0) or 0)
    region_sum = sum(int(v or 0) for v in summary.get("region_count", {}).values())
    view_summaries = summary.get("view_summaries", []) or []
    visible_yes = 0
    text_says_defect = 0
    for v in view_summaries:
        vis = str(v.get("visible_defect", "") or v.get("visible_defect_description", ""))
        if "是" in vis or "有" in vis or "候选" in vis:
            visible_yes += 1
        if view_text_contains_defect(v):
            text_says_defect += 1
    if total == 0 and text_says_defect > 0:
        notes.append("文本描述存在气孔候选，但聚合总数为0，视觉总结自相矛盾")
    if total != region_sum:
        notes.append(f"total_defect_count={total} 与 region_count求和={region_sum} 不一致")
    if not view_summaries:
        notes.append("缺少逐视角 view_summaries")
    elif len(view_summaries) < 9:
        notes.append(f"逐视角结果不足9个，当前只有{len(view_summaries)}个")
    if all(int(v or 0) == 0 for v in summary.get("region_count", {}).values()) and total > 0:
        notes.append("总数大于0但各区域数量全为0")
    return {
        "self_consistency_score_20": max(0, 20 - 5 * len(notes)),
        "self_consistency_notes": notes,
        "view_count": len(view_summaries),
        "text_says_defect_views": text_says_defect,
        "visible_yes_views": visible_yes,
    }


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compare_with_reference(pred: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    pred_total = int(pred.get("total_defect_count", 0) or 0)
    ref_total = int(ref.get("total_defect_count", 0) or 0)
    total_abs_error = abs(pred_total - ref_total)
    total_rel_error = safe_div(total_abs_error, max(ref_total, 1))

    pred_count = pred.get("region_count", {}) or {}
    ref_count = ref.get("region_count", {}) or {}
    abs_errors = {r: abs(int(pred_count.get(r, 0) or 0) - int(ref_count.get(r, 0) or 0)) for r in ALLOWED_REGIONS}
    mae = sum(abs_errors.values()) / len(ALLOWED_REGIONS)

    pred_present = {r for r in ALLOWED_REGIONS if int(pred_count.get(r, 0) or 0) > 0}
    ref_present = {r for r in ALLOWED_REGIONS if int(ref_count.get(r, 0) or 0) > 0}
    tp = len(pred_present & ref_present)
    precision = safe_div(tp, len(pred_present))
    recall = safe_div(tp, len(ref_present))
    f1 = safe_div(2 * precision * recall, precision + recall)

    pred_qlead = set(pred.get("quantity_leading_regions", []) or max_regions(pred_count))
    ref_qlead = set(ref.get("quantity_leading_regions", []) or max_regions(ref_count))
    qlead_hit = bool(pred_qlead & ref_qlead) if ref_qlead else (not pred_qlead)

    pred_risk = set(pred.get("risk_focus_regions", []) or [])
    ref_risk = set(ref.get("risk_focus_regions", []) or [])
    if not ref_risk:
        ref_risk = set(max_regions(ref.get("region_max_area_mm2", {})) + max_regions(ref.get("region_max_eq_diameter_mm", {})))
    risk_hit = bool(pred_risk & ref_risk) if ref_risk else (not pred_risk)

    # 视觉准确性粗评分，满分80，便于和自一致性20拼成100。
    score = 80.0
    score -= min(30.0, total_rel_error * 30.0)
    score -= min(20.0, mae * 3.0)
    score += 10.0 * f1
    if qlead_hit:
        score += 5.0
    else:
        score -= 5.0
    if risk_hit:
        score += 5.0
    else:
        score -= 5.0
    score = max(0.0, min(80.0, score))
    return {
        "reference_available": True,
        "visual_accuracy_score_80": round(score, 2),
        "total_abs_error": total_abs_error,
        "total_rel_error": round(total_rel_error, 4),
        "region_count_abs_errors": abs_errors,
        "region_count_mae": round(mae, 3),
        "region_presence_precision": round(precision, 4),
        "region_presence_recall": round(recall, 4),
        "region_presence_f1": round(f1, 4),
        "quantity_leading_hit": qlead_hit,
        "risk_focus_hit": risk_hit,
        "pred_quantity_leading": sorted(pred_qlead),
        "ref_quantity_leading": sorted(ref_qlead),
        "pred_risk_focus": sorted(pred_risk),
        "ref_risk_focus": sorted(ref_risk),
    }


def build_report(rows: List[Dict[str, Any]], input_path: Path, ref_path: Optional[Path]) -> str:
    lines = ["# 官方VLM视觉层结果验证报告\n\n"]
    lines.append(f"- 输入文件：`{input_path}`\n")
    lines.append(f"- 参考文件：`{ref_path if ref_path else '未提供，仅做自一致性检查'}`\n\n")
    lines.append("## 汇总\n\n")
    lines.append("| 模型 | 总数估计 | 自一致性/20 | 视觉准确/80 | 总分/100 | 主要问题 |\n")
    lines.append("|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        notes = "；".join(r.get("self_consistency_notes", [])) or "-"
        lines.append(f"| {r['model_key']} | {r['pred_total']} | {r['self_consistency_score_20']} | {r.get('visual_accuracy_score_80','')} | {r.get('visual_total_score_100','')} | {notes} |\n")
    lines.append("\n## 分模型详情\n\n")
    for r in rows:
        lines.append(f"### {r['model_key']}\n\n")
        lines.append("```json\n")
        lines.append(json.dumps(r, ensure_ascii=False, indent=2))
        lines.append("\n```\n\n")
    return "".join(lines)


def run_validation(input_path: Path, output_dir: Path, reference_json: Optional[Path]) -> None:
    visual_summaries, raw_input = load_visual_summaries(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = None
    if reference_json and reference_json.exists():
        ref = normalize_summary(load_json(reference_json))
        save_json(output_dir / "reference_normalized.json", ref)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for model_key, raw_summary in visual_summaries.items():
        pred = normalize_summary(raw_summary)
        consistency = self_consistency_checks(pred)
        row = {
            "model_key": model_key,
            "pred_total": pred.get("total_defect_count", 0),
            "pred_region_count": pred.get("region_count", {}),
            **consistency,
        }
        if ref is not None:
            comp = compare_with_reference(pred, ref)
            row.update(comp)
            row["visual_total_score_100"] = round(float(row["visual_accuracy_score_80"]) + float(row["self_consistency_score_20"]), 2)
        else:
            row["reference_available"] = False
            row["visual_accuracy_score_80"] = ""
            row["visual_total_score_100"] = row["self_consistency_score_20"]
        rows.append(row)
        details[model_key] = {"pred_normalized": pred, "validation": row, "raw_summary": raw_summary}

    save_json(output_dir / "visual_validation_results.json", {"input": str(input_path), "reference": str(reference_json) if reference_json else "", "rows": rows, "details": details})
    write_csv(output_dir / "visual_validation_summary.csv", rows, [
        "model_key", "pred_total", "view_count", "text_says_defect_views", "visible_yes_views",
        "self_consistency_score_20", "visual_accuracy_score_80", "visual_total_score_100",
        "total_abs_error", "total_rel_error", "region_count_mae", "region_presence_precision", "region_presence_recall", "region_presence_f1",
        "quantity_leading_hit", "risk_focus_hit", "self_consistency_notes",
    ])
    report = build_report(rows, input_path, reference_json)
    (output_dir / "visual_validation_report.md").write_text(report, encoding="utf-8")
    print("=" * 100)
    print("视觉层验证完成")
    print(f"[done] input  : {input_path}")
    print(f"[done] output : {output_dir}")
    print(f"[done] report : {output_dir / 'visual_validation_report.md'}")
    print("=" * 100)
    for r in rows:
        print(f"{r['model_key']}: total={r['pred_total']}, self={r['self_consistency_score_20']}/20, visual={r.get('visual_accuracy_score_80')}/80, final={r.get('visual_total_score_100')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="官方VLM视觉层结果验证")
    p.add_argument("--input", required=True, help="all_modes_results.json 或 visual_summaries_by_model.json")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--reference-json", default="", help="可选：你的视觉层/人工标注参考摘要 JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(Path(args.input), Path(args.output_dir), Path(args.reference_json) if args.reference_json else None)


if __name__ == "__main__":
    main()
