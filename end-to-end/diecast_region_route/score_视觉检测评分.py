#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉层评分脚本：直接读取 all_modes_results.json，所有路径写死在脚本内部
=====================================================================

用途：
1. 直接从 all_modes_results.json 中提取 API / VLM 的 visual_summary；
2. 与参考视觉结果 ref_visual_summary_38069.json 对比；
3. 输出视觉层评分 visual_layer_score.json；
4. 自动生成 all_modes_results_with_visual_score.json，方便后续汇总端到端综合分；
5. 不需要命令行传参，直接 python 运行即可。

使用方式：
    cd /home/T7/wkq/end-to-end
    /home/T7/public/miniconda3/envs/finetune1/bin/python \
      /home/T7/wkq/end-to-end/API_test/score_visual_layer_from_all_modes_hardcoded.py

如果你想指定某一次结果，不想自动找最新结果，只需要修改下面的 INPUT_ALL_MODES_JSON。
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# 1. 路径配置：全部写死在这里
# ============================================================

# 方式A：指定某一次 all_modes_results.json。
# 如果这里填了具体路径，脚本会优先用这个文件。
# 例子：
# INPUT_ALL_MODES_JSON = "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q/qwen35_plus_graph_300q_20260521_150000/all_modes_results.json"
INPUT_ALL_MODES_JSON = ""

# 方式B：INPUT_ALL_MODES_JSON 为空时，自动从这些目录里找最新的 all_modes_results.json。
# 你当前常用的两个 API 结果目录都放在这里了。
AUTO_FIND_LATEST = True
RESULT_SEARCH_ROOTS = [
    "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q/qwen35_plus_graph_300q_20260523_093416",
]

# 参考视觉结果：建议放你的 YOLO+mask 对齐结果，或者人工修正后的视觉统计 JSON。
# 这个文件格式需要和 visual_summary_for_scoring.json 类似。
REF_VISUAL_JSON = "/home/T7/wkq/end-to-end/API_test/ref_visual_summary_38073.json"

# 输出目录：为空时默认输出到 all_modes_results.json 同级目录下的 visual_score/。
# 如果想固定输出到某处，可以写死：
# OUTPUT_DIR = "/home/T7/wkq/end-to-end/compare_outputs/visual_score_38069"
OUTPUT_DIR = "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q/qwen35_plus_graph_300q_20260523_093416_score/visual"

# 输出文件名
VISUAL_SCORE_JSON_NAME = "visual_layer_score.json"
VISUAL_SCORE_CSV_NAME = "visual_layer_score_breakdown.csv"
PRED_VISUAL_EXTRACTED_NAME = "pred_visual_extracted_from_all_modes.json"
REF_VISUAL_USED_NAME = "ref_visual_used.json"
ALL_MODES_WITH_VISUAL_SCORE_NAME = "all_modes_results_with_visual_score.json"

# 是否把视觉分写回 all_modes_results_with_visual_score.json
WRITE_BACK_ALL_MODES = True


# ============================================================
# 2. 评分区域与规则配置
# ============================================================

VALID_REGIONS = ["一般区域", "牙孔区域", "焊接区域", "重点区域"]
HIGH_IMPACT_REGIONS = ["牙孔区域", "焊接区域", "重点区域"]
INVALID_REGION_TERMS = ["背景", "未知", "重叠", "无法判断", "未标注", "background", "unknown", "overlap"]

# 视觉层总分100：
# 总数15 + 区域数量25 + 高影响区域20 + 面积/等效直径20 + 无效区域过滤10 + 置信度10
SCORE_WEIGHTS = {
    "total_defect_count": 15.0,
    "region_count": 25.0,
    "risk_focus_regions": 20.0,
    "area_and_eq_diameter": 20.0,
    "invalid_region_filter": 10.0,
    "confidence_calibration": 10.0,
}

# 面积/等效直径容差。API 视觉面积是估计值，不要设太严。
AREA_REL_TOLERANCE = 0.45
DIAMETER_REL_TOLERANCE = 0.35


# ============================================================
# 3. 基础工具
# ============================================================

def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def normalize_region_name(x: Any) -> str:
    s = str(x or "").strip()
    alias = {
        "1": "一般区域",
        "2": "牙孔区域",
        "3": "焊接区域",
        "4": "重点区域",
        "Region 1": "一般区域",
        "Region 2": "牙孔区域",
        "Region 3": "焊接区域",
        "Region 4": "重点区域",
        "region 1": "一般区域",
        "region 2": "牙孔区域",
        "region 3": "焊接区域",
        "region 4": "重点区域",
    }
    if s in alias:
        return alias[s]
    for r in VALID_REGIONS:
        if r in s:
            return r
    return s or "无法判断"


def find_latest_all_modes_json(search_roots: List[str]) -> Path:
    candidates: List[Path] = []
    for root in search_roots:
        p = Path(root)
        if not p.exists():
            continue
        candidates.extend(p.rglob("all_modes_results.json"))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "没有在 RESULT_SEARCH_ROOTS 中找到 all_modes_results.json。\n"
            "请检查 RESULT_SEARCH_ROOTS，或者直接填写 INPUT_ALL_MODES_JSON。"
        )

    # 用修改时间选最新文件
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_input_all_modes_path() -> Path:
    if INPUT_ALL_MODES_JSON.strip():
        return Path(INPUT_ALL_MODES_JSON).resolve()
    if AUTO_FIND_LATEST:
        return find_latest_all_modes_json(RESULT_SEARCH_ROOTS).resolve()
    raise ValueError("INPUT_ALL_MODES_JSON 为空，且 AUTO_FIND_LATEST=False，无法确定输入文件。")


def resolve_output_dir(input_all_modes_path: Path) -> Path:
    if OUTPUT_DIR.strip():
        return Path(OUTPUT_DIR).resolve()
    return input_all_modes_path.parent / "visual_score"


# ============================================================
# 4. 从 all_modes_results.json 提取 visual_summary
# ============================================================

def looks_like_visual_summary(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    must_any = {
        "total_defect_count",
        "region_count",
        "region_max_area_mm2",
        "region_max_eq_diameter_mm",
        "defect_candidates",
        "risk_focus_regions",
    }
    return len(keys & must_any) >= 3


def find_visual_summary_recursive(obj: Any, path: str = "$") -> Tuple[Dict[str, Any] | None, str]:
    """
    递归查找 visual_summary。
    优先级：
    1. dict["visual_summary"]
    2. dict["visual_summary_for_scoring"]
    3. cases[*]["visual_summary"]
    4. 任意看起来像视觉摘要的 dict
    """
    if isinstance(obj, dict):
        for k in ["visual_summary", "visual_summary_for_scoring", "pred_visual_summary"]:
            v = obj.get(k)
            if looks_like_visual_summary(v):
                return v, f"{path}.{k}"

        # 优先进入 cases，因为你的 all_modes_results.json 里通常每个 case 也有 visual_summary
        cases = obj.get("cases")
        if isinstance(cases, list):
            for i, case in enumerate(cases):
                found, found_path = find_visual_summary_recursive(case, f"{path}.cases[{i}]")
                if found is not None:
                    return found, found_path

        # 再搜索 modes 等其他字段
        for k, v in obj.items():
            if k == "cases":
                continue
            if looks_like_visual_summary(v):
                return v, f"{path}.{k}"
            if isinstance(v, (dict, list)):
                found, found_path = find_visual_summary_recursive(v, f"{path}.{k}")
                if found is not None:
                    return found, found_path

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found, found_path = find_visual_summary_recursive(item, f"{path}[{i}]")
            if found is not None:
                return found, found_path

    return None, ""


def extract_pred_visual_from_all_modes(all_modes: Dict[str, Any], input_path: Path) -> Tuple[Dict[str, Any], str]:
    # 顶层字段最优先
    for k in ["visual_summary", "visual_summary_for_scoring", "pred_visual_summary"]:
        if looks_like_visual_summary(all_modes.get(k)):
            return all_modes[k], f"{input_path}::{k}"

    pred, found_path = find_visual_summary_recursive(all_modes)
    if pred is None:
        raise ValueError(
            "没有在 all_modes_results.json 中找到 visual_summary。\n"
            "请检查是否包含顶层 visual_summary，或 cases[*].visual_summary。"
        )
    return pred, f"{input_path}::{found_path}"


# ============================================================
# 5. 评分函数
# ============================================================

def count_similarity(pred: Any, ref: Any) -> float:
    """
    数量相似度：0~1
    ref=0,pred=0 得1；
    ref=0,pred>0 得0；
    其他按相对误差扣分。
    """
    pred_v = safe_num(pred)
    ref_v = safe_num(ref)
    if ref_v == 0:
        return 1.0 if pred_v == 0 else 0.0
    return max(0.0, 1.0 - abs(pred_v - ref_v) / max(ref_v, 1.0))


def metric_similarity(pred: Any, ref: Any, tolerance: float = 0.35) -> float:
    """
    面积/直径相似度：0~1
    tolerance=0.35 表示35%以内误差扣得比较轻。
    API面积是估计值，所以不要设太严。
    """
    pred_v = safe_num(pred)
    ref_v = safe_num(ref)
    if ref_v == 0:
        return 1.0 if pred_v == 0 else 0.0

    rel_err = abs(pred_v - ref_v) / max(ref_v, 1e-6)
    return max(0.0, 1.0 - rel_err / tolerance)


def get_region_dict(obj: Dict[str, Any], key: str) -> Dict[str, float]:
    d = obj.get(key, {}) or {}
    if not isinstance(d, dict):
        d = {}
    out = {r: 0.0 for r in VALID_REGIONS}
    for k, v in d.items():
        rr = normalize_region_name(k)
        if rr in out:
            out[rr] = safe_num(v)
    return out


def infer_total_count(obj: Dict[str, Any]) -> float:
    if "total_defect_count" in obj:
        return safe_num(obj.get("total_defect_count", 0))
    rc = get_region_dict(obj, "region_count")
    if any(v > 0 for v in rc.values()):
        return sum(rc.values())
    cands = obj.get("defect_candidates", []) or []
    if isinstance(cands, list):
        return len(cands)
    return 0.0


def score_total_count(pred: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    p = infer_total_count(pred)
    r = infer_total_count(ref)
    sim = count_similarity(p, r)
    return round(SCORE_WEIGHTS["total_defect_count"] * sim, 2), {
        "pred_total": p,
        "ref_total": r,
        "similarity": round(sim, 3),
    }


def score_region_count(pred: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    pred_counts = get_region_dict(pred, "region_count")
    ref_counts = get_region_dict(ref, "region_count")

    sims = {}
    total = 0.0
    max_total = 0.0
    for r in VALID_REGIONS:
        sim = count_similarity(pred_counts.get(r, 0), ref_counts.get(r, 0))
        sims[r] = round(sim, 3)
        weight = 1.3 if r in HIGH_IMPACT_REGIONS else 1.0
        total += sim * weight
        max_total += weight

    score = SCORE_WEIGHTS["region_count"] * total / max(max_total, 1e-6)
    return round(score, 2), {
        "pred_region_count": pred_counts,
        "ref_region_count": ref_counts,
        "region_similarity": sims,
    }


def score_risk_focus(pred: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    pred_set = {normalize_region_name(x) for x in (pred.get("risk_focus_regions", []) or [])}
    ref_set = {normalize_region_name(x) for x in (ref.get("risk_focus_regions", []) or [])}
    pred_set = {x for x in pred_set if x in HIGH_IMPACT_REGIONS}
    ref_set = {x for x in ref_set if x in HIGH_IMPACT_REGIONS}

    # 兜底：如果没有 risk_focus_regions，则用高影响区域中有缺陷的区域推断
    if not pred_set:
        pc = get_region_dict(pred, "region_count")
        pred_set = {r for r in HIGH_IMPACT_REGIONS if pc.get(r, 0) > 0}
    if not ref_set:
        rc = get_region_dict(ref, "region_count")
        ref_set = {r for r in HIGH_IMPACT_REGIONS if rc.get(r, 0) > 0}

    if not pred_set and not ref_set:
        jaccard = 1.0
    else:
        jaccard = len(pred_set & ref_set) / max(len(pred_set | ref_set), 1)

    missed_high = sorted(list(ref_set - pred_set))
    false_high = sorted(list(pred_set - ref_set))

    score = SCORE_WEIGHTS["risk_focus_regions"] * jaccard
    return round(score, 2), {
        "pred_risk_focus": sorted(list(pred_set)),
        "ref_risk_focus": sorted(list(ref_set)),
        "missed_high_impact_regions": missed_high,
        "false_high_impact_regions": false_high,
        "jaccard": round(jaccard, 3),
    }


def score_area_diameter(pred: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    pred_area = get_region_dict(pred, "region_max_area_mm2")
    ref_area = get_region_dict(ref, "region_max_area_mm2")
    pred_dia = get_region_dict(pred, "region_max_eq_diameter_mm")
    ref_dia = get_region_dict(ref, "region_max_eq_diameter_mm")

    total = 0.0
    max_total = 0.0
    details: Dict[str, Any] = {}

    for r in VALID_REGIONS:
        area_sim = metric_similarity(pred_area.get(r, 0), ref_area.get(r, 0), tolerance=AREA_REL_TOLERANCE)
        dia_sim = metric_similarity(pred_dia.get(r, 0), ref_dia.get(r, 0), tolerance=DIAMETER_REL_TOLERANCE)

        weight = 1.4 if r in HIGH_IMPACT_REGIONS else 1.0
        region_sim = 0.45 * area_sim + 0.55 * dia_sim
        total += region_sim * weight
        max_total += weight

        details[r] = {
            "pred_area": pred_area.get(r, 0),
            "ref_area": ref_area.get(r, 0),
            "area_similarity": round(area_sim, 3),
            "pred_eq_diameter": pred_dia.get(r, 0),
            "ref_eq_diameter": ref_dia.get(r, 0),
            "diameter_similarity": round(dia_sim, 3),
            "region_metric_similarity": round(region_sim, 3),
        }

    score = SCORE_WEIGHTS["area_and_eq_diameter"] * total / max(max_total, 1e-6)
    return round(score, 2), details


def score_invalid_region_filter(pred: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    看 API 有没有把背景/重叠/未知/无法判断当作有效缺陷。
    """
    candidates = pred.get("defect_candidates", []) or []
    bad = []

    if not isinstance(candidates, list):
        candidates = []

    for c in candidates:
        if not isinstance(c, dict):
            continue
        region = str(c.get("region_judgement", ""))
        if any(t in region for t in INVALID_REGION_TERMS):
            bad.append({
                "view_index": c.get("view_index"),
                "region_judgement": region,
                "confidence": c.get("confidence"),
            })

    if not candidates:
        # 无候选时不因无效区域扣分；漏检由数量项扣
        score = SCORE_WEIGHTS["invalid_region_filter"]
    else:
        ratio = len(bad) / max(len(candidates), 1)
        score = SCORE_WEIGHTS["invalid_region_filter"] * max(0.0, 1.0 - ratio * 2.0)

    return round(score, 2), {
        "invalid_region_candidates": bad,
        "invalid_region_count": len(bad),
    }


def score_confidence(pred: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    粗略评价置信度是否合理：
    - 高置信候选应尽量落在参考有缺陷的区域；
    - 低置信候选允许作为疑似。
    """
    candidates = pred.get("defect_candidates", []) or []
    ref_counts = get_region_dict(ref, "region_count")

    if not isinstance(candidates, list) or not candidates:
        return 5.0, {
            "note": "无候选，置信度项给中间分；主要由数量和区域项扣分。"
        }

    penalty = 0.0
    detail = []

    for c in candidates:
        if not isinstance(c, dict):
            continue
        region = normalize_region_name(c.get("region_judgement", ""))
        conf = safe_num(c.get("confidence", 0))
        ref_has_region = ref_counts.get(region, 0) > 0

        if conf >= 0.6 and not ref_has_region:
            penalty += 1.0
            detail.append({
                "type": "high_confidence_false_region",
                "view_index": c.get("view_index"),
                "region": region,
                "confidence": conf,
            })
        elif conf < 0.6 and ref_has_region:
            # 疑似候选落在正确区域，可以接受，只轻微扣
            penalty += 0.15

    score = max(0.0, SCORE_WEIGHTS["confidence_calibration"] - penalty * 2.0)
    return round(score, 2), {
        "confidence_warnings": detail,
        "penalty": round(penalty, 3),
    }


def build_comment(total: float, details: Dict[str, Any]) -> str:
    comments = []

    missed = details["risk_focus"]["missed_high_impact_regions"]
    false_high = details["risk_focus"]["false_high_impact_regions"]
    total_sim = details["total_count"]["similarity"]

    if total >= 85:
        comments.append("视觉结构化结果整体较好，区域数量和高影响区域判断基本可信。")
    elif total >= 70:
        comments.append("视觉结构化结果基本可用，但存在部分数量误差或区域风险判断偏差。")
    elif total >= 55:
        comments.append("视觉结构化结果可作为参考，但漏检、过检或区域归属问题较明显。")
    else:
        comments.append("视觉结构化结果可靠性不足，不建议直接作为溯源依据。")

    if total_sim < 0.7:
        comments.append("总气孔数量与参考结果偏差较大。")
    if missed:
        comments.append("漏识别高影响区域：" + "、".join(missed) + "。")
    if false_high:
        comments.append("误判高影响风险区域：" + "、".join(false_high) + "。")
    if details["invalid_filter"]["invalid_region_count"] > 0:
        comments.append("存在背景、重叠区或未知区域误计入缺陷的问题。")

    return "".join(comments)


def score_visual(pred: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    s_total, d_total = score_total_count(pred, ref)
    s_region, d_region = score_region_count(pred, ref)
    s_risk, d_risk = score_risk_focus(pred, ref)
    s_metric, d_metric = score_area_diameter(pred, ref)
    s_invalid, d_invalid = score_invalid_region_filter(pred)
    s_conf, d_conf = score_confidence(pred, ref)

    total = round(s_total + s_region + s_risk + s_metric + s_invalid + s_conf, 2)

    details = {
        "total_count": d_total,
        "region_count": d_region,
        "risk_focus": d_risk,
        "area_diameter": d_metric,
        "invalid_filter": d_invalid,
        "confidence": d_conf,
    }

    return {
        "visual_score": total,
        "score_breakdown": {
            "total_defect_count": s_total,
            "region_count": s_region,
            "risk_focus_regions": s_risk,
            "area_and_eq_diameter": s_metric,
            "invalid_region_filter": s_invalid,
            "confidence_calibration": s_conf,
        },
        "details": details,
        "comment": build_comment(total, details),
    }


# ============================================================
# 6. 汇总输出
# ============================================================

def write_score_csv(path: Path, result: Dict[str, Any]) -> None:
    rows = []
    breakdown = result.get("score_breakdown", {}) or {}
    for k, v in breakdown.items():
        rows.append({"item": k, "score": v})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "score"])
        writer.writeheader()
        writer.writerows(rows)


def attach_visual_score_to_all_modes(all_modes: Dict[str, Any], result: Dict[str, Any], pred_source: str) -> Dict[str, Any]:
    obj = deepcopy(all_modes)
    obj["visual_layer_score"] = result
    obj["visual_layer_score_source"] = pred_source

    # 如果有 cases，也给每个 case 加一个引用字段，方便后续综合评分脚本读取
    cases = obj.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                case["visual_layer_score"] = result.get("visual_score")
                case["visual_layer_score_comment"] = result.get("comment", "")

    return obj


def print_banner(title: str) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# 7. 主流程：无命令行参数，直接运行
# ============================================================

def main() -> None:
    input_path = resolve_input_all_modes_path()
    ref_path = Path(REF_VISUAL_JSON).resolve()
    output_dir = resolve_output_dir(input_path)

    print_banner("[run] visual layer scoring from all_modes_results.json")
    print(f"[input] all_modes_results : {input_path}")
    print(f"[input] reference_visual  : {ref_path}")
    print(f"[out]   output_dir        : {output_dir}")

    all_modes = load_json(input_path)
    ref = load_json(ref_path)

    pred_visual, pred_source = extract_pred_visual_from_all_modes(all_modes, input_path)
    result = score_visual(pred_visual, ref)
    result["pred_source"] = pred_source
    result["ref_source"] = str(ref_path)
    result["input_all_modes_results"] = str(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    visual_score_path = output_dir / VISUAL_SCORE_JSON_NAME
    score_csv_path = output_dir / VISUAL_SCORE_CSV_NAME
    pred_extract_path = output_dir / PRED_VISUAL_EXTRACTED_NAME
    ref_used_path = output_dir / REF_VISUAL_USED_NAME
    enriched_path = input_path.parent / ALL_MODES_WITH_VISUAL_SCORE_NAME

    save_json(visual_score_path, result)
    write_score_csv(score_csv_path, result)
    save_json(pred_extract_path, pred_visual)
    save_json(ref_used_path, ref)

    if WRITE_BACK_ALL_MODES:
        enriched = attach_visual_score_to_all_modes(all_modes, result, pred_source)
        save_json(enriched_path, enriched)

    print_banner("[done] visual layer score")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("-" * 100)
    print(f"[done] visual_score_json : {visual_score_path}")
    print(f"[done] score_csv         : {score_csv_path}")
    print(f"[done] pred_visual_json  : {pred_extract_path}")
    print(f"[done] ref_visual_json   : {ref_used_path}")
    if WRITE_BACK_ALL_MODES:
        print(f"[done] enriched_results  : {enriched_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
