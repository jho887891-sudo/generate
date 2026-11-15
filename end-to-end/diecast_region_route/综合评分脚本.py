#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量综合评分脚本：五个尺寸 all_modes_results.json -> 溯源自动80分 + 视觉100分 + 汇总表

适用你的当前目录结构：
/home/T7/wkq/end-to-end/compare_outputs/0.6b-14b_300q/
├── e2e_0.6b_sample_38073_300q_token_text_graphrag/all_modes_results.json
├── e2e_1.7b_sample_38073_300q_token_text_graphrag/all_modes_results.json
├── e2e_4b_sample_38073_300q_token_text_graphrag/all_modes_results.json
├── e2e_8b_sample_38073_300q_token_text_graphrag/all_modes_results.json
└── e2e_14b_sample_38073_300q_token_text_graphrag/all_modes_results.json

重要修改：
1. 不再使用 rglob 递归扫描结果目录；
2. 只检查每个模型目录一级下的 all_modes_results.json；
3. 避免误扫 combined_auto_visual_score、visual_score、历史评分输出目录；
4. 全流程带 flush=True 进度打印，避免看起来像“卡住”；
5. 默认复用已有评分结果，不重复计算；如果要强制重算，把 RECOMPUTE_TRACE_SCORE / RECOMPUTE_VISUAL_SCORE 改成 True。

输出：
每个模型结果目录下：
combined_auto_visual_score/
├── trace_auto80/
├── visual100/
├── combined_score_summary.json
├── combined_score_summary.csv
├── manual_review_items_for_this_result.json
└── manual_score_template_for_this_result.json

总汇总目录：
/home/T7/wkq/end-to-end/compare_outputs/0.6b-14b_300q/combined_5size_scores/
├── latest_master_score_summary.json
├── latest_master_score_summary.csv
├── latest_master_score_report.md
├── latest_master_manual_review_pack_for_gpt.md
├── latest_master_manual_review_pack_for_gpt.json
└── latest_master_manual_score_template.json

端到端公式：
溯源平均分 = 自动平均分 + 专家平均分
端到端综合分 = 视觉层分 × 0.3 + 溯源平均分 × 0.7

未补专家分之前：
自动参考综合分 = 视觉层分 × 0.3 + (自动平均分 / 80 × 100) × 0.7
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# 1. 路径配置：按服务器固定写死
# ============================================================

PROJECT_ROOT = Path("/home/T7/wkq/end-to-end")
COMPARE_OUTPUTS_ROOT = PROJECT_ROOT / "compare_outputs"

TRACE_SCORER_CANDIDATES = [
    PROJECT_ROOT / "diecast_region_route" / "score_端到端溯源结果.py",
    PROJECT_ROOT / "API_test" / "score_端到端溯源结果.py",
    PROJECT_ROOT / "score_端到端溯源结果.py",
]

VISUAL_SCORER_CANDIDATES = [
    PROJECT_ROOT / "diecast_region_route" / "score_视觉检测评分.py",
    PROJECT_ROOT / "API_test" / "score_视觉检测评分.py",
    PROJECT_ROOT / "score_视觉检测评分.py",
]

# 这里必须是当前五个模型对应样本的参考视觉统计。
# 你已经改成 38073，就保持这个路径。
REF_VISUAL_JSON = PROJECT_ROOT / "API_test" / "ref_visual_summary_38073.json"

# 五个模型结果目录：脚本只检查这些目录一级下的 all_modes_results.json。
MODEL_RESULT_ROOTS = [
    COMPARE_OUTPUTS_ROOT / "e2e_4b_base_only_sample_38073_300q_svp_text_graphrag",
]

# 一般不需要兜底扫描。你现在目录固定，建议 False。
ENABLE_FALLBACK_SCAN = False

# 如果将来目录名变了，可以改成 True，只做一级兜底，不递归。
FALLBACK_SCAN_ROOTS = [COMPARE_OUTPUTS_ROOT]
FALLBACK_REQUIRED_KEYWORDS = ["sample_38073", "300q_token_text"]

EXCLUDE_PATH_KEYWORDS = [
    "_score",
    "visual_score",
    "combined_auto_visual_score",
    "combined_5size_scores",
    "qwen35_plus",
    "old",
    "测试版",
    "语料+图谱300问结果",
    "图谱300问测试",
]

PER_RESULT_SCORE_DIR_NAME = "combined_auto_visual_score"
MASTER_OUTPUT_ROOT = COMPARE_OUTPUTS_ROOT / "base_scores_text"
MASTER_MANUAL_JSON = MASTER_OUTPUT_ROOT / "master_manual_scores_filled.json"

REVIEW_PACK_LIMIT_PER_RESULT = 50

VISUAL_WEIGHT = 0.30
TRACE_WEIGHT = 0.70

# 默认复用已有分数，避免每次都重新跑 5×300。
# 如果你修改了评分规则，想全部重算，改成 True。
RECOMPUTE_TRACE_SCORE = False
RECOMPUTE_VISUAL_SCORE = False


# ============================================================
# 2. 基础工具函数
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def sanitize_name(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__").replace(":", "_")
    text = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180] or "result"


def resolve_existing_path(candidates: List[Path], name: str) -> Path:
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"{name} 不存在，已检查：\n" + "\n".join(str(p) for p in candidates)
    )


def import_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入：{py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_model_label(input_json: Path) -> str:
    s = str(input_json)
    m = re.search(r"e2e_([0-9.]+b)", s, flags=re.I)
    if m:
        return m.group(1).upper()
    return input_json.parent.name


def result_id_from_input(input_json: Path) -> str:
    try:
        rel = input_json.parent.relative_to(COMPARE_OUTPUTS_ROOT)
        return sanitize_name(str(rel))
    except Exception:
        return sanitize_name(input_json.parent.name)


def excluded(path: Path) -> bool:
    s = str(path)
    return any(k in s for k in EXCLUDE_PATH_KEYWORDS)


# ============================================================
# 3. 只做一级目录检查，不递归扫描
# ============================================================

def discover_all_modes() -> List[Path]:
    """
    只扫描五个模型结果目录下的一级 all_modes_results.json。
    不递归进入子目录，避免扫描 combined_auto_visual_score、visual_score 等评分目录。
    """
    found: List[Path] = []

    log("=" * 100)
    log("[scan] 开始检查显式五个模型目录")
    log(f"[scan] root = {COMPARE_OUTPUTS_ROOT}")
    log("=" * 100)

    for root in MODEL_RESULT_ROOTS:
        candidate = root / "all_modes_results.json"
        log(f"[scan] check: {candidate}")

        if candidate.exists() and candidate.is_file():
            if excluded(candidate):
                log(f"[scan] skip excluded: {candidate}")
                continue
            found.append(candidate.resolve())
            log(f"[scan] found: {candidate}")
        else:
            log(f"[scan] missing: {candidate}")

    # 如果显式五个目录都没找到，才做一级兜底扫描。
    # 注意：这里也不是递归，只扫描 compare_outputs/0.6b-14b_300q 下一级子目录。
    if not found and ENABLE_FALLBACK_SCAN:
        log("=" * 100)
        log("[scan] 显式目录未找到结果，开始一级兜底扫描")
        log("=" * 100)

        for root in FALLBACK_SCAN_ROOTS:
            if not root.exists():
                log(f"[scan] fallback root missing: {root}")
                continue

            for subdir in root.iterdir():
                if not subdir.is_dir():
                    continue

                candidate = subdir / "all_modes_results.json"
                s = str(candidate)

                if not candidate.exists() or not candidate.is_file():
                    continue
                if excluded(candidate):
                    continue
                if all(k in s for k in FALLBACK_REQUIRED_KEYWORDS):
                    found.append(candidate.resolve())
                    log(f"[scan] fallback found: {candidate}")

    uniq: Dict[str, Path] = {}
    for p in found:
        uniq[str(p)] = p

    results = sorted(uniq.values(), key=lambda x: str(x))

    log("=" * 100)
    log(f"[scan] 最终找到 {len(results)} 个 all_modes_results.json")
    for p in results:
        log(f"  - {p}")
    log("=" * 100)

    return results


# ============================================================
# 4. 专家评分读取
# ============================================================

def manual_score_from_item(item: Any) -> Tuple[float, str]:
    if not isinstance(item, dict):
        return 0.0, ""

    if "score" in item:
        score = max(0.0, min(20.0, safe_float(item.get("score", 0))))
        return score, str(item.get("comment", ""))

    total = 0.0
    for key, max_score in [
        ("expert_reasonableness", 5),
        ("sample_consistency", 5),
        ("mechanism_depth", 4),
        ("verification_value", 3),
        ("report_usability", 3),
    ]:
        total += max(0.0, min(float(max_score), safe_float(item.get(key, 0))))

    return max(0.0, min(20.0, total)), str(item.get("comment", ""))


def load_master_manual() -> Dict[str, Any]:
    if not MASTER_MANUAL_JSON.exists():
        return {}

    obj = load_json(MASTER_MANUAL_JSON)
    if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
        return obj["results"]
    return obj if isinstance(obj, dict) else {}


def get_manual(master: Dict[str, Any], result_id: str, qid: str, mode: str) -> Tuple[float, str]:
    item = master.get(result_id, {}).get(qid, {}).get(mode, {})
    return manual_score_from_item(item)


# ============================================================
# 5. 单文件评分：溯源80分 + 视觉100分
# ============================================================

def run_trace80(trace_mod, input_json: Path, out_dir: Path) -> Dict[str, Any]:
    result_file = out_dir / "score_results.json"

    if result_file.exists() and not RECOMPUTE_TRACE_SCORE:
        log(f"[trace80] reuse: {result_file}")
        return load_json(result_file)

    log(f"[trace80] start scoring: {input_json}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 这里调用你已有的端到端溯源评分脚本函数。
    trace_mod.run_scoring(
        input_path=input_json,
        output_dir=out_dir,
        manual_json=None,
        review_pack_limit=max(1, int(REVIEW_PACK_LIMIT_PER_RESULT)),
    )

    log(f"[trace80] done: {result_file}")
    return load_json(result_file)


def run_visual100(visual_mod, input_json: Path, out_dir: Path) -> Dict[str, Any]:
    result_file = out_dir / "visual_layer_score.json"

    if result_file.exists() and not RECOMPUTE_VISUAL_SCORE:
        log(f"[visual100] reuse: {result_file}")
        return load_json(result_file)

    log(f"[visual100] start scoring: {input_json}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_modes = visual_mod.load_json(input_json)
    ref = visual_mod.load_json(REF_VISUAL_JSON)

    pred_visual, pred_source = visual_mod.extract_pred_visual_from_all_modes(all_modes, input_json)
    result = visual_mod.score_visual(pred_visual, ref)
    result["pred_source"] = pred_source
    result["ref_source"] = str(REF_VISUAL_JSON)
    result["input_all_modes_results"] = str(input_json)

    visual_mod.save_json(out_dir / "visual_layer_score.json", result)
    visual_mod.write_score_csv(out_dir / "visual_layer_score_breakdown.csv", result)
    visual_mod.save_json(out_dir / "pred_visual_extracted_from_all_modes.json", pred_visual)
    visual_mod.save_json(out_dir / "ref_visual_used.json", ref)

    if hasattr(visual_mod, "attach_visual_score_to_all_modes"):
        enriched = visual_mod.attach_visual_score_to_all_modes(all_modes, result, pred_source)
        visual_mod.save_json(input_json.parent / "all_modes_results_with_visual_score.json", enriched)

    log(f"[visual100] done: {result_file}")
    return result


def summarize_trace(
    trace_result: Dict[str, Any],
    master_manual: Dict[str, Any],
    result_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    details = trace_result.get("details", []) or []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []

    for d in details:
        qid = str(d.get("question_id", ""))
        mode = str(d.get("mode", ""))

        manual_score, manual_comment = get_manual(master_manual, result_id, qid, mode)
        auto = safe_float(d.get("auto_score_80", 0))

        row = {
            "result_id": result_id,
            "question_id": qid,
            "question_index": d.get("question_index", ""),
            "sample_id": d.get("sample_id", ""),
            "mode": mode,
            "mode_desc": d.get("mode_desc", ""),
            "auto_score_80": round(auto, 3),
            "manual_score_20": round(manual_score, 3),
            "trace_score_100": round(auto + manual_score, 3),
            "manual_comment": manual_comment,
            "question": d.get("question", ""),
            "answer": d.get("answer", ""),
            "breakdown": d.get("breakdown", []),
        }

        rows.append(row)
        grouped[mode].append(row)

    summary: List[Dict[str, Any]] = []

    for mode, rs in grouped.items():
        autos = [safe_float(r["auto_score_80"]) for r in rs]
        manuals = [safe_float(r["manual_score_20"]) for r in rs]
        traces = [safe_float(r["trace_score_100"]) for r in rs]

        manual_filled = sum(1 for v in manuals if v > 0)
        status = "filled" if manual_filled > 0 else "pending"

        auto_avg = round(sum(autos) / max(1, len(autos)), 3)
        manual_avg = round(sum(manuals) / max(1, len(manuals)), 3)
        trace_avg = round(sum(traces) / max(1, len(traces)), 3)

        summary.append({
            "mode": mode,
            "count": len(rs),
            "auto_avg_80": auto_avg,
            "auto_normalized_100_reference": round(auto_avg / 80.0 * 100.0, 3),
            "manual_filled_count": manual_filled,
            "manual_status": status,
            "manual_avg_20": manual_avg,
            "trace_avg_100": trace_avg if status == "filled" else "",
        })

    return summary, rows


def integrate_scores(mode_summary: List[Dict[str, Any]], visual_score: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for r in mode_summary:
        auto_norm = safe_float(r.get("auto_normalized_100_reference", 0))
        auto_proxy_e2e = round(visual_score * VISUAL_WEIGHT + auto_norm * TRACE_WEIGHT, 3)

        if r.get("manual_status") == "filled":
            trace_avg = safe_float(r.get("trace_avg_100", 0))
            final_e2e = round(visual_score * VISUAL_WEIGHT + trace_avg * TRACE_WEIGHT, 3)
        else:
            final_e2e = ""

        out.append({
            **r,
            "visual_score_100": round(visual_score, 3),
            "visual_weight": VISUAL_WEIGHT,
            "trace_weight": TRACE_WEIGHT,
            "auto_proxy_e2e_score_reference": auto_proxy_e2e,
            "e2e_final_score_100": final_e2e,
            "score_formula": "e2e_final = visual_score*0.3 + trace_avg_100*0.7",
        })

    return out


def low_score_review_items(
    result_id: str,
    model_label: str,
    detail_rows: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    selected = sorted(detail_rows, key=lambda x: safe_float(x.get("auto_score_80", 0)))[:limit]
    items: List[Dict[str, Any]] = []

    for r in selected:
        notes: List[str] = []

        for b in r.get("breakdown", []) or []:
            if not isinstance(b, dict):
                continue
            ns = b.get("notes", []) or []
            if ns:
                notes.append(
                    f"{b.get('zh_name', b.get('name', ''))}: "
                    f"{b.get('score')}/{b.get('max_score')}，"
                    + "；".join(map(str, ns))
                )

        items.append({
            "result_id": result_id,
            "model_label": model_label,
            "question_id": r.get("question_id", ""),
            "mode": r.get("mode", ""),
            "auto_score_80": r.get("auto_score_80", 0),
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "deduction_notes": notes,
        })

    return items


def build_manual_template_for_result(
    result_id: str,
    model_label: str,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tpl: Dict[str, Any] = {
        "_result_id": result_id,
        "_model_label": model_label,
        "_manual_total": 20,
        "_schema": {
            "expert_reasonableness": "0-5 工程合理性",
            "sample_consistency": "0-5 样本一致性",
            "mechanism_depth": "0-4 机理深度",
            "verification_value": "0-3 验证价值",
            "report_usability": "0-3 报告可用性",
            "score": "0-20 总分；不填score时脚本累加五项",
            "comment": "评分理由",
        },
    }

    for it in items:
        tpl.setdefault(it["question_id"], {})[it["mode"]] = {
            "expert_reasonableness": 0,
            "sample_consistency": 0,
            "mechanism_depth": 0,
            "verification_value": 0,
            "report_usability": 0,
            "score": 0,
            "comment": "",
        }

    return tpl


def score_one(trace_mod, visual_mod, input_json: Path, master_manual: Dict[str, Any]) -> Dict[str, Any]:
    result_id = result_id_from_input(input_json)
    model_label = infer_model_label(input_json)

    out_root = input_json.parent / PER_RESULT_SCORE_DIR_NAME
    trace_out = out_root / "trace_auto80"
    visual_out = out_root / "visual100"

    log("\n" + "=" * 100)
    log(f"[score] {model_label} | {result_id}")
    log(f"[input] {input_json}")
    log(f"[out]   {out_root}")
    log("=" * 100)

    trace_result = run_trace80(trace_mod, input_json, trace_out)
    visual_result = run_visual100(visual_mod, input_json, visual_out)
    visual_score = safe_float(visual_result.get("visual_score", 0))

    mode_summary, detail_rows = summarize_trace(trace_result, master_manual, result_id)
    integrated = integrate_scores(mode_summary, visual_score)
    review_items = low_score_review_items(
        result_id=result_id,
        model_label=model_label,
        detail_rows=detail_rows,
        limit=REVIEW_PACK_LIMIT_PER_RESULT,
    )

    per_summary = {
        "result_id": result_id,
        "model_label": model_label,
        "input_all_modes_results": str(input_json),
        "output_dir": str(out_root),
        "trace_score_dir": str(trace_out),
        "visual_score_dir": str(visual_out),
        "reference_visual_json": str(REF_VISUAL_JSON),
        "visual_score": visual_score,
        "visual_comment": visual_result.get("comment", ""),
        "trace_mode_summary": mode_summary,
        "integrated_by_mode": integrated,
        "manual_scores_source": str(MASTER_MANUAL_JSON) if MASTER_MANUAL_JSON.exists() else "",
        "manual_status": "filled" if any(x.get("manual_status") == "filled" for x in integrated) else "pending",
        "score_formula": {
            "trace_avg_100": "auto_avg_80 + manual_avg_20",
            "e2e_final_score_100": "visual_score*0.3 + trace_avg_100*0.7",
            "auto_proxy_e2e_score_reference": "visual_score*0.3 + (auto_avg_80/80*100)*0.7，仅作未补专家分时的自动参考",
        },
    }

    save_json(out_root / "combined_score_summary.json", per_summary)
    save_json(out_root / "trace_details_with_manual_placeholder.json", detail_rows)
    save_json(out_root / "manual_review_items_for_this_result.json", review_items)
    save_json(
        out_root / "manual_score_template_for_this_result.json",
        build_manual_template_for_result(result_id, model_label, review_items),
    )

    write_csv(
        out_root / "combined_score_summary.csv",
        integrated,
        [
            "mode",
            "count",
            "visual_score_100",
            "auto_avg_80",
            "auto_normalized_100_reference",
            "manual_filled_count",
            "manual_status",
            "manual_avg_20",
            "trace_avg_100",
            "auto_proxy_e2e_score_reference",
            "e2e_final_score_100",
            "visual_weight",
            "trace_weight",
            "score_formula",
        ],
    )

    per_summary["_review_items"] = review_items
    return per_summary


# ============================================================
# 6. 总汇总输出
# ============================================================

def build_master_review_pack_md(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# 五尺寸多模态模型专家20分审阅包\n\n")
    lines.append("请按工程合理性5、样本一致性5、机理深度4、验证价值3、报告可用性3补专家20分。\n")
    lines.append("最终返回 JSON，建议保存为：\n\n")
    lines.append(f"`{MASTER_MANUAL_JSON}`\n\n")
    lines.append("返回结构示例：\n\n```json\n")
    lines.append(json.dumps({
        "results": {
            "result_id示例": {
                "q00001": {
                    "token_text": {
                        "expert_reasonableness": 0,
                        "sample_consistency": 0,
                        "mechanism_depth": 0,
                        "verification_value": 0,
                        "report_usability": 0,
                        "score": 0,
                        "comment": "评分理由"
                    }
                }
            }
        }
    }, ensure_ascii=False, indent=2))
    lines.append("\n```\n\n")

    for i, item in enumerate(items, 1):
        lines.append(
            f"## {i}. {item['model_label']} | {item['result_id']} | "
            f"{item['question_id']} | {item['mode']}\n\n"
        )
        lines.append(f"- 自动分：{item['auto_score_80']}/80\n")
        lines.append(f"- 问题：{item.get('question', '')}\n\n")

        if item.get("deduction_notes"):
            lines.append("自动扣分提示：\n")
            for note in item["deduction_notes"]:
                lines.append(f"- {note}\n")
            lines.append("\n")

        lines.append("模型回答：\n\n```text\n")
        lines.append(str(item.get("answer", "")).strip())
        lines.append("\n```\n\n请补专家20分。\n\n")

    return "".join(lines)


def build_report_md(summary: Dict[str, Any]) -> str:
    rows = summary.get("rows", []) or []
    lines: List[str] = []

    lines.append("# 五尺寸多模态模型综合评分汇总\n\n")
    lines.append(f"- 生成时间：{summary.get('created_at')}\n")
    lines.append(f"- 视觉参考文件：`{summary.get('ref_visual_json')}`\n")
    lines.append(f"- 专家评分文件：`{summary.get('manual_scores_json')}`\n")
    lines.append(
        f"- 专家评分状态："
        f"{'已读取' if summary.get('manual_scores_exists') else '未填写，当前只包含自动80分和视觉100分'}\n\n"
    )

    lines.append("## 评分公式\n\n")
    lines.append("```text\n")
    lines.append("溯源平均分 = 自动平均分 + 专家平均分\n")
    lines.append("端到端综合分 = 视觉层分 × 0.3 + 溯源平均分 × 0.7\n")
    lines.append("未补专家分前：自动参考综合分 = 视觉层分 × 0.3 + (自动平均分/80×100) × 0.7，仅作参考。\n")
    lines.append("```\n\n")

    lines.append("## 汇总表\n\n")
    lines.append(
        "| 模型 | 模式 | 数量 | 视觉层分/100 | 自动均分/80 | 自动归一参考/100 | "
        "专家状态 | 专家均分/20 | 溯源均分/100 | 自动参考综合分 | 最终端到端综合分 |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")

    for r in rows:
        lines.append(
            f"| {r.get('model_label','')} | {r.get('mode','')} | {r.get('count','')} | "
            f"{r.get('visual_score_100','')} | {r.get('auto_avg_80','')} | "
            f"{r.get('auto_normalized_100_reference','')} | {r.get('manual_status','')} | "
            f"{r.get('manual_avg_20','')} | {r.get('trace_avg_100','')} | "
            f"{r.get('auto_proxy_e2e_score_reference','')} | {r.get('e2e_final_score_100','')} |\n"
        )

    lines.append("\n## 解读\n\n")
    lines.append("1. 视觉层分衡量模型看图和区域结构化是否准确。\n")
    lines.append("2. 自动80分衡量溯源回答的格式、区域证据、参数链、建议可验证性和幻觉控制。\n")
    lines.append("3. 专家20分补齐后，最终端到端综合分才是正式用于论文/汇报的分数。\n")
    lines.append("4. 若某模型溯源分高但视觉分低，说明它文字分析可以，但工业X光缺陷检测/区域统计不稳定。\n")

    return "".join(lines)


def model_sort_key(row: Dict[str, Any]):
    m = re.search(r"([0-9.]+)", str(row.get("model_label", "")))
    return (
        safe_float(m.group(1), 999.0) if m else 999.0,
        str(row.get("model_label", "")),
    )


# ============================================================
# 7. 主流程
# ============================================================

def main() -> None:
    t0 = time.time()

    log("=" * 100)
    log("[start] 多尺寸多模态模型综合评分脚本启动")
    log(f"[project] {PROJECT_ROOT}")
    log(f"[compare_root] {COMPARE_OUTPUTS_ROOT}")
    log(f"[ref_visual] {REF_VISUAL_JSON}")
    log(f"[recompute_trace] {RECOMPUTE_TRACE_SCORE}")
    log(f"[recompute_visual] {RECOMPUTE_VISUAL_SCORE}")
    log("=" * 100)

    trace_path = resolve_existing_path(TRACE_SCORER_CANDIDATES, "溯源评分脚本")
    visual_path = resolve_existing_path(VISUAL_SCORER_CANDIDATES, "视觉评分脚本")

    if not REF_VISUAL_JSON.exists():
        raise FileNotFoundError(f"参考视觉结果不存在：{REF_VISUAL_JSON}")

    log(f"[trace_scorer]  {trace_path}")
    log(f"[visual_scorer] {visual_path}")

    log("[import] 正在导入评分脚本...")
    trace_mod = import_module(trace_path, "trace_scorer_mod")
    visual_mod = import_module(visual_path, "visual_scorer_mod")
    log("[import] 导入完成")

    inputs = discover_all_modes()
    if not inputs:
        raise FileNotFoundError(
            "没有找到 all_modes_results.json。\n"
            "请检查 MODEL_RESULT_ROOTS 中五个目录是否正确，且 all_modes_results.json 是否在这些目录一级下。"
        )

    run_dir = MASTER_OUTPUT_ROOT / f"run_{now_str()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    master_manual = load_master_manual()

    log("=" * 100)
    log("[run] 多目录 all_modes_results.json 综合评分")
    log(f"[manual_json] {MASTER_MANUAL_JSON if MASTER_MANUAL_JSON.exists() else '未找到，专家分待补'}")
    log(f"[master_out]  {run_dir}")
    log(f"[inputs]      {len(inputs)}")
    for p in inputs:
        log(f"  - {p}")
    log("=" * 100)

    per_results: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []

    for idx, input_json in enumerate(inputs, 1):
        log(f"\n[progress] {idx}/{len(inputs)} 开始处理：{input_json}")

        try:
            res = score_one(trace_mod, visual_mod, input_json, master_manual)
            review_items.extend(res.pop("_review_items", []))
            per_results.append(res)
            log(f"[progress] {idx}/{len(inputs)} 完成：{input_json}")
        except Exception as e:
            log(f"[ERROR] {input_json}: {repr(e)}")
            log(traceback.format_exc())
            per_results.append({
                "result_id": result_id_from_input(input_json),
                "model_label": infer_model_label(input_json),
                "input_all_modes_results": str(input_json),
                "error": repr(e),
            })

    flat_rows: List[Dict[str, Any]] = []

    for res in per_results:
        for row in res.get("integrated_by_mode", []) or []:
            flat_rows.append({
                "model_label": res.get("model_label", ""),
                "result_id": res.get("result_id", ""),
                "input_all_modes_results": res.get("input_all_modes_results", ""),
                "visual_comment": res.get("visual_comment", ""),
                **row,
            })

    flat_rows.sort(key=model_sort_key)

    summary = {
        "created_at": now_str(),
        "project_root": str(PROJECT_ROOT),
        "compare_outputs_root": str(COMPARE_OUTPUTS_ROOT),
        "ref_visual_json": str(REF_VISUAL_JSON),
        "trace_scorer": str(trace_path),
        "visual_scorer": str(visual_path),
        "manual_scores_json": str(MASTER_MANUAL_JSON),
        "manual_scores_exists": MASTER_MANUAL_JSON.exists(),
        "score_formula": {
            "trace_avg_100": "auto_avg_80 + manual_avg_20",
            "e2e_final_score_100": "visual_score*0.3 + trace_avg_100*0.7",
            "auto_proxy_e2e_score_reference": "visual_score*0.3 + (auto_avg_80/80*100)*0.7",
            "visual_weight": VISUAL_WEIGHT,
            "trace_weight": TRACE_WEIGHT,
        },
        "result_count": len(per_results),
        "rows": flat_rows,
        "per_results": per_results,
        "output_run_dir": str(run_dir),
    }

    save_json(run_dir / "master_score_summary.json", summary)
    save_json(run_dir / "per_result_score_summaries.json", per_results)

    write_csv(
        run_dir / "master_score_summary.csv",
        flat_rows,
        [
            "model_label",
            "result_id",
            "mode",
            "count",
            "visual_score_100",
            "auto_avg_80",
            "auto_normalized_100_reference",
            "manual_filled_count",
            "manual_status",
            "manual_avg_20",
            "trace_avg_100",
            "auto_proxy_e2e_score_reference",
            "e2e_final_score_100",
            "visual_comment",
            "input_all_modes_results",
        ],
    )

    review_pack = {
        "_usage": "请补专家20分。补完后保存为 MASTER_OUTPUT_ROOT/master_manual_scores_filled.json，再重新运行评分脚本。",
        "_save_as": str(MASTER_MANUAL_JSON),
        "_schema": {
            "results": {
                "result_id": {
                    "q00001": {
                        "mode": {
                            "expert_reasonableness": 0,
                            "sample_consistency": 0,
                            "mechanism_depth": 0,
                            "verification_value": 0,
                            "report_usability": 0,
                            "score": 0,
                            "comment": ""
                        }
                    }
                }
            }
        },
        "items": review_items,
    }

    save_json(run_dir / "master_manual_review_pack_for_gpt.json", review_pack)
    (run_dir / "master_manual_review_pack_for_gpt.md").write_text(
        build_master_review_pack_md(review_items),
        encoding="utf-8",
    )

    blank_manual = {"results": {}}
    for item in review_items:
        blank_manual["results"].setdefault(item["result_id"], {}).setdefault(item["question_id"], {})[item["mode"]] = {
            "expert_reasonableness": 0,
            "sample_consistency": 0,
            "mechanism_depth": 0,
            "verification_value": 0,
            "report_usability": 0,
            "score": 0,
            "comment": "",
        }

    save_json(run_dir / "master_manual_score_template.json", blank_manual)
    (run_dir / "master_score_report.md").write_text(build_report_md(summary), encoding="utf-8")

    MASTER_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (MASTER_OUTPUT_ROOT / "latest_run_path.txt").write_text(str(run_dir), encoding="utf-8")

    for name in [
        "master_score_summary.json",
        "master_score_summary.csv",
        "master_score_report.md",
        "master_manual_review_pack_for_gpt.json",
        "master_manual_review_pack_for_gpt.md",
        "master_manual_score_template.json",
    ]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, MASTER_OUTPUT_ROOT / ("latest_" + name))

    log("\n" + "=" * 100)
    log("[done] 批量评分完成")
    log("=" * 100)
    log(f"[done] run_dir        : {run_dir}")
    log(f"[done] summary_json   : {run_dir / 'master_score_summary.json'}")
    log(f"[done] summary_csv    : {run_dir / 'master_score_summary.csv'}")
    log(f"[done] report_md      : {run_dir / 'master_score_report.md'}")
    log(f"[done] review_pack_md : {run_dir / 'master_manual_review_pack_for_gpt.md'}")
    log(f"[done] review_pack_js : {run_dir / 'master_manual_review_pack_for_gpt.json'}")
    log(f"[done] latest_path    : {MASTER_OUTPUT_ROOT / 'latest_run_path.txt'}")
    log("-" * 100)
    log("汇总预览：")

    for r in flat_rows:
        log(
            f"{str(r.get('model_label','')).rjust(6)} | {r.get('mode','')} | "
            f"visual={r.get('visual_score_100')} | "
            f"auto80={r.get('auto_avg_80')} | "
            f"manual={r.get('manual_status')}:{r.get('manual_avg_20')} | "
            f"auto_proxy_e2e={r.get('auto_proxy_e2e_score_reference')} | "
            f"final_e2e={r.get('e2e_final_score_100')}"
        )

    log(f"[time] {time.time() - t0:.1f}s")
    log("=" * 100)


if __name__ == "__main__":
    main()
