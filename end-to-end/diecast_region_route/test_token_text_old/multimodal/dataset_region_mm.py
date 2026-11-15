from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset

from .config_region_mm import RegionAwareMMConfig
from .vision_backbone import preprocess_image


STRICT_FORMAT_REQUIREMENT = """【格式要求】
1. 输出必须且只能包含【结论】【溯源分析】【建议方向】三部分。
2. 必须直接从【结论】开始，不要输出前言、解释性正文、思考过程或额外标题。
3. 【结论】只写1到2句话，直接给判断。
4. 【溯源分析】必须写3点，编号为1. 2. 3.，重点围绕“区域统计 -> 参数链 -> 机理判断/优先级”展开。
5. 【建议方向】必须写3点，编号为1. 2. 3.，只给宏观排查方向和验证方向，不给具体参数设定值。
6. 如果输入提供了区域缺陷统计、区域最大面积或区域最大等效直径，必须明确使用并严格对应原区域。
7. 如果问题前提与当前样本统计不一致，必须先按当前样本统计纠偏后再分析。
8. 只能依据当前输入分析，不要引入未提供字段，不要写“当前未提供某信息”。
""".strip()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _parse_params_from_info_text(info_text: str, numeric_fields: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not info_text:
        return out

    lines = [x.strip() for x in info_text.splitlines() if x.strip()]
    kv_pattern = re.compile(r"^([^=]+)=(.+)$")

    for line in lines:
        m = kv_pattern.match(line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k in numeric_fields:
            out[k] = _safe_float(v, 0.0)

    return out


def _normalize_params(params: Dict[str, Any], info_text: str, numeric_fields: List[str]) -> Dict[str, float]:
    merged: Dict[str, float] = {}

    if isinstance(params, dict):
        for k in numeric_fields:
            if k in params:
                merged[k] = _safe_float(params.get(k), 0.0)

    recovered = _parse_params_from_info_text(info_text, numeric_fields)
    for k in numeric_fields:
        if k not in merged:
            merged[k] = recovered.get(k, 0.0)

    return merged


def _build_empty_boxes_by_view(image_paths: List[str]) -> List[List[List[float]]]:
    return [[] for _ in image_paths]


def _build_empty_defect_meta_by_view(image_paths: List[str]) -> List[List[Dict[str, Any]]]:
    return [[] for _ in image_paths]


def _normalize_answer_text(rec: Dict[str, Any]) -> str:
    """
    v4 数据集默认使用 answer 字段。
    为兼容后续可能的字段命名，也允许 answer_v4_format / final_answer。
    """
    for key in ["answer", "answer_v4_format", "final_answer"]:
        val = str(rec.get(key, "")).strip()
        if val:
            return val
    return ""


def _normalize_question_type(rec: Dict[str, Any]) -> str:
    question_type = str(rec.get("question_type", "open_diagnosis")).strip() or "open_diagnosis"
    return question_type


def _build_question_type_hint(rec: Dict[str, Any], cfg: RegionAwareMMConfig) -> str:
    question_type = _normalize_question_type(rec)
    hint = cfg.question_type_hints.get(question_type, cfg.question_type_hints.get("open_diagnosis", ""))

    lines = [f"【问题类型】{question_type}"]
    if hint:
        lines.append(f"【类型提示】{hint}")

    focus_regions = rec.get("focus_regions", [])
    if isinstance(focus_regions, list) and focus_regions:
        clean = [str(x).strip() for x in focus_regions if str(x).strip()]
        if clean:
            lines.append("【关注区域】" + "；".join(clean))

    mechanism_candidates = rec.get("mechanism_candidates", [])
    if isinstance(mechanism_candidates, list) and mechanism_candidates:
        clean = [str(x).strip() for x in mechanism_candidates if str(x).strip()]
        if clean:
            lines.append("【候选机理】" + "；".join(clean))

    # v4 数据集会包含这些布尔字段；没有也不影响
    extra_requirements = []
    if rec.get("requires_conflict_check", False):
        extra_requirements.append("如果问题前提与当前样本统计不一致，必须先纠偏后再分析。")
    if rec.get("requires_binary_decision", False):
        extra_requirements.append("如果问题存在两个备选项，必须明确更偏向哪一项，并说明另一项为何不是当前优先判断。")
    if rec.get("requires_priority_judgement", False):
        extra_requirements.append("如果问题强调优先级，必须明确第一优先和第二优先的排查或判断顺序。")
    if rec.get("requires_causal_explanation", False):
        extra_requirements.append("回答时需要说明主因链，不要只给结论。")
    if rec.get("requires_validation", False):
        extra_requirements.append("建议方向里要体现如何验证判断是否成立。")
    if rec.get("requires_adjustment_advice", False):
        extra_requirements.append("建议方向里要体现如何调整或优化，但不要给具体数值。")

    if extra_requirements:
        lines.append("【补充要求】" + " ".join(extra_requirements))

    lines.append(STRICT_FORMAT_REQUIREMENT)

    return "\n".join(lines).strip()


def _build_base_instruction(rec: Dict[str, Any]) -> str:
    raw = str(rec.get("instruction", "")).strip()
    if raw:
        return raw
    return (
        "你是一名铝合金压铸缺陷诊断助手。"
        "请严格按照【结论】【溯源分析】【建议方向】三段式回答，"
        "其中【溯源分析】必须3点，【建议方向】必须3点。"
    )


class RegionMMJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path | str, cfg: RegionAwareMMConfig):
        self.path = Path(jsonl_path)
        self.cfg = cfg
        self.records = _load_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        workpiece_id = str(rec.get("workpiece_id", ""))
        sample_id = rec.get("sample_id", rec.get("sample_no", ""))

        base_instruction = _build_base_instruction(rec)
        question = str(rec.get("question", "")).strip()
        answer = _normalize_answer_text(rec)
        info_text = str(rec.get("info_text", "")).strip()

        summary = rec.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        params = rec.get("params", {})
        if not isinstance(params, dict):
            params = {}

        params = _normalize_params(params, info_text, self.cfg.numeric_fields)

        image_paths = rec.get("image_paths", [])
        if not isinstance(image_paths, list):
            image_paths = []

        view_items = rec.get("view_items", [])
        if not isinstance(view_items, list):
            view_items = []

        question_type = _normalize_question_type(rec)
        question_type_hint = _build_question_type_hint(rec, self.cfg)

        instruction = base_instruction
        if self.cfg.use_question_type_hint and self.cfg.inject_question_type_to_instruction and question_type_hint:
            instruction = base_instruction.rstrip() + "\n" + question_type_hint

        if self.cfg.use_question_type_hint and self.cfg.inject_question_type_to_info_text and question_type_hint:
            info_text = (info_text.rstrip() + "\n" + question_type_hint).strip()

        boxes_by_view = rec.get("boxes_by_view", None)
        if not isinstance(boxes_by_view, list):
            boxes_by_view = _build_empty_boxes_by_view(image_paths)

        defect_meta_by_view = rec.get("defect_meta_by_view", None)
        if not isinstance(defect_meta_by_view, list):
            defect_meta_by_view = _build_empty_defect_meta_by_view(image_paths)

        images = []
        valid_image_paths = []

        capped_paths = image_paths[: self.cfg.max_views_per_sample]
        for p in capped_paths:
            try:
                img = preprocess_image(p, self.cfg.vision_image_size)
                images.append(img)
                valid_image_paths.append(p)
            except Exception as exc:
                print(f"[WARN] 读取图像失败，跳过: {p} | {exc}")

        if len(images) == 0:
            raise RuntimeError(f"样本 {workpiece_id} 没有可用图像。")

        n = len(images)
        boxes_by_view = boxes_by_view[:n]
        defect_meta_by_view = defect_meta_by_view[:n]
        view_items = view_items[:n] if view_items else [{"view_index": i + 1} for i in range(n)]

        return {
            "workpiece_id": workpiece_id,
            "sample_id": sample_id,
            "instruction": instruction,
            "question": question,
            "answer": answer,
            "summary": summary,
            "params": params,
            "info_text": info_text,
            "image_paths": valid_image_paths,
            "images": images,
            "boxes_by_view": boxes_by_view,
            "defect_meta_by_view": defect_meta_by_view,
            "view_items": view_items,
            "question_type": question_type,
            "question_type_hint": question_type_hint,
            "mechanism_candidates": rec.get("mechanism_candidates", []),
            "focus_regions": rec.get("focus_regions", []),
            "format_schema": rec.get("format_schema", {}),
            "answer_raw_before_v4_format": rec.get("answer_raw_before_v4_format", rec.get("answer_raw", "")),
        }


def collate_region_mm(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "workpiece_id": [x["workpiece_id"] for x in batch],
        "sample_id": [x["sample_id"] for x in batch],
        "instruction": [x["instruction"] for x in batch],
        "question": [x["question"] for x in batch],
        "answer": [x["answer"] for x in batch],
        "summary": [x["summary"] for x in batch],
        "params": [x["params"] for x in batch],
        "info_text": [x["info_text"] for x in batch],
        "image_paths": [x["image_paths"] for x in batch],
        "images": [x["images"] for x in batch],
        "boxes_by_view": [x["boxes_by_view"] for x in batch],
        "defect_meta_by_view": [x["defect_meta_by_view"] for x in batch],
        "view_items": [x["view_items"] for x in batch],
        "question_type": [x["question_type"] for x in batch],
        "question_type_hint": [x["question_type_hint"] for x in batch],
        "mechanism_candidates": [x["mechanism_candidates"] for x in batch],
        "focus_regions": [x["focus_regions"] for x in batch],
        "format_schema": [x["format_schema"] for x in batch],
        "answer_raw_before_v4_format": [x["answer_raw_before_v4_format"] for x in batch],
    }


# =========================================================
# 向后兼容旧版推理脚本
# =========================================================
RegionAwareJSONLDataset = RegionMMJsonlDataset
