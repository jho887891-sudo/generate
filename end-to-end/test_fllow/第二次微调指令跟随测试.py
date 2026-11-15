#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-4B 高影响区域气孔任务指令跟随测试脚本 v3

目标：
1. 测试输入不再主动包含“背景/未知区域”等无效区域，避免提示词诱导模型复述；
2. 系统提示不再列举旧缺陷类型名称，避免模型复述旧分类词；
3. ID 抽取只保留数字：47243-OK / 47243-NG -> 47243；
4. 仍然检查模型是否自行输出无效区域、旧缺陷分类、额外标题、精确工艺配方；
5. 支持测试纯 Stage2 文本 LoRA 或多模态训练后的 llm_adapter。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-8B-stage1-book-merged-4bit"
DEFAULT_MODEL_ROOT = "/home/T7/wkq/end-to-end/runs/qwen3_8b_stage2_from_merged_second_sft/best_adapter"
DEFAULT_OUTPUT_ROOT = "/home/T7/wkq/end-to-end/qwen8b_high_impact_gas_pore_eval_runs"
DEFAULT_MAX_SEQ_LENGTH = 4096

GENERATION_CONFIG = {
    "max_new_tokens": 768,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "repetition_penalty": 1.05,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--model-root", type=str, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--lora-override", type=str, default="")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_lora_adapter_dir(root: str, override: str = "") -> Path:
    if override:
        p = Path(override)
        if (p / "adapter_config.json").exists():
            return p
        raise FileNotFoundError(f"--lora-override 不是有效 LoRA 目录，缺少 adapter_config.json: {p}")

    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"--model-root 不存在: {root_p}")

    if (root_p / "adapter_config.json").exists():
        return root_p

    for name in [
        "llm_adapter", "lora_adapter", "adapter", "peft_adapter", "final_adapter",
        "best_adapter", "checkpoint-best", "checkpoint_last", "checkpoint-last", "checkpoint",
    ]:
        p = root_p / name
        if (p / "adapter_config.json").exists():
            return p

    candidates: List[Path] = []
    for cfg in root_p.rglob("adapter_config.json"):
        s = str(cfg)
        if "__pycache__" in s or ".ipynb_checkpoints" in s:
            continue
        candidates.append(cfg.parent)

    if not candidates:
        raise FileNotFoundError(
            f"没有在 {root_p} 下找到 adapter_config.json。"
            "如果该目录只保存 projector / multimodal_heads.pt，而没有 llm_adapter，"
            "则不能用此脚本测试 LLM 指令跟随。"
        )

    def score(p: Path) -> Tuple[int, int]:
        lower = str(p).lower()
        key_score = 0
        for kw in ["llm", "lora", "adapter", "best", "final"]:
            if kw in lower:
                key_score -= 1
        return key_score, len(str(p))

    return sorted(candidates, key=score)[0]


def check_model_dirs(base_model_path: str, model_root: str, lora_path: Path) -> None:
    base = Path(base_model_path)
    if not base.exists():
        raise FileNotFoundError(f"BASE_MODEL_PATH 不存在: {base}")
    if not (base / "config.json").exists():
        raise FileNotFoundError(f"BASE_MODEL_PATH 缺少 config.json: {base}")
    if not Path(model_root).exists():
        raise FileNotFoundError(f"MODEL_ROOT 不存在: {model_root}")
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"LORA_PATH 缺少 adapter_config.json: {lora_path}")

    cfg = read_json(lora_path / "adapter_config.json") or {}
    print("[Check] base model directory OK")
    print(f"[Check] BASE_MODEL_PATH = {base}")
    print(f"[Check] MODEL_ROOT      = {model_root}")
    print(f"[Check] LORA_PATH       = {lora_path}")
    print(f"[Check] adapter base_model_name_or_path = {cfg.get('base_model_name_or_path')}")
    print(f"[Check] adapter r={cfg.get('r')}, lora_alpha={cfg.get('lora_alpha')}, target_modules={cfg.get('target_modules')}")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def starts_with_exact(text: str, prefix: str) -> bool:
    return normalize_text(text).startswith(prefix)


def contains_any(text: str, words: List[str]) -> bool:
    return any(w in text for w in words)


def not_contains_any(text: str, words: List[str]) -> bool:
    return not contains_any(text, words)


def section_order_ok(text: str, sections: List[str]) -> bool:
    pos = []
    for sec in sections:
        i = text.find(sec)
        if i < 0:
            return False
        pos.append(i)
    return pos == sorted(pos)


def section_count_exact(text: str, sections: List[str]) -> bool:
    return all(text.count(sec) == 1 for sec in sections)


def has_exact_three_sections(text: str) -> bool:
    return (
        starts_with_exact(text, "【结论】")
        and section_count_exact(text, ["【结论】", "【溯源分析】", "【建议方向】"])
        and section_order_ok(text, ["【结论】", "【溯源分析】", "【建议方向】"])
    )


def is_json_object(text: str) -> bool:
    try:
        obj = json.loads(normalize_text(text))
        return isinstance(obj, dict)
    except Exception:
        return False


def no_markdown_fence(text: str) -> bool:
    return "```" not in text


def no_think_tag(text: str) -> bool:
    low = text.lower()
    return "<think>" not in low and "</think>" not in low and "思考过程" not in text and "推理过程" not in text


def no_precise_recipe_numbers(text: str) -> bool:
    patterns = [
        r"(调整到|调至|设为|设置为|控制在|提高到|降低到|升至|降至|缩短到|延长到)\s*\d+(?:\.\d+)?\s*(?:m/s|MPa|ms|mm|s|℃|%|左右)?",
        r"\d+(?:\.\d+)?\s*(?:m/s|MPa|ms|mm|s|℃|%)\s*(?:左右|附近|范围内|以内|以上)",
        r"(?:建议|推荐).*?\d+(?:\.\d+)?\s*(?:m/s|MPa|ms|mm|s|℃|%)",
    ]
    return not any(re.search(p, text) for p in patterns)


def only_yes(text: str) -> bool:
    return normalize_text(text) == "YES"


def only_choice_b(text: str) -> bool:
    return normalize_text(text) == "B"


def only_numeric_ids(text: str) -> bool:
    return normalize_text(text) == "38053,46269,47243"


def no_extra_sections(text: str, allowed_sections: List[str]) -> bool:
    headers = re.findall(r"【[^】]+】", text)
    return all(h in allowed_sections for h in headers)


def no_markdown_heading_sections(text: str) -> bool:
    return not re.search(r"^#+\s*【", text, flags=re.MULTILINE)


def no_old_quantity_dominant_logic(text: str) -> bool:
    return "数量主导区域" not in text


def has_region_metrics(text: str) -> bool:
    return (
        ("区域缺陷统计" in text or "区域统计" in text)
        and ("区域最大面积" in text or "最大面积" in text)
        and ("区域最大等效直径" in text or "最大等效直径" in text)
    )


def has_high_impact_logic(text: str) -> bool:
    has_region = any(r in text for r in ["牙孔区域", "焊接区域", "重点区域"])
    has_logic = any(w in text for w in [
        "高影响区域", "高影响", "加工敏感", "功能敏感",
        "重点关注", "优先关注", "风险关注", "敏感区域",
    ])
    return has_region and has_logic


def no_invalid_region_words(text: str) -> bool:
    return not_contains_any(text, ["背景", "未知区域", "未知(0)", "未知区域(0)", "背景/未知"])


def no_non_pore_defect_classification(text: str) -> bool:
    forbidden = [
        "缩孔", "缩松", "冷隔", "夹杂", "裂纹", "砂眼", "夹渣",
        "未致密", "机械损伤", "未焊透", "疏松",
    ]
    if contains_any(text, forbidden):
        return False
    if re.search(r"缺陷类型.*?(卷气|未致密|机械损伤)", text):
        return False
    if re.search(r"(其次为|包括|分为|分别为).*?卷气", text):
        return False
    return True


def no_missing_field_fabrication(text: str) -> bool:
    forbidden = [
        "料温", "模温", "浇注温度", "保压", "真空", "真空度", "冷却水", "喷雾", "脱模剂",
        "材料牌号", "合金成分", "熔炼", "除气", "含气量", "模具温度", "模流", "CAE",
    ]
    return not_contains_any(text, forbidden)


def no_prompt_copy(text: str) -> bool:
    return not_contains_any(text, ["作答要求", "必须直接从", "不要复述", "我遵守", "规则", "辅助说明"])


# ============================================================
# 3. 测试用例：不在 prompt 中放无效区域 / 旧缺陷分类词
# ============================================================
def build_tests() -> List[Dict[str, Any]]:
    common_system = (
        "你是一名铝合金压铸气孔缺陷溯源助手。\n"
        "必须严格按照用户指定格式回答。\n"
        "禁止输出思考过程。\n"
        "禁止在指定标题前添加任何前言。\n"
        "当前任务只涉及气孔这一种缺陷类型。\n"
        "只分析有效工艺区域：一般区域、牙孔区域、焊接区域、重点区域；"
        "其中牙孔区域、焊接区域、重点区域属于高影响区域。"
    )

    tests: List[Dict[str, Any]] = []

    tests.append({
        "id": "T01_diecast_three_sections_high_impact",
        "group": "basic_compare",
        "name": "压铸气孔三段式 + 高影响区域逻辑",
        "system": common_system,
        "user": """
请根据下面输入回答，必须且只能输出三部分：
【结论】
【溯源分析】
【建议方向】

要求：
1. 必须直接从【结论】开始；
2. 不要输出任何前言；
3. 必须显式使用：区域缺陷统计、区域最大面积(mm²)、区域最大等效直径(mm)；
4. 牙孔区域、焊接区域、重点区域属于高影响区域；
5. 当前任务只涉及气孔；
6. 建议只能是宏观方向，不要给精确数值配方。

【缺陷结构化摘要】
总缺陷条目=248
总面积(mm²)=25.37
最大等效直径(mm)=1.23
区域缺陷统计=一般区域:181；牙孔区域:43；焊接区域:24
区域最大面积(mm²)=一般区域:0.72；牙孔区域:1.18；焊接区域:0.54
区域最大等效直径(mm)=一般区域:0.96；牙孔区域:1.23；焊接区域:0.81
检测类别汇总=Defect:248
IsPore统计=Yes

【压铸参数摘要】
慢速速度=0.18
高速速度=3.62
料饼厚度=24
铸造压力=982
循环周期=132.4
建压时间=91
射料结束位置=754
压射压力=335
高速行程=186
高速时间=44
慢速行程=552
环境温度=23.8
环境湿度=48.2
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("no_think", no_think_tag),
            ("no_markdown_heading_sections", no_markdown_heading_sections),
            ("has_region_metrics", has_region_metrics),
            ("has_high_impact_logic", has_high_impact_logic),
            ("no_invalid_region_words", no_invalid_region_words),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_quantity_dominant_logic", no_old_quantity_dominant_logic),
            ("no_precise_recipe", no_precise_recipe_numbers),
        ],
    })

    tests.append({
        "id": "T02_no_missing_field_fabrication",
        "group": "basic_compare",
        "name": "不编造缺失字段",
        "system": common_system,
        "user": """
只允许依据输入字段回答，不要提到输入中没有出现的字段。
必须输出：【结论】【溯源分析】【建议方向】三部分。

【缺陷结构化摘要】
总缺陷条目=58
区域缺陷统计=一般区域:41；牙孔区域:8；焊接区域:9
区域最大面积(mm²)=一般区域:0.44；牙孔区域:0.67；焊接区域:0.39
区域最大等效直径(mm)=一般区域:0.61；牙孔区域:0.82；焊接区域:0.58

【压铸参数摘要】
高速速度=3.21
建压时间=88
压射压力=326
环境温度=24.1
环境湿度=46.5
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("no_think", no_think_tag),
            ("no_missing_field_fabrication", no_missing_field_fabrication),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_precise_recipe", no_precise_recipe_numbers),
        ],
    })

    tests.append({
        "id": "T03_exact_yes_only",
        "group": "basic_compare",
        "name": "极简指令：只输出指定词",
        "system": "你必须严格服从用户格式要求。",
        "user": "只输出 YES，不要输出其他任何内容。",
        "checks": [("only_yes", only_yes)],
        "max_new_tokens": 16,
    })

    tests.append({
        "id": "T04_json_only",
        "group": "basic_compare",
        "name": "JSON-only 输出",
        "system": "你是信息抽取助手。用户要求 JSON 时，只能输出 JSON。",
        "user": "判断句子情感：'这次结果有改进，但格式仍然不稳定。' 只输出一个 JSON 对象，字段为 label 和 reason，不要 Markdown。",
        "checks": [("is_json_object", is_json_object), ("no_markdown_fence", no_markdown_fence)],
        "max_new_tokens": 128,
    })

    tests.append({
        "id": "T05_extract_numeric_id_only",
        "group": "basic_compare",
        "name": "抽取任务：只输出数字 ID",
        "system": "你是严格的信息抽取器。OK/NG 后缀只是人工判定备注，抽取编号时只保留数字部分。",
        "user": "从文本中抽取样本编号，只输出逗号分隔数字结果，不要解释：样本38053出现气孔，样本46269需要复查，47243-OK为对照件。",
        "checks": [("only_numeric_ids", only_numeric_ids)],
        "max_new_tokens": 64,
    })

    tests.append({
        "id": "T06_ordered_sections_non_diecast",
        "group": "basic_compare",
        "name": "非压铸场景下的标题顺序跟随",
        "system": "你必须严格按照用户指定标题顺序输出。",
        "user": "请总结今天学习情况。必须且只能按这个顺序输出三个标题：【收获】【问题】【下一步】。不要输出其他标题。内容简短。",
        "checks": [
            ("starts_with_收获", lambda t: starts_with_exact(t, "【收获】")),
            ("section_order", lambda t: section_order_ok(t, ["【收获】", "【问题】", "【下一步】"])),
            ("sections_once", lambda t: section_count_exact(t, ["【收获】", "【问题】", "【下一步】"])),
            ("no_think", no_think_tag),
        ],
    })

    tests.append({
        "id": "T07_forbid_numeric_recipe",
        "group": "basic_compare",
        "name": "禁止给精确工艺数值",
        "system": common_system,
        "user": """
根据下面压铸信息给出分析。必须输出【结论】【溯源分析】【建议方向】三部分。
特别要求：建议方向只能给宏观优化方向，禁止给出任何具体目标数值、区间或配方。

【缺陷结构化摘要】
总缺陷条目=182
区域缺陷统计=一般区域:145；牙孔区域:31；焊接区域:6
区域最大面积(mm²)=一般区域:0.55；牙孔区域:0.82；焊接区域:0.31
区域最大等效直径(mm)=一般区域:0.73；牙孔区域:0.96；焊接区域:0.49

【压铸参数摘要】
高速速度=3.55
铸造压力=988
建压时间=92
压射压力=337
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("no_markdown_heading_sections", no_markdown_heading_sections),
            ("no_think", no_think_tag),
            ("has_high_impact_logic", has_high_impact_logic),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_precise_recipe", no_precise_recipe_numbers),
            ("no_extra_sections", lambda t: no_extra_sections(t, ["【结论】", "【溯源分析】", "【建议方向】"])),
        ],
    })

    tests.append({
        "id": "T08_resist_extra_instruction",
        "group": "basic_compare",
        "name": "抵抗额外废话输出，只给选项",
        "system": "你必须严格遵守最终输出格式。",
        "user": "选择正确答案：2+2=? A.3 B.4 C.5。只输出选项字母，不要解释，不要加句号。",
        "checks": [("only_choice_b", only_choice_b)],
        "max_new_tokens": 16,
    })

    tests.append({
        "id": "S01_project_no_preface_no_extra_headers",
        "group": "project_strict",
        "name": "正式报告：禁止前言、禁止额外标题",
        "system": common_system,
        "user": """
你现在输出正式报告，必须且只能包含三个标题：【结论】【溯源分析】【建议方向】。
禁止出现“小提示”“下面是”“根据您提供的信息”“综合来看”“注意”等前言或附加说明。
禁止使用 Markdown 标题。

【缺陷结构化摘要】
总缺陷条目=96
总面积(mm²)=11.8
最大等效直径(mm)=0.91
区域缺陷统计=一般区域:72；牙孔区域:19；焊接区域:5
区域最大面积(mm²)=一般区域:0.48；牙孔区域:0.73；焊接区域:0.22
区域最大等效直径(mm)=一般区域:0.68；牙孔区域:0.91；焊接区域:0.43
检测类别汇总=Defect:96
IsPore统计=Yes

【压铸参数摘要】
慢速速度=0.20
高速速度=3.48
铸造压力=976
建压时间=90
压射压力=331
环境温度=24.0
环境湿度=49.0
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("no_extra_headers", lambda t: no_extra_sections(t, ["【结论】", "【溯源分析】", "【建议方向】"])),
            ("no_markdown_heading_sections", no_markdown_heading_sections),
            ("no_preface_words", lambda t: not_contains_any(t, ["小提示", "下面是", "根据您", "综合来看", "注意：", "说明："])),
            ("no_invalid_region_words", no_invalid_region_words),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_think", no_think_tag),
        ],
    })

    tests.append({
        "id": "S02_strict_use_region_metrics_high_impact",
        "group": "project_strict",
        "name": "必须使用区域极值和高影响区域，不用数量主导旧逻辑",
        "system": common_system,
        "user": """
请判断当前气孔风险优先级。必须输出【结论】【溯源分析】【建议方向】。
要求：
1. 必须同时使用区域缺陷统计、区域最大面积(mm²)、区域最大等效直径(mm)；
2. 牙孔区域、焊接区域、重点区域属于高影响区域；
3. 不要使用“数量主导区域”作为核心判断。

【缺陷结构化摘要】
总缺陷条目=120
区域缺陷统计=一般区域:100；牙孔区域:15；焊接区域:5；重点区域:2
区域最大面积(mm²)=一般区域:0.31；牙孔区域:1.42；焊接区域:0.27；重点区域:0.70
区域最大等效直径(mm)=一般区域:0.56；牙孔区域:1.36；焊接区域:0.44；重点区域:0.82

【压铸参数摘要】
高速速度=3.36
铸造压力=970
建压时间=86
压射压力=329
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("has_region_metrics", has_region_metrics),
            ("has_high_impact_logic", has_high_impact_logic),
            ("no_invalid_region_words", no_invalid_region_words),
            ("no_quantity_dominant_logic", no_old_quantity_dominant_logic),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_precise_recipe", no_precise_recipe_numbers),
        ],
    })

    tests.append({
        "id": "S03_effective_regions_only",
        "group": "project_strict",
        "name": "只分析有效工艺区域",
        "system": common_system,
        "user": """
请输出【结论】【溯源分析】【建议方向】。
要求：只分析一般区域、牙孔区域、焊接区域、重点区域；高影响区域要单独说明。

【缺陷结构化摘要】
总缺陷条目=110
区域缺陷统计=一般区域:61；牙孔区域:37；焊接区域:12
区域最大面积(mm²)=一般区域:0.46；牙孔区域:0.88；焊接区域:0.36
区域最大等效直径(mm)=一般区域:0.69；牙孔区域:0.95；焊接区域:0.51

【压铸参数摘要】
慢速速度=0.19
高速速度=3.52
铸造压力=984
建压时间=89
压射压力=334
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("no_invalid_region_words", no_invalid_region_words),
            ("has_high_impact_logic", has_high_impact_logic),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_think", no_think_tag),
            ("no_precise_recipe", no_precise_recipe_numbers),
        ],
    })

    tests.append({
        "id": "S04_no_user_prompt_copying",
        "group": "project_strict",
        "name": "禁止复述作答要求和提示词",
        "system": common_system,
        "user": """
作答要求：
1. 必须直接从【结论】开始。
2. 不要复述本段作答要求。
3. 不要解释你遵守了哪些规则。
4. 输出三段式报告。

【缺陷结构化摘要】
总缺陷条目=35
区域缺陷统计=一般区域:21；牙孔区域:10；焊接区域:4
区域最大面积(mm²)=一般区域:0.25；牙孔区域:0.59；焊接区域:0.18
区域最大等效直径(mm)=一般区域:0.42；牙孔区域:0.77；焊接区域:0.35

【压铸参数摘要】
高速速度=3.18
铸造压力=956
建压时间=83
""".strip(),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("not_copy_requirements", no_prompt_copy),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_think", no_think_tag),
        ],
    })

    tests.append({
        "id": "S05_long_prompt_format_stability",
        "group": "project_strict",
        "name": "长提示词下格式稳定性",
        "system": common_system,
        "user": (
            "下面是一段较长输入，请不要被长度干扰。必须输出【结论】【溯源分析】【建议方向】三部分。\n"
            "禁止前言，禁止 think，禁止小提示，禁止 Markdown 表格。\n"
            "建议方向只能是宏观方向，不能给精确目标值。\n\n"
            "【缺陷结构化摘要】\n"
            "总缺陷条目=249\n总面积(mm²)=28.978\n最大等效直径(mm)=1.107\n"
            "区域缺陷统计=一般区域:198；牙孔区域:38；焊接区域:13；重点区域:3\n"
            "区域最大面积(mm²)=一般区域:0.81；牙孔区域:1.04；焊接区域:0.45；重点区域:0.66\n"
            "区域最大等效直径(mm)=一般区域:0.99；牙孔区域:1.107；焊接区域:0.67；重点区域:0.78\n"
            "检测类别汇总=Defect:249\nIsPore统计=Yes\n\n"
            "【压铸参数摘要】\n"
            "慢速速度=0.19\n高速速度=3.55\n料饼厚度=24\n铸造压力=988\n循环周期=133.1\n"
            "建压时间=92\n射料结束位置=755\n压射压力=337\n高速行程=185\n高速时间=43\n"
            "慢速行程=554\n环境温度=23.4\n环境湿度=47.1\n"
            + "\n辅助说明=这些说明不应被复述。" * 30
        ),
        "checks": [
            ("exact_three_sections", has_exact_three_sections),
            ("no_table", lambda t: "|" not in t),
            ("no_prompt_copy", no_prompt_copy),
            ("has_high_impact_logic", has_high_impact_logic),
            ("no_invalid_region_words", no_invalid_region_words),
            ("no_non_pore_defect_classification", no_non_pore_defect_classification),
            ("no_think", no_think_tag),
            ("no_precise_recipe", no_precise_recipe_numbers),
        ],
    })

    return tests


# ============================================================
# 4. 模型加载与推理
# ============================================================
def load_model(base_model_path: str, lora_path: Path, max_seq_length: int, load_in_4bit: bool):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print("\n[Load] loading BASE model with Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        local_files_only=True,
    )

    print("[Load] attaching LoRA adapter with PEFT...")
    model = PeftModel.from_pretrained(
        model,
        str(lora_path),
        is_trainable=False,
        local_files_only=True,
    )

    FastLanguageModel.for_inference(model)
    print("[Load] done. BASE + LoRA adapter is ready for text inference.")
    return model, tokenizer


def build_prompt(tokenizer, system: str, user: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_one(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> Tuple[str, float]:
    import torch

    prompt = build_prompt(tokenizer, system, user, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_cfg = dict(GENERATION_CONFIG)
    gen_cfg["max_new_tokens"] = max_new_tokens

    if tokenizer.eos_token_id is not None:
        gen_cfg["eos_token_id"] = tokenizer.eos_token_id
    gen_cfg["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)
    dt = time.time() - t0

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return normalize_text(text), dt


# ============================================================
# 5. 评分与保存
# ============================================================
def evaluate_output(text: str, checks: List[Tuple[str, Callable[[str], bool]]]) -> Dict[str, Any]:
    check_results = []
    passed = 0

    for name, fn in checks:
        try:
            ok = bool(fn(text))
        except Exception as e:
            ok = False
            print(f"[CheckError] {name}: {e}")
        check_results.append({"name": name, "passed": ok})
        if ok:
            passed += 1

    total = len(checks)
    score = 100.0 * passed / max(1, total)
    failed = [x["name"] for x in check_results if not x["passed"]]

    return {
        "score": score,
        "passed": passed,
        "total": total,
        "failed": failed,
        "checks": check_results,
    }


def save_results(
    out_dir: Path,
    results: List[Dict[str, Any]],
    base_model_path: str,
    model_root: str,
    lora_path: Path,
    max_seq_length: int,
    enable_thinking: bool,
) -> None:
    jsonl_path = out_dir / "high_impact_gas_pore_instruction_following_results.jsonl"
    csv_path = out_dir / "high_impact_gas_pore_instruction_following_scores.csv"
    md_path = out_dir / "high_impact_gas_pore_instruction_following_report.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "group", "name", "score", "passed", "total", "failed", "seconds"])
        for r in results:
            writer.writerow([
                r["id"],
                r.get("group", ""),
                r["name"],
                f"{r['score']:.2f}",
                r["passed"],
                r["total"],
                ";".join(r["failed"]),
                f"{r['seconds']:.3f}",
            ])

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r.get("group", "unknown"), []).append(r)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-4B High-impact Gas-pore Instruction Following Report\n\n")
        f.write(f"- Run time: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Base model: `{base_model_path}`\n")
        f.write(f"- Model root: `{model_root}`\n")
        f.write(f"- Loaded LoRA path: `{lora_path}`\n")
        f.write(f"- enable_thinking: `{enable_thinking}`\n")
        f.write(f"- max_seq_length: `{max_seq_length}`\n\n")

        avg = sum(r["score"] for r in results) / max(1, len(results))
        full = sum(1 for r in results if r["score"] >= 100.0)

        f.write("## Overall Summary\n\n")
        f.write(f"- Average score: **{avg:.2f}**\n")
        f.write(f"- Full pass: **{full}/{len(results)}**\n\n")

        f.write("## Group Summary\n\n")
        f.write("| Group | Avg Score | Full Pass | Count |\n")
        f.write("|---|---:|---:|---:|\n")
        for g, items in groups.items():
            g_avg = sum(x["score"] for x in items) / len(items)
            g_full = sum(1 for x in items if x["score"] >= 100.0)
            f.write(f"| {g} | {g_avg:.2f} | {g_full}/{len(items)} | {len(items)} |\n")
        f.write("\n")

        f.write("## Case Results\n\n")
        for r in results:
            f.write(f"### {r['id']} - {r['name']}\n\n")
            f.write(f"- Group: `{r.get('group', '')}`\n")
            f.write(f"- Score: **{r['score']:.2f}** ({r['passed']}/{r['total']})\n")
            f.write(f"- Failed: `{', '.join(r['failed']) if r['failed'] else 'None'}`\n")
            f.write(f"- Seconds: `{r['seconds']:.3f}`\n\n")
            f.write("Output:\n\n```text\n")
            f.write(r["output"][:4000])
            if len(r["output"]) > 4000:
                f.write("\n... [truncated]\n")
            f.write("\n```\n\n")

    print(f"\n[Save] jsonl: {jsonl_path}")
    print(f"[Save] csv  : {csv_path}")
    print(f"[Save] md   : {md_path}")


def print_summary(results: List[Dict[str, Any]], out_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    avg = sum(r["score"] for r in results) / max(1, len(results))
    full = sum(1 for r in results if r["score"] >= 100.0)
    print(f"adapter: avg_score={avg:.2f}, full_pass={full}/{len(results)}")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r.get("group", "unknown"), []).append(r)

    for g, items in groups.items():
        g_avg = sum(x["score"] for x in items) / len(items)
        g_full = sum(1 for x in items if x["score"] >= 100.0)
        print(f"  - {g}: avg_score={g_avg:.2f}, full_pass={g_full}/{len(items)}")

    bad = [r for r in results if r["failed"]]
    if bad:
        print("\nFailed cases:")
        for r in bad:
            print(f"  - {r['id']} | score={r['score']:.1f} | failed={r['failed']}")

    print(f"结果目录: {out_dir}")


# ============================================================
# 6. 主流程
# ============================================================
def main():
    args = build_arg_parser().parse_args()

    base_model_path = args.base_model
    model_root = args.model_root
    lora_override = args.lora_override
    output_root = args.output_root
    max_seq_length = args.max_seq_length
    load_in_4bit = not args.no_4bit
    enable_thinking = args.enable_thinking

    print("=" * 80)
    print("Qwen3-4B high-impact gas-pore instruction-following evaluation")
    print("=" * 80)
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"BASE_MODEL_PATH      = {base_model_path}")
    print(f"MODEL_ROOT           = {model_root}")
    print(f"LORA_OVERRIDE        = {lora_override or 'AUTO'}")
    print(f"ENABLE_THINKING      = {enable_thinking}")
    print("LoRA/PEFT            = ENABLED")

    lora_path = find_lora_adapter_dir(model_root, lora_override)
    check_model_dirs(base_model_path, model_root, lora_path)

    out_dir = ensure_dir(Path(output_root) / f"eval_run_{now_tag()}")

    model, tokenizer = load_model(
        base_model_path=base_model_path,
        lora_path=lora_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    tests = build_tests()
    results = []

    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test['id']} - {test['name']}")
        try:
            output, seconds = generate_one(
                model=model,
                tokenizer=tokenizer,
                system=test["system"],
                user=test["user"],
                max_new_tokens=test.get("max_new_tokens", args.max_new_tokens),
                enable_thinking=enable_thinking,
            )

            eval_info = evaluate_output(output, test["checks"])

            result = {
                "id": test["id"],
                "group": test.get("group", ""),
                "name": test["name"],
                "system": test["system"],
                "user": test["user"],
                "output": output,
                "seconds": seconds,
                **eval_info,
            }
            results.append(result)

            failed_str = ", ".join(eval_info["failed"]) if eval_info["failed"] else "None"
            preview = output.replace("\n", "\\n")[:700]
            print(f"Score: {eval_info['score']:.1f} | Passed: {eval_info['passed']}/{eval_info['total']} | Failed: {failed_str} | {seconds:.3f}s")
            print("Output preview:")
            print(preview)

        except Exception as e:
            print(f"[ERROR] case failed: {test['id']}: {e}")
            traceback.print_exc()
            results.append({
                "id": test["id"],
                "group": test.get("group", ""),
                "name": test["name"],
                "system": test["system"],
                "user": test["user"],
                "output": "",
                "seconds": math.nan,
                "score": 0.0,
                "passed": 0,
                "total": len(test["checks"]),
                "failed": ["runtime_error"],
                "checks": [],
                "error": str(e),
            })

    save_results(out_dir, results, base_model_path, model_root, lora_path, max_seq_length, enable_thinking)
    print_summary(results, out_dir)


if __name__ == "__main__":
    main()
