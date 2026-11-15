#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态训练后 LLM 指令跟随专项测试脚本（Qwen3-4B）

测试模式：
1. llm_text：
   只测试 BASE + LoRA，不加任何多模态 prefix。
   用于判断 LLM/LoRA 本身是否还听话。

2. zero_prefix：
   在 prompt 前加一段全零 prefix token。
   用于判断“只要加 prefix 位置”是否会破坏指令跟随。

3. region_numeric_prefix：
   在 prompt 前加区域/参数 prefix token。
   用于判断“映射层 token”是否会破坏格式输出。

判断逻辑：
- llm_text 高、zero_prefix 低：前缀位置干扰。
- llm_text 高、region_numeric_prefix 低：区域/参数映射 token 干扰。
- llm_text 本身低：LLM/LoRA 本身已经不稳定。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from multimodal.config_region_mm import RegionAwareMMConfig
from multimodal.model_region_mm import RegionAwareMultiModalModel


# ============================================================
# 默认路径
# ============================================================

FORCE_BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-8B-stage1-book-merged-4bit"

DEFAULT_MODEL_DIR = (
    "/home/T7/wkq/end-to-end/runs/"
    "qwen3_8b_region_mm_frozen_llm_from_merged_stage1/best"
)

FALLBACK_STAGE2_ADAPTER_PATH = (
    "/home/T7/wkq/end-to-end/runs/"
    "qwen3_8b_stage2_from_merged_second_sft/best_adapter"
)

DEFAULT_OUTPUT_DIR = "/home/T7/wkq/end-to-end/test_outputs/mm_instruction_follow_eval_8b"

DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_ZERO_PREFIX_LEN = 32


# ============================================================
# 检查函数
# ============================================================

SECTION_RE = re.compile(
    r"【结论】(?P<c>.*?)【溯源分析】(?P<a>.*?)【建议方向】(?P<s>.*)",
    re.DOTALL,
)
HEADER_RE = re.compile(r"【[^】]+】")


def norm(x: str) -> str:
    return (x or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_think(x: str) -> str:
    x = re.sub(r"<think>.*?</think>", "", x or "", flags=re.DOTALL | re.IGNORECASE)
    x = re.sub(r"<\s*/?\s*think\s*>", "", x, flags=re.IGNORECASE)
    return x.strip()


def only_yes(x: str) -> bool:
    return norm(x) == "YES"


def only_b(x: str) -> bool:
    return norm(x) == "B"


def json_only(x: str) -> bool:
    try:
        return isinstance(json.loads(norm(x)), dict)
    except Exception:
        return False


def starts_with_conclusion(x: str) -> bool:
    return norm(x).startswith("【结论】")


def exact_three_sections(x: str) -> bool:
    x = norm(x)
    allow = ["【结论】", "【溯源分析】", "【建议方向】"]

    if not x.startswith("【结论】"):
        return False

    if not all(x.count(h) == 1 for h in allow):
        return False

    headers = HEADER_RE.findall(x)
    if any(h not in allow for h in headers):
        return False

    return x.find("【结论】") < x.find("【溯源分析】") < x.find("【建议方向】")


def has_content(x: str) -> bool:
    m = SECTION_RE.search(x or "")
    if not m:
        return False

    return (
        len(m.group("c").strip()) >= 6
        and len(m.group("a").strip()) >= 30
        and len(m.group("s").strip()) >= 20
    )


def numbered_analysis(x: str) -> bool:
    m = SECTION_RE.search(x or "")
    if not m:
        return False

    a = m.group("a")
    return all(re.search(rf"(^|\n)\s*{i}[.．、]", a) for i in [1, 2, 3])


def numbered_suggestion(x: str) -> bool:
    m = SECTION_RE.search(x or "")
    if not m:
        return False

    s = m.group("s")
    return all(re.search(rf"(^|\n)\s*{i}[.．、]", s) for i in [1, 2, 3])


def no_bad_headers(x: str) -> bool:
    bad = [
        "【区域风险】",
        "【区域风险判断】",
        "【参数关联】",
        "【参数关联建议】",
        "【溯源方向】",
        "【检测依据】",
        "【后续建议】",
        "【图谱辅助证据】",
        "【输入说明】",
        "【作答要求】",
    ]
    return not any(b in (x or "") for b in bad)


def no_think(x: str) -> bool:
    y = (x or "").lower()
    return (
        "<think>" not in y
        and "</think>" not in y
        and "思考过程" not in (x or "")
        and "推理过程" not in (x or "")
    )


def no_prompt_copy(x: str) -> bool:
    bad = [
        "输出模板",
        "回答格式模板",
        "参考回答模板",
        "作答要求",
        "最终要求",
        "硬性要求",
        "请严格",
        "我将严格",
        "输入模式",
        "辅助说明",
    ]
    return not any(b in (x or "") for b in bad)


def no_graph_debug(x: str) -> bool:
    bad = [
        "命中实体",
        "命中规则",
        "图谱命中",
        "source_rules",
        "graph_debug",
        "R1_",
        "R2_",
        "R3_",
        "调试",
    ]
    return not any(b in (x or "") for b in bad)


def no_non_pore_words(x: str) -> bool:
    bad = [
        "缩孔",
        "缩松",
        "冷隔",
        "夹杂",
        "裂纹",
        "砂眼",
        "夹渣",
        "未致密",
        "机械损伤",
        "未焊透",
        "疏松",
    ]
    return not any(b in (x or "") for b in bad)


def no_invalid_regions(x: str) -> bool:
    bad = ["背景", "未知区域", "未知(0)", "未知区域(0)", "背景/未知"]
    return not any(b in (x or "") for b in bad)


def no_precise_recipe(x: str) -> bool:
    patterns = [
        r"(调整到|调至|设为|设置为|控制在|提高到|降低到|升至|降至|缩短到|延长到)\s*\d+(?:\.\d+)?\s*(?:m/s|MPa|ms|mm|s|℃|%|左右)?",
        r"\d+(?:\.\d+)?\s*(?:m/s|MPa|ms|mm|s|℃|%)\s*(?:左右|附近|范围内|以内|以上|以下)",
        r"(?:建议|推荐).*?\d+(?:\.\d+)?\s*(?:m/s|MPa|ms|mm|s|℃|%)",
    ]
    return not any(re.search(p, x or "") for p in patterns)


def has_region_metrics(x: str) -> bool:
    x = x or ""
    return (
        ("区域缺陷统计" in x or "区域气孔" in x or "区域缺陷数量" in x)
        and ("区域最大面积" in x or "最大面积" in x)
        and ("区域最大等效直径" in x or "最大等效直径" in x)
    )


def has_param_chain(x: str) -> bool:
    keys = [
        "高速速度",
        "高速行程",
        "高速时间",
        "建压时间",
        "压射压力",
        "铸造压力",
        "慢速速度",
        "料饼厚度",
    ]
    return sum(1 for k in keys if k in (x or "")) >= 2


def has_high_impact(x: str) -> bool:
    x = x or ""
    return any(r in x for r in ["牙孔区域", "焊接区域", "重点区域"]) and any(
        w in x for w in ["高影响区域", "风险关注", "优先关注", "加工敏感", "功能敏感", "局部风险"]
    )


def binary_clear(x: str) -> bool:
    return any(w in (x or "") for w in ["更偏向", "优先判断为", "不是当前优先", "不宜直接判断为", "相比之下"])


CHECKS: Dict[str, Callable[[str], bool]] = {
    "only_yes": only_yes,
    "only_b": only_b,
    "json_only": json_only,
    "starts_with_conclusion": starts_with_conclusion,
    "exact_three_sections": exact_three_sections,
    "has_content": has_content,
    "numbered_analysis": numbered_analysis,
    "numbered_suggestion": numbered_suggestion,
    "no_bad_headers": no_bad_headers,
    "no_think": no_think,
    "no_prompt_copy": no_prompt_copy,
    "no_graph_debug": no_graph_debug,
    "no_non_pore_words": no_non_pore_words,
    "no_invalid_regions": no_invalid_regions,
    "no_precise_recipe": no_precise_recipe,
    "has_region_metrics": has_region_metrics,
    "has_param_chain": has_param_chain,
    "has_high_impact": has_high_impact,
    "binary_clear": binary_clear,
}


# ============================================================
# 测试用例
# ============================================================

BASE_SYSTEM = "你必须严格服从用户输出要求。不要输出思考过程。"

PROJECT_SYSTEM = """你是一名铝合金压铸气孔缺陷溯源助手。
必须直接从【结论】开始。
只能输出【结论】【溯源分析】【建议方向】三部分。
【溯源分析】和【建议方向】必须写 1. 2. 3. 三个编号点。
缺陷类型统一按气孔处理。
只分析有效工艺区域：一般区域、牙孔区域、焊接区域、重点区域；其中牙孔区域、焊接区域、重点区域属于高影响区域。
建议方向只给宏观排查方向，不给具体参数设定值。
"""

DOMAIN_INPUT = """【视觉层文字总结】
缺陷类型统一按气孔处理。
有效工艺区域包括：一般区域、牙孔区域、焊接区域、重点区域；其中牙孔区域、焊接区域、重点区域为高影响区域。
总气孔条目=249
总面积(mm²)=28.978
最大等效直径(mm)=1.107
区域缺陷统计={"一般区域": 198, "牙孔区域": 38, "焊接区域": 13, "重点区域": 3}
区域最大面积(mm²)={"一般区域": 0.81, "牙孔区域": 1.04, "焊接区域": 0.45, "重点区域": 0.66}
区域最大等效直径(mm)={"一般区域": 0.99, "牙孔区域": 1.107, "焊接区域": 0.67, "重点区域": 0.78}

【压铸参数摘要】
慢速速度=0.19
高速速度=3.55
料饼厚度=24
铸造压力=988
循环周期=133.1
建压时间=92
射料结束位置=755
压射压力=337
高速行程=185
高速时间=43
慢速行程=554
环境温度=23.4
环境湿度=47.1
"""


def build_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "B01_yes_only",
            "group": "basic",
            "system": BASE_SYSTEM,
            "user": "只输出 YES，不要输出其他任何内容。",
            "checks": ["only_yes"],
            "max_new_tokens": 16,
            "modes": ["llm_text", "zero_prefix"],
        },
        {
            "id": "B02_choice_only",
            "group": "basic",
            "system": BASE_SYSTEM,
            "user": "选择正确答案：2+2=? A.3 B.4 C.5。只输出选项字母，不要解释，不要加句号。",
            "checks": ["only_b"],
            "max_new_tokens": 16,
            "modes": ["llm_text", "zero_prefix"],
        },
        {
            "id": "B03_json_only",
            "group": "basic",
            "system": "你是信息抽取助手。用户要求 JSON 时，只能输出 JSON。",
            "user": "判断句子情感：'这次模型格式更稳定，但仍要验证多模态前缀影响。' 只输出一个 JSON 对象，字段为 label 和 reason。",
            "checks": ["json_only", "no_think"],
            "max_new_tokens": 128,
            "modes": ["llm_text", "zero_prefix"],
        },
        {
            "id": "P01_project_strict",
            "group": "project",
            "system": PROJECT_SYSTEM,
            "user": DOMAIN_INPUT + "\n【问题】请基于当前样本的气孔分布、区域极值和压铸参数链进行溯源分析，并给出建议方向。",
            "checks": [
                "starts_with_conclusion",
                "exact_three_sections",
                "has_content",
                "numbered_analysis",
                "numbered_suggestion",
                "no_bad_headers",
                "no_prompt_copy",
                "no_think",
                "no_graph_debug",
                "no_non_pore_words",
                "no_invalid_regions",
                "no_precise_recipe",
                "has_region_metrics",
                "has_param_chain",
                "has_high_impact",
            ],
            "max_new_tokens": 768,
            "modes": ["llm_text", "zero_prefix", "region_numeric_prefix"],
        },
        {
            "id": "P02_no_extra_titles",
            "group": "project",
            "system": PROJECT_SYSTEM,
            "user": DOMAIN_INPUT + "\n【问题】不要使用【区域风险判断】【参数关联建议】【溯源方向】这些标题，只能使用【结论】【溯源分析】【建议方向】。",
            "checks": [
                "starts_with_conclusion",
                "exact_three_sections",
                "numbered_analysis",
                "numbered_suggestion",
                "no_bad_headers",
                "no_prompt_copy",
                "no_think",
                "no_non_pore_words",
                "no_invalid_regions",
            ],
            "max_new_tokens": 768,
            "modes": ["llm_text", "zero_prefix", "region_numeric_prefix"],
        },
        {
            "id": "P03_binary_decision",
            "group": "project",
            "system": PROJECT_SYSTEM,
            "user": DOMAIN_INPUT + "\n【问题】当前一般区域气孔数量明显多于牙孔区域，是否说明主要矛盾更偏向整体充型稳定性，而不是牙孔区域局部异常？必须明确更偏向哪一项，并说明另一项为何不是当前优先判断。",
            "checks": [
                "starts_with_conclusion",
                "exact_three_sections",
                "numbered_analysis",
                "numbered_suggestion",
                "binary_clear",
                "has_region_metrics",
                "has_param_chain",
                "has_high_impact",
                "no_bad_headers",
                "no_prompt_copy",
                "no_think",
                "no_precise_recipe",
            ],
            "max_new_tokens": 768,
            "modes": ["llm_text", "zero_prefix", "region_numeric_prefix"],
        },
        {
            "id": "P04_forbid_recipe",
            "group": "project",
            "system": PROJECT_SYSTEM,
            "user": DOMAIN_INPUT + "\n【问题】高速速度、高速行程和高速时间之间的匹配关系，可能怎样影响当前气孔数量和区域分布？建议方向禁止给任何具体目标值或数值区间。",
            "checks": [
                "starts_with_conclusion",
                "exact_three_sections",
                "numbered_analysis",
                "numbered_suggestion",
                "has_region_metrics",
                "has_param_chain",
                "no_precise_recipe",
                "no_bad_headers",
                "no_non_pore_words",
                "no_invalid_regions",
                "no_think",
            ],
            "max_new_tokens": 768,
            "modes": ["llm_text", "zero_prefix", "region_numeric_prefix"],
        },
        {
            "id": "P05_prompt_copy_resistance",
            "group": "project",
            "system": PROJECT_SYSTEM,
            "user": DOMAIN_INPUT + "\n【问题】请不要复述我的作答要求，也不要解释你遵守了哪些规则。直接输出正式三段式报告。",
            "checks": [
                "starts_with_conclusion",
                "exact_three_sections",
                "numbered_analysis",
                "numbered_suggestion",
                "no_prompt_copy",
                "no_bad_headers",
                "no_think",
                "no_non_pore_words",
            ],
            "max_new_tokens": 768,
            "modes": ["llm_text", "zero_prefix", "region_numeric_prefix"],
        },
    ]


# ============================================================
# 模型加载
# ============================================================

def load_mm_model(model_dir: Path, force_base: str, fallback_adapter: str) -> RegionAwareMultiModalModel:
    meta_path = model_dir / "meta.json"
    meta = {}

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    cfg = RegionAwareMMConfig()

    # 强制使用当前书本知识合并基座，避免 meta 旧路径污染
    cfg.llm_base_path = force_base

    adapter_dir = model_dir / "llm_adapter"

    if adapter_dir.exists():
        cfg.llm_stage2_adapter_path = str(adapter_dir)
    elif meta.get("llm_stage2_adapter_path"):
        cfg.llm_stage2_adapter_path = meta["llm_stage2_adapter_path"]
    else:
        cfg.llm_stage2_adapter_path = fallback_adapter

    if meta.get("max_seq_length"):
        cfg.max_seq_length = int(meta["max_seq_length"])

    cfg.continue_train_stage1_adapter = False
    cfg.continue_train_stage2_adapter = True
    cfg.train_new_lora_adapter = False

    print("=" * 80, flush=True)
    print("[LOAD] RegionAwareMultiModalModel", flush=True)
    print(f"[LOAD] model_dir              = {model_dir}", flush=True)
    print(f"[LOAD] forced llm_base_path   = {cfg.llm_base_path}", flush=True)
    print(f"[LOAD] llm_stage2_adapter     = {cfg.llm_stage2_adapter_path}", flush=True)
    print(f"[LOAD] max_seq_length         = {cfg.max_seq_length}", flush=True)
    print("=" * 80, flush=True)

    model = RegionAwareMultiModalModel(cfg)

    head_path = model_dir / "multimodal_heads.pt"

    if head_path.exists():
        print(f"[LOAD] loading multimodal heads: {head_path}", flush=True)
        heads = torch.load(head_path, map_location="cpu")
        model.projector.load_state_dict(heads["projector"], strict=True)
        model.param_encoder.load_state_dict(heads["param_encoder"], strict=True)
        model.region_prior.load_state_dict(heads["region_prior"], strict=True)

        if model.prefix_fusion is not None and heads.get("prefix_fusion") is not None:
            model.prefix_fusion.load_state_dict(heads["prefix_fusion"], strict=True)
    else:
        print(f"[WARN] multimodal_heads.pt not found: {head_path}; prefix tests may be limited", flush=True)

    model.eval()
    if hasattr(model, "llm"):
        model.llm.eval()

    return model


# ============================================================
# 生成
# ============================================================

def build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def clean_generation(text: str) -> str:
    text = strip_think(text or "")
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")

    m = re.search(r"【结论】", text)
    if m:
        text = text[m.start():]

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_zero_prefix(model: RegionAwareMultiModalModel, prefix_len: int) -> torch.Tensor:
    hidden = int(model.llm.get_input_embeddings().weight.shape[1])
    return torch.zeros((prefix_len, hidden), dtype=model.llm_embed_dtype, device=model.runtime_device)


def build_region_numeric_prefix(model: RegionAwareMultiModalModel) -> torch.Tensor:
    summary = {
        "total_defect_count": 249,
        "total_area_mm2": 28.978,
        "max_eq_diameter_mm": 1.107,
        "region_count": {"一般区域": 198, "牙孔区域": 38, "焊接区域": 13, "重点区域": 3},
        "region_max_area_mm2": {"一般区域": 0.81, "牙孔区域": 1.04, "焊接区域": 0.45, "重点区域": 0.66},
        "region_max_eq_diameter_mm": {"一般区域": 0.99, "牙孔区域": 1.107, "焊接区域": 0.67, "重点区域": 0.78},
    }

    params = {
        "慢速速度": 0.19,
        "高速速度": 3.55,
        "料饼厚度": 24,
        "铸造压力": 988,
        "循环周期": 133.1,
        "建压时间": 92,
        "射料结束位置": 755,
        "压射压力": 337,
        "高速行程": 185,
        "高速时间": 43,
        "慢速行程": 554,
        "环境温度": 23.4,
        "环境湿度": 47.1,
    }

    try:
        region_tokens = model.encode_region_prior_tokens(summary, [], [])
        numeric_tokens = model.encode_numeric_tokens(params)
        return model._compose_prefix_tokens([region_tokens, numeric_tokens])
    except Exception as e:
        print(f"[WARN] region_numeric_prefix failed: {repr(e)}; fallback zero", flush=True)
        return build_zero_prefix(model, DEFAULT_ZERO_PREFIX_LEN)


@torch.no_grad()
def generate_text(model, system: str, user: str, max_new_tokens: int) -> str:
    prompt = build_prompt(model.tokenizer, system, user)

    model.tokenizer.truncation_side = "left"
    model.tokenizer.padding_side = "left"

    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.cfg.max_seq_length,
    ).to(model.runtime_device)

    out = model.llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        use_cache=True,
    )

    input_len = inputs["input_ids"].shape[-1]
    return clean_generation(model.tokenizer.decode(out[0][input_len:], skip_special_tokens=True))


@torch.no_grad()
def generate_with_prefix(model, system: str, user: str, prefix_tokens: torch.Tensor, max_new_tokens: int) -> str:
    prompt = build_prompt(model.tokenizer, system, user)

    model.tokenizer.truncation_side = "left"
    model.tokenizer.padding_side = "left"

    max_prompt_len = max(128, int(model.cfg.max_seq_length) - int(prefix_tokens.size(0)))

    prompt_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_len,
    ).input_ids.to(model.runtime_device)

    text_embeds = model.llm.get_input_embeddings()(prompt_ids).squeeze(0)
    text_embeds = text_embeds.to(dtype=model.llm_embed_dtype, device=model.runtime_device)

    inputs_embeds = torch.cat([prefix_tokens, text_embeds], dim=0).unsqueeze(0)
    attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=model.runtime_device)

    out = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        use_cache=True,
    )

    return clean_generation(model.tokenizer.decode(out[0], skip_special_tokens=True))


# ============================================================
# 评分与保存
# ============================================================

def evaluate_case(text: str, check_names: List[str]) -> Dict[str, Any]:
    passed = 0
    checks = []

    for name in check_names:
        try:
            ok = bool(CHECKS[name](text))
        except Exception:
            ok = False

        checks.append({"name": name, "passed": ok})
        passed += int(ok)

    total = len(check_names)
    failed = [x["name"] for x in checks if not x["passed"]]

    return {
        "score": 100.0 * passed / max(total, 1),
        "passed": passed,
        "total": total,
        "failed": failed,
        "checks": checks,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    by_group: Dict[str, List[Dict[str, Any]]] = {}

    for r in results:
        by_mode.setdefault(r["run_mode"], []).append(r)
        by_group.setdefault(r["group"], []).append(r)

    def stat(items):
        avg = sum(x["score"] for x in items) / max(1, len(items))
        full = sum(1 for x in items if x["score"] >= 100)
        return {"avg_score": avg, "full_pass": full, "count": len(items)}

    mode_summary = {k: stat(v) for k, v in by_mode.items()}
    group_summary = {k: stat(v) for k, v in by_group.items()}

    diagnosis = []
    llm = mode_summary.get("llm_text", {}).get("avg_score")
    zero = mode_summary.get("zero_prefix", {}).get("avg_score")
    rn = mode_summary.get("region_numeric_prefix", {}).get("avg_score")

    if llm is not None:
        diagnosis.append(
            "llm_text 高：纯文本 LLM/LoRA 指令跟随基本正常。"
            if llm >= 90
            else "llm_text 低：不加前缀时 LLM/LoRA 本身就不稳定。"
        )

    if llm is not None and zero is not None:
        diagnosis.append(
            "zero_prefix 明显低于 llm_text：前缀位置会干扰指令跟随。"
            if zero + 8 < llm
            else "zero_prefix 与 llm_text 差距不大：单纯前缀位置不是主要问题。"
        )

    if llm is not None and rn is not None:
        diagnosis.append(
            "region_numeric_prefix 明显低于 llm_text：区域/参数映射 token 可能干扰格式输出。"
            if rn + 8 < llm
            else "region_numeric_prefix 与 llm_text 差距不大：映射层前缀对指令跟随破坏不明显。"
        )

    return {
        "mode_summary": mode_summary,
        "group_summary": group_summary,
        "diagnosis": diagnosis,
    }


def save_outputs(out_dir: Path, results: List[Dict[str, Any]], summary: Dict[str, Any], args):
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "mm_instruction_follow_results.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "mm_instruction_follow_scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "run_mode", "score", "passed", "total", "failed"])
        for r in results:
            w.writerow([
                r["id"],
                r["group"],
                r["run_mode"],
                f"{r['score']:.2f}",
                r["passed"],
                r["total"],
                ";".join(r["failed"]),
            ])

    with (out_dir / "mm_instruction_follow_report.md").open("w", encoding="utf-8") as f:
        f.write("# 多模态大模型指令跟随专项评估报告\n\n")

        f.write("## 模式汇总\n\n")
        f.write("| 模式 | 平均分 | 满分通过 | 用例数 |\n")
        f.write("|---|---:|---:|---:|\n")
        for mode, s in summary["mode_summary"].items():
            f.write(f"| {mode} | {s['avg_score']:.2f} | {s['full_pass']}/{s['count']} | {s['count']} |\n")

        f.write("\n## 诊断结论\n\n")
        for d in summary["diagnosis"]:
            f.write(f"- {d}\n")

        f.write("\n## 逐项结果\n\n")
        for r in results:
            f.write(f"### {r['id']} / {r['run_mode']} / {r['score']:.2f}\n\n")
            f.write(f"- failed: `{', '.join(r['failed']) if r['failed'] else 'None'}`\n\n")
            f.write("```text\n")
            f.write((r["output"] or "")[:5000])
            f.write("\n```\n\n")

    print("[save]", out_dir, flush=True)


# ============================================================
# 主流程
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--base-model", default=FORCE_BASE_MODEL_PATH)
    p.add_argument("--stage2-adapter", default=FALLBACK_STAGE2_ADAPTER_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--zero-prefix-len", type=int, default=DEFAULT_ZERO_PREFIX_LEN)
    p.add_argument("--only-mode", choices=["all", "llm_text", "zero_prefix", "region_numeric_prefix"], default="all")
    return p


def main():
    args = build_arg_parser().parse_args()

    model = load_mm_model(Path(args.model_dir), args.base_model, args.stage2_adapter)

    tests = build_tests()

    prefixes = {
        "zero_prefix": build_zero_prefix(model, args.zero_prefix_len),
        "region_numeric_prefix": build_region_numeric_prefix(model),
    }

    results = []

    for test in tests:
        modes = test["modes"]

        if args.only_mode != "all":
            modes = [m for m in modes if m == args.only_mode]

        for mode in modes:
            print("=" * 80, flush=True)
            print(f"[RUN] {test['id']} | {mode}", flush=True)

            t0 = time.time()

            try:
                if mode == "llm_text":
                    out = generate_text(
                        model,
                        test["system"],
                        test["user"],
                        int(test.get("max_new_tokens", args.max_new_tokens)),
                    )
                else:
                    out = generate_with_prefix(
                        model,
                        test["system"],
                        test["user"],
                        prefixes[mode],
                        int(test.get("max_new_tokens", args.max_new_tokens)),
                    )

                ev = evaluate_case(out, test["checks"])

                item = {
                    "id": test["id"],
                    "group": test["group"],
                    "run_mode": mode,
                    "system": test["system"],
                    "user": test["user"],
                    "output": out,
                    "seconds": time.time() - t0,
                    **ev,
                }

                results.append(item)

                print(f"[score] {ev['score']:.2f} failed={ev['failed']}", flush=True)
                print(out[:700].replace("\n", "\\n"), flush=True)

            except Exception as e:
                traceback.print_exc()

                results.append({
                    "id": test["id"],
                    "group": test["group"],
                    "run_mode": mode,
                    "output": "",
                    "seconds": time.time() - t0,
                    "score": 0,
                    "passed": 0,
                    "total": len(test["checks"]),
                    "failed": ["runtime_error"],
                    "checks": [],
                    "error": repr(e),
                })

    summary = summarize(results)

    print("\n[SUMMARY]", flush=True)
    for mode, s in summary["mode_summary"].items():
        print(f"{mode}: avg={s['avg_score']:.2f} full={s['full_pass']}/{s['count']}", flush=True)

    for d in summary["diagnosis"]:
        print(" - " + d, flush=True)

    save_outputs(Path(args.output_dir), results, summary, args)


if __name__ == "__main__":
    main()