#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B-Instruct 压铸气孔缺陷溯源九视角整工件测试脚本
====================================================================

实验目的：
1. 只调用官方 Qwen2.5-VL-7B-Instruct；
2. 输入同一工件的 9 张 X 光检测图及其对应模板图、蒙版图、压铸参数和问题；
3. 让官方 VLM 自己判断气孔候选、区域归属和区域风险；
4. 保存官方 VLM 的视觉层总结，方便人工检查识别是否准确；
5. 基于官方 VLM 自己的视觉总结 + 压铸参数 + 问题生成溯源回答；
6. 输出 all_modes_results.json，结构兼容 standard_e2e_scoring_v2.py。

注意：
- 本脚本不会调用你的 YOLO / SAHI / SIFT / 模板匹配视觉层；
- 本脚本不会把你的视觉层统计结果喂给官方 VLM；
- 官方 VLM 输出的面积/直径只能视为相对估计，不是标定后的真实 mm²/mm。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


# ============================================================
# 1. 固定提示词：官方 VLM 专用，不引用你的 token / YOLO / GraphRAG
# ============================================================

ALLOWED_REGION_NAMES = ["一般区域", "牙孔区域", "焊接区域", "重点区域"]
REGION_ID_TO_NAME = {
    "1": "一般区域",
    "2": "牙孔区域",
    "3": "焊接区域",
    "4": "重点区域",
}
IGNORE_REGION_TEXT = "0、5、6 或无法判断区域均不计入有效缺陷区域。"

VISUAL_SYSTEM_PROMPT = """你是一名铝合金压铸 X 光图缺陷识别助手。
你只能依据当前给出的 X 光检测图、模板参考图、区域蒙版图和区域编号说明进行视觉判断。
不要声称使用了外部检测算法，不要声称使用了 YOLO、SAHI、模板匹配、知识图谱或多模态 token。
""".strip()

VISUAL_USER_PROMPT_TAIL = f"""
请根据上面每个视角的三张图自行完成视觉判断。

图像含义：
1. X 光检测图：白色高亮、亮斑或异常亮白区域优先视为气孔候选。
2. 模板参考图：用于辅助理解当前视角和工件结构，不是检测结果。
3. 区域蒙版图：用于判断气孔候选大致属于哪个工艺区域。

区域编号说明：
- 1：一般区域
- 2：牙孔区域
- 3：焊接区域
- 4：重点区域
- {IGNORE_REGION_TEXT}

判断要求：
1. 缺陷类型统一按气孔处理。
2. 请估计各有效区域的气孔候选数量。
3. 面积和等效直径无法从图片精确标定时，不要编造真实 mm² 或 mm 数值；请用 0-5 的相对等级表示，其中 0 表示未观察到，5 表示最突出。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域主要作为整体波动参考。
5. 如果无法判断，请写明不确定性，但仍需给出基于图像的估计。

必须按下面格式输出，且【视觉层总结JSON】中的 JSON 必须是合法 JSON，不能加注释：

【视觉层总结JSON】
{{
  "defect_type": "气孔",
  "metric_note": "面积和等效直径为官方VLM基于图像的相对估计等级，不是标定后的真实物理量",
  "total_defect_count": 0,
  "total_area_mm2": 0,
  "max_eq_diameter_mm": 0,
  "region_count": {{
    "一般区域": 0,
    "牙孔区域": 0,
    "焊接区域": 0,
    "重点区域": 0
  }},
  "region_max_area_mm2": {{
    "一般区域": 0,
    "牙孔区域": 0,
    "焊接区域": 0,
    "重点区域": 0
  }},
  "region_max_eq_diameter_mm": {{
    "一般区域": 0,
    "牙孔区域": 0,
    "焊接区域": 0,
    "重点区域": 0
  }},
  "quantity_leading_regions": [],
  "risk_focus_regions": [],
  "view_summaries": [
    {{
      "view_index": 1,
      "visible_defect_description": "",
      "region_judgement": "",
      "uncertainty": ""
    }}
  ]
}}

【视觉层总结文本】
1. 气孔候选总体判断：
2. 各区域气孔数量估计：
3. 各区域最大气孔相对大小判断：
4. 高影响区域风险判断：
5. 视觉判断不确定性：
""".strip()

TRACE_SYSTEM_PROMPT = """你是一名铝合金压铸气孔缺陷溯源助手。
你只能依据当前提供的官方多模态模型视觉总结、压铸参数和用户问题作答。
不要声称使用了外部视觉检测算法、YOLO、SAHI、模板匹配、知识图谱或多模态 token。
""".strip()

TRACE_USER_PROMPT_TEMPLATE = """
【官方VLM自行识别的视觉层总结】
{visual_summary_text}

【视觉层结构化估计】
{visual_summary_json}

【压铸参数】
{params_text}

【问题】
{question}

【作答边界】
1. 最终回答必须直接从【结论】开始。
2. 最终回答必须且只能包含【结论】【溯源分析】【建议方向】三部分。
3. 缺陷类型统一按气孔处理。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。
5. 必须区分“数量主导区域”和“风险关注区域”。数量主导区域主要看气孔数量，风险关注区域主要看高影响区域、最大气孔相对大小和区域敏感性。
6. 溯源分析必须结合压铸参数，重点关注高速速度、高速行程、高速时间、建压时间、压射压力、铸造压力、料饼厚度、射料结束位置之间的参数链。
7. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。
8. 不要输出思考过程。
9. 不要输出背景区域、未知区域、重叠区域，也不要分析 0、5、6 区域。
10. 不要引入缩孔、缩松、裂纹、冷隔、夹杂、夹渣等非气孔缺陷。
11. 面积和等效直径来自官方VLM图像估计时，只能作为相对风险参考，不得说成真实标定测量值。
12. 如果视觉判断存在不确定性，需要在溯源中说明“当前判断依赖模型视觉估计，需以后续结构化检测或人工复核确认”，但不能逃避回答。

【必须遵守的输出格式】
【结论】
用一段话给出当前样本的排查优先级判断，不能复述问题，不能出现问号。

【溯源分析】
1. 解释区域气孔分布和高影响区域风险，明确数量主导区域与风险关注区域。
2. 结合高速速度、高速行程、高速时间分析充型卷气风险。
3. 结合建压时间、压射压力、铸造压力等参数分析气孔保留和压实风险，并给出综合优先级。

【建议方向】
1. 给出优先排查的工艺链路。
2. 给出参数与区域气孔表现的联动复核方向。
3. 给出同料号、同视角、同工况样本的对照验证方向。

现在开始作答。
""".strip()


# ============================================================
# 2. 基础 IO
# ============================================================

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# 3. 配置和图片路径
# ============================================================

def collect_sorted_images(folder: str | Path, suffixes=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in suffixes])
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {folder}")
    return files


def resolve_view_triplets(cfg: Dict[str, Any], max_views: int) -> List[Dict[str, Any]]:
    """按 view_index 组装：检测图 + 模板图 + 蒙版图。默认九个视角属于同一个工件。"""
    if cfg.get("image_paths"):
        image_paths = [Path(x) for x in cfg["image_paths"]]
    else:
        image_paths = collect_sorted_images(cfg["image_dir"])

    template_folder = Path(cfg["template_folder"])
    mask_folder = Path(cfg["mask_folder"])
    template_sorted = collect_sorted_images(template_folder) if template_folder.exists() else []
    mask_sorted = collect_sorted_images(mask_folder, suffixes=(".png", ".jpg", ".jpeg", ".bmp")) if mask_folder.exists() else []

    view_start_index = int(cfg.get("view_start_index", 1))
    if max_views <= 0:
        max_views = len(image_paths)
    # 一个工件通常有 9 张 X 光检测图；这里不会拆成多个样本，而是一次性作为同一工件输入 VLM。

    triplets: List[Dict[str, Any]] = []
    for local_i, img_path in enumerate(image_paths[:max_views], start=1):
        view_index = view_start_index + local_i - 1
        indexed_template = template_folder / f"image{view_index}.jpg"
        indexed_mask = mask_folder / f"image{view_index}_mask.png"

        if indexed_template.exists():
            template_path = indexed_template
        elif local_i - 1 < len(template_sorted):
            template_path = template_sorted[local_i - 1]
        else:
            raise FileNotFoundError(f"找不到第 {view_index} 视角模板图: {indexed_template}")

        if indexed_mask.exists():
            mask_path = indexed_mask
        elif local_i - 1 < len(mask_sorted):
            mask_path = mask_sorted[local_i - 1]
        else:
            raise FileNotFoundError(f"找不到第 {view_index} 视角蒙版图: {indexed_mask}")

        triplets.append({
            "view_index": view_index,
            "image_path": str(img_path),
            "template_path": str(template_path),
            "mask_path": str(mask_path),
            "image_name": img_path.name,
            "template_name": template_path.name,
            "mask_name": mask_path.name,
        })
    return triplets


def format_params(params: Dict[str, Any]) -> str:
    order = [
        "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
        "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
        "慢速行程", "环境温度", "环境湿度", "探伤图编号", "样本编号",
    ]
    lines: List[str] = []
    used = set()
    for k in order:
        if k in params and params[k] not in (None, ""):
            lines.append(f"{k}={params[k]}")
            used.add(k)
    for k, v in params.items():
        if k not in used and v not in (None, ""):
            lines.append(f"{k}={v}")
    return "\n".join(lines)


# ============================================================
# 4. Qwen2.5-VL 模型加载与生成
# ============================================================

def import_qwen_runtime():
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "缺少官方 qwen-vl-utils 插件。请先执行：\n"
            "/home/T7/public/miniconda3/envs/finetune1/bin/python -m pip install -U \"qwen-vl-utils[decord]\" "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn\n"
            f"原始错误: {e}"
        ) from e

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as e:
        raise RuntimeError(
            "当前 transformers 版本可能不支持 Qwen2.5-VL。请检查 transformers 是否过旧。\n"
            "推荐先不要动 torch，只升级 transformers/accelerate/qwen-vl-utils。\n"
            f"原始错误: {e}"
        ) from e
    return Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info


def choose_torch_dtype(dtype_name: str):
    dtype_name = (dtype_name or "auto").lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def load_qwen25vl_model(cfg: Dict[str, Any], cli_model_path: str = ""):
    model_path = Path(cli_model_path or cfg.get("model_path", ""))
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"模型目录缺少 config.json，请检查路径是否指到模型根目录: {model_path}")

    Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info = import_qwen_runtime()

    dtype = choose_torch_dtype(str(cfg.get("torch_dtype", "auto")))
    device_map = cfg.get("device_map", "auto")
    local_files_only = bool(cfg.get("local_files_only", True))

    print("=" * 100)
    print(f"[model] loading Qwen2.5-VL from: {model_path}", flush=True)
    print(f"[model] dtype={dtype}, device_map={device_map}, local_files_only={local_files_only}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.eval()

    min_pixels = int(cfg.get("min_pixels", 256 * 28 * 28))
    max_pixels = int(cfg.get("max_pixels", 1024 * 28 * 28))
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )

    return model, processor, process_vision_info


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def qwen_generate(
    model,
    processor,
    process_vision_info,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = get_model_device(model)
    inputs = inputs.to(device)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(temperature and temperature > 0),
        num_beams=1,
        use_cache=True,
    )
    if temperature and temperature > 0:
        gen_kwargs["temperature"] = float(temperature)

    generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


# ============================================================
# 5. Prompt 组装
# ============================================================

def build_visual_messages(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    content.append({
        "type": "text",
        "text": (
            "下面给出若干组压铸件 X 光视角。每组严格按照：X光检测图、模板参考图、区域蒙版图 的顺序提供。\n"
            "你需要自己观察白色高亮气孔候选，并结合蒙版图估计区域归属。"
        ),
    })
    for item in triplets:
        vi = item["view_index"]
        content.append({"type": "text", "text": f"\n【第{vi}组-X光检测图】白色高亮/亮斑优先视为气孔候选。"})
        content.append({"type": "image", "image": item["image_path"]})
        content.append({"type": "text", "text": f"【第{vi}组-模板参考图】用于理解该视角结构。"})
        content.append({"type": "image", "image": item["template_path"]})
        content.append({"type": "text", "text": f"【第{vi}组-区域蒙版图】用于估计气孔候选所在区域。"})
        content.append({"type": "image", "image": item["mask_path"]})
    content.append({"type": "text", "text": "\n" + VISUAL_USER_PROMPT_TAIL})
    return [
        {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_trace_messages(
    question: str,
    params_text: str,
    visual_text: str,
    visual_summary: Dict[str, Any],
    triplets: Optional[List[Dict[str, Any]]] = None,
    trace_use_images: bool = False,
) -> List[Dict[str, Any]]:
    prompt = TRACE_USER_PROMPT_TEMPLATE.format(
        visual_summary_text=visual_text.strip(),
        visual_summary_json=json.dumps(visual_summary, ensure_ascii=False, indent=2),
        params_text=params_text.strip(),
        question=question.strip(),
    )
    if not trace_use_images or not triplets:
        return [
            {"role": "system", "content": TRACE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": "下面再次提供图像，仅作为复核视觉总结使用。"})
    for item in triplets:
        vi = item["view_index"]
        content.append({"type": "text", "text": f"\n【第{vi}组-X光检测图】"})
        content.append({"type": "image", "image": item["image_path"]})
        content.append({"type": "text", "text": f"【第{vi}组-模板参考图】"})
        content.append({"type": "image", "image": item["template_path"]})
        content.append({"type": "text", "text": f"【第{vi}组-区域蒙版图】"})
        content.append({"type": "image", "image": item["mask_path"]})
    content.append({"type": "text", "text": "\n" + prompt})
    return [
        {"role": "system", "content": TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ============================================================
# 6. 视觉总结解析和规范化
# ============================================================

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # 优先从 ```json 代码块中解析
    code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates = code_blocks + [text]
    decoder = json.JSONDecoder()
    for cand in candidates:
        for m in re.finditer(r"\{", cand):
            try:
                obj, _ = decoder.raw_decode(cand[m.start():])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def to_number(x: Any, default: float = 0.0) -> float:
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


def normalize_region_dict(raw: Any, value_type: str = "int") -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {r: 0 for r in ALLOWED_REGION_NAMES}
    out: Dict[str, Any] = {r: 0 for r in ALLOWED_REGION_NAMES}
    for k, v in raw.items():
        ks = str(k).strip()
        region = REGION_ID_TO_NAME.get(ks, ks)
        if region not in ALLOWED_REGION_NAMES:
            continue
        out[region] = to_int(v, 0) if value_type == "int" else round(to_number(v, 0.0), 3)
    return out


def compute_max_regions(d: Dict[str, Any]) -> List[str]:
    vals = {k: to_number(v, 0.0) for k, v in d.items() if to_number(v, 0.0) > 0}
    if not vals:
        return []
    mx = max(vals.values())
    return [k for k, v in vals.items() if abs(v - mx) < 1e-9]


def normalize_visual_summary(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    obj = obj or {}
    region_count = normalize_region_dict(obj.get("region_count"), "int")
    region_area = normalize_region_dict(obj.get("region_max_area_mm2"), "float")
    region_dia = normalize_region_dict(obj.get("region_max_eq_diameter_mm"), "float")

    total_count = to_int(obj.get("total_defect_count"), sum(region_count.values()))
    if total_count <= 0:
        total_count = sum(region_count.values())

    total_area = to_number(obj.get("total_area_mm2"), 0.0)
    if total_area <= 0:
        total_area = round(sum(to_number(v) for v in region_area.values()), 3)

    max_dia = to_number(obj.get("max_eq_diameter_mm"), 0.0)
    if max_dia <= 0 and region_dia:
        max_dia = max(to_number(v) for v in region_dia.values())

    quantity_leading = obj.get("quantity_leading_regions")
    if not isinstance(quantity_leading, list) or not quantity_leading:
        quantity_leading = compute_max_regions(region_count)
    quantity_leading = [r for r in quantity_leading if r in ALLOWED_REGION_NAMES]

    risk_focus = obj.get("risk_focus_regions")
    if not isinstance(risk_focus, list) or not risk_focus:
        risk_focus = sorted(set(compute_max_regions(region_area) + compute_max_regions(region_dia)))
    risk_focus = [r for r in risk_focus if r in ALLOWED_REGION_NAMES]

    return {
        "defect_type": "气孔",
        "source": "Qwen2.5-VL-7B-Instruct visual estimation",
        "metric_note": "面积和等效直径为官方VLM基于图像的相对估计等级，不是标定后的真实物理量",
        "total_defect_count": total_count,
        "total_area_mm2": round(total_area, 3),
        "max_eq_diameter_mm": round(max_dia, 3),
        "region_count": region_count,
        "region_max_area_mm2": region_area,
        "region_max_eq_diameter_mm": region_dia,
        "quantity_leading_regions": quantity_leading,
        "risk_focus_regions": risk_focus,
        "view_summaries": obj.get("view_summaries", []) if isinstance(obj.get("view_summaries"), list) else [],
        "raw_parsed_visual_json": obj,
    }


def extract_visual_text(raw: str) -> str:
    raw = raw or ""
    m = re.search(r"【视觉层总结文本】(?P<t>.*)", raw, flags=re.S)
    if m:
        return m.group("t").strip()
    # 兜底：去掉 JSON 部分，保留原文
    return raw.strip()


def visual_summary_to_text(summary: Dict[str, Any]) -> str:
    return "\n".join([
        "1. 气孔候选总体判断：官方VLM基于图像估计共观察到有效气孔候选约" + str(summary.get("total_defect_count", 0)) + "处。",
        "2. 各区域气孔数量估计：" + "、".join([f"{k}{v}处" for k, v in summary.get("region_count", {}).items()]) + "。",
        "3. 各区域最大气孔相对大小判断：面积等级 " + json.dumps(summary.get("region_max_area_mm2", {}), ensure_ascii=False) + "；等效直径等级 " + json.dumps(summary.get("region_max_eq_diameter_mm", {}), ensure_ascii=False) + "。",
        "4. 高影响区域风险判断：风险关注区域为" + ("、".join(summary.get("risk_focus_regions", [])) or "未能明确判断") + "。",
        "5. 视觉判断不确定性：该结果为官方VLM视觉估计，需以后续结构化检测或人工复核确认。",
    ])


# ============================================================
# 7. 答案规范化与输出结构
# ============================================================

def normalize_trace_answer(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    m = re.search(r"【结论】", text)
    if m:
        text = text[m.start():].strip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def build_all_modes_results(
    cfg: Dict[str, Any],
    run_dir: Path,
    visual_summary: Dict[str, Any],
    visual_text: str,
    triplets: List[Dict[str, Any]],
    case_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "official_vlm_single_model_qwen25vl7b_v1",
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "model_path": cfg.get("model_path", ""),
        "config": {
            "sample_id": cfg.get("sample_id"),
            "workpiece_id": cfg.get("workpiece_id"),
            "image_dir": cfg.get("image_dir"),
            "template_folder": cfg.get("template_folder"),
            "mask_folder": cfg.get("mask_folder"),
            "max_views": cfg.get("max_views"),
            "trace_use_images": cfg.get("trace_use_images", False),
        },
        "run_dir": str(run_dir),
        "selected_views": triplets,
        "visual_summary": visual_summary,
        "vlm_visual_summary_text": visual_text,
        "mode_definitions": {
            "qwen25vl7b_official_vlm": {
                "mode_desc": "官方Qwen2.5-VL-7B直接读取X光图、模板图、蒙版图，自行形成视觉总结后进行气孔溯源",
                "uses_user_visual_layer": False,
                "uses_yolo_sahi_sift": False,
            }
        },
        "cases": case_results,
    }


def write_trace_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_index", "sample_id", "mode", "question", "answer_chars", "starts_with_conclusion", "answer_file"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# ============================================================
# 8. 主流程
# ============================================================

def run(cfg: Dict[str, Any], cli_args: argparse.Namespace) -> None:
    if cli_args.model_path:
        cfg["model_path"] = cli_args.model_path
    if cli_args.max_views is not None:
        cfg["max_views"] = int(cli_args.max_views)

    result_root = ensure_dir(cfg.get("result_output_dir", "./qwen25vl7b_diecast_results"))
    run_dir = result_root / f"qwen25vl7b_run_{timestamp()}"
    ensure_dir(run_dir)

    save_json(run_dir / "runtime_config.json", cfg)

    max_views = int(cfg.get("max_views", 9))
    triplets = resolve_view_triplets(cfg, max_views=max_views)
    save_json(run_dir / "selected_views.json", triplets)

    print("=" * 100)
    print("[input] selected view triplets:")
    print(json.dumps(triplets, ensure_ascii=False, indent=2))

    model, processor, process_vision_info = load_qwen25vl_model(cfg, cli_model_path=cli_args.model_path)

    # -------------------------
    # Step 1: 官方 VLM 自行视觉总结
    # -------------------------
    print("\n" + "=" * 100)
    print("[step 1/2] generating official VLM visual summary...", flush=True)
    visual_messages = build_visual_messages(triplets)
    save_text(run_dir / "vlm_visual_prompt.txt", json.dumps(visual_messages, ensure_ascii=False, indent=2))

    visual_raw = qwen_generate(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        messages=visual_messages,
        max_new_tokens=int(cfg.get("max_new_tokens_visual", 1024)),
        temperature=float(cfg.get("temperature", 0.0)),
    )
    save_text(run_dir / "vlm_visual_raw.txt", visual_raw)

    parsed_visual = extract_first_json_object(visual_raw)
    visual_summary = normalize_visual_summary(parsed_visual)
    visual_text = extract_visual_text(visual_raw)
    if not visual_text:
        visual_text = visual_summary_to_text(visual_summary)

    save_json(run_dir / "vlm_visual_summary_full.json", {
        "raw": visual_raw,
        "parsed_json": parsed_visual,
        "normalized_summary": visual_summary,
        "visual_text": visual_text,
    })
    save_json(run_dir / "vlm_visual_summary_for_scoring.json", visual_summary)
    save_text(run_dir / "vlm_visual_summary.txt", visual_text)

    print("[visual summary normalized]")
    print(json.dumps(visual_summary, ensure_ascii=False, indent=2), flush=True)

    # -------------------------
    # Step 2: 基于官方 VLM 自己视觉总结做溯源
    # -------------------------
    print("\n" + "=" * 100)
    print("[step 2/2] generating trace answers...", flush=True)

    params = cfg.get("params", {}) or {}
    params_text = format_params(params)
    questions = cfg.get("questions", []) or []
    if isinstance(questions, str):
        questions = [questions]
    if not questions:
        raise ValueError("配置文件中没有 questions。")

    mode_key = "qwen25vl7b_official_vlm"
    mode_desc = "官方Qwen2.5-VL-7B直接读取X光图、模板图、蒙版图，自行形成视觉总结后进行气孔溯源"
    trace_use_images = bool(cfg.get("trace_use_images", False))

    cases: List[Dict[str, Any]] = []
    jsonl_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for idx, question in enumerate(questions, start=1):
        print("\n" + "-" * 100)
        print(f"[trace] Q{idx:02d}/{len(questions)}: {question}", flush=True)

        q_dir = ensure_dir(run_dir / f"q{idx:02d}")
        trace_messages = build_trace_messages(
            question=question,
            params_text=params_text,
            visual_text=visual_text,
            visual_summary=visual_summary,
            triplets=triplets,
            trace_use_images=trace_use_images,
        )
        save_text(q_dir / "trace_prompt_messages.json", json.dumps(trace_messages, ensure_ascii=False, indent=2))

        raw_answer = qwen_generate(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            messages=trace_messages,
            max_new_tokens=int(cfg.get("max_new_tokens_trace", cfg.get("max_new_tokens", 512))),
            temperature=float(cfg.get("temperature", 0.0)),
        )
        final_answer = normalize_trace_answer(raw_answer)
        save_text(q_dir / "raw_answer.txt", raw_answer)
        save_text(q_dir / "final_answer.txt", final_answer)

        print(final_answer, flush=True)

        mode_result = {
            "mode": mode_key,
            "mode_desc": mode_desc,
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "uses_user_visual_layer": False,
            "uses_yolo_sahi_sift": False,
            "visual_summary_used": visual_summary,
            "vlm_visual_summary_text": visual_text,
        }
        case = {
            "question_index": idx,
            "sample_id": cfg.get("sample_id", cfg.get("workpiece_id", "unknown")),
            "workpiece_id": cfg.get("workpiece_id", cfg.get("sample_id", "unknown")),
            "question": question,
            "question_type": "official_vlm_trace",
            "visual_summary": visual_summary,
            "vlm_visual_summary_text": visual_text,
            "params": params,
            "image_triplets": triplets,
            "modes": {
                mode_key: mode_result,
            },
        }
        save_json(q_dir / "case_result.json", case)
        cases.append(case)

        row = {
            "question_index": idx,
            "sample_id": cfg.get("sample_id", cfg.get("workpiece_id", "unknown")),
            "workpiece_id": cfg.get("workpiece_id", cfg.get("sample_id", "unknown")),
            "mode": mode_key,
            "mode_desc": mode_desc,
            "question": question,
            "visual_summary": visual_summary,
            "vlm_visual_summary_text": visual_text,
            "params": params,
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "answer_path": str(q_dir / "final_answer.txt"),
        }
        jsonl_rows.append(row)
        summary_rows.append({
            "question_index": idx,
            "sample_id": row["sample_id"],
            "mode": mode_key,
            "question": question,
            "answer_chars": len(final_answer),
            "starts_with_conclusion": str(final_answer.startswith("【结论】")),
            "answer_file": str(q_dir / "final_answer.txt"),
        })

    output_obj = build_all_modes_results(
        cfg=cfg,
        run_dir=run_dir,
        visual_summary=visual_summary,
        visual_text=visual_text,
        triplets=triplets,
        case_results=cases,
    )
    save_json(run_dir / "all_modes_results.json", output_obj)
    save_json(run_dir / "all_results.json", output_obj)
    write_jsonl(run_dir / "trace_answers.jsonl", jsonl_rows)
    write_trace_summary_csv(run_dir / "trace_summary.csv", summary_rows)

    # 简单报告
    report_lines = []
    report_lines.append("# Qwen2.5-VL-7B 官方多模态模型压铸气孔溯源测试\n")
    report_lines.append(f"- 模型路径：`{cfg.get('model_path', '')}`\n")
    report_lines.append(f"- 样本ID：`{cfg.get('sample_id', '')}`\n")
    report_lines.append(f"- 工件ID：`{cfg.get('workpiece_id', '')}`\n")
    report_lines.append(f"- 视角数量：{len(triplets)}\n")
    report_lines.append(f"- 实际输入图片数量：{len(triplets) * 3}（每个视角包含X光检测图、模板图、蒙版图）\n")
    report_lines.append(f"- 是否在溯源阶段再次输入图片：{trace_use_images}\n")
    report_lines.append("\n## 官方VLM视觉层总结\n\n")
    report_lines.append("```json\n")
    report_lines.append(json.dumps(visual_summary, ensure_ascii=False, indent=2))
    report_lines.append("\n```\n\n")
    report_lines.append(visual_text + "\n\n")
    for case in cases:
        r = case["modes"][mode_key]
        report_lines.append(f"## Q{case['question_index']:02d}\n\n")
        report_lines.append(case["question"] + "\n\n")
        report_lines.append(r["final_answer"] + "\n\n")
    save_text(run_dir / "qwen25vl7b_trace_report.md", "".join(report_lines))

    print("\n" + "=" * 100)
    print("[done] Qwen2.5-VL-7B 测试完成")
    print(f"[done] result_dir           : {run_dir}")
    print(f"[done] visual summary       : {run_dir / 'vlm_visual_summary.txt'}")
    print(f"[done] scoring input        : {run_dir / 'all_modes_results.json'}")
    print(f"[done] trace answers jsonl  : {run_dir / 'trace_answers.jsonl'}")
    print(f"[done] report               : {run_dir / 'qwen25vl7b_trace_report.md'}")
    print("=" * 100)

    # 释放显存
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B 官方多模态压铸气孔溯源单模型测试")
    parser.add_argument("--config", type=str, required=True, help="配置 JSON 路径")
    parser.add_argument("--model-path", type=str, default="", help="覆盖配置中的 model_path")
    parser.add_argument("--max-views", type=int, default=None, help="覆盖配置中的 max_views；正式整工件测试用 9，传 0 表示读取 image_dir 下全部图片")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    cfg = load_json(args.config)
    run(cfg, args)


if __name__ == "__main__":
    main()
