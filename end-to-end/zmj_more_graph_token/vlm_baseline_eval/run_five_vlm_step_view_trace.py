#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐视角视觉判断 + 五个官方多模态大模型顺序压铸气孔溯源测试
============================================================================

做什么：
1. 五个官方/开源 VLM 顺序加载、顺序释放，避免显存叠加；
2. 每个模型逐视角判断：每次只输入 1 组视角 = X光检测图 + 模板图 + 蒙版图；
3. 每个模型先保存逐视角视觉判断，再聚合成该模型自己的视觉层总结；
4. 每个模型再基于自己的视觉总结 + 压铸参数 + 问题生成溯源回答；
5. 溯源结果输出 all_modes_results.json，适配 standard_e2e_scoring_v2.py；
6. 视觉层结果单独保存，配套 validate_vlm_visual_results.py 做验证。

重要边界：
- 不调用你的 YOLO/SAHI/SIFT 视觉层；
- 不给训练语料、GraphRAG、图谱证据和你的视觉统计；
- 官方 VLM 自己看图、自己判断，判断错误也保留，用于论文对比。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

ALLOWED_REGION_NAMES = ["一般区域", "牙孔区域", "焊接区域", "重点区域"]
REGION_ID_TO_NAME = {"1": "一般区域", "2": "牙孔区域", "3": "焊接区域", "4": "重点区域"}
HIGH_IMPACT_REGIONS = ["牙孔区域", "焊接区域", "重点区域"]

# ============================================================
# 基础工具
# ============================================================

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text or ""), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def choose_dtype(dtype_name: str):
    s = str(dtype_name or "auto").lower()
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def get_first_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_open(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def collect_sorted_images(folder: str | Path, suffixes=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in suffixes])
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {folder}")
    return files


def resize_image_for_vlm(src: str | Path, dst: str | Path, max_side: int = 1280, quality: int = 95) -> str:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h))) if max_side > 0 else 1.0
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    # 对 X 光尽量少压缩；质量可以在 config 中调高。
    img.save(dst, format="JPEG", quality=int(quality), optimize=True)
    return str(dst)


def find_model_dir(root: str | Path, patterns: List[str]) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend([p for p in root.rglob(pat) if p.is_dir()])
    # 优先选择含 config.json 的目录
    candidates = sorted(set(candidates), key=lambda p: (0 if (p / "config.json").exists() else 1, len(str(p)), str(p)))
    return candidates[0] if candidates else None


def resolve_model_path(model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> Path:
    raw = str(model_cfg.get("path", "")).strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    root = model_cfg.get("baseline_root") or global_cfg.get("baseline_root") or "/home/T7/wkq/end-to-end/vlm_baseline_eval/vlm_baselines"
    patterns = model_cfg.get("path_globs") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    found = find_model_dir(root, patterns)
    if found is not None:
        model_cfg["path"] = str(found)
        return found
    raise FileNotFoundError(f"模型路径不存在且自动搜索失败: path={raw}, root={root}, patterns={patterns}")


# ============================================================
# 路径与配置
# ============================================================

def resolve_view_triplets(cfg: Dict[str, Any], run_dir: Path, max_views: int) -> List[Dict[str, Any]]:
    if cfg.get("image_paths"):
        image_paths = [Path(x) for x in cfg["image_paths"]]
    else:
        image_paths = collect_sorted_images(cfg["image_dir"])

    template_folder = Path(cfg["template_folder"])
    mask_folder = Path(cfg["mask_folder"])
    template_sorted = collect_sorted_images(template_folder)
    mask_sorted = collect_sorted_images(mask_folder, suffixes=(".png", ".jpg", ".jpeg", ".bmp"))

    view_start_index = int(cfg.get("view_start_index", 1))
    if max_views <= 0:
        max_views = len(image_paths)

    resize_enabled = bool(cfg.get("resize_images_for_vlm", True))
    max_side = int(cfg.get("vlm_image_max_side", 1280))
    quality = int(cfg.get("vlm_image_quality", 95))
    resized_dir = ensure_dir(run_dir / "vlm_resized_inputs")

    triplets: List[Dict[str, Any]] = []
    for local_i, img_path in enumerate(image_paths[:max_views], start=1):
        view_index = view_start_index + local_i - 1
        indexed_template = template_folder / f"image{view_index}.jpg"
        indexed_mask = mask_folder / f"image{view_index}_mask.png"
        template_path = indexed_template if indexed_template.exists() else template_sorted[local_i - 1]
        mask_path = indexed_mask if indexed_mask.exists() else mask_sorted[local_i - 1]
        if resize_enabled:
            img_use = resize_image_for_vlm(img_path, resized_dir / f"view{view_index:02d}_xray.jpg", max_side=max_side, quality=quality)
            tpl_use = resize_image_for_vlm(template_path, resized_dir / f"view{view_index:02d}_template.jpg", max_side=max_side, quality=quality)
            mask_use = resize_image_for_vlm(mask_path, resized_dir / f"view{view_index:02d}_mask.jpg", max_side=max_side, quality=quality)
        else:
            img_use, tpl_use, mask_use = str(img_path), str(template_path), str(mask_path)
        triplets.append({
            "view_index": view_index,
            "image_path": str(img_path),
            "template_path": str(template_path),
            "mask_path": str(mask_path),
            "vlm_image_path": img_use,
            "vlm_template_path": tpl_use,
            "vlm_mask_path": mask_use,
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
    # 避免官方模型把慢速速度错当高速速度。
    if "慢速速度" in params and "高速速度" in params:
        lines.append("注意：慢速速度与高速速度是两个不同字段，慢速速度不能当作高速速度使用。")
    return "\n".join(lines)


# ============================================================
# 提示词
# ============================================================

def build_single_view_visual_text(view: Dict[str, Any]) -> str:
    vi = view["view_index"]
    return f"""
你是一名铝合金压铸 X 光图缺陷识别助手。

当前只给出同一工件的第 {vi} 个视角，图像顺序固定为：
1. 第 {vi} 视角 X 光检测图；
2. 第 {vi} 视角模板参考图；
3. 第 {vi} 视角区域蒙版图。

任务说明：
1. X 光检测图中白色高亮、亮斑或异常亮白区域优先视为气孔候选。
2. 模板参考图只用于理解当前视角和结构，不是检测结果。
3. 区域蒙版图用于估计气孔候选落在哪个工艺区域。
4. 区域编号：1=一般区域，2=牙孔区域，3=焊接区域，4=重点区域。
5. 0、5、6 或无法判断区域不计入有效缺陷区域。
6. 缺陷类型统一按气孔处理。
7. 你不能声称使用 YOLO、SAHI、模板匹配、知识图谱或多模态 token。
8. 如果看不清，就写“不确定”，但必须给出基于图像的估计。
9. 面积和等效直径无法从图片精确标定，只能输出 0-5 相对等级：0=未观察到，5=最突出。

请严格输出合法 JSON，不要写 markdown，不要写解释文本。JSON 格式如下：
{{
  "view_index": {vi},
  "defect_type": "气孔",
  "visible_defect": "是/否/不确定",
  "estimated_total_count": 0,
  "region_count": {{
    "一般区域": 0,
    "牙孔区域": 0,
    "焊接区域": 0,
    "重点区域": 0
  }},
  "region_max_area_level": {{
    "一般区域": 0,
    "牙孔区域": 0,
    "焊接区域": 0,
    "重点区域": 0
  }},
  "region_max_diameter_level": {{
    "一般区域": 0,
    "牙孔区域": 0,
    "焊接区域": 0,
    "重点区域": 0
  }},
  "candidate_description": "描述气孔候选的可见位置、亮度、数量和不确定性",
  "region_judgement": "描述候选大致属于哪个区域；无法判断就写无法判断",
  "confidence": 0.0,
  "uncertainty": "说明看不清、区域难对齐、缺陷太小等不确定性"
}}
""".strip()


def build_qwen_visual_messages(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_single_view_visual_text(view)},
        {"type": "text", "text": f"【第{view['view_index']}视角-X光检测图】"},
        {"type": "image", "image": view["vlm_image_path"]},
        {"type": "text", "text": f"【第{view['view_index']}视角-模板参考图】"},
        {"type": "image", "image": view["vlm_template_path"]},
        {"type": "text", "text": f"【第{view['view_index']}视角-区域蒙版图】"},
        {"type": "image", "image": view["vlm_mask_path"]},
    ]
    return [{"role": "user", "content": content}]


def build_pipeline_visual_messages(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_single_view_visual_text(view)},
        {"type": "text", "text": f"【第{view['view_index']}视角-X光检测图】"},
        {"type": "image", "url": view["vlm_image_path"]},
        {"type": "text", "text": f"【第{view['view_index']}视角-模板参考图】"},
        {"type": "image", "url": view["vlm_template_path"]},
        {"type": "text", "text": f"【第{view['view_index']}视角-区域蒙版图】"},
        {"type": "image", "url": view["vlm_mask_path"]},
    ]
    return [{"role": "user", "content": content}]


def build_minicpm_visual_prompt_and_images(view: Dict[str, Any]) -> Tuple[str, List[Image.Image]]:
    prompt = build_single_view_visual_text(view)
    images = [
        image_open(view["vlm_image_path"]),
        image_open(view["vlm_template_path"]),
        image_open(view["vlm_mask_path"]),
    ]
    prompt += f"\n\n接下来三张图依次为：第{view['view_index']}视角X光检测图、模板参考图、区域蒙版图。"
    return prompt, images


def aggregate_visual_views(model_key: str, model_name: str, view_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    region_count = {r: 0 for r in ALLOWED_REGION_NAMES}
    region_area = {r: 0.0 for r in ALLOWED_REGION_NAMES}
    region_dia = {r: 0.0 for r in ALLOWED_REGION_NAMES}
    total = 0
    view_summaries = []

    for vr in view_results:
        parsed = vr.get("parsed", {}) or {}
        cnt = normalize_region_dict(parsed.get("region_count"), "int")
        area = normalize_region_dict(parsed.get("region_max_area_level") or parsed.get("region_max_area_mm2"), "float")
        dia = normalize_region_dict(parsed.get("region_max_diameter_level") or parsed.get("region_max_eq_diameter_mm"), "float")
        est_total = to_int(parsed.get("estimated_total_count", parsed.get("total_defect_count")), sum(cnt.values()))
        if est_total <= 0:
            est_total = sum(int(v) for v in cnt.values())
        total += est_total
        for r in ALLOWED_REGION_NAMES:
            region_count[r] += int(cnt.get(r, 0) or 0)
            region_area[r] = max(region_area[r], float(area.get(r, 0.0) or 0.0))
            region_dia[r] = max(region_dia[r], float(dia.get(r, 0.0) or 0.0))
        view_summaries.append({
            "view_index": parsed.get("view_index", vr.get("view_index")),
            "visible_defect": parsed.get("visible_defect", ""),
            "estimated_total_count": est_total,
            "region_count": cnt,
            "candidate_description": parsed.get("candidate_description", ""),
            "region_judgement": parsed.get("region_judgement", ""),
            "confidence": to_number(parsed.get("confidence"), 0.0),
            "uncertainty": parsed.get("uncertainty", ""),
            "raw_text_path": vr.get("raw_text_path", ""),
        })

    qlead = compute_max_regions(region_count)
    risk = sorted(set(compute_max_regions(region_area) + compute_max_regions(region_dia) + [r for r in qlead if r in HIGH_IMPACT_REGIONS]))
    summary = {
        "defect_type": "气孔",
        "source": f"{model_name} per-view visual estimation",
        "model_key": model_key,
        "metric_note": "面积和等效直径为官方VLM逐视角图像估计等级，不是标定后的真实物理量",
        "total_defect_count": int(total),
        "total_area_mm2": round(sum(region_area.values()), 3),
        "max_eq_diameter_mm": round(max(region_dia.values()) if region_dia else 0.0, 3),
        "region_count": region_count,
        "region_max_area_mm2": {k: round(float(v), 3) for k, v in region_area.items()},
        "region_max_eq_diameter_mm": {k: round(float(v), 3) for k, v in region_dia.items()},
        "quantity_leading_regions": qlead,
        "risk_focus_regions": risk,
        "view_summaries": view_summaries,
        "aggregation_note": "该视觉总结由官方VLM逐视角判断结果聚合得到，未使用本项目YOLO/SAHI/SIFT视觉层。",
    }
    text_lines = [
        f"1. 气孔候选总体判断：逐视角估计有效气孔候选合计约 {summary['total_defect_count']} 处。",
        "2. 各区域气孔数量估计：" + "、".join([f"{k}{v}处" for k, v in region_count.items()]) + "。",
        "3. 各区域最大气孔相对大小判断：面积等级 " + json.dumps(summary["region_max_area_mm2"], ensure_ascii=False) + "；等效直径等级 " + json.dumps(summary["region_max_eq_diameter_mm"], ensure_ascii=False) + "。",
        "4. 高影响区域风险判断：" + ("、".join(risk) if risk else "未能从官方VLM视觉判断中形成明确高影响区域风险") + "。",
        "5. 视觉判断不确定性：该结果来自通用VLM逐视角估计，需与人工标注或本项目结构化视觉层对照验证。",
    ]
    return summary, "\n".join(text_lines)


TRACE_PROMPT_TEMPLATE = """
你是一名铝合金压铸气孔缺陷溯源助手。
你只能依据当前提供的官方多模态模型逐视角视觉总结、压铸参数和用户问题作答。
不要声称使用了外部视觉检测算法、YOLO、SAHI、模板匹配、知识图谱或多模态 token。

【官方VLM逐视角视觉层总结】
{visual_summary_text}

【逐视角聚合后的结构化估计】
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


def build_trace_prompt(question: str, params_text: str, visual_text: str, visual_summary: Dict[str, Any]) -> str:
    return TRACE_PROMPT_TEMPLATE.format(
        visual_summary_text=visual_text.strip(),
        visual_summary_json=json.dumps(visual_summary, ensure_ascii=False, indent=2),
        params_text=params_text.strip(),
        question=question.strip(),
    )


def build_trace_messages_text_only(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


# ============================================================
# JSON/答案解析
# ============================================================

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
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
    out: Dict[str, Any] = {r: 0 for r in ALLOWED_REGION_NAMES}
    if not isinstance(raw, dict):
        return out
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


def normalize_answer(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    m = re.search(r"【结论】", text)
    if m:
        text = text[m.start():].strip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ============================================================
# 后端适配器
# ============================================================

class BaseBackend:
    def __init__(self, model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.global_cfg = global_cfg
        self.model_path = resolve_model_path(model_cfg, global_cfg)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.pipe = None

    @property
    def key(self) -> str:
        return str(self.model_cfg["key"])

    @property
    def name(self) -> str:
        return str(self.model_cfg.get("name") or self.key)

    def load(self) -> None:
        raise NotImplementedError

    def generate_visual_view(self, view: Dict[str, Any], max_new_tokens: int) -> str:
        raise NotImplementedError

    def generate_trace(self, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in ["pipe", "model", "processor", "tokenizer"]:
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    del obj
                    setattr(self, attr, None)
            except Exception:
                pass
        clear_cuda()


class Qwen25VLBackend(BaseBackend):
    def load(self) -> None:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.process_vision_info = process_vision_info
        dtype = choose_dtype(self.model_cfg.get("torch_dtype", self.global_cfg.get("torch_dtype", "auto")))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path), torch_dtype=dtype,
            device_map=self.model_cfg.get("device_map", self.global_cfg.get("device_map", "auto")),
            local_files_only=bool(self.model_cfg.get("local_files_only", self.global_cfg.get("local_files_only", True))),
            trust_remote_code=True,
        )
        self.model.eval()
        min_pixels = int(self.model_cfg.get("min_pixels", self.global_cfg.get("min_pixels", 256 * 28 * 28)))
        max_pixels = int(self.model_cfg.get("max_pixels", self.global_cfg.get("max_pixels", 1280 * 28 * 28)))
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), min_pixels=min_pixels, max_pixels=max_pixels, local_files_only=True, trust_remote_code=True)

    @torch.no_grad()
    def _generate(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(get_first_device(self.model))
        generated = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1, use_cache=True)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def generate_visual_view(self, view: Dict[str, Any], max_new_tokens: int) -> str:
        return self._generate(build_qwen_visual_messages(view), max_new_tokens)

    def generate_trace(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate(build_trace_messages_text_only(prompt), max_new_tokens)


class Qwen3VLBackend(BaseBackend):
    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        dtype = choose_dtype(self.model_cfg.get("torch_dtype", self.global_cfg.get("torch_dtype", "auto")))
        kwargs = dict(
            device_map=self.model_cfg.get("device_map", self.global_cfg.get("device_map", "auto")),
            local_files_only=bool(self.model_cfg.get("local_files_only", self.global_cfg.get("local_files_only", True))),
            trust_remote_code=True,
        )
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path), dtype=dtype, **kwargs)
        except TypeError:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path), torch_dtype=dtype, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)

    @torch.no_grad()
    def _generate(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        try:
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(get_first_device(self.model))
        except Exception:
            from qwen_vl_utils import process_vision_info
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(get_first_device(self.model))
        generated = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1, use_cache=True)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else inputs.input_ids
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def generate_visual_view(self, view: Dict[str, Any], max_new_tokens: int) -> str:
        return self._generate(build_qwen_visual_messages(view), max_new_tokens)

    def generate_trace(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate(build_trace_messages_text_only(prompt), max_new_tokens)


class InternVLBackend(BaseBackend):
    """InternVL3/3.5 HF trust_remote_code 原生 chat，避免 pipeline 重复传 trust_remote_code。"""
    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        dtype = choose_dtype(self.model_cfg.get("torch_dtype", self.global_cfg.get("torch_dtype", "bf16")))
        dtype = torch.bfloat16 if dtype == "auto" else dtype
        self.input_size = int(self.model_cfg.get("input_size", 448))
        self.max_tiles_per_image = int(self.model_cfg.get("max_tiles_per_image", 6))
        kwargs = dict(
            trust_remote_code=True,
            local_files_only=bool(self.model_cfg.get("local_files_only", self.global_cfg.get("local_files_only", True))),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        device_map = self.model_cfg.get("device_map", self.global_cfg.get("device_map", "auto"))
        if device_map:
            kwargs["device_map"] = device_map
        try:
            self.model = AutoModel.from_pretrained(str(self.model_path), use_flash_attn=False, **kwargs)
        except TypeError:
            self.model = AutoModel.from_pretrained(str(self.model_path), **kwargs)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True, use_fast=False, local_files_only=kwargs["local_files_only"])
        self.dtype = dtype

    def _build_transform(self, input_size: int):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image: Image.Image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_image_tensor(self, image_file: str) -> torch.Tensor:
        transform = self._build_transform(self.input_size)
        image = Image.open(image_file).convert('RGB')
        images = self._dynamic_preprocess(image, image_size=self.input_size, max_num=self.max_tiles_per_image, use_thumbnail=True)
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)

    @torch.no_grad()
    def _chat_with_images(self, prompt: str, image_paths: Optional[List[str]], max_new_tokens: int) -> str:
        generation_config = dict(max_new_tokens=int(max_new_tokens), do_sample=False)
        if image_paths:
            tensors = []
            num_patches_list = []
            for p in image_paths:
                pv = self._load_image_tensor(p)
                tensors.append(pv)
                num_patches_list.append(pv.size(0))
            pixel_values = torch.cat(tensors, dim=0).to(dtype=self.dtype, device=get_first_device(self.model))
            image_prefix = "".join([f"Image-{i+1}: <image>\n" for i in range(len(image_paths))])
            question = image_prefix + prompt
            attempts = [
                lambda: self.model.chat(self.tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list),
                lambda: self.model.chat(self.tokenizer, pixel_values, question, generation_config),
            ]
        else:
            attempts = [
                lambda: self.model.chat(self.tokenizer, None, prompt, generation_config),
                lambda: self.model.chat(self.tokenizer, prompt, generation_config),
            ]
        last_err = None
        for fn in attempts:
            try:
                res = fn()
                if isinstance(res, tuple):
                    res = res[0]
                return str(res).strip()
            except Exception as e:
                last_err = e
        raise RuntimeError(f"InternVL chat 调用失败: {last_err}")

    def generate_visual_view(self, view: Dict[str, Any], max_new_tokens: int) -> str:
        prompt = build_single_view_visual_text(view)
        imgs = [view["vlm_image_path"], view["vlm_template_path"], view["vlm_mask_path"]]
        return self._chat_with_images(prompt, imgs, max_new_tokens)

    def generate_trace(self, prompt: str, max_new_tokens: int) -> str:
        return self._chat_with_images(prompt, None, max_new_tokens)


class MiniCPMVBackend(BaseBackend):
    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        dtype = choose_dtype(self.model_cfg.get("torch_dtype", self.global_cfg.get("torch_dtype", "bf16")))
        dtype = torch.bfloat16 if dtype == "auto" else dtype
        kwargs = dict(trust_remote_code=True, local_files_only=bool(self.model_cfg.get("local_files_only", self.global_cfg.get("local_files_only", True))))
        try:
            self.model = AutoModel.from_pretrained(str(self.model_path), torch_dtype=dtype, device_map=self.model_cfg.get("device_map", self.global_cfg.get("device_map", "auto")), **kwargs)
        except TypeError:
            self.model = AutoModel.from_pretrained(str(self.model_path), torch_dtype=dtype, **kwargs)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True, local_files_only=kwargs["local_files_only"])

    def _chat(self, prompt: str, images: Optional[List[Image.Image]], max_new_tokens: int) -> str:
        msgs = [{"role": "user", "content": (images or []) + [prompt]}]
        generation_config = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        attempts = [
            lambda: self.model.chat(msgs=msgs, tokenizer=self.tokenizer, sampling=False, stream=False, generation_config=generation_config),
            lambda: self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, sampling=False, stream=False, generation_config=generation_config),
            lambda: self.model.chat(image=images, msgs=[{"role": "user", "content": prompt}], tokenizer=self.tokenizer, sampling=False, stream=False, generation_config=generation_config),
            lambda: self.model.chat(self.tokenizer, msgs, sampling=False, generation_config=generation_config),
        ]
        last_err = None
        for fn in attempts:
            try:
                res = fn()
                if isinstance(res, tuple):
                    res = res[0]
                return str(res).strip()
            except Exception as e:
                last_err = e
        raise RuntimeError(f"MiniCPM-V chat 调用失败: {last_err}")

    def generate_visual_view(self, view: Dict[str, Any], max_new_tokens: int) -> str:
        prompt, images = build_minicpm_visual_prompt_and_images(view)
        return self._chat(prompt, images, max_new_tokens)

    def generate_trace(self, prompt: str, max_new_tokens: int) -> str:
        return self._chat(prompt, None, max_new_tokens)


class LlavaOneVisionBackend(BaseBackend):
    """LLaVA-OneVision 原生 Processor + Model，配置里路径不存在时会自动按 llava/onevision 关键词查找。"""
    def load(self) -> None:
        from transformers import AutoProcessor
        dtype = choose_dtype(self.model_cfg.get("torch_dtype", self.global_cfg.get("torch_dtype", "fp16")))
        dtype = torch.float16 if dtype == "auto" else dtype
        local_files_only = bool(self.model_cfg.get("local_files_only", self.global_cfg.get("local_files_only", True)))
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=local_files_only, trust_remote_code=True)
        # 依次尝试新版/旧版类，避免环境差异导致启动失败。
        last_err = None
        for class_name in ["LlavaOnevisionForConditionalGeneration", "AutoModelForVision2Seq", "AutoModelForImageTextToText"]:
            try:
                mod = __import__("transformers", fromlist=[class_name])
                cls = getattr(mod, class_name)
                self.model = cls.from_pretrained(
                    str(self.model_path),
                    torch_dtype=dtype,
                    device_map=self.model_cfg.get("device_map", self.global_cfg.get("device_map", "auto")),
                    local_files_only=local_files_only,
                    trust_remote_code=True,
                )
                break
            except Exception as e:
                last_err = e
                self.model = None
        if self.model is None:
            raise RuntimeError(f"LLaVA-OneVision 模型加载失败: {last_err}")
        self.model.eval()

    @torch.no_grad()
    def _generate_with_images(self, prompt: str, images: Optional[List[Image.Image]], max_new_tokens: int) -> str:
        images = images or []
        # LLaVA OneVision 的 chat template 通常用多个 image 占位符。
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images if images else None, text=text, return_tensors="pt")
        inputs = inputs.to(get_first_device(self.model))
        output = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1)
        input_len = inputs["input_ids"].shape[-1]
        gen = output[:, input_len:]
        return self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def generate_visual_view(self, view: Dict[str, Any], max_new_tokens: int) -> str:
        prompt = build_single_view_visual_text(view) + f"\n\n接下来三张图依次为：第{view['view_index']}视角X光检测图、模板参考图、区域蒙版图。"
        images = [image_open(view["vlm_image_path"]), image_open(view["vlm_template_path"]), image_open(view["vlm_mask_path"])]
        return self._generate_with_images(prompt, images, max_new_tokens)

    def generate_trace(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate_with_images(prompt, None, max_new_tokens)


BACKEND_CLASSES = {
    "qwen25": Qwen25VLBackend,
    "qwen3": Qwen3VLBackend,
    "internvl": InternVLBackend,
    "minicpm": MiniCPMVBackend,
    "llava_onevision": LlavaOneVisionBackend,
}


def make_backend(model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> BaseBackend:
    backend_name = str(model_cfg.get("backend", ""))
    cls = BACKEND_CLASSES.get(backend_name)
    if cls is None:
        raise ValueError(f"未知 backend={backend_name}，可选: {list(BACKEND_CLASSES)}")
    return cls(model_cfg, global_cfg)


# ============================================================
# 运行逻辑
# ============================================================

def run_visual_per_view(backend: BaseBackend, model_cfg: Dict[str, Any], triplets: List[Dict[str, Any]], visual_dir: Path) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    ensure_dir(visual_dir)
    view_rows: List[Dict[str, Any]] = []
    max_tokens = int(model_cfg.get("max_new_tokens_visual_view", model_cfg.get("max_new_tokens_visual", 768)))
    for view in triplets:
        vi = int(view["view_index"])
        print(f"[visual][{backend.key}] view {vi} start", flush=True)
        vdir = ensure_dir(visual_dir / f"view_{vi:02d}")
        save_json(vdir / "input_triplet.json", view)
        save_text(vdir / "visual_prompt.txt", build_single_view_visual_text(view))
        raw = backend.generate_visual_view(view, max_tokens)
        raw_path = vdir / "visual_raw.txt"
        save_text(raw_path, raw)
        parsed = extract_first_json_object(raw) or {}
        # 补充/归一化关键信息。
        parsed.setdefault("view_index", vi)
        parsed["region_count"] = normalize_region_dict(parsed.get("region_count"), "int")
        parsed["region_max_area_level"] = normalize_region_dict(parsed.get("region_max_area_level") or parsed.get("region_max_area_mm2"), "float")
        parsed["region_max_diameter_level"] = normalize_region_dict(parsed.get("region_max_diameter_level") or parsed.get("region_max_eq_diameter_mm"), "float")
        parsed["estimated_total_count"] = to_int(parsed.get("estimated_total_count", parsed.get("total_defect_count")), sum(parsed["region_count"].values()))
        if parsed["estimated_total_count"] <= 0:
            parsed["estimated_total_count"] = sum(int(v) for v in parsed["region_count"].values())
        save_json(vdir / "visual_parsed.json", parsed)
        row = {
            "model_key": backend.key,
            "model_name": backend.name,
            "view_index": vi,
            "parsed": parsed,
            "raw": raw,
            "raw_text_path": str(raw_path),
            "parsed_json_path": str(vdir / "visual_parsed.json"),
            "triplet": view,
        }
        view_rows.append(row)
        print(f"[visual][{backend.key}] view {vi} parsed_count={parsed.get('estimated_total_count')} region_count={parsed.get('region_count')}", flush=True)
    summary, text = aggregate_visual_views(backend.key, backend.name, view_rows)
    save_json(visual_dir / "visual_views_all.json", view_rows)
    write_jsonl(visual_dir / "visual_views_all.jsonl", view_rows)
    save_json(visual_dir / "visual_summary_aggregate.json", summary)
    save_text(visual_dir / "visual_summary_aggregate.txt", text)
    return summary, text, view_rows


def run_one_model(model_cfg: Dict[str, Any], global_cfg: Dict[str, Any], triplets: List[Dict[str, Any]], model_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    ensure_dir(model_dir)
    model_key = str(model_cfg["key"])
    model_name = str(model_cfg.get("name", model_key))
    started = time.time()
    backend: Optional[BaseBackend] = None
    try:
        backend = make_backend(model_cfg, global_cfg)
        model_cfg["resolved_path"] = str(backend.model_path)
        save_json(model_dir / "model_config_resolved.json", model_cfg)
        print("\n" + "=" * 120)
        print(f"[model] START {model_key} | {model_name}")
        print(f"[model] path    = {backend.model_path}")
        print(f"[model] backend = {model_cfg.get('backend')}")
        print("=" * 120, flush=True)
        backend.load()

        visual_dir = ensure_dir(model_dir / "visual_layer")
        visual_summary, visual_text, view_rows = run_visual_per_view(backend, model_cfg, triplets, visual_dir)

        questions = global_cfg.get("questions", []) or []
        params = global_cfg.get("params", {}) or {}
        params_text = format_params(params)
        trace_dir = ensure_dir(model_dir / "trace_layer")
        mode_key = model_key
        mode_desc = f"{model_name} 逐视角读取X光图、模板图、蒙版图，自行形成视觉总结后进行气孔溯源"

        cases: List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []
        trace_summary_rows: List[Dict[str, Any]] = []
        max_trace_tokens = int(model_cfg.get("max_new_tokens_trace", global_cfg.get("max_new_tokens_trace", 1024)))
        for qi, question in enumerate(questions, start=1):
            print("\n" + "-" * 100)
            print(f"[trace][{model_key}] Q{qi:02d}/{len(questions)}", flush=True)
            q_dir = ensure_dir(trace_dir / f"q{qi:02d}")
            prompt = build_trace_prompt(question, params_text, visual_text, visual_summary)
            save_text(q_dir / "trace_prompt.txt", prompt)
            raw_answer = backend.generate_trace(prompt, max_trace_tokens)
            final_answer = normalize_answer(raw_answer)
            save_text(q_dir / "raw_answer.txt", raw_answer)
            save_text(q_dir / "final_answer.txt", final_answer)
            print(final_answer[:1200], flush=True)

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
            global_q_index = int(model_cfg.get("_model_order", 0)) * 1000 + qi
            case = {
                "question_index": global_q_index,
                "original_question_index": qi,
                "sample_id": global_cfg.get("sample_id", global_cfg.get("workpiece_id", "unknown")),
                "workpiece_id": global_cfg.get("workpiece_id", global_cfg.get("sample_id", "unknown")),
                "model_key": model_key,
                "model_name": model_name,
                "question": question,
                "question_type": "official_vlm_trace_per_view",
                "visual_summary": visual_summary,
                "vlm_visual_summary_text": visual_text,
                "params": params,
                "image_triplets": triplets,
                "modes": {mode_key: mode_result},
            }
            save_json(q_dir / "case_result.json", case)
            cases.append(case)
            row = {
                "model_key": model_key,
                "model_name": model_name,
                "question_index": qi,
                "global_question_index": global_q_index,
                "sample_id": case["sample_id"],
                "workpiece_id": case["workpiece_id"],
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
            trace_rows.append(row)
            trace_summary_rows.append({
                "model_key": model_key,
                "model_name": model_name,
                "question_index": qi,
                "global_question_index": global_q_index,
                "sample_id": case["sample_id"],
                "mode": mode_key,
                "question": question,
                "answer_chars": len(final_answer),
                "starts_with_conclusion": str(final_answer.startswith("【结论】")),
                "answer_file": str(q_dir / "final_answer.txt"),
            })

        save_json(model_dir / "model_all_modes_results.json", {
            "schema_version": "official_vlm_single_model_per_view_v1",
            "model_key": model_key,
            "model_name": model_name,
            "model_path": str(backend.model_path),
            "visual_summary": visual_summary,
            "visual_text": visual_text,
            "visual_view_rows": view_rows,
            "cases": cases,
        })
        write_jsonl(trace_dir / "trace_answers.jsonl", trace_rows)
        write_csv(trace_dir / "trace_summary.csv", trace_summary_rows, ["model_key", "model_name", "question_index", "global_question_index", "sample_id", "mode", "question", "answer_chars", "starts_with_conclusion", "answer_file"])

        report = [f"# {model_name} 逐视角视觉判断与溯源测试\n\n"]
        report.append(f"- 模型路径：`{backend.model_path}`\n")
        report.append(f"- 后端：`{model_cfg.get('backend', '')}`\n")
        report.append(f"- 视角数量：{len(triplets)}\n\n")
        report.append("## 逐视角聚合视觉总结\n\n```json\n")
        report.append(json.dumps(visual_summary, ensure_ascii=False, indent=2))
        report.append("\n```\n\n")
        report.append(visual_text + "\n\n")
        for row in trace_rows:
            report.append(f"## Q{row['question_index']:02d}\n\n{row['question']}\n\n{row['final_answer']}\n\n")
        save_text(model_dir / "model_per_view_trace_report.md", "".join(report))

        status = {"model_key": model_key, "model_name": model_name, "status": "ok", "seconds": round(time.time() - started, 2), "case_count": len(cases), "model_dir": str(model_dir), "resolved_path": str(backend.model_path)}
        return cases, trace_rows, status, visual_summary

    except Exception as e:
        err = traceback.format_exc()
        save_text(model_dir / "ERROR.txt", err)
        print(f"[ERROR][{model_key}] {e}", flush=True)
        print(err, flush=True)
        status = {"model_key": model_key, "model_name": model_name, "status": "error", "seconds": round(time.time() - started, 2), "error": str(e), "error_file": str(model_dir / "ERROR.txt"), "model_dir": str(model_dir)}
        return [], [], status, {}
    finally:
        try:
            if backend is not None:
                backend.unload()
        except Exception:
            pass
        clear_cuda()


def select_models(cfg: Dict[str, Any], only_models: str = "") -> List[Dict[str, Any]]:
    models = [m for m in cfg.get("models", []) if m.get("enabled", True)]
    if only_models:
        want = [x.strip() for x in only_models.split(",") if x.strip()]
        models = [m for m in models if str(m.get("key")) in want]
    if not models:
        raise ValueError("没有可运行模型，请检查 config.models 或 --models")
    for i, m in enumerate(models, start=1):
        m["_model_order"] = i
    return models


def run_all(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.max_views is not None:
        cfg["max_views"] = int(args.max_views)
    if args.output_dir:
        cfg["result_output_dir"] = args.output_dir

    result_root = ensure_dir(cfg.get("result_output_dir", "./five_vlm_per_view_results"))
    run_dir = ensure_dir(result_root / f"five_vlm_per_view_run_{now_ts()}")
    save_json(run_dir / "runtime_config.json", cfg)

    max_views = int(cfg.get("max_views", 9))
    triplets = resolve_view_triplets(cfg, run_dir=run_dir, max_views=max_views)
    save_json(run_dir / "selected_views.json", triplets)

    models = select_models(cfg, args.models)
    print("=" * 120)
    print(f"[run] output_dir = {run_dir}")
    print(f"[run] models     = {[m['key'] for m in models]}")
    print(f"[run] views      = {len(triplets)} groups; per-view visual input = 3 images")
    print(f"[run] questions  = {len(cfg.get('questions', []))}")
    print("=" * 120, flush=True)

    all_cases: List[Dict[str, Any]] = []
    all_trace_rows: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []
    visual_summaries_by_model: Dict[str, Any] = {}

    for model_cfg in models:
        model_key = str(model_cfg["key"])
        model_dir = ensure_dir(run_dir / model_key)
        cases, rows, status, visual_summary = run_one_model(model_cfg, cfg, triplets, model_dir)
        statuses.append(status)
        all_cases.extend(cases)
        all_trace_rows.extend(rows)
        if visual_summary:
            visual_summaries_by_model[model_key] = visual_summary
        save_json(run_dir / "model_run_status_partial.json", statuses)
        save_json(run_dir / "visual_summaries_by_model_partial.json", visual_summaries_by_model)
        write_jsonl(run_dir / "trace_answers_all_partial.jsonl", all_trace_rows)
        if status["status"] == "error" and bool(cfg.get("stop_on_error", False)):
            break

    output_obj = {
        "schema_version": "official_vlm_five_models_per_view_seq_v1",
        "sample_id": cfg.get("sample_id"),
        "workpiece_id": cfg.get("workpiece_id"),
        "run_dir": str(run_dir),
        "selected_views": triplets,
        "visual_summaries_by_model": visual_summaries_by_model,
        "mode_definitions": {
            str(m["key"]): {
                "mode_desc": f"{m.get('name', m['key'])} 逐视角读取X光图、模板图、蒙版图，自行形成视觉总结后进行气孔溯源",
                "model_path": m.get("path", ""),
                "resolved_path": m.get("resolved_path", m.get("path", "")),
                "backend": m.get("backend", ""),
                "uses_user_visual_layer": False,
                "uses_yolo_sahi_sift": False,
            }
            for m in models
        },
        "model_run_status": statuses,
        "cases": all_cases,
    }
    save_json(run_dir / "all_modes_results.json", output_obj)
    save_json(run_dir / "all_models_results.json", output_obj)
    save_json(run_dir / "visual_summaries_by_model.json", visual_summaries_by_model)
    write_jsonl(run_dir / "trace_answers_all.jsonl", all_trace_rows)

    summary_rows = []
    for r in all_trace_rows:
        ans = str(r.get("final_answer", ""))
        summary_rows.append({
            "model_key": r.get("model_key", ""),
            "model_name": r.get("model_name", ""),
            "question_index": r.get("question_index", ""),
            "global_question_index": r.get("global_question_index", ""),
            "sample_id": r.get("sample_id", ""),
            "mode": r.get("mode", ""),
            "question": r.get("question", ""),
            "answer_chars": len(ans),
            "starts_with_conclusion": str(ans.startswith("【结论】")),
            "answer_file": r.get("answer_path", ""),
        })
    write_csv(run_dir / "trace_summary_all.csv", summary_rows, ["model_key", "model_name", "question_index", "global_question_index", "sample_id", "mode", "question", "answer_chars", "starts_with_conclusion", "answer_file"])
    write_csv(run_dir / "model_run_summary.csv", statuses, ["model_key", "model_name", "status", "seconds", "case_count", "error", "resolved_path", "model_dir", "error_file"])

    report = ["# 五个官方多模态大模型逐视角视觉判断 + 顺序溯源测试\n\n"]
    report.append(f"- 样本ID：`{cfg.get('sample_id', '')}`\n")
    report.append(f"- 工件ID：`{cfg.get('workpiece_id', '')}`\n")
    report.append(f"- 视角组数：{len(triplets)}\n")
    report.append(f"- 视觉阶段：逐视角输入，每次 1 张X光 + 1 张模板 + 1 张蒙版。\n")
    report.append(f"- 问题数量：{len(cfg.get('questions', []))}\n")
    report.append(f"- 溯源评分输入：`{run_dir / 'all_modes_results.json'}`\n")
    report.append(f"- 视觉验证输入：`{run_dir / 'visual_summaries_by_model.json'}`\n\n")
    report.append("## 模型运行状态\n\n| 模型 | 状态 | 耗时(s) | 解析路径 | 输出目录 |\n|---|---|---:|---|---|\n")
    for s in statuses:
        report.append(f"| {s.get('model_key')} | {s.get('status')} | {s.get('seconds')} | `{s.get('resolved_path','')}` | `{s.get('model_dir')}` |\n")
    report.append("\n")
    for model_key, summary in visual_summaries_by_model.items():
        report.append(f"## {model_key} 视觉总结\n\n```json\n")
        report.append(json.dumps(summary, ensure_ascii=False, indent=2))
        report.append("\n```\n\n")
    save_text(run_dir / "five_vlm_per_view_seq_report.md", "".join(report))

    print("\n" + "=" * 120)
    print("[done] 五模型逐视角视觉判断 + 顺序溯源测试完成")
    print(f"[done] result_dir          : {run_dir}")
    print(f"[done] trace scoring input : {run_dir / 'all_modes_results.json'}")
    print(f"[done] visual summaries    : {run_dir / 'visual_summaries_by_model.json'}")
    print(f"[done] status csv          : {run_dir / 'model_run_summary.csv'}")
    print(f"[done] report              : {run_dir / 'five_vlm_per_view_seq_report.md'}")
    print("=" * 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="五个官方VLM逐视角视觉判断 + 顺序溯源测试")
    parser.add_argument("--config", type=str, required=True, help="配置 JSON 路径")
    parser.add_argument("--models", type=str, default="", help="只跑指定模型key，逗号分隔")
    parser.add_argument("--max-views", type=int, default=None, help="覆盖配置 max_views；正式用 9，调试用 1")
    parser.add_argument("--output-dir", type=str, default="", help="覆盖结果输出目录")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    cfg = load_json(args.config)
    run_all(cfg, args)


if __name__ == "__main__":
    main()
