#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-Plus：LabelMe标注示例 + 九视角 mask_vis 一次视觉识别 + 图谱文件辅助 + 300问题批量溯源测试
================================================================================

用途：
1. 图片、mask_vis、模板、参数只属于同一个工件样本；
2. 第一次只调用 qwen3.5-plus 读取 9 张合成图，生成一次视觉层结构化总结；
3. 后续 300 个问题全部复用这份视觉总结，不再上传图片，避免重复 image tokens；
4. 读取 LabelMe 标注样本，生成缺陷外观参考图，只在视觉识别阶段输入一次；
5. 读取 Neo4j 导出的 JSON 节点/关系文件，在本地做轻量检索，每个问题只给少量相关图谱证据；
6. 输出 all_modes_results.json，结构兼容 standard_e2e_scoring_v2.py：cases -> modes -> final_answer。

注意：
- 本脚本不调用你的 YOLO/SAHI/SIFT 视觉层；
- 不连接 Neo4j 服务，只读取导出的 JSON 文件；
- 图谱证据只作为机理辅助，不覆盖当前视觉统计和压铸参数；
- LabelMe 标注只作为气孔外观参考，不作为当前工件统计事实；
- API Key 从环境变量 DASHSCOPE_API_KEY 读取，不写入代码。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

# =========================
# 默认路径与参数
# =========================

DEFAULT_IMAGE_DIR = "/home/T7/wkq/end-to-end/diecast_region_route/images/47243"
DEFAULT_TEMPLATE_DIR = "/home/T7/wkq/end-to-end/yolo-X-data/11193 NG"
DEFAULT_MASK_VIS_DIR = "/home/T7/wkq/end-to-end/yolo-X-data/masks_vis"
DEFAULT_LABELME_DIR = "/home/T7/wkq/end-to-end/API_test/labelme"

DEFAULT_QUESTIONS_JSON = "/home/T7/wkq/end-to-end/API_test/Pasted code.json"
DEFAULT_GRAPH_NODES_JSON = "/home/T7/wkq/end-to-end/API_test/neo4j_query_table_data_2026-4-17.json"
DEFAULT_GRAPH_EDGES_JSON = "/home/T7/wkq/end-to-end/API_test/neo4j_连接节点_2026-4-17.json"

DEFAULT_OUTPUT_ROOT = "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q"

DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

DEFAULT_SAMPLE_ID = "38069"
DEFAULT_WORKPIECE_ID = "38069"
DEFAULT_MODE_KEY = "qwen35_plus_labelme_graph_api"

DEFAULT_PARAMS = {
    "慢速速度": 0.19,
    "高速速度": 3.65,
    "料饼厚度": 28,
    "铸造压力": 926,
    "循环周期": 138.1,
    "建压时间": 114,
    "射料结束位置": 748,
    "压射压力": 316,
    "高速行程": 205,
    "高速时间": 46,
    "慢速行程": 525,
    "环境温度": 23.4,
    "环境湿度": 47.1,
    "探伤图编号": 38053,
    "样本编号": 38053,
}

REGION_MAP = {
    "1": "一般区域",
    "2": "牙孔区域",
    "3": "焊接区域",
    "4": "重点区域",
    "5": "重叠区域",
    "6": "背景",
    "0": "背景或未标注区域",
}
VALID_REGIONS = ["一般区域", "牙孔区域", "焊接区域", "重点区域"]
HIGH_IMPACT_REGIONS = ["牙孔区域", "焊接区域", "重点区域"]

# 标定参数
DEFAULT_IMAGE_HEIGHT_PIXELS = 3080
DEFAULT_CALIBRATION_LENGTH_MM = 249.89
DEFAULT_PIXEL_SIZE_MM_PER_PIXEL = 0.032

FORBIDDEN_DEFECT_TERMS = ["缩孔", "缩松", "冷隔", "裂纹", "夹杂", "夹渣", "热裂", "冷裂", "冷豆", "变形", "白点"]
FORBIDDEN_UNPROVIDED_TERMS = [
    "排气槽", "溢流井", "溢流槽", "内浇口", "喷涂积碳", "脱模剂", "熔炼含气量",
    "合金成分", "料温", "模温", "真空系统", "除气工艺", "熔炼除气", "热处理环节",
]

# =========================
# 基础工具
# =========================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text or "", encoding="utf-8")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def image_to_base64_data_url(image_path: str | Path, mime: str = "image/jpeg") -> str:
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def format_params(params: Dict[str, Any]) -> str:
    order = [
        "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
        "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
        "慢速行程", "环境温度", "环境湿度", "探伤图编号", "样本编号",
    ]
    lines = []
    used = set()
    for k in order:
        if k in params:
            lines.append(f"{k}={params[k]}")
            used.add(k)
    for k, v in params.items():
        if k not in used:
            lines.append(f"{k}={v}")
    return "\n".join(lines)


def load_params_from_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return dict(DEFAULT_PARAMS)
    obj = load_json(path)
    if isinstance(obj, dict) and "params" in obj and isinstance(obj["params"], dict):
        return obj["params"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"参数JSON格式不支持: {path}")


def load_questions(path: str | Path) -> List[str]:
    obj = load_json(path)
    out: List[str] = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                q = item.strip()
            elif isinstance(item, dict):
                q = str(item.get("question") or item.get("content") or item.get("input") or "").strip()
            else:
                q = str(item).strip()
            if q:
                out.append(q)
    elif isinstance(obj, dict):
        for key in ["questions", "data", "items"]:
            if isinstance(obj.get(key), list):
                return load_questions_from_list(obj[key])
        q = str(obj.get("question") or "").strip()
        if q:
            out.append(q)
    if not out:
        raise ValueError(f"未从问题文件中读取到问题: {path}")
    return out


def load_questions_from_list(items: List[Any]) -> List[str]:
    out = []
    for item in items:
        if isinstance(item, str):
            q = item.strip()
        elif isinstance(item, dict):
            q = str(item.get("question") or item.get("content") or item.get("input") or "").strip()
        else:
            q = str(item).strip()
        if q:
            out.append(q)
    return out

# =========================
# 图像合成
# =========================

def collect_xray_images(image_dir: str | Path, max_views: int = 9) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"X光图目录不存在: {image_dir}")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if len(files) < max_views:
        raise FileNotFoundError(f"X光图不足 {max_views} 张: {image_dir}, 当前 {len(files)} 张")
    return files[:max_views]


def resolve_template_path(template_dir: str | Path, view_index: int) -> Path:
    template_dir = Path(template_dir)
    candidates = [
        template_dir / f"image{view_index}.jpg",
        template_dir / f"image{view_index}.png",
        template_dir / f"image{view_index}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"第{view_index}视角模板图不存在: {candidates}")


def resolve_mask_vis_path(mask_vis_dir: str | Path, view_index: int) -> Path:
    mask_vis_dir = Path(mask_vis_dir)
    candidates = [
        mask_vis_dir / f"image{view_index}_vis.png",
        mask_vis_dir / f"image{view_index}_mask_vis.png",
        mask_vis_dir / f"image{view_index}.png",
        mask_vis_dir / f"image{view_index}_vis.jpg",
        mask_vis_dir / f"image{view_index}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"第{view_index}视角mask_vis不存在: {candidates}")


def resize_keep_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def paste_center(canvas: Image.Image, img: Image.Image, box: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x = x1 + (bw - img.size[0]) // 2
    y = y1 + (bh - img.size[1]) // 2
    canvas.paste(img, (x, y))


def make_overlay(xray: Image.Image, mask_vis: Image.Image, size: Tuple[int, int]) -> Image.Image:
    x = xray.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    m = mask_vis.convert("RGB").resize(size, Image.Resampling.NEAREST)
    return Image.blend(x, m, alpha=0.42)


def get_font(size: int = 24) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_composite_view(xray_path: Path, template_path: Path, mask_vis_path: Path, out_path: Path, max_side: int = 760) -> Path:
    xray = Image.open(xray_path).convert("RGB")
    mask_vis = Image.open(mask_vis_path).convert("RGB")
    template = Image.open(template_path).convert("RGB")

    panel_w = max_side
    panel_h = int(max_side * 0.72)
    header_h = 60
    legend_h = 120
    gap = 20
    canvas_w = panel_w * 2 + gap * 3
    canvas_h = header_h + panel_h * 2 + legend_h + gap * 4
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = get_font(28)
    font = get_font(22)

    draw.text((gap, 14), f"VIEW {out_path.stem.replace('view_', '')}: X光图 / mask_vis / overlay / 模板图", fill=(0, 0, 0), font=font_title)
    boxes = [
        (gap, header_h + gap, gap + panel_w, header_h + gap + panel_h),
        (gap * 2 + panel_w, header_h + gap, gap * 2 + panel_w * 2, header_h + gap + panel_h),
        (gap, header_h + gap * 2 + panel_h, gap + panel_w, header_h + gap * 2 + panel_h * 2),
        (gap * 2 + panel_w, header_h + gap * 2 + panel_h, gap * 2 + panel_w * 2, header_h + gap * 2 + panel_h * 2),
    ]
    labels = ["A X光检测图：偏白/灰白圆点、亮斑优先视为气孔候选", "B mask_vis：数字编号为区域依据", "C overlay：优先判断白点落在哪个编号区域", "D 模板参考图：仅辅助理解视角"]
    imgs = [xray, mask_vis, make_overlay(xray, mask_vis, (panel_w, panel_h)), template]
    for box, lab, im in zip(boxes, labels, imgs):
        draw.rectangle(box, outline=(0, 0, 0), width=2)
        resized = resize_keep_ratio(im, panel_w - 8, panel_h - 40)
        paste_center(canvas, resized, (box[0] + 4, box[1] + 36, box[2] - 4, box[3] - 4))
        draw.text((box[0] + 8, box[1] + 6), lab, fill=(0, 0, 0), font=font)

    y = header_h + gap * 3 + panel_h * 2
    legend_lines = [
        "区域编号：1=一般区域，2=牙孔区域，3=焊接区域，4=重点区域，5=重叠区域(忽略)，6=背景(忽略)，0=背景/未标注(忽略)。",
        "识别规则：有效区域1/2/3/4内的偏白色、浅白色、灰白色圆点、亮点、亮斑、亮团，尤其孤立近圆形/椭圆形，应列为气孔候选；",
        "但结构边缘、零件轮廓、mask数字、彩色区域边界和模板线条不能当成气孔。无法确定时列为疑似并降低confidence。",
    ]
    for i, line in enumerate(legend_lines):
        draw.text((gap, y + i * 34), line, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JPEG压缩以降低图像token，但保留较高质量。
    canvas.save(out_path, quality=88)
    return out_path


def build_composites(args: argparse.Namespace, run_dir: Path) -> List[Dict[str, Any]]:
    image_paths = collect_xray_images(args.image_dir, args.max_views)
    out_dir = ensure_dir(run_dir / "composite_inputs")
    triplets: List[Dict[str, Any]] = []
    for i, xray in enumerate(image_paths, start=1):
        tpl = resolve_template_path(args.template_dir, i)
        mask = resolve_mask_vis_path(args.mask_vis_dir, i)
        comp = out_dir / f"view_{i:02d}_composite.jpg"
        make_composite_view(xray, tpl, mask, comp, max_side=args.composite_panel_size)
        triplets.append({
            "view_index": i,
            "image_path": str(xray),
            "template_path": str(tpl),
            "mask_vis_path": str(mask),
            "composite_path": str(comp),
            "image_name": xray.name,
        })
    return triplets



def _normalize_stem_for_labelme(p: Path) -> str:
    return p.stem.strip()


def collect_labelme_pairs(labelme_dir: str | Path, max_examples: int = 4) -> List[Dict[str, Any]]:
    """读取 LabelMe 图片+json 配对。没有目录或没有配对时返回空列表，不中断主流程。"""
    root = Path(labelme_dir)
    if not root.exists():
        return []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = { _normalize_stem_for_labelme(p): p for p in sorted(root.iterdir()) if p.suffix.lower() in image_exts }
    jsons = { _normalize_stem_for_labelme(p): p for p in sorted(root.iterdir()) if p.suffix.lower() == ".json" }
    pairs = []
    for stem in sorted(set(images) & set(jsons)):
        if len(pairs) >= max_examples:
            break
        pairs.append({"stem": stem, "image_path": str(images[stem]), "json_path": str(jsons[stem])})
    return pairs


def labelme_shape_to_bbox(points: List[Any]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for pt in points or []:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                xs.append(float(pt[0])); ys.append(float(pt[1]))
            except Exception:
                pass
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def parse_labelme_json(json_path: str | Path) -> Dict[str, Any]:
    obj = load_json(json_path)
    shapes = obj.get("shapes") if isinstance(obj, dict) else []
    parsed = []
    label_count: Dict[str, int] = defaultdict(int)
    for i, sh in enumerate(shapes or [], start=1):
        if not isinstance(sh, dict):
            continue
        label = str(sh.get("label") or "缺陷").strip()
        points = sh.get("points") or []
        x1, y1, x2, y2 = labelme_shape_to_bbox(points)
        label_count[label] += 1
        parsed.append({
            "shape_id": i,
            "label": label,
            "shape_type": sh.get("shape_type") or "",
            "point_count": len(points) if isinstance(points, list) else 0,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return {
        "json_path": str(json_path),
        "imagePath": obj.get("imagePath", "") if isinstance(obj, dict) else "",
        "label_count": dict(label_count),
        "shapes": parsed,
    }


def draw_labelme_reference_image(image_path: str | Path, json_path: str | Path, out_path: str | Path, panel_side: int = 900) -> Path:
    """生成一张给 VLM 看的 LabelMe 标注参考图：左原图，右标注框/多边形。"""
    img = Image.open(image_path).convert("RGB")
    meta = parse_labelme_json(json_path)
    shapes_raw = (load_json(json_path).get("shapes") or []) if Path(json_path).exists() else []
    font_title = get_font(28)
    font = get_font(22)

    w, h = img.size
    scale = min(panel_side / max(w, 1), panel_side / max(h, 1))
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    left = img.resize((rw, rh), Image.Resampling.LANCZOS)
    right = left.copy()
    draw = ImageDraw.Draw(right)

    for idx, sh in enumerate(shapes_raw, start=1):
        if not isinstance(sh, dict):
            continue
        label = str(sh.get("label") or "气孔").strip()
        pts = []
        for pt in sh.get("points") or []:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts.append((float(pt[0]) * scale, float(pt[1]) * scale))
        if len(pts) >= 2:
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(255, 0, 0), width=4)
            x1 = min(x for x, y in pts); y1 = min(y for x, y in pts)
            x2 = max(x for x, y in pts); y2 = max(y for x, y in pts)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
            draw.text((x1, max(0, y1 - 28)), f"{idx}:{label}", fill=(255, 0, 0), font=font)

    header_h = 76
    gap = 22
    footer_h = 110
    canvas_w = rw * 2 + gap * 3
    canvas_h = header_h + rh + footer_h + gap * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)
    d.text((gap, 18), "LabelMe人工标注气孔外观参考：左原图，右标注位置", fill=(0, 0, 0), font=font_title)
    canvas.paste(left, (gap, header_h))
    canvas.paste(right, (gap * 2 + rw, header_h))
    d.rectangle((gap, header_h, gap + rw, header_h + rh), outline=(0,0,0), width=2)
    d.rectangle((gap * 2 + rw, header_h, gap * 2 + rw * 2, header_h + rh), outline=(0,0,0), width=2)
    d.text((gap, header_h + rh + gap), "用途：只帮助理解气孔/疑似气孔在X光图中的白色圆点、亮斑、团簇状外观。", fill=(0,0,0), font=font)
    d.text((gap, header_h + rh + gap + 34), "边界：这些标注不是当前目标工件统计结果，不能把标注数量迁移到当前样本。", fill=(0,0,0), font=font)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)
    return out


def build_labelme_reference(args: argparse.Namespace, run_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    pairs = collect_labelme_pairs(args.labelme_dir, args.max_label_examples)
    if not pairs and not getattr(args, "allow_no_labelme", False):
        raise FileNotFoundError(
            f"未在LabelMe目录找到图片+同名json配对: {args.labelme_dir}。"
            "这个实验组必须输入LabelMe缺陷标签示例；如果只是调试，可加 --allow-no-labelme。"
        )
    out_dir = ensure_dir(run_dir / "labelme_reference_inputs")
    refs = []
    for i, pair in enumerate(pairs, start=1):
        meta = parse_labelme_json(pair["json_path"])
        ref_img = out_dir / f"labelme_ref_{i:02d}.jpg"
        draw_labelme_reference_image(pair["image_path"], pair["json_path"], ref_img)
        label_count_text = "；".join([f"{k}:{v}" for k, v in (meta.get("label_count") or {}).items()]) or "未读取到标签"
        refs.append({
            "example_index": i,
            "stem": pair["stem"],
            "image_path": pair["image_path"],
            "json_path": pair["json_path"],
            "reference_image": str(ref_img),
            "label_count": meta.get("label_count", {}),
            "shapes": meta.get("shapes", []),
            "summary": f"示例{i} 标签统计：{label_count_text}",
        })

    if refs:
        lines = [
            "【LabelMe人工标注参考】",
            "以下标注样本只用于帮助识别气孔外观，不是当前目标工件统计事实。",
            "请学习这些标注中气孔/疑似气孔在X光图中的常见外观：偏白色或灰白色的近圆点、椭圆亮斑、局部亮团、团簇状白点。",
            "不能把LabelMe示例的数量、位置、区域直接迁移到当前九视角目标工件；当前统计只以目标工件九视角图像和mask_vis为准。",
        ]
        for r in refs:
            lines.append("- " + r["summary"])
    else:
        lines = [
            "【LabelMe人工标注参考】",
            "未读取到LabelMe图片/json配对。本次只根据目标工件九视角X光图、mask_vis和模板图进行视觉估计。",
        ]
    return refs, "\n".join(lines)

# =========================
# API 调用
# =========================

def api_call(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: int = 180,
    retries: int = 3,
    sleep_base: float = 2.0,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": False,
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1200]}")
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = sleep_base * attempt
                print(f"[api][warn] attempt {attempt}/{retries} failed: {e}; sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
            else:
                raise
    assert last_err is not None
    raise last_err


def extract_content(resp: Dict[str, Any]) -> str:
    try:
        msg = resp["choices"][0]["message"]
        return (msg.get("content") or "").strip()
    except Exception:
        return ""


def add_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, (int, float)):
            out[k] = out.get(k, 0) + v
        elif isinstance(v, dict):
            out[k] = add_usage(out.get(k, {}), v)
    return out

# =========================
# 视觉识别与解析
# =========================

def visual_prompt(labelme_reference_text: str = "") -> str:
    region_text = "；".join([f"{k}={v}" for k, v in REGION_MAP.items()])
    return f"""你是一名铝合金压铸件X光图气孔候选识别助手。

{labelme_reference_text}

当前输入还包含同一个目标工件的九个视角合成图。每张合成图包含：
A X光检测图；B mask_vis区域编号图；C X光与mask_vis半透明叠加图；D模板参考图。

请先利用LabelMe人工标注参考理解“气孔长什么样”，再对目标工件九视角图像做独立判断。

区域编号说明：{region_text}。
只有 1/2/3/4 是有效工艺区域；5、6、0 不计入有效缺陷统计。

视觉识别规则：
1. 在本项目X光图中，落在有效区域内的偏白色、浅白色、灰白色局部圆点、亮点、亮斑、亮团，尤其孤立近圆形/椭圆形白点，应优先作为气孔候选。
2. 牙孔区域、焊接区域、重点区域属于高影响区域。一般区域主要作为整体波动参考。
3. 不要把零件边缘、孔洞轮廓、加强筋边界、mask_vis数字、彩色区域边界、模板线条当成气孔。
4. 如果不确定但像气孔，请仍列入 defect_candidates，并把 confidence 降低到 0.35~0.60。
5. 必须估计每个候选的 area_px。面积不确定时给粗略像素面积，不要输出mm²或mm，物理量由程序换算。

你必须只输出一个合法JSON对象，不要输出markdown，不要输出```json，不要输出自然语言解释。
JSON格式：
{{
  "defect_type": "气孔",
  "defect_candidates": [
    {{
      "view_index": 1,
      "region_judgement": "牙孔区域",
      "center_x_px": 0,
      "center_y_px": 0,
      "bbox_width_px": 0,
      "bbox_height_px": 0,
      "area_px": 0,
      "equivalent_diameter_px": 0,
      "morphology": "近圆形亮点/椭圆形亮斑/团簇状亮点",
      "visible_defect_description": "简短描述，不超过30字",
      "confidence": 0.0
    }}
  ],
  "uncertainty_note": "简短说明"
}}
""".strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # 优先直接解析
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 提取最大花括号块
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        block = cleaned[start:end + 1]
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return salvage_candidates(text)


def salvage_candidates(text: str) -> Dict[str, Any]:
    """当JSON被截断时，尽量从defect_candidates中抢救完整候选对象。"""
    cands: List[Dict[str, Any]] = []
    # 找每个候选对象中必要字段；容忍不完整外层JSON。
    # 先按大括号粗提取片段。
    fragments = re.findall(r"\{[^{}]*\"view_index\"[^{}]*?\}", text, flags=re.S)
    for frag in fragments:
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict) and "view_index" in obj:
                cands.append(obj)
        except Exception:
            pass
    if not cands:
        return {}
    return {"defect_type": "气孔", "defect_candidates": cands, "uncertainty_note": "JSON截断后从候选片段抢救解析。"}


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_region_name(x: Any) -> str:
    s = str(x or "").strip()
    alias = {
        "1": "一般区域", "Region 1": "一般区域", "region 1": "一般区域",
        "2": "牙孔区域", "Region 2": "牙孔区域", "region 2": "牙孔区域",
        "3": "焊接区域", "Region 3": "焊接区域", "region 3": "焊接区域",
        "4": "重点区域", "Region 4": "重点区域", "region 4": "重点区域",
    }
    if s in alias:
        return alias[s]
    for r in VALID_REGIONS:
        if r in s:
            return r
    return "无法判断"


def summarize_visual_json(
    raw_obj: Dict[str, Any],
    pixel_size: float,
    image_height_pixels: int,
    calibration_length_mm: float,
    confidence_keep_threshold: float = 0.35,
) -> Dict[str, Any]:
    raw_candidates = raw_obj.get("defect_candidates") or []
    candidates: List[Dict[str, Any]] = []
    region_count = {r: 0 for r in VALID_REGIONS}
    region_area = {r: 0.0 for r in VALID_REGIONS}
    region_dia = {r: 0.0 for r in VALID_REGIONS}
    total_area = 0.0
    max_dia = 0.0

    for idx, c in enumerate(raw_candidates, start=1):
        if not isinstance(c, dict):
            continue
        region = normalize_region_name(c.get("region_judgement"))
        conf = safe_float(c.get("confidence"), 0.0)
        area_px = safe_float(c.get("area_px"), 0.0)
        if region not in VALID_REGIONS:
            continue
        if conf < confidence_keep_threshold:
            # 太低的候选只放弃，不计入统计。
            continue
        if area_px <= 0:
            w = safe_float(c.get("bbox_width_px"), 0.0)
            h = safe_float(c.get("bbox_height_px"), 0.0)
            area_px = max(0.0, w * h * 0.78)
        area_mm2 = area_px * (pixel_size ** 2)
        eq_dia_mm = 2.0 * math.sqrt(max(area_mm2, 0.0) / math.pi) if area_mm2 > 0 else 0.0
        cc = dict(c)
        cc["candidate_id"] = idx
        cc["region_judgement"] = region
        cc["area_px"] = round(area_px, 3)
        cc["area_mm2"] = round(area_mm2, 4)
        cc["eq_diameter_mm"] = round(eq_dia_mm, 4)
        cc["confidence"] = round(conf, 3)
        cc["confidence_level"] = "明确气孔候选" if conf >= 0.6 else "疑似气孔候选"
        candidates.append(cc)
        region_count[region] += 1
        region_area[region] = max(region_area[region], area_mm2)
        region_dia[region] = max(region_dia[region], eq_dia_mm)
        total_area += area_mm2
        max_dia = max(max_dia, eq_dia_mm)

    total_count = len(candidates)
    quantity_leading = []
    if total_count:
        m = max(region_count.values())
        quantity_leading = [r for r, v in region_count.items() if v == m and v > 0]

    risk_focus = []
    # 高影响区域中只要有候选，进入风险关注；如果最大尺寸特别突出，也进入。
    for r in HIGH_IMPACT_REGIONS:
        if region_count.get(r, 0) > 0:
            risk_focus.append(r)
    if not risk_focus and total_count:
        # 兜底：最大面积区域
        m_area = max(region_area.values())
        risk_focus = [r for r, v in region_area.items() if v == m_area and v > 0]

    view_summaries = []
    by_view: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        try:
            vi = int(c.get("view_index", 0))
        except Exception:
            vi = 0
        by_view[vi].append(c)
    for vi in range(1, 10):
        arr = by_view.get(vi, [])
        if arr:
            regs = sorted(set(x.get("region_judgement", "") for x in arr))
            view_summaries.append({
                "view_index": vi,
                "visible_defect_description": "；".join([str(x.get("visible_defect_description", ""))[:30] for x in arr])[:120],
                "region_judgement": "、".join([x for x in regs if x]),
                "estimated_count": len(arr),
                "uncertainty": "中" if any(x.get("confidence", 0) < 0.6 for x in arr) else "低",
            })
        else:
            view_summaries.append({
                "view_index": vi,
                "visible_defect_description": "未形成有效气孔候选",
                "region_judgement": "无法判断",
                "estimated_count": 0,
                "uncertainty": "中",
            })

    return {
        "defect_type": "气孔",
        "source": "qwen3.5-plus visual estimation using 9 composite images with mask_vis",
        "metric_note": "面积和等效直径由模型估计的像素面积经标定公式换算，非YOLO/SAHI/SIFT检测结果。",
        "calibration": {
            "image_height_pixels": image_height_pixels,
            "calibration_length_mm": calibration_length_mm,
            "pixel_size_mm_per_pixel": pixel_size,
            "area_formula": "area_mm2 = area_px * pixel_size_mm_per_pixel^2",
            "eq_diameter_formula": "eq_diameter_mm = 2 * sqrt(area_mm2 / pi)",
            "confidence_keep_threshold": confidence_keep_threshold,
        },
        "total_defect_count": total_count,
        "total_area_mm2": round(total_area, 4),
        "max_eq_diameter_mm": round(max_dia, 4),
        "region_count": region_count,
        "region_max_area_mm2": {k: round(v, 4) for k, v in region_area.items()},
        "region_max_eq_diameter_mm": {k: round(v, 4) for k, v in region_dia.items()},
        "quantity_leading_regions": quantity_leading,
        "risk_focus_regions": risk_focus,
        "defect_candidates": candidates,
        "view_summaries": view_summaries,
        "uncertainty_note": str(raw_obj.get("uncertainty_note") or ""),
        "raw_parsed_visual_json": raw_obj,
    }


def build_visual_fact_block(summary: Dict[str, Any]) -> str:
    rc = summary.get("region_count", {}) or {}
    ra = summary.get("region_max_area_mm2", {}) or {}
    rd = summary.get("region_max_eq_diameter_mm", {}) or {}
    cands = summary.get("defect_candidates", []) or []
    lines = [
        "【当前视觉估计结果】",
        "缺陷类型统一按气孔处理；视觉估计只统计一般区域、牙孔区域、焊接区域、重点区域内的有效候选，背景/重叠/未知区域不计入。",
        f"有效气孔候选总数为{summary.get('total_defect_count', 0)}个，总面积约{summary.get('total_area_mm2', 0)}mm²，最大等效直径约{summary.get('max_eq_diameter_mm', 0)}mm。",
        "区域气孔数量：" + "；".join([f"{r}{int(rc.get(r, 0))}个" for r in VALID_REGIONS]) + "。",
        "区域最大面积：" + "；".join([f"{r}{float(ra.get(r, 0)):.4f}mm²" for r in VALID_REGIONS]) + "。",
        "区域最大等效直径：" + "；".join([f"{r}{float(rd.get(r, 0)):.4f}mm" for r in VALID_REGIONS]) + "。",
    ]
    if summary.get("quantity_leading_regions"):
        lines.append("数量主导区域：" + "、".join(summary["quantity_leading_regions"]) + "。")
    if summary.get("risk_focus_regions"):
        lines.append("风险关注区域：" + "、".join(summary["risk_focus_regions"]) + "。")
    if cands:
        lines.append("候选缺陷简表：")
        for c in cands[:12]:
            lines.append(
                f"- VIEW {c.get('view_index')}，{c.get('region_judgement')}，"
                f"面积约{c.get('area_mm2')}mm²，等效直径约{c.get('eq_diameter_mm')}mm，"
                f"置信度{c.get('confidence')}（{c.get('confidence_level')}），{str(c.get('visible_defect_description',''))[:40]}。"
            )
    if summary.get("uncertainty_note"):
        lines.append("视觉不确定性说明：" + str(summary.get("uncertainty_note"))[:300])
    lines.append("注意：置信度低于0.6的候选只能表述为疑似气孔风险，不得作为确定缺陷结论。")
    return "\n".join(lines)

# =========================
# 图谱文件读取与本地检索
# =========================

def text_of_node(node: Dict[str, Any]) -> str:
    props = node.get("properties") or {}
    labels = node.get("labels") or []
    vals = []
    for k in ["name", "standard_name", "raw_entity", "entity_type", "remark"]:
        v = props.get(k)
        if v:
            vals.append(str(v))
    vals.extend(map(str, labels))
    return "；".join(vals)


def text_of_edge(edge: Dict[str, Any]) -> str:
    vals = []
    for side in ["start_props", "end_props"]:
        props = edge.get(side) or {}
        for k in ["name", "standard_name", "raw_entity", "entity_type"]:
            v = props.get(k)
            if v:
                vals.append(str(v))
    rel_props = edge.get("rel_props") or {}
    for k in ["evidence_text", "relation_name"]:
        v = rel_props.get(k)
        if v:
            vals.append(str(v))
    vals.append(str(edge.get("rel_type") or ""))
    return "；".join(vals)


def load_graph_files(nodes_path: str | Path, edges_path: str | Path) -> Dict[str, Any]:
    nodes = load_json(nodes_path)
    edges = load_json(edges_path)
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    node_records = []
    for n in nodes:
        if isinstance(n, dict):
            node_records.append({"obj": n, "text": text_of_node(n)})
    edge_records = []
    for e in edges:
        if isinstance(e, dict):
            edge_records.append({"obj": e, "text": text_of_edge(e)})
    return {"nodes": node_records, "edges": edge_records, "nodes_path": str(nodes_path), "edges_path": str(edges_path)}


def should_reject_graph_text(text: str, question: str) -> bool:
    # 当前任务统一按气孔处理；若检索到其他缺陷词，除非问题明确提到，否则过滤。
    for t in FORBIDDEN_DEFECT_TERMS:
        if t in text and t not in question:
            return True
    return False


def question_terms(question: str, visual_summary: Dict[str, Any]) -> List[str]:
    base_terms = ["气孔", "气泡", "压铸充型", "排气", "工艺参数", "压力传递"]
    candidates = [
        "一般区域", "牙孔区域", "焊接区域", "重点区域", "高影响区域", "数量", "面积", "等效直径",
        "高速速度", "高速行程", "高速时间", "建压时间", "压射压力", "铸造压力",
        "料饼厚度", "射料结束位置", "慢速速度", "慢速行程", "环境温度", "环境湿度",
        "充型", "卷气", "排气", "压力", "压实", "补偿", "验证", "同工况", "位置", "坐标",
    ]
    terms = list(base_terms)
    for t in candidates:
        if t in question:
            terms.append(t)
    for r in visual_summary.get("risk_focus_regions", []) or []:
        terms.append(str(r))
    for r in visual_summary.get("quantity_leading_regions", []) or []:
        terms.append(str(r))
    # 去重保序
    seen = set()
    out = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score_graph_text(text: str, terms: List[str]) -> float:
    score = 0.0
    for t in terms:
        if t and t in text:
            score += 2.0 if len(t) >= 4 else 1.0
    # 气孔/区域/参数相关加权
    for t in ["气孔", "气泡", "牙孔区域", "焊接区域", "高速", "建压", "压射", "铸造压力", "排气", "充型", "卷气", "工艺参数"]:
        if t in text:
            score += 0.8
    return score


def retrieve_graph_context(
    graph: Dict[str, Any],
    question: str,
    visual_summary: Dict[str, Any],
    max_items: int = 12,
    max_chars: int = 1500,
) -> Tuple[str, Dict[str, Any]]:
    terms = question_terms(question, visual_summary)
    hits: List[Tuple[float, str, str]] = []
    for rec in graph.get("nodes", []):
        text = rec["text"]
        if should_reject_graph_text(text, question):
            continue
        s = score_graph_text(text, terms)
        if s > 0:
            hits.append((s, "node", text))
    for rec in graph.get("edges", []):
        text = rec["text"]
        if should_reject_graph_text(text, question):
            continue
        s = score_graph_text(text, terms)
        if s > 0:
            hits.append((s + 0.3, "edge", text))
    hits.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for s, typ, text in hits:
        # 清理rule id等内部字段风格
        clean = re.sub(r"R\d+_\d+", "", text)
        clean = re.sub(r"4:[a-zA-Z0-9\-:]+", "", clean)
        clean = re.sub(r"；{2,}", "；", clean).strip("； ")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        selected.append({"score": round(s, 3), "type": typ, "text": clean[:260]})
        if len(selected) >= max_items:
            break

    # 兜底图谱先验，防止某些问题没有命中。
    header = [
        "【图谱辅助证据】以下内容来自导出的Neo4j图谱文件，只作为机理先验；若与当前视觉统计或压铸参数冲突，必须以当前样本输入为准。",
        "可用图谱先验包括：气孔/气泡与压铸充型环节、排气条件、工艺参数异常相关；牙孔区域、焊接区域等区域风险需要结合当前样本数量和极值判断；高速时间、高速速度、高速行程、压射压力、建压时间、铸造压力等应按参数链联动分析。",
    ]
    lines = list(header)
    if selected:
        lines.append("当前问题相关图谱线索：")
        for item in selected:
            lines.append("- " + item["text"])
    lines.append("禁止在最终回答中输出规则编号、图谱命中列表、Neo4j内部ID；图谱不能替代当前样本事实。")
    ctx = "\n".join(lines)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n（图谱辅助证据已截断）"
    debug = {"terms": terms, "selected": selected, "hit_count": len(hits)}
    return ctx, debug

# =========================
# 提示词
# =========================

def trace_system_prompt() -> str:
    return """你是一名铝合金压铸气孔缺陷溯源助手。

硬性规则：
1. 最终回答必须直接从【结论】开始。
2. 最终回答必须且只能包含【结论】【溯源分析】【建议方向】三部分。
3. 缺陷类型统一按气孔处理，不要引入缩孔、缩松、裂纹、冷隔、夹杂、夹渣等非气孔缺陷。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。
5. 必须区分数量主导区域和风险关注区域。
6. 必须结合压铸参数链：高速速度、高速行程、高速时间、建压时间、压射压力、铸造压力、料饼厚度、射料结束位置等。
7. 图谱证据只辅助机理判断，不能覆盖当前样本视觉统计和压铸参数。
8. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。
9. 不要输出思考过程，不要复述整段输入。
10. 不要输出图谱命中列表、规则编号、Neo4j内部ID或调试信息。
11. 不要写“官方VLM”“qwen3.5-plus”“API模型”等来源词，统一表述为“当前视觉估计结果”。
12. 禁止把低置信度视觉候选说成确定缺陷；置信度低于0.6时只能写“疑似”或“需复核”。
13. 不要点名未提供的具体工艺部件或材料因素，如排气槽、溢流井、内浇口、喷涂积碳、脱模剂、熔炼含气量、合金成分、料温、模温、真空系统、除气工艺等；可以概括为“充型/排气相关工艺链路”。
""".strip()


def classify_question(q: str) -> str:
    if re.search(r"是否|能不能|能否|是不是|可不可以", q):
        return "yes_no_judgment"
    if re.search(r"如何验证|验证|对照|跟踪|复核|调整|改善|排查|建议", q):
        return "adjustment_validation"
    if re.search(r"关系|匹配|联动|影响|作用", q):
        return "relation_mechanism"
    if re.search(r"为什么|原因|主因|来源", q):
        return "cause_explanation"
    return "open_diagnosis"


def trace_user_prompt(
    question: str,
    visual_fact_block: str,
    params: Dict[str, Any],
    graph_context: str,
) -> str:
    qtype = classify_question(question)
    return f"""{visual_fact_block}

【压铸参数】
{format_params(params)}

【图谱辅助证据】
{graph_context}

【问题类型】{qtype}
【问题】
{question}

【作答要求】
1. 【结论】必须直接给出排查优先级或风险优先级，不得复述问题原句。
2. 【溯源分析】必须写 1. 2. 3. 三个编号点：
   1) 区域证据：说明数量主导区域、面积/等效直径风险关注区域、高影响区域；
   2) 参数链：分析高速充型链和压力建立链，不做单参数归因；
   3) 综合优先级：说明当前更应先复核哪条链路、哪个区域，哪些判断需要同工况验证。
3. 【建议方向】必须写 1. 2. 3. 三个编号点：
   1) 优先排查的宏观工艺链路；
   2) 参数与区域气孔表现的联动复核；
   3) 同料号、同视角、同工况、前后样本或趋势收敛验证。
4. 不要输出JSON、字典、等号清单，不要输出图谱命中列表和规则编号。

现在开始作答。""".strip()


def normalize_answer(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<\s*/?\s*think\s*>", "", text, flags=re.I)
    # 清理不适合最终报告的来源词
    text = text.replace("官方VLM", "当前视觉估计结果")
    text = text.replace("qwen3.5-plus", "当前视觉估计结果")
    text = text.replace("API模型", "当前视觉估计结果")
    text = text.replace("模型视觉", "当前视觉")
    # 清理规则编号
    text = re.sub(r"R\d+_\d+", "", text)
    text = re.sub(r"图谱命中列表.*", "", text)
    text = re.sub(r"规则编号.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    m = re.search(r"【结论】", text)
    if m:
        text = text[m.start():].strip()
    return text

# =========================
# 主流程
# =========================

def run_visual_once(args: argparse.Namespace, run_dir: Path, api_key: str) -> Tuple[Dict[str, Any], str, Dict[str, Any], List[Dict[str, Any]]]:
    # 允许复用已有视觉摘要，避免重复上传图片。
    if args.reuse_visual_json:
        vis = load_json(args.reuse_visual_json)
        raw_text = "[reuse_visual_json] 已复用既有视觉摘要，未重复上传图片。"
        return vis, raw_text, {}, []

    existing = run_dir / "visual_summary_for_scoring.json"
    if args.resume and existing.exists():
        vis = load_json(existing)
        raw_text = (run_dir / "visual_raw.txt").read_text(encoding="utf-8") if (run_dir / "visual_raw.txt").exists() else "[resume]"
        return vis, raw_text, {}, []

    labelme_refs, labelme_reference_text = build_labelme_reference(args, run_dir)
    save_json(run_dir / "labelme_reference.json", labelme_refs)
    save_text(run_dir / "labelme_reference_prompt.txt", labelme_reference_text)

    triplets = build_composites(args, run_dir)
    content: List[Dict[str, Any]] = [{"type": "text", "text": visual_prompt(labelme_reference_text)}]

    # 先输入 LabelMe 标注参考图，只在视觉阶段输入一次，帮助模型理解气孔外观。
    for ref in labelme_refs:
        content.append({"type": "image_url", "image_url": {"url": image_to_base64_data_url(ref["reference_image"], "image/jpeg")}})

    # 再输入当前目标工件九视角合成图。
    for item in triplets:
        content.append({"type": "image_url", "image_url": {"url": image_to_base64_data_url(item["composite_path"], "image/jpeg")}})

    messages = [{"role": "user", "content": content}]
    resp = api_call(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        max_tokens=args.visual_max_tokens,
        temperature=0.0,
        timeout=args.timeout,
        retries=args.retries,
    )
    raw = extract_content(resp)
    raw_obj = extract_json_object(raw)
    visual_summary = summarize_visual_json(
        raw_obj,
        pixel_size=args.pixel_size,
        image_height_pixels=args.image_height_pixels,
        calibration_length_mm=args.calibration_length_mm,
        confidence_keep_threshold=args.confidence_keep_threshold,
    )

    save_text(run_dir / "visual_raw.txt", raw)
    save_json(run_dir / "visual_raw_response.json", resp)
    save_json(run_dir / "visual_summary_for_scoring.json", visual_summary)
    save_text(run_dir / "visual_summary_generated.txt", build_visual_fact_block(visual_summary))
    save_json(run_dir / "selected_views.json", triplets)
    return visual_summary, raw, resp.get("usage", {}), triplets


def run_trace_questions(args: argparse.Namespace, run_dir: Path, api_key: str, visual_summary: Dict[str, Any], graph: Dict[str, Any], questions: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    trace_jsonl = run_dir / "trace_answers.jsonl"
    summary_csv = run_dir / "trace_summary.csv"
    qroot = ensure_dir(run_dir / "questions")
    usage_all: Dict[str, Any] = {}
    cases_by_index: Dict[int, Dict[str, Any]] = {}

    # Resume：从已有jsonl恢复。
    if args.resume and trace_jsonl.exists():
        for row in read_jsonl(trace_jsonl):
            try:
                cases_by_index[int(row.get("question_index"))] = row
            except Exception:
                pass

    visual_fact_block = build_visual_fact_block(visual_summary)
    graph_debug_all = []

    start = max(1, args.start_index)
    end = len(questions)
    if args.limit and args.limit > 0:
        end = min(end, start + args.limit - 1)

    for idx in range(start, end + 1):
        question = questions[idx - 1]
        if idx in cases_by_index:
            print(f"[skip] Q{idx:03d} already exists", flush=True)
            continue

        graph_context, graph_debug = retrieve_graph_context(
            graph=graph,
            question=question,
            visual_summary=visual_summary,
            max_items=args.graph_top_k,
            max_chars=args.graph_context_max_chars,
        )
        graph_debug_all.append({"question_index": idx, "question": question, **graph_debug})
        qdir = ensure_dir(qroot / f"q{idx:03d}")
        save_text(qdir / "graph_context.txt", graph_context)
        save_json(qdir / "graph_debug.json", graph_debug)

        messages = [
            {"role": "system", "content": trace_system_prompt()},
            {"role": "user", "content": trace_user_prompt(question, visual_fact_block, params, graph_context)},
        ]
        print(f"[trace] Q{idx:03d}/{len(questions)} start", flush=True)
        resp = api_call(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.answer_max_tokens,
            temperature=0.0,
            timeout=args.timeout,
            retries=args.retries,
        )
        raw = extract_content(resp)
        answer = normalize_answer(raw)
        usage_all = add_usage(usage_all, resp.get("usage", {}))

        mode_result = {
            "mode": DEFAULT_MODE_KEY,
            "mode_desc": "Qwen3.5-Plus API：LabelMe标注样本辅助视觉识别 + 九视角视觉总结一次复用 + 导出Neo4j图谱文件轻量检索辅助 + 参数 + 问题",
            "raw_answer": raw,
            "final_answer": answer,
            "uses_user_visual_layer": False,
            "uses_yolo_sahi_sift": False,
            "uses_graph_export_files": True,
            "uses_labelme_reference": True,
            "graph_context": graph_context,
            "usage": resp.get("usage", {}),
        }
        case = {
            "question_index": idx,
            "sample_id": args.sample_id,
            "workpiece_id": args.workpiece_id,
            "question": question,
            "question_type": classify_question(question),
            "visual_summary": visual_summary,
            "visual_fact_block": visual_fact_block,
            "params": params,
            "graph_context": graph_context,
            "modes": {DEFAULT_MODE_KEY: mode_result},
        }
        save_text(qdir / "final_answer.txt", answer)
        save_text(qdir / "raw_answer.txt", raw)
        save_text(qdir / "user_prompt.txt", messages[1]["content"])
        save_json(qdir / "e2e_result.json", mode_result)
        append_jsonl(trace_jsonl, {
            "question_index": idx,
            "question": question,
            "final_answer": answer,
            "raw_answer": raw,
            "usage": resp.get("usage", {}),
        })
        cases_by_index[idx] = case
        print(f"[trace] Q{idx:03d} done | tokens={resp.get('usage', {}).get('total_tokens', '')}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

        # 每题都保存一次总结果，防止中断丢数据。
        write_outputs(args, run_dir, visual_summary, params, graph, questions, cases_by_index, usage_all, graph_debug_all)

    write_summary_csv(summary_csv, cases_by_index)
    save_json(run_dir / "graph_retrieval_debug.json", graph_debug_all)
    return usage_all


def write_summary_csv(path: Path, cases_by_index: Dict[int, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_index", "question", "answer_preview"])
        writer.writeheader()
        for idx in sorted(cases_by_index):
            case = cases_by_index[idx]
            ans = case["modes"][DEFAULT_MODE_KEY].get("final_answer", "")
            writer.writerow({"question_index": idx, "question": case.get("question", ""), "answer_preview": ans[:160].replace("\n", " ")})


def write_outputs(args: argparse.Namespace, run_dir: Path, visual_summary: Dict[str, Any], params: Dict[str, Any], graph: Dict[str, Any], questions: List[str], cases_by_index: Dict[int, Dict[str, Any]], usage_trace: Dict[str, Any], graph_debug_all: List[Dict[str, Any]]) -> None:
    cases = [cases_by_index[i] for i in sorted(cases_by_index)]
    output_obj = {
        "schema_version": "qwen35_plus_graph_300q_v1",
        "model": args.model,
        "sample_id": args.sample_id,
        "workpiece_id": args.workpiece_id,
        "run_dir": str(run_dir),
        "questions_json": args.questions_json,
        "graph_nodes_json": args.graph_nodes_json,
        "graph_edges_json": args.graph_edges_json,
        "labelme_dir": args.labelme_dir,
        "image_dir": args.image_dir,
        "mask_vis_dir": args.mask_vis_dir,
        "template_dir": args.template_dir,
        "visual_summary": visual_summary,
        "visual_fact_block": build_visual_fact_block(visual_summary),
        "params": params,
        "mode_definitions": {
            DEFAULT_MODE_KEY: {
                "mode_desc": "Qwen3.5-Plus API：LabelMe标注样本辅助视觉识别；图像只识别一次，300问题复用视觉总结；图谱来自导出JSON文件本地检索。",
                "uses_user_visual_layer": False,
                "uses_yolo_sahi_sift": False,
                "uses_graph_export_files": True,
                "uses_labelme_reference": True,
            }
        },
        "cases": cases,
    }
    save_json(run_dir / "all_modes_results.json", output_obj)
    save_json(run_dir / "all_results.json", output_obj)
    save_json(run_dir / "usage_trace_sum.json", usage_trace)
    save_json(run_dir / "graph_retrieval_debug.json", graph_debug_all)


def build_report(run_dir: Path, args: argparse.Namespace, questions: List[str], visual_summary: Dict[str, Any], visual_usage: Dict[str, Any], trace_usage: Dict[str, Any]) -> str:
    total_usage = add_usage(visual_usage, trace_usage)
    lines = [
        "# Qwen3.5-Plus LabelMe标注示例 + 九视角一次视觉识别 + 图谱文件辅助 + 300问题批量溯源测试\n",
        f"## 样本\n{args.sample_id}\n",
        "## 输入\n",
        f"- X光图目录：`{args.image_dir}`\n",
        f"- LabelMe标注目录：`{args.labelme_dir}`\n",
        f"- mask_vis目录：`{args.mask_vis_dir}`\n",
        f"- 模板目录：`{args.template_dir}`\n",
        f"- 问题文件：`{args.questions_json}`，读取问题数：{len(questions)}\n",
        f"- 图谱节点文件：`{args.graph_nodes_json}`\n",
        f"- 图谱关系文件：`{args.graph_edges_json}`\n",
        "- 图片视觉识别只调用一次，后续问题不再重复上传图片。\n",
        "## 视觉结构化摘要\n```json\n",
        json.dumps(visual_summary, ensure_ascii=False, indent=2),
        "\n```\n",
        "## Token用量\n```json\n",
        json.dumps({"visual_usage": visual_usage, "trace_usage_sum": trace_usage, "total_usage": total_usage}, ensure_ascii=False, indent=2),
        "\n```\n",
        "## 评分文件\n`all_modes_results.json`\n",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-Plus 一次视觉识别 + 图谱文件 + 300问题批量测试")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--template-dir", default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--mask-vis-dir", default=DEFAULT_MASK_VIS_DIR)
    parser.add_argument("--labelme-dir", default=DEFAULT_LABELME_DIR, help="LabelMe标注样本目录，包含图片和同名json")
    parser.add_argument("--max-label-examples", type=int, default=4, help="最多输入多少组LabelMe标注示例")
    parser.add_argument("--allow-no-labelme", action="store_true", help="调试用：即使没有LabelMe配对也继续运行；正式实验不要开启")
    parser.add_argument("--questions-json", default=DEFAULT_QUESTIONS_JSON)
    parser.add_argument("--graph-nodes-json", default=DEFAULT_GRAPH_NODES_JSON)
    parser.add_argument("--graph-edges-json", default=DEFAULT_GRAPH_EDGES_JSON)
    parser.add_argument("--params-json", default="", help="可选，压铸参数JSON；不传则用脚本默认params")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--workpiece-id", default=DEFAULT_WORKPIECE_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-views", type=int, default=9)
    parser.add_argument("--composite-panel-size", type=int, default=760)
    parser.add_argument("--visual-max-tokens", type=int, default=1200)
    parser.add_argument("--answer-max-tokens", type=int, default=720)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2, help="每个问题之间的等待秒数，避免过快触发限流")
    parser.add_argument("--start-index", type=int, default=1, help="从第几个问题开始，1-based")
    parser.add_argument("--limit", type=int, default=0, help="最多跑多少个问题；0表示跑到末尾")
    parser.add_argument("--resume", action="store_true", help="断点续跑，跳过已有trace_answers.jsonl中的问题")
    parser.add_argument("--reuse-run-dir", default="", help="复用已有输出目录继续跑")
    parser.add_argument("--reuse-visual-json", default="", help="复用已有visual_summary_for_scoring.json，完全不重复视觉识别")
    parser.add_argument("--only-visual", action="store_true", help="只跑一次视觉识别，不跑问题")
    parser.add_argument("--graph-top-k", type=int, default=12)
    parser.add_argument("--graph-context-max-chars", type=int, default=1500)
    parser.add_argument("--image-height-pixels", type=int, default=DEFAULT_IMAGE_HEIGHT_PIXELS)
    parser.add_argument("--calibration-length-mm", type=float, default=DEFAULT_CALIBRATION_LENGTH_MM)
    parser.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE_MM_PER_PIXEL)
    parser.add_argument("--confidence-keep-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY，例如：export DASHSCOPE_API_KEY='sk-xxxx'")

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.reuse_run_dir:
        run_dir = ensure_dir(args.reuse_run_dir)
    else:
        run_dir = ensure_dir(Path(args.output_root) / f"qwen35_plus_graph_300q_{ts}")

    print("=" * 100)
    print("[run] Qwen3.5-Plus LabelMe reference + 9-view visual once + graph files + batch questions")
    print(f"[run] output_dir = {run_dir}")
    print(f"[run] model      = {args.model}")
    print(f"[run] questions  = {args.questions_json}")
    print(f"[run] graph      = {args.graph_nodes_json} | {args.graph_edges_json}")
    print(f"[run] labelme    = {args.labelme_dir} | max_examples={args.max_label_examples}")
    print("=" * 100, flush=True)

    params = load_params_from_json(args.params_json or None)
    questions = load_questions(args.questions_json)
    save_json(run_dir / "run_args.json", vars(args))
    save_json(run_dir / "params.json", params)
    save_json(run_dir / "questions_loaded.json", questions)

    # Stage 1: 视觉识别一次
    print("[stage 1/2] visual summary once", flush=True)
    visual_summary, visual_raw, visual_usage, triplets = run_visual_once(args, run_dir, api_key)
    save_text(run_dir / "visual_fact_block.txt", build_visual_fact_block(visual_summary))
    print(f"[stage 1/2] visual done | count={visual_summary.get('total_defect_count')} | tokens={visual_usage.get('total_tokens','reuse')}", flush=True)

    if args.only_visual:
        save_text(run_dir / "report.md", build_report(run_dir, args, questions, visual_summary, visual_usage, {}))
        print(f"[done] only visual: {run_dir}")
        return

    # Stage 2: 加载图谱文件，本地检索，逐问题文本回答
    print("[stage 2/2] load graph files", flush=True)
    graph = load_graph_files(args.graph_nodes_json, args.graph_edges_json)
    print(f"[graph] nodes={len(graph.get('nodes', []))} edges={len(graph.get('edges', []))}", flush=True)

    trace_usage = run_trace_questions(args, run_dir, api_key, visual_summary, graph, questions, params)
    save_text(run_dir / "report.md", build_report(run_dir, args, questions, visual_summary, visual_usage, trace_usage))

    total_usage = add_usage(visual_usage, trace_usage)
    save_json(run_dir / "usage_summary.json", {"visual_usage": visual_usage, "trace_usage_sum": trace_usage, "total_usage": total_usage})

    print("=" * 100)
    print("[done] batch test finished")
    print(f"[done] output_dir    : {run_dir}")
    print(f"[done] scoring_input : {run_dir / 'all_modes_results.json'}")
    print(f"[done] visual_summary: {run_dir / 'visual_summary_for_scoring.json'}")
    print(f"[done] trace_jsonl   : {run_dir / 'trace_answers.jsonl'}")
    print(f"[done] usage         : {run_dir / 'usage_summary.json'}")
    print(json.dumps({"visual_usage": visual_usage, "trace_usage_sum": trace_usage, "total_usage": total_usage}, ensure_ascii=False, indent=2))
    print("=" * 100)


if __name__ == "__main__":
    main()
