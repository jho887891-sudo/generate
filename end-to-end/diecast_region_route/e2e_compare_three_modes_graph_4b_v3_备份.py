from __future__ import annotations

import gc
import json
import os
import re
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import cv2
import numpy as np
import torch
from peft import PeftModel
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from multimodal.config_region_mm import RegionAwareMMConfig
from multimodal.model_region_mm import RegionAwareMultiModalModel
from multimodal.vision_backbone import preprocess_image

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


DEFAULT_CONFIG_JSON = Path("/home/T7/wkq/end-to-end/diecast_region_route/e2e_compare_graph_config_hardcoded.json")
LOCAL_CONFIG_JSON = Path(__file__).resolve().with_name("e2e_compare_graph_config_hardcoded.json")
CONFIG_JSON = str(DEFAULT_CONFIG_JSON if DEFAULT_CONFIG_JSON.exists() else LOCAL_CONFIG_JSON)

SYSTEM_PROMPT = """你是一名铝合金压铸气孔缺陷溯源助手。

硬性规则：
1. 最终回答必须直接从【结论】开始。
2. 最终回答必须且只能包含【结论】【溯源分析】【建议方向】三部分。
3. 缺陷类型统一按气孔处理。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。
5. 回答必须依据当前输入中的视觉层文字总结、多模态前缀token、压铸参数和图谱辅助证据。
6. 图谱证据只辅助机理判断，不能覆盖当前样本；最终不要输出图谱命中列表、规则编号或调试信息。
7. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。
8. 不要输出思考过程，不要复述整段输入。
9.【溯源分析】和【建议方向】每点控制在一两句话以内！！！避免过长段落。
"""

FEW_SHOT_EXAMPLE = """【回答示例：只学习格式和组织方式，不要复制示例中的数字、区域名称或结论】

【示例输入】
【视觉层文字总结】
缺陷类型统一按气孔处理。
有效工艺区域包括：一般区域、牙孔区域、焊接区域；其中牙孔区域、焊接区域为高影响区域。
总气孔条目=136
总面积(mm²)=18.0905
最大等效直径(mm)=1.0448
区域缺陷统计={"一般区域":119,"牙孔区域":13,"焊接区域":4}
区域最大面积(mm²)={"一般区域":0.2744,"牙孔区域":0.385,"焊接区域":0.2499}
区域最大等效直径(mm)={"一般区域":0.7746,"牙孔区域":1.0448,"焊接区域":0.8308}

【压铸参数摘要】
慢速速度=0.19
高速速度=3.65
料饼厚度=28
铸造压力=926
循环周期=138.1
建压时间=114
射料结束位置=748
压射压力=316
高速行程=205
高速时间=46
慢速行程=525

【问题】
当前牙孔区域和焊接区域存在气孔，应优先从哪些参数链进行溯源？

【示例输出】
【结论】
当前样本应优先围绕气孔在高影响区域中的极值表现进行溯源；虽然一般区域气孔数量最多，但牙孔区域的最大等效直径更突出，不能只按数量判断风险。

【溯源分析】
1. 一般区域气孔数量最多，说明整体充型或卷气波动需要关注；牙孔区域虽然数量较少，但最大等效直径高于其他区域，且属于后续加工相关的高影响区域，应作为风险复核重点；焊接区域数量较少，但同样属于高影响区域，需保留为辅助风险点。
2. 当前高速速度、高速行程和高速时间共同决定高速充型阶段的覆盖范围与金属液前沿稳定性；建压时间、压射压力和铸造压力共同影响气孔被压实和压力传递是否及时；料饼厚度和射料结束位置会影响后段余料补偿与压力持续性。
3. 结合区域极值、速度行程链和压力建立链，当前更应优先排查高速充型阶段卷气带入后，在牙孔等高影响区域保留的问题；图谱证据只能作为卷气、排气和压力传递机理的辅助依据，最终判断仍以当前样本的区域统计和参数链为准。

【建议方向】
1. 优先围绕高速速度、高速行程和高速时间的匹配关系，复核牙孔区域气孔位置和极值是否随高速充型稳定性变化而收敛。
2. 同步复核建压时间、压射压力和铸造压力之间的压力传递关系，观察高影响区域气孔尺寸是否随压实窗口改善而下降。
3. 选取同料号、同视角、同工况样本，建立区域气孔数量、区域最大面积、区域最大等效直径和关键参数的对照表，验证调整方向是否具有可复现性。
"""

FIXED_REQUIREMENT = ""

QUESTION_TYPE_HINTS = {
    "effect_analysis": "当前问题属于影响分析类。重点说明参数变化可能如何影响缺陷分布趋势、区域风险和工艺判断。",
    "cause_explanation": "当前问题属于原因解释类。重点说明现象为什么出现，区分主因链与次因链。",
    "relation_mechanism": "当前问题属于参数/机理关系类。重点说明变量关系，以及这种关系为何影响缺陷判断。",
    "adjustment_validation": "当前问题属于调整/验证类。重点说明先观察什么、如何验证是否有效、如何做前后样本对比。",
    "inspection_identification": "当前问题属于探伤识别/区分类。重点说明影像趋势、区域特征和判定依据。",
    "yes_no_judgment": "当前问题属于是非判断类。必须先给出明确判断，再说明依据和限制条件。",
    "binary_decision": "当前问题属于二选一判断类。必须明确更偏向哪一项，并说明另一项为何不是当前优先判断。",
    "open_diagnosis": "当前问题属于开放诊断类。先抓数量主导趋势和风险关注趋势，再分析参数链和优先级。",
}

STATIC_HINT_TERMS = [
    "一般区域", "牙孔区域", "焊接区域", "重点区域",
    "气孔", "气泡", "缩孔", "缩松", "裂纹", "冷隔", "热裂", "冷裂", "冷豆", "变形",
    "卷气", "补缩", "补缩失效", "压力传递", "充型稳定性",
    "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
    "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
    "慢速行程", "环境温度", "环境湿度",
]

SECTION_PATTERN = re.compile(r"【结论】(?P<conclusion>.*?)【溯源分析】(?P<analysis>.*?)【建议方向】(?P<suggestion>.*)", re.DOTALL)
THINK_TAG_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
neo4j_driver = None


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: str | Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============================================================
# 视觉层区域白名单与过滤规则
# ============================================================
# 当前视觉层只允许进入后续统计 / token / 提示词的区域：
#   1：一般区域
#   2：牙孔区域
#   3：焊接区域
#   4：重点区域
#
# 其他区域，包括蒙版值 0 / 5 / 6、未知映射、未配置映射、重叠区等，
# 一律只作为视觉检测过程中的屏蔽区域，不写入 defects、boxes、
# aggregate_summary、view_items、visual_fact_block 和最终提示词。
#
# 注意：
# - 函数名 is_background_or_unknown_region 保留，是为了兼容后续已有调用点；
# - 实际含义已经扩展为：凡是不在 ALLOWED_REGION_NAMES 内的区域都过滤。
# ============================================================
ALLOWED_REGION_NAMES = ("一般区域", "牙孔区域", "焊接区域", "重点区域")
ALLOWED_REGION_NAME_SET = set(ALLOWED_REGION_NAMES)
ALLOWED_REGION_VALUES = {"1", "1.0", "2", "2.0", "3", "3.0", "4", "4.0"}

IGNORED_REGION_KEYWORDS = ("背景", "未知", "重叠")
IGNORED_REGION_VALUES = {"0", "0.0", "5", "5.0", "6", "6.0"}


def is_background_or_unknown_region(region_str: Any, region_val: Any | None = None) -> bool:
    """Return True for regions that must be removed before statistics/tokenization.

    当前只保留：
    - 一般区域
    - 牙孔区域
    - 焊接区域
    - 重点区域

    除上述四类之外的区域全部过滤，避免进入视觉统计、区域 token 和提示词。
    """
    s = str(region_str or "").strip()
    v = str(region_val or "").strip()

    if not s and not v:
        return True

    if any(k in s for k in IGNORED_REGION_KEYWORDS):
        return True

    if s in {"None", "null"} or s in IGNORED_REGION_VALUES:
        return True

    if v in IGNORED_REGION_VALUES:
        return True

    # 如果已经映射成中文区域名，则只允许白名单区域进入统计。
    if s.endswith("区域") and s not in ALLOWED_REGION_NAME_SET:
        return True

    # 未配置映射时会变成类似“未知区域(7)”，上面已按关键词过滤。
    # 这里再兜底：非白名单区域名全部过滤。
    if s not in ALLOWED_REGION_NAME_SET:
        return True

    return False


def filter_region_metric_dict(data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Keep only whitelisted effective process regions in summary metric dictionaries."""
    if not isinstance(data, dict):
        return {}
    return {
        str(k): v
        for k, v in data.items()
        if str(k) in ALLOWED_REGION_NAME_SET and not is_background_or_unknown_region(k)
    }


def clean_visual_summary(summary: Dict[str, Any] | None) -> Dict[str, Any]:
    """Keep only whitelisted effective process-region fields in an aggregate/per-view summary."""
    s = dict(summary or {})
    s["region_count"] = filter_region_metric_dict(s.get("region_count"))
    s["region_max_area_mm2"] = filter_region_metric_dict(s.get("region_max_area_mm2"))
    s["region_max_eq_diameter_mm"] = filter_region_metric_dict(s.get("region_max_eq_diameter_mm"))
    # Do not recompute total_area here because per-defect areas may not exist in an aggregate summary.
    # Live detection is filtered before aggregation, so totals are already clean in the normal path.
    return s


def filter_visual_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safety filter for both live visual output and dataset-loaded bundles.
    It removes all non-whitelisted-region defects from defect_meta_by_view and keeps boxes aligned.
    """
    if not isinstance(bundle, dict):
        return bundle
    new_bundle = dict(bundle)
    new_defect_meta_by_view = []
    new_boxes_by_view = []
    for defects, boxes in zip(bundle.get("defect_meta_by_view", []) or [], bundle.get("boxes_by_view", []) or []):
        clean_defects = []
        clean_boxes = []
        for j, d in enumerate(defects or []):
            region_name = d.get("Region Type", d.get("region_type", ""))
            region_val = d.get("Region Value", d.get("region_value", ""))
            if is_background_or_unknown_region(region_name, region_val):
                continue
            clean_defects.append(d)
            if j < len(boxes or []):
                clean_boxes.append((boxes or [])[j])
            elif d.get("bbox"):
                clean_boxes.append(d["bbox"])
        new_defect_meta_by_view.append(clean_defects)
        new_boxes_by_view.append(clean_boxes)
    if new_defect_meta_by_view:
        new_bundle["defect_meta_by_view"] = new_defect_meta_by_view
        new_bundle["boxes_by_view"] = new_boxes_by_view
    new_bundle["aggregate_summary"] = clean_visual_summary(bundle.get("aggregate_summary", {}))
    view_items = []
    for item in bundle.get("view_items", []) or []:
        x = dict(item)
        x["region_count"] = filter_region_metric_dict(x.get("region_count"))
        view_items.append(x)
    if view_items:
        new_bundle["view_items"] = view_items
    return new_bundle


def collect_image_paths(cfg: Dict[str, Any]) -> List[str]:
    if cfg.get("image_paths"):
        return list(cfg["image_paths"])
    image_dir = cfg.get("image_dir")
    if not image_dir:
        raise ValueError("配置里需要提供 image_paths 或 image_dir")
    files = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return [str(p) for p in files]


def get_test_questions(cfg_json: Dict[str, Any]) -> List[str]:
    def clean_items(obj: Any) -> List[str]:
        if obj is None:
            return []
        if isinstance(obj, str):
            q = obj.strip()
            return [q] if q else []
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, dict):
                    q = str(item.get("question", "")).strip()
                else:
                    q = str(item).strip()
                if q:
                    out.append(q)
            return out
        if isinstance(obj, dict):
            q = str(obj.get("question", "")).strip()
            return [q] if q else []
        q = str(obj).strip()
        return [q] if q else []

    questions = clean_items(cfg_json.get("questions"))
    if not questions:
        questions = clean_items(cfg_json.get("question"))
    return questions[:300]


def load_jsonl_record(jsonl_path: str | Path, sample_id: Any, q_id: Any | None = None) -> Dict[str, Any]:
    sample_id_str = str(sample_id)
    q_id_str = "" if q_id is None else str(q_id)
    path = Path(jsonl_path)
    if not path.exists():
        local = Path(__file__).resolve().with_name(path.name)
        if local.exists():
            path = local
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("sample_id", "")) != sample_id_str:
                continue
            if q_id is not None and str(rec.get("q_id", "")) != q_id_str:
                continue
            return rec
    raise FileNotFoundError(f"未在语料中找到 sample_id={sample_id}, q_id={q_id}: {path}")


def bundle_from_dataset_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    image_paths = list(rec.get("image_paths", []))
    view_items = list(rec.get("view_items", []))
    boxes_by_view = list(rec.get("boxes_by_view", []))
    defect_meta_by_view = list(rec.get("defect_meta_by_view", []))
    summary = dict(rec.get("summary", {}) or {})
    if not image_paths:
        raise ValueError("语料记录缺少 image_paths，无法做多模态测试。")
    return {
        "aggregate_summary": summary,
        "boxes_by_view": boxes_by_view,
        "defect_meta_by_view": defect_meta_by_view,
        "view_items": view_items,
        "image_paths": image_paths,
    }


def print_graph_evidence(idx: int, graph_context: str, graph_debug: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"【Q{idx:02d} 图谱命中调试】")
    compact_debug = {
        "enabled": graph_debug.get("enabled", False),
        "available": graph_debug.get("available", False),
        "status": graph_debug.get("status", ""),
        "seed_terms": graph_debug.get("seed_terms", []),
        "seed_entities": graph_debug.get("seed_entities", []),
        "defect_names": graph_debug.get("defect_names", []),
        "rule_ids": graph_debug.get("rule_ids", []),
        "rule_hits": graph_debug.get("rule_hits", 0),
        "evidence_hits": graph_debug.get("evidence_hits", 0),
        "question_type": graph_debug.get("question_type", ""),
        "focus_regions": graph_debug.get("focus_regions", []),
        "mechanism_candidates": graph_debug.get("mechanism_candidates", []),
    }
    print(json.dumps(compact_debug, ensure_ascii=False, indent=2), flush=True)
    print(f"\n【Q{idx:02d} 图谱证据文本】")
    if graph_context:
        print(graph_context[:4000], flush=True)
        if len(graph_context) > 4000:
            print("（图谱证据终端显示已截断，完整内容见 qXX/graph_context.txt）", flush=True)
    else:
        print("未检索到图谱证据，或图谱检索失败。", flush=True)


def classify_question(q: str) -> str:
    q = q.strip()
    if re.search(r"还是|优先.*还是|是否可以直接判断", q):
        return "binary_decision"
    if re.search(r"(有改善作用吗|有压实作用吗|会不会|是否需要|是否可以|是否合理|是否同步|能否|正常吗)", q):
        return "yes_no_judgment"
    if re.search(r"(关系|比值|区别|如何计算|协同影响|交互作用|设定顺序|乘积|匹配|关联吗|有关联吗)", q):
        return "relation_mechanism"
    if re.search(r"(如何准确区分|如何区分|如何判定|影像特征|是什么缺陷|是什么现象|探伤图上的解释|什么意思|如何评级)", q):
        return "inspection_identification"
    if re.search(r"(如何调整|如何改善|如何通过.*改善|如何通过参数改善|如何参数调整|如何预防|应如何调整|如何判断最佳值|如何监测|如何现场快速验证|如何优化|如何确保|如何选择|如何避免|如何排查|针对性解决|怎么办|如何平衡|如何确定|如何解决|培训区分)", q):
        return "adjustment_validation"
    if re.search(r"(为什么|原因|可能是什么原因|可能原因|有何后果|会导致什么|导致什么|会产生什么缺陷|后果是什么|怎么形成的|主要成因是什么)", q):
        return "cause_explanation"
    if re.search(r"(有何影响|对.+影响|有何不同|如何变化|有何参考|风险|极限作用|有什么特征|效果如何|好处)", q):
        return "effect_analysis"
    return "open_diagnosis"


def detect_mechanism_candidates(q: str) -> List[str]:
    candidates = []
    for term in ["卷气类缺陷保留", "补缩失效类缺陷保留", "气孔", "缩孔", "缩松", "裂纹", "冷隔", "热裂", "变形"]:
        base = term.replace("类缺陷保留", "")
        if base in q or term in q:
            candidates.append(term)
    if "卷气" in q and "补缩" in q:
        candidates = ["卷气类缺陷保留", "补缩失效类缺陷保留"]
    return unique_keep_order(candidates)


def region_flags(q: str) -> List[str]:
    regs = []
    for term in ["一般区域", "牙孔区域", "焊接区域", "重点区域"]:
        if term in q:
            regs.append(term)
    return regs


def build_question_type_hint(question: str) -> Tuple[str, str, List[str], List[str]]:
    qt = classify_question(question)
    regions = region_flags(question)
    mechanisms = detect_mechanism_candidates(question)
    lines = [f"【问题类型】{qt}"]
    lines.append(f"【类型提示】{QUESTION_TYPE_HINTS.get(qt, QUESTION_TYPE_HINTS['open_diagnosis'])}")
    if regions:
        lines.append("【关注区域】" + "；".join(regions))
    if mechanisms:
        lines.append("【候选机理】" + "；".join(mechanisms))
    return qt, "\n".join(lines), regions, mechanisms


def extract_binary_choice_from_text(text: str, candidates: List[str]) -> str:
    if len(candidates) != 2:
        return ""
    text = text or ""
    a, b = candidates
    has_a = a in text or a.replace("类缺陷保留", "") in text
    has_b = b in text or b.replace("类缺陷保留", "") in text
    if has_a and not has_b:
        return a
    if has_b and not has_a:
        return b
    better = re.search(r"(更偏向|更可能|优先判断为|倾向于)([^。\n]+)", text)
    if better:
        seg = better.group(2)
        if a in seg or a.replace("类缺陷保留", "") in seg:
            return a
        if b in seg or b.replace("类缺陷保留", "") in seg:
            return b
    return ""


def count_defects_in_circle(center, defects_list, radius_mm, pixel_size_mm_per_pixel, max_area=None, is_pore=False):
    count = 0
    cx, cy = center
    radius_px = radius_mm / pixel_size_mm_per_pixel
    for d in defects_list:
        dist = np.sqrt((d["X Position"] - cx) ** 2 + (d["Y Position"] - cy) ** 2)
        if dist <= radius_px and (max_area is None or d["Area (mm²)"] <= max_area) and d["IsPore"] == is_pore:
            count += 1
    return count


def align_mask_to_image(img, template, mask):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    kp2, des2 = sift.detectAndCompute(template_gray, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    if len(good_matches) < 4:
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    pts_img = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_template = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(pts_template, pts_img, cv2.RANSAC, 5.0)
    if M is None:
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = img_gray.shape
    return cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)


def aggregate_summaries(per_view: List[Dict[str, Any]]) -> Dict[str, Any]:
    region_count = defaultdict(int)
    region_max_area_mm2 = defaultdict(float)
    region_max_eq_diameter_mm = defaultdict(float)
    class_summary = defaultdict(int)
    total_defect_count = 0
    total_area_mm2 = 0.0
    max_eq_diameter_mm = 0.0
    is_pore_count = 0
    for item in per_view:
        s = item["summary"]
        total_defect_count += int(s.get("total_defect_count", 0))
        total_area_mm2 += float(s.get("total_area_mm2", 0.0))
        max_eq_diameter_mm = max(max_eq_diameter_mm, float(s.get("max_eq_diameter_mm", 0.0)))
        is_pore_count += int(s.get("is_pore_count", 0))
        for k, v in (s.get("region_count", {}) or {}).items():
            region_count[k] += int(v)
        for k, v in (s.get("region_max_area_mm2", {}) or {}).items():
            region_max_area_mm2[k] = max(region_max_area_mm2[k], float(v))
        for k, v in (s.get("region_max_eq_diameter_mm", {}) or {}).items():
            region_max_eq_diameter_mm[k] = max(region_max_eq_diameter_mm[k], float(v))
        for k, v in (s.get("class_summary", {}) or {}).items():
            class_summary[k] += int(v)
    return {
        "total_defect_count": total_defect_count,
        "total_area_mm2": round(total_area_mm2, 3),
        "max_eq_diameter_mm": round(max_eq_diameter_mm, 3),
        "region_count": dict(region_count),
        "region_max_area_mm2": {k: round(v, 3) for k, v in region_max_area_mm2.items()},
        "region_max_eq_diameter_mm": {k: round(v, 3) for k, v in region_max_eq_diameter_mm.items()},
        "class_summary": dict(class_summary),
        "is_pore_count": is_pore_count,
    }


class VisualStageRunner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_path = cfg["visual_model_path"]
        self.calibration_json = cfg["calibration_json"]
        self.region_types_json = cfg["region_types_json"]
        self.standard_json = cfg["standard_json"]
        self.template_folder = cfg["template_folder"]
        self.mask_folder = cfg["mask_folder"]
        self.visual_output_dir = Path(cfg["visual_output_dir"])
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.3))
        self.model_image_size = int(cfg.get("model_image_size", 1280))
        self.slice_height = int(cfg.get("slice_height", 640))
        self.slice_width = int(cfg.get("slice_width", 640))
        self.overlap_height_ratio = float(cfg.get("overlap_height_ratio", 0.3))
        self.overlap_width_ratio = float(cfg.get("overlap_width_ratio", 0.3))
        self.use_mask_alignment = bool(cfg.get("use_mask_alignment", True))
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(max(1, min(8, os.cpu_count() or 8)))
        except Exception:
            pass
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.pixel_size = load_json(self.calibration_json)["pixel_size_mm_per_pixel"]
        self.region_types = load_json(self.region_types_json)
        # 兜底映射：只允许 1-4 四个有效工艺区域进入统计和提示词。
        # 其他像素值即使被 region_types.json 映射成文字，也会被白名单过滤掉。
        self.region_types.setdefault("1", "一般区域")
        self.region_types.setdefault("2", "牙孔区域")
        self.region_types.setdefault("3", "焊接区域")
        self.region_types.setdefault("4", "重点区域")

        self.standards = load_json(self.standard_json)
        print(f"[vision] loading SAHI model: {self.model_path}", flush=True)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov11",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
            image_size=self.model_image_size,
        )

    def release(self):
        if hasattr(self, "detection_model"):
            del self.detection_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def resolve_assets_by_index(self, view_index: int) -> Tuple[str, str]:
        template_path = str(Path(self.template_folder) / f"image{view_index}.jpg")
        mask_path = str(Path(self.mask_folder) / f"image{view_index}_mask.png")
        if not Path(template_path).exists():
            raise FileNotFoundError(f"模板不存在: {template_path}")
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"mask 不存在: {mask_path}")
        return template_path, mask_path

    def analyze_single_image(self, image_path: str, view_index: int):
        template_path, mask_path = self.resolve_assets_by_index(view_index)
        image = cv2.imread(str(image_path))
        template = cv2.imread(template_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or template is None or mask is None:
            raise ValueError(f"无法读取图像/模板或掩膜: {image_path}")
        mask_g = align_mask_to_image(image, template, mask) if self.use_mask_alignment else cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = get_sliced_prediction(
            image, self.detection_model,
            slice_height=self.slice_height, slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio, overlap_width_ratio=self.overlap_width_ratio,
        )
        defects = []
        boxes = []
        defect_id = 1
        region_count = defaultdict(int)
        region_max_area_mm2 = defaultdict(float)
        region_max_eq_diameter_mm = defaultdict(float)
        class_summary = defaultdict(int)
        is_pore_count = 0

        print(f"[vision] view {view_index}: raw detections = {len(result.object_prediction_list)}", flush=True)

        for obj in result.object_prediction_list:
            if obj.mask is None:
                continue
            mask_bool = obj.mask.bool_mask
            area_px = int(np.sum(mask_bool))
            area_mm2 = area_px * (self.pixel_size ** 2)
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            best_contour = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(best_contour)
            cx, cy = int(x), int(y)
            radius_px = int(round(r))
            dia_mm = 2 * r * self.pixel_size
            x1, y1, w, h = cv2.boundingRect(best_contour)
            x2 = x1 + w
            y2 = y1 + h
            candidate_box = [float(x1), float(y1), float(x2), float(y2)]
            cx_clamped = int(np.clip(cx, 0, mask_g.shape[1] - 1))
            cy_clamped = int(np.clip(cy, 0, mask_g.shape[0] - 1))
            region_val = str(mask_g[cy_clamped, cx_clamped])
            region_str = self.region_types.get(region_val, f"未知区域({region_val})")
            if is_background_or_unknown_region(region_str, region_val):
                continue
            is_pore = obj.category.name.lower() in ("pore", "porosity")
            status = "OK"
            std = self.standards.get(region_str, {})
            if is_pore and not std.get("allow_pores", True):
                status = "NG"
            elif (not is_pore) and not std.get("allow_defects", True):
                status = "NG"
            if status == "OK":
                max_area = std.get("max_pore_area_mm2", float("inf")) if is_pore else std.get("max_defect_area_mm2", float("inf"))
                if area_mm2 > max_area:
                    status = "NG"
            else:
                max_area = None
            if status == "OK":
                obs_r = std.get("observation_diameter_mm", 0) / 2
                max_num = std.get("max_pores_per_observation", float("inf")) if is_pore else std.get("max_defects_per_observation", float("inf"))
                if max_num > 0 and obs_r > 0:
                    near = count_defects_in_circle((cx, cy), defects, obs_r, self.pixel_size, max_area=max_area, is_pore=is_pore)
                    if near >= max_num:
                        status = "NG"
            defect_info = {
                "Defect ID": defect_id, "Region Type": region_str, "Region Value": region_val,
                "Diameter (mm)": round(dia_mm, 3), "Area (mm²)": round(area_mm2, 3),
                "X Position": cx, "Y Position": cy, "Radius (px)": radius_px,
                "IsPore": is_pore, "Status": status, "Category": obj.category.name,
                "Contour": best_contour.reshape(-1, 2).tolist(), "view_index": view_index,
                "defect_id": defect_id,
                "region_value": int(region_val) if str(region_val).isdigit() else region_val,
                "region_type": region_str,
                "x": float(cx),
                "y": float(cy),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "area_mm2": round(area_mm2, 4),
                "diameter_mm": round(dia_mm, 4),
                "is_pore": bool(is_pore),
                "defect_type": "气孔" if is_pore else "缺陷",
                "class_name": "气孔" if is_pore else "缺陷",
                "raw_class_name": obj.category.name,
                "confidence": round(float(getattr(obj, "score", 0.0).value if hasattr(getattr(obj, "score", None), "value") else getattr(obj, "score", 0.0) or 0.0), 4),
            }
            boxes.append(candidate_box)
            defects.append(defect_info)
            defect_id += 1
            region_count[region_str] += 1
            region_max_area_mm2[region_str] = max(region_max_area_mm2[region_str], area_mm2)
            region_max_eq_diameter_mm[region_str] = max(region_max_eq_diameter_mm[region_str], dia_mm)
            class_summary["Pore" if is_pore else "Defect"] += 1
            if is_pore:
                is_pore_count += 1

        print(f"[vision] view {view_index}: kept defects = {len(defects)}", flush=True)
        summary = {
            "total_defect_count": len(defects),
            "total_area_mm2": round(sum(d["Area (mm²)"] for d in defects), 3),
            "max_eq_diameter_mm": round(max([d["Diameter (mm)"] for d in defects], default=0), 3),
            "region_count": dict(region_count),
            "region_max_area_mm2": {k: round(v, 3) for k, v in region_max_area_mm2.items()},
            "region_max_eq_diameter_mm": {k: round(v, 3) for k, v in region_max_eq_diameter_mm.items()},
            "class_summary": dict(class_summary),
            "is_pore_count": is_pore_count, "view_index": view_index, "image_name": Path(image_path).name,
        }
        return defects, boxes, summary

    def run(self, image_paths: List[str]) -> Dict[str, Any]:
        ensure_dir(self.visual_output_dir)
        per_view = []
        boxes_by_view = []
        defect_meta_by_view = []
        view_items = []
        for i, image_path in enumerate(image_paths, start=1):
            print(f"[vision] analyzing view {i}: {image_path}", flush=True)
            defects, boxes, summary = self.analyze_single_image(image_path, i)
            per_view.append({"image_path": image_path, "summary": summary})
            boxes_by_view.append(boxes)
            defect_meta_by_view.append(defects)
            view_items.append({
                "view_index": i, "image_name": Path(image_path).name, "display_name": Path(image_path).name,
                "image_stem": Path(image_path).stem, "image_path": image_path,
                "total_defect_count": summary["total_defect_count"], "total_area_mm2": summary["total_area_mm2"],
                "max_eq_diameter_mm": summary["max_eq_diameter_mm"], "region_count": summary["region_count"], "visual_ready": True,
            })
            if bool(self.cfg.get("save_intermediate_visual_files", False)):
                save_json(self.visual_output_dir / f"view_{i:02d}_defects.json", defects)
                save_json(self.visual_output_dir / f"view_{i:02d}_summary.json", summary)
        aggregate_summary = aggregate_summaries(per_view)
        bundle = {
            "aggregate_summary": aggregate_summary,
            "boxes_by_view": boxes_by_view,
            "defect_meta_by_view": defect_meta_by_view,
            "view_items": view_items,
            "image_paths": image_paths,
        }
        if bool(self.cfg.get("save_intermediate_visual_files", False)):
            save_json(self.visual_output_dir / "aggregate_summary.json", aggregate_summary)
            save_json(self.visual_output_dir / "vision_bundle.json", bundle)
        return bundle




try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


def init_neo4j(cfg_json: Dict[str, Any]):
    global neo4j_driver
    if neo4j_driver is not None:
        return
    if GraphDatabase is None:
        raise RuntimeError("neo4j python驱动未安装，请先 pip install neo4j")
    neo4j_driver = GraphDatabase.driver(
        cfg_json["neo4j_uri"],
        auth=(cfg_json["neo4j_user"], cfg_json["neo4j_password"]),
    )
    neo4j_driver.verify_connectivity()


def ensure_graph_indexes(cfg_json: Dict[str, Any]):
    if neo4j_driver is None:
        return
    entity_index = cfg_json["entity_fulltext_index"]
    rule_index = cfg_json["rule_fulltext_index"]
    database = cfg_json["neo4j_database"]

    cyphers = [
        f"""
        CREATE FULLTEXT INDEX {entity_index} IF NOT EXISTS
        FOR (n:Entity)
        ON EACH [n.name, n.standard_name, n.raw_entity, n.source_rules]
        """,
        f"""
        CREATE FULLTEXT INDEX {rule_index} IF NOT EXISTS
        FOR (r:Rule)
        ON EACH [r.rule_id, r.name, r.source]
        """,
    ]
    with neo4j_driver.session(database=database) as session:
        for c in cyphers:
            try:
                session.run(c)
            except Exception as e:
                print(f"[graph][warn] create index failed: {e}", flush=True)


def load_graph_dictionary(cfg_json: Dict[str, Any]) -> Dict[str, Any]:
    database = cfg_json["neo4j_database"]
    cypher = """
    MATCH (n:Entity)
    RETURN labels(n) AS labels,
           coalesce(n.name, '') AS name,
           coalesce(n.standard_name, '') AS standard_name,
           coalesce(n.raw_entity, '') AS raw_entity
    LIMIT 50000
    """
    all_terms = set()
    by_label = defaultdict(set)
    with neo4j_driver.session(database=database) as session:
        rows = session.run(cypher)
        for r in rows:
            labels = r["labels"] or []
            candidates = [(r["name"] or "").strip(), (r["standard_name"] or "").strip(), (r["raw_entity"] or "").strip()]
            for term in candidates:
                if not term or len(term) < 2 or len(term) > 60:
                    continue
                all_terms.add(term)
                for lb in labels:
                    by_label[lb].add(term)
    for t in STATIC_HINT_TERMS:
        all_terms.add(t)
    return {"all_terms": sorted(all_terms, key=len, reverse=True), "by_label": by_label}


def split_rule_ids(text: str) -> List[str]:
    return re.findall(r"R\d+_\d+", text or "")


def extract_seed_terms(info_text: str, question: str, graph_dictionary: Dict[str, Any], max_terms: int = 12) -> List[str]:
    text = f"{info_text}\n{question}"
    found = []
    for term in graph_dictionary["all_terms"]:
        if term and term in text:
            found.append(term)
        if len(found) >= max_terms * 3:
            break
    found = unique_keep_order(found)
    priority_terms = []
    other_terms = []
    for t in found:
        if any(k in t for k in ["区域", "缺陷", "卷气", "补缩", "缩孔", "缩松", "气孔", "问题"]):
            priority_terms.append(t)
        else:
            other_terms.append(t)
    return (priority_terms + other_terms)[:max_terms]


def build_lucene_query(seed_terms: List[str], question: str) -> str:
    if seed_terms:
        return " OR ".join([f'"{x}"' for x in seed_terms])
    q = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", question or "").strip()
    return q[:80] if q else "缺陷"


def choose_display_name(item: Dict[str, Any]) -> str:
    return (item.get("standard_name") or item.get("name") or item.get("raw_entity") or "").strip()


def classify_entity(item: Dict[str, Any]) -> str:
    labels = item.get("labels", []) or []
    priority = ["Region", "Phenomenon", "Problem", "Factor", "Action", "Defect", "Cause", "Feature", "ProcessFactor", "MoldFactor", "Countermeasure", "Branch"]
    for p in priority:
        if p in labels:
            return p
    return "Other"


def dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for x in items:
        name = choose_display_name(x)
        cat = classify_entity(x)
        key = (cat, name)
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def search_seed_entities(cfg_json: Dict[str, Any], lucene_query: str, seed_terms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    entity_index = cfg_json["entity_fulltext_index"]
    database = cfg_json["neo4j_database"]
    cypher = f"""
    CALL db.index.fulltext.queryNodes('{entity_index}', $search_text) YIELD node, score
    WHERE node:Entity
    WITH node, score,
         CASE
           WHEN 'Defect' IN labels(node) THEN 6
           WHEN 'Problem' IN labels(node) THEN 5
           WHEN 'Phenomenon' IN labels(node) THEN 4
           WHEN 'Region' IN labels(node) THEN 4
           WHEN 'Factor' IN labels(node) THEN 4
           WHEN 'Action' IN labels(node) THEN 3
           WHEN 'Cause' IN labels(node) THEN 3
           WHEN 'ProcessFactor' IN labels(node) THEN 3
           WHEN 'MoldFactor' IN labels(node) THEN 3
           WHEN 'Countermeasure' IN labels(node) THEN 3
           WHEN 'Feature' IN labels(node) THEN 2
           ELSE 1
         END AS label_boost,
         CASE WHEN coalesce(node.layer, '') = 'evidence' THEN 1.5 ELSE 0 END AS layer_boost
    RETURN labels(node) AS labels,
           coalesce(node.entity_type, '') AS entity_type,
           coalesce(node.name, '') AS name,
           coalesce(node.standard_name, '') AS standard_name,
           coalesce(node.raw_entity, '') AS raw_entity,
           coalesce(node.source_rules, '') AS source_rules,
           coalesce(node.layer, '') AS layer,
           coalesce(node.graph_group, '') AS graph_group,
           score + label_boost + layer_boost AS final_score
    ORDER BY final_score DESC
    LIMIT $limit
    """
    try:
        with neo4j_driver.session(database=database) as session:
            rows = list(session.run(cypher, search_text=lucene_query, limit=max(1, min(limit, 20))))
            result = [r.data() for r in rows]
            if result:
                return result
    except Exception as e:
        print(f"[graph][warn] fulltext search failed: {e}", flush=True)

    if not seed_terms:
        return []

    fallback_terms = [t for t in seed_terms if len(t) >= 2][:8]
    cypher_fallback = """
    MATCH (n:Entity)
    WHERE any(term IN $terms WHERE
        coalesce(n.name, '') CONTAINS term OR
        coalesce(n.standard_name, '') CONTAINS term OR
        coalesce(n.raw_entity, '') CONTAINS term
    )
    RETURN labels(n) AS labels,
           coalesce(n.entity_type, '') AS entity_type,
           coalesce(n.name, '') AS name,
           coalesce(n.standard_name, '') AS standard_name,
           coalesce(n.raw_entity, '') AS raw_entity,
           coalesce(n.source_rules, '') AS source_rules,
           coalesce(n.layer, '') AS layer,
           coalesce(n.graph_group, '') AS graph_group,
           0.0 AS final_score
    LIMIT $limit
    """
    with neo4j_driver.session(database=database) as session:
        rows = session.run(cypher_fallback, terms=fallback_terms, limit=max(1, min(limit, 30)))
        return [r.data() for r in rows]


def search_defect_taxonomy(cfg_json: Dict[str, Any], defect_name: str, per_defect_limit: int = 10) -> List[Dict[str, Any]]:
    if not defect_name:
        return []
    database = cfg_json["neo4j_database"]
    cypher = """
    MATCH (n:Entity)
    WHERE any(lb IN labels(n) WHERE lb IN ['Cause', 'Feature', 'ProcessFactor', 'MoldFactor', 'Countermeasure'])
      AND (coalesce(n.raw_entity, '') CONTAINS $defect OR coalesce(n.name, '') CONTAINS $defect OR coalesce(n.standard_name, '') CONTAINS $defect)
    RETURN labels(n) AS labels,
           coalesce(n.name, '') AS name,
           coalesce(n.standard_name, '') AS standard_name,
           coalesce(n.raw_entity, '') AS raw_entity,
           coalesce(n.source_rules, '') AS source_rules
    LIMIT $limit
    """
    with neo4j_driver.session(database=database) as session:
        rows = session.run(cypher, defect=defect_name, limit=max(1, min(per_defect_limit, 20)))
        return [r.data() for r in rows]


def fetch_rule_nodes(cfg_json: Dict[str, Any], rule_ids: List[str]) -> List[Dict[str, Any]]:
    if not rule_ids:
        return []
    database = cfg_json["neo4j_database"]
    cypher = """
    MATCH (r:Rule)
    WHERE r.rule_id IN $rule_ids
    RETURN r.rule_id AS rule_id,
           coalesce(r.source, '') AS source,
           coalesce(r.name, '') AS name
    ORDER BY r.rule_id
    """
    with neo4j_driver.session(database=database) as session:
        rows = session.run(cypher, rule_ids=rule_ids)
        return [r.data() for r in rows]


def fetch_rule_evidence(cfg_json: Dict[str, Any], rule_ids: List[str], limit: int = 60) -> List[Dict[str, Any]]:
    if not rule_ids:
        return []
    database = cfg_json["neo4j_database"]
    cypher = """
    MATCH (e:Entity)
    WHERE coalesce(e.source_rules, '') <> '' AND any(rid IN $rule_ids WHERE e.source_rules CONTAINS rid)
    RETURN labels(e) AS labels,
           coalesce(e.entity_type, '') AS entity_type,
           coalesce(e.name, '') AS name,
           coalesce(e.standard_name, '') AS standard_name,
           coalesce(e.raw_entity, '') AS raw_entity,
           coalesce(e.source_rules, '') AS source_rules,
           coalesce(e.layer, '') AS layer
    LIMIT $limit
    """
    with neo4j_driver.session(database=database) as session:
        rows = session.run(cypher, rule_ids=rule_ids, limit=max(1, min(limit, 120)))
        return [r.data() for r in rows]


def format_graph_context(seed_terms, seed_entities, defect_knowledge, rule_nodes, rule_evidence) -> str:
    if not seed_entities and not defect_knowledge and not rule_evidence:
        return ""
    lines = [
        "【图谱证据】",
        "以下内容来自知识图谱检索，仅作为辅助证据与约束先验；若与当前样本统计冲突，必须以当前样本输入为准。"
    ]
    if seed_terms:
        lines += ["一、命中检索词：", "；".join(seed_terms[:10])]
    if seed_entities:
        lines.append("二、命中实体：")
        for item in seed_entities[:8]:
            cat = classify_entity(item)
            name = choose_display_name(item)
            if name:
                lines.append(f"- {name}（{cat}）")
    if defect_knowledge:
        lines.append("三、通用缺陷知识：")
        grouped = defaultdict(list)
        for item in defect_knowledge:
            grouped[classify_entity(item)].append(choose_display_name(item))
        for cat, zh in [("Cause", "相关原因"), ("Feature", "相关表现"), ("ProcessFactor", "工艺因素"), ("MoldFactor", "模具因素"), ("Countermeasure", "改进措施")]:
            vals = unique_keep_order([x for x in grouped.get(cat, []) if x])[:6]
            if vals:
                lines.append(f"- {zh}：{'；'.join(vals)}")
    if rule_evidence:
        lines.append("四、规则证据归并：")
        grouped = defaultdict(list)
        for item in rule_evidence:
            grouped[classify_entity(item)].append(choose_display_name(item))
        for cat, zh in [("Region", "相关区域"), ("Phenomenon", "相关现象"), ("Problem", "相关问题"), ("Factor", "相关因素"), ("Action", "可参考动作")]:
            vals = unique_keep_order([x for x in grouped.get(cat, []) if x])[:8]
            if vals:
                lines.append(f"- {zh}：{'；'.join(vals)}")
    if rule_nodes:
        lines.append("五、关联规则：")
        rule_texts = []
        for r in rule_nodes[:10]:
            rid = r.get("rule_id", "")
            src = r.get("source", "")
            rule_texts.append(f"{rid}（{src}）" if src else rid)
        if rule_texts:
            lines.append("；".join(rule_texts))
    text = "\n".join(lines).strip()
    if len(text) > 2600:
        text = text[:2600] + "\n（图谱证据已截断）"
    return text


def retrieve_graph_context(cfg_json: Dict[str, Any], info_text: str, question: str, graph_top_k: int = 10):
    init_neo4j(cfg_json)
    ensure_graph_indexes(cfg_json)
    graph_dictionary = load_graph_dictionary(cfg_json)

    seed_terms = extract_seed_terms(info_text, question, graph_dictionary, max_terms=12)
    lucene_query = build_lucene_query(seed_terms, question)
    seed_entities = dedup_items(search_seed_entities(cfg_json, lucene_query, seed_terms, limit=graph_top_k))

    defect_names = []
    for item in seed_entities:
        labels = item.get("labels", []) or []
        if "Defect" in labels or "Problem" in labels:
            name = choose_display_name(item)
            if name:
                defect_names.append(name)
    defect_names = unique_keep_order(defect_names)[:3]

    defect_knowledge = []
    for defect in defect_names:
        defect_knowledge.extend(search_defect_taxonomy(cfg_json, defect, per_defect_limit=10))
    defect_knowledge = dedup_items(defect_knowledge)

    rule_ids = []
    for item in seed_entities:
        rule_ids.extend(split_rule_ids(item.get("source_rules", "")))
    for item in defect_knowledge:
        rule_ids.extend(split_rule_ids(item.get("source_rules", "")))
    rule_ids = unique_keep_order(rule_ids)[:12]

    rule_nodes = fetch_rule_nodes(cfg_json, rule_ids)
    rule_evidence = dedup_items(fetch_rule_evidence(cfg_json, rule_ids, limit=60))
    graph_context = format_graph_context(seed_terms, seed_entities, defect_knowledge, rule_nodes, rule_evidence)

    graph_debug = {
        "enabled": True,
        "available": True,
        "status": "connected",
        "seed_terms": seed_terms,
        "lucene_query": lucene_query,
        "seed_entities": [choose_display_name(x) for x in seed_entities[:10]],
        "defect_names": defect_names,
        "rule_ids": rule_ids,
        "rule_hits": len(rule_nodes),
        "evidence_hits": len(rule_evidence),
    }
    return graph_context, graph_debug


def patch_force_single_gpu_build_llm(adapter_override: str | None):
    def _patched_build_llm(self, cfg: RegionAwareMMConfig):
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if cfg.load_in_4bit else None
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_base_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        if adapter_override:
            llm = PeftModel.from_pretrained(llm, adapter_override, is_trainable=False)
        elif cfg.llm_stage2_adapter_path and cfg.continue_train_stage2_adapter:
            llm = PeftModel.from_pretrained(llm, cfg.llm_stage2_adapter_path, is_trainable=False)
        return llm
    RegionAwareMultiModalModel._build_llm = _patched_build_llm


def wait_for_gpu(min_free_gb: float = 40.0, timeout_sec: int = 180):
    if not torch.cuda.is_available():
        return
    start = time.time()
    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        print(f"[llm] GPU free: {free_gb:.2f}/{total_gb:.2f} GiB", flush=True)
        if free_gb >= min_free_gb:
            return
        if time.time() - start > timeout_sec:
            print("[llm][warn] wait timeout, continue anyway", flush=True)
            return
        time.sleep(3)


def find_current_weight(cfg_json: Dict[str, Any], cfg: RegionAwareMMConfig):
    override = cfg_json.get("mm_weight_path")
    if override:
        p = Path(override)
        if p.is_dir():
            return "light", p
        if p.is_file():
            return "full", p
    best_dir = Path(cfg.output_dir) / "best"
    if (best_dir / "multimodal_heads.pt").exists():
        return "light", best_dir
    best_pt = Path(cfg.output_dir) / "best.pt"
    if best_pt.exists():
        return "full", best_pt
    epoch1_dir = Path(cfg.output_dir) / "epoch_1"
    if (epoch1_dir / "multimodal_heads.pt").exists():
        return "light", epoch1_dir
    epoch1_pt = Path(cfg.output_dir) / "epoch_1.pt"
    if epoch1_pt.exists():
        return "full", epoch1_pt
    raise FileNotFoundError("未找到可用的当前权重，请检查 mm_weight_path 或 output_dir")


def load_current_model(cfg_json: Dict[str, Any]) -> RegionAwareMultiModalModel:
    cfg = RegionAwareMMConfig()
    weight_type, weight_path = find_current_weight(cfg_json, cfg)
    print(f"[llm] current weight type: {weight_type}", flush=True)
    print(f"[llm] current weight path: {weight_path}", flush=True)

    adapter_override = None
    if weight_type == "light":
        adapter_dir = weight_path / "llm_adapter"
        if adapter_dir.exists():
            adapter_override = str(adapter_dir)

    patch_force_single_gpu_build_llm(adapter_override)
    model = RegionAwareMultiModalModel(cfg)

    if weight_type == "light":
        mm_heads = torch.load(weight_path / "multimodal_heads.pt", map_location="cpu")
        model.projector.load_state_dict(mm_heads["projector"], strict=True)
        model.param_encoder.load_state_dict(mm_heads["param_encoder"], strict=True)
        model.region_prior.load_state_dict(mm_heads["region_prior"], strict=True)
        if model.prefix_fusion is not None and mm_heads.get("prefix_fusion") is not None:
            model.prefix_fusion.load_state_dict(mm_heads["prefix_fusion"], strict=True)
        print(f"[llm] loaded lightweight heads from: {weight_path}", flush=True)
    else:
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[llm] missing keys: {len(missing)}", flush=True)
        print(f"[llm] unexpected keys: {len(unexpected)}", flush=True)
    model.eval()
    return model



# 重新加入视觉文字总结，并支持三种对比模式。

GENERAL_SYSTEM_PROMPT = """你是一名铝合金压铸气孔缺陷溯源助手。

硬性规则：
1. 最终回答必须直接从【结论】开始。
2. 最终回答必须且只能包含【结论】【溯源分析】【建议方向】三部分。
3. 缺陷类型统一按气孔处理。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。
5. 回答必须依据当前模式中明确提供的视觉信息、压铸参数、问题和图谱辅助证据；没有提供的视觉通道不能假设存在。
6. 图谱证据只辅助机理判断，不能覆盖当前样本；最终不要输出图谱命中列表、规则编号或调试信息。
7. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。
8. 不要输出思考过程，不要复述整段输入。
9. 【结论】必须是诊断判断句，必须直接回答问题，不能复制、改写或变相复述【问题】原句。
10. 【结论】不能出现问号，不能使用“如何、是否、应如何、哪些参数链、怎么判断”等提问式表达。
11. 【结论】必须包含明确优先级判断，例如“优先围绕……排查”“当前更应关注……”“主导矛盾更偏向……”。
12. 【溯源分析】三个编号点必须分别围绕“区域证据→参数链→综合优先级”展开，禁止三点内容重复。
13. 【建议方向】三个编号点必须分别对应“优先排查→联动复核→对照验证”，禁止重复同一句话。
14. 视觉层已经只保留有效工艺区域，最终回答不得扩展到当前视觉统计之外的区域。
15. 最终回答必须是自然语言分析，不得逐行复制输入中的统计字段，不得输出JSON、字典、等号清单或类似“区域缺陷统计={...}”的原始字段。
16. 【溯源分析】第1点只能解释区域证据的含义，不能把视觉层文字总结原样搬运。
"""

MODE_DEFINITIONS = {
    "token_text": {
        "mode_desc": "视觉文字总结 + 视觉/区域前缀token + 参数 + 问题",
        "include_visual_text": True,
        "use_visual_prefix": True,
        "use_region_prefix": True,
        "use_numeric_prefix": True,
    },
    "token_only": {
        "mode_desc": "视觉/区域前缀token + 参数 + 问题，不给视觉文字总结",
        "include_visual_text": False,
        "use_visual_prefix": True,
        "use_region_prefix": True,
        "use_numeric_prefix": True,
    },
    "text_only": {
        "mode_desc": "视觉文字总结 + 参数 + 问题，不输入视觉/区域前缀token",
        "include_visual_text": True,
        "use_visual_prefix": False,
        "use_region_prefix": False,
        "use_numeric_prefix": True,
    },
}

FORBIDDEN_OUTPUT_MARKERS = [
    "【当前样本统计】", "【多模态token】", "【当前诊断】", "当前图片token为",
    "坐标映射结果为", "当前数量分布为", "当前面积分布为", "当前极值分布为",
    "给出第一优先排查方向", "给出第二优先排查方向", "给出后续对比验证方向",
    "图谱命中列表", "规则编号", "调试信息",
]


def get_enabled_modes(cfg_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    requested = cfg_json.get("compare_modes") or cfg_json.get("modes")
    if not requested:
        return MODE_DEFINITIONS
    if isinstance(requested, str):
        requested = [x.strip() for x in requested.split(",") if x.strip()]
    out = {}
    for key in requested:
        if key not in MODE_DEFINITIONS:
            raise ValueError(f"未知对比模式: {key}，可选: {list(MODE_DEFINITIONS)}")
        out[key] = MODE_DEFINITIONS[key]
    return out


def compose_instruction_with_template(mode_key: str = "", mode_desc: str = "") -> str:
    mode_requirements = {
        "token_text": "当前模式同时提供视觉文字总结和视觉/区域前缀信息；回答时要结合视觉统计、区域风险、压铸参数和问题。",
        "token_only": "当前模式不提供视觉文字总结；视觉信息只来自视觉/区域前缀信息，但压铸参数和问题仍然通过文本提供，必须结合参数链回答。",
        "text_only": "当前模式不输入视觉/区域前缀信息；视觉信息只来自视觉文字总结，同时压铸参数和问题仍然通过文本提供，必须结合参数链回答。",
    }.get(mode_key, "严格依据当前输入作答。")
    return f"""{GENERAL_SYSTEM_PROMPT}

当前对比实验模式：{mode_key}（{mode_desc}）。
{mode_requirements}

输出前自我约束：
- 【结论】必须回答问题，不得把【问题】改写成结论。
- 【结论】第一句建议使用“当前样本应优先……”“当前更应……”“当前主导风险更偏向……”这类判断句。
- 不要说明当前使用了哪种输入模式。
请直接作答，不要解释格式要求。"""


def sanitize_graph_context_for_prompt(graph_context: str, max_chars: int = 700) -> str:
    """Keep GraphRAG mounted, but prevent rule ids/lists from dominating the prompt."""
    graph_context = (graph_context or "").strip()
    if not graph_context:
        return ""
    graph_context = re.sub(r"\n{3,}", "\n\n", graph_context).strip()
    graph_context = re.sub(r"\n?五、关联规则：.*", "", graph_context, flags=re.DOTALL).strip()
    graph_context = re.sub(r"R\d+_\d+(?:（[^）]*）)?", "", graph_context)
    graph_context = re.sub(r"；{2,}", "；", graph_context)
    if len(graph_context) > max_chars:
        graph_context = graph_context[:max_chars] + "\n（图谱辅助证据已截断，完整内容仅用于调试保存）"
    return graph_context


def strip_visual_summary_from_info_text(info_text: str) -> str:
    """Keep only sample id and process-parameter text in the user prompt."""
    if not info_text:
        return ""
    if "【压铸参数摘要】" not in info_text:
        return info_text.strip()
    sample_line = ""
    for line in info_text.splitlines():
        if line.strip().startswith("【样本编号】"):
            sample_line = line.strip()
            break
    param_block = "【压铸参数摘要】" + info_text.split("【压铸参数摘要】", 1)[1]
    return (sample_line + "\n" + param_block).strip() if sample_line else param_block.strip()



PARAM_PROMPT_ORDER = [
    "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
    "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
    "慢速行程", "环境温度", "环境湿度", "温度", "湿度",
    "探伤图编号", "样本编号",
]


def format_params_for_prompt(params: Dict[str, Any] | None) -> str:
    """Format cfg_json['params'] into user prompt text.

    修复点：配置里只有 params、没有 info_text 时，原脚本会导致【压铸参数】为空。
    """
    if not isinstance(params, dict) or not params:
        return ""

    lines: List[str] = []
    used = set()
    for key in PARAM_PROMPT_ORDER:
        if key in params and params[key] not in ("", None):
            lines.append(f"{key}={params[key]}")
            used.add(key)

    for key, value in params.items():
        if key in used or value in ("", None):
            continue
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}={value}")

    return "\n".join(lines).strip()


def build_info_text_for_prompt(cfg_json: Dict[str, Any]) -> str:
    """Build compact info text containing sample id and die-casting parameters."""
    sample_id = cfg_json.get("sample_id", cfg_json.get("workpiece_id", ""))
    params_text = format_params_for_prompt(cfg_json.get("params", {}))

    lines: List[str] = []
    if sample_id not in ("", None):
        lines.append(f"【样本编号】{sample_id}")
    if params_text:
        lines.append("【压铸参数摘要】")
        lines.append(params_text)
    return "\n".join(lines).strip()


def _compact_json(obj: Any) -> str:
    return json.dumps(obj or {}, ensure_ascii=False, separators=(",", ":"))


def build_visual_fact_block(summary: Dict[str, Any]) -> str:
    """
    Convert the filtered visual-layer output into a natural-language evidence summary.
    Do not use JSON/dict/equal-sign field style here, otherwise the LLM tends to copy
    the fields into 【溯源分析】 instead of explaining them.
    """
    s = clean_visual_summary(summary)
    region_count = filter_region_metric_dict(s.get("region_count"))
    region_area = filter_region_metric_dict(s.get("region_max_area_mm2"))
    region_dia = filter_region_metric_dict(s.get("region_max_eq_diameter_mm"))

    valid_region_set = set(region_count.keys()) | set(region_area.keys()) | set(region_dia.keys())
    valid_regions = [r for r in ALLOWED_REGION_NAMES if r in valid_region_set]
    high_regions = [r for r in valid_regions if r in ["牙孔区域", "焊接区域", "重点区域"]]
    general_regions = [r for r in valid_regions if r == "一般区域"]

    def fmt_count_items(d: Dict[str, Any]) -> str:
        if not d:
            return "未形成有效区域数量统计"
        return "、".join([f"{k}{int(v)}条" for k, v in d.items()])

    def fmt_float_items(d: Dict[str, Any], unit: str = "mm²") -> str:
        if not d:
            return "未形成有效区域极值统计"
        return "、".join([f"{k}{float(v):.3f}{unit}" for k, v in d.items()])

    count_regions = []
    if region_count:
        max_count = max(int(v) for v in region_count.values())
        count_regions = [k for k, v in region_count.items() if int(v) == max_count]

    area_regions = []
    if region_area:
        max_area = max(float(v) for v in region_area.values())
        area_regions = [k for k, v in region_area.items() if abs(float(v) - max_area) < 1e-9]

    dia_regions = []
    if region_dia:
        max_dia = max(float(v) for v in region_dia.values())
        dia_regions = [k for k, v in region_dia.items() if abs(float(v) - max_dia) < 1e-9]

    lines = [
        "【视觉层文字总结】",
        "以下内容是模型作答依据，不是最终回答模板；最终回答必须转化为自然语言分析。",
        "缺陷类型统一按气孔处理；视觉统计仅包含一般区域、牙孔区域、焊接区域和重点区域。",
    ]
    if valid_regions:
        lines.append("有效工艺区域包括" + "、".join(valid_regions) + "。")
    if high_regions:
        lines.append("高影响区域包括" + "、".join(high_regions) + "，这些区域在优先级判断中需要高于普通参考区域。")
    if general_regions:
        lines.append("普通参考区域包括" + "、".join(general_regions) + "，主要用于观察整体波动。")

    lines.extend([
        f"过滤后共保留{int(s.get('total_defect_count', 0) or 0)}条有效气孔，总面积约{s.get('total_area_mm2', 0)}mm²，最大等效直径约{s.get('max_eq_diameter_mm', 0)}mm。",
        "区域气孔数量分布为：" + fmt_count_items(region_count) + "。",
        "各区域最大面积为：" + fmt_float_items(region_area, "mm²") + "。",
        "各区域最大等效直径为：" + fmt_float_items(region_dia, "mm") + "。",
    ])

    trend_parts = []
    if count_regions:
        trend_parts.append("从数量看，" + "、".join(count_regions) + "是数量主导区域")
    if area_regions:
        trend_parts.append("从最大面积看，" + "、".join(area_regions) + "是面积风险关注区域")
    if dia_regions:
        trend_parts.append("从最大等效直径看，" + "、".join(dia_regions) + "是尺寸风险关注区域")
    if trend_parts:
        lines.append("区域趋势判断：" + "；".join(trend_parts) + "。")
    return "\n".join(lines).strip()


def build_user_prompt(
    info_text: str,
    question: str,
    graph_context: str = "",
    visual_fact_block: str = "",
    mode_key: str = "token_text",
    question_type_hint: str = "",
    params: Dict[str, Any] | None = None,
) -> str:
    info_text = strip_visual_summary_from_info_text((info_text or "").strip())
    params_text = format_params_for_prompt(params)

    # 关键修复：即使配置没有 info_text，也必须把 cfg_json["params"] 写入提示词。
    if not info_text and params_text:
        info_text = params_text
    elif params_text and not any(k in info_text for k in PARAM_PROMPT_ORDER if k in (params or {})):
        info_text = (info_text + "\n" + params_text).strip()

    question = (question or "").strip()
    graph_context = (graph_context or "").strip()
    visual_fact_block = (visual_fact_block or "").strip()
    question_type_hint = (question_type_hint or "").strip()

    visual_block = f"{visual_fact_block}\n\n" if visual_fact_block else ""
    graph_block = f"\n【图谱辅助证据】\n{graph_context}\n" if graph_context else ""
    qtype_block = f"\n【当前问题类型提示】\n{question_type_hint}\n" if question_type_hint else ""

    if mode_key == "token_text":
        boundary = """1. 当前视觉信息同时来自视觉层文字总结和视觉/区域前缀信息；压铸参数、图谱辅助证据和问题必须同时作为回答依据。
2. 视觉文字总结与视觉/区域前缀信息存在表达差异时，以过滤后的视觉层文字总结中的统计值为准。
3. 仅依据有效工艺区域进行缺陷区域分析，不扩展到视觉总结之外的区域；回答时必须把统计值转化为自然语言判断，不能输出原始统计字段。"""
    elif mode_key == "token_only":
        boundary = """1. 当前不提供视觉文字总结；图片、缺陷位置、区域归类和区域风险只通过视觉/区域前缀信息输入。
2. 压铸参数、图谱辅助证据和问题仍然通过文本输入，压铸参数也会通过参数前缀信息输入，必须结合参数链回答。
3. 不要编造总缺陷数、区域计数、面积、等效直径；如需描述区域信息，用趋势性表达。"""
    elif mode_key == "text_only":
        boundary = """1. 当前不输入视觉/区域前缀信息；视觉信息只来自视觉层文字总结。
2. 压铸参数、图谱辅助证据和问题仍然通过文本输入，压铸参数也会通过参数前缀信息输入，必须结合参数链回答。
3. 必须用自然语言显式使用视觉层文字总结中的区域数量分布、区域最大面积和区域最大等效直径，不能复制输入字段或输出字典。"""
    else:
        boundary = "1. 严格依据当前输入作答，不使用未提供的信息。"

    return f"""{visual_block}【压铸参数】
{info_text}
{qtype_block}{graph_block}
【问题】
{question}

【作答边界】
{boundary}
4. 缺陷统一按气孔处理，不要引入缩孔、缩松、裂纹、冷隔、夹杂、夹渣等非气孔缺陷。
5. 不要输出背景区域、未知区域、未知(0)、背景/未知、重叠区域，也不要引入当前视觉统计之外的区域。
6. 不要输出图谱命中列表、规则编号、调试信息或中间字段。
7. 不要输出【当前样本统计】【多模态token】【当前诊断】等字段。
8. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。

【必须遵守的输出格式】
第一行必须是【结论】。
【结论】必须直接给当前样本的判断，不允许复述或改写【问题】原句。
【结论】不允许以“针对”“对于”“关于”开头，不允许出现“？”、“如何”、“是否”、“应如何”、“哪些参数链”、“怎么判断”等提问式表达。
【结论】必须给出排查优先级或风险优先级，不能只列问题中的评价指标。
【溯源分析】下面必须写 1. 2. 3. 三个编号点，且三点不能重复：1用自然语言解释区域证据的含义，明确数量主导区域与风险关注区域；2分析高速充型、建压、压射压力、铸造压力等参数链；3给出区域风险与参数链结合后的综合优先级。
【建议方向】下面必须写 1. 2. 3. 三个编号点，且三点不能重复：1写优先排查的工艺链路；2写参数与区域缺陷的联动复核；3写同料号、同视角、同工况、前后样本或趋势收敛验证。
不要把编号点合并成一段话。

现在开始作答。"""

def _run_generate_with_optional_stream(
    model: RegionAwareMultiModalModel,
    generate_kwargs: Dict[str, Any],
    stream_to_stdout: bool = True,
) -> str:
    """
    Run llm.generate and optionally stream decoded text to terminal.
    This is only runtime visibility, not a detection/repair/evaluation step.
    """
    if not stream_to_stdout:
        output_ids = model.llm.generate(**generate_kwargs)
        return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    streamer = TextIteratorStreamer(
        model.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=1.0,
    )
    generate_kwargs = dict(generate_kwargs)
    generate_kwargs["streamer"] = streamer

    chunks: List[str] = []
    err_box: List[BaseException] = []

    def _worker():
        try:
            model.llm.generate(**generate_kwargs)
        except BaseException as e:
            err_box.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    last_tick = time.time()
    while t.is_alive():
        try:
            chunk = next(streamer)
            if chunk:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        except StopIteration:
            break
        except Exception:
            if time.time() - last_tick > 10:
                print("\n[llm] generation still running...", flush=True)
                last_tick = time.time()
    # Drain remaining text after generation completes.
    while True:
        try:
            chunk = next(streamer)
            if chunk:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        except StopIteration:
            break
        except Exception:
            break
    t.join()
    print("", flush=True)
    if err_box:
        raise err_box[0]
    return "".join(chunks)

@torch.no_grad()
def generate_one_compare_mode(
    model: RegionAwareMultiModalModel,
    instruction: str,
    user_prompt: str,
    images: Sequence[torch.Tensor],
    boxes_by_view: Sequence[Sequence[Sequence[float]]],
    defect_meta_by_view: Sequence[Sequence[Dict[str, Any]]],
    summary: Dict[str, Any],
    params: Dict[str, Any],
    view_items: Sequence[Dict[str, Any]] | None = None,
    use_visual_prefix: bool = True,
    use_region_prefix: bool = True,
    use_numeric_prefix: bool = True,
    max_new_tokens: int = 1280,
    generation_max_time_sec: float | None = None,
    stream_to_stdout: bool = True,
) -> str:
    """Generate with independently controlled visual/region/numeric prefix tokens."""
    view_items = list(view_items or [])
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model.tokenizer.truncation_side = "left"
    model.tokenizer.padding_side = "left"

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    prefix_parts = []
    if use_visual_prefix:
        prefix_parts.append(model.encode_sample_visual_tokens(images, boxes_by_view, view_items))
    if use_region_prefix:
        prefix_parts.append(model.encode_region_prior_tokens(summary, defect_meta_by_view, view_items))
    if use_numeric_prefix:
        prefix_parts.append(model.encode_numeric_tokens(params))

    if prefix_parts:
        prefix_tokens = model._compose_prefix_tokens(prefix_parts)
        prompt_ids = model.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max(128, model.cfg.max_seq_length - prefix_tokens.size(0)),
        ).input_ids.to(model.runtime_device)
        text_embeds = model.llm.get_input_embeddings()(prompt_ids).squeeze(0)
        text_embeds = text_embeds.to(dtype=model.llm_embed_dtype, device=model.runtime_device)
        inputs_embeds = torch.cat([prefix_tokens, text_embeds], dim=0).unsqueeze(0)
        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=model.runtime_device)
        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        if generation_max_time_sec and generation_max_time_sec > 0:
            generate_kwargs["max_time"] = float(generation_max_time_sec)
        return _run_generate_with_optional_stream(model, generate_kwargs, stream_to_stdout=stream_to_stdout)
    else:
        enc = model.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=model.cfg.max_seq_length,
        )
        input_ids = enc.input_ids.to(model.runtime_device)
        attention_mask = enc.attention_mask.to(model.runtime_device)
        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        if generation_max_time_sec and generation_max_time_sec > 0:
            generate_kwargs["max_time"] = float(generation_max_time_sec)
        return _run_generate_with_optional_stream(model, generate_kwargs, stream_to_stdout=stream_to_stdout)


def normalize_output(text: str) -> str:
    if not text:
        return text
    text = THINK_BLOCK_RE.sub("", text).strip()
    text = THINK_TAG_RE.sub("", text)
    text = re.sub(r"【图谱证据】.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*图谱检索结果.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*调试信息.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"现在请回答当前问题[:：]?\s*$", "", text).strip()
    m = re.search(r"【结论】", text)
    if m:
        text = text[m.start():].strip()
    return text.strip()


def summarize_graph_debug(graph_debug: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "available": graph_debug.get("available", False),
        "status": graph_debug.get("status", ""),
        "seed_terms": graph_debug.get("seed_terms", []),
        "seed_entities": graph_debug.get("seed_entities", []),
        "defect_names": graph_debug.get("defect_names", []),
        "rule_ids": graph_debug.get("rule_ids", []),
        "rule_hits": graph_debug.get("rule_hits", 0),
        "evidence_hits": graph_debug.get("evidence_hits", 0),
    }


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    cfg_json = load_json(CONFIG_JSON)
    cfg_json["enable_file_upload"] = False
    cfg_json.setdefault("save_intermediate_visual_files", False)
    questions = get_test_questions(cfg_json)
    if not questions:
        raise ValueError("配置中需要提供 question 或 questions。")

    modes = get_enabled_modes(cfg_json)

    print("[main] stage 1/2: vision / filtered region-token source", flush=True)
    image_paths = collect_image_paths(cfg_json)
    vision = VisualStageRunner(cfg_json)
    bundle = vision.run(image_paths)
    bundle = filter_visual_bundle(bundle)
    print(f"\n[done] visual output: {cfg_json['visual_output_dir']}", flush=True)
    vision.release()
    del vision
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print("\n" + "=" * 100)
    print("【视觉/区域token摘要：仅保留有效工艺区域】")
    print(json.dumps(bundle["aggregate_summary"], ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("【图谱/生成/上传状态】")
    print(json.dumps({
        "graph_rag": bool(cfg_json.get("use_graph_rag", False)),
        "compare_modes": list(modes.keys()),
        "valid_region_only": True,
        "file_upload": False,
        "save_intermediate_visual_files": bool(cfg_json.get("save_intermediate_visual_files", False)),
    }, ensure_ascii=False, indent=2))

    wait_for_gpu(min_free_gb=float(cfg_json.get("min_free_gpu_gb", 16.0)), timeout_sec=180)
    print("[main] stage 2/2: multimodal language model", flush=True)

    model = load_current_model(cfg_json)
    images = [preprocess_image(p, RegionAwareMMConfig().vision_image_size) for p in bundle["image_paths"]]
    result_dir = Path(cfg_json["result_output_dir"])
    ensure_dir(result_dir)

    visual_fact_block = build_visual_fact_block(bundle["aggregate_summary"])
    params_prompt_text = build_info_text_for_prompt(cfg_json)
    save_text(result_dir / "visual_fact_block.txt", visual_fact_block)
    save_text(result_dir / "params_prompt_block.txt", params_prompt_text)
    save_json(result_dir / "visual_summary_filtered.json", bundle["aggregate_summary"])

    cases = []
    graph_debug_all = []

    for idx, question in enumerate(questions, start=1):
        question_type, question_type_hint, focus_regions, mechanism_candidates = build_question_type_hint(question)
        print(f"[main] Q{idx:02d} detected question_type: {question_type}", flush=True)

        graph_context = ""
        graph_debug = {
            "enabled": bool(cfg_json.get("use_graph_rag", False)),
            "available": False,
            "status": "disabled",
            "seed_terms": [],
            "lucene_query": "",
            "seed_entities": [],
            "defect_names": [],
            "rule_ids": [],
            "rule_hits": 0,
            "evidence_hits": 0,
            "question_type": question_type,
            "focus_regions": focus_regions,
            "mechanism_candidates": mechanism_candidates,
        }

        if cfg_json.get("use_graph_rag", False):
            try:
                graph_context, graph_debug_base = retrieve_graph_context(
                    cfg_json=cfg_json,
                    info_text=build_info_text_for_prompt(cfg_json),
                    question=question,
                    graph_top_k=int(cfg_json.get("graph_top_k", 10)),
                )
                graph_debug.update(graph_debug_base)
                graph_debug["question_type"] = question_type
                graph_debug["focus_regions"] = focus_regions
                graph_debug["mechanism_candidates"] = mechanism_candidates
            except Exception as e:
                graph_debug["available"] = False
                graph_debug["status"] = f"graph retrieval failed: {e}"
                print(f"[graph][warn][Q{idx:02d}] retrieve_graph_context failed: {e}", flush=True)

        print_graph_evidence(idx, graph_context, graph_debug)
        graph_debug_all.append(graph_debug)

        graph_context_for_prompt = sanitize_graph_context_for_prompt(
            graph_context,
            max_chars=int(cfg_json.get("prompt_graph_max_chars", 700)),
        ) if cfg_json.get("use_graph_rag", False) else ""

        case = {
            "question_index": idx,
            "workpiece_id": cfg_json.get("workpiece_id", str(cfg_json.get("sample_id", "unknown"))),
            "weight_path": cfg_json.get("mm_weight_path", ""),
            "question": question,
            "question_type": question_type,
            "question_type_hint": question_type_hint,
            "focus_regions": focus_regions,
            "mechanism_candidates": mechanism_candidates,
            "visual_summary": bundle["aggregate_summary"],
            "visual_fact_block": visual_fact_block,
            "graph_context": graph_context,
            "graph_debug": graph_debug,
            "params": cfg_json.get("params", {}),
            "image_paths": bundle["image_paths"],
            "modes": {},
        }

        q_dir = result_dir / f"q{idx:02d}"
        ensure_dir(q_dir)
        save_json(q_dir / "graph_debug.json", graph_debug)
        save_text(q_dir / "graph_context.txt", graph_context or "未检索到图谱证据。")

        for mode_key, mode_cfg in modes.items():
            print(f"[llm] Q{idx:02d} mode={mode_key} start generate | max_new_tokens={int(cfg_json.get('max_new_tokens', 1280))} | stream={bool(cfg_json.get('stream_to_stdout', True))}", flush=True)
            _gen_t0 = time.time()
            prompt_visual_block = visual_fact_block if mode_cfg["include_visual_text"] else ""
            user_prompt = build_user_prompt(
                info_text=build_info_text_for_prompt(cfg_json),
                question=question,
                graph_context=graph_context_for_prompt,
                visual_fact_block=prompt_visual_block,
                mode_key=mode_key,
                question_type_hint=question_type_hint,
                params=cfg_json.get("params", {}),
            )
            instruction = compose_instruction_with_template(mode_key, mode_cfg["mode_desc"])
            raw_answer = generate_one_compare_mode(
                model=model,
                instruction=instruction,
                user_prompt=user_prompt,
                images=images,
                boxes_by_view=bundle["boxes_by_view"],
                defect_meta_by_view=bundle["defect_meta_by_view"],
                summary=bundle["aggregate_summary"],
                params=cfg_json.get("params", {}),
                view_items=bundle["view_items"],
                use_visual_prefix=bool(mode_cfg.get("use_visual_prefix", False)),
                use_region_prefix=bool(mode_cfg.get("use_region_prefix", False)),
                use_numeric_prefix=bool(mode_cfg.get("use_numeric_prefix", True)),
                max_new_tokens=int(cfg_json.get("max_new_tokens", 1280)),
                generation_max_time_sec=float(cfg_json.get("generation_max_time_sec", 180)),
                stream_to_stdout=bool(cfg_json.get("stream_to_stdout", True)),
            )
            print(f"[llm] Q{idx:02d} mode={mode_key} generate finished in {time.time() - _gen_t0:.1f}s", flush=True)
            answer = normalize_output(raw_answer)

            mode_result = {
                "mode": mode_key,
                "mode_desc": mode_cfg["mode_desc"],
                "include_visual_text": bool(mode_cfg["include_visual_text"]),
                "use_visual_prefix": bool(mode_cfg.get("use_visual_prefix", False)),
                "use_region_prefix": bool(mode_cfg.get("use_region_prefix", False)),
                "use_numeric_prefix": bool(mode_cfg.get("use_numeric_prefix", True)),
                "raw_answer": raw_answer,
                "final_answer": answer,
                "one_pass_generation": True,
                "user_prompt": user_prompt,
                "instruction": instruction,
            }
            case["modes"][mode_key] = mode_result

            mode_dir = q_dir / mode_key
            ensure_dir(mode_dir)
            save_json(mode_dir / "e2e_result.json", mode_result)
            save_text(mode_dir / "final_answer.txt", answer)
            save_text(mode_dir / "raw_answer.txt", raw_answer)
            save_text(mode_dir / "user_prompt.txt", user_prompt)

            print("\n" + "=" * 100)
            print(f"【Q{idx:02d} {mode_key}：{mode_cfg['mode_desc']}】")
            print(answer)
            print(f"\n[llm][Q{idx:02d}][{mode_key}] one_pass_generation=True", flush=True)

        save_json(q_dir / "case_result.json", case)
        cases.append(case)

    output_obj = {
        "schema_version": "e2e_compare_three_modes_v2",
        "config_json": CONFIG_JSON,
        "workpiece_id": cfg_json.get("workpiece_id", str(cfg_json.get("sample_id", "unknown"))),
        "weight_path": cfg_json.get("mm_weight_path", ""),
        "valid_region_only": True,
        "mode_definitions": modes,
        "visual_summary": bundle["aggregate_summary"],
        "visual_fact_block": visual_fact_block,
        "params_prompt_block": build_info_text_for_prompt(cfg_json),
        "cases": cases,
    }
    save_json(result_dir / "all_modes_results.json", output_obj)
    # 兼容旧文件名，但内容使用新结构。
    save_json(result_dir / "all_results.json", output_obj)
    save_json(result_dir / "graph_debug_all.json", graph_debug_all)

    with (result_dir / "e2e_compare_report.md").open("w", encoding="utf-8") as f:
        f.write("# 端到端三模式对比测试结果\n\n")
        f.write(f"## 工件ID\n{cfg_json.get('workpiece_id', str(cfg_json.get('sample_id', 'unknown')))}\n\n")
        f.write(f"## 使用权重\n{cfg_json.get('mm_weight_path', '')}\n\n")
        f.write("## 生成方式\n")
        f.write("- token_text：视觉文字总结 + 视觉/区域前缀token + 参数 + 问题。\n")
        f.write("- token_only：视觉/区域前缀token + 参数 + 问题，不给视觉文字总结。\n")
        f.write("- text_only：视觉文字总结 + 参数 + 问题，不输入视觉/区域前缀token。\n")
        f.write("- 视觉层仅保留一般区域、牙孔区域、焊接区域和重点区域进入统计、提示词和token元信息。\n\n")
        f.write("## 过滤后视觉层摘要\n```json\n")
        f.write(json.dumps(bundle["aggregate_summary"], ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        for case in cases:
            f.write(f"## Q{case['question_index']:02d} {case['question_type']}\n")
            f.write(case["question"] + "\n\n")
            f.write("### 图谱命中摘要\n```json\n")
            f.write(json.dumps(summarize_graph_debug(case["graph_debug"]), ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            for mode_key, r in case["modes"].items():
                f.write(f"### {mode_key}：{r['mode_desc']}\n\n")
                f.write(r["final_answer"] + "\n\n")

    print(f"\n[done] result output: {result_dir}", flush=True)
    print(f"[done] scoring input: {result_dir / 'all_modes_results.json'}", flush=True)


if __name__ == "__main__":
    main()
