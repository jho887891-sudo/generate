#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准端到端/纯文本语料统一评分脚本 v2
============================================================

用途：
1. 评分 e2e_compare_three_modes_graph_4b_v4_prompt_clean.py 生成的 all_modes_results.json；
2. 直接评分纯文本 LoRA 语料 JSONL，例如：instruction/input/output 格式；
3. 输出自动80分、人工20分模板、GPT人工评分审阅包、低分样本、重复率报告。

自动评分满分 80 分：
1. 格式与指令遵循：12
2. 问题相关性与回答完整性：8
3. 视觉/区域证据使用：14
4. 高影响区域与风险优先级：12
5. 参数链分析：14
6. 建议方向与可验证性：10
7. 禁忌项与幻觉控制：10

人工/专家合理性满分 20 分：
- 首次运行会生成 manual_score_template.json；
- 人工填完后，重新运行并指定 --manual-json，即得到最终 100 分。

推荐用法：
python standard_e2e_scoring_v2.py \
  --input /path/to/all_modes_results.json \
  --output-dir /path/to/scores

python standard_e2e_scoring_v2.py \
  --input /path/to/stage2_text_lora_pure_all_repaired.jsonl \
  --output-dir /path/to/corpus_scores

python standard_e2e_scoring_v2.py \
  --input /path/to/stage2_text_lora_pure_text_repaired.zip \
  --output-dir /path/to/corpus_scores
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# ============================================================
# 可按需修改的默认路径
# ============================================================

DEFAULT_INPUT_PATH = Path(
    "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q/qwen35_plus_graph_300q_20260523_093416/all_modes_results.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q/qwen35_plus_graph_300q_20260523_093416_score"
)

DEFAULT_MANUAL_JSON = Path(
    "/home/T7/wkq/end-to-end/compare_outputs/qwen35_plus_labelme_graph_300q/qwen35_plus_graph_300q_20260523_093416/manual_score_template.json"
)

SECTION_TITLES = ["【结论】", "【溯源分析】", "【建议方向】"]

HIGH_IMPACT_REGIONS = ["牙孔区域", "焊接区域", "重点区域"]
KNOWN_REGIONS = [
    "一般区域", "牙孔区域", "焊接区域", "重点区域", "厚大区域",
    "背景区域", "未知区域", "背景", "未知",
]
VALID_REGION_CANDIDATES = ["一般区域", "牙孔区域", "焊接区域", "重点区域", "厚大区域"]

PARAM_FIELDS = [
    "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
    "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
    "慢速行程", "环境温度", "环境湿度", "温度", "湿度",
]

# 评分用词，不是要求模型必须死记这些词，但自动评分需要抓取可解释线索。
CORE_FILLING_TERMS = ["高速速度", "高速行程", "高速时间", "充型", "金属液前沿", "卷气", "排气", "流动稳定", "流动前沿"]
CORE_PRESSURE_TERMS = ["建压时间", "压射压力", "铸造压力", "增压", "压实", "压力传递", "保压", "补缩压力"]
VALIDATION_TERMS = ["验证", "复核", "对照", "跟踪", "同料号", "同视角", "同工况", "前后样本", "趋势", "收敛", "批次"]
LINKAGE_TERMS = ["联动", "匹配", "协同", "共同影响", "参数链", "组合", "耦合", "同步"]
PRIORITY_TERMS = ["优先", "首先", "重点", "第一优先", "主线", "先从", "优先级"]

FORBIDDEN_TERMS_STRICT = [
    "图谱命中列表", "规则编号", "调试信息", "多模态token", "当前样本统计",
    "【多模态token】", "【当前诊断】", "坐标映射结果", "RAG调试", "debug",
]

# 当前任务默认所有缺陷统一按气孔处理。若答案把问题扩大为其他缺陷，会扣分。
NON_GAS_DEFECT_TERMS = ["缩孔", "缩松", "冷隔", "裂纹", "夹杂", "夹渣", "砂眼", "热裂", "冷裂"]

# 如果当前视觉统计中没出现这些区域，答案又把它们当“当前样本区域”分析，需要扣分。
UNSUPPORTED_REGION_TERMS = [
    "分型面", "溢流槽", "浇口附近", "热节区域", "热节",
    "加工面", "边缘区域", "中心区域", "筋条根部", "孔边区域",
]

QUESTION_START_RE = re.compile(r"^\s*(如何|是否|哪些|怎么|为什么|应如何|能否|有何|是不是)")


# ============================================================
# 基础工具
# ============================================================

def clean_text(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", clean_text(text))


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"JSONL解析失败: {path}:{lineno}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_text(text: str) -> str:
    return hashlib.sha1(clean_text(text).encode("utf-8")).hexdigest()


def number_skeleton(text: str) -> str:
    """把数字归一化，用于判断模板重复。"""
    t = compact_text(text)
    t = re.sub(r"[-+]?\d+(?:\.\d+)?", "<NUM>", t)
    t = re.sub(r"[A-Za-z0-9_./\\:-]{8,}", "<ID>", t)
    return t


# ============================================================
# 解析答案/视觉/参数/问题
# ============================================================

def split_sections(answer: str) -> Dict[str, str]:
    answer = clean_text(answer)
    result = {"conclusion": "", "analysis": "", "suggestion": ""}
    pattern = re.compile(r"【结论】(?P<c>.*?)【溯源分析】(?P<a>.*?)【建议方向】(?P<s>.*)", re.S)
    m = pattern.search(answer)
    if not m:
        return result
    result["conclusion"] = m.group("c").strip()
    result["analysis"] = m.group("a").strip()
    result["suggestion"] = m.group("s").strip()
    return result


def parse_loose_mapping(raw: str) -> Dict[str, float]:
    """解析 {"一般区域": 1} 或 一般区域:1；牙孔区域:2 这类结构。"""
    raw = clean_text(raw)
    if not raw:
        return {}

    # 先尝试 JSON / Python dict 风格
    cleaned = raw.strip()
    cleaned = cleaned.replace("，", ",").replace("：", ":")
    try_candidates = [cleaned]
    if "'" in cleaned and '"' not in cleaned:
        try_candidates.append(cleaned.replace("'", '"'))
    for cand in try_candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items() if _is_number(v) and not is_background_or_unknown(str(k))}
        except Exception:
            pass

    out: Dict[str, float] = {}
    for region in VALID_REGION_CANDIDATES + ["背景", "未知", "背景区域", "未知区域", "未知(0)"]:
        pat = rf"{re.escape(region)}\s*[:：=＝]\s*([-+]?\d+(?:\.\d+)?)"
        m = re.search(pat, raw)
        if m and not is_background_or_unknown(region):
            out[region] = float(m.group(1))
    return out


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def is_background_or_unknown(region: str) -> bool:
    return any(x in str(region) for x in ["背景", "未知", "0"])


def get_region_dict(summary: Dict[str, Any], key: str) -> Dict[str, float]:
    if not isinstance(summary, dict):
        return {}
    aliases = {
        "region_count": ["region_count", "区域缺陷统计", "区域气孔统计", "区域数量统计"],
        "region_max_area_mm2": ["region_max_area_mm2", "区域最大面积(mm²)", "区域最大面积(mm2)", "区域最大面积"],
        "region_max_eq_diameter_mm": ["region_max_eq_diameter_mm", "区域最大等效直径(mm)", "区域最大等效直径"],
    }
    raw = None
    for k in aliases.get(key, [key]):
        if k in summary:
            raw = summary.get(k)
            break
    if isinstance(raw, dict):
        out: Dict[str, float] = {}
        for k, v in raw.items():
            ks = str(k)
            if is_background_or_unknown(ks):
                continue
            try:
                out[ks] = float(v)
            except Exception:
                continue
        return out
    if isinstance(raw, str):
        return parse_loose_mapping(raw)
    return {}


def max_regions(metric: Dict[str, float]) -> List[str]:
    if not metric:
        return []
    max_v = max(metric.values())
    return [k for k, v in metric.items() if abs(v - max_v) < 1e-9]


def expected_visual_roles(summary: Dict[str, Any]) -> Dict[str, List[str]]:
    region_count = get_region_dict(summary, "region_count")
    region_area = get_region_dict(summary, "region_max_area_mm2")
    region_dia = get_region_dict(summary, "region_max_eq_diameter_mm")
    return {
        "quantity_leading_regions": max_regions(region_count),
        "area_risk_regions": max_regions(region_area),
        "diameter_risk_regions": max_regions(region_dia),
        "valid_regions": sorted(set(region_count) | set(region_area) | set(region_dia)),
    }


def extract_block(text: str, start_titles: List[str], end_titles: Optional[List[str]] = None) -> str:
    text = clean_text(text)
    start_pat = "|".join(re.escape(x) for x in start_titles)
    if end_titles:
        end_pat = "|".join(re.escape(x) for x in end_titles)
        pat = re.compile(rf"(?:{start_pat})\s*(.*?)(?=(?:{end_pat})|\Z)", re.S)
    else:
        pat = re.compile(rf"(?:{start_pat})\s*(.*)", re.S)
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def extract_question_from_text(text: str) -> str:
    text = clean_text(text)
    patterns = [
        r"【(?:用户问题|问题|提问)】\s*(.+?)(?=\n【|\Z)",
        r"(?:用户问题|问题|提问)\s*[:：]\s*(.+?)(?=\n|\Z)",
        r"question\s*[:：=]\s*(.+?)(?=\n|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            q = clean_text(m.group(1))
            # 只取第一段，避免把后续摘要吞进去
            q = q.split("\n")[0].strip()
            if q:
                return q
    # 兜底：找带问号的一行
    for line in text.splitlines():
        line = line.strip()
        if 4 <= len(line) <= 120 and ("？" in line or "?" in line):
            return line
    return ""


def extract_summary_from_text(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    summary: Dict[str, Any] = {}

    scalar_patterns = {
        "total_defect_count": [r"总(?:气孔|缺陷)条目\s*[=＝:：]\s*(\d+)", r"总数量\s*[=＝:：]\s*(\d+)"],
        "total_area_mm2": [r"总面积\s*\(?mm²?\)?\s*[=＝:：]\s*([-+]?\d+(?:\.\d+)?)"],
        "max_eq_diameter_mm": [r"最大等效直径\s*\(?mm\)?\s*[=＝:：]\s*([-+]?\d+(?:\.\d+)?)"],
    }
    for key, pats in scalar_patterns.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                try:
                    summary[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
                except Exception:
                    summary[key] = m.group(1)
                break

    mapping_patterns = {
        "region_count": [r"区域缺陷统计\s*[=＝:：]\s*(.+?)(?=\n|$)", r"区域气孔统计\s*[=＝:：]\s*(.+?)(?=\n|$)"],
        "region_max_area_mm2": [r"区域最大面积\s*\(?mm²?\)?\s*[=＝:：]\s*(.+?)(?=\n|$)"],
        "region_max_eq_diameter_mm": [r"区域最大等效直径\s*\(?mm\)?\s*[=＝:：]\s*(.+?)(?=\n|$)"],
    }
    for key, pats in mapping_patterns.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                summary[key] = parse_loose_mapping(m.group(1))
                break
    return summary


def extract_params_from_text(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    params: Dict[str, Any] = {}
    for p in PARAM_FIELDS:
        # 避免 温度 抢 环境温度，湿度 抢 环境湿度；但都保留也不影响评分
        pat = rf"(?<!环境){re.escape(p)}\s*[=＝:：]\s*([-+]?\d+(?:\.\d+)?)" if p in ["温度", "湿度"] else rf"{re.escape(p)}\s*[=＝:：]\s*([-+]?\d+(?:\.\d+)?)"
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            try:
                params[p] = float(val)
            except Exception:
                params[p] = val
    # 常见字段名归一
    if "温度" in params and "环境温度" not in params:
        params["环境温度"] = params["温度"]
    if "湿度" in params and "环境湿度" not in params:
        params["环境湿度"] = params["湿度"]
    return params


def normalize_record_to_cases_from_jsonl(records: List[Dict[str, Any]], source_name: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records, 1):
        input_text = clean_text(rec.get("input") or rec.get("prompt") or rec.get("question") or "")
        answer = clean_text(rec.get("output") or rec.get("answer") or rec.get("final_answer") or "")
        question = clean_text(rec.get("question") or extract_question_from_text(input_text) or f"语料样本{idx:05d}")

        summary = rec.get("visual_summary") if isinstance(rec.get("visual_summary"), dict) else extract_summary_from_text(input_text)
        params = rec.get("params") if isinstance(rec.get("params"), dict) else extract_params_from_text(input_text)

        sample_id = rec.get("sample_id") or rec.get("workpiece_id") or rec.get("work_id") or rec.get("id") or f"sample_{idx:05d}"
        cases.append({
            "question_index": idx,
            "sample_id": sample_id,
            "question": question,
            "question_type": rec.get("question_type", "jsonl_corpus"),
            "visual_summary": summary or {},
            "params": params or {},
            "modes": {
                "corpus_output": {
                    "mode_desc": "JSONL语料输出",
                    "final_answer": answer,
                }
            },
            "_raw_input": input_text,
            "_source_name": source_name,
        })
    return {"cases": cases, "source_format": "jsonl", "source_name": source_name}


def load_input_any(path: Path) -> Tuple[Dict[str, Any], List[Path]]:
    """返回统一格式 data，以及临时解压文件列表。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    cleanup_files: List[Path] = []

    if path.suffix.lower() == ".zip":
        tmpdir = Path(tempfile.mkdtemp(prefix="score_zip_"))
        cleanup_files.append(tmpdir)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmpdir)
        jsonl_files = sorted(tmpdir.rglob("*.jsonl"))
        json_files = sorted(tmpdir.rglob("*.json"))
        # 优先选 all_repaired / all / train+val 以外的总文件
        preferred = [p for p in jsonl_files if any(k in p.name.lower() for k in ["all", "merged", "pure"])]
        if preferred:
            chosen = preferred[0]
            records = list(iter_jsonl(chosen))
            return normalize_record_to_cases_from_jsonl(records, str(path)), cleanup_files
        if jsonl_files:
            # 如果没有 all，就把所有 jsonl 合并；去重由后续报告处理
            records: List[Dict[str, Any]] = []
            for jf in jsonl_files:
                records.extend(iter_jsonl(jf))
            return normalize_record_to_cases_from_jsonl(records, str(path)), cleanup_files
        if json_files:
            return load_input_any(json_files[0])[0], cleanup_files
        raise FileNotFoundError(f"ZIP中未找到 json/jsonl 文件: {path}")

    if path.suffix.lower() == ".jsonl":
        records = list(iter_jsonl(path))
        return normalize_record_to_cases_from_jsonl(records, str(path)), cleanup_files

    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict) and "cases" in data:
            data.setdefault("source_format", "e2e_all_modes")
            data.setdefault("source_name", str(path))
            return data, cleanup_files
        if isinstance(data, list):
            return normalize_record_to_cases_from_jsonl(data, str(path)), cleanup_files
        if isinstance(data, dict) and all(k in data for k in ["instruction", "input", "output"]):
            return normalize_record_to_cases_from_jsonl([data], str(path)), cleanup_files
        raise ValueError(f"无法识别JSON结构: {path}")

    raise ValueError(f"不支持的输入格式: {path}")


# ============================================================
# 通用判断函数
# ============================================================

def mentions_any(text: str, terms: List[str]) -> bool:
    return any(t in text for t in terms if t)


def count_mentions(text: str, terms: List[str]) -> int:
    return sum(1 for t in terms if t and t in text)


def answer_contains_region_role(answer: str, regions: List[str], role_terms: List[str]) -> bool:
    for r in regions:
        if r in answer and any(t in answer for t in role_terms):
            return True
    return False


def has_numbered_three_points(text: str) -> bool:
    text = clean_text(text)
    return all(re.search(rf"(^|\n)\s*{i}[\.、)]", text) for i in [1, 2, 3])


def has_at_least_numbered_points(text: str, n: int = 3) -> bool:
    points = re.findall(r"(^|\n)\s*\d+[\.、)]", clean_text(text))
    return len(points) >= n


def has_heavy_json_or_field_copy(answer: str) -> bool:
    # 大段字典或把输入字段用等号整体搬运，会让评分虚高，但不适合正式报告。
    if re.search(r"\{[^{}]{10,}\}", answer):
        return True
    field_like = re.findall(
        r"(区域缺陷统计|区域最大面积|区域最大等效直径|数量主导区域|面积风险关注区域|等效直径风险关注区域)\s*[=＝:：]",
        answer,
    )
    return len(field_like) >= 2


def repeated_content_penalty(answer: str) -> bool:
    lines = [re.sub(r"\s+", "", x) for x in answer.splitlines() if len(re.sub(r"\s+", "", x)) >= 18]
    seen: Dict[str, int] = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1
        if seen[line] >= 3:
            return True

    sections = split_sections(answer)
    points = re.findall(r"(?:^|\n)\s*\d+[\.、)]\s*(.+?)(?=(?:\n\s*\d+[\.、)])|\Z)", sections.get("analysis", ""), flags=re.S)
    points += re.findall(r"(?:^|\n)\s*\d+[\.、)]\s*(.+?)(?=(?:\n\s*\d+[\.、)])|\Z)", sections.get("suggestion", ""), flags=re.S)
    compact_points = [compact_text(p)[:80] for p in points if len(compact_text(p)) > 20]
    return len(compact_points) != len(set(compact_points))


def copied_question_as_conclusion(answer: str, question: str) -> bool:
    sections = split_sections(answer)
    conclusion = clean_text(sections.get("conclusion", ""))
    if not conclusion:
        return False
    if "？" in conclusion or "?" in conclusion:
        return True
    if QUESTION_START_RE.search(conclusion):
        return True
    if conclusion.startswith("针对") or conclusion.startswith("对于") or conclusion.startswith("关于"):
        # 不是所有“针对”都错，但用于语料输出时极易学成复述问题，严格扣。
        return True

    q = compact_text(question)
    c = compact_text(conclusion)
    if not q or not c or len(q) < 12:
        return False

    # 检测长片段复述
    hit = 0
    for chunk_len in [16, 24, 32]:
        if len(q) < chunk_len:
            continue
        step = max(1, chunk_len // 2)
        for i in range(0, len(q) - chunk_len + 1, step):
            chunk = q[i:i + chunk_len]
            if chunk and chunk in c:
                hit += 1
                break
    return hit >= 2


def unavailable_param_mentions(answer: str, params: Dict[str, Any]) -> List[str]:
    if not isinstance(params, dict):
        params = {}
    available = set(params.keys())
    # 温度/湿度别名互通
    if "环境温度" in available:
        available.add("温度")
    if "温度" in available:
        available.add("环境温度")
    if "环境湿度" in available:
        available.add("湿度")
    if "湿度" in available:
        available.add("环境湿度")
    return sorted({p for p in PARAM_FIELDS if p in answer and available and p not in available})


def unsupported_region_hits(answer: str, valid_regions: List[str]) -> List[str]:
    hits: List[str] = []
    valid = set(valid_regions)
    for r in VALID_REGION_CANDIDATES:
        if r in answer and r not in valid:
            hits.append(r)
    for r in UNSUPPORTED_REGION_TERMS:
        if r in answer and r not in valid:
            hits.append(r)
    return sorted(set(hits))


def focus_terms_from_question(question: str, summary: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, List[str]]:
    q = clean_text(question)
    roles = expected_visual_roles(summary)
    q_regions = [r for r in roles["valid_regions"] if r in q]
    q_params = [p for p in PARAM_FIELDS if p in q]
    q_topics = []
    topic_terms = [
        "气孔", "卷气", "排气", "充型", "建压", "压力", "高速", "慢速", "料饼", "环境", "温度", "湿度",
        "面积", "等效直径", "数量", "风险", "焊接", "牙孔", "宏观", "验证", "趋势", "稳定性",
    ]
    for t in topic_terms:
        if t in q:
            q_topics.append(t)
    return {"regions": q_regions, "params": q_params, "topics": q_topics}


# ============================================================
# 自动评分：80分
# ============================================================

def score_format(answer: str, question: str) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    text = clean_text(answer)
    sections = split_sections(text)

    if text.startswith("【结论】"):
        score += 2
    else:
        notes.append("未直接从【结论】开始")

    if all(t in text for t in SECTION_TITLES):
        score += 2
    else:
        notes.append("三段式标题不完整")

    if all(text.count(t) == 1 for t in SECTION_TITLES):
        score += 2
    else:
        notes.append("三段式标题出现次数异常")

    positions = [text.find(t) for t in SECTION_TITLES]
    if all(p >= 0 for p in positions) and positions == sorted(positions):
        score += 1
    else:
        notes.append("标题顺序错误")

    if all(len(compact_text(sections.get(k, ""))) >= 12 for k in ["conclusion", "analysis", "suggestion"]):
        score += 1
    else:
        notes.append("存在空段或内容过短")

    conclusion = sections.get("conclusion", "")
    if conclusion and not copied_question_as_conclusion(text, question):
        score += 1.5
    else:
        notes.append("结论疑似提问式表达或复述问题")

    if has_numbered_three_points(sections.get("analysis", "")):
        score += 1.25
    else:
        notes.append("溯源分析未按1/2/3编号")

    if has_numbered_three_points(sections.get("suggestion", "")):
        score += 1.25
    else:
        notes.append("建议方向未按1/2/3编号")

    return min(score, 12), notes


def score_relevance(answer: str, question: str, summary: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    text = clean_text(answer)
    focus = focus_terms_from_question(question, summary, params)

    if not copied_question_as_conclusion(text, question):
        score += 2
    else:
        notes.append("结论疑似复述问题")

    if mentions_any(text, ["判断", "优先", "主导", "风险", "排查", "复核", "说明", "表明"]):
        score += 1.5
    else:
        notes.append("缺少明确判断或优先级表达")

    if not focus["regions"] or any(r in text for r in focus["regions"]):
        score += 1.5
    else:
        notes.append("未回应问题关注区域")

    if not focus["params"] or any(p in text for p in focus["params"]):
        score += 1.0
    else:
        notes.append("未回应问题关注参数")

    if not focus["topics"] or count_mentions(text, focus["topics"]) >= min(2, len(focus["topics"])):
        score += 1.0
    else:
        notes.append("对问题主题回应不足")

    if "气孔" in text or "孔" in question:
        score += 0.75
    else:
        notes.append("未围绕气孔展开")

    if len(compact_text(text)) >= 180:
        score += 0.25
    else:
        notes.append("回答偏短，完整性不足")

    return min(score, 8), notes


def score_visual_evidence(answer: str, summary: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    text = clean_text(answer)
    roles = expected_visual_roles(summary)
    has_any_summary = any(roles[k] for k in ["quantity_leading_regions", "area_risk_regions", "diameter_risk_regions"])

    if not has_any_summary:
        # 如果输入本身没有视觉统计，不给满分，但避免错误地全部扣死。
        notes.append("输入未解析到区域视觉统计，视觉证据评分受限")
        return 6.0, notes

    if mentions_any(text, ["数量", "条目", "分布", "气孔数量", "区域"]):
        score += 2
    else:
        notes.append("未使用区域数量/分布信息")

    if roles["quantity_leading_regions"] and answer_contains_region_role(text, roles["quantity_leading_regions"], ["数量", "主导", "最多", "集中"]):
        score += 3
    else:
        notes.append("未正确体现数量主导区域")

    if mentions_any(text, ["面积", "最大面积"]):
        score += 2
    else:
        notes.append("未使用面积指标")

    if roles["area_risk_regions"] and answer_contains_region_role(text, roles["area_risk_regions"], ["面积", "风险", "突出", "极值", "最大"]):
        score += 2
    else:
        notes.append("未正确体现面积风险区域")

    if mentions_any(text, ["等效直径", "直径", "极值"]):
        score += 2
    else:
        notes.append("未使用最大等效直径指标")

    if roles["diameter_risk_regions"] and answer_contains_region_role(text, roles["diameter_risk_regions"], ["等效直径", "直径", "风险", "突出", "极值", "最大"]):
        score += 2
    else:
        notes.append("未正确体现等效直径风险区域")

    if mentions_any(text, ["不能只按数量", "不只按数量", "数量不能直接代表风险", "数量主导", "风险关注"]):
        score += 1
    else:
        notes.append("未体现数量主导与风险关注的区别")

    return min(score, 14), notes


def score_region_priority(answer: str, summary: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    text = clean_text(answer)
    roles = expected_visual_roles(summary)
    valid_regions = roles["valid_regions"]
    high_present = [r for r in HIGH_IMPACT_REGIONS if r in valid_regions]

    if high_present:
        if any(r in text for r in high_present):
            score += 3
        else:
            notes.append("未结合当前样本高影响区域")
    else:
        # 当前样本没有高影响区域统计时，要求不要硬编高影响区域。
        if not any(r in text for r in HIGH_IMPACT_REGIONS):
            score += 2
        else:
            notes.append("当前样本未提供高影响区域，却引入了牙孔/焊接/重点区域")

    if mentions_any(text, ["高影响区域", "风险关注", "优先关注", "敏感区域", "功能区域", "后续加工", "装配"]):
        score += 3
    else:
        notes.append("未体现高影响区域风险逻辑")

    if "一般区域" in valid_regions:
        if "一般区域" in text and mentions_any(text, ["整体波动", "参考", "数量主导", "充型波动", "整体稳定"]):
            score += 2
        else:
            notes.append("未合理说明一般区域的参考意义")
    else:
        score += 1.5

    risk_regions = sorted(set(roles["area_risk_regions"] + roles["diameter_risk_regions"]))
    if risk_regions and any(r in text for r in risk_regions) and mentions_any(text, ["优先", "风险", "复核", "排查", "关注"]):
        score += 3
    else:
        notes.append("未把极值风险区域转化为排查优先级")

    hits = unsupported_region_hits(text, valid_regions)
    if not hits:
        score += 1
    else:
        notes.append("引入当前样本未提供区域：" + "、".join(hits))

    return min(score, 12), notes


def score_parameter_chain(answer: str, question: str, params: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    text = clean_text(answer)
    focus = focus_terms_from_question(question, {}, params)
    mentioned_params = [p for p in PARAM_FIELDS if p in text]
    available_params = [p for p in PARAM_FIELDS if isinstance(params, dict) and p in params]

    if len(mentioned_params) >= 4:
        score += 4
    elif len(mentioned_params) >= 2:
        score += 2.5
        notes.append("参数提及数量偏少")
    elif available_params:
        notes.append("基本未结合压铸参数")
    else:
        score += 1.5
        notes.append("输入未解析到参数，参数链评分受限")

    if not focus["params"] or any(p in mentioned_params for p in focus["params"]):
        score += 1
    else:
        notes.append("问题关注参数没有进入分析")

    if mentions_any(text, CORE_FILLING_TERMS):
        score += 2.5
    else:
        notes.append("缺少高速充型/卷气链分析")

    if mentions_any(text, CORE_PRESSURE_TERMS):
        score += 3.5
    else:
        notes.append("缺少建压/压射压力/铸造压力链分析")

    if mentions_any(text, LINKAGE_TERMS):
        score += 1.5
    else:
        notes.append("缺少参数联动表达")

    if mentions_any(text, ["牙孔区域", "焊接区域", "重点区域", "一般区域", "区域"]) and mentions_any(text, ["参数", "高速", "建压", "压力", "充型"]):
        score += 1.5
    else:
        notes.append("参数链未与区域风险建立联系")

    bad_params = unavailable_param_mentions(text, params)
    if bad_params:
        # 不在此项重扣，主要在禁忌项扣。这里只提示并小扣。
        score -= min(1.0, 0.25 * len(bad_params))
        notes.append("提及输入未提供参数：" + "、".join(bad_params))

    return max(0.0, min(score, 14)), notes


def score_suggestion(answer: str) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    sections = split_sections(answer)
    suggestion = clean_text(sections.get("suggestion", ""))

    if has_numbered_three_points(suggestion):
        score += 2
    else:
        notes.append("建议方向未分三点")

    if mentions_any(suggestion, PRIORITY_TERMS):
        score += 2
    else:
        notes.append("建议缺少优先级")

    if mentions_any(suggestion, LINKAGE_TERMS):
        score += 2
    else:
        notes.append("建议缺少联动复核")

    if mentions_any(suggestion, VALIDATION_TERMS):
        score += 2
    else:
        notes.append("建议缺少对照验证")

    if not re.search(r"(调整到|控制在|提高到|降低到|设为|调至|设定为|保持在)\s*[-+]?\d", suggestion):
        score += 2
    else:
        notes.append("出现具体数值配方")

    return min(score, 10), notes


def score_forbidden(answer: str, question: str, summary: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 10.0
    notes: List[str] = []
    text = clean_text(answer)
    roles = expected_visual_roles(summary)
    valid_regions = roles["valid_regions"]

    for term in FORBIDDEN_TERMS_STRICT:
        if term in text:
            score -= 1.5
            notes.append(f"出现禁忌字段：{term}")

    # 背景/未知区域：若只是说“不计入”，也会干扰训练正式输出；轻扣。
    for term in ["背景区域", "未知区域", "未知(0)", "背景/未知"]:
        if term in text:
            score -= 1.0
            notes.append(f"出现不应进入正式报告的区域字段：{term}")

    if "气孔" in text or "气孔" in question:
        for term in NON_GAS_DEFECT_TERMS:
            if term in text:
                score -= 1.0
                notes.append(f"混入非气孔缺陷概念：{term}")

    hits = unsupported_region_hits(text, valid_regions)
    if hits:
        score -= min(3.0, 1.0 * len(hits))
        notes.append("出现当前样本未提供区域：" + "、".join(hits))

    bad_params = unavailable_param_mentions(text, params)
    if bad_params:
        score -= min(2.0, 0.5 * len(bad_params))
        notes.append("出现输入未提供参数：" + "、".join(bad_params))

    if has_heavy_json_or_field_copy(text):
        score -= 2.5
        notes.append("存在JSON/字典/等号字段搬运")

    if repeated_content_penalty(text):
        score -= 2.0
        notes.append("存在明显重复表达")

    if copied_question_as_conclusion(text, question):
        score -= 2.0
        notes.append("结论疑似复述问题")

    if re.search(r"R\d+_\d+", text):
        score -= 2.0
        notes.append("输出了图谱规则编号")

    if "<think>" in text.lower() or "</think>" in text.lower() or "思考过程" in text[:80]:
        score -= 3.0
        notes.append("出现think/思考过程")

    return max(0.0, round(score, 2)), notes


def score_answer(answer: str, question: str, summary: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    answer = clean_text(answer)
    items: List[Tuple[str, str, float, Callable[[], Tuple[float, List[str]]]]] = [
        ("format_instruction", "格式与指令遵循", 12, lambda: score_format(answer, question)),
        ("question_relevance", "问题相关性与回答完整性", 8, lambda: score_relevance(answer, question, summary, params)),
        ("visual_region_evidence", "视觉/区域证据使用", 14, lambda: score_visual_evidence(answer, summary)),
        ("region_priority", "高影响区域与风险优先级", 12, lambda: score_region_priority(answer, summary)),
        ("parameter_chain", "参数链分析", 14, lambda: score_parameter_chain(answer, question, params)),
        ("suggestion_verifiability", "建议方向与可验证性", 10, lambda: score_suggestion(answer)),
        ("forbidden_control", "禁忌项与幻觉控制", 10, lambda: score_forbidden(answer, question, summary, params)),
    ]
    breakdown = []
    for key, zh, max_score, fn in items:
        sc, notes = fn()
        sc = max(0.0, min(float(max_score), float(sc)))
        breakdown.append({"name": key, "zh_name": zh, "score": round(sc, 2), "max_score": max_score, "notes": notes})
    auto_score = round(sum(x["score"] for x in breakdown), 2)
    return {"auto_score_80": auto_score, "breakdown": breakdown}


# ============================================================
# 人工评分模板与审阅包
# ============================================================

def manual_score_from_item(item: Dict[str, Any]) -> Tuple[float, str]:
    if not isinstance(item, dict):
        return 0.0, ""
    if "score" in item:
        try:
            score = float(item.get("score", 0))
        except Exception:
            score = 0.0
        return max(0.0, min(20.0, score)), str(item.get("comment", ""))

    keys = [
        ("expert_reasonableness", 5),
        ("sample_consistency", 5),
        ("mechanism_depth", 4),
        ("verification_value", 3),
        ("report_usability", 3),
    ]
    total = 0.0
    for k, maxv in keys:
        try:
            total += max(0.0, min(float(maxv), float(item.get(k, 0))))
        except Exception:
            pass
    return max(0.0, min(20.0, total)), str(item.get("comment", ""))


def get_manual_score(manual: Dict[str, Any], qid: str, mode: str) -> Tuple[float, str]:
    if not isinstance(manual, dict):
        return 0.0, ""
    item = manual.get(qid, {}).get(mode, {})
    return manual_score_from_item(item)


def build_manual_template(data: Dict[str, Any], max_items: int = 0) -> Dict[str, Any]:
    tpl: Dict[str, Any] = {
        "_scoring_standard": {
            "manual_total": 20,
            "expert_reasonableness": "0-5 工程专家角度是否合理，是否符合压铸气孔溯源经验",
            "sample_consistency": "0-5 是否严格贴合当前样本视觉统计和参数，不乱引入未提供区域/参数",
            "mechanism_depth": "0-4 是否能解释区域现象与卷气、排气、压力传递、压实等机理",
            "verification_value": "0-3 建议是否能被现场复核、对照实验或趋势跟踪验证",
            "report_usability": "0-3 语言是否适合作为正式诊断报告，是否简洁稳定",
        }
    }
    count = 0
    for case in data.get("cases", []):
        if max_items and count >= max_items:
            break
        qid = f"q{int(case.get('question_index', 0) or 0):05d}"
        tpl[qid] = {}
        for mode in (case.get("modes", {}) or {}).keys():
            tpl[qid][mode] = {
                "expert_reasonableness": 0,
                "sample_consistency": 0,
                "mechanism_depth": 0,
                "verification_value": 0,
                "report_usability": 0,
                "score": 0,
                "comment": "人工/专家合理性评分，合计0-20分。",
            }
        count += 1
    return tpl


def build_manual_review_pack_json(
    data: Dict[str, Any],
    details: List[Dict[str, Any]],
    max_items: int = 50,
    prefer_low_score: bool = True,
) -> Dict[str, Any]:
    detail_map = {(d.get("question_id"), d.get("mode")): d for d in details}
    case_map = {f"q{int(c.get('question_index', 0) or 0):05d}": c for c in data.get("cases", [])}

    sorted_details = sorted(details, key=lambda x: float(x.get("auto_score_80", 0))) if prefer_low_score else details
    selected_keys: List[Tuple[str, str]] = []
    for d in sorted_details:
        key = (d.get("question_id"), d.get("mode"))
        if key not in selected_keys:
            selected_keys.append(key)
        if max_items and len(selected_keys) >= max_items:
            break

    grouped: Dict[str, List[str]] = defaultdict(list)
    for qid, mode in selected_keys:
        grouped[qid].append(mode)

    items: List[Dict[str, Any]] = []
    for qid, modes in grouped.items():
        case = case_map.get(qid, {})
        item = {
            "question_id": qid,
            "question_index": case.get("question_index", 0),
            "sample_id": case.get("sample_id", ""),
            "question": case.get("question", ""),
            "question_type": case.get("question_type", ""),
            "visual_summary": case.get("visual_summary", {}) or {},
            "params": case.get("params", {}) or {},
            "modes": [],
        }
        for mode in modes:
            d = detail_map.get((qid, mode), {})
            deduction_notes = []
            for b in d.get("breakdown", []) or []:
                notes = b.get("notes", []) or []
                if notes:
                    deduction_notes.append({
                        "item": b.get("zh_name", b.get("name", "")),
                        "score": b.get("score", 0),
                        "max_score": b.get("max_score", 0),
                        "notes": notes,
                    })
            item["modes"].append({
                "mode": mode,
                "mode_desc": d.get("mode_desc", ""),
                "auto_score_80": d.get("auto_score_80", 0),
                "auto_deduction_notes": deduction_notes,
                "answer": d.get("answer", ""),
                "manual_score_blank": {
                    "expert_reasonableness": 0,
                    "sample_consistency": 0,
                    "mechanism_depth": 0,
                    "verification_value": 0,
                    "report_usability": 0,
                    "score": 0,
                    "comment": "",
                },
            })
        items.append(item)

    return {
        "_usage": "本文件用于人工/ChatGPT补20分。默认只抽取低分样本，避免审阅包过大。",
        "manual_scoring_standard": {
            "total": 20,
            "expert_reasonableness": {"max": 5, "meaning": "工程专家角度是否合理，是否符合压铸气孔溯源经验。"},
            "sample_consistency": {"max": 5, "meaning": "是否严格贴合当前样本视觉统计和参数，不乱引入未提供区域/参数。"},
            "mechanism_depth": {"max": 4, "meaning": "是否能解释区域现象与卷气、排气、充型稳定性、压力传递、压实等机理。"},
            "verification_value": {"max": 3, "meaning": "建议是否能被现场复核、对照实验或趋势跟踪验证。"},
            "report_usability": {"max": 3, "meaning": "语言是否适合作为正式诊断报告，是否简洁、稳定、可用。"},
        },
        "return_json_schema": {
            "q00001": {
                "corpus_output": {
                    "expert_reasonableness": 0,
                    "sample_consistency": 0,
                    "mechanism_depth": 0,
                    "verification_value": 0,
                    "report_usability": 0,
                    "score": 0,
                    "comment": "人工评分理由，指出主要优点和扣分点。",
                }
            }
        },
        "items": items,
    }


def build_manual_review_pack_md(pack: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 人工20分评分审阅包\n\n")
    lines.append("请根据人工评分标准，为每个 question_id / mode 补充人工20分评分。最终只返回 JSON。\n\n")
    lines.append("## 人工评分标准（20分）\n\n")
    lines.append("| 项目 | 分值 | 说明 |\n|---|---:|---|\n")
    std = pack["manual_scoring_standard"]
    for key in ["expert_reasonableness", "sample_consistency", "mechanism_depth", "verification_value", "report_usability"]:
        lines.append(f"| {key} | {std[key]['max']} | {std[key]['meaning']} |\n")
    lines.append("\n## 返回 JSON 格式示例\n\n```json\n")
    lines.append(json.dumps(pack["return_json_schema"], ensure_ascii=False, indent=2))
    lines.append("\n```\n\n")

    for item in pack["items"]:
        lines.append(f"## {item['question_id']}：{item.get('question_type', '')}\n\n")
        lines.append(f"**样本ID：** {item.get('sample_id', '')}\n\n")
        lines.append(f"**问题：** {item.get('question', '')}\n\n")
        lines.append("**视觉统计摘要：**\n\n```json\n")
        lines.append(json.dumps(item.get("visual_summary", {}), ensure_ascii=False, indent=2))
        lines.append("\n```\n\n")
        lines.append("**压铸参数：**\n\n```json\n")
        lines.append(json.dumps(item.get("params", {}), ensure_ascii=False, indent=2))
        lines.append("\n```\n\n")
        for m in item.get("modes", []):
            lines.append(f"### {item['question_id']} - {m['mode']}：{m.get('mode_desc', '')}\n\n")
            lines.append(f"自动分：{m.get('auto_score_80', 0)}/80\n\n")
            if m.get("auto_deduction_notes"):
                lines.append("自动扣分提示：\n")
                for dn in m["auto_deduction_notes"]:
                    notes = "；".join(map(str, dn.get("notes", [])))
                    lines.append(f"- {dn.get('item', '')}: {dn.get('score', 0)}/{dn.get('max_score', 0)}，{notes}\n")
                lines.append("\n")
            lines.append("模型回答：\n\n```text\n")
            lines.append(m.get("answer", "").strip())
            lines.append("\n```\n\n")
            lines.append("请为此回答补人工20分，并给 comment。\n\n")
    return "".join(lines)


# ============================================================
# 统计报告
# ============================================================

def judgment_from_avg(avg: float) -> str:
    if avg >= 90:
        return "优秀：可作为主方法结果"
    if avg >= 80:
        return "良好：整体可用，仍需检查个别幻觉/泛化"
    if avg >= 70:
        return "一般：有诊断能力，但稳定性不足"
    if avg >= 60:
        return "偏弱：只能作为对照或中间结果"
    return "较差：不建议作为正式实验主结果"


def auto_judgment_from_80(avg: float) -> str:
    ratio = avg / 80 * 100
    return judgment_from_avg(ratio)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return round(xs[0], 2)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(xs[int(k)], 2)
    return round(xs[f] * (c - k) + xs[c] * (k - f), 2)


def build_duplicate_report(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    answers = [clean_text(d.get("answer", "")) for d in details]
    exact = Counter(sha1_text(a) for a in answers if a)
    skel = Counter(sha1_text(number_skeleton(a)) for a in answers if a)
    exact_dup_items = sum(c for c in exact.values() if c > 1)
    skel_dup_items = sum(c for c in skel.values() if c > 1)

    # 找重复样例
    hash_to_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in details:
        h = sha1_text(clean_text(d.get("answer", "")))
        if len(hash_to_examples[h]) < 5:
            hash_to_examples[h].append({
                "question_id": d.get("question_id"),
                "mode": d.get("mode"),
                "score": d.get("auto_score_80"),
                "question": d.get("question", ""),
            })
    top_exact = []
    for h, c in exact.most_common(20):
        if c <= 1:
            continue
        top_exact.append({"hash": h, "count": c, "examples": hash_to_examples.get(h, [])})

    return {
        "total_answers": len(answers),
        "unique_exact_answers": len(exact),
        "exact_duplicate_items": exact_dup_items,
        "exact_duplicate_rate": round(exact_dup_items / max(1, len(answers)), 4),
        "unique_number_skeleton_answers": len(skel),
        "skeleton_duplicate_items": skel_dup_items,
        "skeleton_duplicate_rate": round(skel_dup_items / max(1, len(answers)), 4),
        "top_exact_duplicates": top_exact,
    }


def build_note_counter(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for d in details:
        for b in d.get("breakdown", []) or []:
            for n in b.get("notes", []) or []:
                counter[str(n)] += 1
    return [{"note": k, "count": v} for k, v in counter.most_common()]


def rows_to_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def build_markdown_report(
    input_path: Path,
    manual_json: Optional[Path],
    summary_rows: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    duplicate_report: Dict[str, Any],
    note_counter: List[Dict[str, Any]],
    max_detail_items: int = 100,
) -> str:
    md: List[str] = []
    md.append("# 端到端/纯文本语料标准评分报告 v2\n\n")
    md.append(f"- 输入文件：`{input_path}`\n")
    md.append(f"- 人工评分文件：`{manual_json if manual_json else '未使用，人工分默认0'}`\n")
    md.append("- 自动评分满分80，人工/专家合理性满分20。\n\n")

    md.append("## 模式均分\n\n")
    md.append("| 模式 | 数量 | 自动均分/80 | P50 | P10 | 最低 | 最高 | 人工均分/20 | 最终均分/100 | 判断 |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in summary_rows:
        md.append(
            f"| {r['mode']} | {r['count']} | {r['auto_avg_80']} | {r['auto_p50_80']} | {r['auto_p10_80']} | "
            f"{r['auto_min_80']} | {r['auto_max_80']} | {r['manual_avg_20']} | {r['final_avg_100']} | {r['judgment']} |\n"
        )

    md.append("\n## 重复率报告\n\n")
    md.append(f"- 回答总数：{duplicate_report['total_answers']}\n")
    md.append(f"- 完全唯一回答数：{duplicate_report['unique_exact_answers']}\n")
    md.append(f"- 完全重复条目占比：{duplicate_report['exact_duplicate_rate']}\n")
    md.append(f"- 数字归一化后唯一骨架数：{duplicate_report['unique_number_skeleton_answers']}\n")
    md.append(f"- 数字归一化后重复条目占比：{duplicate_report['skeleton_duplicate_rate']}\n\n")

    md.append("## 高频扣分项 Top 30\n\n")
    md.append("| 扣分提示 | 次数 |\n|---|---:|\n")
    for item in note_counter[:30]:
        md.append(f"| {item['note']} | {item['count']} |\n")

    md.append("\n## 低分样本 Top 100\n\n")
    md.append("| 问题ID | 模式 | 自动分/80 | 问题 |\n|---|---|---:|---|\n")
    low = sorted(details, key=lambda x: float(x.get("auto_score_80", 0)))[:max_detail_items]
    for d in low:
        q = clean_text(d.get("question", "")).replace("|", " ")[:80]
        md.append(f"| {d.get('question_id')} | {d.get('mode')} | {d.get('auto_score_80')} | {q} |\n")

    md.append("\n## 低分样本详细扣分\n\n")
    for d in low:
        md.append(f"### {d.get('question_id')} - {d.get('mode')}\n\n")
        md.append(f"自动分：{d.get('auto_score_80')}/80，人工分：{d.get('manual_score_20')}/20，最终：{d.get('final_score_100')}/100\n\n")
        md.append(f"问题：{d.get('question', '')}\n\n")
        for b in d.get("breakdown", []) or []:
            notes = "；".join(b.get("notes", [])) if b.get("notes") else "无明显扣分"
            md.append(f"- {b.get('zh_name')}: {b.get('score')}/{b.get('max_score')}，{notes}\n")
        md.append("\n")
    return "".join(md)



# ============================================================
# 兼容 all_modes_results 直接 answer/raw_answer 结构
# ============================================================

DIRECT_ANSWER_DEFAULT_MODE = "text_only"


def _infer_direct_answer_mode(data: Dict[str, Any], case: Dict[str, Any]) -> str:
    """
    只用于兼容输入结构，不改变评分逻辑。
    标准格式为 case["modes"][mode]["final_answer"]。
    有些脚本输出为 case["answer"] / case["raw_answer"]，这里统一包装成一个 mode。
    """
    # 如果外部显式写了 mode，优先使用
    for key in ["mode", "compare_mode", "input_mode"]:
        v = case.get(key) or data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
    for key in ["mode", "compare_mode", "input_mode"]:
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 常见 SVP-Text / text-only 输出
    schema = str(data.get("schema_version", "")).lower()
    exp = str(meta.get("experiment_name", "")).lower()
    if "text" in schema or "text" in exp or "svp" in schema:
        return "text_only"

    return DIRECT_ANSWER_DEFAULT_MODE


def _normalize_modes_dict(modes: Any) -> Dict[str, Dict[str, Any]]:
    """
    兼容 modes 里 value 为字符串或其他字段名的情况。
    只做结构包装，不改回答内容，不改评分规则。
    """
    if not isinstance(modes, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for mode, val in modes.items():
        mode_name = str(mode)
        if isinstance(val, dict):
            r = dict(val)
            # 兼容部分脚本把回答写成 output/raw_answer/text
            if "final_answer" not in r:
                for k in ["answer", "raw_answer", "output", "text", "response"]:
                    if r.get(k):
                        r["final_answer"] = r.get(k)
                        break
            out[mode_name] = r
        else:
            out[mode_name] = {
                "mode_desc": "兼容模式：modes value 为非 dict，已包装为 final_answer",
                "final_answer": clean_text(val),
            }
    return out


def normalize_cases_for_scoring(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将不同 all_modes_results.json 结构统一成评分脚本内部需要的结构：
        case["modes"][mode]["final_answer"]

    兼容结构：
        case["answer"]
        case["raw_answer"]
        case["final_answer"]
        case["output"]
        case["response"]

    注意：这里只补齐输入结构，不改变任何评分函数和分值规则。
    """
    if not isinstance(data, dict):
        return data

    cases = data.get("cases", [])
    if not isinstance(cases, list):
        return data

    top_visual_summary = data.get("visual_summary", {}) if isinstance(data.get("visual_summary", {}), dict) else {}
    meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}

    compatible_count = 0
    for i, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            continue

        # 补 question_index，避免 q00000
        if not case.get("question_index"):
            case["question_index"] = i

        # 补 visual_summary：很多结果只在顶层保存一次视觉统计
        if not isinstance(case.get("visual_summary"), dict) or not case.get("visual_summary"):
            if top_visual_summary:
                case["visual_summary"] = top_visual_summary

        # 补 params：很多结果没有 case["params"]，但 user_prompt 里有参数块
        if not isinstance(case.get("params"), dict) or not case.get("params"):
            if case.get("user_prompt"):
                case["params"] = extract_params_from_text(case.get("user_prompt", ""))
            else:
                case["params"] = data.get("params", {}) if isinstance(data.get("params", {}), dict) else {}

        # 补 sample_id：不影响评分，只影响报告展示
        if not case.get("sample_id"):
            case["sample_id"] = (
                case.get("workpiece_id")
                or meta.get("sample_id")
                or meta.get("workpiece_id")
                or meta.get("experiment_name")
                or data.get("sample_id")
                or data.get("workpiece_id")
                or ""
            )

        # 标准 modes 结构
        modes = _normalize_modes_dict(case.get("modes", {}) or {})

        # 兼容 direct answer 结构
        if not modes:
            answer = (
                case.get("final_answer")
                or case.get("answer")
                or case.get("raw_answer")
                or case.get("output")
                or case.get("response")
                or ""
            )
            if clean_text(answer):
                mode_name = _infer_direct_answer_mode(data, case)
                modes = {
                    mode_name: {
                        "mode_desc": "兼容模式：由 case.answer / case.raw_answer 包装得到",
                        "final_answer": answer,
                    }
                }
                compatible_count += 1

        case["modes"] = modes

    data["_compatibility_normalization"] = {
        "enabled": True,
        "direct_answer_cases_wrapped": compatible_count,
        "note": "仅将 case.answer/raw_answer 等直接回答结构包装为 case.modes[mode].final_answer；评分函数和分值规则未改变。"
    }
    return data


# ============================================================
# 主流程
# ============================================================

def run_scoring(
    input_path: Path,
    output_dir: Path,
    manual_json: Optional[Path] = None,
    review_pack_limit: int = 50,
) -> None:
    data, _cleanup = load_input_any(input_path)
    data = normalize_cases_for_scoring(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    manual = load_json(manual_json) if manual_json and manual_json.exists() else {}

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for case in data.get("cases", []):
        q_index = int(case.get("question_index", 0) or 0)
        qid = f"q{q_index:05d}"
        question = clean_text(case.get("question", ""))
        summary = case.get("visual_summary", data.get("visual_summary", {})) or {}
        params = case.get("params", {}) or {}
        sample_id = case.get("sample_id", "")

        for mode, r in (case.get("modes", {}) or {}).items():
            answer = clean_text(r.get("final_answer") or r.get("answer") or "")
            auto = score_answer(answer, question, summary, params)
            manual_score, manual_comment = get_manual_score(manual, qid, mode)
            final_score = round(auto["auto_score_80"] + manual_score, 2)

            row = {
                "question_id": qid,
                "question_index": q_index,
                "sample_id": sample_id,
                "mode": mode,
                "mode_desc": r.get("mode_desc", ""),
                "auto_score_80": auto["auto_score_80"],
                "manual_score_20": round(manual_score, 2),
                "final_score_100": final_score,
                "manual_comment": manual_comment,
                "question": question,
            }
            rows.append(row)
            details.append({**row, "breakdown": auto["breakdown"], "answer": answer, "visual_summary": summary, "params": params})

    mode_scores: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        mode_scores[r["mode"]].append(r)

    summary_rows: List[Dict[str, Any]] = []
    for mode, rs in mode_scores.items():
        autos = [float(r["auto_score_80"]) for r in rs]
        manuals = [float(r["manual_score_20"]) for r in rs]
        finals = [float(r["final_score_100"]) for r in rs]
        final_avg = round(sum(finals) / max(1, len(finals)), 2)
        auto_avg = round(sum(autos) / max(1, len(autos)), 2)
        summary_rows.append({
            "mode": mode,
            "count": len(rs),
            "auto_avg_80": auto_avg,
            "auto_p50_80": percentile(autos, 0.5),
            "auto_p10_80": percentile(autos, 0.1),
            "auto_min_80": round(min(autos), 2) if autos else 0,
            "auto_max_80": round(max(autos), 2) if autos else 0,
            "manual_avg_20": round(sum(manuals) / max(1, len(manuals)), 2),
            "final_avg_100": final_avg,
            "judgment": judgment_from_avg(final_avg) if any(manuals) else auto_judgment_from_80(auto_avg),
        })
    summary_rows.sort(key=lambda x: x["auto_avg_80"], reverse=True)

    duplicate_report = build_duplicate_report(details)
    note_counter = build_note_counter(details)

    result_obj = {
        "input_path": str(input_path),
        "source_format": data.get("source_format", "unknown"),
        "manual_json": str(manual_json) if manual_json else "",
        "scoring_standard": {
            "auto_total": 80,
            "manual_total": 20,
            "auto_breakdown": [
                {"name": "格式与指令遵循", "score": 12},
                {"name": "问题相关性与回答完整性", "score": 8},
                {"name": "视觉/区域证据使用", "score": 14},
                {"name": "高影响区域与风险优先级", "score": 12},
                {"name": "参数链分析", "score": 14},
                {"name": "建议方向与可验证性", "score": 10},
                {"name": "禁忌项与幻觉控制", "score": 10},
            ],
            "manual_breakdown": [
                {"name": "工程专家合理性", "score": 5},
                {"name": "当前样本一致性", "score": 5},
                {"name": "机理分析深度", "score": 4},
                {"name": "现场验证价值", "score": 3},
                {"name": "报告可用性", "score": 3},
            ],
        },
        "summary_by_mode": summary_rows,
        "duplicate_report": duplicate_report,
        "note_counter": note_counter,
        "rows": rows,
        "details": details,
    }

    save_json(output_dir / "score_results.json", result_obj)
    save_json(output_dir / "score_details.json", details)
    save_json(output_dir / "score_summary_by_mode.json", summary_rows)
    save_json(output_dir / "duplicate_report.json", duplicate_report)
    save_json(output_dir / "note_counter.json", note_counter)
    save_json(output_dir / "manual_score_template.json", build_manual_template(data))

    low_details = sorted(details, key=lambda x: float(x.get("auto_score_80", 0)))[:200]
    write_jsonl(output_dir / "low_score_samples.jsonl", low_details)

    review_pack = build_manual_review_pack_json(data, details, max_items=review_pack_limit, prefer_low_score=True)
    save_json(output_dir / "manual_review_pack_for_gpt.json", review_pack)
    (output_dir / "manual_review_pack_for_gpt.md").write_text(build_manual_review_pack_md(review_pack), encoding="utf-8")

    rows_to_csv(
        output_dir / "score_results.csv",
        rows,
        [
            "question_id", "question_index", "sample_id", "mode", "mode_desc",
            "auto_score_80", "manual_score_20", "final_score_100",
            "manual_comment", "question",
        ],
    )
    rows_to_csv(
        output_dir / "score_summary_by_mode.csv",
        summary_rows,
        [
            "mode", "count", "auto_avg_80", "auto_p50_80", "auto_p10_80",
            "auto_min_80", "auto_max_80", "manual_avg_20", "final_avg_100", "judgment",
        ],
    )

    report_md = build_markdown_report(input_path, manual_json, summary_rows, details, duplicate_report, note_counter)
    (output_dir / "score_report.md").write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 100)
    print("评分完成：standard_e2e_scoring_v2.py")
    print("=" * 100)
    print(f"[done] input       : {input_path}")
    print(f"[done] output_dir  : {output_dir}")
    print(f"[done] result json : {output_dir / 'score_results.json'}")
    print(f"[done] result csv  : {output_dir / 'score_results.csv'}")
    print(f"[done] report      : {output_dir / 'score_report.md'}")
    print(f"[done] low samples : {output_dir / 'low_score_samples.jsonl'}")
    print(f"[done] duplicates  : {output_dir / 'duplicate_report.json'}")
    print(f"[done] manual tpl  : {output_dir / 'manual_score_template.json'}")
    print(f"[done] GPT review  : {output_dir / 'manual_review_pack_for_gpt.md'}")
    print("=" * 100)
    for r in summary_rows:
        print(f"{r['mode']}: auto_avg={r['auto_avg_80']}/80, p50={r['auto_p50_80']}, min={r['auto_min_80']}, max={r['auto_max_80']}, n={r['count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端/纯文本语料统一评分脚本 v2")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="输入 all_modes_results.json / JSONL / ZIP")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--manual-json", type=str, default="", help="人工20分JSON；不传则自动查找输出目录下 manual_score_template.json")
    parser.add_argument("--review-pack-limit", type=int, default=50, help="GPT人工审阅包最多抽取多少条低分样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    manual_json: Optional[Path]
    if args.manual_json:
        manual_json = Path(args.manual_json)
    else:
        candidate = output_dir / "manual_score_template.json"
        manual_json = candidate if candidate.exists() else None

    print("=" * 100)
    print("标准端到端/纯文本语料统一评分脚本 v2")
    print("=" * 100)
    print(f"[Path] input       = {input_path}")
    print(f"[Path] output_dir  = {output_dir}")
    print(f"[Path] manual_json = {manual_json if manual_json else '未使用，人工分默认0'}")
    print("=" * 100)

    run_scoring(
        input_path=input_path,
        output_dir=output_dir,
        manual_json=manual_json,
        review_pack_limit=max(1, int(args.review_pack_limit)),
    )


if __name__ == "__main__":
    main()
