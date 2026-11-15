# -*- coding: utf-8 -*-
"""
二次微调语料清洗脚本：文件夹输入版 / numeric_id / 保留缺陷与参数信息版

输入：
- 一个文件夹，里面是需要清洗的 .json / .jsonl 语料。

输出：
- 一个文件夹，里面是清洗后的可训练语料。

关键修正：
1. 不再因为一行里出现“背景/未知区域”就删除整行；
2. 只从区域统计字段中删除“背景/未知区域”对应项；
3. 保留一般区域、牙孔区域、焊接区域、重点区域等有效缺陷信息；
4. 保留全部压铸参数信息；
5. 温度 / 空气温度 / 温度(℃) 统一为环境温度；
6. 湿度 / 空气湿度 / 湿度(%) 统一为环境湿度；
7. ID 统一只保留数字：47243-OK / 47243-NG -> 47243；
8. 缺陷类型统一按气孔；
9. 输出适合纯文本 Stage2 LoRA 的 instruction/input/output 数据；
10. 同时输出 full 版本，保留 image_paths / view_items / summary / params 等字段，方便后续多模态使用。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT_DIR = "/home/T7/wkq/end-to-end/model_data/stage2_second_sft"
DEFAULT_OUTPUT_DIR = "/home/T7/wkq/end-to-end/model_data/stage2_text_lora_cleaned_numeric_id_v2"

SEED = 3407
TRAIN_RATIO = 0.90

HIGH_IMPACT_REGIONS = ["牙孔区域", "焊接区域", "重点区域"]

IGNORED_REGION_WORDS = [
    "背景",
    "未知区域",
    "未知区域(0)",
    "未知(0)",
    "背景/未知",
]

NON_PORE_WORDS = [
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

SYSTEM_INSTRUCTION = (
    "你是一名铝合金压铸气孔缺陷溯源助手。"
    "必须严格按照【结论】【溯源分析】【建议方向】三部分作答。"
    "缺陷类型统一按气孔处理。"
    "牙孔区域、焊接区域、重点区域属于高影响区域；一般区域只作为整体波动参考。"
    "溯源分析必须结合缺陷数量、区域缺陷统计、区域最大面积、区域最大等效直径和压铸参数链。"
    "建议方向只给宏观排查方向、联动关系和优先级，不给具体参数设定值。"
)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def normalize_numeric_id(x: Any) -> str:
    s = str(x).strip()
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def clean_think(text: Any) -> str:
    text = str(text)
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def replace_non_pore_words(text: str) -> str:
    for w in NON_PORE_WORDS:
        text = text.replace(w, "气孔")
    return text


def normalize_region_name(name: Any) -> str:
    s = str(name).strip()
    mapping = {
        "牙孔": "牙孔区域",
        "焊接": "焊接区域",
        "重点": "重点区域",
        "general": "一般区域",
        "region1": "一般区域",
        "Region 1": "一般区域",
    }
    return mapping.get(s, s)


def is_ignored_region(name: Any) -> bool:
    s = str(name).strip()
    return any(w in s for w in IGNORED_REGION_WORDS) or s in {"0", "unknown", "background"}


def filter_region_dict(d: Any) -> Dict[str, Any]:
    """
    从字典形式的区域统计中删除背景/未知区域，
    保留一般区域、牙孔区域、焊接区域、重点区域等有效项。
    """
    if not isinstance(d, dict):
        return {}

    out = {}
    for k, v in d.items():
        region = normalize_region_name(k)
        if is_ignored_region(region):
            continue
        out[region] = v

    return out


def clean_region_kv_string(value: str) -> str:
    """
    清洗类似：
    一般区域:181；牙孔区域:43；焊接区域:24；背景:18；未知区域(0):9

    只删除背景/未知项，不删除整行。
    """
    value = str(value).strip()
    if not value:
        return value

    # 如果是 JSON dict 字符串，优先尝试 json.loads
    if value.startswith("{") and value.endswith("}"):
        try:
            obj = json.loads(value)
            obj = filter_region_dict(obj)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    # 分隔符兼容：； ; ， , 
    parts = re.split(r"[；;,，]", value)
    kept = []

    for part in parts:
        p = part.strip()
        if not p:
            continue

        # 没有冒号的直接保留，除非是背景/未知
        if ":" not in p and "：" not in p:
            if is_ignored_region(p):
                continue
            kept.append(p)
            continue

        if "：" in p:
            k, v = p.split("：", 1)
            sep = "："
        else:
            k, v = p.split(":", 1)
            sep = ":"

        region = normalize_region_name(k)
        if is_ignored_region(region):
            continue

        kept.append(f"{region}{sep}{v.strip()}")

    return "；".join(kept)


def clean_region_line(line: str) -> str:
    """
    对区域相关行进行清洗，保留有效区域。
    """
    if "=" not in line:
        return line

    key, value = line.split("=", 1)

    region_line_keys = [
        "区域缺陷统计",
        "区域最大面积",
        "区域最大等效直径",
        "区域分布",
        "region_count",
        "region_max_area",
        "region_max_eq_diameter",
    ]

    if any(k in key for k in region_line_keys):
        cleaned = clean_region_kv_string(value)
        return f"{key}={cleaned}"

    return line


def normalize_temperature_humidity_text(text: str) -> str:
    text = text.replace("温度(℃)", "环境温度")
    text = text.replace("湿度(%)", "环境湿度")
    text = text.replace("空气温度", "环境温度")
    text = text.replace("空气湿度", "环境湿度")
    text = text.replace("温度=", "环境温度=")
    text = text.replace("湿度=", "环境湿度=")
    return text


def sanitize_existing_input(text: str) -> str:
    """
    清洗已有完整 input。
    注意：不删除整行缺陷信息，只清洗区域字段中的背景/未知项。
    """
    text = clean_think(text)
    text = replace_non_pore_words(text)
    text = normalize_temperature_humidity_text(text)

    lines = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            lines.append(line)
            continue

        # 区域规则行如果提到背景/未知，改成不包含背景/未知字样
        if raw.startswith("【区域规则】"):
            lines.append("【区域规则】牙孔区域、焊接区域、重点区域为高影响区域；一般区域只作为整体波动参考。")
            continue

        # 区域统计/极值行：只删除背景/未知的项
        if any(k in raw for k in ["区域缺陷统计", "区域最大面积", "区域最大等效直径", "区域分布", "region_count"]):
            lines.append(clean_region_line(raw))
            continue

        # 普通说明里如果只是在说背景/未知忽略，就删掉这一句
        if any(w in raw for w in IGNORED_REGION_WORDS):
            # 如果不是区域统计行，直接跳过这一行，避免模型学会输出背景/未知
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def remove_background_unknown_sentences(text: str) -> str:
    """
    用于 output：删除包含背景/未知的分句。
    """
    parts = re.split(r"(?<=[。；;！!？?\n])", text)
    kept = []

    for p in parts:
        if any(w in p for w in IGNORED_REGION_WORDS):
            continue
        kept.append(p)

    return "".join(kept).strip()


def normalize_params(params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}

    out = dict(params)

    for k in ["环境温度", "空气温度", "温度", "温度(℃)"]:
        if k in out and out[k] not in [None, ""]:
            out["环境温度"] = out[k]
            break

    for k in ["环境湿度", "空气湿度", "湿度", "湿度(%)"]:
        if k in out and out[k] not in [None, ""]:
            out["环境湿度"] = out[k]
            break

    for k in ["探伤图编号", "样本编号", "workpiece_id", "sample_id"]:
        if k in out and out[k] not in [None, ""]:
            out[k] = normalize_numeric_id(out[k])

    return out


def get_from_many(obj: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        if k in obj and obj[k] not in [None, ""]:
            return obj[k]
    return default


def extract_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    summary = obj.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    merged = {}
    merged.update(summary)

    for key in [
        "总缺陷条目",
        "总面积(mm²)",
        "最大等效直径(mm)",
        "区域缺陷统计",
        "区域最大面积(mm²)",
        "区域最大等效直径(mm)",
        "total_defect_count",
        "total_area_mm2",
        "max_eq_diameter_mm",
        "region_count",
        "region_max_area",
        "region_max_eq_diameter",
    ]:
        if key in obj and key not in merged:
            merged[key] = obj[key]

    return merged


def build_defect_summary_text(obj: Dict[str, Any]) -> str:
    summary = extract_summary(obj)

    total_count = get_from_many(summary, ["总缺陷条目", "total_defect_count", "total_count"], "")
    total_area = get_from_many(summary, ["总面积(mm²)", "total_area_mm2", "total_area"], "")
    max_diam = get_from_many(summary, ["最大等效直径(mm)", "max_eq_diameter_mm", "max_eq_diameter"], "")

    region_count = filter_region_dict(summary.get("区域缺陷统计") or summary.get("region_count") or {})
    region_max_area = filter_region_dict(summary.get("区域最大面积(mm²)") or summary.get("region_max_area") or {})
    region_max_diam = filter_region_dict(summary.get("区域最大等效直径(mm)") or summary.get("region_max_eq_diameter") or {})

    lines = []
    lines.append("【缺陷类型】气孔")
    lines.append("【区域规则】牙孔区域、焊接区域、重点区域为高影响区域；一般区域只作为整体波动参考。")
    lines.append("【缺陷结构化摘要】")
    lines.append(f"总缺陷条目={total_count}")
    lines.append(f"总面积(mm²)={total_area}")
    lines.append(f"最大等效直径(mm)={max_diam}")
    lines.append(f"区域缺陷统计={json.dumps(region_count, ensure_ascii=False)}")
    lines.append(f"区域最大面积(mm²)={json.dumps(region_max_area, ensure_ascii=False)}")
    lines.append(f"区域最大等效直径(mm)={json.dumps(region_max_diam, ensure_ascii=False)}")
    return "\n".join(lines)


def build_params_text(obj: Dict[str, Any]) -> str:
    params = normalize_params(obj.get("params", {}))

    fields = [
        "慢速速度",
        "高速速度",
        "料饼厚度",
        "铸造压力",
        "循环周期",
        "建压时间",
        "射料结束位置",
        "压射压力",
        "高速行程",
        "高速时间",
        "慢速行程",
        "环境温度",
        "环境湿度",
    ]

    lines = ["【压铸参数摘要】"]
    for k in fields:
        if k in params and params[k] not in [None, ""]:
            lines.append(f"{k}={params[k]}")

    return "\n".join(lines)


def sanitize_question(q: str) -> str:
    q = clean_think(q)
    q = replace_non_pore_words(q)
    q = remove_background_unknown_sentences(q)
    return q.strip()


def sanitize_answer(a: str) -> str:
    a = clean_think(a)
    a = replace_non_pore_words(a)
    a = remove_background_unknown_sentences(a)

    for p in ["好的，", "下面是", "以下是", "我来分析", "根据您提供的信息"]:
        if a.startswith(p):
            a = a[len(p):].strip()

    title_replacements = {
        "### 【结论】": "【结论】",
        "###【结论】": "【结论】",
        "### 结论": "【结论】",
        "# 结论": "【结论】",
        "### 【溯源分析】": "【溯源分析】",
        "###【溯源分析】": "【溯源分析】",
        "### 溯源分析": "【溯源分析】",
        "# 溯源分析": "【溯源分析】",
        "### 【建议方向】": "【建议方向】",
        "###【建议方向】": "【建议方向】",
        "### 建议方向": "【建议方向】",
        "# 建议方向": "【建议方向】",
        "【现场数据链】": "【溯源分析】",
        "【检测依据】": "【溯源分析】",
        "【后续建议】": "【建议方向】",
        "【溯源建议】": "【建议方向】",
        "【建议方向']": "【建议方向】",
    }

    for old, new in title_replacements.items():
        a = a.replace(old, new)

    if not a.startswith("【结论】") or "【溯源分析】" not in a or "【建议方向】" not in a:
        a = (
            "【结论】\n"
            "当前样本应统一按气孔进行溯源判断，重点关注牙孔区域、焊接区域和重点区域等高影响区域。\n\n"
            "【溯源分析】\n"
            + a
            + "\n\n【建议方向】\n"
            "建议围绕区域气孔数量、区域最大面积、区域最大等效直径和关键压铸参数建立联动对照，观察高影响区域气孔是否收敛。"
        )

    return a.strip()


def build_full_input(obj: Dict[str, Any]) -> str:
    raw_input = str(obj.get("input", "")).strip()

    if "【缺陷结构化摘要】" in raw_input and "【压铸参数摘要】" in raw_input and "【问题】" in raw_input:
        return sanitize_existing_input(raw_input)

    question = obj.get("question") or obj.get("问题") or raw_input
    if not question:
        question = "请基于当前样本的气孔分布、区域统计和压铸参数进行溯源分析，并给出建议方向。"

    question = sanitize_question(question)

    sample_id = obj.get("sample_id") or obj.get("workpiece_id") or obj.get("sample_no") or ""
    sample_id = normalize_numeric_id(sample_id) if sample_id else ""

    parts = []
    if sample_id:
        parts.append(f"【样本编号】{sample_id}")

    parts.append(build_defect_summary_text(obj))
    parts.append(build_params_text(obj))
    parts.append("【问题】")
    parts.append(question)
    parts.append(
        "【作答要求】严格按照【结论】【溯源分析】【建议方向】三部分作答；"
        "只分析气孔；必须结合缺陷数量、区域分布、区域最大面积、区域最大等效直径和压铸参数；"
        "牙孔区域、焊接区域、重点区域作为高影响区域；"
        "建议只给宏观方向，不给具体数值设定。"
    )

    return "\n".join(parts).strip()


def get_answer(obj: Dict[str, Any]) -> str:
    if "output" in obj:
        return str(obj.get("output", ""))
    if "answer" in obj:
        return str(obj.get("answer", ""))
    if "response" in obj:
        return str(obj.get("response", ""))
    return ""


def make_text_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    instruction = str(obj.get("instruction", "")).strip()
    if not instruction:
        instruction = SYSTEM_INSTRUCTION

    input_text = build_full_input(obj)
    output_text = sanitize_answer(get_answer(obj))

    if not input_text or not output_text:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def make_full_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rec = dict(obj)
    text_rec = make_text_record(obj)
    if text_rec is None:
        return None

    rec["instruction"] = text_rec["instruction"]
    rec["input"] = text_rec["input"]
    rec["output"] = text_rec["output"]
    rec["answer"] = text_rec["output"]
    rec["defect_type"] = "气孔"
    rec["high_impact_regions"] = HIGH_IMPACT_REGIONS

    for k in ["sample_id", "workpiece_id", "sample_no"]:
        if k in rec:
            rec[k] = normalize_numeric_id(rec[k])

    if "params" in rec:
        rec["params"] = normalize_params(rec["params"])

    if "summary" in rec and isinstance(rec["summary"], dict):
        s = dict(rec["summary"])
        for k in [
            "区域缺陷统计",
            "region_count",
            "区域最大面积(mm²)",
            "region_max_area",
            "区域最大等效直径(mm)",
            "region_max_eq_diameter",
        ]:
            if k in s:
                s[k] = filter_region_dict(s[k])
        rec["summary"] = s

    return rec


def find_input_files(input_dir: Path) -> List[Path]:
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".json", ".jsonl"]:
            files.append(p)
    return sorted(files)


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception as e:
                    print(f"[WARN] JSONL 解析失败 {path} 第{line_idx}行: {e}")
        return records

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend([x for x in data if isinstance(x, dict)])
            elif isinstance(data, dict):
                for key in ["data", "train", "samples", "records"]:
                    if isinstance(data.get(key), list):
                        records.extend([x for x in data[key] if isinstance(x, dict)])
                        break
                else:
                    records.append(data)
        except Exception as e:
            print(f"[WARN] JSON 解析失败 {path}: {e}")
        return records

    return records


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def add_id_replay_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    replay = []

    examples = [
        ("样本38053出现气孔，样本46269需要复查，47243-OK为对照件。", "38053,46269,47243"),
        ("工件47243-NG需要复查，样本38053为异常件，样本46269为对照件。", "47243,38053,46269"),
        ("请抽取编号：44100、46269、47243-OK、47243-NG。", "44100,46269,47243,47243"),
    ]

    for src, tgt in examples:
        replay.append({
            "instruction": "你是严格的信息抽取器。OK/NG 后缀只是人工判定备注，抽取编号时只保留数字部分。",
            "input": f"从下面文本中抽取编号，只输出逗号分隔数字结果，不要解释：{src}",
            "output": tgt,
        })

    return rows + replay * 20


def main():
    args = build_arg_parser().parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    train_ratio = args.train_ratio
    seed = args.seed

    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_input_files(input_dir)
    print(f"[INFO] 输入文件夹: {input_dir}")
    print(f"[INFO] 找到 JSON/JSONL 文件数: {len(files)}")

    raw_records = []
    file_counter = {}

    for p in files:
        recs = read_json_or_jsonl(p)
        if recs:
            raw_records.extend(recs)
            file_counter[str(p.relative_to(input_dir))] = len(recs)

    print(f"[INFO] 原始记录数: {len(raw_records)}")

    text_rows = []
    full_rows = []
    skipped = 0

    for obj in raw_records:
        tr = make_text_record(obj)
        fr = make_full_record(obj)

        if tr is None or fr is None:
            skipped += 1
            continue

        text_rows.append(tr)
        full_rows.append(fr)

    text_rows = add_id_replay_samples(text_rows)

    rng = random.Random(seed)
    rng.shuffle(text_rows)

    n_train = int(len(text_rows) * train_ratio)
    train_rows = text_rows[:n_train]
    val_rows = text_rows[n_train:]

    text_all = out_dir / "stage2_text_lora_cleaned_numeric_id_all.jsonl"
    text_train = out_dir / "stage2_text_lora_cleaned_numeric_id_train.jsonl"
    text_val = out_dir / "stage2_text_lora_cleaned_numeric_id_val.jsonl"

    write_jsonl(text_all, text_rows)
    write_jsonl(text_train, train_rows)
    write_jsonl(text_val, val_rows)

    full_all = out_dir / "stage2_full_cleaned_numeric_id_all.jsonl"
    full_train = out_dir / "stage2_full_cleaned_numeric_id_train.jsonl"
    full_val = out_dir / "stage2_full_cleaned_numeric_id_val.jsonl"

    rng.shuffle(full_rows)
    n_full_train = int(len(full_rows) * train_ratio)

    write_jsonl(full_all, full_rows)
    write_jsonl(full_train, full_rows[:n_full_train])
    write_jsonl(full_val, full_rows[n_full_train:])

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "raw_files": file_counter,
        "raw_records": len(raw_records),
        "skipped_records": skipped,
        "text_records_with_id_replay": len(text_rows),
        "text_train": len(train_rows),
        "text_val": len(val_rows),
        "full_records": len(full_rows),
        "outputs": {
            "text_all": str(text_all),
            "text_train": str(text_train),
            "text_val": str(text_val),
            "full_all": str(full_all),
            "full_train": str(full_train),
            "full_val": str(full_val),
        },
        "rules": {
            "id_rule": "OK/NG 后缀只作为人工判定备注，训练主字段统一保留数字 ID",
            "defect_type": "气孔",
            "high_impact_regions": HIGH_IMPACT_REGIONS,
            "ignored_regions": IGNORED_REGION_WORDS,
            "temperature_rule": "温度/温度(℃)/空气温度/环境温度统一为环境温度",
            "humidity_rule": "湿度/湿度(%)/空气湿度/环境湿度统一为环境湿度",
        }
    }

    report_path = out_dir / "clean_report_numeric_id.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[DONE] 清洗完成")
    print(f"text train: {text_train}")
    print(f"text val  : {text_val}")
    print(f"text all  : {text_all}")
    print(f"full train: {full_train}")
    print(f"full val  : {full_val}")
    print(f"report    : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()