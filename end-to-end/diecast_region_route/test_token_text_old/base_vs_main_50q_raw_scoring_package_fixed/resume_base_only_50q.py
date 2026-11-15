#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_MODEL_PATH = os.environ.get(
    "BASE_LLM_PATH",
    "/home/T7/wkq/end-to-end/models/Qwen3-4B-Instruct-2507-bnb-4bit",
)

OUT_ROOT = Path(os.environ.get(
    "OUT_ROOT",
    "/home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071",
))

CONFIG_JSON = Path(os.environ.get(
    "CONFIG_JSON",
    "/home/T7/wkq/end-to-end/diecast_region_route/test_token_text/base_vs_main_50q_config.json",
))

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))


SYSTEM_PROMPT = """你是一名铝合金压铸气孔缺陷溯源助手。

硬性规则：
1. 最终回答必须直接从【结论】开始。
2. 最终回答必须且只能包含【结论】【溯源分析】【建议方向】三部分。
3. 缺陷类型统一按气孔处理。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。
5. 回答必须依据当前输入中的视觉层文字总结、压铸参数和图谱辅助证据；没有提供的信息不要编造。
6. 图谱证据只辅助机理判断，不能覆盖当前样本；最终不要输出图谱命中列表、规则编号或调试信息。
7. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。
8. 严禁输出任何推理过程、思考过程、自我检查、格式说明、重写提示或过渡文字。
9. 不得输出“我将”“下面”“先按照”“重新输出”“根据要求”“格式如下”“分析如下”等引导语。
10. 不得输出 <think>、</think>、思考、推理、过程、草稿、检查、示例、作答要求 等内容。
11. 【溯源分析】和【建议方向】分别按 1、2、3 编号，每点一到两句话。
12. 【建议方向】第3点写完后立即停止，不要继续输出任何内容。
"""


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_text(path: Path, default=""):
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def compact_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def visual_summary_to_text(vs: dict) -> str:
    region_count = vs.get("region_count", {}) or {}
    region_area = vs.get("region_max_area_mm2", {}) or {}
    region_dia = vs.get("region_max_eq_diameter_mm", {}) or {}

    return "\n".join([
        "缺陷类型统一按气孔处理。",
        "有效工艺区域包括：一般区域、牙孔区域、焊接区域、重点区域；其中牙孔区域、焊接区域、重点区域为高影响区域。",
        f"总气孔条目={vs.get('total_defect_count', '')}",
        f"总面积(mm²)={vs.get('total_area_mm2', '')}",
        f"最大等效直径(mm)={vs.get('max_eq_diameter_mm', '')}",
        "区域缺陷统计=" + json.dumps(region_count, ensure_ascii=False),
        "区域最大面积(mm²)=" + json.dumps(region_area, ensure_ascii=False),
        "区域最大等效直径(mm)=" + json.dumps(region_dia, ensure_ascii=False),
    ])


def params_to_text(params: dict) -> str:
    if not isinstance(params, dict):
        return ""
    keys = [
        "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
        "建压时间", "射料结束位置", "压射压力",
        "高速行程", "高速时间", "慢速行程",
        "环境温度", "环境湿度",
    ]
    lines = []
    for k in keys:
        if k in params and params[k] not in [None, ""]:
            lines.append(f"{k}={params[k]}")
    return "\n".join(lines)


def build_user_prompt(case_meta: dict, cfg_json: dict, graph_context: str) -> str:
    question = case_meta.get("question", "")
    question_type_hint = case_meta.get("question_type_hint", "")
    visual_summary = case_meta.get("visual_summary", {}) or cfg_json.get("visual_summary", {}) or {}
    params = case_meta.get("params", {}) or cfg_json.get("params", {}) or {}

    visual_text = visual_summary_to_text(visual_summary)
    params_text = params_to_text(params)

    if not graph_context.strip():
        graph_context = "未检索到有效图谱证据；最终判断以当前视觉统计和压铸参数为准。"

    return f"""【视觉层文字总结】
{visual_text}

【压铸参数摘要】
{params_text}

【图谱辅助证据】
{graph_context}

【当前问题类型提示】
{question_type_hint}

【问题】
{question}

现在开始作答。只输出最终答案，不输出任何推理过程、格式说明、自我检查或重写提示。
"""


def extract_main_answer(qdir: Path) -> str:
    mdir = qdir / "main_token_text_raw"
    candidates = [
        mdir / "raw_answer.txt",
        mdir / "final_answer.txt",
        mdir / "answer.txt",
    ]
    for p in candidates:
        txt = read_text(p, "")
        if txt:
            return txt
    return ""


def build_all_results(cases):
    return {
        "experiment": "main_token_text_raw vs base_text_only_raw",
        "raw_scoring": True,
        "base_model_path": BASE_MODEL_PATH,
        "output_root": str(OUT_ROOT),
        "cases": cases,
    }


def main():
    print("=" * 100)
    print("Resume base-only 50Q text_only raw generation")
    print(f"BASE_MODEL_PATH = {BASE_MODEL_PATH}")
    print(f"OUT_ROOT        = {OUT_ROOT}")
    print(f"CONFIG_JSON     = {CONFIG_JSON}")
    print("=" * 100)

    if not OUT_ROOT.exists():
        raise FileNotFoundError(f"OUT_ROOT 不存在: {OUT_ROOT}")

    cfg_json = load_json(CONFIG_JSON, {}) or {}

    qdirs = sorted([p for p in OUT_ROOT.iterdir() if p.is_dir() and re.match(r"q\d+", p.name)])
    if not qdirs:
        raise RuntimeError(f"没有找到 q001/q002 目录: {OUT_ROOT}")

    print(f"[scan] found qdirs = {len(qdirs)}")
    print("[load] loading base model...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    cases = []

    for qdir in qdirs:
        qname = qdir.name
        qnum = int(re.sub(r"\D+", "", qname))

        main_meta_path = qdir / "case_result_partial_main.json"
        case_meta = load_json(main_meta_path, {}) or {}

        if not case_meta:
            print(f"[skip] {qname}: missing case_result_partial_main.json")
            continue

        base_dir = qdir / "base_text_only_raw"
        base_raw_path = base_dir / "raw_answer.txt"

        question = case_meta.get("question", "")
        graph_context = read_text(qdir / "graph_context.txt", "")

        if base_raw_path.exists() and read_text(base_raw_path):
            print(f"[skip] {qname}: base already exists")
            raw_answer = read_text(base_raw_path)
            prompt = read_text(base_dir / "prompt.txt")
        else:
            print(f"[base] {qname} start generate")

            user_prompt = build_user_prompt(case_meta, cfg_json, graph_context)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.08,
                    no_repeat_ngram_size=8,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                )

            gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            raw_answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            write_text(base_dir / "prompt.txt", prompt)
            write_text(base_dir / "raw_answer.txt", raw_answer)
            write_text(base_dir / "final_answer.txt", raw_answer)

            save_json(qdir / "case_result_partial_base.json", {
                "question_index": qnum,
                "workpiece_id": case_meta.get("workpiece_id", ""),
                "base_model_path": BASE_MODEL_PATH,
                "question": question,
                "question_type": case_meta.get("question_type", ""),
                "question_type_hint": case_meta.get("question_type_hint", ""),
                "focus_regions": case_meta.get("focus_regions", []),
                "mechanism_candidates": case_meta.get("mechanism_candidates", []),
                "visual_summary": case_meta.get("visual_summary", {}),
                "params": case_meta.get("params", cfg_json.get("params", {})),
                "raw_answer": raw_answer,
                "final_answer": raw_answer,
                "final_equals_raw": True,
                "mode": "base_text_only_raw",
            })

            print(f"[base] {qname} done")

        main_raw = extract_main_answer(qdir)

        cases.append({
            "question_index": qnum,
            "workpiece_id": case_meta.get("workpiece_id", ""),
            "question": question,
            "question_type": case_meta.get("question_type", ""),
            "question_type_hint": case_meta.get("question_type_hint", ""),
            "focus_regions": case_meta.get("focus_regions", []),
            "mechanism_candidates": case_meta.get("mechanism_candidates", []),
            "visual_summary": case_meta.get("visual_summary", {}),
            "params": case_meta.get("params", cfg_json.get("params", {})),
            "modes": {
                "main_token_text_raw": {
                    "mode_desc": "主方法：4B + token_text + 多模态映射层 + GraphRAG，raw评分",
                    "raw_answer": main_raw,
                    "final_answer": main_raw,
                    "final_equals_raw": True,
                },
                "base_text_only_raw": {
                    "mode_desc": "未微调4B基座：text_only + 同提示词约束 + GraphRAG，raw评分",
                    "raw_answer": raw_answer,
                    "final_answer": raw_answer,
                    "final_equals_raw": True,
                },
            }
        })

    cases.sort(key=lambda x: int(x.get("question_index", 0)))
    all_results = build_all_results(cases)

    save_json(OUT_ROOT / "all_modes_results.json", all_results)
    save_json(OUT_ROOT / "all_results_base_vs_main_raw.json", all_results)

    print("=" * 100)
    print("[done] base-only resume finished")
    print(f"[done] all_modes_results = {OUT_ROOT / 'all_modes_results.json'}")
    print(f"[done] cases = {len(cases)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
