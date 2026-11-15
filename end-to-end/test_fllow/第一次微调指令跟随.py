# -*- coding: utf-8 -*-
"""
测试 Qwen3-32B 第一次微调 LoRA 挂载后的指令跟随能力。

用途：
1. 基座模型路径固定为 myQwen3-32B-unsloth-bnb-4bit；
2. LoRA 路径固定为第一次微调后的 adapter：Qwen3-32B-lora-adapter-8-23；
3. 测试用例与“基座模型指令跟随测试.py”保持一致，便于直接比较分数；
4. 输出 jsonl/csv/markdown 报告，用来判断第一次微调是否已经造成格式漂移、前言输出、think 输出或极简指令失效。

运行：
/home/T7/public/miniconda3/envs/finetune1/bin/python /home/T7/wkq/end-to-end/util/第一次微调LoRA指令跟随测试.py
"""

import os

# 单卡测试更稳定；如果想指定其它卡，改这里，例如 "1"、"2"。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import csv
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import torch
from unsloth import FastLanguageModel
from peft import PeftModel


# =========================
# 1. 模型路径配置
# =========================
BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-1.7B-Base"
STAGE1_LORA_PATH = "/home/T7/wkq/end-to-end/runs/qwen3_1.7b_stage1_expert_clean_best/best_adapter"
OUTPUT_ROOT = "/home/T7/wkq/end-to-end/test_outputs/qwen3_1.7B_stage1_lora_instruction_eval_runs"

MAX_SEQ_LENGTH = 8192
LOAD_IN_4BIT = True
DTYPE = None
ENABLE_THINKING = False

GENERATION_KWARGS = {
    "max_new_tokens": 768,
    "do_sample": False,
    "repetition_penalty": 1.05,
}


# =========================
# 2. 测试用例
# =========================
TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "T01_diecast_three_sections_strict",
        "name": "压铸诊断三段式格式跟随",
        "system": "你是一名铝合金压铸缺陷诊断助手。必须严格按照用户要求输出。",
        "user": """请根据下面输入进行诊断。

硬性要求：
1. 输出必须且只能包含三部分：【结论】【溯源分析】【建议方向】。
2. 必须直接从【结论】开始，前面不能有任何前言、解释、寒暄或思考过程。
3. 不允许输出 <think>、</think>、思考过程、小提示。
4. 溯源分析必须使用区域缺陷统计、区域最大面积(mm²)、区域最大等效直径(mm)，并区分“数量主导区域”和“风险关注区域”。
5. 不要编造输入中没有的字段，不要写“未提供料温/模温/真空”等缺失字段说明。
6. 建议方向只能给宏观调整方向，不要给精确数值设定。

【样本编号】46269
【缺陷结构化摘要】
总缺陷条目=249
总面积(mm²)=28.978
最大等效直径(mm)=1.107
区域缺陷统计=一般区域:181；牙孔区域:43；焊接区域:25
区域最大面积(mm²)=一般区域:0.92；牙孔区域:1.18；焊接区域:0.51
区域最大等效直径(mm)=一般区域:1.02；牙孔区域:1.23；焊接区域:0.81
检测类别汇总=Defect:249
IsPore统计=Yes
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
温度(℃)=23.4
湿度(%)=47.1

问题：该工件的主要缺陷风险来自哪里？""",
        "checks": [
            "three_headers_exact",
            "starts_with_conclusion",
            "no_think",
            "no_preface",
            "must_mention_quantity_and_risk_region",
            "must_use_region_metrics",
            "no_missing_field_disclaimer",
        ],
    },
    {
        "id": "T02_no_missing_field_fabrication",
        "name": "不编造缺失字段",
        "system": "你是一名铝合金压铸缺陷诊断助手。必须严格按照用户要求输出。",
        "user": """只根据给出的字段回答，不能补充没有给出的工艺字段。

输出格式必须且只能为：
【结论】
【溯源分析】
【建议方向】

禁止：
- 禁止在【结论】之前输出任何内容；
- 禁止提到输入没有提供的字段，例如料温、模温、真空度、喷涂量、保压时间；
- 禁止写“未提供xxx”之类的话。

【缺陷结构化摘要】
总缺陷条目=58
总面积(mm²)=6.44
最大等效直径(mm)=0.82
区域缺陷统计=一般区域:44；牙孔区域:8；焊接区域:6
区域最大面积(mm²)=一般区域:0.44；牙孔区域:0.67；焊接区域:0.39
区域最大等效直径(mm)=一般区域:0.75；牙孔区域:0.82；焊接区域:0.64
【压铸参数摘要】
慢速速度=0.22
高速速度=3.21
铸造压力=960
建压时间=88
温度(℃)=24.0
湿度(%)=48.5

问题：请判断当前缺陷主导区域和优先调整方向。""",
        "checks": [
            "three_headers_exact",
            "starts_with_conclusion",
            "no_think",
            "no_preface",
            "no_missing_field_disclaimer",
            "no_forbidden_unprovided_fields",
        ],
    },
    {
        "id": "T03_exact_yes_only",
        "name": "极简指令：只输出指定词",
        "system": "你必须严格服从用户的输出格式要求。",
        "user": "请只输出 YES。不要输出任何其它字符、标点、解释、空行或前后缀。",
        "checks": ["exact_yes"],
    },
    {
        "id": "T04_json_only",
        "name": "JSON-only 输出",
        "system": "你必须严格服从用户的输出格式要求。",
        "user": """请判断下面文本的情绪，只输出 JSON，不要 Markdown，不要解释，不要代码块。
JSON 格式必须为：{"label":"positive|neutral|negative","reason":"不超过12个中文字符"}

文本：这个版本比上一个稳定多了，但是输出格式还需要再控制。""",
        "checks": ["valid_json_only", "no_markdown_code_fence"],
    },
    {
        "id": "T05_extract_only",
        "name": "抽取任务：只输出逗号分隔 ID",
        "system": "你必须严格服从用户的输出格式要求。",
        "user": """从下面文本中抽取所有工件编号。只输出编号，用英文逗号分隔，不要解释。
文本：本批次包含工件 38053、工件46269，另一个样本编号为 47243-OK。""",
        "checks": ["extract_ids_only"],
    },
    {
        "id": "T06_ordered_sections_non_diecast",
        "name": "非压铸场景下的标题顺序跟随",
        "system": "你必须严格服从用户的输出格式要求。",
        "user": """请把下面内容改写成学习总结。
硬性格式：只能包含【收获】【问题】【下一步】三个标题，必须按这个顺序输出，标题前不能有任何内容。
内容：今天学习了模型微调流程，但对数据清洗和评价指标还不够熟。""",
        "checks": ["custom_three_headers_exact", "starts_with_custom_first_header", "no_think", "no_preface"],
        "custom_headers": ["【收获】", "【问题】", "【下一步】"],
    },
    {
        "id": "T07_forbid_numeric_recipe",
        "name": "禁止给精确工艺数值",
        "system": "你是一名铝合金压铸缺陷诊断助手。必须严格按照用户要求输出。",
        "user": """请给出宏观建议。禁止给出任何精确数值目标，例如“提高到3.8m/s”“压力调到980bar”“建压时间设为90ms”。
输出必须为三段：【结论】【溯源分析】【建议方向】，且必须直接从【结论】开始。

输入：总缺陷条目=120；区域缺陷统计=一般区域:90；牙孔区域:25；焊接区域:5；高速速度=3.2；铸造压力=940；建压时间=96。
问题：如何调整？""",
        "checks": [
            "three_headers_exact",
            "starts_with_conclusion",
            "no_think",
            "no_preface",
            "no_precise_numeric_recipe",
        ],
    },
    {
        "id": "T08_resist_extra_instruction",
        "name": "抵抗额外废话输出，只给选项",
        "system": "你必须严格服从用户最终的硬性格式要求。",
        "user": """请先认真分析，但最终答案不要展示分析过程。
最终硬性要求：答案只能输出一个大写字母 A、B、C 或 D。
问题：1+1 等于几？
A. 1
B. 2
C. 3
D. 4""",
        "checks": ["exact_single_choice_b"],
    },
]


# =========================
# 3. 模型加载：基座 + 第一次微调 LoRA
# =========================
def assert_base_model_dir(model_path: str) -> None:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"BASE_MODEL_PATH 不存在: {model_path}")

    # 基座目录应包含 config/tokenizer/model shard。
    adapter_markers = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    found = [x for x in adapter_markers if (p / x).exists()]
    if found:
        raise RuntimeError(
            "当前 BASE_MODEL_PATH 看起来是 LoRA adapter 目录，不是基座模型目录。\n"
            f"检测到文件: {found}\n"
            f"当前路径: {model_path}\n"
            "请把 BASE_MODEL_PATH 改成包含 config.json、tokenizer.json、model-xxxxx.safetensors 的基座模型目录。"
        )

    needed_any = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if not any((p / x).exists() for x in needed_any):
        raise RuntimeError(f"当前目录不像基座模型目录，缺少常见配置文件: {model_path}")

    print("[Check] base model directory OK")
    print(f"[Check] BASE_MODEL_PATH = {model_path}")


def assert_lora_adapter_dir(adapter_path: str) -> None:
    p = Path(adapter_path)
    if not p.exists():
        raise FileNotFoundError(f"STAGE1_LORA_PATH 不存在: {adapter_path}")

    if not (p / "adapter_config.json").exists():
        raise RuntimeError(
            "当前 STAGE1_LORA_PATH 不像 LoRA adapter 目录，缺少 adapter_config.json。\n"
            f"当前路径: {adapter_path}"
        )

    if not ((p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()):
        raise RuntimeError(
            "当前 STAGE1_LORA_PATH 缺少 adapter 权重文件：adapter_model.safetensors 或 adapter_model.bin。\n"
            f"当前路径: {adapter_path}"
        )

    print("[Check] stage1 LoRA adapter directory OK")
    print(f"[Check] STAGE1_LORA_PATH = {adapter_path}")

    # 只打印 adapter_config 里的关键信息，方便确认没有拿错 LoRA。
    try:
        cfg = json.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = cfg.get("base_model_name_or_path", "")
        r = cfg.get("r", "")
        alpha = cfg.get("lora_alpha", "")
        target_modules = cfg.get("target_modules", "")
        print(f"[Check] adapter base_model_name_or_path = {base_name}")
        print(f"[Check] adapter r={r}, lora_alpha={alpha}, target_modules={target_modules}")
    except Exception as e:
        print(f"[Warn] adapter_config.json 可读取但解析摘要失败: {e}")


def load_base_with_stage1_lora(base_model_path: str, lora_adapter_path: str):
    assert_base_model_dir(base_model_path)
    assert_lora_adapter_dir(lora_adapter_path)

    print("\n[Load] loading BASE model with Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
    )

    print("[Load] attaching STAGE1 LoRA adapter with PEFT...")
    try:
        model = PeftModel.from_pretrained(
            model,
            lora_adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
    except TypeError:
        # 兼容旧版本 peft：个别版本不支持 local_files_only 参数。
        model = PeftModel.from_pretrained(
            model,
            lora_adapter_path,
            is_trainable=False,
        )

    FastLanguageModel.for_inference(model)
    model.eval()

    print("[Load] done. BASE + STAGE1 LoRA is ready for inference.")
    return tokenizer, model


# =========================
# 4. 推理
# =========================
def build_chat_input(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()
def generate_one(tokenizer, model, system_prompt: str, user_prompt: str) -> str:
    prompt_text = build_chat_input(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **GENERATION_KWARGS,
    )
    new_ids = output_ids[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# =========================
# 5. 规则检查
# =========================
def normalize_text(s: str) -> str:
    return s.replace("\r\n", "\n").strip()


def has_markdown_code_fence(s: str) -> bool:
    return "```" in s


def check_three_headers_exact(output: str, headers: List[str]) -> Tuple[bool, str]:
    s = normalize_text(output)
    counts = {h: s.count(h) for h in headers}
    if any(v != 1 for v in counts.values()):
        return False, f"标题出现次数异常: {counts}"

    positions = [s.find(h) for h in headers]
    if any(p < 0 for p in positions):
        return False, f"缺少标题: {headers}"
    if positions != sorted(positions):
        return False, f"标题顺序错误: {positions}"

    extra_headers = ["【分析】", "【说明】", "【总结】", "【小提示】", "【注意】", "【判断】"]
    extras = [h for h in extra_headers if h in s and h not in headers]
    if extras:
        return False, f"出现额外标题: {extras}"
    return True, "OK"


def starts_with_header(output: str, header: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    if not s.startswith(header):
        return False, f"不是直接从 {header} 开始，开头为: {s[:50]!r}"
    return True, "OK"


def no_preface(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    bad_prefixes = [
        "好的", "可以", "当然", "下面", "以下", "根据", "我将", "我会", "首先", "分析如下", "诊断如下",
        "Here", "Sure", "Based on",
    ]
    for p in bad_prefixes:
        if s.startswith(p):
            return False, f"出现前言/寒暄开头: {p}"
    return True, "OK"


def no_think(output: str) -> Tuple[bool, str]:
    s = normalize_text(output).lower()
    bad = ["<think>", "</think>", "思考过程", "推理过程", "小提示", "internal reasoning"]
    hit = [b for b in bad if b.lower() in s]
    if hit:
        return False, f"出现禁止的思考/提示内容: {hit}"
    return True, "OK"


def no_missing_field_disclaimer(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    patterns = [r"未提供", r"没有提供", r"缺少", r"缺失", r"未给出", r"未包含", r"无法判断.*因为.*未"]
    for pat in patterns:
        if re.search(pat, s):
            return False, f"出现缺失字段说明: {pat}"
    return True, "OK"


def no_forbidden_unprovided_fields(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    forbidden = ["料温", "模温", "真空", "真空度", "喷涂", "保压时间", "模具温度", "浇注温度"]
    hit = [w for w in forbidden if w in s]
    if hit:
        return False, f"提到了输入未提供字段: {hit}"
    return True, "OK"


def must_mention_quantity_and_risk_region(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    required = ["数量主导区域", "风险关注区域"]
    miss = [w for w in required if w not in s]
    if miss:
        return False, f"未明确区分: {miss}"
    return True, "OK"


def must_use_region_metrics(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    groups = {
        "区域统计/计数": ["区域缺陷统计", "181", "43", "25", "计数", "数量"],
        "区域最大面积": ["区域最大面积", "最大面积", "1.18", "0.92", "面积"],
        "区域最大等效直径": ["区域最大等效直径", "最大等效直径", "1.23", "1.02", "等效直径"],
    }
    miss = []
    for name, words in groups.items():
        if not any(w in s for w in words):
            miss.append(name)
    if miss:
        return False, f"未显式使用区域指标: {miss}"
    return True, "OK"


def exact_yes(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    if s != "YES":
        return False, f"不是精确 YES，实际输出: {s!r}"
    return True, "OK"


def exact_single_choice_b(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    if s != "B":
        return False, f"不是精确 B，实际输出: {s!r}"
    return True, "OK"


def valid_json_only(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    if has_markdown_code_fence(s):
        return False, "出现 Markdown 代码块"
    if not (s.startswith("{") and s.endswith("}")):
        return False, f"不是纯 JSON 对象: {s[:40]!r} ... {s[-40:]!r}"
    try:
        obj = json.loads(s)
    except Exception as e:
        return False, f"JSON 解析失败: {e}"
    if set(obj.keys()) != {"label", "reason"}:
        return False, f"JSON 字段不符合要求: {list(obj.keys())}"
    if obj.get("label") not in ["positive", "neutral", "negative"]:
        return False, f"label 不合法: {obj.get('label')!r}"
    if len(str(obj.get("reason", ""))) > 12:
        return False, f"reason 超过 12 个中文字符: {obj.get('reason')!r}"
    return True, "OK"


def no_markdown_code_fence(output: str) -> Tuple[bool, str]:
    if has_markdown_code_fence(output):
        return False, "出现 Markdown 代码块"
    return True, "OK"


def extract_ids_only(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    compact = re.sub(r"\s+", "", s)
    acceptable = {"38053,46269,47243", "38053,46269,47243-OK"}
    if compact not in acceptable:
        return False, f"抽取结果不符合要求，实际输出: {s!r}"
    return True, "OK"


def no_precise_numeric_recipe(output: str) -> Tuple[bool, str]:
    s = normalize_text(output)
    patterns = [
        r"(调到|设为|设置为|提高到|降低到|控制在|稳定在|调整为)\s*[0-9]+(\.[0-9]+)?\s*(m/s|bar|ms|mm|℃|%)?",
        r"[0-9]+(\.[0-9]+)?\s*(m/s|bar|ms|mm|℃|%)\s*(左右|附近|以内|以上|以下)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return False, f"疑似给出精确数值配方: {m.group(0)}"
    return True, "OK"


CHECK_FN = {
    "three_headers_exact": lambda out, case: check_three_headers_exact(out, ["【结论】", "【溯源分析】", "【建议方向】"]),
    "starts_with_conclusion": lambda out, case: starts_with_header(out, "【结论】"),
    "custom_three_headers_exact": lambda out, case: check_three_headers_exact(out, case["custom_headers"]),
    "starts_with_custom_first_header": lambda out, case: starts_with_header(out, case["custom_headers"][0]),
    "no_preface": lambda out, case: no_preface(out),
    "no_think": lambda out, case: no_think(out),
    "no_missing_field_disclaimer": lambda out, case: no_missing_field_disclaimer(out),
    "no_forbidden_unprovided_fields": lambda out, case: no_forbidden_unprovided_fields(out),
    "must_mention_quantity_and_risk_region": lambda out, case: must_mention_quantity_and_risk_region(out),
    "must_use_region_metrics": lambda out, case: must_use_region_metrics(out),
    "exact_yes": lambda out, case: exact_yes(out),
    "exact_single_choice_b": lambda out, case: exact_single_choice_b(out),
    "valid_json_only": lambda out, case: valid_json_only(out),
    "no_markdown_code_fence": lambda out, case: no_markdown_code_fence(out),
    "extract_ids_only": lambda out, case: extract_ids_only(out),
    "no_precise_numeric_recipe": lambda out, case: no_precise_numeric_recipe(out),
}


def evaluate_output(output: str, case: Dict[str, Any]) -> Dict[str, Any]:
    details = []
    passed = 0
    total = len(case["checks"])
    for check_name in case["checks"]:
        ok, msg = CHECK_FN[check_name](output, case)
        details.append({"check": check_name, "pass": ok, "message": msg})
        if ok:
            passed += 1
    return {
        "passed": passed,
        "total": total,
        "score": round(passed / total * 100, 2) if total else 0,
        "details": details,
    }


# =========================
# 6. 保存结果
# =========================
def save_results(results: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "stage1_lora_instruction_following_results.jsonl"
    csv_path = out_dir / "stage1_lora_instruction_following_scores.csv"
    md_path = out_dir / "stage1_lora_instruction_following_report.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_tag", "case_id", "case_name", "score", "passed", "total", "failed_checks", "latency_sec"],
        )
        writer.writeheader()
        for r in results:
            failed = [d["check"] for d in r["eval"]["details"] if not d["pass"]]
            writer.writerow({
                "model_tag": r["model_tag"],
                "case_id": r["case_id"],
                "case_name": r["case_name"],
                "score": r["eval"]["score"],
                "passed": r["eval"]["passed"],
                "total": r["eval"]["total"],
                "failed_checks": ";".join(failed),
                "latency_sec": r["latency_sec"],
            })

    avg = sum(r["eval"]["score"] for r in results) / max(1, len(results))
    full_pass = sum(1 for r in results if r["eval"]["score"] == 100)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-32B 第一次微调 LoRA 指令跟随能力测试报告\n\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"基座模型路径：`{BASE_MODEL_PATH}`\n\n")
        f.write(f"LoRA 路径：`{STAGE1_LORA_PATH}`\n\n")
        f.write("LoRA/PEFT：`已加载第一次微调 adapter`\n\n")
        f.write(f"ENABLE_THINKING：`{ENABLE_THINKING}`\n\n")
        f.write(f"GENERATION_KWARGS：`{GENERATION_KWARGS}`\n\n")
        f.write(f"平均分：**{avg:.2f}**\n\n")
        f.write(f"满分用例：**{full_pass}/{len(results)}**\n\n")
        f.write("| 用例 | 得分 | 通过/总数 | 失败项 | 耗时(s) |\n")
        f.write("|---|---:|---:|---|---:|\n")
        for r in results:
            failed = [d["check"] for d in r["eval"]["details"] if not d["pass"]]
            f.write(
                f"| {r['case_id']} {r['case_name']} | {r['eval']['score']} | "
                f"{r['eval']['passed']}/{r['eval']['total']} | {', '.join(failed) if failed else '无'} | {r['latency_sec']} |\n"
            )
        f.write("\n")

        for r in results:
            f.write(f"## {r['case_id']} {r['case_name']}\n\n")
            f.write(f"得分：{r['eval']['score']}，通过：{r['eval']['passed']}/{r['eval']['total']}\n\n")
            for d in r["eval"]["details"]:
                mark = "✅" if d["pass"] else "❌"
                f.write(f"- {mark} `{d['check']}`：{d['message']}\n")
            f.write("\n模型输出：\n\n")
            f.write("```text\n")
            f.write(r["output"])
            f.write("\n```\n\n")

    print(f"\n[Save] jsonl: {jsonl_path}")
    print(f"[Save] csv  : {csv_path}")
    print(f"[Save] md   : {md_path}")


# =========================
# 7. 主流程
# =========================
def main() -> None:
    run_name = datetime.now().strftime("stage1_lora_run_%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_ROOT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen3-32B STAGE1-LORA instruction-following evaluation")
    print("=" * 80)
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"BASE_MODEL_PATH      = {BASE_MODEL_PATH}")
    print(f"STAGE1_LORA_PATH    = {STAGE1_LORA_PATH}")
    print("LoRA/PEFT            = ENABLED: stage1 adapter")

    tokenizer, model = load_base_with_stage1_lora(BASE_MODEL_PATH, STAGE1_LORA_PATH)

    results = []
    for idx, case in enumerate(TEST_CASES, 1):
        print(f"\n[{idx}/{len(TEST_CASES)}] {case['id']} - {case['name']}")
        start = time.time()
        try:
            output = generate_one(tokenizer, model, case["system"], case["user"])
            err = None
        except Exception:
            output = ""
            err = traceback.format_exc()
            print(err)

        latency = round(time.time() - start, 3)
        ev = evaluate_output(output, case) if err is None else {
            "passed": 0,
            "total": len(case["checks"]),
            "score": 0,
            "details": [{"check": "runtime_error", "pass": False, "message": err}],
        }
        failed = [d["check"] for d in ev["details"] if not d["pass"]]
        print(f"Score: {ev['score']} | Passed: {ev['passed']}/{ev['total']} | Failed: {failed if failed else 'None'} | {latency}s")
        print("Output preview:")
        print(output[:500].replace("\n", "\\n"))

        results.append({
            "model_tag": "stage1_lora",
            "case_id": case["id"],
            "case_name": case["name"],
            "system": case["system"],
            "user": case["user"],
            "output": output,
            "eval": ev,
            "latency_sec": latency,
            "error": err,
        })

    save_results(results, out_dir)

    avg_score = sum(r["eval"]["score"] for r in results) / max(1, len(results))
    strict_pass = sum(1 for r in results if r["eval"]["score"] == 100)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"stage1_lora: avg_score={avg_score:.2f}, full_pass={strict_pass}/{len(results)}")
    print(f"结果目录: {out_dir}")


if __name__ == "__main__":
    main()
