# -*- coding: utf-8 -*-
"""
二次微调纯文本 LoRA 训练脚本

当前路线：
Qwen3-4B-stage1-book-merged-4bit
→ Stage2 参数 / 区域气孔溯源 LoRA

重点：
1. BASE_MODEL_PATH 使用已经合并 Stage1 书本知识 LoRA 的模型；
2. 不再加载 Stage1 LoRA；
3. 手动读取 JSONL，避免 params / summary 等复杂字段结构不一致导致 datasets 报错；
4. 纯文本训练时 input 不是只有问题，而是完整包含：
   - 缺陷数量
   - 区域缺陷统计
   - 区域最大面积
   - 区域最大等效直径
   - 压铸参数
   - 问题
5. 温度 / 空气温度 / 温度(℃) 统一理解为环境温度；
   湿度 / 空气湿度 / 湿度(%) 统一理解为环境湿度；
6. 缺陷类型统一按气孔处理；
7. 背景 / 未知区域忽略；
8. 牙孔区域、焊接区域、重点区域作为高影响区域；
9. 只对 assistant 回复计算 loss。
"""

import os
import re
import json
from pathlib import Path

# 必须放在 transformers / unsloth 导入前
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only


# ============================================================
# 路径配置
# ============================================================

# 合并后的 Stage1 书本知识基座模型
BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-4B-Instruct-2507-bnb-4bit"

# 第二次微调语料
TRAIN_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage2_text_lora_pure_text_repaired_v2_no_question_echo/data/stage2_text_lora_pure_train_repaired_v2_no_question_echo.jsonl"
VAL_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage2_text_lora_pure_text_repaired_v2_no_question_echo/data/stage2_text_lora_pure_val_repaired_v2_no_question_echo.jsonl"

# 如果你的语料实际放在 model_data/stage2_second_sft，就改成：
# TRAIN_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage2_second_sft/stage2_high_impact_gas_pore_train.jsonl"
# VAL_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage2_second_sft/stage2_high_impact_gas_pore_val.jsonl"

RUN_NAME = "qwen3_4b_只进行第二次微调"
OUTPUT_DIR = f"/home/T7/wkq/end-to-end/runs/{RUN_NAME}"

BEST_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "best_adapter")
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "training_summary.json")


# ============================================================
# 训练参数
# ============================================================

MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

NUM_TRAIN_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.05
EVAL_SAVE_STEPS = 50

SEED = 3407


# ============================================================
# 基础工具函数
# ============================================================

def must_exist(path: str, name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在: {p}")
    return p


def clean_think(text: str) -> str:
    text = str(text)
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def save_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_from_many(obj, keys, default=""):
    if not isinstance(obj, dict):
        return default

    for k in keys:
        if k in obj and obj[k] not in [None, ""]:
            return obj[k]

    return default


# ============================================================
# 参数与区域清洗
# ============================================================

def normalize_params(params):
    """
    统一压铸车间空气温湿度字段。
    你的语料里：
    - 温度 / 温度(℃) / 空气温度 / 环境温度 都按环境温度处理；
    - 湿度 / 湿度(%) / 空气湿度 / 环境湿度 都按环境湿度处理。
    """
    if not isinstance(params, dict):
        return {}

    out = dict(params)

    # 环境温度
    for k in ["环境温度", "空气温度", "温度", "温度(℃)"]:
        if k in out and out.get(k) not in [None, ""]:
            out["环境温度"] = out.get(k)
            break

    # 环境湿度
    for k in ["环境湿度", "空气湿度", "湿度", "湿度(%)"]:
        if k in out and out.get(k) not in [None, ""]:
            out["环境湿度"] = out.get(k)
            break

    if "探伤图编号" in out and "样本编号" not in out:
        out["样本编号"] = out["探伤图编号"]

    return out


def is_ignored_region(region_name: str) -> bool:
    s = str(region_name).strip()
    return (
        "背景" in s
        or "未知" in s
        or s in {"0", "unknown", "background", "未知(0)", "未知区域(0)"}
    )


def normalize_region_name(name: str) -> str:
    s = str(name).strip()

    mapping = {
        "general": "一般区域",
        "region1": "一般区域",
        "Region 1": "一般区域",
        "牙孔": "牙孔区域",
        "tooth": "牙孔区域",
        "thread": "牙孔区域",
        "焊接": "焊接区域",
        "weld": "焊接区域",
        "重点": "重点区域",
        "key": "重点区域",
    }

    return mapping.get(s, s)


def filter_region_dict(d):
    """
    背景 / 未知区域直接忽略。
    """
    if not isinstance(d, dict):
        return {}

    valid = {}

    for k, v in d.items():
        region = normalize_region_name(k)

        if is_ignored_region(region):
            continue

        valid[region] = v

    return valid


def sanitize_question(q):
    q = clean_think(str(q))

    # 当前任务只按气孔处理
    for w in ["缩孔", "缩松", "冷隔", "夹杂", "裂纹", "砂眼", "夹渣", "未致密", "机械损伤"]:
        q = q.replace(w, "气孔")

    return q.strip()


def sanitize_answer(answer):
    answer = clean_think(str(answer))

    # 当前任务只按气孔处理
    for w in ["缩孔", "缩松", "冷隔", "夹杂", "裂纹", "砂眼", "夹渣", "未致密", "机械损伤"]:
        answer = answer.replace(w, "气孔")

    # 不希望模型学习输出背景/未知区域
    sentences = re.split(r"(?<=[。；;])", answer)
    kept = []

    for sent in sentences:
        if any(x in sent for x in ["背景", "未知区域", "未知(0)", "未知区域(0)", "背景/未知"]):
            continue
        kept.append(sent)

    answer = "".join(kept).strip()

    return answer


# ============================================================
# 构造完整文本输入：不是只输入 question
# ============================================================

def build_defect_summary_text(obj):
    summary = obj.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    total_count = get_from_many(
        summary,
        ["总缺陷条目", "total_defect_count", "total_count"],
        get_from_many(obj, ["总缺陷条目", "total_defect_count", "total_count"], "")
    )

    total_area = get_from_many(
        summary,
        ["总面积(mm²)", "total_area_mm2", "total_area"],
        get_from_many(obj, ["总面积(mm²)", "total_area_mm2", "total_area"], "")
    )

    max_diam = get_from_many(
        summary,
        ["最大等效直径(mm)", "max_eq_diameter_mm", "max_eq_diameter"],
        get_from_many(obj, ["最大等效直径(mm)", "max_eq_diameter_mm", "max_eq_diameter"], "")
    )

    region_count = (
        summary.get("区域缺陷统计")
        or summary.get("region_count")
        or obj.get("区域缺陷统计")
        or obj.get("region_count")
        or {}
    )
    region_count = filter_region_dict(region_count)

    region_max_area = (
        summary.get("区域最大面积(mm²)")
        or summary.get("region_max_area")
        or obj.get("区域最大面积(mm²)")
        or obj.get("region_max_area")
        or {}
    )
    region_max_area = filter_region_dict(region_max_area)

    region_max_diam = (
        summary.get("区域最大等效直径(mm)")
        or summary.get("region_max_eq_diameter")
        or obj.get("区域最大等效直径(mm)")
        or obj.get("region_max_eq_diameter")
        or {}
    )
    region_max_diam = filter_region_dict(region_max_diam)

    lines = []
    lines.append("【缺陷类型】气孔")
    lines.append("【区域规则】牙孔区域、焊接区域、重点区域为高影响区域；背景和未知区域忽略。")
    lines.append("【缺陷结构化摘要】")
    lines.append(f"总缺陷条目={total_count}")
    lines.append(f"总面积(mm²)={total_area}")
    lines.append(f"最大等效直径(mm)={max_diam}")
    lines.append(f"区域缺陷统计={json.dumps(region_count, ensure_ascii=False)}")
    lines.append(f"区域最大面积(mm²)={json.dumps(region_max_area, ensure_ascii=False)}")
    lines.append(f"区域最大等效直径(mm)={json.dumps(region_max_diam, ensure_ascii=False)}")

    return "\n".join(lines)


def build_params_text(obj):
    params = obj.get("params", {})
    params = normalize_params(params)

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


def build_full_input_text(obj):
    """
    纯文本 LoRA 训练时，input 不能只有 question。
    如果 obj["input"] 已经包含完整摘要，就直接用；
    否则用 summary / params / question 重新拼完整输入。
    """
    raw_input = str(obj.get("input", "")).strip()

    # 如果原始 input 已经是完整三段输入，就保留并统一温湿度命名
    if (
        "【缺陷结构化摘要】" in raw_input
        and "【压铸参数摘要】" in raw_input
        and "【问题】" in raw_input
    ):
        text = raw_input
        text = text.replace("温度(℃)", "环境温度")
        text = text.replace("湿度(%)", "环境湿度")
        text = text.replace("空气温度", "环境温度")
        text = text.replace("空气湿度", "环境湿度")

        # 只按气孔处理
        text = sanitize_question(text)
        return text.strip()

    question = (
        obj.get("question")
        or obj.get("问题")
        or raw_input
        or "请基于当前样本的气孔分布和压铸参数进行溯源分析，并给出建议方向。"
    )
    question = sanitize_question(question)

    sample_id = (
        obj.get("sample_id")
        or obj.get("workpiece_id")
        or obj.get("sample_no")
        or ""
    )

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
        "牙孔区域、焊接区域、重点区域作为高影响区域；背景和未知区域忽略；"
        "建议只给宏观方向，不给具体数值设定。"
    )

    return "\n".join(parts).strip()


def load_text_sft_jsonl(path: str):
    """
    手动读取 JSONL，避免 datasets 因 params 字段结构不一致报错。
    但不会丢弃缺陷摘要和参数，而是把它们整理进 input 文本。
    """
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] 第 {line_idx} 行 JSON 解析失败，跳过: {e}")
                continue

            instruction = str(obj.get("instruction", "")).strip()

            if not instruction:
                instruction = (
                    "你是一名铝合金压铸气孔缺陷溯源助手。"
                    "必须严格按照【结论】【溯源分析】【建议方向】三部分作答。"
                    "缺陷类型统一按气孔处理。"
                    "牙孔区域、焊接区域、重点区域属于高影响区域；背景和未知区域忽略。"
                )

            user_input = build_full_input_text(obj)

            if "output" in obj:
                output = str(obj.get("output", "")).strip()
            elif "answer" in obj:
                output = str(obj.get("answer", "")).strip()
            elif "response" in obj:
                output = str(obj.get("response", "")).strip()
            else:
                output = ""

            output = sanitize_answer(output)

            if not user_input or not output:
                print(f"[WARN] 第 {line_idx} 行 input/output 为空，跳过")
                continue

            records.append({
                "instruction": instruction,
                "input": user_input,
                "output": output,
            })

    if not records:
        raise RuntimeError(f"没有读到有效训练样本: {path}")

    print(f"[Data] loaded {len(records)} text SFT samples from {path}")
    print("=" * 80)
    print("[Data] sample input preview:")
    print(records[0]["input"][:1500])
    print("=" * 80)
    print("[Data] sample output preview:")
    print(records[0]["output"][:1000])
    print("=" * 80)

    return Dataset.from_list(records)


# ============================================================
# Chat template
# ============================================================

def convert_to_conversations(examples):
    n = len(next(iter(examples.values())))

    instructions = examples["instruction"] if "instruction" in examples else [""] * n
    inputs = examples["input"]
    outputs = examples["output"]

    conversations = []

    for instruction, user_input, answer in zip(instructions, inputs, outputs):
        instruction = str(instruction).strip()
        user_input = str(user_input).strip()
        answer = clean_think(answer)

        messages = []

        if instruction:
            messages.append({
                "role": "system",
                "content": instruction,
            })

        messages.append({
            "role": "user",
            "content": user_input,
        })

        messages.append({
            "role": "assistant",
            "content": answer,
        })

        conversations.append(messages)

    return {"conversations": conversations}


def apply_template(examples, tokenizer):
    texts = []

    for conversation in examples["conversations"]:
        try:
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

        text = clean_think(text)
        texts.append(text)

    return {"text": texts}


def prepare_text_dataset(raw_dataset, tokenizer):
    temp = raw_dataset.map(convert_to_conversations, batched=True)

    formatted = temp.map(
        lambda x: apply_template(x, tokenizer),
        batched=True,
    )

    remove_cols = [c for c in formatted.column_names if c != "text"]
    final_dataset = formatted.remove_columns(remove_cols).shuffle(seed=SEED)

    return final_dataset


def save_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 80)
    print("Stage2 pure text LoRA training from merged Stage1 base")
    print("=" * 80)
    print("BASE_MODEL_PATH =", BASE_MODEL_PATH)
    print("TRAIN_DATA_PATH =", TRAIN_DATA_PATH)
    print("VAL_DATA_PATH   =", VAL_DATA_PATH)
    print("OUTPUT_DIR      =", OUTPUT_DIR)
    print("=" * 80)

    must_exist(BASE_MODEL_PATH, "合并后的 Stage1 基座模型")
    must_exist(TRAIN_DATA_PATH, "第二次训练集")
    must_exist(VAL_DATA_PATH, "第二次验证集")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[Data] loading train / val jsonl manually")
    raw = {
        "train": load_text_sft_jsonl(TRAIN_DATA_PATH),
        "validation": load_text_sft_jsonl(VAL_DATA_PATH),
    }

    print("[Model] loading merged Stage1 base")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
    )

    print("[Data] formatting dataset")
    train_dataset = prepare_text_dataset(raw["train"], tokenizer)
    eval_dataset = prepare_text_dataset(raw["validation"], tokenizer)

    print("train size =", len(train_dataset))
    print("val size   =", len(eval_dataset))
    print("=" * 80)
    print("sample preview:")
    print(train_dataset[0]["text"][:1500])
    print("=" * 80)

    print("[LoRA] attaching new Stage2 LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,

        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=-1,

        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        logging_steps=5,

        eval_strategy="steps",
        eval_steps=EVAL_SAVE_STEPS,

        save_strategy="steps",
        save_steps=EVAL_SAVE_STEPS,
        save_total_limit=5,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=True,
        fp16=False,

        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    # 关键：只训练 assistant 回复，避免 system/user/input 参与 loss
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("=" * 80)
    print("[Train] start Stage2 training")
    print("=" * 80)

    trainer_stats = trainer.train()

    print("=" * 80)
    print("[Train] finished")
    print("train metrics:")
    print(trainer_stats.metrics)
    print("=" * 80)

    print("[Best] best checkpoint =", trainer.state.best_model_checkpoint)
    print("[Best] best eval_loss  =", trainer.state.best_metric)

    # load_best_model_at_end=True 后，trainer.model 已经是 eval_loss 最低的 checkpoint
    print("[Save] saving best Stage2 adapter")
    trainer.model.save_pretrained(BEST_ADAPTER_DIR, safe_serialization=True)
    tokenizer.save_pretrained(BEST_ADAPTER_DIR)

    print("[Save] saving final/current Stage2 adapter")
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    summary = {
        "stage": "stage2_text_lora_from_merged_stage1",
        "base_model_path": BASE_MODEL_PATH,
        "train_data_path": TRAIN_DATA_PATH,
        "val_data_path": VAL_DATA_PATH,
        "output_dir": OUTPUT_DIR,
        "best_adapter_dir": BEST_ADAPTER_DIR,
        "final_adapter_dir": FINAL_ADAPTER_DIR,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "train_metrics": trainer_stats.metrics,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "eval_save_steps": EVAL_SAVE_STEPS,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "note": "Stage1 LoRA 已经合并进 BASE_MODEL_PATH，本脚本不再加载 Stage1 LoRA。",
    }

    save_json(SUMMARY_PATH, summary)

    print("=" * 80)
    print("二次纯文本 LoRA 训练完成")
    print("best_adapter :", BEST_ADAPTER_DIR)
    print("final_adapter:", FINAL_ADAPTER_DIR)
    print("summary      :", SUMMARY_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()