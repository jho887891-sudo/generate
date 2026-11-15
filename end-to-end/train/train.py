import os
import re
import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# ============================================================
# 离线加载
# ============================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============================================================
# 路径配置
# ============================================================
BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-0.6B"

# 第一次微调：书本知识语料
TRAIN_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage1_first_sft/stage1_book_al_diecast_train.jsonl"
VAL_DATA_PATH   = "/home/T7/wkq/end-to-end/model_data/stage1_first_sft/stage1_book_al_diecast_val.jsonl"

RUN_NAME = "qwen3_0.6b_stage1_expert_clean_best"
OUTPUT_DIR = f"/home/T7/wkq/end-to-end/runs/{RUN_NAME}"

BEST_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "best_adapter")
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "training_summary.json")

# ============================================================
# 训练参数
# ============================================================
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True

NUM_TRAIN_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.05

# 约每 20 step 验证/保存一次。训练结束后自动加载 eval_loss 最低的 checkpoint。
EVAL_SAVE_STEPS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("Qwen3-0.6B LoRA SFT | Stage1 book knowledge | select best by eval_loss")
print("=" * 80)
print("BASE_MODEL_PATH  =", BASE_MODEL_PATH)
print("TRAIN_DATA_PATH =", TRAIN_DATA_PATH)
print("VAL_DATA_PATH   =", VAL_DATA_PATH)
print("OUTPUT_DIR      =", OUTPUT_DIR)
print("BEST_ADAPTER_DIR=", BEST_ADAPTER_DIR)
print("=" * 80)


def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 不存在: {path}")


def clean_think(text: str) -> str:
    text = str(text)
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def get_output_field(examples):
    if "output" in examples:
        return examples["output"]
    if "answer" in examples:
        return examples["answer"]
    if "response" in examples:
        return examples["response"]
    raise KeyError("数据中找不到 output / answer / response 字段")


def convert_to_conversations(examples):
    n = len(next(iter(examples.values())))

    instructions = examples["instruction"] if "instruction" in examples else [""] * n
    inputs = examples["input"] if "input" in examples else (
        examples["question"] if "question" in examples else [""] * n
    )
    outputs = get_output_field(examples)

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
    final_dataset = formatted.remove_columns(remove_cols).shuffle(seed=3407)

    return final_dataset


def main():
    must_exist(BASE_MODEL_PATH, "基础模型路径")
    must_exist(TRAIN_DATA_PATH, "训练集路径")
    must_exist(VAL_DATA_PATH, "验证集路径")

    print("[Data] loading train / val jsonl")
    raw = load_dataset(
        "json",
        data_files={
            "train": TRAIN_DATA_PATH,
            "validation": VAL_DATA_PATH,
        },
    )

    print("[Model] loading base model")
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

    print("[LoRA] attaching LoRA")
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
        random_state=3407,
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

        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    # 只训练 assistant 回复，避免 user prompt / 输入字段参与 loss
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("=" * 80)
    print("[Train] start training")
    print("=" * 80)

    trainer_stats = trainer.train()

    print("=" * 80)
    print("[Train] finished")
    print("train metrics:")
    print(trainer_stats.metrics)
    print("=" * 80)

    print("[Best] best checkpoint =", trainer.state.best_model_checkpoint)
    print("[Best] best eval_loss  =", trainer.state.best_metric)

    trainer.model.save_pretrained(BEST_ADAPTER_DIR, safe_serialization=True)
    tokenizer.save_pretrained(BEST_ADAPTER_DIR)

    trainer.model.save_pretrained(FINAL_ADAPTER_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    summary = {
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
        "max_seq_length": MAX_SEQ_LENGTH,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("训练完成")
    print("best_adapter:", BEST_ADAPTER_DIR)
    print("summary     :", SUMMARY_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
