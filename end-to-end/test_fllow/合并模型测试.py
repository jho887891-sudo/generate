# -*- coding: utf-8 -*-
"""
测试合并后的 Stage1 模型是否能正常回答问题。

用途：
1. 只加载合并后的模型，不加载任何 LoRA；
2. 验证模型是否能正常生成；
3. 简单测试书本知识、格式跟随、压铸问答能力；
4. 输出结果保存到 txt，方便对比。

当前默认测试：
/home/T7/wkq/end-to-end/models/Qwen3-4B-stage1-book-merged-4bit
"""

import os
import json
from pathlib import Path
from datetime import datetime

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from unsloth import FastLanguageModel


# ============================================================
# 路径配置
# ============================================================
MERGED_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-0.6B-Base"

OUTPUT_DIR = "/home/T7/wkq/end-to-end/test_outputs/merged_stage1_model_test"

MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
DTYPE = None

ENABLE_THINKING = False

GENERATION_CONFIG = {
    "max_new_tokens": 768,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "repetition_penalty": 1.05,
}


# ============================================================
# 工具函数
# ============================================================
def must_exist(path: str, name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在: {p}")
    return p


def clean_output(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    return text.strip()


def build_prompt(tokenizer, user_text: str, system_text: str = "") -> str:
    messages = []

    if system_text:
        messages.append({
            "role": "system",
            "content": system_text,
        })

    messages.append({
        "role": "user",
        "content": user_text,
    })

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return text


@torch.inference_mode()
def generate_answer(model, tokenizer, user_text: str, system_text: str = "") -> str:
    prompt = build_prompt(tokenizer, user_text, system_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    gen_cfg = dict(GENERATION_CONFIG)

    if tokenizer.eos_token_id is not None:
        gen_cfg["eos_token_id"] = tokenizer.eos_token_id

    if tokenizer.pad_token_id is not None:
        gen_cfg["pad_token_id"] = tokenizer.pad_token_id
    else:
        gen_cfg["pad_token_id"] = tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        **gen_cfg,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return clean_output(answer)


def main():
    print("=" * 80)
    print("Test merged Stage1 model")
    print("=" * 80)
    print("MERGED_MODEL_PATH =", MERGED_MODEL_PATH)
    print("LOAD_IN_4BIT      =", LOAD_IN_4BIT)
    print("MAX_SEQ_LENGTH    =", MAX_SEQ_LENGTH)
    print("=" * 80)

    must_exist(MERGED_MODEL_PATH, "合并后的模型路径")

    model_dir = Path(MERGED_MODEL_PATH)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"缺少 config.json，可能不是有效模型目录: {model_dir}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Load] loading merged model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MERGED_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
    )

    FastLanguageModel.for_inference(model)
    model.eval()

    print("[Load] done.")
    print("=" * 80)

    system_text = (
        "你是一名铝合金压铸领域专家，回答要准确、简洁、工程化。"
        "如果用户要求固定格式，必须严格遵守。"
    )

    test_cases = [
        {
            "id": "T01_basic_yes",
            "question": "只输出 YES，不要输出其他内容。",
            "system": "",
        },
        {
            "id": "T02_diecast_book_knowledge",
            "question": "在压铸模具中，分流锥的主要作用是什么？",
            "system": system_text,
        },
        {
            "id": "T03_diecast_defect_knowledge",
            "question": "铝合金压铸件产生气孔的常见原因有哪些？请用中文简要说明。",
            "system": system_text,
        },
        {
            "id": "T04_three_sections",
            "question": (
                "请严格按照【结论】【溯源分析】【建议方向】三部分回答："
                "铝合金压铸件牙孔区域出现气孔，应该从哪些方向排查？"
            ),
            "system": system_text,
        },
        {
            "id": "T05_no_lora_stack_check",
            "question": (
                "请说明：压铸件一般区域气孔数量较多，但牙孔区域属于后续加工区域，"
                "为什么不能只按数量判断风险？"
            ),
            "system": system_text,
        },
    ]

    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {case['id']}")
        print("-" * 80)
        print("Q:", case["question"])

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            user_text=case["question"],
            system_text=case.get("system", ""),
        )

        print("A:")
        print(answer)
        print("-" * 80)

        results.append({
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
        })

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = output_dir / f"merged_stage1_test_{tag}.txt"
    json_path = output_dir / f"merged_stage1_test_{tag}.json"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"MERGED_MODEL_PATH = {MERGED_MODEL_PATH}\n")
        f.write(f"TIME = {tag}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"[{r['id']}]\n")
            f.write("Q:\n")
            f.write(r["question"] + "\n\n")
            f.write("A:\n")
            f.write(r["answer"] + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "merged_model_path": MERGED_MODEL_PATH,
            "time": tag,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("测试完成")
    print("txt :", txt_path)
    print("json:", json_path)
    print("=" * 80)


if __name__ == "__main__":
    main()