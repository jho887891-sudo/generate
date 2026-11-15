# -*- coding: utf-8 -*-
"""
把 Qwen3-4B 4bit 基座模型和第一次书本知识 LoRA 合并成新的 4bit merged 模型。

当前基座是 bnb-4bit：
/home/T7/wkq/end-to-end/models/Qwen3-4B-Instruct-2507-bnb-4bit

所以不能稳定使用 merged_16bit。
这里使用 forced_merged_4bit / merged_4bit_forced 兼容写法。

合并后输出：
/home/T7/wkq/end-to-end/models/Qwen3-4B-stage1-book-merged-4bit
"""

import os
import json
import shutil
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel


# ============================================================
# 路径配置
# ============================================================
BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-0.6B"

STAGE1_LORA_PATH = "/home/T7/wkq/end-to-end/runs/qwen3_0.6b_stage1_expert_clean_best/best_adapter"

TMP_LORA_DIR = "/home/T7/wkq/end-to-end/tmp/0.6B_stage1_lora_for_merge"

MERGED_OUTPUT_DIR = "/home/T7/wkq/end-to-end/models/Qwen3-0.6B-stage1-book-merged-4bit"

MAX_SEQ_LENGTH = 8192
LOAD_IN_4BIT = True
DTYPE = None


def must_exist(path: str, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在: {p}")
    return p


def prepare_tmp_lora_adapter() -> Path:
    src = must_exist(STAGE1_LORA_PATH, "Stage1 LoRA 路径")
    cfg_path = src / "adapter_config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少 adapter_config.json: {cfg_path}")

    tmp = Path(TMP_LORA_DIR)
    if tmp.exists():
        shutil.rmtree(tmp)

    shutil.copytree(src, tmp)

    tmp_cfg_path = tmp / "adapter_config.json"
    cfg = json.loads(tmp_cfg_path.read_text(encoding="utf-8"))

    old_base = cfg.get("base_model_name_or_path")
    cfg["base_model_name_or_path"] = BASE_MODEL_PATH

    tmp_cfg_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Check] Stage1 LoRA adapter copied to tmp")
    print("[Check] original base_model_name_or_path =", old_base)
    print("[Check] fixed    base_model_name_or_path =", cfg["base_model_name_or_path"])
    print("[Check] r =", cfg.get("r"))
    print("[Check] lora_alpha =", cfg.get("lora_alpha"))
    print("[Check] target_modules =", cfg.get("target_modules"))

    return tmp


def list_output_files(out_dir: str):
    p = Path(out_dir)
    print("=" * 80)
    print("[Check] output dir exists:", p.exists())
    print("[Check] output dir:", p)

    if p.exists():
        files = sorted(p.iterdir())
        print("[Check] file count:", len(files))
        for f in files[:50]:
            size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
            print(f"  - {f.name}  {size_mb:.2f} MB")
    print("=" * 80)


def save_merged_model(model, tokenizer):
    """
    兼容不同 Unsloth 版本的 4bit 强制合并参数名。
    """
    methods = [
        "forced_merged_4bit",     # 你的日志提示的名字
        "merged_4bit_forced",     # 部分 Unsloth 版本使用这个名字
        "merged_4bit",            # 兜底
    ]

    last_error = None

    for method in methods:
        try:
            print(f"[Merge] trying save_method={method}")
            model.save_pretrained_merged(
                MERGED_OUTPUT_DIR,
                tokenizer,
                save_method=method,
            )
            print(f"[Merge] success with save_method={method}")
            return
        except Exception as e:
            print(f"[Warn] save_method={method} failed: {repr(e)}")
            last_error = e

    raise RuntimeError(f"所有 4bit 合并方式都失败，最后错误: {repr(last_error)}")


def main():
    print("=" * 80)
    print("Merge Qwen3-0.6B 4bit base + Stage1 book LoRA")
    print("=" * 80)
    print("BASE_MODEL_PATH   =", BASE_MODEL_PATH)
    print("STAGE1_LORA_PATH  =", STAGE1_LORA_PATH)
    print("TMP_LORA_DIR      =", TMP_LORA_DIR)
    print("MERGED_OUTPUT_DIR =", MERGED_OUTPUT_DIR)
    print("=" * 80)

    must_exist(BASE_MODEL_PATH, "0.6B 基座模型路径")
    tmp_lora_dir = prepare_tmp_lora_adapter()

    out = Path(MERGED_OUTPUT_DIR)
    if out.exists():
        print(f"[Warn] 删除旧输出目录: {out}")
        shutil.rmtree(out)

    print("[Load] loading BASE + Stage1 LoRA through Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(tmp_lora_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
    )

    print("[Merge] saving merged 4bit model...")
    save_merged_model(model, tokenizer)

    list_output_files(MERGED_OUTPUT_DIR)

    print("合并完成。")
    print("后续二次微调请使用：")
    print(f'BASE_MODEL_PATH = "{MERGED_OUTPUT_DIR}"')
    print("并且不要再加载 Stage1 LoRA。")


if __name__ == "__main__":
    main()