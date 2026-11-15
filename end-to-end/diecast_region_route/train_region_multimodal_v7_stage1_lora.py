from __future__ import annotations

import os
import sys
import time

# 必须放在 torch / transformers / peft 导入前
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 80, flush=True)
print("[BOOT] start train_region_multimodal_frozen_llm_4b.py", flush=True)
print("[BOOT] python =", sys.executable, flush=True)
print("[BOOT] time   =", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("=" * 80, flush=True)

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Any

print("[IMPORT] importing numpy / torch ...", flush=True)
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

print("[IMPORT] importing multimodal config/dataset/model ...", flush=True)
from multimodal.config_region_mm import RegionAwareMMConfig
from multimodal.dataset_region_mm import RegionMMJsonlDataset, collate_region_mm
from multimodal.model_region_mm import RegionAwareMultiModalModel

print("[IMPORT] all imports finished", flush=True)


# ============================================================
# 当前推荐路线
# ============================================================
# Qwen3-8B-stage1-book-merged-4bit
# + Stage2 纯文本 LoRA
# + 冻结 LLM
# + 只训练 projector / param_encoder / region_prior / prefix_fusion
# ============================================================

BASE_MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-4B-Instruct-2507-bnb-4bit"

STAGE2_ADAPTER_PATH = (
    "/home/T7/wkq/end-to-end/runs/"
    "qwen3_4b_stage1_expert_clean_best/best_adapter"
)

TRAIN_JSONL = (
    "/home/T7/wkq/end-to-end/model_data/"
    "mm_mapping_unique_by_workpiece/data/stage2_mm_mapping_train_unique_by_workpiece.jsonl"
)

VAL_JSONL = (
    "/home/T7/wkq/end-to-end/model_data/"
    "mm_mapping_unique_by_workpiece/data/stage2_mm_mapping_val_unique_by_workpiece.jsonl"
)

OUTPUT_DIR = (
    "/home/T7/wkq/end-to-end/runs/"
    "qwen3_4b_region_mm_frozen_llm_基座模型"
)


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def must_exist(path: str | Path, name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在: {p}")
    return p


def summarize_question_types(records, prefix: str):
    counter = Counter()

    for r in records:
        qt = str(r.get("question_type", "open_diagnosis"))
        counter[qt] += 1

    total = sum(counter.values())

    print("\n" + "=" * 80)
    print(f"[info] {prefix} 问题类型分布")
    print(f"[info] total = {total}")

    for k, v in counter.most_common():
        ratio = 0.0 if total == 0 else v / total
        print(f"  - {k}: {v} ({ratio:.2%})")

    print("=" * 80 + "\n")


def check_answer_format(records, prefix: str):
    ok = 0
    bad = []

    for i, r in enumerate(records):
        ans = str(r.get("answer", r.get("output", "")))

        cond = (
            "【结论】" in ans
            and "【溯源分析】" in ans
            and "【建议方向】" in ans
        )

        if cond:
            ok += 1
        else:
            bad.append((i, r.get("sample_id"), r.get("question", "")[:50]))

    total = len(records)
    ratio = 0 if total == 0 else ok / total
    print(f"[info] {prefix} 三段式标题合格: {ok}/{total} ({ratio:.2%})")

    if bad:
        print("[warn] 发现格式不合格样本，前5条：")
        for item in bad[:5]:
            print("  ", item)


def check_dataset_paths(cfg: RegionAwareMMConfig):
    missing = []

    for p in [cfg.train_jsonl, cfg.val_jsonl]:
        if not Path(p).exists():
            missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            "下面这些数据集文件不存在：\n" + "\n".join(missing)
        )

    print(f"[info] train_jsonl = {cfg.train_jsonl}")
    print(f"[info] val_jsonl   = {cfg.val_jsonl}")


def check_model_paths(cfg: RegionAwareMMConfig):
    missing = []

    if not Path(cfg.llm_base_path).exists():
        missing.append(f"llm_base_path 不存在: {cfg.llm_base_path}")

    if not Path(cfg.llm_stage2_adapter_path).exists():
        missing.append(f"llm_stage2_adapter_path 不存在: {cfg.llm_stage2_adapter_path}")

    if missing:
        raise FileNotFoundError("\n".join(missing))

    print(f"[info] llm_base_path           = {cfg.llm_base_path}")
    print(f"[info] llm_stage2_adapter_path = {cfg.llm_stage2_adapter_path}")


def freeze_llm_parameters(model: RegionAwareMultiModalModel):
    """
    冻结 LLM，包括 Stage2 LoRA adapter。
    只训练多模态新增模块，避免再次污染文本指令跟随能力。
    """
    if not hasattr(model, "llm"):
        print("[warn] model 没有 llm 属性，无法冻结 LLM")
        return

    total = 0

    for p in model.llm.parameters():
        p.requires_grad = False
        total += p.numel()

    print(f"[freeze] frozen llm params: {total:,}", flush=True)

    # 确保多模态模块可训练
    trainable_modules = [
        "projector",
        "param_encoder",
        "region_prior",
        "prefix_fusion",
    ]

    for name in trainable_modules:
        module = getattr(model, name, None)

        if module is None:
            print(f"[warn] module not found: {name}", flush=True)
            continue

        for p in module.parameters():
            p.requires_grad = True

        n = sum(p.numel() for p in module.parameters())
        print(f"[trainable] {name} enabled, params={n:,}", flush=True)


def print_trainable_params(model: torch.nn.Module):
    total = 0
    trainable = 0

    module_counter: Dict[str, int] = {}

    for name, p in model.named_parameters():
        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()
            root = name.split(".")[0]
            module_counter[root] = module_counter.get(root, 0) + p.numel()

    print("=" * 80)
    print(f"[params] total     : {total:,}")
    print(f"[params] trainable : {trainable:,}")
    print(f"[params] ratio     : {0 if total == 0 else trainable / total:.6%}")
    print("[params] trainable by module:")
    for k, v in sorted(module_counter.items(), key=lambda x: -x[1]):
        print(f"  - {k}: {v:,}")
    print("=" * 80)

    if trainable == 0:
        raise RuntimeError("没有可训练参数，请检查多模态模块是否正确设置 requires_grad=True")


def evaluate(model: RegionAwareMultiModalModel, loader: DataLoader):
    model.eval()

    # 冻结 LLM 时，保持 eval
    if hasattr(model, "llm"):
        model.llm.eval()

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            out = model(batch)
            loss = out["loss"]

            if torch.isnan(loss):
                continue

            total_loss += float(loss.item())
            total_steps += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return 0.0 if total_steps == 0 else total_loss / total_steps


def copy_stage2_adapter_to_checkpoint(cfg: RegionAwareMMConfig, save_dir: Path):
    """
    因为 LLM 冻结，Stage2 adapter 不会变化。
    为了推理脚本加载方便，直接复制一份到 checkpoint/llm_adapter。
    """
    src = Path(cfg.llm_stage2_adapter_path)
    dst = save_dir / "llm_adapter"

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)


def save_lightweight_checkpoint(
    model: RegionAwareMultiModalModel,
    cfg: RegionAwareMMConfig,
    save_dir: Path,
    epoch: int,
    val_loss: float,
):
    """
    轻量保存：
    1. 保存多模态新增模块；
    2. 复制冻结的 Stage2 LLM adapter；
    3. 保存 meta.json。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    mm_state = {
        "projector": model.projector.state_dict(),
        "param_encoder": model.param_encoder.state_dict(),
        "region_prior": model.region_prior.state_dict(),
        "prefix_fusion": model.prefix_fusion.state_dict()
        if model.prefix_fusion is not None
        else None,
        "epoch": epoch,
        "val_loss": val_loss,
    }

    torch.save(mm_state, save_dir / "multimodal_heads.pt")

    copy_stage2_adapter_to_checkpoint(cfg, save_dir)

    meta = {
        "epoch": epoch,
        "val_loss": val_loss,

        "training_mode": "frozen_llm_train_mapping_only",

        "llm_base_path": str(cfg.llm_base_path),
        "llm_stage1_adapter_path": str(getattr(cfg, "llm_stage1_adapter_path", "")),
        "llm_stage2_adapter_path": str(cfg.llm_stage2_adapter_path),

        "continue_train_stage1_adapter": False,
        "continue_train_stage2_adapter": False,
        "train_new_lora_adapter": False,
        "freeze_llm_base": True,
        "llm_frozen_after_load": True,

        "yolo_weight_path": str(cfg.yolo_weight_path),
        "vision_image_size": cfg.vision_image_size,
        "yolo_feature_layer": cfg.yolo_feature_layer,
        "vision_feat_dim": cfg.vision_feat_dim,

        "max_views_per_sample": cfg.max_views_per_sample,
        "max_rois_per_view": cfg.max_rois_per_view,
        "max_defect_tokens": cfg.max_defect_tokens,
        "max_prefix_tokens": cfg.max_prefix_tokens,

        "use_external_summary_tokens": bool(cfg.use_external_summary_tokens),
        "use_coordinate_region_summary": bool(cfg.use_coordinate_region_summary),
        "use_prefix_fusion_mapper": bool(cfg.use_prefix_fusion_mapper),
        "fusion_token_count": int(cfg.fusion_token_count),
        "use_visual_summary_text": bool(cfg.use_visual_summary_text),

        "max_seq_length": cfg.max_seq_length,
        "max_user_tokens": cfg.max_user_tokens,
        "max_answer_tokens": cfg.max_answer_tokens,

        "numeric_fields": list(cfg.numeric_fields),
        "region_order": list(cfg.region_order),
        "high_impact_regions": list(getattr(cfg, "high_impact_regions", [])),
        "ignored_regions": list(getattr(cfg, "ignored_regions", [])),

        "train_jsonl": str(cfg.train_jsonl),
        "val_jsonl": str(cfg.val_jsonl),
        "output_dir": str(cfg.output_dir),
    }

    with (save_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============================================================
# 参数解析
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--train-jsonl", type=str, default=TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=str, default=VAL_JSONL)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)

    parser.add_argument("--base-model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--stage2-adapter", type=str, default=STAGE2_ADAPTER_PATH)

    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-acc", type=int, default=None)

    # 默认冻结 LLM，不建议打开
    parser.add_argument(
        "--train-llm",
        action="store_true",
        help="如果指定，则不冻结 LLM adapter，会继续训练 LLM。默认不建议。",
    )

    return parser


# ============================================================
# 主流程
# ============================================================

def main():
    args = build_arg_parser().parse_args()

    cfg = RegionAwareMMConfig()

    # ============================================================
    # 数据和输出路径
    # ============================================================
    cfg.train_jsonl = Path(args.train_jsonl)
    cfg.val_jsonl = Path(args.val_jsonl)
    cfg.test_jsonl = Path(args.val_jsonl)
    cfg.output_dir = Path(args.output_dir)

    # ============================================================
    # 4B + merged Stage1 + Stage2 adapter 路线
    # ============================================================
    cfg.llm_base_path = args.base_model
    cfg.llm_stage2_adapter_path = args.stage2_adapter

    cfg.num_epochs = args.epochs

    if args.lr is not None:
        cfg.learning_rate = args.lr

    if args.max_seq_length is not None:
        cfg.max_seq_length = args.max_seq_length

    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size

    if args.grad_acc is not None:
        cfg.gradient_accumulation_steps = args.grad_acc

    # ============================================================
    # 关键训练模式
    # 默认：冻结 LLM，只训练映射层
    # ============================================================
    cfg.continue_train_stage1_adapter = False

    if args.train_llm:
        cfg.continue_train_stage2_adapter = True
        cfg.freeze_llm_base = True
    else:
        cfg.continue_train_stage2_adapter = True
        cfg.freeze_llm_base = True

    cfg.train_new_lora_adapter = False

    # 4B 稳定配置
    cfg.use_visual_summary_text = False
    cfg.use_external_summary_tokens = False
    cfg.use_coordinate_region_summary = True
    cfg.use_prefix_fusion_mapper = True

    cfg.max_answer_tokens = max(getattr(cfg, "max_answer_tokens", 256), 512)
    cfg.max_user_tokens = max(getattr(cfg, "max_user_tokens", 384), 768)

    cfg.use_question_type_hint = True
    cfg.inject_question_type_to_instruction = True
    cfg.inject_question_type_to_info_text = False

    print("=" * 80, flush=True)
    print("4B Region-aware multimodal training | frozen LLM, train mapping only", flush=True)
    print("=" * 80, flush=True)
    print(f"[info] train_jsonl    = {cfg.train_jsonl}", flush=True)
    print(f"[info] val_jsonl      = {cfg.val_jsonl}", flush=True)
    print(f"[info] output_dir     = {cfg.output_dir}", flush=True)
    print(f"[info] base_model     = {cfg.llm_base_path}", flush=True)
    print(f"[info] stage2_adapter = {cfg.llm_stage2_adapter_path}", flush=True)
    print(f"[info] train_llm      = {args.train_llm}", flush=True)
    print(f"[info] epochs         = {cfg.num_epochs}", flush=True)
    print(f"[info] lr             = {cfg.learning_rate}", flush=True)
    print(f"[info] max_seq_length = {cfg.max_seq_length}", flush=True)
    print("=" * 80, flush=True)

    set_seed(cfg.seed)
    check_dataset_paths(cfg)
    check_model_paths(cfg)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("[Data] loading dataset ...", flush=True)
    train_ds = RegionMMJsonlDataset(cfg.train_jsonl, cfg)
    val_ds = RegionMMJsonlDataset(cfg.val_jsonl, cfg)

    summarize_question_types(train_ds.records, "train")
    summarize_question_types(val_ds.records, "val")
    check_answer_format(train_ds.records, "train")
    check_answer_format(val_ds.records, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_region_mm,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_region_mm,
        pin_memory=False,
    )

    print("[Model] building RegionAwareMultiModalModel ...", flush=True)
    model = RegionAwareMultiModalModel(cfg)

    if not args.train_llm:
        print("[Mode] freeze LLM and train multimodal heads only", flush=True)
        freeze_llm_parameters(model)
    else:
        print("[Mode] train LLM adapter + multimodal heads", flush=True)
        print("[WARN] 当前不建议打开 --train-llm，可能再次污染指令跟随能力。", flush=True)

    print_trainable_params(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val = None
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()

        # 关键：冻结 LLM 时，保持 LLM eval 模式
        if not args.train_llm and hasattr(model, "llm"):
            model.llm.eval()

        running_loss = 0.0
        log_count = 0

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{cfg.num_epochs}",
        )

        for step, batch in enumerate(progress, start=1):
            out = model(batch)
            loss = out["loss"]

            if torch.isnan(loss):
                print("[warn] loss is NaN, skip", flush=True)
                optimizer.zero_grad(set_to_none=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                continue

            loss_for_backward = loss / cfg.gradient_accumulation_steps
            loss_for_backward.backward()

            running_loss += float(loss.item())
            log_count += 1
            global_step += 1

            if step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    cfg.max_grad_norm,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if global_step % cfg.log_every == 0:
                avg_loss = running_loss / max(log_count, 1)
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0.0
                log_count = 0

        if len(train_loader) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                cfg.max_grad_norm,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = evaluate(model, val_loader)
        print(f"[epoch {epoch + 1}] val_loss = {val_loss:.4f}", flush=True)

        epoch_dir = Path(cfg.output_dir) / f"epoch_{epoch + 1}"
        save_lightweight_checkpoint(
            model=model,
            cfg=cfg,
            save_dir=epoch_dir,
            epoch=epoch + 1,
            val_loss=val_loss,
        )
        print(f"[save] {epoch_dir}", flush=True)

        if best_val is None or val_loss < best_val:
            best_val = val_loss

            best_dir = Path(cfg.output_dir) / "best"
            save_lightweight_checkpoint(
                model=model,
                cfg=cfg,
                save_dir=best_dir,
                epoch=epoch + 1,
                val_loss=val_loss,
            )
            print(f"[save best] {best_dir}", flush=True)

    print("\n训练完成。", flush=True)
    print(f"[best] val_loss = {best_val:.4f}", flush=True)
    print(f"[output] {cfg.output_dir}", flush=True)


if __name__ == "__main__":
    main()