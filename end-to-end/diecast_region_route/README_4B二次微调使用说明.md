# 4B 基座 + 第一次书本知识 LoRA + 二次多模态微调代码说明

## 这版改了哪些地方

1. `multimodal/config_region_mm.py`
   - 基座模型从 32B 改为 4B 默认路径：
     `/home/T7/zyx/finetune/myQwen3-4B-unsloth-bnb-4bit`
   - 第一次书本知识 LoRA 默认路径改为：
     `/home/T7/zyx/finetune/util/Qwen3-4B-stage1-lora-adapter`
   - 二次多模态输出目录改为：
     `/home/T7/zyx/finetune/diecast_region_route/region_mm_outputs_v7_4b_stage1_lora`
   - 支持环境变量覆盖路径：
     - `DIECAST_LLM_BASE_PATH`
     - `DIECAST_STAGE1_ADAPTER_PATH`
     - `DIECAST_STAGE2_ADAPTER_PATH`

2. `train_region_multimodal_v7_stage1_lora.py`
   - 默认继续训练第一次书本知识 LoRA：`continue_train_stage1_adapter=True`
   - 不从旧的 stage2 adapter 开始：`continue_train_stage2_adapter=False`
   - 新增命令行参数：
     - `--base-model`
     - `--stage1-adapter`
     - `--max-seq-length`
   - 输出目录默认改为 4B 版本。

3. `multimodal/model_region_mm.py`
   - 保留轻量版 kbit prepare，避免不必要的 fp32 cast。
   - 加入 `tokenizer.truncation_side='left'` 和 `padding_side='left'`，长 prompt 时优先保留后面的实际问题和输出约束。

4. `e2e_test_graph_v7_format.json`
   - `mm_weight_path` 改为 4B 二次微调输出目录。

## 重要提醒

第一次书本知识 LoRA 必须和 4B 基座对应，不能把 32B 的 LoRA 直接挂到 4B 基座上。

如果你的 4B 第一次 LoRA 路径不是默认路径，请直接改 `config_region_mm.py`，或者运行时传参。

## 推荐训练命令

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

/home/T7/public/miniconda3/envs/finetune1/bin/python \
/home/T7/zyx/finetune/diecast_region_route/train_region_multimodal_v7_stage1_lora.py \
  --base-model /home/T7/zyx/finetune/myQwen3-4B-unsloth-bnb-4bit \
  --stage1-adapter /home/T7/zyx/finetune/util/Qwen3-4B-stage1-lora-adapter \
  --output-dir /home/T7/zyx/finetune/diecast_region_route/region_mm_outputs_v7_4b_stage1_lora \
  --epochs 2 \
  --max-seq-length 1536
```

## 如果路径已经写在 config 里，也可以直接运行

```bash
/home/T7/public/miniconda3/envs/finetune1/bin/python \
/home/T7/zyx/finetune/diecast_region_route/train_region_multimodal_v7_stage1_lora.py
```

## 二次微调输出

训练完成后主要看：

```text
region_mm_outputs_v7_4b_stage1_lora/best/
├── llm_adapter/
├── multimodal_heads.pt
└── meta.json
```

测试配置里的 `mm_weight_path` 指向：

```text
/home/T7/zyx/finetune/diecast_region_route/region_mm_outputs_v7_4b_stage1_lora/best
```
