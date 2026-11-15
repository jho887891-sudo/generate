# SVP-Text LoRA 对比实验

这组脚本用于比较同一输入形式下的两个 LoRA。

固定基座：

```text
/home/T7/wkq/end-to-end/models/Qwen3-4B-Instruct-2507-bnb-4bit
```

输入形式：

```text
图片 -> 视觉层识别 -> 视觉层文字总结 -> 写入 prompt -> LLM 溯源
```

问题、参数、图片默认都读取：

```text
/home/T7/wkq/end-to-end/diecast_region_route/e2e_compare_graph_config_hardcoded.json
```

## 两个实验入口

实验 1：只进行第二次微调 LoRA

```bash
cd /home/T7/wkq/end-to-end/diecast_region_route/svp_text_lora_compare
/home/T7/public/miniconda3/envs/finetune1/bin/python test_lora_second_only.py
```

挂载 LoRA：

```text
/home/T7/wkq/end-to-end/runs/qwen3_4b_只进行第二次微调
```

实验 2：stage1 expert clean best LoRA

```bash
cd /home/T7/wkq/end-to-end/diecast_region_route/svp_text_lora_compare
/home/T7/public/miniconda3/envs/finetune1/bin/python test_lora_stage1_expert.py
```

挂载 LoRA：

```text
/home/T7/wkq/end-to-end/runs/qwen3_4b_stage1_expert_clean_best
```

脚本会自动兼容以下目录形式：

```text
LoRA目录/
LoRA目录/best_adapter/
LoRA目录/adapter/
LoRA目录/llm_adapter/
```

只要里面存在 `adapter_config.json`，就会自动选中。

## 输出

默认输出到：

```text
/home/T7/wkq/end-to-end/compare_outputs/svp_text_lora_compare
```

每次运行会生成独立目录，例如：

```text
lora_second_only_20260529_180000/
lora_stage1_expert_20260529_180000/
```

主要文件：

```text
experiment_meta.json
visual_fact_block.txt
visual_summary.json
answers.md
all_results.json
partial_results.json
q001/
  user_prompt.txt
  answer.txt
  raw_answer.txt
  case_result.json
```

## 常用参数

只跑前 5 个问题做冒烟测试：

```bash
python test_lora_second_only.py --limit 5
python test_lora_stage1_expert.py --limit 5
```

指定配置文件：

```bash
python test_lora_second_only.py --config /home/T7/wkq/end-to-end/diecast_region_route/e2e_compare_graph_config_hardcoded.json
```

启用 GraphRAG：

```bash
python test_lora_second_only.py --use-graph
```

默认不启用 GraphRAG，这样对比更集中在 LoRA 本身。

