# 五模型逐视角视觉判断 + 顺序溯源测试

## 这版解决的问题

1. 不再把 27 张图一次性喂给模型，而是逐视角判断：每次输入 `X光检测图 + 模板图 + 蒙版图` 三张图。
2. 视觉层总结和溯源回答分开保存。
3. `all_modes_results.json` 适配你的 `standard_e2e_scoring_v2.py`。
4. 修复 InternVL 启动失败：不再走容易重复传 `trust_remote_code` 的 pipeline，而是走 `AutoModel + AutoTokenizer + model.chat`。
5. 修复 LLaVA 路径失败：配置里支持 `path_globs` 自动查找 `llava/onevision` 相关目录。
6. 新增 `validate_vlm_visual_results.py`，用于验证官方 VLM 的视觉层结果。

## 放置位置

把这些文件放到：

```bash
/home/T7/wkq/end-to-end/vlm_baseline_eval/
```

包括：

```text
run_five_vlm_step_view_trace.py
config_five_vlm_step_view_sample38069.json
run_five_vlm_step_view_trace.sh
validate_vlm_visual_results.py
```

## 先查 LLaVA / InternVL 真实路径

```bash
find /home/T7/wkq/end-to-end/vlm_baseline_eval/vlm_baselines -maxdepth 3 -type d \
  \( -iname "*llava*" -o -iname "*onevision*" -o -iname "*InternVL*" \)
```

如果 LLaVA 实际目录不是 `LLaVA-OneVision`，可以不改配置，脚本会用 `path_globs` 自动搜索；如果搜索不到，再手动改 `config_five_vlm_step_view_sample38069.json` 里的 `llava_onevision.path`。

## 调试命令

先跑一个模型、一个视角：

```bash
cd /home/T7/wkq/end-to-end/vlm_baseline_eval
export CUDA_VISIBLE_DEVICES=0
MAX_VIEWS=1 MODELS=qwen25_vl_7b bash run_five_vlm_step_view_trace.sh
```

再测试两个之前失败的模型：

```bash
MAX_VIEWS=1 MODELS=internvl35_8b_hf,llava_onevision bash run_five_vlm_step_view_trace.sh
```

## 正式跑五个模型九视角

```bash
cd /home/T7/wkq/end-to-end/vlm_baseline_eval
mkdir -p logs
RUN_LOG="logs/five_vlm_per_view_$(date +%Y%m%d_%H%M%S).out"
nohup env CUDA_VISIBLE_DEVICES=0 MAX_VIEWS=9 MODELS=qwen25_vl_7b,qwen3_vl_8b,internvl35_8b_hf,minicpm_v45,llava_onevision \
  bash run_five_vlm_step_view_trace.sh > "$RUN_LOG" 2>&1 &
echo "PID=$! LOG=/home/T7/wkq/end-to-end/vlm_baseline_eval/$RUN_LOG"
```

查看日志：

```bash
tail -n 100 /home/T7/wkq/end-to-end/vlm_baseline_eval/logs/five_vlm_per_view_*.out
```

## 输出目录

默认输出到：

```text
/home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/
```

关键文件：

```text
all_modes_results.json                  # 溯源评分输入，给你的 standard_e2e_scoring_v2.py
all_models_results.json                 # 同内容备份
visual_summaries_by_model.json          # 五个模型视觉层聚合总结
model_run_summary.csv                   # 哪些模型成功/失败
trace_answers_all.jsonl                 # 所有溯源回答
trace_summary_all.csv                   # 所有回答简表
five_vlm_per_view_seq_report.md         # 总报告
```

每个模型内部：

```text
模型名/visual_layer/view_01/visual_raw.txt       # 第1视角视觉原始输出
模型名/visual_layer/view_01/visual_parsed.json   # 第1视角视觉解析结果
模型名/visual_layer/visual_summary_aggregate.json
模型名/visual_layer/visual_summary_aggregate.txt
模型名/trace_layer/q01/final_answer.txt
```

## 溯源评分

```bash
/home/T7/public/miniconda3/envs/finetune1/bin/python /你的评分脚本路径/standard_e2e_scoring_v2.py \
  --input /home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/all_modes_results.json \
  --output-dir /home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/trace_score
```

## 视觉层验证

只做自一致性检查：

```bash
/home/T7/public/miniconda3/envs/finetune1/bin/python /home/T7/wkq/end-to-end/vlm_baseline_eval/validate_vlm_visual_results.py \
  --input /home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/all_modes_results.json \
  --output-dir /home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/visual_validation
```

如果你有自己的视觉层参考结果，比如：

```text
/home/T7/wkq/end-to-end/compare_outputs/e2e_8b_sample_38069/visual_summary_filtered.json
```

可以做对比验证：

```bash
/home/T7/public/miniconda3/envs/finetune1/bin/python /home/T7/wkq/end-to-end/vlm_baseline_eval/validate_vlm_visual_results.py \
  --input /home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/all_modes_results.json \
  --reference-json /home/T7/wkq/end-to-end/compare_outputs/e2e_8b_sample_38069/visual_summary_filtered.json \
  --output-dir /home/T7/wkq/end-to-end/compare_outputs/five_vlm_per_view_sample_38069_9views/five_vlm_per_view_run_时间戳/visual_validation
```

输出：

```text
visual_validation_results.json
visual_validation_summary.csv
visual_validation_report.md
```
