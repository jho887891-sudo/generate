# Qwen2.5-VL-7B 九视角整工件压铸气孔溯源测试

这个版本按你的实验口径处理：**九张 X 光检测图属于同一个工件**，不是九个样本。

视觉总结阶段一次性输入：

- 9 张 X 光检测图
- 9 张模板参考图
- 9 张区域蒙版图

合计 27 张图。模型先自行输出视觉层总结，再基于该视觉总结、压铸参数和问题输出溯源回答。

## 放置位置

把三个文件放到：

```bash
/home/T7/wkq/end-to-end/vlm_baseline_eval/
```

## 运行

```bash
cd /home/T7/wkq/end-to-end/vlm_baseline_eval
export CUDA_VISIBLE_DEVICES=0
bash run_qwen25vl7b_diecast_9views_call.sh
```

## 输出

输出目录默认：

```text
/home/T7/wkq/end-to-end/compare_outputs/qwen25_vl_7b_sample_38069_9views/qwen25vl7b_run_时间戳/
```

重点文件：

```text
vlm_visual_summary.txt                 # 官方VLM自己的视觉层总结
vlm_visual_summary_for_scoring.json    # 解析后的视觉统计估计
trace_answers.jsonl                    # 每个问题的原始记录
trace_summary.csv                      # 简要汇总
all_modes_results.json                 # 直接给你的评分脚本读取
qwen25vl7b_trace_report.md             # 汇总报告
```

## 显存不够时

正式实验保持 `MAX_VIEWS=9`。如果只是排查环境，可以临时：

```bash
MAX_VIEWS=1 bash run_qwen25vl7b_diecast_9views_call.sh
```

跑通后再恢复九视角。
