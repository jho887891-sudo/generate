# 三模式端到端对比 + 自动评分（路径写死版）

## 这版的改动

- 对比脚本默认配置路径已写死：`/home/T7/wkq/end-to-end/test_fllow/e2e_compare_graph_4b_config_hardcoded.json`。
- 正常运行不需要输入任何路径参数。
- 配置文件输入使用 `image_dir`，不是 `image_paths`。
- 图片匹配准则：按文件名自然排序后，第1张图对应 `image1.jpg/image1_mask.png`，第2张图对应 `image2.jpg/image2_mask.png`，以此类推。
- 默认不从 `dataset_jsonl` 读取视觉结果，而是从 `image_dir` 重新跑视觉层。
- 视觉层里背景/未知区域缺陷直接跳过，不进入统计、token 元数据或文字总结。

## 安装到服务器

```bash
cd /home/T7/wkq/end-to-end
unzip -o e2e_4b_compare_scoring_hardcoded_v2.zip
cp e2e_4b_compare_scoring_hardcoded_v2/e2e_compare_three_modes_graph_4b.py /home/T7/wkq/end-to-end/test_fllow/
cp e2e_4b_compare_scoring_hardcoded_v2/score_e2e_comparison.py /home/T7/wkq/end-to-end/test_fllow/
cp e2e_4b_compare_scoring_hardcoded_v2/e2e_compare_graph_4b_config_hardcoded.json /home/T7/wkq/end-to-end/test_fllow/
```

## 修改图片文件夹

只改配置文件里的：

```json
"image_dir": "/home/T7/wkq/end-to-end/diecast_region_route_4B/images/47242"
```

## 运行三模式对比

```bash
cd /home/T7/wkq/end-to-end/diecast_region_route_4B
/home/T7/public/miniconda3/envs/finetune1/bin/python -u /home/T7/wkq/end-to-end/test_fllow/e2e_compare_three_modes_graph_4b.py
```

输出：

```text
/home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38053/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38053/compare_report.md
```

## 运行评分

```bash
/home/T7/public/miniconda3/envs/finetune1/bin/python /home/T7/wkq/end-to-end/test_fllow/score_e2e_comparison.py
```

默认读取：

```text
/home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38053/all_modes_results.json
```

默认输出：

```text
/home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38053_scores/
```
