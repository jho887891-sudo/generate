# Stage2 纯文本 LoRA 清洗语料

## 文件

- `data/stage2_text_lora_pure_train.jsonl`
- `data/stage2_text_lora_pure_val.jsonl`
- `data/stage2_text_lora_pure_all.jsonl`
- `reports/clean_report.json`

## 清洗结果

- train: 1501
- val: 278
- all: 1779

## 清洗规则

1. 只保留纯文本训练字段：`instruction`、`input`、`output`。
2. 去掉多模态映射字段：`image_paths`、`view_items`、`boxes_by_view`、`defect_meta_by_view` 等。
3. 保留并写入 `input` 的内容：
   - 总气孔条目
   - 总面积
   - 最大等效直径
   - 区域缺陷统计
   - 区域最大面积
   - 区域最大等效直径
   - 压铸参数
   - 用户问题
4. 背景/未知区域只删除对应项，不删除整行有效区域信息。
5. 温度/湿度统一为环境温度/环境湿度。
6. 缺陷类型统一按气孔。
7. output 重新生成严格三段式：
   - 【结论】
   - 【溯源分析】
   - 【建议方向】

## 训练代码路径建议

```python
TRAIN_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage2_text_lora_pure/data/stage2_text_lora_pure_train.jsonl"
VAL_DATA_PATH = "/home/T7/wkq/end-to-end/model_data/stage2_text_lora_pure/data/stage2_text_lora_pure_val.jsonl"
```
