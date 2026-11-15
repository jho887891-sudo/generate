# 多模态映射层语料按工件去重报告

## 结论

原始文件共有 1779 条多模态问答样本，但按 sample_id 统计只有 186 个独立工件。已按“一工件一条”生成去重测试集。

> 你说的 187 条，我在当前压缩包里实际检查到的是 186 个独立 sample_id；没有发现第 187 个独立工件。

## 去重结果

| 数据集 | 原始条数 | 独立工件数 | 去重后条数 | 删除重复问答 | 每工件问题数范围 |
|---|---:|---:|---:|---:|---|
| all | 1779 | 186 | 186 | 1593 | 8～13 |
| train | 1501 | 158 | 158 | 1343 | 8～13 |
| val | 278 | 28 | 28 | 250 | 8～13 |

## 保留样本选择规则

同一个 sample_id 下只保留 1 条，优先保留更适合测试整体诊断能力的问题：open_diagnosis > cause_explanation > effect_analysis > binary_decision > yes_no_judgment > adjustment_validation > relation_mechanism > inspection_identification。

如果同类型有多条，则优先保留包含“主要风险、怎么判断、当前缺陷、气孔风险、优先验证、整体充型、参数”等关键词的问题。

## 可用性校验

- all: 9张图路径完整=True；summary完整=True；params完整=True；output完整=True。
- train: 9张图路径完整=True；summary完整=True；params完整=True；output完整=True。
- val: 9张图路径完整=True；summary完整=True；params完整=True；output完整=True。
- train/val sample_id 交叉数量：0。

## 文件说明

- `data/stage2_mm_mapping_all_unique_by_workpiece.jsonl`：全量独立工件测试集，186条。
- `data/stage2_mm_mapping_train_unique_by_workpiece.jsonl`：训练集独立工件版，158条。
- `data/stage2_mm_mapping_val_unique_by_workpiece.jsonl`：验证集独立工件版，28条。
- `reports/removed_duplicate_questions.csv`：每个工件被删除的问题明细，方便追溯。
