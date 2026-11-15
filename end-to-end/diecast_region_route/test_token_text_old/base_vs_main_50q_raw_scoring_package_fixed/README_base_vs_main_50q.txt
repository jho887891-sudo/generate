【修复说明】
1. 已修复 CONFIG_JSON 不读取 E2E_CONFIG_JSON 的问题。
2. 已补全 build_info_text_for_prompt / format_params_for_prompt，确保配置中的 params 会进入【压铸参数】提示词。
3. 评分对象仍为 raw：final_answer = raw_answer。

# 主方法 vs 未微调基座模型：50问 raw 输出对比测试包

## 测试目的

用同一套视觉统计、压铸参数、GraphRAG 检索证据和 50 个问题，对比：

1. `main_token_text_raw`：你的主方法，token_text，使用视觉文字总结 + 视觉/区域 prefix token + 参数 + GraphRAG。
2. `base_text_only_raw`：未微调基座模型，text_only，只使用视觉文字总结 + 参数 + GraphRAG + 问题，不加载 LoRA，不加载多模态映射层。

本包已经把 `final_answer` 设置为 `raw_answer`，因此你的评分脚本如果读取 `final_answer`，实际评分对象就是 raw。

## 问题来源

从你上传的 300 个问题中按均匀间隔抽取 50 个，覆盖区域风险、参数链、GraphRAG、坐标/视角验证、单参数误判等类型。

## 文件说明

```text
base_vs_main_50q_raw_scoring.py          # 测试脚本
base_vs_main_50q_config.json             # 配置文件，已包含50个问题
questions_50_from_300.json               # 50个问题及原始索引
run_base_vs_main_50q_raw.sh              # 前台运行脚本
start_background_base_vs_main_50q_raw.sh # 后台运行脚本
score_raw_with_standard_v2.sh            # 调用你的评分脚本评分
```

## 使用方式

解压到：

```bash
/home/T7/wkq/end-to-end/diecast_region_route/test_token_text
```

前台运行：

```bash
cd /home/T7/wkq/end-to-end/diecast_region_route/test_token_text
bash run_base_vs_main_50q_raw.sh
```

后台运行：

```bash
cd /home/T7/wkq/end-to-end/diecast_region_route/test_token_text
bash start_background_base_vs_main_50q_raw.sh
```

查看日志：

```bash
tail -f /home/T7/wkq/end-to-end/compare_outputs/logs_base_vs_main_50q/overnight_base_vs_main_50q_raw.log
```

## 重要：基座模型路径

配置文件中默认写了：

```text
/home/T7/wkq/end-to-end/models/Qwen3-4B-unsloth-bnb-4bit
```

如果你服务器真实的未微调基座模型不在这个路径，请二选一：

### 方法一：直接改配置

修改：

```text
base_vs_main_50q_config.json -> base_llm_path
```

### 方法二：运行时覆盖

```bash
BASE_LLM_PATH=/你的/未微调/Qwen3-4B路径 bash run_base_vs_main_50q_raw.sh
```

## 输出结果

```text
/home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071/all_modes_results.json
```

里面每题包含两个 mode：

```text
main_token_text_raw
base_text_only_raw
```

并且：

```text
final_answer = raw_answer
normalized_answer_for_reference = normalize_output(raw_answer)
```

## 评分

如果你的评分脚本是 `standard_e2e_scoring_v2.py`，可以运行：

```bash
bash score_raw_with_standard_v2.sh
```

如果评分脚本路径不同：

```bash
SCORER=/path/to/standard_e2e_scoring_v2.py bash score_raw_with_standard_v2.sh
```

## 建议检查点

跑完后建议看：

```bash
cat /home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071/q001/main_token_text_raw/raw_answer.txt
cat /home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071/q001/base_text_only_raw/raw_answer.txt
```

如果基座模型 raw 也出现重复标题、提示词回显或 `<think>`，说明这类异常不是主方法独有，更可能与长提示词、模型原生指令跟随能力、解码边界有关。

【2026-05-22 二次修复说明】
1. 修复 build_user_prompt() 缺少 question_type_hint 参数导致的 TypeError。
2. 修复 get_test_questions() 固定截断为 12 题的问题，现在按配置 max_questions=50 执行。
3. 重新运行后日志应显示 question_count = 50。
