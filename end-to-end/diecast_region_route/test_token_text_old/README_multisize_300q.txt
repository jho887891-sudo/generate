0.6B / 1.7B / 4B / 8B / 14B 多模态大模型 token_text 300问对比测试包

一、这个包做了什么
1. 沿用你上传的 4B/600q GraphRAG token_text 测试逻辑。
2. 只运行 token_text，不跑 token_only/text_only。
3. 从原 600 问配置中固定抽取前 300 问，保存为 questions_300_fixed_from_600q.json。
4. 0.6B、1.7B、4B、8B、14B 五个尺寸全部读取同一个 questions_300_fixed_from_600q.json，保证问题完全一致。
5. 每个尺寸单独输出一个 all_modes_results.json，内部仍然保留：
   - raw_answer：模型原始生成内容
   - final_answer：经过 normalize_output 抽取和清洗后的正式答案
6. final_answer 不是 raw_answer 的复制，仍然保留原脚本的抽取/清洗逻辑。
7. 增加 resume=true：如果 qXX/case_result.json 已经存在，重启后会跳过已完成问题。
8. 后台稳定运行方式采用 shell 逐尺寸顺序启动，每个尺寸是一个独立 Python 进程，便于释放显存，避免 5 个尺寸挤在同一进程里。

二、文件说明
1. e2e_token_text_only_300q_multisize_graphrag.py
   通用测试脚本。每次只测一个尺寸，但通过配置文件切换 0.6B/1.7B/4B/8B/14B。

2. e2e_token_text_config_4b_600q_sample38063_graphrag.json
   你上传的原始 4B/600q 配置，作为生成 300 问配置的基础。

3. questions_300_fixed_from_600q.json
   从原配置抽取的固定 300 个问题，五个尺寸共用。

4. model_jobs_5size_300q.json
   五个尺寸的模型路径表。服务器上实际目录名如果和默认写法不同，只改这个文件即可。

5. make_5size_300q_configs.py
   根据原始配置 + 固定问题 + model_jobs_5size_300q.json 生成五个尺寸的配置文件。

6. configs/e2e_token_text_config_*.json
   已经预生成的五个尺寸配置文件。

7. run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh
   稳定后台运行入口。

三、部署位置
建议放到：
/home/T7/wkq/end-to-end/diecast_region_route/test_token_text_multisize_300q

部署命令：
mkdir -p /home/T7/wkq/end-to-end/diecast_region_route/test_token_text_multisize_300q
cd /home/T7/wkq/end-to-end/diecast_region_route/test_token_text_multisize_300q
# 把本包全部文件复制到这个目录

四、运行前必须检查的路径
打开：
model_jobs_5size_300q.json

重点检查每个尺寸的：
1. mm_weight_path
2. mm_output_dir
3. llm_base_path
4. llm_stage2_adapter_path

默认命名按你现有 4B 路径风格写成：
qwen3_0p6b_region_mm_frozen_llm_from_merged_stage1
qwen3_1p7b_region_mm_frozen_llm_from_merged_stage1
qwen3_4b_region_mm_frozen_llm_from_merged_stage1
qwen3_8b_region_mm_frozen_llm_from_merged_stage1
qwen3_14b_region_mm_frozen_llm_from_merged_stage1

如果你服务器上实际目录是 qwen3_0_6b、qwen3_0.6b 或其他命名，改 model_jobs_5size_300q.json 即可，不需要改主测试代码。

五、前台试跑
建议先只跑 0.6B，确认流程通：
bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh 0p6b

再跑 4B：
bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh 4b

六、后台跑五个尺寸
mkdir -p /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_5size_300q_graphrag
nohup bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh > /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_5size_300q_graphrag/overnight_5size_300q.log 2>&1 &

查看主日志：
tail -f /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_5size_300q_graphrag/overnight_5size_300q.log

查看某个尺寸日志：
ls -lh /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_5size_300q_graphrag
tail -f /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_5size_300q_graphrag/4b_300q_graphrag_*.log

七、输出位置
默认输出：
/home/T7/wkq/end-to-end/compare_outputs/e2e_0p6b_sample_样本号_300q_token_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_1p7b_sample_样本号_300q_token_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_样本号_300q_token_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_8b_sample_样本号_300q_token_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_14b_sample_样本号_300q_token_text_graphrag/all_modes_results.json

每个问题仍然有：
q01/token_text/raw_answer.txt
q01/token_text/final_answer.txt
q01/token_text/e2e_result.json
q01/token_text/user_prompt.txt

八、中断后继续跑
脚本默认：
resume=true
SKIP_DONE=1

含义：
1. 如果某个尺寸 all_modes_results.json 已经存在，run 脚本默认跳过该尺寸。
2. 如果某个尺寸中途断了，重新跑同一个尺寸时，主脚本会跳过已有 qXX/case_result.json，从断点附近继续。

强制重跑某个尺寸：
SKIP_DONE=0 bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh 4b

如果要彻底重跑，先删除对应输出目录：
rm -rf /home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_样本号_300q_token_text_graphrag

九、几个关键修复点
1. 不再把脚本写死为 4B。
2. load_current_model 会从当前尺寸 config 读取 llm_base_path、llm_stage2_adapter_path、mm_output_dir、mm_weight_path。
3. 每个尺寸共用同一个 questions_300_fixed_from_600q.json。
4. 保留 raw_answer 和 final_answer，并保留 normalize_output 抽取逻辑。
5. 用独立进程逐尺寸运行，后台更稳。
