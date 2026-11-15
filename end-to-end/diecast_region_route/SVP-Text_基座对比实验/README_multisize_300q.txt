0.6B / 1.7B / 4B / 8B / 14B SVP-Text 300问对比测试包

一、这个包现在做什么
1. 沿用原 4B/600q GraphRAG 测试数据、图片、参数和固定 300 个问题。
2. 当前主入口改为 SVP-Text：图片先经过视觉层识别，再把视觉层文字总结写入 prompt 进行溯源。
3. 不再加载 multimodal_heads.pt，不输入视觉/区域/参数 prefix token。
4. 五个尺寸保持各自 llm_base_path 不变，并挂载对应的 *_stage2_from_merged_Third_sft/best_adapter。
5. 每个尺寸单独输出一个 all_modes_results.json，内部保留 raw_answer 和 final_answer。
6. resume=true：如果 qXX/case_result.json 已存在，重启后跳过已完成问题。

二、文件说明
1. e2e_svp_text_300q_multisize_graphrag.py
   当前主测试脚本。每次只测一个尺寸，通过 E2E_CONFIG_JSON 切换 0.6B/1.7B/4B/8B/14B。

2. e2e_token_text_only_300q_multisize_graphrag.py
   历史多模态 token_text 入口，保留作备份；当前运行脚本不再调用它。

3. e2e_token_text_config_4b_600q_sample38063_graphrag.json
   原始 4B/600q 配置，作为生成 300 问配置的基础。

4. questions_300_fixed_from_600q.json
   固定 300 个问题，五个尺寸共用。

5. model_jobs_5size_300q.json
   五个尺寸的基座和 Stage2 LoRA 路径表。服务器上实际目录名不同，只改这个文件即可。

6. make_5size_300q_configs.py
   根据原始配置 + 固定问题 + model_jobs_5size_300q.json 生成五个尺寸配置。

7. run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh
   稳定后台运行入口，名称保留，但实际调用 SVP-Text 主脚本。

三、运行前重点检查
打开 model_jobs_5size_300q.json，重点检查每个尺寸：
1. llm_base_path
2. llm_stage2_adapter_path
3. min_free_gpu_gb

SVP-Text 主脚本不会读取 mm_weight_path / mm_output_dir。

四、前台试跑
建议先只跑 0.6B：
bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh 0p6b

再跑 4B：
bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh 4b

五、后台跑五个尺寸
mkdir -p /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q
nohup bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh > /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q/overnight_5size_300q_svp_text.log 2>&1 &

查看主日志：
tail -f /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q/overnight_5size_300q_svp_text.log

五点一、40G 显存推荐分组跑
如果服务器可用显存约 40G，推荐：
第一批并行跑 0.6B / 1.7B / 4B；
第一批完成后，第二批顺序跑 8B / 14B。

后台命令：
mkdir -p /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q
nohup bash run_svp_text_batch_0p6_1p7_4_then_8_14.sh > /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q/staged_5size.log 2>&1 &

查看日志：
tail -f /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q/staged_5size.log

不建议 8B 和 14B 在 40G 显存上同时跑；如果你确认显存足够，可以强制并行：
RUN_LARGE_PARALLEL=1 nohup bash run_svp_text_batch_0p6_1p7_4_then_8_14.sh > /home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q/staged_5size_parallel_large.log 2>&1 &

六、输出位置
默认输出：
/home/T7/wkq/end-to-end/compare_outputs/e2e_0.6b_sample_样本号_300q_svp_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_1.7b_sample_样本号_300q_svp_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_样本号_300q_svp_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_8b_sample_样本号_300q_svp_text_graphrag/all_modes_results.json
/home/T7/wkq/end-to-end/compare_outputs/e2e_14b_sample_样本号_300q_svp_text_graphrag/all_modes_results.json

每个问题保存为：
q01/svp_text/raw_answer.txt
q01/svp_text/final_answer.txt
q01/svp_text/e2e_result.json
q01/svp_text/user_prompt.txt

七、中断后继续跑
脚本默认：
resume=true
SKIP_DONE=1

含义：
1. 如果某个尺寸 all_modes_results.json 已经存在，run 脚本默认跳过该尺寸。
2. 如果某个尺寸中途断了，重新跑同一个尺寸时，主脚本会跳过已有 qXX/case_result.json。

强制重跑某个尺寸：
SKIP_DONE=0 bash run_token_text_0p6_1p7_4_8_14b_300q_graphrag.sh 4b

如果要彻底重跑，先删除对应输出目录：
rm -rf /home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_样本号_300q_svp_text_graphrag

八、关键变化
1. 基座模型路径保持各尺寸配置不变。
2. 溯源分析统一挂载 llm_stage2_adapter_path，即 *_stage2_from_merged_Third_sft/best_adapter。
3. 视觉层仍然运行，但只把识别结果转成文字总结放进 prompt。
4. LLM 推理只走 input_ids/attention_mask，不再走 inputs_embeds 和 prefix tokens。
