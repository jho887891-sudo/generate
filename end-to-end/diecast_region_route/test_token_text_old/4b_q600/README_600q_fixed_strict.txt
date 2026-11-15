4B token_text 600问 GraphRAG 稳定性测试包（参数提示词 + 严格输出清洗修复版）

本包用于替换原 600 问稳定性测试包，保留 GraphRAG，只运行 token_text 模式。

修复内容：
1. 修复【压铸参数】为空的问题：
   - 新增 format_params_for_prompt(params)
   - 新增 build_info_text_for_prompt(cfg_json)
   - 从配置文件 params 字段自动生成【压铸参数摘要】
   - GraphRAG 检索也使用同一份“样本编号 + 压铸参数摘要”作为上下文

2. 增加强约束：
   - 用户提示词中明确要求：最终答案必须从【结论】开始
   - 最终答案必须在【建议方向】第3点结束后停止
   - 【结论】前不得出现 <think>、空标签、解释、问题原文或其他字符

3. 增强 normalize_output(text)：
   - 删除 <think>...</think>、<think>、</think>
   - 严格截取第一个【结论】开始的内容
   - 只保留【结论】【溯源分析】【建议方向】三段
   - 自动截断【建议方向】后的【问题】、【示例输出】、【作答边界】等提示词回显
   - 清理图谱规则编号、多模态token、图谱命中列表、调试信息等禁忌内容

4. 结果目录额外保存：
   - params_prompt_block.txt
   用于确认压铸参数确实进入提示词。

部署方式：
1. 解压本包，把三个文件放到服务器目录：
   /home/T7/wkq/end-to-end/diecast_region_route/test_token_text/

2. 运行：
   cd /home/T7/wkq/end-to-end/diecast_region_route/test_token_text
   bash run_token_text_4b_600q_graphrag.sh

3. 后台运行建议：
   mkdir -p /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_600q_graphrag
   nohup bash run_token_text_4b_600q_graphrag.sh > /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_600q_graphrag/overnight_4b_600q_graphrag_fixed_strict.log 2>&1 &

4. 查看日志：
   tail -f /home/T7/wkq/end-to-end/compare_outputs/logs_token_text_600q_graphrag/overnight_4b_600q_graphrag_fixed_strict.log

5. 跑完后检查参数是否进入提示词：
   cat /home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38066_600q_token_text_graphrag/params_prompt_block.txt

6. 检查任意问题提示词：
   cat /home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38066_600q_token_text_graphrag/q01/token_text/user_prompt.txt

7. 评分输入文件：
   /home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38066_600q_token_text_graphrag/all_modes_results.json

注意：
- 本包保留 use_graph_rag=true。
- 本包强制 compare_modes=["token_text"]。
- raw_answer 仍会保存原始生成内容，可能包含模型原始输出痕迹；正式评分和报告请使用 final_answer。
