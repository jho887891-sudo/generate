# SVP-Text 多模态溯源系统

这是一个可以整体拎走部署的 SVP-Text 溯源系统目录。

推荐放置位置：

```text
/home/T7/wkq/end-to-end/SVP-Text_多模态溯源系统
```

系统默认会寻找：

```text
/home/T7/wkq/end-to-end/diecast_region_route
```

并沿用其中的视觉层代码、配置文件和模型路径。

## 方法说明

SVP-Text 不是去掉图片输入，而是去掉大语言模型内部的多模态 prefix token 融合层。

实际数据流：

```text
用户上传图片 + 压铸参数 + 溯源问题
  -> 视觉层模型识别图片
  -> 模板/mask 完成区域归属
  -> 视觉统计转成文字总结
  -> 视觉文字总结 + 参数 + 问题 + 可选图谱证据 写入 prompt
  -> LLM 输出溯源结果
```

输出格式固定为：

```text
【结论】
...
【溯源分析】
1. ...
2. ...
3. ...
【建议方向】
1. ...
2. ...
3. ...
```

## 目录结构

```text
SVP-Text_多模态溯源系统/
  run_server.py
  start_server.sh
  requirements.txt
  README.md
  configs/
    svp_text_config.example.json
  svp_text_app/
    main.py
    trace_service.py
    llm_service.py
    prompting.py
    config.py
    bootstrap.py
    static/
      index.html
      styles.css
      app.js
```

## 安装依赖

如果原来的三模式脚本已经能在服务器上运行，通常只需要补 Web 依赖：

```bash
cd /home/T7/wkq/end-to-end/SVP-Text_多模态溯源系统
/home/T7/public/miniconda3/envs/finetune1/bin/pip install fastapi uvicorn python-multipart
```

完整安装：

```bash
cd /home/T7/wkq/end-to-end/SVP-Text_多模态溯源系统
/home/T7/public/miniconda3/envs/finetune1/bin/pip install -r requirements.txt
```

## 启动

```bash
cd /home/T7/wkq/end-to-end/SVP-Text_多模态溯源系统
bash start_server.sh
```

也可以直接运行：

```bash
cd /home/T7/wkq/end-to-end/SVP-Text_多模态溯源系统
/home/T7/public/miniconda3/envs/finetune1/bin/python run.py
```

访问：

```text
http://服务器IP:8088
```

## 常用环境变量

```bash
export SVP_TEXT_END_TO_END_ROOT=/home/T7/wkq/end-to-end
export DIECAST_REGION_ROUTE_ROOT=/home/T7/wkq/end-to-end/diecast_region_route
export SVP_TEXT_CONFIG_JSON=/home/T7/wkq/end-to-end/diecast_region_route/e2e_compare_graph_config_hardcoded.json
export SVP_TEXT_RUNTIME_DIR=/home/T7/wkq/end-to-end/compare_outputs/svp_text_runtime
export SVP_TEXT_PORT=8088
```

LLM 路径可以覆盖：

```bash
export DIECAST_LLM_BASE_PATH=/home/T7/wkq/end-to-end/models/Qwen3-4B-stage1-book-merged-4bit
export SVP_TEXT_LLM_ADAPTER_PATH=/home/T7/wkq/end-to-end/runs/qwen3_4b_stage2_from_merged_Third_sft/best_adapter
```

如果你填写的是二次微调运行目录，系统也会自动尝试使用其中的 `best_adapter`：

```text
/home/T7/wkq/end-to-end/runs/qwen3_4b_stage2_from_merged_Third_sft
```

当前默认 LoRA 使用二次微调文本 LoRA，不再默认挂载多模态阶段的 `mm_weight_path/llm_adapter`。

## 页面使用

1. 上传一张或多张探伤图片。
2. 填写样本编号。
3. 填写压铸参数，可以用表单，也可以切换 JSON。
4. 输入溯源问题，支持一行一个问题，一次提交多个问题。
5. 根据需要打开图谱辅助证据。
6. 点击“开始溯源分析”，页面会显示完整任务进度。
7. 任务完成后可查看多问题答案、视觉层识别图、视觉层文字总结，并下载本次输出文件。

页面顶部的配置区只显示当前载入文件名，完整路径可通过 `/api/config` 查看。

运行结果会保存到：

```text
/home/T7/wkq/end-to-end/compare_outputs/svp_text_runtime
```

每次请求都会生成独立运行目录，里面包含：

```text
request_config.json
visual/
result/svp_text_result.json
result/answer.txt
result/visual_fact_block.txt
result/user_prompt.txt
```

系统会在运行目录下维护 SQLite 历史库：

```text
svp_text_history.sqlite3
```

页面右侧会读取近 7 天溯源任务，用户可直接下载历史任务压缩包。

## API

```http
POST /api/trace/start
Content-Type: multipart/form-data
```

字段：

```text
images        多张图片
params_json   压铸参数 JSON
questions     溯源问题，支持一行一个问题
sample_id     样本编号
use_graph_rag 是否启用图谱辅助证据
```

健康检查：

```text
GET /api/health
GET /api/config
GET /api/jobs/{job_id}
GET /api/history?days=7
GET /api/runs/{run_id}/download
```
