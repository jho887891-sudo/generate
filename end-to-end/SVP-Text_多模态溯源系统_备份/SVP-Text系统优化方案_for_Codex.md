# SVP-Text 多模态溯源系统优化方案（给 Codex）

## 1. 总体定位

当前项目已经具备一个可运行的 SVP-Text 原型：FastAPI 后端 + 原生 HTML/CSS/JS 前端。核心链路是：上传探伤图片和压铸参数 → 调用视觉层 VisualStageRunner → 过滤并聚合缺陷区域统计 → 将视觉统计转成文字事实块 → 可选图谱检索 → 构造 prompt → Qwen/LoRA 文本模型生成溯源回答 → 保存结果和历史记录。

从硕士课题系统角度，当前系统不能只作为“能调用模型的网页”，而应改成“主方法演示 + 证据链可视化 + 可复现实验管理”的系统。页面需要让答辩老师一眼看懂：SVP-Text 为什么不是简单聊天机器人，而是把视觉模型、区域模板、压铸参数、图谱证据和 LLM 溯源组合成一个有证据链的工业缺陷溯源流程。

## 2. 当前代码结构

- `svp_text_app/main.py`：FastAPI 路由，包含首页、健康检查、配置、异步任务启动、任务查询、历史和下载接口。
- `svp_text_app/trace_service.py`：主业务链路，负责保存图片、构造运行目录、调用视觉层、生成视觉文字总结、调用 LLM、写结果、写历史。
- `svp_text_app/llm_service.py`：加载基础模型和 LoRA，执行文本生成。
- `svp_text_app/prompting.py`：构造 SVP-Text system prompt、视觉事实块、问题类型提示和最终 user prompt。
- `svp_text_app/static/index.html`：当前单页布局。
- `svp_text_app/static/app.js`：前端参数表单、图片预览、任务提交、轮询、结果渲染。
- `svp_text_app/static/styles.css`：页面样式。

Python 语法检查已通过：`python -m py_compile run.py run_server.py svp_text_app/*.py`。

完整模型运行未在当前环境验证，因为它依赖服务器上的 `diecast_region_route`、视觉模型、Qwen 基座模型和 LoRA 权重。

## 3. P0 必须修改的问题

### 3.1 样本编号和运行目录必须做安全清洗

当前 `_make_run_dir(sample_id)` 直接把 `sample_id` 拼进 `run_id`，如果样本编号包含 `/`、`..`、空格或特殊字符，会造成目录结构混乱，严重时可能影响文件访问和下载接口。

修改要求：

- 在 `trace_service.py` 增加 `sanitize_run_tag(text: str) -> str`。
- 只保留中文、英文、数字、下划线、短横线。
- 其他字符替换为 `_`。
- 限制长度，例如 40 字符。
- `_make_run_dir` 使用清洗后的 tag。
- API 返回 `display_sample_id` 保留原始样本编号，`run_id` 只使用安全字符串。

示例：

```python
def sanitize_run_tag(text: str, max_len: int = 40) -> str:
    raw = str(text or "sample").strip()
    raw = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", raw)
    raw = raw.strip("._-") or "sample"
    return raw[:max_len]
```

### 3.2 文件访问路径检查要改为 `relative_to`

当前 `resolve_run_file` 使用字符串 `startswith` 判断路径是否在运行目录下，不够严谨。

修改要求：

```python
try:
    target.relative_to(run_dir)
except ValueError:
    raise ValueError("非法文件路径。")
```

并且对 `run_id` 做格式校验，只允许安全字符。

### 3.3 GPU 任务必须串行化或队列化

当前 `/api/trace/start` 使用 `BackgroundTasks`，但没有全局任务队列和 GPU 锁。多个用户或多次点击可能并发加载视觉模型和 LLM，导致显存爆炸或生成结果互相影响。

修改要求：

- 在 `SVPTextTraceService` 内增加全局 `threading.Lock` 或 `asyncio.Queue`。
- `run_job` 进入 `_analyze_saved_job` 前必须获取 GPU 锁。
- 前端提交后禁用按钮已经有，但后端仍要防并发。
- 任务状态增加 `queued_position` 或至少明确显示“等待 GPU 队列”。

最小改法：

```python
class SVPTextTraceService:
    def __init__(self):
        self.jobs = {}
        self._gpu_lock = threading.Lock()

    def run_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return
        try:
            self._set_status(job, status="running", stage="queued", progress=3, message="任务进入 GPU 推理队列。")
            with self._gpu_lock:
                result = self._analyze_saved_job(job)
            ...
```

### 3.4 LLM 上下文长度不能硬编码 2048

`llm_service.py` 中 `generate(..., max_seq_length=2048)` 会导致 prompt 被截断。SVP-Text 的关键证据在 prompt 前部，当前 tokenizer 又设置了 `truncation_side = "left"`，可能把视觉事实和系统约束截掉。

修改要求：

- 配置里增加 `max_seq_length`，默认 4096 或 8192。
- `_analyze_saved_job` 调用 `llm.generate` 时传入 `max_seq_length=int(cfg_json.get("max_seq_length", settings.max_seq_length))`。
- 输出结果中保存 `prompt_char_count`、`prompt_token_count`、`max_seq_length`、`was_truncated`。
- 图谱证据先截断，不能让图谱挤掉视觉事实和压铸参数。
- 不建议默认 `truncation_side="left"`。如必须截断，应优先压缩 graph_context 和历史文本，保留 system prompt、视觉事实、参数和问题。

### 3.5 前端 JSON/表单切换会丢失额外参数

当前参数表单只渲染固定字段。用户在 JSON 里添加额外字段后，切回表单再提交，额外字段容易被丢弃。

修改要求：

- 方案 A：保持 JSON 为权威输入，表单只编辑常用字段，提交时合并 JSON 中的额外字段。
- 方案 B：增加“添加参数”功能，动态渲染额外字段。
- 推荐方案 A：`state.extraParams` 保存不在 `paramFields` 中的字段。

### 3.6 API 结果结构需要面向前端和论文统一

当前 `result` 中有 `visual_summary`、`visual_fact_block`、`visual_gallery`、`results`，但缺少页面可直接渲染的结构化表格数据。

修改要求：

返回结构建议调整为：

```json
{
  "method": "SVP-Text",
  "run_id": "...",
  "sample_id": "...",
  "created_at": "...",
  "input": {
    "image_count": 2,
    "params": {},
    "questions": []
  },
  "pipeline": [
    {"key": "vision", "name": "视觉识别", "status": "completed", "duration_ms": 1234},
    {"key": "visual_text", "name": "视觉统计转文本", "status": "completed", "duration_ms": 100},
    {"key": "graph", "name": "图谱证据检索", "status": "skipped/completed", "duration_ms": 0},
    {"key": "llm", "name": "LLM 溯源生成", "status": "completed", "duration_ms": 5000}
  ],
  "visual": {
    "summary": {},
    "fact_block": "...",
    "gallery": [],
    "region_cards": [],
    "defect_table": []
  },
  "params": {
    "raw": {},
    "table": []
  },
  "answers": [
    {
      "index": 1,
      "question": "...",
      "question_type": "...",
      "answer": "...",
      "evidence": {
        "focus_regions": [],
        "mechanism_candidates": [],
        "graph_used": false
      },
      "artifacts": {
        "prompt_url": "/api/runs/.../files/result/prompts/q01_prompt.txt",
        "answer_url": "/api/runs/.../files/result/answers/q01_answer.txt"
      }
    }
  ],
  "download_url": "..."
}
```

## 4. 页面渲染改造方案

### 4.1 页面总布局

建议把当前三栏页面改为“顶部状态 + 左侧输入 + 中央证据链 + 右侧结果/历史”的研究型界面。

推荐布局：

```text
┌─────────────────────────────────────────────────────────────┐
│ SVP-Text 多模态压铸气孔溯源系统                              │
│ 方法标签：Vision → Structured Visual Prompt → Text LLM        │
│ 服务状态 / 配置 / 模型 / LoRA / GPU队列状态                    │
└─────────────────────────────────────────────────────────────┘

┌───────────────┬────────────────────────────┬────────────────┐
│ 输入区         │ 方法证据链与视觉结果          │ 溯源答案与历史   │
│ 1 图片上传      │ A 流程时间线                 │ 结果卡片         │
│ 2 样本参数      │ B 视觉识别图                 │ 证据折叠         │
│ 3 问题与开关    │ C 区域统计卡片               │ 下载/历史        │
│ 4 提交          │ D 参数表/Prompt预览          │                │
└───────────────┴────────────────────────────┴────────────────┘
```

### 4.2 顶部应该渲染什么

顶部不能只显示模型文件名，应显示“方法身份”和“运行配置摘要”。

渲染字段：

- 方法名：`SVP-Text`
- 方法说明短句：`视觉层先识别缺陷并生成结构化文本，再由文本 LLM 完成溯源。`
- 配置文件名：`config_name`
- 视觉模型名：`visual_model_name`
- LLM 基座名：`llm_base_name`
- LoRA 名：`llm_adapter_name`
- 图谱开关默认状态
- 当前 GPU 队列状态或当前任务状态

### 4.3 输入区怎么渲染

输入区应分 4 个卡片：

1. 样本信息
   - 样本编号
   - 探伤图编号
   - 料号/批次（可选）
   - 备注（可选）

2. 图片上传
   - 支持拖拽上传
   - 显示缩略图、文件名、大小、顺序编号
   - 支持删除单张图片和调整顺序

3. 压铸参数
   - 常用参数表单
   - JSON 高级编辑
   - 参数校验结果：缺失字段、高风险字段、非数字字段
   - 额外参数保留，不要丢失

4. 溯源问题
   - 多问题输入
   - 提供问题模板按钮：
     - “总体风险判断”
     - “牙孔区域异常分析”
     - “焊接区域异常分析”
     - “参数链优先级判断”
     - “同工况验证计划”
   - 图谱辅助开关
   - 高级选项折叠：`max_new_tokens`、`max_seq_length`、`graph_top_k`

### 4.4 中央证据链怎么渲染

中央是硕士课题展示的重点。不能只显示一张带框图，应显示“证据链”。

#### A. 流程时间线

渲染 `pipeline`：

```text
1 图片上传完成
2 视觉模型识别完成
3 区域模板/mask 归属完成
4 视觉统计转文本完成
5 图谱证据检索完成/跳过
6 LLM 溯源生成完成
7 输出打包完成
```

每一步显示：状态、耗时、产物链接。

#### B. 视觉识别图

当前 `_draw_visual_results` 已生成带框图，但页面要增强：

- 图片支持点击放大。
- 每张图显示：视角编号、缺陷数、最大等效直径、最大面积、主要区域。
- 框颜色图例：一般区域、牙孔区域、焊接区域、重点区域。
- 允许切换：原图 / 识别图 / mask/区域图（如果后端能提供）。

#### C. 区域统计卡片

根据 `visual_summary` 渲染四类区域卡片：

- 一般区域
- 牙孔区域
- 焊接区域
- 重点区域

每张卡片显示：

- 气孔数量
- 总面积
- 最大面积
- 最大等效直径
- 风险提示：高影响区域优先关注

#### D. 缺陷明细表

后端从 `bundle.defect_meta_by_view` 整理 `defect_table`，前端表格渲染：

| 视角 | 缺陷编号 | 区域 | bbox | 面积mm² | 等效直径mm | 置信度 |
|---|---|---|---|---|---|---|

这个表格对论文答辩很重要，因为它证明答案来自可追溯的视觉证据，而不是 LLM 自由发挥。

#### E. Prompt 证据块

增加“Prompt 预览”折叠区，显示：

- 视觉事实块
- 压铸参数摘要
- 问题类型提示
- 图谱辅助证据
- 最终 user prompt 下载链接

不要默认展开全部，避免页面太长。

### 4.5 右侧结果区怎么渲染

每个问题单独渲染一张答案卡片。

卡片结构：

```text
Q01 当前样本应优先从高速充型链还是压力建立链判断？
标签：open_diagnosis / graph: off / focus: 牙孔区域, 焊接区域

【结论】...
【溯源分析】...
【建议方向】...

按钮：复制答案 / 查看Prompt / 下载答案 / 展开证据
```

答案卡片下面增加“教师审阅友好”的结构化标签：

- 关注区域：`focus_regions`
- 候选机理：`mechanism_candidates`
- 是否使用图谱：`graph_debug.enabled`
- 输出格式校验：是否包含【结论】【溯源分析】【建议方向】

### 4.6 历史区怎么渲染

当前历史只能下载，不够。应改为：

- 点击历史项可加载详情。
- 支持按样本编号、日期、图谱开关过滤。
- 每条历史显示：样本编号、图片数、问题数、完成时间、摘要、下载按钮。
- 增加“对比两个运行”的入口，后续可做消融对比。

## 5. 后端 API 建议

保留现有接口，同时增加：

### 5.1 获取运行结果

```http
GET /api/runs/{run_id}/result
```

返回 `result/svp_text_result.json`。

### 5.2 获取运行产物列表

```http
GET /api/runs/{run_id}/artifacts
```

返回可下载文件列表：answer、answers.md、visual_summary.json、visual_fact_block.txt、prompts、annotated_images、zip。

### 5.3 参数校验

```http
POST /api/params/validate
```

输入参数 JSON，返回缺失字段、非数字字段、建议补充字段。此接口不做机理判断，只做格式和完整性检查。

### 5.4 SSE 进度流，可替代轮询

```http
GET /api/jobs/{job_id}/events
```

当前 1.5 秒轮询可以保留，但更好的展示方式是 SSE。若时间紧张，可以先保留轮询。

### 5.5 图谱调试信息单独接口

不要在主答案里混入太多图谱 debug。建议：

```http
GET /api/runs/{run_id}/graph-debug?question_index=1
```

用于论文调试展示，不默认显示给普通用户。

## 6. 论文系统应该体现的功能模块

建议在论文“系统设计与实现”章节按以下模块写：

1. 数据输入模块
   - 多视角探伤图上传
   - 压铸工艺参数录入
   - 溯源问题输入

2. 视觉识别模块
   - 基于已有视觉层模型识别气孔
   - 结合模板/mask 进行区域归属
   - 输出缺陷位置、数量、面积、等效直径

3. 视觉语义化模块
   - 将视觉统计转为结构化自然语言事实块
   - 作为 SVP-Text 的核心输入

4. 图谱辅助模块
   - 可选启用
   - 只辅助机理判断，不覆盖当前样本事实

5. LLM 溯源生成模块
   - 基于视觉事实、工艺参数、问题类型提示生成答案
   - 固定输出结构：结论、溯源分析、建议方向

6. 证据链可视化模块
   - 带框图
   - 区域统计卡片
   - 缺陷明细表
   - prompt 证据预览

7. 结果管理模块
   - 多问题答案
   - 运行目录
   - 历史记录
   - 一键下载

## 7. 文件级修改清单

### `trace_service.py`

必须改：

- 增加 `sanitize_run_tag`。
- 修复 `_make_run_dir`。
- 修复 `resolve_run_file`。
- 增加 GPU 锁。
- 记录每个阶段耗时，生成 `pipeline`。
- 从 `bundle` 中抽取 `region_cards` 和 `defect_table`。
- 结果 JSON 改成面向页面渲染的结构。
- `_append_history` 保存更完整摘要：图谱开关、状态、耗时、样本编号。

建议新增函数：

```python
def build_region_cards(summary: dict) -> list[dict]: ...
def build_defect_table(bundle: dict) -> list[dict]: ...
def validate_answer_format(answer: str) -> dict: ...
def make_artifact_urls(run_dir: Path, index: int) -> dict: ...
```

### `llm_service.py`

必须改：

- `generate` 增加 `max_seq_length` 入参使用配置值。
- 对 `generate` 加锁，避免同一个模型并发 generate。
- 返回或记录 token 统计。
- 不要默认左截断关键证据；如果保留左截断，必须在结果中明确 `was_truncated=true`。

### `prompting.py`

建议改：

- 将视觉事实块、参数块、图谱块分别限制长度。
- Prompt 中更明确地区分“当前样本事实”和“辅助知识”。
- 增加 `build_prompt_sections(...)`，返回结构化 sections，前端可渲染 prompt 预览。

### `main.py`

建议改：

- 增加 `/api/runs/{run_id}/result`。
- 增加 `/api/runs/{run_id}/artifacts`。
- 增加 `/api/params/validate`。
- 上传图片做数量、后缀、大小限制。
- HTTPException 不要全部返回 400，区分 404、422、500。

### `static/index.html`

建议改：

- 增加方法流程时间线容器：`pipelineTimeline`。
- 增加区域统计容器：`regionCards`。
- 增加缺陷明细表容器：`defectTable`。
- 增加 Prompt 预览折叠区：`promptPreview`。
- 增加历史详情加载区。
- 增加图片放大 modal。

### `static/app.js`

建议改：

- 增加统一 `apiFetch`，集中处理错误。
- 增加 `renderPipeline(result.pipeline)`。
- 增加 `renderRegionCards(result.visual.region_cards)`。
- 增加 `renderDefectTable(result.visual.defect_table)`。
- 增加 `renderPromptPreview(answer)`。
- 增加 `copyAnswer`、`copyVisualText`、`loadRunResult`。
- 参数 JSON 和表单合并，避免丢失额外字段。
- 图片预览支持删除和重排。

### `static/styles.css`

建议改：

- 增加时间线样式。
- 增加区域统计卡片样式。
- 增加表格样式。
- 增加答案卡片标签样式。
- 增加图片 modal 样式。
- 响应式布局保留。

## 8. 建议的前端渲染函数

Codex 可以在 `app.js` 中增加：

```javascript
function renderPipeline(items = []) {
  const root = $("pipelineTimeline");
  root.innerHTML = "";
  if (!items.length) {
    root.innerHTML = `<div class="empty-state">暂无流程信息。</div>`;
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = `timeline-item ${escapeHtml(item.status || "")}`;
    row.innerHTML = `
      <span class="dot"></span>
      <div>
        <strong>${escapeHtml(item.name || item.key || "")}</strong>
        <p>${escapeHtml(item.message || "")}</p>
        <em>${item.duration_ms ? `${item.duration_ms} ms` : ""}</em>
      </div>
    `;
    root.appendChild(row);
  }
}

function renderRegionCards(cards = []) {
  const root = $("regionCards");
  root.innerHTML = "";
  if (!cards.length) {
    root.innerHTML = `<div class="empty-state">暂无区域统计。</div>`;
    return;
  }
  for (const card of cards) {
    const el = document.createElement("article");
    el.className = `region-card ${card.high_impact ? "high-impact" : ""}`;
    el.innerHTML = `
      <h3>${escapeHtml(card.region || "未知区域")}</h3>
      <div class="metric-row"><span>气孔数量</span><strong>${card.count ?? 0}</strong></div>
      <div class="metric-row"><span>最大面积</span><strong>${card.max_area_mm2 ?? "-"} mm²</strong></div>
      <div class="metric-row"><span>最大等效直径</span><strong>${card.max_eq_diameter_mm ?? "-"} mm</strong></div>
      <p>${escapeHtml(card.note || "")}</p>
    `;
    root.appendChild(el);
  }
}

function renderDefectTable(rows = []) {
  const root = $("defectTable");
  if (!rows.length) {
    root.innerHTML = `<div class="empty-state">暂无缺陷明细。</div>`;
    return;
  }
  root.innerHTML = `
    <table class="data-table">
      <thead>
        <tr><th>视角</th><th>编号</th><th>区域</th><th>面积mm²</th><th>等效直径mm</th><th>置信度</th></tr>
      </thead>
      <tbody>${rows.map((r) => `
        <tr>
          <td>${escapeHtml(r.view_index ?? "")}</td>
          <td>${escapeHtml(r.defect_id ?? "")}</td>
          <td>${escapeHtml(r.region_type ?? "")}</td>
          <td>${escapeHtml(r.area_mm2 ?? "")}</td>
          <td>${escapeHtml(r.eq_diameter_mm ?? "")}</td>
          <td>${escapeHtml(r.confidence ?? "")}</td>
        </tr>`).join("")}</tbody>
    </table>
  `;
}
```

## 9. 验收标准

### 功能验收

- 可以上传多张图片并完成一次溯源。
- 任务进度能显示完整阶段：queued、vision、visual_summary、graph、llm、packaging、completed。
- 页面能显示：带框图、区域统计卡片、缺陷明细表、视觉文字总结、多问题答案、prompt 预览、下载链接。
- 历史记录可点击查看详情，不只是下载。
- 图谱开关打开或关闭时，结果中能明确显示 graph used/skipped。
- 运行结果 zip 中包含：`svp_text_result.json`、`answers.md`、`visual_summary.json`、`visual_fact_block.txt`、`prompts/`、`answers/`、`annotated_images/`、`request_config.json`。

### 安全和稳定性验收

- 样本编号包含特殊字符时不会造成目录穿越或下载路径错误。
- 多次连续点击提交不会并发跑爆 GPU。
- 大文件、非法后缀、空问题、非法 JSON 能被前后端拦截。
- LLM prompt 不会静默截断关键视觉事实和参数。
- 服务重启后，历史结果仍可通过 `/api/runs/{run_id}/result` 查看。

### 论文展示验收

- 答辩演示时，老师能看到“视觉证据 → 结构化文本 → prompt → LLM 输出”的完整链条。
- 系统能证明 SVP-Text 的核心创新点：不是直接把图片塞给多模态 LLM，而是把视觉层结果转成可控、可审计、可复现的文本证据。
- 页面上有足够的中间产物支撑方法解释，而不是只有最终答案。

## 10. 可直接给 Codex 的任务描述

请基于当前 `SVP-Text_多模态溯源系统` 项目做一次系统级优化，不改变主方法 SVP-Text 的算法链路。优化目标是把它从“可运行 demo”升级为“硕士论文可展示的证据链溯源系统”。

重点要求：

1. 后端安全与稳定性
   - 修复 `sample_id` 拼接运行目录的安全问题，增加运行 tag 清洗。
   - 修复 `resolve_run_file` 的路径检查，使用 `Path.relative_to`。
   - 增加 GPU 推理全局锁或单任务队列，避免多个任务并发加载视觉模型和 LLM。
   - 上传图片增加数量、大小、后缀校验。
   - 增加 `/api/runs/{run_id}/result` 和 `/api/runs/{run_id}/artifacts`。

2. LLM 和 prompt
   - `max_seq_length` 从配置读取，不要硬编码 2048。
   - 记录 prompt 字符数、token 数、是否截断。
   - 保证视觉事实、压铸参数和问题不会被图谱证据挤掉。
   - 对每个答案做格式校验，输出是否包含【结论】【溯源分析】【建议方向】。

3. 结果结构
   - 在 `svp_text_result.json` 中增加 `pipeline`、`visual.region_cards`、`visual.defect_table`、`answers[].evidence`、`answers[].artifacts`。
   - 保留原始字段以兼容旧页面，但新页面优先使用结构化字段。

4. 前端页面
   - 保留原生 HTML/CSS/JS，不强制引入大型框架。
   - 新增方法流程时间线、区域统计卡片、缺陷明细表、Prompt 预览折叠区、答案证据标签、历史详情加载。
   - 图片上传支持删除和顺序展示。
   - 参数表单和 JSON 模式不能丢失额外字段。
   - 历史记录支持点击加载详情，不能只提供下载。

5. 视觉渲染
   - 视觉识别图支持点击放大。
   - 显示颜色图例：一般区域、牙孔区域、焊接区域、重点区域。
   - 每张图显示缺陷数、主要区域、最大面积、最大等效直径。

6. 验收
   - 现有启动方式 `bash start_server.sh` 不要破坏。
   - 原有接口 `/api/trace/start`、`/api/jobs/{job_id}`、`/api/history`、`/api/runs/{run_id}/download` 保持可用。
   - Python 代码能通过 `python -m py_compile run.py run_server.py svp_text_app/*.py`。
   - 前端无明显控制台错误。
   - 生成结果 zip 包内容完整。
