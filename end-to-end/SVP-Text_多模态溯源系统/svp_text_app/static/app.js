const paramFields = [
  ["慢速速度", 0.19],
  ["高速速度", 3.65],
  ["料饼厚度", 28],
  ["铸造压力", 926],
  ["循环周期", 138.1],
  ["建压时间", 114],
  ["射料结束位置", 748],
  ["压射压力", 316],
  ["高速行程", 205],
  ["高速时间", 46],
  ["慢速行程", 525],
  ["环境温度", 23.4],
  ["环境湿度", 47.1],
  ["探伤图编号", 1],
  ["样本编号", 1],
];

const questionTemplates = {
  总体风险判断: "当前样本的气孔风险应优先从哪条工艺链路判断？请结合区域统计、参数链和排查优先级说明。",
  牙孔区域异常分析: "牙孔区域气孔数量或极值突出时，可能对应哪些压铸参数链异常？",
  焊接区域异常分析: "焊接区域出现气孔聚集时，应如何结合视觉证据和工艺参数判断主要风险来源？",
  参数链优先级判断: "当前样本应优先排查高速充型链、压力建立链还是料饼/循环链？请给出排序依据。",
  同工况验证计划: "针对当前样本，应如何制定同料号、同视角、同工况的后续验证计划？",
};

const state = {
  jsonMode: false,
  pollTimer: null,
  currentJobId: "",
  selectedFiles: [],
  extraParams: {},
  historyItems: [],
  lastResult: null,
  activePage: "analysis",
};

const $ = (id) => document.getElementById(id);
const paramFieldNames = new Set(paramFields.map(([name]) => name));

function setActivePage(page) {
  state.activePage = page;
  document.querySelectorAll(".page-view").forEach((view) => {
    view.classList.toggle("active", view.id === `page-${page}`);
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === page);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fileName(path) {
  if (!path) return "";
  return String(path).replaceAll("\\", "/").split("/").filter(Boolean).pop() || String(path);
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败：${response.status}`);
  }
  return payload;
}

function coerceValue(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return "";
  const number = Number(text);
  return Number.isFinite(number) ? number : text;
}

function buildParamGrid(withDefaults = false) {
  const grid = $("paramsGrid");
  grid.innerHTML = "";
  for (const [name, value] of paramFields) {
    const label = document.createElement("label");
    label.className = "param-field";
    label.innerHTML = `<span>${escapeHtml(name)}</span><input data-param="${escapeHtml(name)}" type="text" />`;
    const input = label.querySelector("input");
    if (withDefaults) input.value = value;
    grid.appendChild(label);
  }
  syncJsonFromGrid();
}

function readParamsFromGrid() {
  const params = { ...state.extraParams };
  document.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    const value = coerceValue(input.value);
    if (value !== "") params[key] = value;
    else delete params[key];
  });
  const sampleId = $("sampleId").value.trim();
  const imageNo = $("imageNo").value.trim();
  const batchNo = $("batchNo").value.trim();
  if (sampleId) params["样本编号"] = sampleId;
  if (imageNo) params["探伤图编号"] = imageNo;
  if (batchNo) params["料号/批次"] = batchNo;
  return params;
}

function rememberExtraParams(params) {
  state.extraParams = {};
  for (const [key, value] of Object.entries(params || {})) {
    if (!paramFieldNames.has(key) && !["料号/批次"].includes(key)) {
      state.extraParams[key] = value;
    }
  }
}

function syncJsonFromGrid() {
  const params = readParamsFromGrid();
  $("paramsJson").value = JSON.stringify(params, null, 2);
  validateParamsClient(params);
}

function syncGridFromJson() {
  const params = JSON.parse($("paramsJson").value.trim() || "{}");
  rememberExtraParams(params);
  document.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    input.value = params[key] ?? "";
  });
  $("sampleId").value = params["样本编号"] ?? $("sampleId").value;
  $("imageNo").value = params["探伤图编号"] ?? $("imageNo").value;
  $("batchNo").value = params["料号/批次"] ?? $("batchNo").value;
  validateParamsClient(params);
}

function validateParamsClient(params) {
  const required = ["慢速速度", "高速速度", "料饼厚度", "铸造压力", "建压时间", "射料结束位置"];
  const missing = required.filter((key) => params[key] === undefined || params[key] === "");
  const nonNumeric = Object.entries(params)
    .filter(([, value]) => value !== "" && value !== null && Number.isNaN(Number(value)))
    .map(([key]) => key);
  const parts = [];
  parts.push(missing.length ? `缺失：${missing.join("、")}` : "关键字段完整");
  if (nonNumeric.length) parts.push(`非数字字段：${nonNumeric.join("、")}`);
  const extraCount = Object.keys(state.extraParams).length;
  if (extraCount) parts.push(`已保留额外参数 ${extraCount} 项`);
  $("paramValidation").textContent = parts.join("；");
}

function setProgress(progress, stage, message) {
  const pct = Math.max(0, Math.min(100, Number(progress) || 0));
  $("progressFill").style.width = `${pct}%`;
  $("progressPercent").textContent = `${pct}%`;
  $("progressStage").textContent = stage || "等待输入";
  $("progressMessage").textContent = message || "";
}

function renderPreviews() {
  const grid = $("previewGrid");
  grid.innerHTML = "";
  if (!state.selectedFiles.length) return;
  state.selectedFiles.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "preview-item";
    item.innerHTML = `
      <img src="${URL.createObjectURL(file)}" alt="${escapeHtml(file.name)}" />
      <span>${index + 1}. ${escapeHtml(file.name)}</span>
      <em>${(file.size / 1024 / 1024).toFixed(2)} MB</em>
      <div class="preview-actions">
        <button type="button" data-move-up="${index}" ${index === 0 ? "disabled" : ""}>↑</button>
        <button type="button" data-move-down="${index}" ${index === state.selectedFiles.length - 1 ? "disabled" : ""}>↓</button>
        <button type="button" data-remove-file="${index}">×</button>
      </div>
    `;
    grid.appendChild(item);
  });
}

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
        <em>${item.duration_ms !== undefined ? `${escapeHtml(item.duration_ms)} ms` : ""}</em>
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
      <div class="metric-row"><span>气孔数量</span><strong>${escapeHtml(card.count ?? 0)}</strong></div>
      <div class="metric-row"><span>总面积</span><strong>${escapeHtml(card.total_area_mm2 ?? "-")} mm²</strong></div>
      <div class="metric-row"><span>最大面积</span><strong>${escapeHtml(card.max_area_mm2 ?? "-")} mm²</strong></div>
      <div class="metric-row"><span>最大等效直径</span><strong>${escapeHtml(card.max_eq_diameter_mm ?? "-")} mm</strong></div>
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
        <tr><th>视角</th><th>编号</th><th>区域</th><th>bbox</th><th>面积mm²</th><th>等效直径mm</th><th>置信度</th></tr>
      </thead>
      <tbody>${rows.map((r) => `
        <tr>
          <td>${escapeHtml(r.view_index)}</td>
          <td>${escapeHtml(r.defect_id)}</td>
          <td>${escapeHtml(r.region_type)}</td>
          <td>${escapeHtml(Array.isArray(r.bbox) ? r.bbox.join(", ") : r.bbox || "")}</td>
          <td>${escapeHtml(r.area_mm2 ?? "")}</td>
          <td>${escapeHtml(r.eq_diameter_mm ?? "")}</td>
          <td>${escapeHtml(r.confidence ?? "")}</td>
        </tr>`).join("")}</tbody>
    </table>
  `;
}

function renderParamsTable(paramsPayload) {
  const root = $("paramsTable");
  if (!root) return;
  const rawParams = paramsPayload?.raw || (
    paramsPayload && !Array.isArray(paramsPayload) && !paramsPayload.table && !paramsPayload.validation
      ? paramsPayload
      : {}
  );
  const rows = Array.isArray(paramsPayload?.table)
    ? paramsPayload.table
    : Object.entries(rawParams).map(([name, value]) => ({ name, display: value, is_numeric: !Number.isNaN(Number(value)) }));
  if (!rows.length) {
    root.innerHTML = `<div class="empty-state">暂无参数。</div>`;
    return;
  }
  root.innerHTML = `
    <table class="data-table">
      <thead><tr><th>参数</th><th>值</th><th>类型</th></tr></thead>
      <tbody>${rows.map((row) => `
        <tr>
          <td>${escapeHtml(row.name)}</td>
          <td>${escapeHtml(row.display ?? row.value ?? "")}</td>
          <td>${row.is_numeric ? "数值" : "文本"}</td>
        </tr>`).join("")}</tbody>
    </table>
  `;
}

function renderPromptPreview(item) {
  const root = $("promptPreview");
  if (!item) {
    root.innerHTML = `<div class="empty-state">选择答案卡片中的 Prompt 可查看。</div>`;
    return;
  }
  const sections = item.prompt_sections || {};
  root.innerHTML = `
    <div class="prompt-meta">
      <span>${escapeHtml(item.prompt_token_count ?? "-")} tokens</span>
      <span>${escapeHtml(item.max_seq_length ?? "-")} max</span>
      <span>${item.was_truncated ? "已截断" : "未截断"}</span>
    </div>
    <h3>视觉事实块</h3><pre>${escapeHtml(sections.visual_fact_block || "")}</pre>
    <h3>压铸参数摘要</h3><pre>${escapeHtml(sections.params_block || "")}</pre>
    <h3>问题类型提示</h3><pre>${escapeHtml(sections.question_type_hint || "")}</pre>
    <h3>图谱辅助证据</h3><pre>${escapeHtml(sections.graph_context || "未使用")}</pre>
  `;
}

async function renderArtifacts(result) {
  const summary = $("downloadSummary");
  const list = $("artifactList");
  if (!summary || !list) return;
  list.innerHTML = "";
  if (!result || !result.run_id) {
    summary.innerHTML = `<div class="empty-state">完成一次任务后可下载本次输出包。</div>`;
    return;
  }
  summary.innerHTML = `
    <div class="download-card">
      <strong>${escapeHtml(result.sample_id || result.run_id)}</strong>
      <span>${escapeHtml(result.created_at || "")}</span>
      <a class="download-btn" href="${escapeHtml(result.download_url || "#")}">下载 zip 输出包</a>
    </div>
  `;
  try {
    const payload = await apiFetch(`/api/runs/${encodeURIComponent(result.run_id)}/artifacts`);
    const items = payload.items || [];
    list.innerHTML = items.length ? items.map((item) => `
      <a class="artifact-item" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">
        <strong>${escapeHtml(item.name)}</strong>
        <span>${escapeHtml(item.path)}</span>
      </a>
    `).join("") : `<div class="empty-state">暂无产物列表。</div>`;
  } catch (err) {
    list.innerHTML = `<div class="empty-state">产物列表读取失败：${escapeHtml(err.message)}</div>`;
  }
}

function renderAnswers(results = []) {
  const list = $("answersList");
  list.innerHTML = "";
  if (!results.length) {
    list.innerHTML = `<div class="empty-state">暂未生成答案。</div>`;
    return;
  }
  for (const item of results) {
    const format = item.format_check || {};
    const evidence = item.evidence || {};
    const artifacts = item.artifacts || {};
    const card = document.createElement("article");
    card.className = "answer-card";
    card.innerHTML = `
      <div class="answer-head">
        <span>Q${String(item.index).padStart(2, "0")}</span>
        <strong>${escapeHtml(item.question || "")}</strong>
      </div>
      <div class="tag-row">
        <span>${escapeHtml(item.question_type || "open_diagnosis")}</span>
        <span>graph: ${evidence.graph_used ? "on" : "off"}</span>
        <span>${format.valid ? "格式完整" : "格式待复核"}</span>
      </div>
      <pre>${escapeHtml(item.answer || "")}</pre>
      <div class="answer-actions">
        <button class="ghost-btn" type="button" data-copy-answer="${item.index}">复制答案</button>
        <button class="ghost-btn" type="button" data-show-prompt="${item.index}">查看 Prompt</button>
        ${artifacts.prompt_url ? `<a class="ghost-link" href="${artifacts.prompt_url}" target="_blank" rel="noreferrer">Prompt 文件</a>` : ""}
        ${artifacts.answer_url ? `<a class="ghost-link" href="${artifacts.answer_url}" target="_blank" rel="noreferrer">答案文件</a>` : ""}
      </div>
      <details class="answer-evidence">
        <summary>证据标签</summary>
        <p>关注区域：${escapeHtml((evidence.focus_regions || []).join("、") || "未指定")}</p>
        <p>候选机理：${escapeHtml((evidence.mechanism_candidates || []).join("、") || "未指定")}</p>
      </details>
    `;
    list.appendChild(card);
  }
}

function renderVisualGallery(items = []) {
  const gallery = $("visualGallery");
  gallery.innerHTML = "";
  $("visualCount").textContent = items.length ? `${items.length} 张` : "暂无";
  if (!items.length) {
    gallery.innerHTML = `<div class="empty-state">暂无视觉识别图。</div>`;
    return;
  }
  for (const item of items) {
    const card = document.createElement("figure");
    card.className = "visual-item";
    card.innerHTML = `
      <button type="button" class="image-open" data-image-url="${escapeHtml(item.url)}" data-caption="${escapeHtml(item.image_name || "")}">
        <img src="${item.url}" alt="${escapeHtml(item.image_name || "")}" />
      </button>
      <figcaption>
        <strong>${escapeHtml(item.image_name || `view_${item.view_index}`)}</strong>
        <span>${item.defect_count || 0} 个 / 最大 ${escapeHtml(item.max_eq_diameter_mm ?? "-")} mm</span>
      </figcaption>
      <p>${escapeHtml((item.major_regions || []).join("、") || "无主要区域")}</p>
    `;
    gallery.appendChild(card);
  }
}

function renderResult(result) {
  state.lastResult = result || {};
  const visual = result.visual || {};
  const answers = result.answers || result.results || [];
  $("visualText").textContent = visual.fact_block || result.visual_fact_block || "暂无";
  renderPipeline(result.pipeline || []);
  renderRegionCards(visual.region_cards || []);
  renderDefectTable(visual.defect_table || []);
  renderParamsTable(result.params || { raw: result.raw_params || result.params || {} });
  renderAnswers(answers);
  renderVisualGallery(visual.gallery || result.visual_gallery || []);
  renderPromptPreview(null);
  renderArtifacts(result);
  const link = $("downloadLink");
  if (result.download_url) {
    link.href = result.download_url;
    link.classList.remove("disabled");
  }
}

async function loadConfig() {
  try {
    const health = await apiFetch("/api/health");
    $("healthStatus").textContent = health.status === "ok" ? "服务在线" : "服务异常";
    $("healthStatus").classList.toggle("ok", health.status === "ok");

    const cfg = await apiFetch("/api/config");
    $("configPath").textContent = cfg.config_name || fileName(cfg.config_json);
    $("visionPath").textContent = cfg.visual_model_name || fileName(cfg.visual_model_path);
    $("llmPath").textContent = cfg.llm_base_name || fileName(cfg.llm_base_path);
    $("adapterPath").textContent = cfg.llm_adapter_name || fileName(cfg.llm_adapter_path);
    $("useGraph").checked = Boolean(cfg.use_graph_rag_default);
    $("maxNewTokens").placeholder = String(cfg.max_new_tokens || 1280);
    $("maxSeqLength").placeholder = String(cfg.max_seq_length || 4096);
    $("graphTopK").placeholder = String(cfg.graph_top_k || 10);
  } catch (err) {
    $("healthStatus").textContent = "服务未连接";
  }
}

function renderHistoryList() {
  const query = $("historySearch").value.trim().toLowerCase();
  const items = state.historyItems.filter((item) => {
    const text = `${item.sample_id || ""} ${item.run_id || ""}`.toLowerCase();
    return !query || text.includes(query);
  });
  $("historyCount").textContent = `${items.length} 项`;
  const list = $("historyList");
  list.innerHTML = "";
  if (!items.length) {
    list.innerHTML = `<div class="empty-state">暂无历史文件。</div>`;
    return;
  }
  for (const item of items) {
    const row = document.createElement("button");
    row.className = "history-item";
    row.type = "button";
    row.dataset.runId = item.run_id;
    row.innerHTML = `
      <strong>${escapeHtml(item.sample_id || item.run_id)}</strong>
      <span>${escapeHtml(item.created_at || "")}</span>
      <em>${item.question_count || 0} 问题 / ${item.image_count || 0} 图片 / graph ${Number(item.use_graph_rag) ? "on" : "off"}</em>
    `;
    list.appendChild(row);
  }
}

async function loadHistory() {
  try {
    const payload = await apiFetch("/api/history?days=7");
    state.historyItems = payload.items || [];
    renderHistoryList();
  } catch (err) {
    $("historyList").innerHTML = `<div class="empty-state">历史读取失败。</div>`;
  }
}

async function loadRunResult(runId) {
  const result = await apiFetch(`/api/runs/${encodeURIComponent(runId)}/result`);
  renderResult(result);
  $("historyDetail").innerHTML = `
    <strong>${escapeHtml(result.sample_id || result.run_id)}</strong>
    <span>${escapeHtml(result.created_at || "")}</span>
    <a class="ghost-link" href="${escapeHtml(result.download_url || "#")}">下载历史输出</a>
  `;
  setActivePage("results");
}

async function pollJob(jobId) {
  try {
    const payload = await apiFetch(`/api/jobs/${jobId}`);
    setProgress(payload.progress, payload.stage, payload.message);
    if (payload.status === "queued") {
      $("progressMessage").textContent = payload.message || "等待 GPU 队列。";
    }
    if (payload.status === "completed") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      $("runBtn").disabled = false;
      $("runBtn").textContent = "开始溯源分析";
      renderResult(payload.result || {});
      await loadHistory();
      setActivePage("results");
    } else if (payload.status === "failed") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      $("runBtn").disabled = false;
      $("runBtn").textContent = "开始溯源分析";
      setProgress(payload.progress || 0, "任务失败", payload.error || "任务执行失败。");
    }
  } catch (err) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    $("runBtn").disabled = false;
    $("runBtn").textContent = "开始溯源分析";
    setProgress(0, "连接失败", err.message);
  }
}

function runtimeOption(id) {
  const text = $(id).value.trim();
  return text ? text : $(id).placeholder;
}

async function runTrace() {
  if (!state.selectedFiles.length) {
    alert("请先上传图片。");
    return;
  }

  let params = {};
  let paramsText = "";
  try {
    if (state.jsonMode) syncGridFromJson();
    else syncJsonFromGrid();
    paramsText = $("paramsJson").value.trim() || "{}";
    params = JSON.parse(paramsText);
  } catch (err) {
    alert(`参数 JSON 无法解析：${err.message}`);
    return;
  }

  validateParamsClient(params);
  const questions = $("questions").value.trim();
  if (!questions) {
    alert("请输入至少一个溯源问题。");
    return;
  }

  const form = new FormData();
  state.selectedFiles.forEach((file) => form.append("images", file));
  form.append("sample_id", $("sampleId").value.trim());
  form.append("params_json", paramsText);
  form.append("questions", questions);
  form.append("use_graph_rag", $("useGraph").checked ? "true" : "false");
  form.append("max_new_tokens", runtimeOption("maxNewTokens"));
  form.append("max_seq_length", runtimeOption("maxSeqLength"));
  form.append("graph_top_k", runtimeOption("graphTopK"));

  $("runBtn").disabled = true;
  $("runBtn").textContent = "任务运行中";
  $("downloadLink").classList.add("disabled");
  $("downloadLink").href = "#";
  renderAnswers([]);
  renderVisualGallery([]);
  renderPipeline([]);
  renderRegionCards([]);
  renderDefectTable([]);
  $("visualText").textContent = "等待视觉层输出...";
  setProgress(2, "提交任务", "正在上传图片和创建溯源任务。");
  setActivePage("results");

  try {
    const payload = await apiFetch("/api/trace/start", { method: "POST", body: form });
    state.currentJobId = payload.job_id;
    setProgress(payload.progress, payload.stage, payload.message);
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(() => pollJob(state.currentJobId), 1500);
    await pollJob(state.currentJobId);
  } catch (err) {
    $("runBtn").disabled = false;
    $("runBtn").textContent = "开始溯源分析";
    setProgress(0, "任务创建失败", err.message);
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => setActivePage(button.dataset.page));
  });
  $("imageInput").addEventListener("change", (event) => {
    state.selectedFiles = Array.from(event.target.files || []);
    renderPreviews();
  });
  $("previewGrid").addEventListener("click", (event) => {
    const remove = event.target.dataset.removeFile;
    const up = event.target.dataset.moveUp;
    const down = event.target.dataset.moveDown;
    if (remove !== undefined) state.selectedFiles.splice(Number(remove), 1);
    if (up !== undefined) {
      const index = Number(up);
      [state.selectedFiles[index - 1], state.selectedFiles[index]] = [state.selectedFiles[index], state.selectedFiles[index - 1]];
    }
    if (down !== undefined) {
      const index = Number(down);
      [state.selectedFiles[index + 1], state.selectedFiles[index]] = [state.selectedFiles[index], state.selectedFiles[index + 1]];
    }
    renderPreviews();
  });
  $("paramsGrid").addEventListener("input", syncJsonFromGrid);
  ["sampleId", "imageNo", "batchNo"].forEach((id) => $(id).addEventListener("input", syncJsonFromGrid));
  $("paramsJson").addEventListener("input", () => {
    try {
      validateParamsClient(JSON.parse($("paramsJson").value.trim() || "{}"));
    } catch {
      $("paramValidation").textContent = "JSON 暂时无法解析。";
    }
  });
  $("refreshHistoryBtn").addEventListener("click", loadHistory);
  $("historySearch").addEventListener("input", renderHistoryList);

  $("loadDefaultsBtn").addEventListener("click", () => {
    buildParamGrid(true);
    $("sampleId").value = "1";
    $("imageNo").value = "1";
    $("questions").value = [
      questionTemplates["参数链优先级判断"],
      questionTemplates["同工况验证计划"],
    ].join("\n");
    syncJsonFromGrid();
  });

  $("toggleJsonBtn").addEventListener("click", () => {
    state.jsonMode = !state.jsonMode;
    if (state.jsonMode) {
      syncJsonFromGrid();
      $("paramsGrid").classList.add("hidden");
      $("paramsJson").classList.remove("hidden");
      $("toggleJsonBtn").textContent = "表单";
    } else {
      try {
        syncGridFromJson();
      } catch (err) {
        alert(`参数 JSON 无法解析：${err.message}`);
        state.jsonMode = true;
        return;
      }
      $("paramsGrid").classList.remove("hidden");
      $("paramsJson").classList.add("hidden");
      $("toggleJsonBtn").textContent = "JSON";
    }
  });

  document.querySelectorAll(".question-template").forEach((button) => {
    button.addEventListener("click", () => {
      const text = questionTemplates[button.textContent.trim()];
      const current = $("questions").value.trim();
      $("questions").value = current ? `${current}\n${text}` : text;
    });
  });

  $("runBtn").addEventListener("click", runTrace);
  $("copyVisualBtn").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("visualText").textContent);
  });
  $("answersList").addEventListener("click", async (event) => {
    const copyIndex = event.target.dataset.copyAnswer;
    const promptIndex = event.target.dataset.showPrompt;
    const answers = (state.lastResult && (state.lastResult.answers || state.lastResult.results)) || [];
    if (copyIndex) {
      const item = answers.find((x) => Number(x.index) === Number(copyIndex));
      if (item) await navigator.clipboard.writeText(item.answer || "");
    }
    if (promptIndex) {
      const item = answers.find((x) => Number(x.index) === Number(promptIndex));
      renderPromptPreview(item);
      setActivePage("prompt");
    }
  });
  $("visualGallery").addEventListener("click", (event) => {
    const button = event.target.closest("[data-image-url]");
    if (!button) return;
    $("modalImage").src = button.dataset.imageUrl;
    $("modalCaption").textContent = button.dataset.caption || "";
    $("imageModal").classList.remove("hidden");
  });
  $("modalClose").addEventListener("click", () => $("imageModal").classList.add("hidden"));
  $("historyList").addEventListener("click", async (event) => {
    const row = event.target.closest("[data-run-id]");
    if (!row) return;
    try {
      await loadRunResult(row.dataset.runId);
    } catch (err) {
      $("historyDetail").textContent = err.message;
    }
  });
}

buildParamGrid(false);
bindEvents();
loadConfig();
loadHistory();
