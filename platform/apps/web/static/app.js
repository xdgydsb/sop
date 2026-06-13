const state = {
  catalog: null,
  health: null,
  runtime: null,
  selectedStep: 0,
  poller: null,
  refreshing: false
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const eventNames = {
  box_opened: "盒子由关闭变为打开",
  earphone_in_box: "耳机由盒外进入盒内",
  charger_in_box: "充电器由盒外进入盒内",
  green_bag_in_box: "绿色小袋由盒外进入盒内",
  box_closed: "盒子由打开变为关闭"
};

function toast(message, error = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.className = "toast", 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.message || `请求失败 HTTP ${response.status}`);
  }
  return payload.data;
}

function setView(name) {
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  const item = $(`.nav-item[data-view="${name}"]`);
  $("#pageTitle").textContent = item ? item.querySelector("span").textContent : "";
  if (name === "runtime" || name === "evidence") refreshRuntime();
}

function initNavigation() {
  $("#nav").addEventListener("click", event => {
    const button = event.target.closest("[data-view]");
    if (button) setView(button.dataset.view);
  });
  $$("[data-jump]").forEach(button => button.addEventListener("click", () => setView(button.dataset.jump)));
}

function initClock() {
  const render = () => {
    const now = new Date();
    $("#today").textContent = now.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit", weekday: "short" });
    $("#clock").textContent = now.toLocaleTimeString("zh-CN", { hour12: false });
  };
  render();
  setInterval(render, 1000);
}

async function loadCatalog() {
  state.catalog = await api("/api/platform/catalog");
  renderCatalog();
}

function renderCatalog() {
  const model = state.catalog.models[0];
  $("#labelCloud").innerHTML = model.labels.map(label => `<span>${label}</span>`).join("");
  $("#modelName").textContent = `${model.name} ${model.version}`;
  $("#modelStatus").textContent = model.status === "BASELINE_VERIFIED" ? "稳定基线" : "待验证";
  $("#modelStatus").classList.toggle("online", model.status === "BASELINE_VERIFIED");
  $("#modelVerification").textContent = model.verification
    ? `${model.verification} · ${model.verifiedAt}`
    : "尚无真实工位验证记录";
  const releases = state.catalog.releases || [];
  const deployments = state.catalog.deployments || [];
  const latestRelease = releases.at(-1);
  const latestDeployment = deployments.at(-1);
  $("#releaseMetric").textContent = String(releases.length).padStart(2, "0");
  $("#releaseMetricText").textContent = latestRelease
    ? `${latestRelease.name} v${latestRelease.version}`
    : "尚未发布版本";
  $("#draftVersion").textContent = `${state.catalog.sop.name} 草稿 v${state.catalog.sop.version}`;
  $("#draftStatus").textContent = latestRelease
    ? `最新发布 v${latestRelease.version} · ${
        latestRelease.runtimeCompatibility === "LEGACY_STAGE3"
          ? "兼容当前 stage3"
          : "需要通用 Runtime"
      }`
    : "草稿 · 未发布";
  $("#deploymentVersion").textContent = latestDeployment
    ? `${latestDeployment.releaseId} · ${latestDeployment.status}`
    : "尚未部署发布版本";
  renderFlow();
}

function renderFlow() {
  const sop = state.catalog.sop;
  $("#flowNodes").innerHTML = [
    `<button class="flow-node"><em>START</em><b>开始作业周期</b><small>工位与运行链路自检通过</small></button>`,
    ...sop.steps.map((step, index) => `<button class="flow-node ${state.selectedStep === index ? "selected" : ""}" data-step="${index}"><em>S${index + 1}</em><b>${step.name}</b><small>${step.rule}</small></button>`),
    `<button class="flow-node"><em>END</em><b>完成产品周期</b><small>保存判定结果与动作证据</small></button>`
  ].join("");
  $$("#flowNodes [data-step]").forEach(node => node.addEventListener("click", () => selectStep(Number(node.dataset.step))));
}

function selectStep(index) {
  state.selectedStep = index;
  const step = state.catalog.sop.steps[index];
  $("#inspectorEmpty").hidden = true;
  $("#stepForm").hidden = false;
  $("#stepName").value = step.name;
  $("#stepEvent").value = step.event;
  $("#stepRule").value = step.rule;
  $("#stepHold").value = step.holdMs;
  renderFlow();
}

async function saveSop() {
  try {
    const saved = await api("/api/platform/sop", {
      method: "PUT",
      body: JSON.stringify(state.catalog.sop)
    });
    state.catalog.sop = saved;
    toast("SOP草稿已保存");
  } catch (error) {
    toast(error.message, true);
  }
}

async function publishSop() {
  try {
    const saved = await api("/api/platform/sop", {
      method: "PUT",
      body: JSON.stringify(state.catalog.sop)
    });
    state.catalog.sop = saved;
    const release = await api("/api/platform/sop/releases", { method: "POST" });
    await loadCatalog();
    toast(`SOP v${release.version} 已校验并发布`);
  } catch (error) {
    toast(`发布失败：${error.message}`, true);
  }
}

async function deployLatestRelease() {
  const releases = state.catalog.releases || [];
  const latestRelease = releases.at(-1);
  if (!latestRelease) {
    toast("请先校验并发布一个 SOP 版本", true);
    return;
  }
  try {
    await api("/api/platform/deployments", {
      method: "POST",
      body: JSON.stringify({
        releaseId: latestRelease.releaseId,
        stationId: state.catalog.sop.stationId
      })
    });
    await loadCatalog();
    toast(`${latestRelease.releaseId} 已生成部署配置，等待 Worker 确认下发`);
  } catch (error) {
    toast(`部署失败：${error.message}`, true);
  }
}

function initDesigner() {
  $("#stepForm").addEventListener("submit", event => {
    event.preventDefault();
    const step = state.catalog.sop.steps[state.selectedStep];
    step.name = $("#stepName").value.trim();
    step.rule = $("#stepRule").value.trim();
    step.holdMs = Number($("#stepHold").value);
    renderFlow();
    toast("工序属性已应用到草稿");
  });
  $("#deleteStepButton").addEventListener("click", () => {
    if (state.catalog.sop.steps.length <= 1) {
      toast("SOP 至少保留一个步骤", true);
      return;
    }
    const [removed] = state.catalog.sop.steps.splice(state.selectedStep, 1);
    state.selectedStep = Math.max(0, state.selectedStep - 1);
    $("#stepForm").hidden = true;
    $("#inspectorEmpty").hidden = false;
    renderFlow();
    toast(`已从草稿删除“${removed.name}”，保存后写入数据库`);
  });
  $("#saveSopButton").addEventListener("click", saveSop);
  $("#publishSopButton").addEventListener("click", publishSop);
}

async function refreshHealth() {
  try {
    state.health = await api("/api/platform/health");
    renderHealth();
  } catch (error) {
    toast(error.message, true);
  }
}

function renderHealth() {
  const entries = [
    ["平台 API", state.health.platform],
    ["若依 Runtime", state.health.runtime],
    ["海康工业相机", state.health.camera],
    ["GPU 推理链路", state.health.inference]
  ];
  const allOnline = state.health.platform.online && state.health.runtime.online && state.health.camera.online;
  $("#systemState").textContent = allOnline ? "运行链路在线" : "部分服务未连接";
  $(".pulse").classList.toggle("online", allOnline);
  $("#stationStatus").textContent = allOnline ? "在线" : "待连接";
  $("#stationStatus").classList.toggle("online", allOnline);
  $("#stationMetric").textContent = allOnline ? "运行链路正常" : "部分设备未连接";
  $("#healthList").innerHTML = entries.map(([name, item]) => `
    <div class="health-item"><i class="${item.online ? "online" : ""}"></i><div><strong>${name}</strong><small>${item.message}</small></div><b>${item.online ? "ONLINE" : "OFFLINE"}</b></div>
  `).join("");
  $("#diagnosticList").innerHTML = entries.map(([name, item]) => `
    <div class="diagnostic"><span>${name}</span><strong>${item.online ? "运行正常" : "未连接"}</strong><small>${item.latencyMs != null ? `${item.latencyMs} ms · ` : ""}${item.message}</small></div>
  `).join("");

  const stream = $("#liveStream");
  if (state.health.camera.online) {
    const streamUrl = state.health.camera.streamUrl;
    if (stream.dataset.streamUrl !== streamUrl) {
      stream.src = streamUrl;
      stream.dataset.streamUrl = streamUrl;
    }
    stream.classList.add("visible");
    $("#videoEmpty").hidden = true;
    $(".live-dot").classList.add("online");
  } else {
    stream.removeAttribute("src");
    delete stream.dataset.streamUrl;
    stream.classList.remove("visible");
    $("#videoEmpty").hidden = false;
    $(".live-dot").classList.remove("online");
  }
}

function normalizeRuntime(data) {
  if (!data || !data.task) return null;
  return {
    task: data.task,
    steps: data.steps || [],
    events: data.recentEvents || [],
    alarms: data.recentAlarms || []
  };
}

async function refreshRuntime() {
  try {
    state.runtime = normalizeRuntime(await api("/api/platform/runtime"));
  } catch (error) {
    state.runtime = null;
  }
  renderRuntime();
}

function statusClass(status = "") {
  const value = status.toUpperCase();
  if (["PASS", "PASSED", "COMPLETED"].includes(value)) return "pass";
  if (["RUNNING", "ACTIVE", "IN_PROGRESS"].includes(value)) return "running";
  if (["FAIL", "FAILED", "REJECT", "TIMEOUT"].includes(value)) return "fail";
  return "";
}

function renderRuntime() {
  const runtime = state.runtime;
  const configured = state.catalog ? state.catalog.sop.steps : [];
  const steps = runtime && runtime.steps.length ? runtime.steps : configured.map((step, index) => ({
    stepNo: index + 1, stepName: step.name, expectedEvent: step.event, stepStatus: "PENDING", judgeMessage: ""
  }));
  const passed = steps.filter(step => statusClass(step.stepStatus || step.judgeResult) === "pass").length;
  const task = runtime && runtime.task;
  const mode = task ? (task.runtimeMode || task.taskStatus || "READY") : "OFFLINE";
  const expected = task && task.runtimeMessage ? task.runtimeMessage : "等待运行服务";
  const progress = Math.round((passed / Math.max(steps.length, 1)) * 100);

  $("#dashboardMode").textContent = mode;
  $("#dashboardMode").classList.toggle("online", Boolean(task));
  $("#dashboardProgress").style.width = `${progress}%`;
  $("#progressPercent").textContent = `${progress}%`;
  $("#progressText").textContent = task ? `${passed}/${steps.length} 个工序已通过` : "等待Runtime连接";
  $("#heroFps").textContent = task && task.runtimeFps != null ? Number(task.runtimeFps).toFixed(1) : "--";
  $("#runtimeFps").textContent = task && task.runtimeFps != null ? Number(task.runtimeFps).toFixed(1) : "--";
  $("#captureFps").textContent = state.health && state.health.camera.online ? "LIVE" : "--";
  $("#runtimeLatency").textContent = "--";
  $("#liveMessage").textContent = task ? (task.runtimeMessage || "运行状态已同步") : "运行服务未连接";
  $("#overlayMode").textContent = mode;
  $("#overlayExpected").textContent = expected;
  $("#stepCount").textContent = `${passed} / ${steps.length}`;

  $("#runtimeSteps").innerHTML = steps.map((step, index) => {
    const status = step.stepStatus || step.judgeResult || "PENDING";
    const event = step.expectedEvent || step.event || "-";
    return `<div class="runtime-step ${statusClass(status)}"><span class="step-index">S${step.stepNo || index + 1}</span><div><b>${step.stepName || step.name}</b><small>${eventNames[event] || event}</small></div><span class="step-state">${status}</span></div>`;
  }).join("");
  $("#evidenceStrip").innerHTML = steps.map((step, index) => {
    const status = step.stepStatus || step.judgeResult || "PENDING";
    const preview = step.snapshotUrl
      ? `<img src="${step.snapshotUrl}" alt="S${step.stepNo || index + 1}动作证据">`
      : `<div class="evidence-placeholder">S${step.stepNo || index + 1}</div>`;
    return `<div class="evidence-tile ${statusClass(status)}">${preview}<b>S${step.stepNo || index + 1} · ${step.stepName || step.name}</b><span>${step.clipUrl ? "截图与视频已就绪" : status}</span></div>`;
  }).join("");

  const current = steps.find(step => statusClass(step.stepStatus || step.judgeResult) === "running")
    || steps.find(step => !statusClass(step.stepStatus || step.judgeResult));
  $("#judgeTitle").textContent = current ? `当前：${current.stepName || current.name}` : (passed === steps.length ? "作业周期完成" : "等待开始检测");
  $("#judgeReason").textContent = current && current.judgeMessage ? current.judgeMessage : "动作必须满足前置状态、手物交互、状态转移和后置稳定。";
  renderEvidence(steps, runtime);
}

function renderEvidence(steps, runtime) {
  const task = runtime && runtime.task;
  $("#evidenceTask").textContent = task ? task.taskCode : "暂无真实运行任务";
  $("#evidencePassed").textContent = steps.filter(step => statusClass(step.stepStatus || step.judgeResult) === "pass").length;
  $("#evidenceAlarms").textContent = runtime ? runtime.alarms.length : 0;
  $("#evidenceClips").textContent = steps.filter(step => step.clipUrl).length;
  $("#alarmMetric").textContent = runtime ? runtime.alarms.length : "--";
  $("#completedMetric").textContent = task && ["PASSED", "FINISHED"].includes(task.taskStatus) ? "01" : "--";
  $("#evidenceTable").innerHTML = steps.length ? steps.map((step, index) => `
    <tr><td>S${step.stepNo || index + 1} · ${step.stepName || step.name}</td><td>${step.stepStatus || step.judgeResult || "PENDING"}</td><td>${step.judgeMessage || "等待该工序结果"}</td><td>${step.snapshotUrl ? `<a class="clip-link" href="${step.snapshotUrl}" target="_blank">查看截图</a>` : "暂无截图"}${step.clipUrl ? ` · <a class="clip-link" href="${step.clipUrl}" target="_blank">查看动作片段 ↗</a>` : ""}</td></tr>
  `).join("") : `<tr><td colspan="4" class="no-data">暂无任务与证据</td></tr>`;
}

async function control(command) {
  const labels = { start: "开始检测", stop: "停止检测", reset: "重置检测" };
  try {
    const taskCode = state.runtime && state.runtime.task ? state.runtime.task.taskCode : null;
    const data = await api(`/api/platform/runtime/${command}`, {
      method: "POST",
      body: JSON.stringify({ taskCode })
    });
    state.runtime = normalizeRuntime(data);
    renderRuntime();
    toast(`${labels[command]}指令已由真实Runtime接收`);
  } catch (error) {
    toast(`${labels[command]}失败：${error.message}`, true);
  }
}

function initActions() {
  $("#startButton").addEventListener("click", () => control("start"));
  $("#stopButton").addEventListener("click", () => control("stop"));
  $("#resetButton").addEventListener("click", () => control("reset"));
  $("#refreshButton").addEventListener("click", refreshAll);
  $("#stationRefresh").addEventListener("click", refreshAll);
  $("#deployButton").addEventListener("click", deployLatestRelease);
  $("#evidenceRefresh").addEventListener("click", refreshRuntime);
}

async function refreshAll() {
  if (state.refreshing) return;
  state.refreshing = true;
  try {
    await Promise.all([refreshHealth(), refreshRuntime()]);
  } finally {
    state.refreshing = false;
  }
}

async function boot() {
  initNavigation();
  initClock();
  initDesigner();
  initActions();
  try {
    await loadCatalog();
    await refreshAll();
  } catch (error) {
    toast(error.message, true);
  }
  state.poller = setInterval(refreshAll, 4000);
}

window.addEventListener("DOMContentLoaded", boot);
