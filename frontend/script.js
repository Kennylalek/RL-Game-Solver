const API = "http://127.0.0.1:5050/api";

const state = {
  status: null,
  difficulty: "medium",
  sudokuPuzzle: Array(81).fill(0),
  sudokuValidation: null,
  sudokuSteps: [],
  sudokuIndex: 0,
  gameSteps: [],
  gameIndex: 0,
  curves: { sudoku: {}, "2048": [] },
  lastComparison: null,
  apiOnline: null,
  busy: null,
};

const REPLAY_OPTIONS = {
  sudoku: [
    ["rl", "RL Q + MRV"],
    ["mrv", "MRV Heuristic"],
    ["backtracking", "Pure Backtracking"],
    ["random_mrv", "Random MRV"],
  ],
  "2048": [
    ["hybrid", "Hybrid Q + Expectimax"],
    ["q", "RL Q Policy"],
    ["expectimax", "Expectimax"],
    ["random", "Random Policy"],
  ],
};

const SNAKE_WEIGHTS = [
  15, 14, 13, 12,
  8, 9, 10, 11,
  7, 6, 5, 4,
  0, 1, 2, 3,
];

function byId(id) {
  return document.getElementById(id);
}

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function optionalNumber(id) {
  const value = byId(id)?.value;
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function boundedNumber(id, fallback, min, max) {
  const parsed = optionalNumber(id);
  if (parsed === undefined) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function readHyperparameters() {
  return {
    alpha: boundedNumber("hp-alpha", 0.3, 0.001, 1),
    gamma: boundedNumber("hp-gamma", 0.95, 0, 1),
    epsilon: boundedNumber("hp-epsilon", 1, 0, 1),
    eps_decay: boundedNumber("hp-eps-decay", 0.995, 0.5, 1),
    eps_min: boundedNumber("hp-eps-min", 0.05, 0, 1),
  };
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (error) {
    throw new Error(`Cannot reach Flask API at ${API}. Start backend/app.py, then reload or wait for reconnect.`);
  }

  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error(`Backend returned non-JSON response (${response.status}). Check Flask logs.`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function showError(message) {
  const alert = byId("alert");
  alert.textContent = message;
  alert.classList.add("active");
}

function clearError() {
  const alert = byId("alert");
  alert.textContent = "";
  alert.classList.remove("active");
}

async function execute(label, operation) {
  state.busy = label;
  clearError();
  updateButtons();
  try {
    await operation();
    await refreshStatus({ silent: true });
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    state.busy = null;
    updateButtons();
  }
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function pollJob(jobId, onUpdate) {
  while (true) {
    const job = await requestJson(`/jobs/${jobId}`);
    onUpdate(job);
    if (job.status === "complete") {
      return job;
    }
    if (job.status === "error") {
      throw new Error(job.error || job.message || "Live job failed");
    }
    await delay(350);
  }
}

async function executeLiveJob(label, startOperation, onUpdate, onComplete) {
  state.busy = label;
  clearError();
  updateButtons();
  try {
    const started = await startOperation();
    if (!started.job_id) {
      throw new Error("Backend did not return a live job id");
    }
    const finalJob = await pollJob(started.job_id, onUpdate);
    if (onComplete) {
      onComplete(finalJob);
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    state.busy = null;
    await refreshStatus({ silent: true });
    updateButtons();
  }
}

function updateButtons() {
  const offline = state.apiOnline === false;
  const sudokuReady = state.status?.training?.sudoku?.[state.difficulty] === "ready";
  const sudokuAlgorithm = byId("sudoku-algorithm")?.value || "rl";
  const gameReady = state.status?.training?.["2048"] === "ready";
  const gamePolicy = byId("game-policy")?.value || "hybrid";
  const gameNeedsTrainedAgent = gamePolicy === "q" || gamePolicy === "hybrid";

  byId("btn-sudoku-train").disabled = offline || state.busy !== null;
  byId("btn-sudoku-puzzle").disabled = offline || state.busy !== null;
  byId("btn-sudoku-solve").disabled = offline || state.busy !== null || (sudokuAlgorithm === "rl" && !sudokuReady);
  byId("btn-sudoku-save").disabled = offline || state.busy !== null || !sudokuReady;
  byId("btn-sudoku-load").disabled = offline || state.busy !== null;
  byId("btn-sudoku-reset").disabled = offline || state.busy !== null;
  byId("btn-game-train").disabled = offline || state.busy !== null;
  byId("btn-game-play").disabled = offline || state.busy !== null || (gameNeedsTrainedAgent && !gameReady);
  byId("btn-game-save").disabled = offline || state.busy !== null || !gameReady;
  byId("btn-game-load").disabled = offline || state.busy !== null;
  byId("btn-game-reset").disabled = offline || state.busy !== null;
  byId("btn-compare").disabled = offline || state.busy !== null;
  byId("btn-export").disabled = offline || state.busy !== null;
  byId("btn-replay").disabled = offline || state.busy !== null;

  byId("btn-sudoku-train").textContent = state.busy === "sudoku-train" ? "Training live..." : "Train";
  byId("btn-sudoku-solve").textContent = state.busy === "sudoku-solve" ? "Solving live..." : "Solve";
  byId("btn-game-train").textContent = state.busy === "game-train" ? "Training live..." : "Train";
  byId("btn-game-play").textContent = state.busy === "game-play" ? "Playing live..." : "Play";
  byId("btn-compare").textContent = state.busy === "compare" ? "Running..." : "Run comparison";
  byId("btn-export").textContent = state.busy === "export" ? "Exporting..." : "Export artifacts";
  byId("btn-replay").textContent = state.busy === "replay" ? "Running..." : "Run replay";
}

function updateStatusPills() {
  const sudokuStatus = state.status?.training?.sudoku?.[state.difficulty] || "idle";
  const gameStatus = state.status?.training?.["2048"] || "idle";
  const sudokuPill = byId("status-sudoku");
  const gamePill = byId("status-2048");
  if (state.apiOnline === false) {
    sudokuPill.textContent = "Sudoku offline";
    gamePill.textContent = "2048 offline";
    sudokuPill.classList.remove("ready");
    gamePill.classList.remove("ready");
    return;
  }

  sudokuPill.textContent = `Sudoku ${sudokuStatus}`;
  gamePill.textContent = `2048 ${gameStatus}`;
  sudokuPill.classList.toggle("ready", sudokuStatus === "ready");
  gamePill.classList.toggle("ready", gameStatus === "ready");
}

async function refreshStatus(options = {}) {
  const silent = options.silent === true;
  try {
    state.status = await requestJson("/status");
    state.curves = await requestJson("/analysis/training-curves");
    state.apiOnline = true;
    updateStatusPills();
    updateButtons();
    if (state.busy === null) {
      renderCurves();
    }
    if (!silent) {
      clearError();
    }
    return true;
  } catch (error) {
    state.apiOnline = false;
    updateStatusPills();
    updateButtons();
    if (!silent) {
      showError(error instanceof Error ? error.message : String(error));
    }
    return false;
  }
}

function setActivePanel(name) {
  byId("tab-sudoku").classList.toggle("active", name === "sudoku");
  byId("tab-2048").classList.toggle("active", name === "2048");
  byId("panel-sudoku").classList.toggle("active", name === "sudoku");
  byId("panel-2048").classList.toggle("active", name === "2048");
}

function renderSudokuBoard(board, fixed, highlight) {
  const root = byId("sudoku-board");
  root.innerHTML = "";
  for (let idx = 0; idx < 81; idx += 1) {
    const cell = document.createElement("div");
    const value = board[idx] || 0;
    cell.className = "sudoku-cell";
    if (fixed[idx]) {
      cell.classList.add("is-fixed");
    } else if (value) {
      cell.classList.add("is-placed");
    } else {
      cell.classList.add("is-empty");
    }
    if (highlight === idx) {
      cell.classList.add("is-highlighted");
    }
    cell.textContent = value ? String(value) : "";
    root.appendChild(cell);
  }
}

function renderSudokuStep() {
  const step = state.sudokuSteps[state.sudokuIndex];
  const board = step?.board || state.sudokuPuzzle;
  const highlight = step?.action?.idx ?? null;
  renderSudokuBoard(board, state.sudokuPuzzle, highlight);

  byId("s-stat-step").textContent = state.sudokuSteps.length
    ? `${state.sudokuIndex + 1}/${state.sudokuSteps.length}`
    : "0";
  byId("s-stat-filled").textContent = String(board.filter(Boolean).length);
  byId("s-stat-reward").textContent = step?.reward ?? "-";
  byId("s-stat-message").textContent = step?.message || "Ready";

  const range = byId("sudoku-range");
  range.max = String(Math.max(state.sudokuSteps.length - 1, 0));
  range.value = String(state.sudokuIndex);
  renderSudokuExplain(step);
}

function renderSudokuExplain(step) {
  const root = byId("sudoku-explain");
  root.innerHTML = "";
  if (!step?.explain) {
    root.innerHTML = '<div class="empty-state">Run a Sudoku solve to inspect Q-values.</div>';
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "explain-grid";
  if (step.explain.algorithm) {
    wrapper.appendChild(statNode("Algorithm", step.explain.algorithm));
  }
  if (step.explain.duration_ms !== undefined) {
    wrapper.appendChild(statNode("Duration", `${step.explain.duration_ms} ms`));
  }
  const selected = step.explain.cell === null ? "-" : `${step.explain.row}, ${step.explain.col}`;
  wrapper.appendChild(statNode("Feature state", step.explain.state.join(" / ")));
  wrapper.appendChild(statNode("Selected cell", selected));

  const candidates = document.createElement("div");
  candidates.className = "candidate-list";
  step.explain.candidates.forEach((candidate) => {
    const pill = document.createElement("div");
    pill.className = "candidate-pill";
    const value = document.createElement("span");
    value.textContent = String(candidate.value);
    const q = document.createElement("strong");
    q.textContent = Number(candidate.q).toFixed(2);
    pill.append(value, q);
    candidates.appendChild(pill);
  });
  wrapper.appendChild(candidates);
  root.appendChild(wrapper);
}

function renderSudokuValidation() {
  const root = byId("sudoku-validation");
  root.innerHTML = "";
  const validation = state.sudokuValidation;
  if (!validation) {
    root.innerHTML = '<div class="empty-state">Load a puzzle to validate difficulty.</div>';
    return;
  }

  const grid = document.createElement("div");
  grid.className = "explain-grid";
  grid.appendChild(statNode("Empty cells", `${validation.actual_empty_cells}/${validation.target_empty_cells}`));
  grid.appendChild(statNode("Unique solution", validation.unique_solution ? "Yes" : "No"));
  grid.appendChild(statNode("MRV steps", validation.mrv?.steps ?? "-"));
  grid.appendChild(statNode("Backtracking steps", validation.backtracking?.steps ?? "-"));
  grid.appendChild(statNode("MRV time", `${formatNumber(validation.mrv?.duration_ms ?? 0, 2)} ms`));
  grid.appendChild(statNode("Backtracking time", `${formatNumber(validation.backtracking?.duration_ms ?? 0, 2)} ms`));
  root.appendChild(grid);
}

function renderGameBoard(board) {
  const root = byId("game-board");
  root.innerHTML = "";
  board.flat().forEach((value) => {
    const tile = document.createElement("div");
    tile.className = `game-tile tile-${Math.min(value, 2048)}`;
    tile.textContent = value ? String(value) : "";
    root.appendChild(tile);
  });
}

function renderGameStep() {
  const emptyBoard = Array.from({ length: 4 }, () => Array(4).fill(0));
  const step = state.gameSteps[state.gameIndex];
  const board = step?.board || emptyBoard;
  renderGameBoard(board);

  byId("g-stat-step").textContent = state.gameSteps.length
    ? `${state.gameIndex}/${Math.max(state.gameSteps.length - 1, 0)}`
    : "0";
  byId("g-stat-score").textContent = String(step?.score || 0);
  byId("g-stat-max").textContent = String(Math.max(...board.flat(), 0) || "-");
  byId("g-stat-reward").textContent = step?.reward ?? "-";

  const range = byId("game-range");
  range.max = String(Math.max(state.gameSteps.length - 1, 0));
  range.value = String(state.gameIndex);
  renderGameExplain(step);
  renderGameHeatmap(step);
}

function renderGameExplain(step) {
  const root = byId("game-explain");
  root.innerHTML = "";
  if (!step?.explain) {
    root.innerHTML = '<div class="empty-state">Run a 2048 episode to inspect action values.</div>';
    return;
  }

  const table = document.createElement("div");
  table.className = "action-table";
  table.innerHTML = '<div class="action-row action-head"><span>Action</span><span>Q</span><span>Gain</span><span>Heuristic</span></div>';
  step.explain.actions.forEach((action) => {
    const row = document.createElement("div");
    row.className = `action-row${action.valid ? "" : " muted-row"}`;
    row.innerHTML = `<span>${action.name}</span><span>${Number(action.q).toFixed(2)}</span><span>${action.score_gain}</span><span>${action.heuristic === null ? "-" : Number(action.heuristic).toFixed(1)}</span>`;
    table.appendChild(row);
  });
  root.appendChild(table);
}

function renderGameHeatmap(step) {
  const root = byId("game-heatmap");
  root.innerHTML = "";
  const board = step?.board || Array.from({ length: 4 }, () => Array(4).fill(0));
  const flat = board.flat();
  const maxWeighted = Math.max(...flat.map((value, idx) => value * SNAKE_WEIGHTS[idx]), 1);
  const grid = document.createElement("div");
  grid.className = "heatmap-grid";
  flat.forEach((value, idx) => {
    const weight = SNAKE_WEIGHTS[idx];
    const weighted = value * weight;
    const intensity = Math.max(0.08, weighted / maxWeighted);
    const cell = document.createElement("div");
    cell.className = "heatmap-cell";
    cell.style.background = `rgba(37, 99, 235, ${0.08 + intensity * 0.42})`;
    cell.innerHTML = `<strong>${value || "-"}</strong><span>w${weight}</span>`;
    grid.appendChild(cell);
  });
  root.appendChild(grid);
}

function statNode(label, value) {
  const node = document.createElement("div");
  node.className = "stat";
  const valueNode = document.createElement("div");
  valueNode.className = "stat-value";
  valueNode.textContent = String(value);
  const labelNode = document.createElement("div");
  labelNode.className = "stat-label";
  labelNode.textContent = label;
  node.append(valueNode, labelNode);
  return node;
}

async function loadSudokuPuzzle() {
  const difficulty = byId("sudoku-difficulty").value;
  state.difficulty = difficulty;
  const params = new URLSearchParams({ difficulty });
  const seed = optionalNumber("sudoku-seed");
  if (seed !== undefined) {
    params.set("seed", String(seed));
  }
  const payload = await requestJson(`/sudoku/puzzle?${params.toString()}`);
  state.sudokuPuzzle = payload.board;
  state.sudokuValidation = payload.validation || null;
  state.sudokuSteps = [];
  state.sudokuIndex = 0;
  renderSudokuStep();
  renderSudokuValidation();
}

function renderSudokuComparison(rows) {
  const root = byId("sudoku-comparison");
  const table = document.createElement("div");
  table.className = "table";
  table.innerHTML = '<div class="table-row table-head"><span>Method</span><span>Success</span><span>Avg steps</span><span>Avg ms</span></div>';
  rows.forEach((row) => {
    const tr = document.createElement("div");
    tr.className = "table-row";
    tr.innerHTML = `<span>${row.method}</span><span>${formatNumber(row.success_rate * 100, 1)}%</span><span>${formatNumber(row.avg_steps, 1)}</span><span>${formatNumber(row.avg_duration_ms, 1)}</span>`;
    table.appendChild(tr);
  });
  root.replaceChildren(table);
}

function renderGameComparison(rows) {
  const root = byId("game-comparison");
  const table = document.createElement("div");
  table.className = "table";
  table.innerHTML = '<div class="table-row table-head game-summary"><span>Method</span><span>Avg score</span><span>Best</span><span>Avg tile</span><span>Moves</span></div>';
  rows.forEach((row) => {
    const tr = document.createElement("div");
    tr.className = "table-row game-summary";
    tr.innerHTML = `<span>${row.method}</span><span>${formatNumber(row.avg_score, 0)}</span><span>${row.best_score}</span><span>${formatNumber(row.avg_max_tile, 0)}</span><span>${formatNumber(row.avg_moves, 1)}</span>`;
    table.appendChild(tr);
  });
  root.replaceChildren(table);
}

function renderDashboard(comparison) {
  const root = byId("results-dashboard");
  if (!comparison) {
    root.innerHTML = '<div class="empty-state">Run comparison to summarize the strongest methods.</div>';
    return;
  }

  const bestSudoku = comparison.sudoku.summary[0];
  const bestGame = comparison.game2048.summary[0];
  const cards = [
    ["Best Sudoku", bestSudoku?.method || "-", `${formatNumber((bestSudoku?.success_rate || 0) * 100, 1)}% solved`],
    ["Sudoku effort", bestSudoku ? `${formatNumber(bestSudoku.avg_steps, 1)} avg steps` : "-", bestSudoku ? `${formatNumber(bestSudoku.avg_duration_ms, 1)} ms` : "-"],
    ["Best 2048", bestGame?.method || "-", bestGame ? `${formatNumber(bestGame.avg_score, 0)} avg score` : "-"],
    ["2048 ceiling", bestGame ? `${bestGame.best_max_tile} best tile` : "-", bestGame ? `${formatNumber(bestGame.avg_moves, 1)} avg moves` : "-"],
  ];

  const grid = document.createDocumentFragment();
  cards.forEach(([label, value, detail]) => {
    const node = document.createElement("div");
    node.className = "dashboard-card";
    node.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div><p>${detail}</p>`;
    grid.appendChild(node);
  });
  root.replaceChildren(grid);
}

function renderExportArtifacts(payload) {
  const root = byId("export-artifacts");
  if (!payload?.artifacts) {
    root.innerHTML = "";
    return;
  }
  const artifacts = payload.artifacts;
  const plots = artifacts.plots || [];
  const rows = [
    ["Sudoku CSV", artifacts.sudoku_csv],
    ["2048 CSV", artifacts.game2048_csv],
    ["Summary JSON", artifacts.summary_json],
    ["Plots", plots.length ? plots.join(" | ") : "No plots written"],
  ];
  const output = document.createElement("div");
  output.className = "artifact-list";
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.innerHTML = `<strong>${label}</strong><span>${value}</span>`;
    output.appendChild(row);
  });
  root.replaceChildren(output);
}

function compactHistory(data, series, maxPoints = 240) {
  if (!data || data.length <= maxPoints) {
    return data || [];
  }

  const bucketSize = Math.ceil(data.length / maxPoints);
  const compacted = [];
  for (let start = 0; start < data.length; start += bucketSize) {
    const bucket = data.slice(start, start + bucketSize);
    const row = {
      episode: bucket.at(-1)?.episode ?? start + bucket.length,
      __from: bucket[0]?.episode ?? start + 1,
      __to: bucket.at(-1)?.episode ?? start + bucket.length,
    };
    series.forEach((item) => {
      const values = bucket
        .map((point) => Number(point[item.key]))
        .filter((value) => Number.isFinite(value));
      row[item.key] = values.length
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : 0;
    });
    compacted.push(row);
  }
  return compacted;
}

function renderChart(rootId, data, series, config) {
  const root = byId(rootId);
  root.innerHTML = "";
  if (!data || !data.length) {
    root.innerHTML = '<div class="empty-state">No training history loaded.</div>';
    return;
  }

  const activeSeries = series.filter((item) => data.some((row) => row[item.key] !== undefined));
  const plotted = compactHistory(data, activeSeries);
  const width = 620;
  const height = 260;
  const left = 58;
  const right = 18;
  const top = 20;
  const bottom = 54;
  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  const meta = document.createElement("div");
  meta.className = "chart-meta";
  meta.textContent = plotted.length === data.length
    ? `${data.length} episodes shown`
    : `${data.length} episodes bucketed into ${plotted.length} averaged points`;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <line class="axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line>
    <line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"></line>
  `;

  const allValues = [];
  activeSeries.forEach((item) => {
    plotted.forEach((row) => {
      const value = Number(row[item.key] || 0);
      if (Number.isFinite(value)) {
        allValues.push(value);
      }
    });
  });
  const rawMin = Math.min(...allValues, 0);
  const rawMax = Math.max(...allValues, 1);
  const paddingValue = (rawMax - rawMin || 1) * 0.08;
  const min = Math.floor(rawMin - paddingValue);
  const max = Math.ceil(rawMax + paddingValue);
  const range = max - min || 1;

  function addText(text, x, y, className, rotate = false) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
    node.textContent = text;
    node.setAttribute("x", String(x));
    node.setAttribute("y", String(y));
    node.setAttribute("class", className);
    if (rotate) {
      node.setAttribute("transform", `rotate(-90 ${x} ${y})`);
    }
    svg.appendChild(node);
  }

  addText(String(max), left - 10, top + 4, "axis-tick");
  addText(String(min), left - 10, height - bottom + 4, "axis-tick");
  addText(config.yLabel, 16, height / 2 + 42, "axis-label-y", true);
  addText(config.xLabel, width / 2, height - 12, "axis-label-x");
  addText(String(plotted[0]?.episode ?? 1), left, height - bottom + 20, "axis-tick-x");
  addText(String(plotted.at(-1)?.episode ?? data.length), width - right, height - bottom + 20, "axis-tick-x");

  activeSeries.forEach((item) => {
    const points = plotted.map((row, idx) => {
      const value = Number(row[item.key] || 0);
      const x = left + (idx / Math.max(plotted.length - 1, 1)) * (width - left - right);
      const y = height - bottom - ((value - min) / range) * (height - top - bottom);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", points.join(" "));
    line.setAttribute("stroke", item.color);
    svg.appendChild(line);
  });

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  activeSeries.forEach((item) => {
    const label = document.createElement("span");
    label.innerHTML = `<i style="background:${item.color}"></i>${item.label}`;
    legend.appendChild(label);
  });

  wrap.append(meta, svg, legend);
  root.appendChild(wrap);
}

function renderCurves() {
  renderChart("sudoku-curve", state.curves.sudoku?.[state.difficulty] || [], [
    { key: "reward", label: "Reward", color: "#2563eb" },
    { key: "rolling_reward", label: "Reward avg25", color: "#0f766e" },
    { key: "filled", label: "Filled", color: "#16a34a" },
    { key: "rolling_filled", label: "Filled avg25", color: "#84cc16" },
  ], { xLabel: "Episode", yLabel: "Reward / filled cells" });
  renderChart("game-curve", state.curves["2048"] || [], [
    { key: "score", label: "Score", color: "#ea580c" },
    { key: "rolling_score", label: "Score avg25", color: "#dc2626" },
    { key: "max_tile", label: "Max tile", color: "#7c3aed" },
    { key: "rolling_max_tile", label: "Tile avg25", color: "#2563eb" },
  ], { xLabel: "Episode", yLabel: "Score / max tile" });
}

function updateReplayMethods() {
  const domain = byId("replay-domain").value;
  const options = REPLAY_OPTIONS[domain];
  const defaults = domain === "sudoku"
    ? ["mrv", "backtracking"]
    : ["expectimax", "random"];
  ["replay-method-a", "replay-method-b"].forEach((id, index) => {
    const select = byId(id);
    select.innerHTML = "";
    options.forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      if (value === defaults[index]) {
        option.selected = true;
      }
      select.appendChild(option);
    });
  });
}

function labelForMethod(domain, value) {
  return REPLAY_OPTIONS[domain].find((item) => item[0] === value)?.[1] || value;
}

function miniSudokuBoard(board) {
  const grid = document.createElement("div");
  grid.className = "mini-sudoku-board";
  board.forEach((value) => {
    const cell = document.createElement("div");
    cell.textContent = value ? String(value) : "";
    grid.appendChild(cell);
  });
  return grid;
}

function miniGameBoard(board) {
  const grid = document.createElement("div");
  grid.className = "mini-game-board";
  board.flat().forEach((value) => {
    const cell = document.createElement("div");
    cell.className = `game-tile tile-${Math.min(value, 2048)}`;
    cell.textContent = value ? String(value) : "";
    grid.appendChild(cell);
  });
  return grid;
}

async function runSudokuReplay(method) {
  try {
    const payload = await requestJson("/sudoku/solve", {
      method: "POST",
      body: JSON.stringify({
        difficulty: state.difficulty,
        board: state.sudokuPuzzle,
        algorithm: method,
        seed: optionalNumber("replay-seed"),
      }),
    });
    const finalStep = payload.steps.at(-1);
    return {
      ok: true,
      domain: "sudoku",
      method,
      board: finalStep?.board || state.sudokuPuzzle,
      metrics: [
        ["Solved", payload.solved ? "Yes" : "No"],
        ["Steps", payload.total_steps],
        ["Filled", (finalStep?.board || []).filter(Boolean).length],
      ],
    };
  } catch (error) {
    return { ok: false, domain: "sudoku", method, error: error instanceof Error ? error.message : String(error) };
  }
}

async function runGameReplay(policy) {
  try {
    const payload = await requestJson("/2048/solve", {
      method: "POST",
      body: JSON.stringify({
        policy,
        seed: optionalNumber("replay-seed"),
        max_moves: boundedNumber("replay-max-moves", 400, 20, 2000),
      }),
    });
    const finalStep = payload.steps.at(-1);
    const board = finalStep?.board || Array.from({ length: 4 }, () => Array(4).fill(0));
    return {
      ok: true,
      domain: "2048",
      method: policy,
      board,
      metrics: [
        ["Score", finalStep?.score ?? 0],
        ["Max tile", Math.max(...board.flat(), 0)],
        ["Moves", Math.max(0, payload.total_steps - 1)],
      ],
    };
  } catch (error) {
    return { ok: false, domain: "2048", method: policy, error: error instanceof Error ? error.message : String(error) };
  }
}

function renderReplayResults(domain, results) {
  const root = byId("side-by-side-replay");
  const fragment = document.createDocumentFragment();
  results.forEach((result) => {
    const pane = document.createElement("div");
    pane.className = "replay-pane";
    const title = document.createElement("div");
    title.className = "replay-title";
    title.textContent = labelForMethod(domain, result.method);
    pane.appendChild(title);
    if (!result.ok) {
      const error = document.createElement("div");
      error.className = "empty-state";
      error.textContent = result.error;
      pane.appendChild(error);
      fragment.appendChild(pane);
      return;
    }

    pane.appendChild(domain === "sudoku" ? miniSudokuBoard(result.board) : miniGameBoard(result.board));
    const metrics = document.createElement("div");
    metrics.className = "mini-metrics";
    result.metrics.forEach(([label, value]) => {
      metrics.appendChild(statNode(label, value));
    });
    pane.appendChild(metrics);
    fragment.appendChild(pane);
  });
  root.replaceChildren(fragment);
}

async function runSideBySideReplay() {
  const domain = byId("replay-domain").value;
  const first = byId("replay-method-a").value;
  const second = byId("replay-method-b").value;
  const results = domain === "sudoku"
    ? await Promise.all([runSudokuReplay(first), runSudokuReplay(second)])
    : await Promise.all([runGameReplay(first), runGameReplay(second)]);
  renderReplayResults(domain, results);
}

function wireEvents() {
  byId("tab-sudoku").addEventListener("click", () => setActivePanel("sudoku"));
  byId("tab-2048").addEventListener("click", () => setActivePanel("2048"));

  byId("sudoku-difficulty").addEventListener("change", () => {
    state.difficulty = byId("sudoku-difficulty").value;
    byId("compare-difficulty").value = state.difficulty;
    execute("sudoku-puzzle", loadSudokuPuzzle);
  });
  byId("sudoku-algorithm").addEventListener("change", updateButtons);
  byId("game-policy").addEventListener("change", updateButtons);

  byId("btn-sudoku-puzzle").addEventListener("click", () => execute("sudoku-puzzle", loadSudokuPuzzle));
  byId("btn-sudoku-train").addEventListener("click", () => executeLiveJob(
    "sudoku-train",
    () => requestJson("/sudoku/train", {
      method: "POST",
      body: JSON.stringify({
        difficulty: state.difficulty,
        episodes: Number(byId("sudoku-episodes").value),
        seed: optionalNumber("sudoku-seed"),
        ...readHyperparameters(),
      }),
    }),
    (job) => {
      state.curves.sudoku[state.difficulty] = job.history;
      byId("s-stat-step").textContent = `${job.progress}/${job.total || "?"}`;
      byId("s-stat-message").textContent = job.message;
      byId("status-sudoku").textContent = job.message;
      byId("status-sudoku").classList.remove("ready");
      renderCurves();
    },
    (job) => {
      if (job.result?.training_history) {
        state.curves.sudoku[state.difficulty] = job.result.training_history;
      }
      renderCurves();
    },
  ));
  byId("btn-sudoku-solve").addEventListener("click", () => executeLiveJob(
    "sudoku-solve",
    () => requestJson("/sudoku/solve", {
      method: "POST",
      body: JSON.stringify({
        difficulty: state.difficulty,
        board: state.sudokuPuzzle,
        algorithm: byId("sudoku-algorithm").value,
        seed: optionalNumber("sudoku-seed"),
        live: true,
      }),
    }),
    (job) => {
      state.sudokuSteps = job.steps;
      state.sudokuIndex = Math.max(0, job.steps.length - 1);
      renderSudokuStep();
    },
    (job) => {
      if (job.result?.training_history) {
        state.curves.sudoku[state.difficulty] = job.result.training_history;
      }
      renderCurves();
    },
  ));
  byId("sudoku-range").addEventListener("input", (event) => {
    state.sudokuIndex = Number(event.target.value);
    renderSudokuStep();
  });
  ["save", "load", "reset"].forEach((action) => {
    byId(`btn-sudoku-${action}`).addEventListener("click", () => execute("sudoku-model", async () => {
      const result = await requestJson("/sudoku/model", {
        method: "POST",
        body: JSON.stringify({ difficulty: state.difficulty, action }),
      });
      byId("s-stat-message").textContent = result.message;
    }));
  });

  byId("btn-game-train").addEventListener("click", () => executeLiveJob(
    "game-train",
    () => requestJson("/2048/train", {
      method: "POST",
      body: JSON.stringify({
        episodes: Number(byId("game-episodes").value),
        seed: optionalNumber("game-seed"),
        ...readHyperparameters(),
      }),
    }),
    (job) => {
      state.curves["2048"] = job.history;
      byId("g-stat-step").textContent = `${job.progress}/${job.total || "?"}`;
      byId("g-stat-score").textContent = String(job.history.at(-1)?.score ?? 0);
      byId("g-stat-max").textContent = String(job.history.at(-1)?.max_tile ?? "-");
      byId("g-stat-reward").textContent = job.message;
      byId("status-2048").textContent = job.message;
      byId("status-2048").classList.remove("ready");
      renderCurves();
    },
    (job) => {
      if (job.result?.training_history) {
        state.curves["2048"] = job.result.training_history;
      }
      renderCurves();
    },
  ));
  byId("btn-game-play").addEventListener("click", () => executeLiveJob(
    "game-play",
    () => requestJson("/2048/solve", {
      method: "POST",
      body: JSON.stringify({
        policy: byId("game-policy").value,
        seed: optionalNumber("game-seed"),
        live: true,
        max_moves: boundedNumber("game-max-moves", 1000, 20, 5000),
      }),
    }),
    (job) => {
      state.gameSteps = job.steps;
      state.gameIndex = Math.max(0, job.steps.length - 1);
      renderGameStep();
    },
    (job) => {
      if (job.result?.training_history) {
        state.curves["2048"] = job.result.training_history;
      }
      renderCurves();
    },
  ));
  byId("game-range").addEventListener("input", (event) => {
    state.gameIndex = Number(event.target.value);
    renderGameStep();
  });
  ["save", "load", "reset"].forEach((action) => {
    byId(`btn-game-${action}`).addEventListener("click", () => execute("game-model", async () => {
      const result = await requestJson("/2048/model", {
        method: "POST",
        body: JSON.stringify({ action }),
      });
      byId("g-stat-reward").textContent = result.message;
    }));
  });

  byId("btn-compare").addEventListener("click", () => execute("compare", async () => {
    const difficulty = byId("compare-difficulty").value;
    const runs = Math.max(1, Math.min(10, Number(byId("compare-runs").value) || 3));
    const gameMaxMoves = Math.max(20, Math.min(2000, Number(byId("compare-max-moves").value) || 600));
    const seed = boundedNumber("compare-seed", 101, 0, 999999999);
    const comparison = await requestJson("/analysis/compare", {
      method: "POST",
      body: JSON.stringify({ difficulty, runs, seed, game_max_moves: gameMaxMoves }),
    });
    state.lastComparison = comparison;
    renderSudokuComparison(comparison.sudoku.summary);
    renderGameComparison(comparison.game2048.summary);
    renderDashboard(comparison);
  }));
  byId("btn-export").addEventListener("click", () => execute("export", async () => {
    const difficulty = byId("compare-difficulty").value;
    const runs = Math.max(1, Math.min(10, Number(byId("compare-runs").value) || 3));
    const gameMaxMoves = Math.max(20, Math.min(2000, Number(byId("compare-max-moves").value) || 600));
    const seed = boundedNumber("compare-seed", 101, 0, 999999999);
    const exported = await requestJson("/analysis/export", {
      method: "POST",
      body: JSON.stringify({ difficulty, runs, seed, game_max_moves: gameMaxMoves }),
    });
    state.lastComparison = exported.comparison;
    renderSudokuComparison(exported.comparison.sudoku.summary);
    renderGameComparison(exported.comparison.game2048.summary);
    renderDashboard(exported.comparison);
    renderExportArtifacts(exported);
  }));
  byId("replay-domain").addEventListener("change", updateReplayMethods);
  byId("btn-replay").addEventListener("click", () => execute("replay", runSideBySideReplay));
}

async function init() {
  wireEvents();
  updateReplayMethods();
  renderSudokuStep();
  renderGameStep();
  const online = await refreshStatus();
  if (online) {
    await loadSudokuPuzzle().catch((error) => showError(error instanceof Error ? error.message : String(error)));
  }
  window.setInterval(async () => {
    const wasOffline = state.apiOnline === false;
    const isOnline = await refreshStatus({ silent: true });
    if (isOnline && wasOffline && !state.sudokuPuzzle.some(Boolean)) {
      clearError();
      await loadSudokuPuzzle().catch((error) => showError(error instanceof Error ? error.message : String(error)));
    }
  }, 1800);
}

init().catch((error) => showError(error.message));
