from __future__ import annotations


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Epidemic Containment Dashboard</title>
  <style>
    :root {
      --bg: #f6f2ea;
      --panel: #fffdf8;
      --ink: #182026;
      --muted: #58636e;
      --accent: #205c7a;
      --accent-soft: #d5ebf5;
      --danger: #b63b2a;
      --warn: #d28715;
      --good: #2e7d55;
      --border: #d8d2c8;
      --shadow: 0 10px 24px rgba(24, 32, 38, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(32, 92, 122, 0.10), transparent 28%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }

    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 20px;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: 2.2rem;
      letter-spacing: 0.02em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.45;
    }

    .grid {
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 18px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }

    .controls {
      position: sticky;
      top: 18px;
      height: fit-content;
    }

    .panel h2, .panel h3 {
      margin: 0 0 12px;
      font-size: 1.05rem;
    }

    .field {
      margin-bottom: 12px;
    }

    .field label {
      display: block;
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .field select,
    .field input {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      font: inherit;
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }

    button {
      border: 0;
      border-radius: 12px;
      padding: 11px 12px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
    }

    button.secondary {
      background: #e7ecef;
      color: var(--ink);
    }

    button.warn {
      background: var(--warn);
    }

    button.good {
      background: var(--good);
    }

    button:disabled {
      opacity: 0.55;
      cursor: default;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }

    .chip {
      background: #efe7da;
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 10px;
      font-size: 0.9rem;
      display: inline-flex;
      gap: 8px;
      align-items: center;
    }

    .chip button {
      background: transparent;
      color: var(--danger);
      padding: 0;
      min-width: auto;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .stat {
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }

    .stat .label {
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .stat .value {
      font-size: 1.55rem;
      font-weight: 700;
      letter-spacing: 0.01em;
    }

    .content-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .graph-wrap {
      min-height: 540px;
    }

    svg {
      width: 100%;
      height: 520px;
      display: block;
      background:
        radial-gradient(circle at center, rgba(32, 92, 122, 0.05), transparent 48%),
        linear-gradient(180deg, #fffdfa, #f9f4ea);
      border-radius: 16px;
      border: 1px solid var(--border);
    }

    .legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .legend span::before {
      content: "";
      display: inline-block;
      width: 14px;
      height: 14px;
      margin-right: 8px;
      border-radius: 50%;
      vertical-align: -2px;
    }

    .legend .infection::before { background: #cf4b34; }
    .legend .economy::before { background: #2e7d55; }
    .legend .quarantine::before {
      background: #fff;
      border: 2px solid #111;
    }

    .table-panel {
      overflow: auto;
    }

    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .chart-card {
      padding-bottom: 10px;
    }

    .chart-frame {
      width: 100%;
      height: 220px;
      display: block;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247, 241, 231, 0.95));
      border-radius: 14px;
      border: 1px solid var(--border);
    }

    .chart-legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.86rem;
    }

    .chart-legend span::before {
      content: "";
      display: inline-block;
      width: 14px;
      height: 3px;
      margin-right: 8px;
      vertical-align: middle;
      border-radius: 999px;
      background: #000;
    }

    .chart-legend .actual::before { background: #c63a2f; }
    .chart-legend .reported::before { background: #d28715; }
    .chart-legend .economy::before { background: #2e7d55; }
    .chart-legend .score::before { background: #205c7a; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }

    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: middle;
    }

    th {
      color: var(--muted);
      font-weight: 600;
    }

    .bar {
      position: relative;
      height: 12px;
      background: #ebe7df;
      border-radius: 999px;
      overflow: hidden;
      min-width: 110px;
    }

    .bar > span {
      display: block;
      height: 100%;
      border-radius: 999px;
    }

    .bar.infection > span { background: linear-gradient(90deg, #f6ae6b, #c63a2f); }
    .bar.economy > span { background: linear-gradient(90deg, #78c398, #2e7d55); }

    .history {
      max-height: 220px;
      overflow: auto;
      font-size: 0.92rem;
      line-height: 1.45;
    }

    .history-entry {
      border-bottom: 1px solid var(--border);
      padding: 10px 0;
    }

    .muted {
      color: var(--muted);
    }

    .status {
      min-height: 24px;
      color: var(--danger);
      font-size: 0.92rem;
      margin-top: 8px;
    }

    @media (max-width: 1100px) {
      .grid, .content-grid, .chart-grid {
        grid-template-columns: 1fr;
      }
      .controls {
        position: static;
      }
      .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <h1>Epidemic Containment Dashboard</h1>
        <p>Explore the graph outbreak step by step. You can run the baseline policy, drive the simulation manually, call the configured server-side LLM, and compare reported infections with the hidden true state.</p>
      </div>
    </div>

    <div class="grid">
      <aside class="panel controls">
        <h2>Scenario Control</h2>
        <div class="field">
          <label for="taskName">Task</label>
          <select id="taskName">
            <option value="easy_localized_outbreak">easy_localized_outbreak</option>
            <option value="medium_multi_center_spread">medium_multi_center_spread</option>
            <option value="hard_asymptomatic_high_density">hard_asymptomatic_high_density</option>
          </select>
        </div>
        <div class="field">
          <label for="seedValue">Seed</label>
          <input id="seedValue" type="number" value="42" />
        </div>
        <div class="button-row">
          <button id="resetButton">Reset</button>
          <button id="baselineButton" class="good">Baseline Step</button>
        </div>
        <div class="button-row">
          <button id="llmButton">LLM Step</button>
          <button id="playLlmButton" class="warn">Play LLM</button>
        </div>
        <div class="button-row">
          <button id="playBaselineButton" class="warn">Play Baseline</button>
          <button id="stopPlayButton" class="secondary">Stop</button>
        </div>

        <h3>Manual Action Builder</h3>
        <div class="field">
          <label for="actionKind">Action</label>
          <select id="actionKind">
            <option value="vaccinate">vaccinate</option>
            <option value="quarantine">quarantine</option>
            <option value="lift_quarantine">lift_quarantine</option>
          </select>
        </div>
        <div class="field">
          <label for="nodeId">Node</label>
          <select id="nodeId"></select>
        </div>
        <div class="field">
          <label for="amountValue">Vaccine Amount</label>
          <input id="amountValue" type="number" value="80" step="1" />
        </div>
        <div class="button-row">
          <button id="addActionButton" class="secondary">Add Action</button>
          <button id="clearActionsButton" class="secondary">Clear</button>
        </div>

        <div class="chips" id="pendingActions"></div>

        <div class="button-row">
          <button id="manualStepButton">Step Manual Plan</button>
          <button id="noopButton" class="secondary">No-op Step</button>
        </div>

        <div class="status" id="statusText"></div>
      </aside>

      <main>
        <section class="stats" id="statsGrid"></section>

        <section class="content-grid">
          <div class="panel graph-wrap">
            <h2>Travel Graph</h2>
            <svg id="graphSvg" viewBox="0 0 920 520" preserveAspectRatio="xMidYMid meet"></svg>
            <div class="legend">
              <span class="infection">Node fill tracks actual infection</span>
              <span class="economy">Economy shown in table as green bars</span>
              <span class="quarantine">Black outline means quarantined</span>
            </div>
          </div>

          <div class="panel">
            <h2>Task Summary</h2>
            <div id="taskSummary" class="muted"></div>
            <h3 style="margin-top: 18px;">Recent History</h3>
            <div id="historyLog" class="history"></div>
          </div>
        </section>

        <section class="chart-grid">
          <div class="panel chart-card">
            <h2>Infection Timeline</h2>
            <svg id="infectionChart" class="chart-frame" viewBox="0 0 520 220" preserveAspectRatio="none"></svg>
            <div class="chart-legend">
              <span class="actual">Actual infection</span>
              <span class="reported">Reported infection</span>
            </div>
          </div>
          <div class="panel chart-card">
            <h2>Economy And Score Timeline</h2>
            <svg id="economyChart" class="chart-frame" viewBox="0 0 520 220" preserveAspectRatio="none"></svg>
            <div class="chart-legend">
              <span class="economy">Economy</span>
              <span class="score">Task score</span>
            </div>
          </div>
        </section>

        <section class="panel table-panel">
          <h2>Node Metrics</h2>
          <table>
            <thead>
              <tr>
                <th>Node</th>
                <th>Reported Infection</th>
                <th>Actual Infection</th>
                <th>Economy</th>
                <th>Population</th>
                <th>Quarantine</th>
              </tr>
            </thead>
            <tbody id="nodeTable"></tbody>
          </table>
        </section>
      </main>
    </div>
  </div>

  <script>
    const MAX_PENDING_INTERVENTIONS = 3;
    const INFECTION_CLEAR_EPSILON = 0.0005;
    const STAGNATION_WINDOW = 4;
    const STAGNATION_BAND = 0.0025;
    const pendingInterventions = [];
    let currentPayload = null;
    let autoplay = false;

    function pct(value) {
      return `${(value * 100).toFixed(1)}%`;
    }

    function byId(id) {
      return document.getElementById(id);
    }

    function setStatus(text, isError = true) {
      const el = byId('statusText');
      el.textContent = text || '';
      el.style.color = isError ? '#b63b2a' : '#2e7d55';
    }

    function infectionSignal(payload) {
      const history = payload && payload.state && payload.state.history ? payload.state.history : [];
      const actualSeries = history.map((entry) => entry.actual_total_infection_rate);
      const currentActual = payload && payload.state ? Number(payload.state.actual_total_infection_rate || 0) : 0;
      const recentSeries = actualSeries.slice(-STAGNATION_WINDOW);
      const currentNodes = payload && payload.state && payload.state.nodes ? payload.state.nodes : [];
      const allNodesClear = currentNodes.length > 0 && currentNodes.every((node) => Number(node.actual_infection_rate || 0) <= INFECTION_CLEAR_EPSILON);
      const cleared = currentActual <= INFECTION_CLEAR_EPSILON || allNodesClear;

      if (cleared) {
        return {
          label: 'cleared',
          detail: 'actual infection is effectively zero',
          trend: 'contained',
          shouldStopAutoplay: true,
        };
      }

      if (recentSeries.length >= STAGNATION_WINDOW) {
        const minRecent = Math.min(...recentSeries);
        const maxRecent = Math.max(...recentSeries);
        if (maxRecent - minRecent <= STAGNATION_BAND) {
          return {
            label: 'stagnating',
            detail: `actual infection has stayed within ${(STAGNATION_BAND * 100).toFixed(2)}% for ${STAGNATION_WINDOW} steps`,
            trend: 'flat',
            shouldStopAutoplay: true,
          };
        }
      }

      const previousActual = actualSeries.length >= 2
        ? actualSeries[actualSeries.length - 2]
        : currentActual;
      const delta = currentActual - previousActual;
      let trend = 'steady';
      if (delta <= -STAGNATION_BAND) {
        trend = 'falling';
      } else if (delta >= STAGNATION_BAND) {
        trend = 'rising';
      }

      return {
        label: 'active',
        detail: `actual infection ${trend} at ${pct(currentActual)}`,
        trend,
        shouldStopAutoplay: false,
      };
    }

    function currentNodeObservation(nodeId) {
      if (!currentPayload || !currentPayload.observation) {
        return null;
      }
      return currentPayload.observation.nodes.find((node) => node.node_id === nodeId) || null;
    }

    function remainingManualBudget(excludedIndex = -1) {
      if (!currentPayload || !currentPayload.observation) {
        return 0;
      }
      const reserved = pendingInterventions.reduce((sum, action, index) => {
        if (index === excludedIndex || action.kind !== 'vaccinate') {
          return sum;
        }
        return sum + (Number(action.amount) || 0);
      }, 0);
      return Math.max(0, currentPayload.observation.vaccine_budget - reserved);
    }

    function syncPendingInterventions() {
      if (!currentPayload || !currentPayload.observation) {
        return;
      }
      const next = [];
      let spend = 0;
      pendingInterventions.forEach((action) => {
        const node = currentNodeObservation(action.node_id);
        if (!node) {
          return;
        }
        if (action.kind === 'quarantine' && node.is_quarantined) {
          return;
        }
        if (action.kind === 'lift_quarantine' && !node.is_quarantined) {
          return;
        }
        if (action.kind === 'vaccinate') {
          const amount = Number(action.amount) || 0;
          const available = Math.max(0, currentPayload.observation.vaccine_budget - spend);
          if (available <= 0 || amount <= 0) {
            return;
          }
          const clipped = Number(Math.min(amount, available).toFixed(1));
          spend += clipped;
          next.push({ ...action, amount: clipped });
          return;
        }
        next.push(action);
      });
      if (next.length !== pendingInterventions.length || next.some((action, index) => JSON.stringify(action) !== JSON.stringify(pendingInterventions[index]))) {
        pendingInterventions.splice(0, pendingInterventions.length, ...next);
        renderPendingActions();
      }
    }

    function handleStepPayload(payload, successText) {
      renderDashboard(payload);
      const info = payload.info || {};
      const errors = info.errors || [];
      if (info.action_blocked) {
        autoplay = false;
        setStatus(errors.length ? errors.join('; ') : 'Action blocked before the step advanced.', true);
      } else if (errors.length) {
        setStatus(errors.join('; '), true);
      } else {
        setStatus(successText, false);
      }
      return payload;
    }

    function renderPendingActions() {
      const holder = byId('pendingActions');
      holder.innerHTML = '';
      pendingInterventions.forEach((item, index) => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.innerHTML = `<span>${describeAction(item)}</span>`;
        const btn = document.createElement('button');
        btn.textContent = 'x';
        btn.onclick = () => {
          pendingInterventions.splice(index, 1);
          renderPendingActions();
        };
        chip.appendChild(btn);
        holder.appendChild(chip);
      });
    }

    function describeAction(action) {
      if (action.kind === 'vaccinate') {
        return `vaccinate(${action.node_id}, ${Number(action.amount).toFixed(0)})`;
      }
      return `${action.kind}(${action.node_id})`;
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
      if (!response.ok) {
        const text = await response.text();
        try {
          const parsed = JSON.parse(text);
          throw new Error(parsed.detail || `Request failed with ${response.status}`);
        } catch (_) {
          throw new Error(text || `Request failed with ${response.status}`);
        }
      }
      return response.json();
    }

    function updateNodeOptions(nodes) {
      const select = byId('nodeId');
      const previous = select.value;
      select.innerHTML = '';
      nodes.forEach((node) => {
        const option = document.createElement('option');
        option.value = node.node_id;
        option.textContent = node.node_id;
        select.appendChild(option);
      });
      if (previous && nodes.some((node) => node.node_id === previous)) {
        select.value = previous;
      }
    }

    function renderStats(payload) {
      const { observation, state, evaluation } = payload;
      const signal = infectionSignal(payload);
      const items = [
        ['Step', `${observation.step_count}`],
        ['Horizon', `${observation.max_steps}`],
        ['Vaccine Budget', observation.vaccine_budget.toFixed(1)],
        ['Reported Infection', pct(observation.reported_total_infection_rate)],
        ['Actual Infection', pct(state.actual_total_infection_rate)],
        ['Outbreak', signal.label],
        ['Trend', signal.trend],
        ['Economy', pct(observation.global_economic_score)],
        ['Peak Infection', pct(state.peak_infection_rate)],
        ['Task Score', evaluation.score.toFixed(2)],
        ['Success', evaluation.success ? 'yes' : 'no'],
        ['Reporting Lag', `${observation.reporting_lag_steps} step(s)`],
        ['Done', observation.done ? 'yes' : 'no']
      ];
      byId('statsGrid').innerHTML = items.map(([label, value]) => `
        <div class="stat">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>
      `).join('');
    }

    function colorForInfection(rate) {
      const clamped = Math.max(0, Math.min(1, rate));
      const hue = 32 - clamped * 25;
      const light = 84 - clamped * 34;
      return `hsl(${hue}, 70%, ${light}%)`;
    }

    function layoutNodes(nodes) {
      const cx = 460;
      const cy = 255;
      const radius = nodes.length > 12 ? 190 : 165;
      return nodes.map((node, index) => {
        const theta = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
        return {
          ...node,
          x: cx + Math.cos(theta) * radius,
          y: cy + Math.sin(theta) * radius
        };
      });
    }

    function renderGraph(payload) {
      const svg = byId('graphSvg');
      const stateNodes = payload.state.nodes;
      const laidOut = layoutNodes(stateNodes);
      const nodeMap = Object.fromEntries(laidOut.map((node) => [node.node_id, node]));

      const edgeMarkup = payload.state.edges.map((edge) => {
        const a = nodeMap[edge.source];
        const b = nodeMap[edge.target];
        const stroke = edge.active ? '#86a9b8' : '#d5cfc5';
        const dash = edge.active ? '0' : '8 6';
        return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${stroke}" stroke-width="3" stroke-dasharray="${dash}" opacity="0.85" />`;
      }).join('');

      const nodeMarkup = laidOut.map((node) => {
        const fill = colorForInfection(node.actual_infection_rate);
        const stroke = node.quarantined ? '#111111' : '#ffffff';
        const strokeWidth = node.quarantined ? 6 : 3;
        const radius = 24 + Math.max(0, Math.min(24, node.population / 110));
        return `
          <g>
            <circle cx="${node.x}" cy="${node.y}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" />
            <text x="${node.x}" y="${node.y - 2}" text-anchor="middle" font-size="13" font-weight="700" fill="#182026">${node.node_id}</text>
            <text x="${node.x}" y="${node.y + 15}" text-anchor="middle" font-size="11" fill="#182026">${pct(node.actual_infection_rate)}</text>
          </g>
        `;
      }).join('');

      svg.innerHTML = edgeMarkup + nodeMarkup;
    }

    function renderNodeTable(payload) {
      const reportedMap = Object.fromEntries(payload.observation.nodes.map((node) => [node.node_id, node]));
      const rows = payload.state.nodes.map((node) => {
        const reported = reportedMap[node.node_id];
        return `
          <tr>
            <td><strong>${node.node_id}</strong></td>
            <td>
              <div class="bar infection"><span style="width:${reported.known_infection_rate * 100}%"></span></div>
              <div>${pct(reported.known_infection_rate)}</div>
            </td>
            <td>
              <div class="bar infection"><span style="width:${node.actual_infection_rate * 100}%"></span></div>
              <div>${pct(node.actual_infection_rate)}</div>
            </td>
            <td>
              <div class="bar economy"><span style="width:${node.economic_health * 100}%"></span></div>
              <div>${pct(node.economic_health)}</div>
            </td>
            <td>${node.population}</td>
            <td>${node.quarantined ? 'yes' : 'no'}</td>
          </tr>
        `;
      }).join('');
      byId('nodeTable').innerHTML = rows;
    }

    function renderTaskSummary(payload) {
      const observation = payload.observation;
      const evaluation = payload.evaluation;
      const info = payload.info || {};
      const llm = payload.llm || {};
      const signal = infectionSignal(payload);
      const alerts = observation.alerts && observation.alerts.length ? observation.alerts.join(', ') : 'none';
      const errors = info.errors && info.errors.length ? info.errors.join('; ') : 'none';
      const llmSummary = llm.configured
        ? `${llm.model_name} @ ${llm.api_base_url} (timeout ${llm.timeout_s}s)`
        : 'not configured on the server';
      const decisionSource = info.decision_source || 'manual/no-op';
      byId('taskSummary').innerHTML = `
        <p><strong>${observation.task_name}</strong> (${observation.difficulty})</p>
        <p>${observation.goal}</p>
        <p><strong>Decision Source:</strong> ${decisionSource}</p>
        <p><strong>Outbreak State:</strong> ${signal.label} (${signal.detail})</p>
        <p><strong>LLM Config:</strong> ${llmSummary}</p>
        <p><strong>Alerts:</strong> ${alerts}</p>
        <p><strong>Summary:</strong> ${evaluation.summary}</p>
        <p><strong>Errors:</strong> ${errors}</p>
      `;
    }

    function renderHistory(payload) {
      const entries = [...payload.state.history].slice(-10).reverse();
      if (!entries.length) {
        byId('historyLog').innerHTML = '<div class="muted">No steps yet.</div>';
        return;
      }
      byId('historyLog').innerHTML = entries.map((entry) => `
        <div class="history-entry">
          <div><strong>Step ${entry.step}</strong> | reward ${entry.reward.toFixed(2)}</div>
          <div class="muted">actions: ${entry.action_descriptions.length ? entry.action_descriptions.join(', ') : 'noop'}</div>
          <div class="muted">reported ${pct(entry.reported_total_infection_rate)} | actual ${pct(entry.actual_total_infection_rate)} | economy ${pct(entry.global_economic_score)}</div>
        </div>
      `).join('');
    }

    function buildSeriesPoints(series, width, height, maxValue) {
      if (!series.length) {
        return '';
      }
      const left = 36;
      const right = width - 12;
      const top = 14;
      const bottom = height - 28;
      const usableWidth = Math.max(1, right - left);
      const usableHeight = Math.max(1, bottom - top);
      return series.map((value, index) => {
        const x = left + (series.length === 1 ? usableWidth / 2 : (usableWidth * index) / (series.length - 1));
        const y = bottom - (Math.max(0, Math.min(maxValue, value)) / maxValue) * usableHeight;
        return `${x},${y}`;
      }).join(' ');
    }

    function axisMarkup(width, height) {
      const left = 36;
      const right = width - 12;
      const top = 14;
      const bottom = height - 28;
      const mid = (top + bottom) / 2;
      return `
        <line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#b9b1a4" stroke-width="1.5" />
        <line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#b9b1a4" stroke-width="1.5" />
        <line x1="${left}" y1="${mid}" x2="${right}" y2="${mid}" stroke="#e3ddd3" stroke-width="1" stroke-dasharray="4 6" />
        <text x="8" y="${top + 4}" font-size="11" fill="#58636e">100%</text>
        <text x="14" y="${mid + 4}" font-size="11" fill="#58636e">50%</text>
        <text x="20" y="${bottom + 4}" font-size="11" fill="#58636e">0%</text>
      `;
    }

    function renderCharts(payload) {
      const history = payload.state.history || [];
      const infectionSvg = byId('infectionChart');
      const economySvg = byId('economyChart');
      const width = 520;
      const height = 220;

      const actualSeries = [0].concat(history.map((entry) => entry.actual_total_infection_rate));
      const reportedSeries = [payload.state.reporting_lag_steps > 0 ? 0 : payload.observation.nodes.length ? payload.state.nodes.reduce((acc, node) => acc + 0, 0) : 0]
        .concat(history.map((entry) => entry.reported_total_infection_rate));
      const economySeries = [1].concat(history.map((entry) => entry.global_economic_score));
      const scoreSeries = history.map((_, index) => {
        const subset = history.slice(0, index + 1);
        const peak = Math.max(...subset.map((entry) => entry.actual_total_infection_rate), payload.state.history.length ? payload.state.history[0].actual_total_infection_rate : 0);
        return Math.max(0, Math.min(1, payload.evaluation.score));
      });
      const chartScoreSeries = [payload.evaluation.score].slice(0, 1).concat(scoreSeries);

      const actualPoints = buildSeriesPoints(actualSeries, width, height, 1.0);
      const reportedPoints = buildSeriesPoints(reportedSeries, width, height, 1.0);
      const economyPoints = buildSeriesPoints(economySeries, width, height, 1.0);
      const scorePoints = buildSeriesPoints(chartScoreSeries, width, height, 1.0);

      infectionSvg.innerHTML = `
        ${axisMarkup(width, height)}
        <polyline fill="none" stroke="#c63a2f" stroke-width="3" points="${actualPoints}" />
        <polyline fill="none" stroke="#d28715" stroke-width="3" stroke-dasharray="7 5" points="${reportedPoints}" />
      `;

      economySvg.innerHTML = `
        ${axisMarkup(width, height)}
        <polyline fill="none" stroke="#2e7d55" stroke-width="3" points="${economyPoints}" />
        <polyline fill="none" stroke="#205c7a" stroke-width="3" stroke-dasharray="7 5" points="${scorePoints}" />
      `;
    }

    function renderDashboard(payload) {
      currentPayload = payload;
      syncPendingInterventions();
      updateNodeOptions(payload.observation.nodes);
      renderStats(payload);
      renderGraph(payload);
      renderCharts(payload);
      renderNodeTable(payload);
      renderTaskSummary(payload);
      renderHistory(payload);
    }

    async function resetDashboard() {
      autoplay = false;
      const payload = await postJson('/dashboard/api/reset', {
        task_name: byId('taskName').value,
        seed: Number(byId('seedValue').value)
      });
      pendingInterventions.length = 0;
      renderPendingActions();
      renderDashboard(payload);
      setStatus('Scenario reset.', false);
    }

    async function stepManual(actionPayload) {
      const payload = await postJson('/dashboard/api/step', actionPayload);
      return handleStepPayload(payload, `Step applied. Reward ${payload.reward.toFixed(2)}.`);
    }

    async function baselineStep() {
      const payload = await postJson('/dashboard/api/baseline-step', {});
      return handleStepPayload(payload, `Baseline step complete. Reward ${payload.reward.toFixed(2)}.`);
    }

    async function llmStep() {
      const payload = await postJson('/dashboard/api/llm-step', {});
      const modelName = payload.llm && payload.llm.model_name ? payload.llm.model_name : 'llm';
      return handleStepPayload(payload, `LLM step complete with ${modelName}. Reward ${payload.reward.toFixed(2)}.`);
    }

    async function autoplayBaseline() {
      autoplay = true;
      while (autoplay && currentPayload && !currentPayload.observation.done) {
        const payload = await baselineStep();
        if (payload.info && payload.info.action_blocked) {
          autoplay = false;
          break;
        }
        const signal = infectionSignal(payload);
        if (signal.shouldStopAutoplay) {
          autoplay = false;
          setStatus(`Autoplay stopped because the outbreak is ${signal.label}: ${signal.detail}.`, false);
          break;
        }
        if (payload.observation.done) {
          autoplay = false;
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 350));
      }
    }

    async function autoplayLlm() {
      autoplay = true;
      while (autoplay && currentPayload && !currentPayload.observation.done) {
        const payload = await llmStep();
        if (payload.info && payload.info.action_blocked) {
          autoplay = false;
          break;
        }
        const signal = infectionSignal(payload);
        if (signal.shouldStopAutoplay) {
          autoplay = false;
          setStatus(`Autoplay stopped because the outbreak is ${signal.label}: ${signal.detail}.`, false);
          break;
        }
        if (payload.observation.done) {
          autoplay = false;
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 600));
      }
    }

    byId('actionKind').addEventListener('change', () => {
      byId('amountValue').disabled = byId('actionKind').value !== 'vaccinate';
    });

    byId('addActionButton').onclick = () => {
      const kind = byId('actionKind').value;
      const nodeId = byId('nodeId').value;
      const existingIndex = pendingInterventions.findIndex((item) => item.node_id === nodeId);
      const node = currentNodeObservation(nodeId);
      if (!nodeId) {
        setStatus('Choose a node before adding an action.');
        return;
      }
      if (!node) {
        setStatus('Reset the dashboard before building a manual plan.');
        return;
      }
      if (existingIndex < 0 && pendingInterventions.length >= MAX_PENDING_INTERVENTIONS) {
        setStatus(`Manual plans are capped at ${MAX_PENDING_INTERVENTIONS} interventions.`);
        return;
      }
      if (kind === 'quarantine' && node.is_quarantined) {
        setStatus(`${nodeId} is already quarantined.`);
        return;
      }
      if (kind === 'lift_quarantine' && !node.is_quarantined) {
        setStatus(`${nodeId} is not quarantined.`);
        return;
      }
      const action = { kind, node_id: nodeId };
      let statusMessage = 'Action added to the manual plan.';
      if (kind === 'vaccinate') {
        const requestedAmount = Number(byId('amountValue').value);
        if (!Number.isFinite(requestedAmount) || requestedAmount <= 0) {
          setStatus('Vaccination amount must be a positive number.');
          return;
        }
        const availableBudget = remainingManualBudget(existingIndex);
        if (availableBudget <= 0) {
          setStatus('No vaccine budget remains for the manual plan.');
          return;
        }
        const clippedAmount = Number(Math.min(requestedAmount, availableBudget).toFixed(1));
        action.amount = clippedAmount;
        if (clippedAmount < requestedAmount) {
          statusMessage = `Vaccination clipped to ${clippedAmount.toFixed(1)} so the manual plan stays within budget.`;
        }
      }
      if (existingIndex >= 0) {
        pendingInterventions.splice(existingIndex, 1, action);
        if (statusMessage === 'Action added to the manual plan.') {
          statusMessage = 'Updated the pending action for that node.';
        }
      } else {
        pendingInterventions.push(action);
      }
      renderPendingActions();
      setStatus(statusMessage, false);
    };

    byId('clearActionsButton').onclick = () => {
      pendingInterventions.length = 0;
      renderPendingActions();
      setStatus('Manual plan cleared.', false);
    };

    byId('resetButton').onclick = () => {
      resetDashboard().catch((error) => setStatus(error.message));
    };

    byId('manualStepButton').onclick = () => {
      if (!pendingInterventions.length) {
        setStatus('Manual plan is empty. Use No-op Step if you want to advance without acting.');
        return;
      }
      stepManual({ interventions: pendingInterventions }).then((payload) => {
        if (!(payload.info && payload.info.action_blocked)) {
          pendingInterventions.length = 0;
          renderPendingActions();
        }
      }).catch((error) => setStatus(error.message));
    };

    byId('noopButton').onclick = () => {
      stepManual({ interventions: [] }).catch((error) => setStatus(error.message));
    };

    byId('baselineButton').onclick = () => {
      baselineStep().catch((error) => setStatus(error.message));
    };

    byId('llmButton').onclick = () => {
      llmStep().catch((error) => setStatus(error.message));
    };

    byId('playLlmButton').onclick = () => {
      autoplayLlm().catch((error) => setStatus(error.message));
    };

    byId('playBaselineButton').onclick = () => {
      autoplayBaseline().catch((error) => setStatus(error.message));
    };

    byId('stopPlayButton').onclick = () => {
      autoplay = false;
      setStatus('Autoplay stopped.', false);
    };

    resetDashboard().catch((error) => setStatus(error.message));
  </script>
</body>
</html>
"""
