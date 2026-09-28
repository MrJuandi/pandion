"""Standalone interactive browser viewer for Nightjar simulation and policy replays.

Generates self-contained, dependency-free HTML5 visualization artifacts for auditing
fire spread, observation deliveries, aircraft trajectories, and drop decisions.
Supports single-policy exploration and multi-policy synchronized side-by-side comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

VIEWER_VERSION = "nightjar-viewer-0.1"


def build_viewer_html(replays: Sequence[dict], title: str = "Nightjar Replay Viewer") -> str:
    """Build a complete self-contained HTML page visualizing one or more replays."""
    if not replays:
        raise ValueError("at least one replay is required to generate viewer HTML")

    replays_json = json.dumps(list(replays), sort_keys=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --fire: #ef4444;
      --fire-glow: rgba(239, 68, 68, 0.4);
      --water: #3b82f6;
      --unburned: #334155;
      --nonburnable: #020617;
      --candidate: #eab308;
      --observed-border: #10b981;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background-color: var(--card-bg);
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    header h1 {{
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge {{
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 9999px;
      background-color: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .policy-tabs {{
      display: flex;
      gap: 6px;
      background-color: var(--bg);
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }}
    .tab-btn {{
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 6px 12px;
      font-size: 0.85rem;
      font-weight: 500;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .tab-btn:hover {{ color: var(--text); }}
    .tab-btn.active {{
      background-color: var(--card-bg);
      color: var(--accent);
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }}
    .mode-toggle {{
      background-color: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--card-border);
      padding: 6px 12px;
      font-size: 0.85rem;
      border-radius: 6px;
      cursor: pointer;
    }}
    .mode-toggle.active {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    main {{
      display: flex;
      flex: 1;
      overflow: hidden;
    }}
    .sidebar {{
      width: 320px;
      background-color: var(--card-bg);
      border-right: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 16px;
      gap: 16px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }}
    .metric-card {{
      background-color: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px;
    }}
    .metric-title {{
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-value {{
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 4px;
    }}
    .metric-unit {{
      font-size: 0.75rem;
      font-weight: 400;
      color: var(--text-muted);
    }}
    .section-title {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .decision-log {{
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 320px;
    }}
    .decision-item {{
      background-color: rgba(15, 23, 42, 0.4);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .decision-item:hover {{
      background-color: rgba(56, 189, 248, 0.1);
      border-color: var(--accent);
    }}
    .decision-item.current {{
      border-color: var(--accent);
      background-color: rgba(56, 189, 248, 0.15);
    }}
    .decision-time {{
      font-weight: 600;
      color: var(--accent);
    }}
    .decision-order {{
      color: var(--text);
      margin-top: 2px;
      word-break: break-all;
    }}
    .view-content {{
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 20px;
      gap: 16px;
      overflow-y: auto;
      align-items: center;
    }}
    .grids-container {{
      display: flex;
      gap: 24px;
      justify-content: center;
      flex-wrap: wrap;
      width: 100%;
    }}
    .grid-view-panel {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    .grid-title {{
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 12px;
    }}
    .canvas-wrapper {{
      position: relative;
      width: 440px;
      height: 440px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      background-color: #0b0f19;
      overflow: hidden;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .tooltip {{
      position: absolute;
      background-color: rgba(15, 23, 42, 0.95);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.75rem;
      pointer-events: none;
      z-index: 100;
      display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      white-space: nowrap;
    }}
    .playback-controls {{
      width: 100%;
      max-width: 920px;
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .slider-row {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    .time-display {{
      font-family: monospace;
      font-size: 1rem;
      font-weight: 700;
      color: var(--accent);
      min-width: 130px;
    }}
    .slider {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 4px rgba(56, 189, 248, 0.8);
    }}
    .buttons-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .btn-group {{
      display: flex;
      gap: 6px;
    }}
    .ctrl-btn {{
      background-color: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .ctrl-btn:hover {{
      background-color: rgba(56, 189, 248, 0.2);
      border-color: var(--accent);
    }}
    .ctrl-btn.active {{
      background-color: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 10px 14px;
      background-color: rgba(15, 23, 42, 0.6);
      border-radius: 8px;
      font-size: 0.75rem;
      border: 1px solid var(--card-border);
      width: 100%;
      max-width: 920px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-color {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
    }}
    .toggles-row {{
      display: flex;
      gap: 12px;
      font-size: 0.78rem;
      color: var(--text-muted);
    }}
    .toggles-row label {{
      display: flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <header>
    <h1>
      <span>🔥 Nightjar Replay Viewer</span>
      <span class="badge" id="header-scenario-id">Loading...</span>
    </h1>
    <div style="display: flex; gap: 10px; align-items: center;">
      <div class="policy-tabs" id="policy-tabs"></div>
      <button class="mode-toggle" id="split-toggle-btn" style="display: none;">Side-by-side</button>
    </div>
  </header>

  <main>
    <div class="sidebar">
      <div class="section-title">Replay Summary</div>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-title">Burned Area</div>
          <div class="metric-value"><span id="m-area">-</span> <span class="metric-unit">ha</span></div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Water Dropped</div>
          <div class="metric-value"><span id="m-water">-</span> <span class="metric-unit">L</span></div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Flight Time</div>
          <div class="metric-value"><span id="m-flight">-</span> <span class="metric-unit">s</span></div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Drops / Rejections</div>
          <div class="metric-value"><span id="m-drops">-</span> / <span id="m-rejected">-</span></div>
        </div>
      </div>

      <div class="section-title">Aircraft Telemetry</div>
      <div class="metric-card">
        <div class="metric-title">Tank Water</div>
        <div class="metric-value"><span id="m-tank">-</span> <span class="metric-unit">L</span></div>
        <div style="background: #334155; height: 4px; border-radius: 2px; margin-top: 6px;">
          <div id="tank-bar" style="background: var(--water); height: 100%; width: 100%; border-radius: 2px;"></div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">
          Status: <strong id="m-phase" style="color: var(--text);">-</strong>
        </div>
      </div>

      <div class="section-title">Decision History</div>
      <div class="decision-log" id="decision-log"></div>
    </div>

    <div class="view-content">
      <div class="grids-container" id="grids-container"></div>

      <div class="playback-controls">
        <div class="slider-row">
          <div class="time-display" id="time-display">00:00 / 20:00</div>
          <input type="range" class="slider" id="time-slider" min="0" max="1200" value="0" step="1">
        </div>
        <div class="buttons-row">
          <div class="btn-group">
            <button class="ctrl-btn" id="btn-reset">⏮ Reset</button>
            <button class="ctrl-btn" id="btn-prev">◀ Step -60s</button>
            <button class="ctrl-btn" id="btn-play" style="min-width: 70px;">▶ Play</button>
            <button class="ctrl-btn" id="btn-next">Step +60s ▶</button>
          </div>
          <div class="btn-group">
            <button class="ctrl-btn speed-btn" data-speed="1">1x</button>
            <button class="ctrl-btn speed-btn" data-speed="2">2x</button>
            <button class="ctrl-btn speed-btn active" data-speed="5">5x</button>
            <button class="ctrl-btn speed-btn" data-speed="10">10x</button>
            <button class="ctrl-btn speed-btn" data-speed="25">25x</button>
          </div>
          <div class="toggles-row">
            <label><input type="checkbox" id="chk-obs" checked> Delivered Obs</label>
            <label><input type="checkbox" id="chk-cand" checked> Candidates</label>
            <label><input type="checkbox" id="chk-trail" checked> Trail</label>
          </div>
        </div>
      </div>

      <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #334155;"></div> Unburned Fuel</div>
        <div class="legend-item"><div class="legend-color" style="background: #ef4444;"></div> Fire Reached</div>
        <div class="legend-item"><div class="legend-color" style="background: #3b82f6;"></div> Water Barrier</div>
        <div class="legend-item"><div class="legend-color" style="background: #020617; border: 1px solid #475569;"></div> Unburnable (NONE)</div>
        <div class="legend-item"><div class="legend-color" style="border: 2px dashed #10b981; background: transparent;"></div> Observed Burning</div>
        <div class="legend-item"><div class="legend-color" style="border: 2px solid #eab308; background: transparent;"></div> Valid Target</div>
        <div class="legend-item"><div class="legend-color" style="background: #38bdf8; border-radius: 50%;"></div> Aircraft</div>
        <div class="legend-item"><div class="legend-color" style="background: #06b6d4;"></div> Water Station</div>
      </div>
    </div>
  </main>

  <div class="tooltip" id="tooltip"></div>

  <script>
    const REPLAYS = {replays_json};
    let currentReplayIdx = 0;
    let isSideBySide = false;
    let currentTime = 0;
    let isPlaying = false;
    let playbackSpeed = 5;
    let animFrame = null;
    let lastAnimTimestamp = null;

    const dom = {{
      scenarioId: document.getElementById('header-scenario-id'),
      policyTabs: document.getElementById('policy-tabs'),
      splitToggleBtn: document.getElementById('split-toggle-btn'),
      timeDisplay: document.getElementById('time-display'),
      timeSlider: document.getElementById('time-slider'),
      btnPlay: document.getElementById('btn-play'),
      btnReset: document.getElementById('btn-reset'),
      btnPrev: document.getElementById('btn-prev'),
      btnNext: document.getElementById('btn-next'),
      speedBtns: document.querySelectorAll('.speed-btn'),
      decisionLog: document.getElementById('decision-log'),
      gridsContainer: document.getElementById('grids-container'),
      mArea: document.getElementById('m-area'),
      mWater: document.getElementById('m-water'),
      mFlight: document.getElementById('m-flight'),
      mDrops: document.getElementById('m-drops'),
      mRejected: document.getElementById('m-rejected'),
      mTank: document.getElementById('m-tank'),
      tankBar: document.getElementById('tank-bar'),
      mPhase: document.getElementById('m-phase'),
      tooltip: document.getElementById('tooltip'),
      chkObs: document.getElementById('chk-obs'),
      chkCand: document.getElementById('chk-cand'),
      chkTrail: document.getElementById('chk-trail'),
    }};

    function init() {{
      const baseReplay = REPLAYS[0];
      dom.scenarioId.textContent = baseReplay.scenario_id || "scenario";
      const horizon = baseReplay.horizon_s || 1200;
      dom.timeSlider.max = horizon;

      // Policy Tabs
      if (REPLAYS.length > 1) {{
        dom.splitToggleBtn.style.display = "block";
      }}
      REPLAYS.forEach((rep, idx) => {{
        const btn = document.createElement('button');
        btn.className = `tab-btn ${{idx === 0 ? 'active' : ''}}`;
        btn.textContent = rep.policy || `Policy ${{idx + 1}}`;
        btn.onclick = () => selectPolicy(idx);
        dom.policyTabs.appendChild(btn);
      }});

      dom.splitToggleBtn.onclick = () => {{
        isSideBySide = !isSideBySide;
        dom.splitToggleBtn.classList.toggle('active', isSideBySide);
        setupGrids();
        render();
      }};

      dom.timeSlider.oninput = (e) => {{
        currentTime = parseFloat(e.target.value);
        render();
      }};

      dom.btnPlay.onclick = togglePlay;
      dom.btnReset.onclick = () => {{
        currentTime = 0;
        dom.timeSlider.value = 0;
        render();
      }};
      dom.btnPrev.onclick = () => stepTime(-60);
      dom.btnNext.onclick = () => stepTime(60);

      dom.speedBtns.forEach(btn => {{
        btn.onclick = () => {{
          dom.speedBtns.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          playbackSpeed = parseFloat(btn.dataset.speed);
        }};
      }});

      dom.chkObs.onchange = render;
      dom.chkCand.onchange = render;
      dom.chkTrail.onchange = render;

      setupGrids();
      updateSidebar();
      render();
    }}

    function selectPolicy(idx) {{
      currentReplayIdx = idx;
      document.querySelectorAll('.tab-btn').forEach((b, i) => {{
        b.classList.toggle('active', i === idx);
      }});
      if (isSideBySide) {{
        isSideBySide = false;
        dom.splitToggleBtn.classList.remove('active');
      }}
      setupGrids();
      updateSidebar();
      render();
    }}

    function setupGrids() {{
      dom.gridsContainer.innerHTML = "";
      const toRender = isSideBySide ? REPLAYS : [REPLAYS[currentReplayIdx]];
      toRender.forEach((replay, viewIdx) => {{
        const panel = document.createElement('div');
        panel.className = 'grid-view-panel';
        panel.innerHTML = `
          <div class="grid-title">${{replay.policy || 'Policy'}} (${{replay.affected_area_ha}} ha)</div>
          <div class="canvas-wrapper">
            <canvas id="canvas-${{viewIdx}}" width="440" height="440"></canvas>
          </div>
        `;
        dom.gridsContainer.appendChild(panel);

        const canvas = panel.querySelector('canvas');
        canvas.onmousemove = (e) => handleCanvasHover(e, canvas, replay);
        canvas.onmouseleave = () => {{ dom.tooltip.style.display = "none"; }};
      }});
    }}

    function formatTime(s) {{
      const mins = Math.floor(s / 60);
      const secs = Math.floor(s % 60);
      return `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      dom.btnPlay.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
      if (isPlaying) {{
        lastAnimTimestamp = performance.now();
        animFrame = requestAnimationFrame(playLoop);
      }} else if (animFrame) {{
        cancelAnimationFrame(animFrame);
      }}
    }}

    function playLoop(timestamp) {{
      if (!isPlaying) return;
      const dt = (timestamp - lastAnimTimestamp) / 1000;
      lastAnimTimestamp = timestamp;

      currentTime += dt * playbackSpeed;
      const maxTime = parseFloat(dom.timeSlider.max);
      if (currentTime >= maxTime) {{
        currentTime = maxTime;
        togglePlay();
      }}
      dom.timeSlider.value = currentTime;
      render();
      if (isPlaying) {{
        animFrame = requestAnimationFrame(playLoop);
      }}
    }}

    function stepTime(delta) {{
      currentTime = Math.max(0, Math.min(parseFloat(dom.timeSlider.max), currentTime + delta));
      dom.timeSlider.value = currentTime;
      render();
    }}

    function updateSidebar() {{
      const rep = REPLAYS[currentReplayIdx];
      dom.mArea.textContent = rep.affected_area_ha?.toFixed(2) || "0.00";
      dom.mWater.textContent = rep.aircraft_metrics?.water_dropped_l?.toLocaleString() || "0";
      dom.mFlight.textContent = rep.aircraft_metrics?.flight_s?.toFixed(1) || "0.0";
      dom.mDrops.textContent = rep.drops?.length || "0";
      dom.mRejected.textContent = rep.aircraft_metrics?.rejected_orders || "0";

      // Build decision list
      dom.decisionLog.innerHTML = "";
      (rep.decisions || []).forEach((dec, idx) => {{
        const item = document.createElement('div');
        item.className = 'decision-item';
        item.dataset.time = dec.time_s;
        const targetStr = dec.order.target_m ? ` [${{dec.order.target_m[0].toFixed(0)}}, ${{dec.order.target_m[1].toFixed(0)}}]` : '';
        item.innerHTML = `
          <div class="decision-time">t = ${{dec.time_s}}s (${{formatTime(dec.time_s)}})</div>
          <div class="decision-order">${{dec.order.type}}${{targetStr}}</div>
        `;
        item.onclick = () => {{
          currentTime = dec.time_s;
          dom.timeSlider.value = currentTime;
          render();
        }};
        dom.decisionLog.appendChild(item);
      }});
    }}

    function render() {{
      const maxTime = parseFloat(dom.timeSlider.max);
      dom.timeDisplay.textContent = `${{formatTime(currentTime)}} / ${{formatTime(maxTime)}}`;

      // Update decision log highlight
      const rep = REPLAYS[currentReplayIdx];
      const items = dom.decisionLog.querySelectorAll('.decision-item');
      items.forEach(item => {{
        const decTime = parseFloat(item.dataset.time);
        item.classList.toggle('current', Math.abs(currentTime - decTime) < 30);
      }});

      // Update Aircraft Tank and Phase telemetry
      let latestDec = rep.decisions ? rep.decisions[0] : null;
      (rep.decisions || []).forEach(dec => {{
        if (dec.time_s <= currentTime) latestDec = dec;
      }});
      if (latestDec && latestDec.observation && latestDec.observation.aircraft) {{
        const ac = latestDec.observation.aircraft;
        dom.mTank.textContent = ac.water_l?.toLocaleString() || "0";
        const pct = Math.max(0, Math.min(100, (ac.water_l / 6137) * 100));
        dom.tankBar.style.width = `${{pct}}%`;
        dom.mPhase.textContent = (ac.phase || "idle").toUpperCase();
      }}

      // Render canvases
      const toRender = isSideBySide ? REPLAYS : [REPLAYS[currentReplayIdx]];
      toRender.forEach((replay, viewIdx) => {{
        const canvas = document.getElementById(`canvas-${{viewIdx}}`);
        if (canvas) drawReplayOnCanvas(canvas, replay);
      }});
    }}

    function drawReplayOnCanvas(canvas, replay) {{
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const rows = 11;
      const cols = 11;
      const cellW = w / cols;
      const cellH = h / rows;

      // Fuel & arrival grid
      const arrivals = replay.arrival_time_s || [];
      const firstDec = replay.decisions && replay.decisions[0];
      const fuel = firstDec?.observation?.fuel_model || [];
      const waterBarrier = replay.water_barrier_cells || [];

      // Find drops up to currentTime
      const activeDrops = (replay.drops || []).filter(d => d.time_s <= currentTime);
      const droppedCells = new Set();
      activeDrops.forEach(d => {{
        (d.newly_blocked_cells || []).forEach(cell => droppedCells.add(`${{cell[0]}},${{cell[1]}}`));
      }});

      // Delivered observations at current time
      const latestDec = (replay.decisions || []).reduce((prev, curr) => curr.time_s <= currentTime ? curr : prev, replay.decisions[0]);
      const deliveredObs = latestDec?.observation?.fire_reports || [];
      const observedBurning = new Set();
      deliveredObs.forEach(cell => observedBurning.add(`${{cell[0]}},${{cell[1]}}`));

      const candidates = new Set();
      if (dom.chkCand.checked && latestDec?.candidate_cells) {{
        latestDec.candidate_cells.forEach(c => candidates.add(`${{c[0]}},${{c[1]}}`));
      }}

      // Draw grid cells (row 10 top, row 0 bottom; col 0 left, col 10 right)
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const canvasX = c * cellW;
          const canvasY = (rows - 1 - r) * cellH;

          const isFuelNone = fuel[r] && fuel[r][c] === 'NONE';
          const arrTime = arrivals[r] ? arrivals[r][c] : null;
          const hasFireReached = arrTime !== null && arrTime <= currentTime;
          const isBlocked = droppedCells.has(`${{r}},${{c}}`);

          // Background fill
          if (isFuelNone) {{
            ctx.fillStyle = "#020617";
          }} else if (isBlocked) {{
            ctx.fillStyle = "#1e3a8a"; // Deep water blue
          }} else if (hasFireReached) {{
            ctx.fillStyle = "#dc2626"; // Vibrant fire red
          }} else {{
            ctx.fillStyle = "#334155"; // Neutral unburned fuel
          }}
          ctx.fillRect(canvasX, canvasY, cellW, cellH);

          // Grid line
          ctx.strokeStyle = "#1e293b";
          ctx.lineWidth = 1;
          ctx.strokeRect(canvasX, canvasY, cellW, cellH);

          // Water barrier pattern
          if (isBlocked) {{
            ctx.fillStyle = "rgba(59, 130, 246, 0.4)";
            ctx.fillRect(canvasX + 4, canvasY + 4, cellW - 8, cellH - 8);
          }}

          // Delivered Observation border
          if (dom.chkObs.checked && observedBurning.has(`${{r}},${{c}}`)) {{
            ctx.strokeStyle = "#10b981";
            ctx.lineWidth = 2.5;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(canvasX + 2, canvasY + 2, cellW - 4, cellH - 4);
            ctx.setLineDash([]);
          }}

          // Candidate target marker
          if (candidates.has(`${{r}},${{c}}`)) {{
            ctx.strokeStyle = "#eab308";
            ctx.lineWidth = 2;
            ctx.strokeRect(canvasX + 4, canvasY + 4, cellW - 8, cellH - 8);
          }}
        }}
      }}

      // Draw Water Source Station
      const waterSource = latestDec?.observation?.water_source_m;
      if (waterSource) {{
        const wsX = (waterSource[0] / 500) * (w - cellW) + cellW / 2;
        const wsY = (1 - waterSource[1] / 500) * (h - cellH) + cellH / 2;
        ctx.fillStyle = "#06b6d4";
        ctx.beginPath();
        ctx.arc(wsX, wsY, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}

      // Interpolate Aircraft position along drop/refill transitions
      const acState = getInterpolatedAircraft(replay, currentTime);
      if (acState) {{
        const acX = (acState.x / 500) * (w - cellW) + cellW / 2;
        const acY = (1 - acState.y / 500) * (h - cellH) + cellH / 2;

        // Draw flight trail
        if (dom.chkTrail.checked && acState.trail) {{
          ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          acState.trail.forEach((pt, i) => {{
            const px = (pt[0] / 500) * (w - cellW) + cellW / 2;
            const py = (1 - pt[1] / 500) * (h - cellH) + cellH / 2;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }});
          ctx.lineTo(acX, acY);
          ctx.stroke();
        }}

        // Aircraft glyph
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(acX, acY, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
    }}

    function getInterpolatedAircraft(replay, time) {{
      const decisions = replay.decisions || [];
      if (!decisions.length) return null;
      const initialPos = decisions[0].observation?.aircraft?.position_m || [0, 0];
      const trail = [initialPos];

      let lastPos = initialPos;
      let lastTime = 0;

      for (let i = 0; i < decisions.length; i++) {{
        const dec = decisions[i];
        if (dec.time_s > time) break;
        const curPos = dec.observation?.aircraft?.position_m || lastPos;
        trail.push(curPos);
        lastPos = curPos;
        lastTime = dec.time_s;
      }}

      // Check drop events
      const drops = (replay.drops || []).filter(d => d.time_s <= time);
      drops.forEach(d => {{
        if (d.position_m) trail.push(d.position_m);
      }});

      return {{ x: lastPos[0], y: lastPos[1], trail: trail }};
    }}

    function handleCanvasHover(e, canvas, replay) {{
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const rows = 11;
      const cols = 11;
      const cellW = canvas.width / cols;
      const cellH = canvas.height / rows;

      const c = Math.floor(x / cellW);
      const r = rows - 1 - Math.floor(y / cellH);

      if (r >= 0 && r < rows && c >= 0 && c < cols) {{
        const arrTime = replay.arrival_time_s ? replay.arrival_time_s[r][c] : null;
        const arrStr = arrTime !== null ? `${{arrTime.toFixed(1)}}s` : 'Never reached';
        const isBurning = arrTime !== null && arrTime <= currentTime;
        const fuel = replay.decisions?.[0]?.observation?.fuel_model?.[r]?.[c] || 'FM1';

        dom.tooltip.style.display = 'block';
        dom.tooltip.style.left = `${{e.pageX + 15}}px`;
        dom.tooltip.style.top = `${{e.pageY + 10}}px`;
        dom.tooltip.innerHTML = `
          <strong>Cell [row: ${{r}}, col: ${{c}}]</strong><br>
          Coords: (${{c * 50}}m, ${{r * 50}}m)<br>
          Fuel: ${{fuel}}<br>
          Arrival Time: ${{arrStr}}<br>
          State @ t=${{currentTime.toFixed(0)}}s: <strong>${{isBurning ? '🔥 BURNING' : 'UNBURNED'}}</strong>
        `;
      }}
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    return html


def write_viewer_html(replays: Sequence[dict], output_path: Path, title: str | None = None) -> None:
    """Generate and write viewer HTML file to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_title = title or (replays[0].get("scenario_id", "Nightjar") + " - Replay Viewer")
    html_content = build_viewer_html(replays, title=page_title)
    output_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path, help="Paths to replay JSON files")
    parser.add_argument("--output", type=Path, default=Path("runs/viewer.html"), help="Destination HTML file")
    parser.add_argument("--title", type=str, default=None, help="Custom viewer page title")
    args = parser.parse_args()

    replays = []
    for replay_path in args.replays:
        if not replay_path.exists():
            parser.error(f"replay file not found: {replay_path}")
        replays.append(json.loads(replay_path.read_text()))

    write_viewer_html(replays, args.output, title=args.title)
    print(f"Generated Nightjar Replay Viewer at: {args.output}")


if __name__ == "__main__":
    main()
