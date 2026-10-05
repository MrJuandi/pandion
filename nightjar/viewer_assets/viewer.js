(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const Fire = window.NightjarFire;
  const STATIC = 'Cartografía · sin episodio';
  const finite = (n) => typeof n === 'number' && Number.isFinite(n);
  const clampNum = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const fmt = (n, digits = 1) => finite(n) ? n.toLocaleString('es-ES', {maximumFractionDigits: digits}) : '—';
  const clock = (s) => `${String(Math.floor(Math.max(0, s) / 60)).padStart(2, '0')}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, '0')}`;
  const esc = (s) => String(s ?? '—');
  const policyLabel = (r) => ({heuristic: 'Heurística · ataque directo', containment_heuristic: 'Heurística · contención', no_intervention: 'Sin intervención', random_valid: 'Aleatoria válida', ppo: 'PPO'}[r.policy] || r.policy || '—');
  const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSO', 'SO', 'OSO', 'O', 'ONO', 'NO', 'NNO'];
  const compass = (deg) => COMPASS[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
  const SERIES = ['#38bdf8', '#a78bfa', '#f472b6', '#94a3b8', '#facc15'];

  const ui = {
    scenario: $('scenario-select'), policy: $('policy-select'), comparePolicy: $('compare-policy-select'), compare: $('compare-toggle'),
    slider: $('time-slider'), play: $('btn-play'), panels: $('map-panels'), stage: $('map-stage'), tooltip: $('map-tooltip'),
    timeline: $('timeline'), state: $('state-select'), style: $('terrain-style'), relief: $('relief-scale'), speed: $('speed-select'),
    environment: $('condiciones'), comparisonResults: $('comparison-results'), status: $('renderer-status'), decisions: $('decision-list')
  };
  const scenarios = new Map();
  for (const replay of REPLAYS) {
    const key = String(replay.scenario_id ?? 'escenario');
    if (!scenarios.has(key)) scenarios.set(key, []);
    scenarios.get(key).push(replay);
  }
  let activeScenario = [...scenarios.keys()][0];
  let activeReplay = 0, comparisonReference = 1, compare = false;
  let now = 0, playing = false, speed = Number(ui.speed.value) || 30, lastFrame = 0;
  let camMode = '3d', rendererFailed = false, syncing = false, hudTimer = 0;
  let activeView = ['mapa', 'condiciones', 'comparacion', 'datos'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'mapa';
  let views = [];
  const cache = new WeakMap();

  const currentGroup = () => scenarios.get(activeScenario) || [];
  const episodeGroup = () => currentGroup().filter((r) => r.policy !== STATIC);
  const current = () => currentGroup()[activeReplay] || currentGroup()[0];
  const comparisonGroup = () => [current(), currentGroup()[comparisonReference]].filter(Boolean);
  const displayGroup = () => compare ? comparisonGroup() : [current()];
  const checked = (id) => $(id).checked;
  const isStatic = (r) => r?.policy === STATIC;

  // ---------------------------------------------------------------------------
  // Derived, cached replay facts (all from recorded data)
  // ---------------------------------------------------------------------------
  function facts(r) {
    if (cache.has(r)) return cache.get(r);
    const {rows, cols, cell_size_m: size} = r.grid;
    const elev = r.elevation_m.flat().filter(finite);
    const arrivals = r.arrival_time_s.flat().filter(finite).sort((a, b) => a - b);
    const decisions = Array.isArray(r.decisions) ? r.decisions : [];
    let track = Array.isArray(r.aircraft_track) && r.aircraft_track.length
      ? r.aircraft_track.filter((p) => Array.isArray(p.position_m) && finite(Number(p.time_s))).map((p) => ({t: Number(p.time_s), p: [Number(p.position_m[0]), Number(p.position_m[1])]}))
      : decisions.filter((d) => Array.isArray(d.observation?.aircraft?.position_m)).map((d) => ({t: Number(d.time_s), p: d.observation.aircraft.position_m.map(Number)}));
    track = track.filter((s) => s.p.every(finite)).sort((a, b) => a.t - b.t);
    // Water released per recorded drop: tank level before minus after the completed DROP order.
    const metrics = r.aircraft_metrics || {}, totalWater = metrics.water_dropped_l ?? metrics.water_dropped_liters ?? metrics.total_water_dropped_l;
    const capacity = Math.max(0, ...decisions.map((d) => Number(d.aircraft_water_l)).filter(finite)) || null;
    const releases = [];
    let previous = capacity;
    for (const d of decisions) {
      const water = Number(d.aircraft_water_l), type = orderType(d);
      if (type === 'DROP' && finite(water) && finite(previous)) releases.push({t: Number(d.completed_at_s ?? d.time_s), litres: Math.max(0, previous - water)});
      if (finite(water)) previous = water;
    }
    const drops = (r.drops || []).filter((d) => finite(Number(d.time_s))).map((d) => ({...d, time_s: Number(d.time_s), cells: d.newly_blocked_cells || d.treated_cells || []}));
    if (!releases.length && finite(totalWater) && drops.length) for (const d of drops) releases.push({t: d.time_s, litres: totalWater / drops.length});
    const refills = [], seen = new Set();
    for (const d of decisions) {
      if (orderType(d) !== 'RETURN_TO_REFILL') continue;
      const p = samplePosition(track, Number(d.completed_at_s ?? d.time_s));
      if (!p) continue;
      const key = `${Math.round(p[0] / 25)},${Math.round(p[1] / 25)}`;
      if (!seen.has(key)) { seen.add(key); refills.push(p); }
    }
    let water = [], wx = 0, wy = 0;
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) if (r.water_mask?.[row]?.[col] === true) { water.push([row, col]); wx += col; wy += row; }
    let waterLabel = null;
    if (water.length) {
      const cx = wx / water.length, cy = wy / water.length;
      const best = water.reduce((a, b) => ((b[1] - cx) ** 2 + (b[0] - cy) ** 2 < (a[1] - cx) ** 2 + (a[0] - cy) ** 2 ? b : a));
      waterLabel = [(best[1] + 0.5) * size, (best[0] + 0.5) * size];
    }
    const f = {rows, cols, size, cellHa: size * size / 1e4, minElev: elev.length ? Math.min(...elev) : null, maxElev: elev.length ? Math.max(...elev) : null,
      arrivals, track, capacity, releases, drops, refills, waterLabel};
    cache.set(r, f); return f;
  }
  function orderType(d) { const o = d.order || d.action || {}; return o.type || o.action || o.kind || null; }
  function samplePosition(track, t) {
    if (!track.length) return null;
    if (t <= track[0].t) return track[0].p.slice();
    for (let i = 1; i < track.length; i++) {
      const a = track[i - 1], b = track[i];
      if (t <= b.t) { const f = b.t > a.t ? (t - a.t) / (b.t - a.t) : 1; return [a.p[0] + (b.p[0] - a.p[0]) * f, a.p[1] + (b.p[1] - a.p[1]) * f]; }
    }
    return track.at(-1).p.slice();
  }
  function areaAt(r, t) { const a = facts(r).arrivals; let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m + 1; else hi = m; } return lo * facts(r).cellHa; }
  function episodeTime(r) { return Math.min(now, Number(r.horizon_s) || 0); }

  function aircraftAt(r, t) {
    const f = facts(r);
    if (isStatic(r) || !f.track.length) return null;
    const position = samplePosition(f.track, t);
    // Heading from the active track segment; keep the last movement when stationary.
    let heading = null;
    for (let i = 1; i < f.track.length; i++) {
      const a = f.track[i - 1], b = f.track[i], dx = b.p[0] - a.p[0], dy = b.p[1] - a.p[1];
      if (Math.hypot(dx, dy) < 1e-6) continue;
      if (a.t <= t || !heading) heading = [dx, dy];
      if (b.t >= t && a.t <= t) break;
    }
    heading = heading || [1, 0];
    const len = Math.hypot(...heading) || 1;
    const decisions = r.decisions || [];
    let issued = null, completed = null;
    for (const d of decisions) {
      if (Number(d.time_s) <= t) issued = d;
      if (Number(d.completed_at_s ?? d.time_s) <= t && finite(Number(d.aircraft_water_l))) completed = d;
    }
    const type = issued ? orderType(issued) : null, inProgress = issued && t < Number(issued.completed_at_s ?? issued.time_s);
    const phase = !issued ? 'En espera' : type === 'DROP' ? (inProgress ? 'Hacia descarga' : 'Descarga') : type === 'RETURN_TO_REFILL' ? (inProgress ? 'Hacia recarga' : 'Recargando') : type === 'HOLD' ? 'En espera' : String(type || 'Orden');
    const water = completed ? Number(completed.aircraft_water_l) : f.capacity;
    const accent = type === 'DROP' ? '#38bdf8' : type === 'RETURN_TO_REFILL' ? '#a78bfa' : '#fbbf24';
    return {position, heading: [heading[0] / len, heading[1] / len], phase, water, capacity: f.capacity, accent,
      labelLines: ['NJ-01 · CL-415 ref.', `${phase}${finite(water) ? ` · ${fmt(water, 0)} L` : ''}`]};
  }
  function trailAt(r, t) {
    const f = facts(r); if (isStatic(r) || !f.track.length) return [];
    const out = [];
    for (let i = 0; i < f.track.length && f.track[i].t <= t; i++) {
      const s = f.track[i]; out.push(s.p);
      const next = f.track[i + 1];
      if (next && next.t > t) break;
    }
    const p = samplePosition(f.track, t); if (p) out.push(p);
    return out;
  }
  function windAt(r, t) {
    const env = r.environment || {};
    let vector = null;
    if (env.weather_mode === 'station_intervals' || env.weather_mode === 'synthetic_intervals') {
      const forcing = Array.isArray(env.forcing) ? env.forcing : [];
      const active = forcing.find((f) => Number(f.start_s) <= t && t < Number(f.end_s)) || forcing.filter((f) => Number(f.end_s) <= t).at(-1) || forcing[0];
      vector = active?.midflame_wind_to_m_s;
    } else vector = r.metadata?.wind_to_m_s || r.wind_to_m_s || null;
    const w = vectorValue(vector);
    return w ? {east: Number(vector[0]), north: Number(vector[1]), speed: w.speed, degrees: w.degrees} : null;
  }

  // ---------------------------------------------------------------------------
  // Initialisation and controls
  // ---------------------------------------------------------------------------
  function init() {
    for (const [id, group] of scenarios) {
      const option = document.createElement('option'); option.value = id; option.textContent = group[0].metadata?.name || id; ui.scenario.append(option);
    }
    ui.scenario.addEventListener('change', () => { activeScenario = ui.scenario.value; activeReplay = 0; comparisonReference = 1; now = 0; compare = false; if (activeView === 'comparacion') activeView = 'mapa'; stop(); fillPolicies(); updateCompare(); refresh(); });
    ui.policy.addEventListener('change', () => { activeReplay = Number(ui.policy.value); updateCompare(); refresh(); });
    ui.comparePolicy.addEventListener('change', () => { comparisonReference = Number(ui.comparePolicy.value); refresh(); });
    ui.compare.addEventListener('click', () => { if (ui.compare.disabled) return; setView(compare ? 'mapa' : 'comparacion'); });
    ui.slider.addEventListener('input', () => { now = Number(ui.slider.value); updateTime(); });
    ui.play.addEventListener('click', togglePlay);
    $('btn-reset').addEventListener('click', () => { stop(); now = 0; updateTime(); });
    ui.speed.addEventListener('change', () => { speed = Number(ui.speed.value) || 30; });
    for (const id of ['layer-front', 'layer-burn', 'layer-observations', 'layer-drops', 'layer-flight', 'layer-contours', 'layer-labels', 'layer-grid', 'layer-smoke', 'state-select', 'terrain-style']) $(id).addEventListener('change', updateTime);
    ui.relief.addEventListener('change', refresh);
    $('view-3d').addEventListener('click', () => setCamMode('3d'));
    $('view-2d').addEventListener('click', () => setCamMode('2d'));
    $('cam-chase').addEventListener('click', () => setCamMode('chase'));
    $('cam-front').addEventListener('click', () => setCamMode('front'));
    $('camera-reset').addEventListener('click', resetCamera);
    $('zoom-in').addEventListener('click', () => views.forEach((v) => v.renderer.zoomBy(1.25)));
    $('zoom-out').addEventListener('click', () => views.forEach((v) => v.renderer.zoomBy(1 / 1.25)));
    $('export-png').addEventListener('click', exportPng);
    document.querySelectorAll('.nav-link,.brand').forEach((link) => link.addEventListener('click', (e) => { e.preventDefault(); setView(link.hash.slice(1)); }));
    document.addEventListener('keydown', globalKey);
    if (window.matchMedia('(max-width: 640px)').matches) $('legend').open = false;
    fillPolicies(); updateCompare(); setView(activeView);
    requestAnimationFrame(loop);
  }
  function fillPolicies() {
    ui.policy.replaceChildren();
    currentGroup().forEach((r, i) => { const o = document.createElement('option'); o.value = String(i); o.textContent = policyLabel(r); ui.policy.append(o); });
    ui.policy.value = '0'; activeReplay = 0;
    $('policy-count').textContent = String(currentGroup().length);
  }
  function updateCompare() {
    if (episodeGroup().length < 2) compare = false;
    const peers = currentGroup().map((r, i) => ({r, i})).filter(({r, i}) => i !== activeReplay && !isStatic(r));
    if (!peers.some(({i}) => i === comparisonReference)) comparisonReference = peers[0]?.i ?? activeReplay;
    ui.comparePolicy.replaceChildren();
    peers.forEach(({r, i}) => { const o = document.createElement('option'); o.value = String(i); o.textContent = policyLabel(r); ui.comparePolicy.append(o); });
    ui.comparePolicy.value = String(comparisonReference);
    $('compare-policy-field').hidden = !compare;
    ui.compare.setAttribute('aria-pressed', String(compare));
    ui.compare.disabled = episodeGroup().length < 2;
    ui.compare.querySelector('b').textContent = compare ? 'Salir de la comparación' : 'Comparar dos políticas';
    ui.compare.querySelector('small').textContent = compare ? 'Dos políticas · reloj y cámara compartidos' : 'Selecciona una referencia';
  }
  function horizon() { return compare ? Math.max(0, ...comparisonGroup().map((r) => Number(r.horizon_s) || 0)) : Number(current()?.horizon_s) || 0; }
  function setView(view) {
    activeView = view; stop();
    if (view === 'comparacion') compare = episodeGroup().length > 1;
    else if (view === 'mapa') compare = false;
    updateCompare();
    document.querySelector('.workspace').hidden = !['mapa', 'comparacion'].includes(activeView);
    for (const id of ['condiciones', 'datos']) $(id).hidden = activeView !== id;
    const slot = activeView === 'condiciones' ? $('timeline-slot') : ui.stage;
    if (ui.timeline.parentElement !== slot) slot.append(ui.timeline);
    document.querySelectorAll('.nav-link').forEach((link) => {
      const selected = link.hash === `#${activeView}`;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    });
    try { history.replaceState(null, '', `#${activeView}`); } catch (_) { /* file: URLs in some browsers */ }
    refresh();
  }

  // ---------------------------------------------------------------------------
  // Panels and renderers
  // ---------------------------------------------------------------------------
  function createRenderer(host) {
    if (!rendererFailed) {
      try { return new window.NightjarScene3D(host); } catch (error) { rendererFailed = true; }
    }
    return new window.NightjarFlatMap(host);
  }
  function rebuildPanels(group) {
    const many = group.length > 1;
    ui.panels.classList.toggle('comparing', many);
    while (views.length > group.length) { const v = views.pop(); v.renderer.dispose(); v.panel.remove(); }
    while (views.length < group.length) {
      const index = views.length, panel = document.createElement('div'); panel.className = 'map-panel';
      const host = document.createElement('div'); host.className = 'scene-host'; panel.append(host); ui.panels.append(panel);
      const renderer = createRenderer(host), canvas = renderer.canvas;
      canvas.dataset.index = String(index);
      if (index === 0) canvas.id = 'map-canvas';
      canvas.setAttribute('aria-label', 'Mapa 3D interactivo del episodio');
      const view = {panel, host, renderer, index, title: null, scale: null};
      canvas.addEventListener('pointermove', (e) => { if (!e.buttons) hover(e, view); });
      canvas.addEventListener('pointerleave', () => { ui.tooltip.style.display = 'none'; });
      canvas.addEventListener('pointerdown', () => { ui.tooltip.style.display = 'none'; canvas.focus({preventScroll: true}); });
      canvas.addEventListener('dblclick', (e) => renderer.focusAt(e.clientX, e.clientY));
      canvas.addEventListener('keydown', (e) => canvasKey(e));
      renderer.onCameraChange = (source) => {
        if (syncing || views.length < 2) return;
        syncing = true; const state = source.cameraState();
        for (const other of views) if (other.renderer !== source) other.renderer.applyCameraState(state);
        syncing = false;
      };
      renderer.onModeChange = (mode) => markCamMode(mode);
      renderer.setCameraMode?.(camMode === 'chase' || camMode === 'front' ? '3d' : camMode);
      views.push(view);
    }
    views.forEach((view, index) => {
      if (many && !view.title) {
        view.title = document.createElement('button'); view.title.type = 'button'; view.title.className = 'map-panel-title';
        view.title.addEventListener('click', () => {
          const r = view.replay, next = currentGroup().indexOf(r);
          if (next !== activeReplay) { comparisonReference = activeReplay; activeReplay = next; ui.policy.value = String(activeReplay); updateCompare(); refresh(); }
        });
        view.scale = document.createElement('span'); view.scale.className = 'map-panel-scale';
        view.panel.append(view.title, view.scale);
      } else if (!many && view.title) { view.title.remove(); view.scale.remove(); view.title = null; view.scale = null; }
      view.replay = group[index];
      if (view.title) {
        view.title.textContent = policyLabel(group[index]);
        view.title.title = `Inspeccionar ${policyLabel(group[index])}`;
        const selected = group[index] === current();
        view.panel.classList.toggle('selected-policy', selected); view.title.setAttribute('aria-pressed', String(selected));
      } else view.panel.classList.remove('selected-policy');
      view.renderer.setReplay(group[index], Number(ui.relief.value) || 1);
    });
    updateRendererStatus();
  }
  function updateRendererStatus() {
    const fallback = rendererFailed;
    ui.status.hidden = !fallback;
    if (fallback) ui.status.textContent = 'WebGL no disponible. Vista 2D de compatibilidad; la perspectiva 3D requiere WebGL.';
    for (const v of views) if (fallback) v.renderer.canvas.dataset.renderer = camMode === '2d' ? '2d' : 'canvas-fallback';
    for (const id of ['cam-chase', 'cam-front']) $(id).disabled = fallback;
  }
  function setCamMode(mode) {
    camMode = mode; markCamMode(mode);
    for (const v of views) v.renderer.setCameraMode(mode);
    updateRendererStatus();
  }
  function markCamMode(mode) {
    camMode = mode;
    for (const [id, m] of [['view-3d', '3d'], ['view-2d', '2d'], ['cam-chase', 'chase'], ['cam-front', 'front']]) {
      $(id).classList.toggle('selected', m === mode); $(id).setAttribute('aria-pressed', String(m === mode));
    }
    $('camera-hint').textContent = mode === '2d' ? 'Arrastrar: desplazar · Rueda: zoom · Doble clic: enfocar · Espacio: reproducir'
      : 'Arrastrar: orbitar · Clic dcho o Mayús: desplazar · Rueda: zoom · Doble clic: enfocar · Espacio: reproducir';
  }
  function resetCamera() {
    if (camMode === 'chase' || camMode === 'front') markCamMode('3d');
    for (const v of views) { v.renderer.setCameraMode(camMode); v.renderer.resetCamera(false, true); }
  }
  function exportPng() {
    const canvas = views[0]?.renderer.canvas; if (!canvas) return;
    const link = document.createElement('a');
    link.download = `nightjar-${activeScenario}-${clock(now).replace(':', 'm')}s.png`;
    link.href = canvas.toDataURL('image/png'); link.click();
  }

  // ---------------------------------------------------------------------------
  // Refresh (selection changes) and time updates
  // ---------------------------------------------------------------------------
  function refresh() {
    const r = current(); if (!r) return;
    const H = horizon();
    ui.slider.max = String(H); now = clampNum(now, 0, H);
    const stat = isStatic(r);
    ui.state.disabled = stat;
    if (stat) ui.state.value = 'historical'; else if (ui.state.value === 'historical') ui.state.value = 'combined';
    ui.state.querySelector('[value=historical]').hidden = !stat;
    ui.slider.disabled = stat; ui.play.disabled = stat; $('btn-reset').disabled = stat;
    $('compare-card').hidden = episodeGroup().length < 2 && activeView !== 'comparacion';
    if (!document.querySelector('.workspace').hidden) rebuildPanels(displayGroup());
    updateHeading(r); updateFinalMetrics(r); updateDecisionList(r); updateProvenance(r); updateTicks();
    updateComparisonResults();
    updateTime();
  }
  function updateTime() {
    const r = current(); if (!r) return;
    const H = horizon(); now = clampNum(now, 0, H);
    ui.slider.value = String(now);
    ui.slider.style.setProperty('--progress', `${H ? now / H * 100 : 0}%`);
    $('time-display').textContent = clock(now); $('time-end').textContent = clock(H);
    updateEnvironment(r); updateInspector(r); updateLive(r); updateClock(r);
    const decision = latestDecision(r);
    document.querySelectorAll('.decision-item').forEach((el) => {
      const selected = decision && Number(el.dataset.time) === Number(decision.time_s);
      el.classList.toggle('current', !!selected);
      if (selected) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
    });
    const style = ui.style.value, key = $('terrain-color-key'), f = facts(r);
    key.hidden = style === 'fuel'; key.dataset.style = style;
    key.textContent = style === 'elevation' ? `Altitud: ${fmt(f.minElev, 0)}–${fmt(f.maxElev, 0)} m · huella oculta` : 'Pendiente: 0–45° · huella oculta';
    if (style !== 'fuel' && f.minElev === null) { ui.style.value = 'fuel'; key.hidden = true; }
    if (!document.querySelector('.workspace').hidden) {
      for (const v of views) pushState(v);
      drawGrowth(); drawProfile(r);
    }
  }
  function pushState(view) {
    const r = view.replay; if (!r) return;
    const t = episodeTime(r), f = facts(r), fire = Fire.fireState(r, t);
    const obs = latestObservation(r, t);
    const layers = {front: checked('layer-front'), burn: checked('layer-burn'), reports: checked('layer-observations'), drops: checked('layer-drops'),
      track: checked('layer-flight'), contours: checked('layer-contours'), labels: checked('layer-labels'), grid: checked('layer-grid'), smoke: checked('layer-smoke')};
    view.renderer.setState({now: t, mode: isStatic(r) ? 'historical' : ui.state.value, style: ui.style.value, layers,
      reports: obs?.fire_reports || [], reportsTime: latestDecision(r, t)?.time_s, drops: f.drops.filter((d) => d.time_s <= t), allDrops: f.drops,
      aircraft: aircraftAt(r, t), trail: trailAt(r, t), refills: f.refills, waterLabel: f.waterLabel, wind: windAt(r, t)}, fire);
    view.fire = fire;
  }

  // ---------------------------------------------------------------------------
  // Playback loop
  // ---------------------------------------------------------------------------
  function togglePlay() { if (playing) { stop(); return; } if (now >= horizon()) now = 0; playing = true; ui.play.textContent = 'Pausar'; ui.play.setAttribute('aria-label', 'Pausar'); lastFrame = performance.now(); }
  function stop() { playing = false; ui.play.textContent = 'Reproducir'; ui.play.setAttribute('aria-label', 'Reproducir'); }
  function loop(t) {
    const dt = Math.min(0.1, Math.max(0, (t - lastFrame) / 1000)); lastFrame = t;
    if (playing) { now += dt * speed; if (now >= horizon()) { now = horizon(); stop(); } updateTime(); }
    if (!document.querySelector('.workspace').hidden) {
      for (const v of views) {
        try { v.renderer.frame(t); }
        catch (error) { rendererFailed = true; views.forEach((x) => { x.renderer.dispose(); x.panel.remove(); }); views = []; refresh(); break; }
      }
      if (t - hudTimer > 120) { hudTimer = t; updateCameraHud(); }
    }
    requestAnimationFrame(loop);
  }
  function globalKey(e) {
    const tag = e.target.tagName;
    if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY'].includes(tag) && e.target.type !== 'range') return;
    if (e.key === ' ' && e.target.type !== 'range') { e.preventDefault(); if (!ui.play.disabled) togglePlay(); }
    else if ((e.key === 'r' || e.key === 'R') && !e.metaKey && !e.ctrlKey) { stop(); now = 0; updateTime(); }
  }
  function canvasKey(e) {
    const map = {ArrowLeft: [-0.06, 0], ArrowRight: [0.06, 0], ArrowUp: [0, 0.06], ArrowDown: [0, -0.06], a: [-0.06, 0], d: [0.06, 0], w: [0, 0.06], s: [0, -0.06]};
    if (map[e.key]) { e.preventDefault(); views.forEach((v) => v.renderer.panBy(...map[e.key])); }
    else if (e.key === '+' || e.key === '=') { e.preventDefault(); views.forEach((v) => v.renderer.zoomBy(1.2)); }
    else if (e.key === '-') { e.preventDefault(); views.forEach((v) => v.renderer.zoomBy(1 / 1.2)); }
    else if (e.key === 'Home' || e.key === '0') { e.preventDefault(); resetCamera(); }
    else if (e.key === ' ') { e.preventDefault(); e.stopPropagation(); if (!ui.play.disabled) togglePlay(); }
  }

  // ---------------------------------------------------------------------------
  // HUD
  // ---------------------------------------------------------------------------
  function updateHeading(r) {
    const f = facts(r), grid = r.grid || {}, terrain = r.environment?.terrain?.source || r.metadata?.terrain?.source;
    const dem = terrain && typeof terrain === 'object' ? [terrain.provider, terrain.dataset].filter(Boolean).join(' · ') : (typeof terrain === 'string' ? terrain : '');
    $('map-title').textContent = isStatic(r) ? 'Cartografía histórica inicial · sin episodio' : (r.metadata?.name || r.scenario_id || 'Paisaje');
    $('map-subtitle').textContent = `${grid.rows ?? '?'}×${grid.cols ?? '?'} celdas · ${fmt(grid.cell_size_m, 0)} m · ${fmt(f.minElev, 0)}–${fmt(f.maxElev, 0)} m · ${r.crs || 'CRS no indicado'}${dem ? ` · ${dem}` : ''}`;
    const km2 = grid.rows * grid.cols * f.cellHa / 100;
    $('scenario-kicker').textContent = `DEM ${grid.rows}×${grid.cols} (${fmt(km2, 1)} km²) · ELEV ${fmt(f.minElev, 0)}–${fmt(f.maxElev, 0)} m · ${r.crs || '—'} · ${activeScenario.toUpperCase()}`;
    const t = r.metadata?.terrain, note = $('terrain-note');
    note.hidden = !t?.display_only; note.textContent = t?.display_only ? 'Relieve de referencia: el simulador usó una cuadrícula plana.' : '';
  }
  function updateClock(r) {
    const env = r.environment || {}, origin = env.time_origin_utc || r.time_origin_utc || r.metadata?.acquired_at_utc;
    const date = origin ? new Date(Date.parse(origin) + episodeTime(r) * 1000) : null;
    $('hud-clock').textContent = date && !Number.isNaN(date.valueOf()) ? `${date.toISOString().slice(11, 19)} UTC` : clock(now);
    const decisions = r.decisions || [], k = decisions.filter((d) => Number(d.time_s) <= episodeTime(r)).length;
    $('hud-step').textContent = isStatic(r) ? 'CARTOGRAFÍA · SIN EPISODIO' : `DECISIÓN ${String(k).padStart(2, '0')} / ${String(decisions.length).padStart(2, '0')} · T + ${fmt(now / 60, 1)} MIN`;
    const drops = facts(r).drops.filter((d) => d.time_s <= episodeTime(r)).length;
    $('time-step').textContent = isStatic(r) ? 'Sin episodio' : `Decisión ${k}/${decisions.length} · ${drops} descargas · t = ${fmt(now, 0)} s`;
    const wind = windAt(r, episodeTime(r)), station = activeForcing(r, episodeTime(r))?.station_weather || {};
    const moisture = moisturePercent(activeForcing(r, episodeTime(r))?.moisture?.dead_1h);
    const bits = [];
    if (wind) bits.push(`Viento ${fmt(wind.speed, 1)} m/s hacia ${compass(wind.degrees)}`);
    if (moisture !== null) bits.push(`Comb. muerto 1 h ${fmt(moisture, 1)} %`);
    if (finite(station.air_temperature_c)) bits.push(`${fmt(station.air_temperature_c, 1)} °C`);
    if (finite(station.relative_humidity_pct)) bits.push(`HR ${fmt(station.relative_humidity_pct, 0)} %`);
    const ex = Number(ui.relief.value) || 1;
    bits.push(ex === 1 ? 'Relieve 1:1' : `Relieve exagerado ×${fmt(ex, 1)}`);
    $('hud-conditions').textContent = bits.join(' · ');
    $('compass-wind').style.display = wind ? '' : 'none';
    $('compass-wind-label').textContent = wind ? `${fmt(wind.speed * 3.6, 0)} km/h` : 'sin viento';
  }
  function updateCameraHud() {
    const v = views[0]; if (!v) return;
    const info = v.renderer.cameraInfo(), wind = windAt(v.replay, episodeTime(v.replay));
    $('hud-cam').textContent = `CÁM · INCL ${fmt(info.pitch, 0)}° · AZ ${fmt(info.azimuth, 0)}°${finite(info.altitude) ? ` · ALT ${fmt(info.altitude, 0)} m` : ''}`;
    $('compass-rose').setAttribute('transform', `rotate(${-info.azimuth})`);
    if (wind) $('compass-wind').setAttribute('transform', `rotate(${wind.degrees - info.azimuth})`);
    const nice = (value) => { const p = 10 ** Math.floor(Math.log10(value)), n = value / p; return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * p; };
    const label = (renderer, el) => {
      const i = renderer.cameraInfo(); if (!finite(i.metresPerPixel) || i.metresPerPixel <= 0) return;
      const metres = nice(i.metresPerPixel * 110), px = metres / i.metresPerPixel;
      el.replaceChildren(); const bar = document.createElement('i'); bar.style.width = `${px}px`;
      const text = document.createElement('span'); text.textContent = `${metres >= 1000 ? `${fmt(metres / 1000, 1)} km` : `${fmt(metres, 0)} m`}${i.pitch < 89 ? ' · en el centro' : ''}`;
      el.append(bar, text);
    };
    label(v.renderer, $('map-scale'));
    $('map-scale').hidden = views.length > 1;
    for (const view of views) if (view.scale) label(view.renderer, view.scale);
  }
  function updateTicks() {
    const r = current(), H = horizon(), box = $('timeline-ticks'); box.replaceChildren();
    if (!H || isStatic(r)) return;
    for (const d of r.decisions || []) { const i = document.createElement('i'); i.style.left = `${Number(d.time_s) / H * 100}%`; box.append(i); }
    for (const d of facts(r).drops) { const i = document.createElement('i'); i.className = 'drop'; i.style.left = `${d.time_s / H * 100}%`; box.append(i); }
  }

  // ---------------------------------------------------------------------------
  // Sidebar
  // ---------------------------------------------------------------------------
  function updateFinalMetrics(r) {
    const ids = ['m-area', 'm-water', 'm-flight', 'm-drops', 'm-rejected'];
    if (isStatic(r)) { ids.forEach((id) => { $(id).textContent = '—'; }); return; }
    const m = r.aircraft_metrics || {};
    $('m-area').textContent = fmt(r.affected_area_ha, 2);
    $('m-water').textContent = fmt(m.water_dropped_l ?? m.water_dropped_liters ?? m.total_water_dropped_l, 0);
    const km = m.total_flight_distance_km != null && m.flight_s == null;
    $('m-flight').textContent = fmt(km ? m.total_flight_distance_km : m.flight_s, 1); $('m-flight-unit').textContent = km ? 'km' : 's';
    $('m-drops').textContent = Array.isArray(r.drops) ? String(r.drops.length) : '—';
    $('m-rejected').textContent = fmt(m.rejected_orders ?? m.rejected_orders_count, 0);
  }
  function updateLive(r) {
    const t = episodeTime(r), f = facts(r);
    $('live-time').textContent = `t = ${fmt(t, 0)} s`;
    if (isStatic(r)) {
      for (const id of ['live-area', 'live-front', 'live-water', 'live-phase']) $(id).textContent = '—';
      $('live-area').textContent = fmt(areaAt(r, 0), 0);
      $('live-area-delta').textContent = 'perímetro cartografiado'; $('live-front-cells').textContent = '—'; $('live-drops').textContent = '—'; $('live-tank').textContent = 'sin episodio';
      $('fleet-body').replaceChildren(); $('fleet-count').textContent = '0 unidades';
      return;
    }
    const fire = Fire.fireState(r, t), area = fire.count * f.cellHa, area0 = areaAt(r, 0);
    $('live-area').textContent = fmt(area, area < 100 ? 2 : 0);
    $('live-area-delta').textContent = `+${fmt(area - area0, area - area0 < 100 ? 2 : 0)} ha desde t = 0`;
    $('live-front').textContent = fmt(fire.front.length * f.cellHa, 1);
    $('live-front-cells').textContent = `${fire.front.length} celdas perimetrales`;
    const released = f.releases.filter((x) => x.t <= t);
    $('live-water').textContent = fmt(released.reduce((s, x) => s + x.litres, 0), 0);
    $('live-drops').textContent = `${f.drops.filter((d) => d.time_s <= t).length} descargas registradas`;
    const a = aircraftAt(r, t);
    $('live-phase').textContent = a ? a.phase : '—';
    $('live-tank').textContent = a && finite(a.water) ? `${fmt(a.water, 0)}${a.capacity ? ` / ${fmt(a.capacity, 0)}` : ''} L` : '—';
    const body = $('fleet-body'); body.replaceChildren();
    $('fleet-count').textContent = a ? '1 unidad' : '0 unidades';
    if (a) {
      const tr = document.createElement('tr');
      const cells = [['callsign', 'NJ-01'], [null, 'CL-415 ref.'], [null, a.phase]];
      for (const [cls, text] of cells) { const td = document.createElement('td'); if (cls) { const s = document.createElement('span'); s.className = cls; s.textContent = text; td.append(s); } else td.textContent = text; tr.append(td); }
      const td = document.createElement('td'); td.textContent = finite(a.water) ? `${fmt(a.water, 0)} L` : '—';
      if (a.capacity && finite(a.water)) { const tank = document.createElement('span'); tank.className = 'tank'; const fill = document.createElement('i'); fill.style.width = `${clampNum(a.water / a.capacity * 100, 0, 100)}%`; tank.append(fill); td.append(tank); }
      tr.append(td); body.append(tr);
    }
  }
  function updateDecisionList(r) {
    const list = Array.isArray(r.decisions) ? r.decisions : [];
    ui.decisions.replaceChildren(); $('decision-count').textContent = `${list.length} ${list.length === 1 ? 'registro' : 'registros'}`;
    if (!list.length) { const empty = document.createElement('div'); empty.className = 'empty-state'; empty.textContent = isStatic(r) ? 'Esta capa cartográfica no contiene un episodio de simulación.' : 'Este episodio no contiene decisiones registradas.'; ui.decisions.append(empty); return; }
    for (const d of list) {
      const item = document.createElement('button'); item.type = 'button'; item.className = 'decision-item'; item.dataset.time = String(d.time_s ?? 0);
      const type = orderType(d); if (type === 'DROP') item.classList.add('drop'); if (type === 'RETURN_TO_REFILL') item.classList.add('refill');
      const time = document.createElement('div'); time.className = 'decision-time'; time.textContent = clock(Number(d.time_s) || 0);
      const desc = document.createElement('div'); desc.className = 'decision-desc'; desc.textContent = decisionLabel(d);
      item.append(time, desc); item.addEventListener('click', () => { now = clampNum(Number(d.time_s) || 0, 0, horizon()); updateTime(); });
      ui.decisions.append(item);
    }
  }
  function decisionLabel(d) {
    const order = d.order || d.action || {}, code = orderType(d);
    const kind = ({HOLD: 'Esperar', DROP: 'Descargar', RETURN_TO_REFILL: 'Volver a recargar'})[code] || code || 'Decisión registrada';
    const target = order.target_m || order.position_m;
    return Array.isArray(target) ? `${kind} · (${fmt(target[0], 0)}, ${fmt(target[1], 0)}) m` : String(kind);
  }
  function latestDecision(r, t = Math.min(now, r.horizon_s)) { let found = null; for (const d of (r.decisions || [])) { const time = Number(d.time_s); if (time <= t && (!found || time >= Number(found.time_s))) found = d; } return found; }
  function latestObservation(r, t) { return latestDecision(r, t)?.observation || null; }
  function updateInspector(r) {
    const decision = latestDecision(r), observation = decision?.observation;
    $('inspector-time').textContent = decision ? `Emitida ${clock(Number(decision.time_s))} · hace ${fmt(Math.max(0, Math.min(now, r.horizon_s) - Number(decision.time_s)), 0)} s` : 'Sin decisión disponible';
    $('inspector-action').textContent = decision ? decisionLabel(decision) : r.horizon_s === 0 ? 'Cartografía histórica' : 'Aún no hay decisiones registradas';
    const reports = observation?.fire_reports;
    $('inspector-observation').textContent = observation
      ? `${Array.isArray(reports) ? reports.length : '—'} celdas con detección entregada. ${latestDeliveredWeather(r) ? 'Meteorología entregada disponible.' : 'Sin meteorología entregada.'}`
      : 'Este instante no contiene una observación registrada del controlador.';
  }
  function updateComparisonResults() {
    const box = ui.comparisonResults; box.replaceChildren(); box.classList.remove('message');
    const group = comparisonGroup();
    if (!(compare && group.length > 1)) {
      box.hidden = activeView !== 'comparacion';
      if (!box.hidden) { box.classList.add('message'); box.textContent = 'Este escenario no dispone de dos episodios para comparar.'; }
      return;
    }
    box.hidden = false;
    const table = document.createElement('table'); table.className = 'compare-table';
    const head = document.createElement('thead'), hr = document.createElement('tr');
    ['Política', 'Final (ha)', 'Agua (L)', 'Vuelo', 'Desc.'].forEach((s) => { const th = document.createElement('th'); th.textContent = s; hr.append(th); });
    head.append(hr); const body = document.createElement('tbody');
    group.forEach((r) => {
      const tr = document.createElement('tr'), m = r.aircraft_metrics || {};
      if (r === current()) tr.className = 'active';
      [policyLabel(r), r.affected_area_ha == null ? '—' : fmt(r.affected_area_ha, 2), fmt(m.water_dropped_l ?? m.water_dropped_liters ?? m.total_water_dropped_l, 0),
        m.total_flight_distance_km != null ? `${fmt(m.total_flight_distance_km, 1)} km` : `${fmt(m.flight_s, 1)} s`, isStatic(r) ? '—' : String(r.drops?.length ?? '—')]
        .forEach((v) => { const td = document.createElement('td'); td.textContent = v; tr.append(td); });
      body.append(tr);
    });
    table.append(head, body); box.append(table);
  }

  // ---- charts ---------------------------------------------------------------------
  function prepareCanvas(canvas) {
    const box = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(box.width * dpr)), h = Math.max(1, Math.round(box.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, box.width, box.height);
    return {ctx, w: box.width, h: box.height};
  }
  function seriesColor(r, index) { return r === current() ? '#b8d577' : compare && r === currentGroup()[comparisonReference] ? '#fb923c' : SERIES[index % SERIES.length]; }
  function drawGrowth() {
    const canvas = $('chart-growth'); if (!canvas.offsetParent) return;
    const {ctx, w, h} = prepareCanvas(canvas), group = episodeGroup().length ? episodeGroup() : [current()];
    const H = Math.max(1, ...group.map((r) => Number(r.horizon_s) || 0));
    const values = group.flatMap((r) => [areaAt(r, 0), areaAt(r, Number(r.horizon_s) || 0)]);
    let lo = Math.min(...values), hi = Math.max(...values);
    const span = Math.max(hi - lo, hi * 0.02, 0.5); lo = Math.max(0, lo - span * 0.12); hi = hi + span * 0.12;
    const L = 44, R = 10, T = 10, B = 22, x = (t) => L + t / H * (w - L - R), y = (a) => h - B - (a - lo) / (hi - lo) * (h - T - B);
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'; ctx.fillStyle = '#64748b'; ctx.strokeStyle = '#1d2a3c'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const a = lo + (hi - lo) * i / 4, yy = y(a); ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(w - R, yy); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(fmt(a, a < 10 ? 1 : 0), L - 5, yy + 3); }
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) { const t = H * i / 4; ctx.fillText(`${fmt(t / 60, 0)}′`, x(t), h - 7); }
    const legend = $('growth-legend'); legend.replaceChildren();
    group.forEach((r, index) => {
      const color = seriesColor(r, index), RH = Number(r.horizon_s) || 0, steps = 160;
      ctx.beginPath();
      for (let i = 0; i <= steps; i++) { const t = H * i / steps, a = areaAt(r, Math.min(t, RH)); i ? ctx.lineTo(x(t), y(a)) : ctx.moveTo(x(t), y(a)); }
      ctx.strokeStyle = color; ctx.lineWidth = r === current() ? 2.4 : 1.4; ctx.globalAlpha = r === current() || (compare && r === currentGroup()[comparisonReference]) ? 1 : 0.7; ctx.stroke(); ctx.globalAlpha = 1;
      const a = areaAt(r, Math.min(now, RH));
      ctx.beginPath(); ctx.arc(x(Math.min(now, H)), y(a), 3.2, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
      const item = document.createElement('span'), sw = document.createElement('i'); sw.style.background = color;
      item.append(sw, document.createTextNode(`${policyLabel(r)} · ${fmt(a, a < 100 ? 2 : 0)} ha`)); legend.append(item);
    });
    ctx.setLineDash([4, 3]); ctx.strokeStyle = '#f87171'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(Math.min(now, H)), T); ctx.lineTo(x(Math.min(now, H)), h - B); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#38bdf8'; ctx.textAlign = 'right'; ctx.fillText(`t = +${fmt(now / 60, 1)} min`, w - R, T + 8);
    ctx.textAlign = 'left'; ctx.fillStyle = '#64748b'; ctx.fillText('ha', 4, T + 8);
  }
  function drawProfile(r) {
    const canvas = $('chart-profile'); if (!canvas.offsetParent) return;
    const {ctx, w, h} = prepareCanvas(canvas), f = facts(r), t = episodeTime(r);
    if (f.minElev === null) { ctx.fillStyle = '#64748b'; ctx.font = '11px ui-sans-serif, system-ui'; ctx.fillText('Sin DEM en este escenario', 12, h / 2); $('profile-foot').textContent = '—'; return; }
    const W = f.cols * f.size, Hm = f.rows * f.size;
    const fire = Fire.fireState(r, t), centre = fire.frontCentroid || fire.burnedCentroid || [W / 2, Hm / 2];
    const wind = windAt(r, t), dir = wind && wind.speed > 0 ? [wind.east / wind.speed, wind.north / wind.speed] : [1, 0];
    // Clip the line through the centre along the wind-to direction to the grid rectangle.
    let tmin = -Infinity, tmax = Infinity;
    for (const [p, d, max] of [[centre[0], dir[0], W], [centre[1], dir[1], Hm]]) {
      if (Math.abs(d) < 1e-9) continue;
      const a = (0 - p) / d, b = (max - p) / d; tmin = Math.max(tmin, Math.min(a, b)); tmax = Math.min(tmax, Math.max(a, b));
    }
    if (!finite(tmin) || !finite(tmax) || tmax <= tmin) { tmin = -W / 2; tmax = W / 2; }
    const elevAt = (x, y) => {
      const u = clampNum(x / f.size - 0.5, 0, f.cols - 1), v = clampNum(y / f.size - 0.5, 0, f.rows - 1), j = Math.floor(u), i = Math.floor(v), fu = u - j, fv = v - i;
      const e = (rr, cc) => { const val = r.elevation_m[clampNum(rr, 0, f.rows - 1)]?.[clampNum(cc, 0, f.cols - 1)]; return finite(val) ? val : f.minElev; };
      return (e(i, j) * (1 - fu) + e(i, j + 1) * fu) * (1 - fv) + (e(i + 1, j) * (1 - fu) + e(i + 1, j + 1) * fu) * fv;
    };
    const N = 160, samples = [];
    for (let k = 0; k <= N; k++) {
      const s = tmin + (tmax - tmin) * k / N, x = centre[0] + dir[0] * s, y = centre[1] + dir[1] * s;
      const row = clampNum(Math.floor(y / f.size), 0, f.rows - 1), col = clampNum(Math.floor(x / f.size), 0, f.cols - 1);
      samples.push({s, e: elevAt(x, y), burned: !!fire.burned[row * f.cols + col]});
    }
    const es = samples.map((p) => p.e), lo = Math.min(...es), hi = Math.max(...es), pad = Math.max(10, (hi - lo) * 0.15);
    const L = 44, R = 10, T = 10, B = 20, X = (s) => L + (s - tmin) / (tmax - tmin) * (w - L - R), Y = (e) => h - B - (e - (lo - pad)) / (hi + pad - (lo - pad)) * (h - T - B);
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'; ctx.fillStyle = '#64748b'; ctx.strokeStyle = '#1d2a3c';
    for (const e of [lo, (lo + hi) / 2, hi]) { ctx.beginPath(); ctx.moveTo(L, Y(e)); ctx.lineTo(w - R, Y(e)); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(`${fmt(e, 0)} m`, L - 5, Y(e) + 3); }
    const grad = ctx.createLinearGradient(0, T, 0, h - B); grad.addColorStop(0, 'rgba(56,189,248,.45)'); grad.addColorStop(1, 'rgba(56,189,248,.03)');
    ctx.beginPath(); ctx.moveTo(X(samples[0].s), h - B); samples.forEach((p) => ctx.lineTo(X(p.s), Y(p.e))); ctx.lineTo(X(samples.at(-1).s), h - B); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    for (let k = 1; k < samples.length; k++) {
      const a = samples[k - 1], b = samples[k]; if (!a.burned || !b.burned) continue;
      ctx.beginPath(); ctx.moveTo(X(a.s), h - B); ctx.lineTo(X(a.s), Y(a.e)); ctx.lineTo(X(b.s), Y(b.e)); ctx.lineTo(X(b.s), h - B); ctx.closePath(); ctx.fillStyle = 'rgba(234,88,12,.32)'; ctx.fill();
    }
    ctx.beginPath(); samples.forEach((p, k) => k ? ctx.lineTo(X(p.s), Y(p.e)) : ctx.moveTo(X(p.s), Y(p.e))); ctx.strokeStyle = '#7dd3fc'; ctx.lineWidth = 1.8; ctx.stroke();
    const ec = elevAt(centre[0], centre[1]);
    ctx.beginPath(); ctx.arc(X(0), Y(ec), 4, 0, Math.PI * 2); ctx.fillStyle = fire.frontCentroid ? '#f87171' : '#fbbf24'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = '#fca5a5'; ctx.textAlign = 'center'; ctx.fillText(`${fire.frontCentroid ? 'FRENTE' : 'CENTRO'} · ${fmt(ec, 0)} m`, clampNum(X(0), 60, w - 60), Math.max(T + 8, Y(ec) - 8));
    ctx.fillStyle = '#64748b'; ctx.textAlign = 'left'; ctx.fillText('0', L, h - 6); ctx.textAlign = 'right'; ctx.fillText(`${fmt((tmax - tmin) / 1000, 1)} km`, w - R, h - 6);
    let rise = 0; for (let k = 1; k < samples.length; k++) rise += Math.abs(samples[k].e - samples[k - 1].e);
    const bearing = (Math.atan2(dir[0], dir[1]) * 180 / Math.PI + 360) % 360;
    $('profile-label').textContent = `${compass(bearing + 180)} → ${compass(bearing)}`;
    $('profile-foot').replaceChildren(...[`Corte ${wind ? 'según el viento' : 'oeste–este'}`, `Máx ${fmt(hi, 0)} m`, `Pendiente media ${fmt(rise / (tmax - tmin) * 100, 0)} %`].map((s) => { const e = document.createElement('span'); e.textContent = s; return e; }));
  }

  // ---- tooltip ----------------------------------------------------------------------
  let hoverPending = null;
  function hover(e, view) {
    if (hoverPending) { hoverPending = {e, view}; return; }
    hoverPending = {e, view};
    requestAnimationFrame(() => { const {e: ev, view: v} = hoverPending; hoverPending = null; showTooltip(ev, v); });
  }
  function showTooltip(e, view) {
    const r = view.replay, hit = view.renderer.pick(e.clientX, e.clientY);
    if (!r || !hit) { ui.tooltip.style.display = 'none'; return; }
    const {row, col} = hit, f = facts(r), origin = Array.isArray(r.origin_m) ? r.origin_m : [0, 0];
    const at = r.arrival_time_s?.[row]?.[col], wx = Number(origin[0] || 0) + (col + 0.5) * f.size, wy = Number(origin[1] || 0) + (row + 0.5) * f.size;
    const reported = latestObservation(r, episodeTime(r))?.fire_reports?.some((c) => c[0] === row && c[1] === col);
    const arrival = ui.state.value === 'observation' ? (reported ? 'detección entregada' : 'sin detección entregada') : isStatic(r) ? 'perímetro histórico inicial' : `llegada ${finite(at) ? `${fmt(at, 0)} s (${clock(at)})` : 'no alcanzada'}`;
    const p = r.__painter;
    ui.tooltip.textContent = `Celda [${row}, ${col}] · ${fmt(wx, 0)} E, ${fmt(wy, 0)} N\nCombustible ${r.fuel_grid?.[row]?.[col] ?? '—'}${r.water_mask?.[row]?.[col] ? ' · agua' : ''} · Elevación ${fmt(r.elevation_m?.[row]?.[col], 0)} m · Pendiente ${p ? fmt(p.slopeDeg(row, col), 0) : '—'}°\n${arrival} · ${r.crs || 'CRS no indicado'}`;
    ui.tooltip.style.whiteSpace = 'pre-line'; ui.tooltip.style.display = 'block';
    const box = ui.stage.getBoundingClientRect();
    ui.tooltip.style.left = `${Math.max(8, Math.min(e.clientX - box.left + 14, box.width - ui.tooltip.offsetWidth - 8))}px`;
    ui.tooltip.style.top = `${Math.max(8, Math.min(e.clientY - box.top + 14, box.height - ui.tooltip.offsetHeight - 66))}px`;
  }

  // ---------------------------------------------------------------------------
  // Conditions and provenance
  // ---------------------------------------------------------------------------
  function activeForcing(r, t) {
    const env = r.environment || {}, forcing = Array.isArray(env.forcing) ? env.forcing : [];
    if (env.weather_mode !== 'station_intervals' && env.weather_mode !== 'synthetic_intervals') return null;
    return forcing.find((f) => Number(f.start_s) <= t && t < Number(f.end_s)) || (t >= r.horizon_s ? forcing.filter((f) => Number(f.end_s) <= t).at(-1) : null);
  }
  function updateEnvironment(r) {
    const env = r.environment || {}, stat = isStatic(r);
    const modeName = stat ? 'Condiciones iniciales · sin episodio' : env.weather_mode === 'station_intervals' ? 'Estación · intervalos' : env.weather_mode === 'constant_synthetic' ? 'Viento sintético constante' : env.weather_mode === 'synthetic_intervals' ? 'Viento sintético variable' : 'Sin datos meteorológicos';
    $('environment-mode').textContent = modeName; ui.environment.classList.toggle('is-empty', !env.weather_mode && !stat); ui.environment.classList.toggle('is-static', stat);
    const t = Math.min(now, r.horizon_s);
    const origin = env.time_origin_utc || r.time_origin_utc || r.metadata?.acquired_at_utc, date = origin ? new Date(Date.parse(origin) + (t * 1000)) : null;
    $('environment-time').textContent = `Hora UTC ${date && !Number.isNaN(date.valueOf()) ? date.toISOString().replace('.000Z', 'Z') : '—'} · t = ${fmt(t, 0)} s`;
    let active = null;
    if (env.weather_mode === 'station_intervals' || env.weather_mode === 'synthetic_intervals') active = activeForcing(r, t);
    else if (env.weather_mode === 'constant_synthetic') active = {midflame_wind_to_m_s: r.metadata?.wind_to_m_s || r.wind_to_m_s || null, moisture: null, station_weather: null};
    const wind = vectorValue(active?.midflame_wind_to_m_s), station = active?.station_weather || {};
    $('environment-wind').textContent = wind ? `${fmt(wind.speed, 1)} m/s · ${compass(wind.degrees)}` : '—';
    $('environment-wind-detail').textContent = wind ? `Hacia ${compass(wind.degrees)}${finite(station.wind_speed_2m_m_s) ? ` · estación 2 m ${fmt(station.wind_speed_2m_m_s, 1)} m/s desde ${fmt(station.wind_from_deg, 0)}°` : ''}` : 'Dirección hacia —';
    $('environment-temperature').textContent = finite(station.air_temperature_c) ? `${fmt(station.air_temperature_c, 1)} °C` : '—';
    $('environment-humidity').textContent = finite(station.relative_humidity_pct) ? `${fmt(station.relative_humidity_pct, 0)} %` : '—';
    $('environment-rain').textContent = finite(station.precipitation_mm) ? `${fmt(station.precipitation_mm, 1)} mm` : '—';
    const moisture = moisturePercent(active?.moisture?.dead_1h); $('environment-moisture').textContent = moisture == null ? '—' : `${fmt(moisture, 1)} %`;
    const f = facts(r);
    $('environment-elevation').textContent = f.minElev !== null ? `${fmt(f.minElev, 0)}–${fmt(f.maxElev, 0)} m` : '—';
    const metadataTerrain = r.metadata?.terrain || {}, environmentTerrain = env.terrain || {};
    const terrainSource = environmentTerrain.source ?? metadataTerrain.source ?? {}, terrainLabel = environmentTerrain.label || metadataTerrain.label || '';
    const displayOnly = environmentTerrain.display_only ?? metadataTerrain.display_only;
    const hasSource = typeof terrainSource === 'string' ? terrainSource.trim().length > 0 : !!terrainSource && typeof terrainSource === 'object' && Object.keys(terrainSource).length > 0;
    $('environment-elevation-use').textContent = stat ? 'Capa histórica; no hay episodio' : displayOnly ? 'DEM de referencia; no usado en física' : hasSource ? 'DEM usado por la física' : f.minElev !== null ? 'Elevación disponible; fuente DEM y uso físico no documentados' : 'Sin DEM disponible';
    const ws = env.station || {}, sourceLabel = [ws.code, ws.name, ws.provider, ws.dataset].filter(Boolean).join(' · ');
    const provenance = Array.isArray(env.sources) ? env.sources.map((s) => typeof s === 'string' ? s : (s.provider || s.name || s.dataset)).filter(Boolean).join(', ') : '';
    const availability = active && env.weather_mode === 'station_intervals' ? ` · Intervalo ${active.interval_start_utc || clock(active.start_s)}–${active.interval_end_utc || clock(active.end_s)}; agregado disponible al cierre t = ${fmt(active.available_at_s, 0)} s` : '';
    const uiTerrain = typeof terrainSource === 'string' ? terrainSource : [terrainSource.provider, terrainSource.name, terrainSource.dataset].filter(Boolean).join(' · ');
    $('environment-source').textContent = `${stat ? 'Condiciones iniciales del escenario' : 'Forzamiento actual de simulación'} · meteorología: ${env.weather_mode === 'station_intervals' ? (sourceLabel || provenance || 'estación indicada en el replay') : 'no disponible para el clima sintético'}${availability} · Relieve: ${uiTerrain || terrainLabel || 'fuente no indicada'}`;
    const observation = latestDeliveredWeather(r);
    $('environment-observation').textContent = observation ? describeWeather(observation) : 'No disponible en este instante.';
  }
  function vectorValue(v) { if (!Array.isArray(v) || v.length < 2 || !finite(Number(v[0])) || !finite(Number(v[1]))) return null; const u = Number(v[0]), north = Number(v[1]); return {speed: Math.hypot(u, north), degrees: (Math.atan2(u, north) * 180 / Math.PI + 360) % 360}; }
  function moisturePercent(value) { if (value && typeof value === 'object') value = value.value ?? value.fraction ?? value.moisture_fraction ?? value.pct ?? value.estimated_fraction; if (!finite(value)) return null; return Math.abs(value) <= 1 ? value * 100 : value; }
  function describeWeather(w) { const wind = vectorValue(w.midflame_wind_to_m_s), station = w.station_weather || {}, bits = []; if (wind) bits.push(`viento de simulación ${fmt(wind.speed, 1)} m/s hacia ${compass(wind.degrees)}`); if (finite(station.air_temperature_c)) bits.push(`${fmt(station.air_temperature_c, 1)} °C`); if (finite(station.relative_humidity_pct)) bits.push(`${fmt(station.relative_humidity_pct, 0)} % HR`); if (finite(station.precipitation_mm)) bits.push(`${fmt(station.precipitation_mm, 1)} mm lluvia`); if (w.moisture?.dead_1h != null) bits.push(`combustible muerto 1 h estimado ${fmt(moisturePercent(w.moisture.dead_1h), 1)} %`); return bits.join(' · ') || 'Registro disponible, sin campos reconocidos'; }
  function latestDeliveredWeather(r) { const records = []; for (const d of (r.decisions || [])) { const issued = Number(d.time_s), weather = d.observation?.environment_weather; if (!weather || !Number.isFinite(issued) || issued > Math.min(now, r.horizon_s)) continue; const available = Number(weather.available_at_s); const deliveredAt = Number.isFinite(available) ? available : issued; if (deliveredAt <= now) records.push({weather, deliveredAt, issued}); } records.sort((a, b) => a.deliveredAt - b.deliveredAt || a.issued - b.issued); return records.at(-1)?.weather || null; }
  function updateProvenance(r) {
    const m = r.metadata || {}, terrain = m.terrain || {}, training = m.training || {};
    const splitNames = {development: 'Desarrollo', validation: 'Validación', reserved: 'Reserva'};
    const items = [
      ['Fuente', sourceSummary(m), m.acquired_at_utc ? `Adquirido: ${m.acquired_at_utc}` : 'Época de adquisición no indicada'],
      ['Escenario', m.name || r.scenario_id || '—', `Partición: ${splitNames[m.split] || m.split || '—'} · Semilla: ${m.seed ?? '—'}`],
      ['Relieve', terrain.label || 'Sin capa de elevación', terrainSummary(terrain)],
      ['Modelo de política', training.label || policyLabel(r), trainingSummary(training, m)],
      ['Ejecución', policyLabel(r), `${fmt(r.horizon_s, 0)} s · ${m.seed == null ? 'semilla —' : `semilla ${m.seed}`}`],
      ['Supuestos', typeof m.assumptions === 'string' ? m.assumptions : 'Consulta el detalle técnico', `Versión de esquema: ${r.schema_version || m.schema_version || '—'}`]
    ];
    const box = $('provenance'); box.replaceChildren();
    items.forEach(([label, value, detail]) => { const c = document.createElement('article'); c.className = 'provenance-card'; const a = document.createElement('small'); a.textContent = label; const b = document.createElement('b'); b.textContent = esc(value); const p = document.createElement('p'); p.textContent = esc(detail); c.append(a, b, p); box.append(c); });
    const technical = {source: m.source || null, source_kind: m.source_kind || null, terrain_source: terrain.source || null, terrain_attribution: typeof terrain.source?.attribution === 'string' ? `${terrain.source.attribution.startsWith('produced using') ? '' : 'produced using Copernicus WorldDEM-30 '}${terrain.source.attribution}` : null, training_dataset: training.dataset || null, scenario_hash: m.scenario_hash || null, checkpoint_sha256: training.checkpoint_sha256 || null, assumptions: m.assumptions || null};
    $('technical-provenance').textContent = JSON.stringify(technical, null, 2);
  }
  function sourceSummary(m) { const source = m.source; if (source && typeof source === 'object') { const n = [source.provider || source.name || source.collection || null, source.dataset || source.product || null].filter(Boolean); return n.join(' · ') || 'Fuente documentada'; } return ({copernicus_cems: 'Cartografía Copernicus', synthetic_fixture: 'Mapa sintético'})[m.source_kind] || source || m.source_kind || 'No indicada'; }
  function terrainSummary(t) { const source = t.source; if (source && typeof source === 'object') { const n = [source.provider || source.name || null, source.dataset || source.product || null, source.native_resolution_m ? `${source.native_resolution_m} m` : null].filter(Boolean); return n.join(' · ') || 'Resolución nativa no indicada'; } return typeof source === 'string' ? source : 'Resolución nativa no indicada'; }
  function trainingSummary(t, m) { const dataset = t.dataset, pool = dataset && typeof dataset === 'object' ? (dataset.splits?.development?.length ?? dataset.pool_size ?? dataset.count) : t.dataset_size; return [`Semilla ${t.seed ?? m.seed ?? '—'}`, t.updates == null ? null : `${t.updates} actualizaciones`, pool == null ? null : `${pool} casos de entrenamiento`, t.dev_cases == null ? null : `${t.dev_cases} casos de desarrollo`, t.status || null].filter(Boolean).join(' · ') || 'Estado no documentado'; }

  // Read-only inspection handle for browser regression checks.
  window.NightjarViewer = {get views() { return views; }, get now() { return now; }, get cameraMode() { return camMode; }};
  init();
})();
