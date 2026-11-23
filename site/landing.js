(() => {
  "use strict";
  const E = window.PANDION_EPISODE;
  if (!E) return;

  const ROWS = E.grid.rows, COLS = E.grid.cols, CS = E.grid.cell_size_m;
  const W = COLS * CS, H = ROWS * CS, T_END = E.horizon_s;
  const CELL_HA = (CS * CS) / 10000;
  const RUN_ORDER = ["learned", "heuristic", "none"];
  const COLORS = { water: "#4cc3e0", plumage: "#e9ece6", fire: "#ff6b2c", hot: "#ffb36b", night: "#0a0e13" };
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- derived data ----------
  const runs = {};
  for (const key of RUN_ORDER) {
    const r = E.runs[key];
    const arrivals = [];
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) if (r.arrival[i][j] != null) arrivals.push(r.arrival[i][j]);
    arrivals.sort((a, b) => a - b);
    const tracks = new Map();
    for (const [k, t, x, y] of r.track) {
      if (!tracks.has(k)) tracks.set(k, []);
      tracks.get(k).push({ t, x, y });
    }
    for (const list of tracks.values()) list.sort((a, b) => a.t - b.t);
    const drops = r.drops.map(([k, t, litres, cells]) => ({ k, t, litres, cells })).sort((a, b) => a.t - b.t);
    runs[key] = { ...r, arrivals, tracks, drops };
  }
  const countUpTo = (sorted, t) => { let lo = 0, hi = sorted.length; while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= t) lo = m + 1; else hi = m; } return lo; };

  // perimeter of the no-aircraft outcome, as cell edges in grid units
  const ghostEdges = [];
  {
    const A = E.runs.none.arrival, burned = (i, j) => i >= 0 && j >= 0 && i < ROWS && j < COLS && A[i][j] != null;
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
      if (!burned(i, j)) continue;
      if (!burned(i + 1, j)) ghostEdges.push([j, i + 1, j + 1, i + 1]);
      if (!burned(i - 1, j)) ghostEdges.push([j, i, j + 1, i]);
      if (!burned(i, j + 1)) ghostEdges.push([j + 1, i, j + 1, i + 1]);
      if (!burned(i, j - 1)) ghostEdges.push([j, i, j, i + 1]);
    }
  }


  const img = new Image();
  let imgReady = false;
  const maps = [];
  img.onload = () => { imgReady = true; maps.forEach((m) => m.resize()); };
  img.src = "data/basemap.png";
  function elevationAt(gx, gy) { // gx, gy in cell units, row axis northward
    const x = Math.min(Math.max(gx - 0.5, 0), COLS - 1), y = Math.min(Math.max(gy - 0.5, 0), ROWS - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, COLS - 1), y1 = Math.min(y0 + 1, ROWS - 1);
    const fx = x - x0, fy = y - y0, Z = E.elevation_m;
    return (Z[y0][x0] * (1 - fx) + Z[y0][x1] * fx) * (1 - fy) + (Z[y1][x0] * (1 - fx) + Z[y1][x1] * fx) * fy;
  }

  function hillshade(n) {
    const c = document.createElement("canvas"); c.width = c.height = n;
    const g = c.getContext("2d"), data = g.createImageData(n, n);
    const step = COLS / n, az = (315 * Math.PI) / 180, alt = (40 * Math.PI) / 180, zf = 1.6;
    for (let py = 0; py < n; py++) for (let px = 0; px < n; px++) {
      const gx = (px + 0.5) * step, gy = ROWS - (py + 0.5) * step;
      const dzdx = (elevationAt(gx + step, gy) - elevationAt(gx - step, gy)) / (2 * step * CS);
      const dzdy = (elevationAt(gx, gy + step) - elevationAt(gx, gy - step)) / (2 * step * CS);
      const slope = Math.atan(zf * Math.hypot(dzdx, dzdy));
      const aspect = Math.atan2(dzdy, -dzdx);
      let v = Math.cos(alt) * Math.cos(slope) + Math.sin(alt) * Math.sin(slope) * Math.cos(az - Math.PI / 2 - aspect);
      v = Math.max(0, Math.min(1, v));
      const o = (py * n + px) * 4, s = Math.round(70 + 185 * v);
      data.data[o] = data.data[o + 1] = data.data[o + 2] = s; data.data[o + 3] = 255;
    }
    g.putImageData(data, 0, 0);
    return c;
  }
  let shadeCanvas = null;

  function makeMap(canvas, opts) {
    const ctx = canvas.getContext("2d"), base = document.createElement("canvas");
    let n = 0, scale = 1;
    const px = (x) => (x / W) * n, py = (y) => (1 - y / H) * n;

    function resize() {
      const css = canvas.getBoundingClientRect().width || canvas.offsetWidth;
      if (!css) return;
      n = Math.min(Math.round(css * Math.min(window.devicePixelRatio || 1, 2)), opts.maxPx || 2048);
      scale = n / css;
      canvas.width = canvas.height = n;
      if (imgReady) buildBase();
    }

    function buildBase() {
      base.width = base.height = n;
      const b = base.getContext("2d");
      b.imageSmoothingEnabled = true; b.imageSmoothingQuality = "high";
      b.drawImage(img, 0, 0, n, n);
      b.globalCompositeOperation = "saturation"; b.globalAlpha = 0.4; b.fillStyle = "#777"; b.fillRect(0, 0, n, n);
      if (!shadeCanvas) shadeCanvas = hillshade(256);
      b.globalCompositeOperation = "overlay"; b.globalAlpha = 0.95; b.drawImage(shadeCanvas, 0, 0, n, n);
      b.globalCompositeOperation = "multiply"; b.globalAlpha = 0.6; b.drawImage(shadeCanvas, 0, 0, n, n);
      b.globalCompositeOperation = "source-over"; b.globalAlpha = opts.dim ?? 0.18; b.fillStyle = COLORS.night; b.fillRect(0, 0, n, n);
      b.globalAlpha = 1;
      b.strokeStyle = "rgba(233,236,230,0.06)"; b.lineWidth = Math.max(1, scale * 0.75);
      for (let m = 1000; m < W; m += 1000) { const p = Math.round((m / W) * n) + 0.5; b.beginPath(); b.moveTo(p, 0); b.lineTo(p, n); b.moveTo(0, p); b.lineTo(n, p); b.stroke(); }
    }

    function draw(runKey, t, now, animate) {
      if (!n) return;
      const cw = n / COLS, ch = n / ROWS, run = runs[runKey];
      ctx.clearRect(0, 0, n, n);
      if (imgReady) ctx.drawImage(base, 0, 0);

      const burning = [];
      for (let i = 0; i < ROWS; i++) {
        const A = run.arrival[i], B = run.burnout[i];
        for (let j = 0; j < COLS; j++) {
          const a = A[j];
          if (a == null || a > t) continue;
          const x = j * cw, y = (ROWS - 1 - i) * ch, b = B[j];
          if (b != null && t >= b) { ctx.fillStyle = "rgba(22,11,7,0.84)"; ctx.fillRect(x, y, cw + 0.5, ch + 0.5); }
          else burning.push([x, y, b != null && b > a ? (t - a) / (b - a) : 0.5, i * 41 + j * 7]);
        }
      }
      const flick = animate ? now / 150 : 0;
      for (const [x, y, p, seed] of burning) {
        const k = 0.8 + 0.2 * Math.sin(flick + seed);
        ctx.fillStyle = p < 0.35 ? `rgba(255,190,120,${0.96 * k})` : `rgba(255,107,44,${0.92 * k})`;
        ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
      }
      ctx.globalCompositeOperation = "lighter";
      for (const [x, y] of burning) {
        ctx.fillStyle = "rgba(255,107,44,0.09)"; ctx.fillRect(x - cw, y - ch, cw * 3, ch * 3);
      }
      ctx.globalCompositeOperation = "source-over";

      for (const d of run.drops) {
        if (d.t > t) break;
        for (const [i, j] of d.cells) {
          const x = j * cw, y = (ROWS - 1 - i) * ch;
          ctx.fillStyle = "rgba(76,195,224,0.85)"; ctx.fillRect(x, y, cw, ch);
          ctx.strokeStyle = "rgba(200,242,252,0.95)"; ctx.lineWidth = Math.max(1, scale); ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
        }
        const age = t - d.t;
        if (age < 120) {
          const f = age / 120, [i, j] = d.cells[0] || [0, 0];
          ctx.beginPath(); ctx.arc((j + 0.5) * cw, (ROWS - i - 0.5) * ch, cw * (0.6 + f * 4), 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(76,195,224,${0.9 * (1 - f)})`; ctx.lineWidth = 2 * scale; ctx.stroke();
        }
      }

      if (opts.ghost && runKey !== "none") {
        ctx.save(); ctx.beginPath();
        for (const [x0, y0, x1, y1] of ghostEdges) { ctx.moveTo(x0 * cw, (ROWS - y0) * ch); ctx.lineTo(x1 * cw, (ROWS - y1) * ch); }
        ctx.setLineDash([3 * scale, 4 * scale]); ctx.strokeStyle = "rgba(233,236,230,0.5)"; ctx.lineWidth = 1.1 * scale; ctx.stroke();
        ctx.restore();
      }

      for (const list of run.tracks.values()) {
        ctx.beginPath();
        let started = false;
        for (let s = Math.max(0, t - 180); s <= t; s += 5) {
          const p = aircraftAt(list, s), X = px(p.x), Y = py(p.y);
          if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        }
        ctx.strokeStyle = "rgba(76,195,224,0.5)"; ctx.lineWidth = 1.5 * scale; ctx.stroke();
        const p = aircraftAt(list, t), X = px(p.x), Y = py(p.y);
        if (X < -40 || Y < -40 || X > n + 40 || Y > n + 40) continue;
        ctx.save(); ctx.translate(X, Y); ctx.rotate(Math.atan2(p.hx, p.hy)); ctx.scale(scale * opts.plane, scale * opts.plane);
        ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.strokeStyle = "rgba(10,14,19,0.8)"; ctx.lineWidth = 6; ctx.stroke(WING);
        ctx.strokeStyle = COLORS.plumage; ctx.lineWidth = 2.6; ctx.stroke(WING);
        ctx.restore();
      }
    }

    const m = { canvas, resize, draw, visible: true };
    new ResizeObserver(() => resize()).observe(canvas);
    new IntersectionObserver(([e]) => { m.visible = e.isIntersecting; }).observe(canvas);
    maps.push(m);
    return m;
  }

  const WING = new Path2D("M-17 1 Q-11 -9 -4.5 -4.5 L0 2 L4.5 -4.5 Q11 -9 17 1");

  function aircraftAt(list, t) {
    if (t <= list[0].t) return { x: list[0].x, y: list[0].y, hx: 0, hy: 1 };
    for (let i = 1; i < list.length; i++) {
      const a = list[i - 1], b = list[i];
      if (t <= b.t) {
        const f = b.t === a.t ? 1 : (t - a.t) / (b.t - a.t);
        const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
        return { x: a.x + dx * f, y: a.y + dy * f, hx: len ? dx / len : 0, hy: len ? dy / len : 1, still: len < 1 };
      }
    }
    const l = list[list.length - 1];
    return { x: l.x, y: l.y, hx: 0, hy: 1, still: true };
  }


  // ---------- scene ----------
  let t = 1200, playing = !reduceMotion, lastFrame = 0, holdUntil = 0; // open mid-episode so the fire is already visible
  if (reduceMotion) t = T_END;
  const hero = makeMap(document.getElementById("hero-map"), { plane: 1.6, maxPx: 2048, dim: 0.1 });
  const panels = [...document.querySelectorAll(".panel canvas")].map((c) => ({ key: c.dataset.run, map: makeMap(c, { plane: 0.8, ghost: true, maxPx: 1200 }), live: document.querySelector(`[data-live="${c.dataset.run}"]`) }));

  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s) % 60).padStart(2, "0")}`;
  const areaAt = (key, tt) => Math.round(countUpTo(runs[key].arrivals, tt) * CELL_HA);
  const hTime = document.getElementById("h-time"), hArea = document.getElementById("h-area"), clock = document.getElementById("clock");
  const scrub = document.getElementById("scrub");

  // timeline curves
  const curves = document.getElementById("curves"), cctx = curves.getContext("2d");
  const maxHa = Math.max(...RUN_ORDER.map((k) => runs[k].final_ha));
  function drawCurves() {
    const r = curves.getBoundingClientRect(), d = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(r.width * d), h = Math.round(r.height * d);
    if (!w) return;
    if (curves.width !== w || curves.height !== h) { curves.width = w; curves.height = h; }
    cctx.clearRect(0, 0, w, h);
    cctx.fillStyle = "#0f141b"; cctx.fillRect(0, 0, w, h);
    const cx = (t / T_END) * w;
    cctx.fillStyle = "rgba(233,236,230,0.05)"; cctx.fillRect(0, 0, cx, h);
    for (const k of ["none", "heuristic", "learned"]) {
      cctx.beginPath();
      for (let s = 0; s <= 160; s++) {
        const tt = (s / 160) * T_END, y = h - 6 * d - (areaAt(k, tt) / maxHa) * (h - 12 * d);
        s ? cctx.lineTo((tt / T_END) * w, y) : cctx.moveTo(0, y);
      }
      cctx.strokeStyle = k === "learned" ? COLORS.water : k === "heuristic" ? "rgba(233,236,230,0.6)" : "rgba(255,107,44,0.75)";
      cctx.lineWidth = (k === "learned" ? 2 : 1.25) * d; cctx.stroke();
    }
    cctx.fillStyle = COLORS.plumage; cctx.fillRect(Math.round(cx) - d, 0, 2 * d, h);
  }

  function render(now) {
    const animate = playing && !reduceMotion;
    if (hero.visible) hero.draw("learned", t, now, animate);
    for (const p of panels) if (p.map.visible) p.map.draw(p.key, t, now, animate);
    for (const p of panels) p.live.textContent = areaAt(p.key, t);
    hTime.textContent = fmt(t); hArea.textContent = areaAt("learned", t); clock.textContent = fmt(t);
    scrub.value = String(Math.round(t)); scrub.setAttribute("aria-valuetext", `${Math.floor(t / 60)} minutes`);
    drawCurves();
  }

  const playBtn = document.getElementById("play");
  function setPlaying(v) {
    playing = v; playBtn.classList.toggle("paused", !v);
    playBtn.setAttribute("aria-label", v ? "Pause replay" : "Play replay");
    if (v && t >= T_END) t = 0;
    lastFrame = 0; holdUntil = 0;
  }
  playBtn.addEventListener("click", () => setPlaying(!playing));
  scrub.addEventListener("pointerdown", () => setPlaying(false));
  scrub.addEventListener("input", () => { t = Number(scrub.value); render(performance.now()); });

  const bar = document.querySelector(".bar");
  const onScroll = () => bar.classList.toggle("solid", window.scrollY > 24);
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  const SIM_PER_SEC = 80; // one hour of fire in 45 s
  function frame(now) {
    if (playing) {
      if (holdUntil) { if (now >= holdUntil) { holdUntil = 0; t = 0; } }
      else if (lastFrame) { t = Math.min(T_END, t + ((now - lastFrame) / 1000) * SIM_PER_SEC); if (t >= T_END) holdUntil = now + 3500; }
      lastFrame = now;
    }
    if (hero.visible || panels.some((p) => p.map.visible)) render(now);
    requestAnimationFrame(frame);
  }
  if (reduceMotion) setPlaying(false);
  requestAnimationFrame(frame);
})();
