/* Nightjar 3D scene: interactive Three.js terrain, fire state and aircraft rendering.
 * Every layer is drawn from the replay at the selected time. Display-only choices
 * (vertical exaggeration, aircraft symbol size and altitude, interpolated positions,
 * illustrative smoke) are labelled in the interface and never fed back to a controller.
 */
(() => {
  'use strict';
  const T = window.THREE;
  const finite = (n) => typeof n === 'number' && Number.isFinite(n);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // ---------------------------------------------------------------------------
  // Fire state (pure function of the replay and the selected time)
  // ---------------------------------------------------------------------------
  function isBurnable(r, row, col) {
    if (r.water_mask?.[row]?.[col] === true) return false;
    const fuel = r.fuel_grid?.[row]?.[col];
    if (fuel === null || fuel === undefined || fuel === 'NONE' || fuel === 'none') return false;
    const n = Number(fuel);
    return !(n === 0 || (n >= 91 && n <= 99));
  }

  function fireState(r, now) {
    const rows = r.grid.rows, cols = r.grid.cols, size = r.grid.cell_size_m;
    const burned = new Uint8Array(rows * cols);
    let count = 0, bx = 0, by = 0;
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const a = r.arrival_time_s[row][col];
      if (finite(a) && a <= now) { burned[row * cols + col] = 1; count++; bx += col; by += row; }
    }
    // Front: burned cells with an unburned, burnable 4-neighbour inside the grid.
    const front = [];
    let fx = 0, fy = 0;
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      if (!burned[row * cols + col]) continue;
      let open = false;
      for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nr = row + dr, nc = col + dc;
        if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) continue;
        if (!burned[nr * cols + nc] && isBurnable(r, nr, nc)) { open = true; break; }
      }
      if (open) { front.push({row, col, age: now - r.arrival_time_s[row][col]}); fx += col; fy += row; }
    }
    const centre = (sx, sy, n) => n ? [(sx / n + 0.5) * size, (sy / n + 0.5) * size] : null;
    return {burned, count, front, cellHa: size * size / 1e4,
      frontCentroid: centre(fx, fy, front.length), burnedCentroid: centre(bx, by, count)};
  }

  // ---------------------------------------------------------------------------
  // Palettes
  // ---------------------------------------------------------------------------
  function fuelRGB(code) {
    if (code === null || code === undefined || code === 'NONE' || code === 'none') return [44, 49, 50];
    const n = Number(code);
    if (!Number.isFinite(n)) return [104, 114, 80];
    if (n === 0) return [86, 86, 80];                       // non-burnable / unmapped
    if (n >= 91 && n <= 99) return [92, 90, 84];            // NB
    if (n >= 101 && n <= 109) return [158, 152, 98];        // GR grass
    if (n >= 121 && n <= 124) return [132, 134, 84];        // GS grass-shrub
    if (n >= 141 && n <= 149) return [104, 116, 72];        // SH shrub
    if (n >= 161 && n <= 165) return [78, 102, 66];         // TU timber-understorey
    if (n >= 181 && n <= 189) return [60, 88, 62];          // TL timber litter
    if (n >= 201 && n <= 204) return [124, 108, 82];        // SB slash
    if (n >= 1 && n <= 3) return [158, 152, 98];            // Anderson grass
    if (n >= 4 && n <= 7) return [104, 116, 72];            // Anderson shrub
    if (n >= 8 && n <= 10) return [60, 88, 62];             // Anderson timber
    if (n >= 11 && n <= 13) return [124, 108, 82];          // Anderson slash
    return [110, 118, 82];
  }
  const ramp = (stops, t) => {
    const v = clamp(t, 0, 1) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(v)), f = v - i;
    return stops[i].map((c, k) => c + (stops[i + 1][k] - c) * f);
  };
  const ELEVATION_STOPS = [[47, 90, 82], [109, 138, 92], [185, 168, 116], [239, 227, 194]];
  const SLOPE_STOPS = [[45, 58, 53], [111, 122, 85], [179, 154, 98], [232, 214, 168]];
  const rgb = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;

  // ---------------------------------------------------------------------------
  // Texture painter: one colour map and one emissive map per replay
  // ---------------------------------------------------------------------------
  let noisePattern = null;
  function noiseCanvas() {
    if (noisePattern) return noisePattern;
    const c = document.createElement('canvas'); c.width = c.height = 128;
    const ctx = c.getContext('2d'), img = ctx.createImageData(128, 128);
    let seed = 1234567;
    const rand = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (let i = 0; i < img.data.length; i += 4) { const v = 96 + rand() * 64; img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255; }
    ctx.putImageData(img, 0, 0); noisePattern = c; return c;
  }

  class TexturePainter {
    constructor(r) {
      this.r = r;
      const {rows, cols, cell_size_m: size} = r.grid;
      Object.assign(this, {rows, cols, size});
      this.px = clamp(Math.floor(2048 / Math.max(rows, cols)), 4, 48);
      this.color = document.createElement('canvas'); this.glow = document.createElement('canvas');
      for (const c of [this.color, this.glow]) { c.width = cols * this.px; c.height = rows * this.px; }
      this.cell = document.createElement('canvas');          // 4 px per cell work layer
      this.cell.width = cols * 4; this.cell.height = rows * 4;
      const elev = r.elevation_m.flat().filter(finite);
      this.minElev = elev.length ? Math.min(...elev) : 0; this.maxElev = elev.length ? Math.max(...elev) : 0;
      this.bases = new Map(); this.key = '';
    }
    elevation(row, col) { const v = this.r.elevation_m?.[row]?.[col]; return finite(v) ? v : null; }
    shade(row, col) {
      const e = (rr, cc) => this.elevation(clamp(rr, 0, this.rows - 1), clamp(cc, 0, this.cols - 1)) ?? this.elevation(row, col) ?? 0;
      const dzdx = (e(row, col + 1) - e(row, col - 1)) / (2 * this.size), dzdy = (e(row + 1, col) - e(row - 1, col)) / (2 * this.size);
      const slope = Math.atan(Math.hypot(dzdx, dzdy)), aspect = Math.atan2(dzdy, -dzdx);
      const zenith = Math.PI / 4, azimuth = (360 - 315 + 90) * Math.PI / 180;   // NW sun, 45° altitude
      return Math.cos(zenith) * Math.cos(slope) + Math.sin(zenith) * Math.sin(slope) * Math.cos(azimuth - aspect);
    }
    slopeDeg(row, col) {
      const e = (rr, cc) => this.elevation(clamp(rr, 0, this.rows - 1), clamp(cc, 0, this.cols - 1)) ?? 0;
      const gx = (e(row, col + 1) - e(row, col - 1)) / (2 * this.size), gy = (e(row + 1, col) - e(row - 1, col)) / (2 * this.size);
      return Math.atan(Math.hypot(gx, gy)) * 180 / Math.PI;
    }
    base(style) {
      if (this.bases.has(style)) return this.bases.get(style);
      const {rows, cols, px, r} = this, range = Math.max(1, this.maxElev - this.minElev);
      // Smooth 1 px/cell layer scaled up: soft cartographic transitions between cells.
      const small = document.createElement('canvas'); small.width = cols; small.height = rows;
      const sctx = small.getContext('2d'), img = sctx.createImageData(cols, rows);
      for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
        const e = this.elevation(row, col), t = e === null ? 0 : (e - this.minElev) / range;
        let c = style === 'elevation' ? ramp(ELEVATION_STOPS, t) : style === 'slope' ? ramp(SLOPE_STOPS, this.slopeDeg(row, col) / 45) : fuelRGB(r.fuel_grid?.[row]?.[col]);
        if (style === 'fuel') c = c.map((v, k) => v * (0.86 + 0.24 * t) + [10, 12, 8][k] * t);
        const s = 0.78 + 0.36 * clamp(this.shade(row, col), 0, 1);
        if (e === null && r.elevation_m.some(line => line.some(finite))) c = [30, 34, 36];
        const i = ((rows - 1 - row) * cols + col) * 4;
        img.data[i] = clamp(c[0] * s, 0, 255); img.data[i + 1] = clamp(c[1] * s, 0, 255); img.data[i + 2] = clamp(c[2] * s, 0, 255); img.data[i + 3] = 255;
      }
      sctx.putImageData(img, 0, 0);
      const out = document.createElement('canvas'); out.width = this.color.width; out.height = this.color.height;
      const ctx = out.getContext('2d');
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(small, 0, 0, out.width, out.height);
      ctx.globalCompositeOperation = 'soft-light'; ctx.globalAlpha = 0.35;
      ctx.fillStyle = ctx.createPattern(noiseCanvas(), 'repeat'); ctx.fillRect(0, 0, out.width, out.height);
      ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
      // Water stays crisp: it is categorical and used by the aircraft model.
      this.cellLayer((c, row, col) => r.water_mask?.[row]?.[col] === true ? '#1b5f84' : null);
      ctx.drawImage(this.cell, 0, 0, out.width, out.height);
      this.bases.set(style, out); return out;
    }
    cellLayer(colorAt) {
      const ctx = this.cell.getContext('2d'); ctx.clearRect(0, 0, this.cell.width, this.cell.height);
      for (let row = 0; row < this.rows; row++) for (let col = 0; col < this.cols; col++) {
        const c = colorAt(ctx, row, col); if (!c) continue;
        ctx.fillStyle = c; ctx.fillRect(col * 4, (this.rows - 1 - row) * 4, 4, 4);
      }
      return this.cell;
    }
    cellRect(row, col) { return [col * this.px, (this.rows - 1 - row) * this.px, this.px, this.px]; }
    paint(state, fire) {
      const {now, mode, style, layers} = state, r = this.r;
      const reportsKey = (state.reports || []).length + ':' + (state.reportsTime ?? '');
      const key = [style, mode, fire.count, fire.front.length, Math.floor(now / 30), reportsKey, (state.drops || []).length,
        layers.burn, layers.front, layers.reports, layers.drops, layers.grid].join('|');
      if (key === this.key) return false;
      this.key = key;
      const ctx = this.color.getContext('2d'), g = this.glow.getContext('2d'), {px, rows, cols} = this;
      ctx.drawImage(this.base(style), 0, 0);
      g.fillStyle = '#000'; g.fillRect(0, 0, this.glow.width, this.glow.height);
      const terrainOnly = style !== 'fuel';
      if (mode === 'observation') {
        const set = new Set((state.reports || []).map(c => `${c[0]},${c[1]}`));
        this.cellLayer((c, row, col) => set.has(`${row},${col}`) ? 'rgba(239,120,48,.62)' : null);
        ctx.drawImage(this.cell, 0, 0, this.color.width, this.color.height);
      } else if (!terrainOnly) {
        const historical = mode === 'historical';
        if (layers.burn) {
          this.cellLayer((c, row, col) => fire.burned[row * cols + col] ? (historical ? 'rgba(122,52,26,.86)' : 'rgba(46,38,33,.8)') : null);
          ctx.drawImage(this.cell, 0, 0, this.color.width, this.color.height);
        }
        if (layers.front && fire.front.length) {
          this.cellLayer((c, row, col) => null);
          const cctx = this.cell.getContext('2d');
          for (const f of fire.front) { cctx.fillStyle = f.age < 900 ? '#e4561b' : '#9c3b17'; cctx.fillRect(f.col * 4, (rows - 1 - f.row) * 4, 4, 4); }
          ctx.drawImage(this.cell, 0, 0, this.color.width, this.color.height);
          g.globalCompositeOperation = 'lighter';
          for (const f of fire.front) {
            const heat = historical ? 0.45 : clamp(1 - f.age / 1800, 0.38, 1);
            const [x, y] = [(f.col + 0.5) * px, (rows - 1 - f.row + 0.5) * px];
            const grad = g.createRadialGradient(x, y, px * 0.05, x, y, px * 0.95);
            grad.addColorStop(0, `rgba(255,236,160,${0.8 * heat})`); grad.addColorStop(0.4, `rgba(255,112,24,${0.5 * heat})`);
            grad.addColorStop(1, 'rgba(160,30,0,0)');
            g.fillStyle = grad; g.fillRect(x - px, y - px, px * 2, px * 2);
          }
          g.globalCompositeOperation = 'source-over';
        }
      }
      if (layers.drops && !terrainOnly) {
        const treated = new Set();
        for (const d of state.drops || []) for (const c of d.cells || []) treated.add(`${c[0]},${c[1]}`);
        if (treated.size) {
          this.cellLayer((c, row, col) => treated.has(`${row},${col}`) ? 'rgba(64,190,245,.58)' : null);
          ctx.drawImage(this.cell, 0, 0, this.color.width, this.color.height);
        }
      }
      if (layers.reports && mode !== 'simulator' && !terrainOnly && (state.reports || []).length) {
        const set = new Set(state.reports.map(c => `${c[0]},${c[1]}`));
        ctx.save(); ctx.strokeStyle = '#ffe28a'; ctx.lineWidth = Math.max(1.5, px * 0.08); ctx.setLineDash([px * 0.22, px * 0.14]); ctx.beginPath();
        for (const [row, col] of state.reports) {
          if (row < 0 || col < 0 || row >= rows || col >= cols) continue;
          const [x, y] = this.cellRect(row, col);
          if (!set.has(`${row + 1},${col}`)) { ctx.moveTo(x, y); ctx.lineTo(x + px, y); }
          if (!set.has(`${row - 1},${col}`)) { ctx.moveTo(x, y + px); ctx.lineTo(x + px, y + px); }
          if (!set.has(`${row},${col - 1}`)) { ctx.moveTo(x, y); ctx.lineTo(x, y + px); }
          if (!set.has(`${row},${col + 1}`)) { ctx.moveTo(x + px, y); ctx.lineTo(x + px, y + px); }
        }
        ctx.stroke(); ctx.restore();
      }
      if (layers.grid) {
        ctx.save(); ctx.strokeStyle = 'rgba(230,240,230,.13)'; ctx.lineWidth = 1; ctx.beginPath();
        for (let col = 0; col <= cols; col++) { ctx.moveTo(col * px, 0); ctx.lineTo(col * px, rows * px); }
        for (let row = 0; row <= rows; row++) { ctx.moveTo(0, row * px); ctx.lineTo(cols * px, row * px); }
        ctx.stroke(); ctx.restore();
      }
      return true;
    }
  }

  // ---------------------------------------------------------------------------
  // Models and sprites
  // ---------------------------------------------------------------------------
  function std(color, extra = {}) { return new T.MeshStandardMaterial({color, roughness: 0.5, metalness: 0.12, ...extra}); }
  function part(geometry, material, x = 0, y = 0, z = 0) {
    const m = new T.Mesh(geometry, material); m.position.set(x, y, z); m.castShadow = true; return m;
  }
  /** Stylised amphibious tanker (CL-415 reference), nose along +X, unit length. */
  function buildAmphibian() {
    const group = new T.Group(), yellow = std(0xf2c12e), red = std(0xd83a34), white = std(0xe9edf0), dark = std(0x1d2733, {roughness: 0.3}), grey = std(0x8d97a3);
    const hull = part(new T.CylinderGeometry(0.062, 0.05, 0.86, 18), yellow); hull.rotation.z = Math.PI / 2; group.add(hull);
    const nose = part(new T.SphereGeometry(0.062, 18, 12), yellow, 0.43, 0, 0); nose.scale.set(1.5, 1, 1); group.add(nose);
    const keel = part(new T.BoxGeometry(0.78, 0.045, 0.085), white, 0.02, -0.048, 0); group.add(keel);
    group.add(part(new T.BoxGeometry(0.84, 0.014, 0.128), red, 0, 0.004, 0));
    group.add(part(new T.BoxGeometry(0.1, 0.03, 0.085), dark, 0.4, 0.045, 0));
    group.add(part(new T.BoxGeometry(0.17, 0.022, 1.06), yellow, 0.06, 0.082, 0));
    for (const s of [-1, 1]) {
      group.add(part(new T.BoxGeometry(0.17, 0.024, 0.07), red, 0.06, 0.083, s * 0.5));
      const nacelle = part(new T.CylinderGeometry(0.034, 0.03, 0.24, 14), grey, 0.14, 0.085, s * 0.19); nacelle.rotation.z = Math.PI / 2; group.add(nacelle);
      group.add(part(new T.BoxGeometry(0.03, 0.09, 0.012), grey, 0.04, 0.03, s * 0.44));
      const float = part(new T.CylinderGeometry(0.018, 0.012, 0.13, 10), white, 0.05, -0.02, s * 0.44); float.rotation.z = Math.PI / 2; group.add(float);
    }
    group.add(part(new T.BoxGeometry(0.15, 0.2, 0.016), yellow, -0.4, 0.12, 0));
    group.add(part(new T.BoxGeometry(0.15, 0.04, 0.018), red, -0.41, 0.2, 0));
    group.add(part(new T.BoxGeometry(0.1, 0.012, 0.36), yellow, -0.43, 0.22, 0));
    const rotors = [];
    for (const s of [-1, 1]) {
      const rotor = new T.Group(); rotor.position.set(0.265, 0.085, s * 0.19);
      const spinner = part(new T.ConeGeometry(0.02, 0.05, 12), dark); spinner.rotation.z = -Math.PI / 2; rotor.add(spinner);
      for (let b = 0; b < 4; b++) {
        const arm = new T.Group(); arm.rotation.x = b * Math.PI / 2;
        arm.add(part(new T.BoxGeometry(0.006, 0.13, 0.018), dark, 0, 0.07, 0)); rotor.add(arm);
      }
      const disc = new T.Mesh(new T.CircleGeometry(0.135, 28), new T.MeshBasicMaterial({color: 0xe8eef5, transparent: true, opacity: 0.16, side: T.DoubleSide, depthWrite: false}));
      disc.rotation.y = Math.PI / 2; rotor.add(disc);
      group.add(rotor); rotors.push(rotor);
    }
    group.userData.rotors = rotors;
    return group;
  }

  function makeLabel() {
    const canvas = document.createElement('canvas'); canvas.width = 560; canvas.height = 150;
    const texture = new T.CanvasTexture(canvas); texture.encoding = T.sRGBEncoding; texture.minFilter = T.LinearFilter;
    const material = new T.SpriteMaterial({map: texture, depthTest: false, depthWrite: false, transparent: true, sizeAttenuation: false});
    const sprite = new T.Sprite(material); sprite.renderOrder = 30; sprite.center.set(0.5, 0);
    let last = '';
    function set(lines, accent = '#7dd3fc') {
      const key = lines.join('|') + accent; if (key === last) return; last = key;
      const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = '600 38px Inter, ui-sans-serif, system-ui, sans-serif';
      const width = Math.min(canvas.width - 8, Math.max(...lines.map((l, i) => { ctx.font = `${i ? 500 : 700} ${i ? 30 : 38}px Inter, ui-sans-serif, system-ui, sans-serif`; return ctx.measureText(l).width; })) + 46);
      const x0 = (canvas.width - width) / 2, h = 34 + lines.length * 42;
      ctx.fillStyle = 'rgba(7,12,20,.86)'; ctx.strokeStyle = accent; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x0, 4, width, h, 10) : ctx.rect(x0, 4, width, h); ctx.fill(); ctx.stroke();
      ctx.fillStyle = accent; ctx.fillRect(x0 + 2, 6, 8, h - 4);
      lines.forEach((line, i) => {
        ctx.font = `${i ? 500 : 700} ${i ? 30 : 38}px Inter, ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = i ? '#cbd5e1' : '#f8fafc'; ctx.fillText(line, x0 + 24, 48 + i * 42);
      });
      texture.needsUpdate = true;
      sprite.userData.aspect = canvas.width / canvas.height;
    }
    return {sprite, set};
  }

  function makeParticles(count, color, opacity) {
    const geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(count * 3), 3));
    geometry.setAttribute('aAlpha', new T.Float32BufferAttribute(new Float32Array(count), 1));
    geometry.setAttribute('aSize', new T.Float32BufferAttribute(new Float32Array(count), 1));
    const material = new T.ShaderMaterial({
      uniforms: {uColor: {value: new T.Color(color)}, uScale: {value: 1}, uOrtho: {value: 0}, uOpacity: {value: opacity}},
      vertexShader: 'attribute float aAlpha;attribute float aSize;varying float vAlpha;uniform float uScale;uniform float uOrtho;' +
        'void main(){vAlpha=aAlpha;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=aSize*uScale/(uOrtho>0.5?1.0:max(1.0,-mv.z));gl_Position=projectionMatrix*mv;}',
      fragmentShader: 'uniform vec3 uColor;uniform float uOpacity;varying float vAlpha;' +
        'void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;gl_FragColor=vec4(uColor,smoothstep(0.5,0.05,d)*vAlpha*uOpacity);}',
      transparent: true, depthWrite: false
    });
    const points = new T.Points(geometry, material); points.frustumCulled = false; points.renderOrder = 10;
    return points;
  }
  const hash = (i) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

  function skyTexture() {
    const c = document.createElement('canvas'); c.width = 4; c.height = 256;
    const ctx = c.getContext('2d'), g = ctx.createLinearGradient(0, 0, 0, 256);
    g.addColorStop(0, '#060a12'); g.addColorStop(0.55, '#0d1624'); g.addColorStop(1, '#1d2a38');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 4, 256);
    const t = new T.CanvasTexture(c); t.encoding = T.sRGBEncoding; return t;
  }

  // ---------------------------------------------------------------------------
  // Interactive scene
  // ---------------------------------------------------------------------------
  class NightjarScene3D {
    constructor(host) {
      this.host = host;
      const renderer = new T.WebGLRenderer({antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance'});
      if (!renderer.getContext()) throw new Error('WebGL unavailable');
      this.renderer = renderer;
      renderer.outputEncoding = T.sRGBEncoding; renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.2;
      renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFSoftShadowMap;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.canvas = renderer.domElement; this.canvas.tabIndex = 0; this.canvas.dataset.renderer = 'webgl';
      host.append(this.canvas);
      this.lost = false;
      this.canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });

      const scene = this.scene = new T.Scene();
      scene.background = skyTexture(); scene.fog = new T.Fog(0x111b28, 1e4, 4e4);
      this.persp = new T.PerspectiveCamera(40, 1, 1, 1e6);
      this.ortho = new T.OrthographicCamera(-1, 1, 1, -1, 1, 1e6);
      this.camera = this.persp;
      scene.add(new T.HemisphereLight(0xdde8ff, 0x2b2a22, 0.62));
      scene.add(new T.AmbientLight(0xffffff, 0.06));
      this.sun = new T.DirectionalLight(0xffedd5, 1.55);
      this.sun.castShadow = true; this.sun.shadow.mapSize.set(2048, 2048); this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 1.5;
      scene.add(this.sun, this.sun.target);
      this.fireLight = new T.PointLight(0xff7a2a, 0, 1, 2); scene.add(this.fireLight);

      this.terrainGroup = new T.Group(); scene.add(this.terrainGroup);
      this.overlay = new T.Group(); scene.add(this.overlay);
      this.aircraft = buildAmphibian(); this.aircraft.visible = false; scene.add(this.aircraft);
      this.label = makeLabel(); this.label.sprite.visible = false; scene.add(this.label.sprite);
      this.leader = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3(0, 1, 0)]),
        new T.LineDashedMaterial({color: 0x7dd3fc, dashSize: 10, gapSize: 7, transparent: true, opacity: 0.75}));
      this.leader.visible = false; scene.add(this.leader);
      this.nadir = new T.Mesh(new T.RingGeometry(0.75, 1, 40), new T.MeshBasicMaterial({color: 0x7dd3fc, transparent: true, opacity: 0.7, side: T.DoubleSide, depthWrite: false}));
      this.nadir.rotation.x = -Math.PI / 2; this.nadir.visible = false; scene.add(this.nadir);
      this.trailMax = 4096;
      const trailGeometry = new T.BufferGeometry(); trailGeometry.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(this.trailMax * 3), 3));
      this.trail = new T.Line(trailGeometry, new T.LineBasicMaterial({color: 0x7dd3fc, transparent: true, opacity: 0.8}));
      this.trail.frustumCulled = false; scene.add(this.trail);
      this.groundTrack = new T.Line(trailGeometry.clone(), new T.LineDashedMaterial({color: 0xbfe9ff, dashSize: 30, gapSize: 24, transparent: true, opacity: 0.35, depthWrite: false}));
      this.groundTrack.frustumCulled = false; scene.add(this.groundTrack);
      this.dropGroup = new T.Group(); scene.add(this.dropGroup);
      this.markerGroup = new T.Group(); scene.add(this.markerGroup);
      this.water = makeParticles(420, 0xd8f3ff, 0.9); scene.add(this.water);
      this.smoke = makeParticles(520, 0xc4beb6, 0.16); scene.add(this.smoke);
      this.smokeSources = [];

      this.controls = new T.OrbitControls(this.persp, this.canvas);
      Object.assign(this.controls, {enableDamping: true, dampingFactor: 0.09, screenSpacePanning: true, maxPolarAngle: Math.PI / 2 - 0.05, minPolarAngle: 0.02, panSpeed: 1, zoomSpeed: 1.1, rotateSpeed: 0.8});
      this.controls.addEventListener('change', () => { if (!this.syncing && this.onCameraChange) this.onCameraChange(this); });
      this.controls.addEventListener('start', () => { this.transition = null; if (this.mode === 'chase' || this.mode === 'front') { this.mode = '3d'; this.onModeChange?.(this.mode); } });
      this.raycaster = new T.Raycaster();
      this.mode = '3d'; this.transition = null; this.state = null; this.r = null; this.terrainCache = new Map();
      this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(host);
      this.resize();
    }

    resize() {
      const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
      if (this.size && this.size[0] === w && this.size[1] === h) return;
      this.size = [w, h]; this.renderer.setSize(w, h, false);
      this.canvas.style.width = '100%'; this.canvas.style.height = '100%';
      this.persp.aspect = w / h; this.persp.updateProjectionMatrix(); this.updateOrthoFrustum();
    }
    updateOrthoFrustum() {
      if (!this.extent || !this.size) return;
      const aspect = this.size[0] / this.size[1], half = this.extent * 0.64;
      Object.assign(this.ortho, aspect >= 1 ? {left: -half * aspect, right: half * aspect, top: half, bottom: -half} : {left: -half, right: half, top: half / aspect, bottom: -half / aspect});
      this.ortho.updateProjectionMatrix();
    }

    // ---- terrain -------------------------------------------------------------
    setReplay(r, exaggeration) {
      if (this.r === r && this.exaggeration === exaggeration) return;
      const previousKey = this.terrainKey;
      this.r = r; this.exaggeration = exaggeration;
      this.painter = r.__painter || (r.__painter = new TexturePainter(r));
      const key = terrainKey(r) + '|' + exaggeration;
      let terrain = this.terrainCache.get(key);
      if (!terrain) { terrain = this.buildTerrain(r, exaggeration); this.terrainCache.set(key, terrain); }
      if (this.terrain !== terrain) { if (this.terrain) this.terrainGroup.remove(this.terrain.group); this.terrainGroup.add(terrain.group); this.terrain = terrain; }
      this.colorTexture?.dispose(); this.glowTexture?.dispose();
      this.colorTexture = new T.CanvasTexture(this.painter.color); this.glowTexture = new T.CanvasTexture(this.painter.glow);
      for (const t of [this.colorTexture, this.glowTexture]) { t.encoding = T.sRGBEncoding; t.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy()); }
      terrain.mesh.material.map = this.colorTexture; terrain.mesh.material.emissiveMap = this.glowTexture; terrain.mesh.material.needsUpdate = true;
      this.painter.key = '';
      const {W, H, extent, relief} = terrain;
      Object.assign(this, {W, H, extent, relief, baseY: terrain.baseY});
      this.symbol = clamp(extent * 0.032, 20, 600);
      this.aircraft.scale.setScalar(this.symbol);
      this.nadir.scale.setScalar(this.symbol * 0.32);
      this.leader.material.dashSize = this.symbol * 0.09; this.leader.material.gapSize = this.symbol * 0.06;
      this.groundTrack.material.dashSize = this.symbol * 0.25; this.groundTrack.material.gapSize = this.symbol * 0.2;
      this.sun.position.set(-extent * 0.7, extent * 1.05 + relief, -extent * 0.55);
      Object.assign(this.sun.shadow.camera, {left: -extent * 0.85, right: extent * 0.85, top: extent * 0.85, bottom: -extent * 0.85, near: 1, far: extent * 4});
      this.sun.shadow.camera.updateProjectionMatrix();
      this.scene.fog.near = extent * 2.6; this.scene.fog.far = extent * 9;
      for (const cam of [this.persp, this.ortho]) { cam.near = extent / 4000; cam.far = extent * 40; cam.updateProjectionMatrix(); }
      this.controls.minDistance = extent * 0.02; this.controls.maxDistance = extent * 5;
      this.updateOrthoFrustum();
      this.terrainKey = terrainKey(r);
      if (previousKey !== this.terrainKey) this.resetCamera(true);
      else if (this.lastExaggeration !== exaggeration) this.resetCamera(false);
      this.lastExaggeration = exaggeration;
      this.renderer.shadowMap.needsUpdate = true;
      this.state = null;
    }

    buildTerrain(r, exaggeration) {
      const {rows, cols, cell_size_m: size} = r.grid, W = cols * size, H = rows * size, extent = Math.max(W, H);
      const values = r.elevation_m.flat().filter(finite), hasDEM = values.length > 0;
      const minElev = hasDEM ? Math.min(...values) : 0, maxElev = hasDEM ? Math.max(...values) : 0;
      const sample = (row, col) => r.elevation_m[row]?.[col];
      const corner = new Float32Array((rows + 1) * (cols + 1));
      for (let i = 0; i <= rows; i++) for (let j = 0; j <= cols; j++) {
        let sum = 0, n = 0;
        for (const rr of [i - 1, i]) for (const cc of [j - 1, j]) { const v = sample(rr, cc); if (finite(v)) { sum += v; n++; } }
        corner[i * (cols + 1) + j] = n ? sum / n : minElev;
      }
      const C = (i, j) => corner[clamp(i, 0, rows) * (cols + 1) + clamp(j, 0, cols)];
      const cr = (p0, p1, p2, p3, t) => 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
      // Display surface: Catmull-Rom interpolation of cell-corner heights (no added physical resolution).
      const s = clamp(Math.floor(240 / Math.max(rows, cols)), 1, 6), NR = rows * s, NC = cols * s;
      const heights = new Float32Array((NR + 1) * (NC + 1));
      for (let vi = 0; vi <= NR; vi++) {
        const v = vi / s, i1 = Math.floor(v), tv = v - i1;
        for (let vj = 0; vj <= NC; vj++) {
          const u = vj / s, j1 = Math.floor(u), tu = u - j1;
          const row = (i) => cr(C(i, j1 - 1), C(i, j1), C(i, j1 + 1), C(i, j1 + 2), tu);
          const h = cr(row(i1 - 1), row(i1), row(i1 + 1), row(i1 + 2), tv);
          heights[vi * (NC + 1) + vj] = (clamp(h, minElev, maxElev) - minElev) * exaggeration;
        }
      }
      const positions = new Float32Array((NR + 1) * (NC + 1) * 3), uvs = new Float32Array((NR + 1) * (NC + 1) * 2), indices = [];
      for (let vi = 0; vi <= NR; vi++) for (let vj = 0; vj <= NC; vj++) {
        const k = vi * (NC + 1) + vj;
        positions.set([vj / s * size - W / 2, heights[k], H / 2 - vi / s * size], k * 3);
        uvs.set([vj / NC, vi / NR], k * 2);
      }
      for (let vi = 0; vi < NR; vi++) for (let vj = 0; vj < NC; vj++) {
        if (hasDEM && !finite(sample(Math.floor(vi / s), Math.floor(vj / s)))) continue;
        const a = vi * (NC + 1) + vj, b = a + 1, c = a + NC + 1, d = c + 1;
        indices.push(a, b, c, b, d, c);
      }
      const geometry = new T.BufferGeometry();
      geometry.setAttribute('position', new T.BufferAttribute(positions, 3)); geometry.setAttribute('uv', new T.BufferAttribute(uvs, 2));
      geometry.setIndex(indices); geometry.computeVertexNormals();
      const material = new T.MeshStandardMaterial({roughness: 0.94, metalness: 0, emissive: 0xffffff, emissiveIntensity: 1, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1});
      const mesh = new T.Mesh(geometry, material); mesh.receiveShadow = true; mesh.castShadow = true;
      const group = new T.Group(); group.add(mesh);
      const relief = (maxElev - minElev) * exaggeration;
      const baseY = -Math.max(size * 1.5, relief * 0.16, extent * 0.028);
      // Base walls: show the sampled extent; not a geological section.
      const wall = [], wallColors = [], top = new T.Color(0x5a5145), bottom = new T.Color(0x14110e);
      const edge = (k1, k2) => {
        const p = (k) => [positions[k * 3], positions[k * 3 + 1], positions[k * 3 + 2]];
        const a = p(k1), b = p(k2);
        wall.push(...a, a[0], baseY, a[2], ...b, ...b, a[0], baseY, a[2], b[0], baseY, b[2]);
        for (const c of [top, bottom, top, top, bottom, bottom]) wallColors.push(c.r, c.g, c.b);
      };
      for (let vj = 0; vj < NC; vj++) { edge(vj + 1, vj); edge(NR * (NC + 1) + vj, NR * (NC + 1) + vj + 1); }
      for (let vi = 0; vi < NR; vi++) { edge(vi * (NC + 1), (vi + 1) * (NC + 1)); edge((vi + 1) * (NC + 1) + NC, vi * (NC + 1) + NC); }
      const wallGeometry = new T.BufferGeometry();
      wallGeometry.setAttribute('position', new T.Float32BufferAttribute(wall, 3)); wallGeometry.setAttribute('color', new T.Float32BufferAttribute(wallColors, 3));
      wallGeometry.computeVertexNormals();
      const walls = new T.Mesh(wallGeometry, new T.MeshStandardMaterial({vertexColors: true, roughness: 1, side: T.DoubleSide}));
      walls.receiveShadow = true; group.add(walls);
      // Regional floor and 1 km reference grid for scale beyond the modelled extent.
      const span = Math.ceil(extent * 12 / 1000) * 1000;
      const floor = new T.Mesh(new T.PlaneGeometry(span, span), new T.MeshStandardMaterial({color: 0x0d141c, roughness: 1}));
      floor.rotation.x = -Math.PI / 2; floor.position.y = baseY - 1; floor.receiveShadow = true; group.add(floor);
      const step = extent > 6000 ? 1000 : extent > 1500 ? 500 : 100;
      const grid = new T.GridHelper(span, Math.round(span / step), 0x253445, 0x182431);
      grid.position.y = baseY - 0.5; grid.material.transparent = true; grid.material.opacity = 0.55; group.add(grid);
      // Contours: true equal-elevation intersections of the displayed surface.
      const rawStep = Math.max(1, (maxElev - minElev) / 12), power = 10 ** Math.floor(Math.log10(rawStep));
      const contourStep = (rawStep / power >= 5 ? 5 : rawStep / power >= 2 ? 2 : 1) * power;
      const minor = [], major = [];
      for (let t = 0; t < indices.length; t += 3) {
        const tri = [indices[t], indices[t + 1], indices[t + 2]].map(k => [positions[k * 3], positions[k * 3 + 1], positions[k * 3 + 2]]);
        const lo = Math.min(tri[0][1], tri[1][1], tri[2][1]) / exaggeration + minElev, hi = Math.max(tri[0][1], tri[1][1], tri[2][1]) / exaggeration + minElev;
        for (let level = Math.ceil(lo / contourStep) * contourStep; level < hi; level += contourStep) {
          const z = (level - minElev) * exaggeration, hits = [];
          for (let e = 0; e < 3; e++) {
            const a = tri[e], b = tri[(e + 1) % 3];
            if ((a[1] <= z && z < b[1]) || (b[1] <= z && z < a[1])) { const f = (z - a[1]) / (b[1] - a[1]); hits.push(a[0] + (b[0] - a[0]) * f, z, a[2] + (b[2] - a[2]) * f); }
          }
          if (hits.length === 6) (Math.round(level / contourStep) % 5 === 0 ? major : minor).push(...hits);
        }
      }
      const lines = (data, opacity) => {
        const g = new T.BufferGeometry(); g.setAttribute('position', new T.Float32BufferAttribute(data, 3));
        return new T.LineSegments(g, new T.LineBasicMaterial({color: 0xf1f5e8, transparent: true, opacity, depthWrite: false}));
      };
      const contours = new T.Group(); contours.add(lines(minor, 0.13), lines(major, 0.3)); group.add(contours);
      return {group, mesh, contours, contourStep, heights, NR, NC, s, W, H, size, extent, relief, baseY, minElev, maxElev, rows, cols};
    }

    /** Display-surface height (world y) at local metres; null outside the modelled grid. */
    heightAt(x, y) {
      const t = this.terrain; if (!t) return null;
      const u = x / t.size * t.s, v = y / t.size * t.s;
      if (!(u >= 0 && v >= 0 && u <= t.NC && v <= t.NR)) return null;
      const j = Math.min(t.NC - 1, Math.floor(u)), i = Math.min(t.NR - 1, Math.floor(v)), fu = u - j, fv = v - i, w = t.NC + 1;
      const h = (a, b) => t.heights[a * w + b];
      return (h(i, j) * (1 - fu) + h(i, j + 1) * fu) * (1 - fv) + (h(i + 1, j) * (1 - fu) + h(i + 1, j + 1) * fu) * fv;
    }
    world(x, y, lift = 0) {
      const g = this.heightAt(x, y);
      return new T.Vector3(x - this.W / 2, (g ?? this.baseY) + lift, this.H / 2 - y);
    }

    // ---- dynamic state ---------------------------------------------------------
    setState(state, fire) {
      if (!this.r) return;
      this.state = state; this.fire = fire;
      if (this.painter.paint(state, fire)) { this.colorTexture.needsUpdate = true; this.glowTexture.needsUpdate = true; }
      const L = state.layers;
      this.terrain.contours.visible = L.contours;
      const frontOn = L.front && state.style === 'fuel' && state.mode !== 'observation' && fire.front.length > 0;
      if (frontOn && fire.frontCentroid) {
        const c = this.world(...fire.frontCentroid, this.extent * 0.05);
        this.fireLight.position.copy(c); this.fireLight.distance = this.extent * 0.7;
        this.fireLight.userData.base = clamp(fire.front.length / 25, 0.5, 2.6) * (state.mode === 'historical' ? 0.4 : 1);
      } else this.fireLight.userData.base = 0;
      // Smoke sources: the most recently reached front cells (illustrative layer).
      this.smokeSources = frontOn && L.smoke && state.mode !== 'historical'
        ? fire.front.slice().sort((a, b) => a.age - b.age).slice(0, 80).map(f => this.world((f.col + 0.5) * this.r.grid.cell_size_m, (f.row + 0.5) * this.r.grid.cell_size_m))
        : [];
      this.wind = state.wind;
      this.updateAircraft(state);
      this.updateDrops(state);
      this.updateMarkers(state);
    }

    updateAircraft(state) {
      const a = state.aircraft, show = !!a && state.layers.track;
      for (const o of [this.aircraft, this.label.sprite, this.leader, this.nadir]) o.visible = show;
      const S = this.symbol;
      const agl = S * 1.35;
      const trail = show ? state.trail : [];
      const pos = this.trail.geometry.attributes.position, gpos = this.groundTrack.geometry.attributes.position;
      const n = Math.min(this.trailMax, trail.length);
      for (let i = 0; i < n; i++) {
        const [x, y] = trail[trail.length - n + i], g = this.heightAt(x, y) ?? this.baseY;
        pos.setXYZ(i, x - this.W / 2, g + agl, this.H / 2 - y); gpos.setXYZ(i, x - this.W / 2, g + S * 0.04, this.H / 2 - y);
      }
      this.trail.geometry.setDrawRange(0, n); this.groundTrack.geometry.setDrawRange(0, n);
      pos.needsUpdate = true; gpos.needsUpdate = true; this.groundTrack.computeLineDistances();
      if (!show) return;
      const ground = this.world(a.position[0], a.position[1]);
      const p = ground.clone(); p.y += agl;
      this.aircraft.position.copy(p);
      // heading is [east, north]; the model nose (+X) maps to world (east, -north).
      this.aircraft.rotation.set(0, Math.atan2(a.heading[1], a.heading[0]), 0);
      this.nadir.position.set(ground.x, ground.y + S * 0.05, ground.z);
      this.leader.geometry.setFromPoints([p.clone().add(new T.Vector3(0, -S * 0.08, 0)), ground.clone().add(new T.Vector3(0, S * 0.05, 0))]);
      this.leader.computeLineDistances();
      this.label.sprite.position.set(p.x, p.y + S * 0.42, p.z);
      this.label.set(a.labelLines, a.accent || '#7dd3fc');
    }

    updateDrops(state) {
      const drops = state.layers.drops ? state.allDrops || [] : [];
      const key = drops.filter(d => d.time_s <= state.now).length + ':' + drops.length;
      if (key !== this.dropKey) {
        this.dropKey = key; this.dropGroup.clear();
        for (const d of drops) {
          if (d.time_s > state.now || !Array.isArray(d.position_m)) continue;
          const g = this.world(d.position_m[0], d.position_m[1], this.symbol * 0.06);
          const ring = new T.Mesh(new T.RingGeometry(this.symbol * 0.22, this.symbol * 0.3, 36), new T.MeshBasicMaterial({color: 0x38bdf8, transparent: true, opacity: 0.85, side: T.DoubleSide, depthWrite: false}));
          ring.rotation.x = -Math.PI / 2; ring.position.copy(g); this.dropGroup.add(ring);
        }
      }
      // Water curtain for 45 s of simulation time after each recorded drop (display effect).
      const active = drops.find(d => state.now >= d.time_s && state.now - d.time_s < 45 && Array.isArray(d.position_m));
      this.activeDrop = active ? {drop: active, age: state.now - active.time_s, heading: state.aircraft?.heading || [1, 0]} : null;
    }

    updateMarkers(state) {
      const key = JSON.stringify([state.refills, state.waterLabel, state.layers.labels]);
      if (key === this.markerKey) return;
      this.markerKey = key; this.markerGroup.clear();
      if (!state.layers.labels) return;
      const add = (position, lines, accent) => {
        const label = makeLabel(); label.set(lines, accent);
        const ground = this.world(position[0], position[1]);
        const pole = new T.Mesh(new T.CylinderGeometry(this.symbol * 0.015, this.symbol * 0.015, this.symbol * 0.9, 8), new T.MeshBasicMaterial({color: new T.Color(accent)}));
        pole.position.set(ground.x, ground.y + this.symbol * 0.45, ground.z);
        label.sprite.position.set(ground.x, ground.y + this.symbol * 0.95, ground.z);
        label.sprite.userData.small = true;
        this.markerGroup.add(pole, label.sprite);
      };
      for (const p of state.refills || []) add(p, ['Recarga', 'posición registrada'], '#38bdf8');
      if (state.waterLabel) add(state.waterLabel, ['Agua', 'máscara del escenario'], '#0ea5e9');
    }

    // ---- camera ----------------------------------------------------------------
    setCameraMode(mode) {
      this.mode = mode;
      const ortho = mode === '2d';
      if (ortho !== (this.camera === this.ortho)) {
        const target = this.controls.target.clone();
        this.camera = ortho ? this.ortho : this.persp;
        this.controls.object = this.camera;
        this.controls.enableRotate = !ortho;
        this.controls.minPolarAngle = ortho ? 0 : 0.02; this.controls.maxPolarAngle = ortho ? 0 : Math.PI / 2 - 0.05;
        this.controls.target.copy(target);
      }
      if (mode === '3d' || mode === '2d') this.resetCamera(false, true);
      else if (mode === 'front') {
        const c = this.fire?.frontCentroid || this.fire?.burnedCentroid;
        const target = c ? this.world(c[0], c[1]) : new T.Vector3(0, this.relief * 0.3, 0);
        const dir = this.persp.position.clone().sub(this.controls.target).normalize();
        if (dir.y < 0.35) { dir.y = 0.55; dir.normalize(); }
        this.transition = {position: target.clone().addScaledVector(dir, this.extent * 0.5), target};
      }
    }
    allSprites() { const out = [this.label.sprite]; this.markerGroup.traverse(o => { if (o.isSprite) out.push(o); }); return out; }
    resetCamera(immediate = false, keepMode = false) {
      if (!this.extent) return;
      if (!keepMode && this.mode !== '2d') this.mode = '3d';
      let position, target;
      if (this.camera === this.ortho) {
        target = new T.Vector3(0, 0, 0); position = new T.Vector3(0, this.extent * 3, 0.001);
        this.ortho.zoom = 1; this.ortho.updateProjectionMatrix();
      } else {
        const aspect = this.size ? this.size[0] / this.size[1] : 1.5;
        const dist = this.extent * 1.85 * Math.max(1, 1.1 / aspect) + this.relief * 0.5;
        const polar = 54 * Math.PI / 180, az = -24 * Math.PI / 180;
        target = new T.Vector3(0, this.relief * 0.28, 0);
        position = target.clone().add(new T.Vector3(Math.sin(az) * Math.sin(polar) * dist, Math.cos(polar) * dist, Math.cos(az) * Math.sin(polar) * dist));
      }
      if (immediate) { this.camera.position.copy(position); this.controls.target.copy(target); this.transition = null; this.controls.update(); }
      else this.transition = {position, target};
    }
    zoomBy(factor) {
      this.transition = null;
      if (this.camera === this.ortho) { this.ortho.zoom = clamp(this.ortho.zoom * factor, 0.3, 12); this.ortho.updateProjectionMatrix(); }
      else {
        const offset = this.persp.position.clone().sub(this.controls.target);
        offset.setLength(clamp(offset.length() / factor, this.controls.minDistance, this.controls.maxDistance));
        this.persp.position.copy(this.controls.target).add(offset);
      }
      this.controls.update(); this.onCameraChange?.(this);
    }
    panBy(dx, dy) {
      this.transition = null;
      const forward = new T.Vector3(); this.camera.getWorldDirection(forward);
      if (this.camera === this.ortho) forward.set(0, 0, -1);
      forward.y = 0; forward.normalize();
      const right = new T.Vector3(-forward.z, 0, forward.x);
      const scale = this.camera === this.ortho ? this.extent / this.ortho.zoom : this.persp.position.distanceTo(this.controls.target);
      const delta = right.multiplyScalar(dx * scale).add(forward.multiplyScalar(dy * scale));
      this.camera.position.add(delta); this.controls.target.add(delta); this.controls.update(); this.onCameraChange?.(this);
    }
    cameraState() { return {ortho: this.camera === this.ortho, position: this.camera.position.clone(), target: this.controls.target.clone(), zoom: this.ortho.zoom}; }
    applyCameraState(s) {
      this.syncing = true;
      if (s.ortho !== (this.camera === this.ortho)) this.setCameraMode(s.ortho ? '2d' : '3d');
      this.transition = null; this.camera.position.copy(s.position); this.controls.target.copy(s.target);
      if (s.ortho) { this.ortho.zoom = s.zoom; this.ortho.updateProjectionMatrix(); }
      this.controls.update(); this.syncing = false;
    }
    cameraInfo() {
      const d = this.controls.target.clone().sub(this.camera.position), len = d.length() || 1;
      const ex = this.exaggeration || 1, minElev = this.terrain?.minElev ?? 0;
      const vh = this.camera === this.ortho ? (this.ortho.top - this.ortho.bottom) / this.ortho.zoom : 2 * Math.tan(this.persp.fov * Math.PI / 360) * len;
      return {
        azimuth: this.camera === this.ortho ? 0 : (Math.atan2(d.x, -d.z) * 180 / Math.PI + 360) % 360,
        pitch: this.camera === this.ortho ? 90 : Math.asin(clamp(-d.y / len, -1, 1)) * 180 / Math.PI,
        altitude: this.camera.position.y / ex + minElev,
        metresPerPixel: vh / (this.size ? this.size[1] : 1)
      };
    }
    pick(clientX, clientY) {
      if (!this.terrain) return null;
      const box = this.canvas.getBoundingClientRect();
      const ndc = new T.Vector2((clientX - box.left) / box.width * 2 - 1, -(clientY - box.top) / box.height * 2 + 1);
      this.raycaster.setFromCamera(ndc, this.camera);
      const hit = this.raycaster.intersectObject(this.terrain.mesh)[0];
      if (!hit) return null;
      const x = hit.point.x + this.W / 2, y = this.H / 2 - hit.point.z, size = this.r.grid.cell_size_m;
      const row = Math.floor(y / size), col = Math.floor(x / size);
      if (row < 0 || col < 0 || row >= this.r.grid.rows || col >= this.r.grid.cols) return null;
      return {row, col, point: hit.point};
    }
    focusAt(clientX, clientY) {
      const hit = this.pick(clientX, clientY); if (!hit) return;
      const delta = hit.point.clone().sub(this.controls.target);
      this.transition = {position: this.camera.position.clone().add(delta), target: hit.point.clone()};
    }

    // ---- per-frame animation and rendering ---------------------------------------
    frame(wall) {
      if (this.lost) throw new Error('WebGL context lost');
      this.resize();
      const S = this.symbol || 1;
      if (this.mode === 'chase' && this.aircraft.visible) {
        const yaw = this.aircraft.rotation.y, back = new T.Vector3(-Math.cos(yaw), 0, Math.sin(yaw));
        const side = new T.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
        const position = this.aircraft.position.clone().addScaledVector(back, S * 3).addScaledVector(side, S * 1.2).add(new T.Vector3(0, S * 1.1, 0));
        const target = this.aircraft.position.clone().addScaledVector(back, -S * 1.5);
        this.camera.position.lerp(position, 0.08); this.controls.target.lerp(target, 0.12);
      } else if (this.transition) {
        this.camera.position.lerp(this.transition.position, 0.12); this.controls.target.lerp(this.transition.target, 0.12);
        if (this.camera.position.distanceTo(this.transition.position) < this.extent * 0.002) { this.camera.position.copy(this.transition.position); this.controls.target.copy(this.transition.target); this.transition = null; }
        this.onCameraChange?.(this);
      }
      this.controls.update();
      for (const rotor of this.aircraft.userData.rotors) rotor.rotation.x += 0.9;
      const flicker = 0.92 + 0.08 * Math.sin(wall * 0.011) + 0.05 * Math.sin(wall * 0.027 + 1.3);
      if (this.terrain) this.terrain.mesh.material.emissiveIntensity = flicker * 0.95;
      this.fireLight.intensity = (this.fireLight.userData.base || 0) * flicker;
      const ortho = this.camera === this.ortho;
      const pxPerUnit = ortho ? this.size[1] / ((this.ortho.top - this.ortho.bottom) / this.ortho.zoom) : this.size[1] / (2 * Math.tan(this.persp.fov * Math.PI / 360));
      for (const p of [this.smoke, this.water]) { p.material.uniforms.uScale.value = pxPerUnit * this.renderer.getPixelRatio(); p.material.uniforms.uOrtho.value = ortho ? 1 : 0; }
      this.animateSmoke(wall); this.animateWater();
      // Fixed on-screen label size: perspective sprites are scaled by depth in the r128 shader.
      const worldPerScreen = ortho ? (this.ortho.top - this.ortho.bottom) / this.ortho.zoom : 2 * Math.tan(this.persp.fov * Math.PI / 360);
      for (const sprite of this.allSprites()) {
        const h = 0.058 * (sprite.userData.small ? 0.78 : 1) * worldPerScreen;
        sprite.scale.set(h * (sprite.userData.aspect || 3.7), h, 1);
      }
      this.renderer.render(this.scene, this.camera);
    }
    animateSmoke(wall) {
      const n = this.smokeSources.length, geo = this.smoke.geometry;
      this.smoke.visible = n > 0; if (!n) return;
      const pos = geo.attributes.position, alpha = geo.attributes.aAlpha, size = geo.attributes.aSize;
      const w = this.wind && finite(this.wind.east) ? this.wind : {east: 0.3, north: 0.3};
      const wl = Math.hypot(w.east, w.north) || 1, drift = clamp(wl / 6, 0.25, 1.5);
      const height = this.extent * 0.16, spread = this.extent * 0.22 * drift, S = this.extent * 0.008;
      const t = wall * 0.00004;
      for (let i = 0; i < pos.count; i++) {
        const src = this.smokeSources[i % n], age = (hash(i) + t * (0.8 + hash(i + 7) * 0.6)) % 1;
        const jitter = (hash(i + 3) - 0.5) * S * 2 * (1 + age * 6), jz = (hash(i + 5) - 0.5) * S * 2 * (1 + age * 6);
        pos.setXYZ(i, src.x + w.east / wl * spread * age * age + jitter, src.y + height * Math.pow(age, 0.7), src.z - w.north / wl * spread * age * age + jz);
        alpha.setX(i, Math.sin(Math.PI * Math.min(1, age * 1.15)) * (0.55 + hash(i + 9) * 0.45));
        size.setX(i, S * (1 + age * 5));
      }
      pos.needsUpdate = alpha.needsUpdate = size.needsUpdate = true;
    }
    animateWater() {
      const d = this.activeDrop, geo = this.water.geometry;
      this.water.visible = !!d; if (!d) return;
      const pos = geo.attributes.position, alpha = geo.attributes.aAlpha, size = geo.attributes.aSize, S = this.symbol;
      const ground = this.world(d.drop.position_m[0], d.drop.position_m[1]), drop = S * 1.35;
      const hx = d.heading[0], hz = -d.heading[1], hl = Math.hypot(hx, hz) || 1;
      for (let i = 0; i < pos.count; i++) {
        const along = (hash(i) - 0.5) * S * 2.4, lateral = (hash(i + 11) - 0.5) * S * 0.5;
        const fall = clamp(d.age / 6 + hash(i + 2) * 0.35, 0, 1);
        const spreadMul = 1 + fall * 1.8;
        pos.setXYZ(i, ground.x + hx / hl * along + (-hz / hl) * lateral * spreadMul, ground.y + drop * (1 - fall), ground.z + hz / hl * along + (hx / hl) * lateral * spreadMul);
        alpha.setX(i, (1 - clamp((d.age - 8) / 37, 0, 1)) * (fall < 1 ? 0.85 : 0.4));
        size.setX(i, S * (0.07 + fall * 0.12));
      }
      pos.needsUpdate = alpha.needsUpdate = size.needsUpdate = true;
    }
    dispose() {
      this.resizeObserver.disconnect(); this.controls.dispose();
      this.renderer.dispose(); this.renderer.forceContextLoss?.(); this.canvas.remove();
    }
  }
  function terrainKey(r) {
    let sum = 0; for (const row of r.elevation_m) for (const v of row) if (finite(v)) sum = (sum * 31 + Math.round(v * 10)) % 2147483647;
    return `${r.grid.rows}x${r.grid.cols}@${r.grid.cell_size_m}#${sum}`;
  }

  // ---------------------------------------------------------------------------
  // 2D compatibility map (used only when WebGL is unavailable)
  // ---------------------------------------------------------------------------
  class NightjarFlatMap {
    constructor(host) {
      this.host = host; this.canvas = document.createElement('canvas'); this.canvas.tabIndex = 0; this.canvas.dataset.renderer = 'canvas-fallback';
      this.canvas.style.width = '100%'; this.canvas.style.height = '100%'; host.append(this.canvas);
      this.zoom = 1; this.pan = [0, 0]; this.mode = '2d';
    }
    setReplay(r) { if (this.r === r) return; this.r = r; this.painter = r.__painter || (r.__painter = new TexturePainter(r)); this.painter.key = ''; }
    setState(state, fire) { this.state = state; this.fire = fire; this.painter.paint(state, fire); }
    setCameraMode(mode) { this.mode = mode; }
    resetCamera() { this.zoom = 1; this.pan = [0, 0]; }
    zoomBy(f) { this.zoom = clamp(this.zoom * f, 0.4, 8); }
    panBy(dx, dy) { this.pan[0] -= dx; this.pan[1] += dy; }
    cameraState() { return {zoom: this.zoom, pan: [...this.pan]}; }
    applyCameraState(s) { if (s && s.pan) { this.zoom = s.zoom; this.pan = [...s.pan]; } }
    layout() {
      const box = this.host.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(box.width * dpr)), h = Math.max(1, Math.round(box.height * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      const {cols, rows, cell_size_m: size} = this.r.grid, scale = Math.min(w * 0.9 / cols, h * 0.9 / rows) * this.zoom;
      return {w, h, dpr, scale, x0: w / 2 - cols * scale / 2 + this.pan[0] * w, y0: h / 2 - rows * scale / 2 + this.pan[1] * h, size};
    }
    toScreen(L, x, y) { return [L.x0 + x / L.size * L.scale, L.y0 + (this.r.grid.rows - y / L.size) * L.scale]; }
    cameraInfo() { const L = this.layout(); return {azimuth: 0, pitch: 90, altitude: null, metresPerPixel: L.size / L.scale * L.dpr}; }
    pick(clientX, clientY) {
      const L = this.layout(), box = this.canvas.getBoundingClientRect();
      const x = ((clientX - box.left) * L.dpr - L.x0) / L.scale, y = ((clientY - box.top) * L.dpr - L.y0) / L.scale;
      const col = Math.floor(x), row = this.r.grid.rows - 1 - Math.floor(y);
      return row >= 0 && col >= 0 && row < this.r.grid.rows && col < this.r.grid.cols ? {row, col} : null;
    }
    focusAt() {}
    frame() {
      if (!this.r) return;
      const L = this.layout(), ctx = this.canvas.getContext('2d');
      ctx.fillStyle = '#0d141c'; ctx.fillRect(0, 0, L.w, L.h);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.painter.color, L.x0, L.y0, this.r.grid.cols * L.scale, this.r.grid.rows * L.scale);
      ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = 0.8;
      ctx.drawImage(this.painter.glow, L.x0, L.y0, this.r.grid.cols * L.scale, this.r.grid.rows * L.scale);
      ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
      const s = this.state; if (!s) return;
      if (s.layers.track && s.trail?.length > 1) {
        ctx.beginPath(); s.trail.forEach((p, i) => { const q = this.toScreen(L, p[0], p[1]); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
        ctx.strokeStyle = '#7dd3fc'; ctx.lineWidth = 2 * L.dpr; ctx.stroke();
      }
      if (s.layers.drops) for (const d of s.drops || []) {
        if (!Array.isArray(d.position_m)) continue; const [x, y] = this.toScreen(L, d.position_m[0], d.position_m[1]);
        ctx.beginPath(); ctx.arc(x, y, 5 * L.dpr, 0, Math.PI * 2); ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2 * L.dpr; ctx.stroke();
      }
      if (s.layers.track && s.aircraft) {
        const [x, y] = this.toScreen(L, s.aircraft.position[0], s.aircraft.position[1]);
        ctx.beginPath(); ctx.arc(x, y, 6 * L.dpr, 0, Math.PI * 2); ctx.fillStyle = '#f2c12e'; ctx.fill(); ctx.strokeStyle = '#0b1220'; ctx.lineWidth = 2 * L.dpr; ctx.stroke();
      }
    }
    dispose() { this.canvas.remove(); }
  }

  window.NightjarFire = {fireState, isBurnable};
  window.NightjarScene3D = NightjarScene3D;
  window.NightjarFlatMap = NightjarFlatMap;
})();
