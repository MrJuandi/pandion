/* Aircraft audit facts: each track, tank and water release belongs to one unit. */
(function (root) {
  'use strict';
  const finite = Number.isFinite;
  const index = x => Number.isInteger(x?.aircraft_index) ? x.aircraft_index : 0;
  const orderType = d => { const o = d.order || d.action || {}; return o.type || o.action || o.kind; };
  function derive(replay) {
    const units = new Map();
    const unit = i => { if (!units.has(i)) units.set(i, {index: i, track: [], decisions: [], releases: [], capacity: null, initialWater: null}); return units.get(i); };
    for (const initial of replay.aircraft_initial_state || []) {
      const u = unit(initial.index); u.capacity = initial.capacity_l; u.initialWater = initial.water_l;
    }
    for (const p of replay.aircraft_track || []) {
      if (Array.isArray(p.position_m) && p.position_m.every(finite) && finite(p.time_s)) unit(index(p)).track.push({t: p.time_s, p: p.position_m.slice()});
    }
    for (const d of replay.decisions || []) {
      const u = unit(index(d)), ac = d.observation?.aircraft;
      u.decisions.push(d);
      if (u.capacity === null && finite(ac?.capacity_l)) u.capacity = ac.capacity_l;
      if (u.initialWater === null && finite(ac?.water_l)) u.initialWater = ac.water_l;
      if (!(replay.aircraft_track || []).length && Array.isArray(ac?.position_m)) u.track.push({t: Number(d.time_s), p: ac.position_m.slice()});
    }
    const drops = replay.drops || [];
    for (const u of units.values()) {
      u.track.sort((a, b) => a.t - b.t);
      u.decisions.sort((a, b) => Number(a.time_s) - Number(b.time_s));
      if (u.capacity === null) u.capacity = Math.max(0, ...u.decisions.map(d => Number(d.aircraft_water_l)).filter(finite)) || null;
      if (u.initialWater === null) u.initialWater = u.capacity;
      const exact = drops.filter(d => index(d) === u.index && finite(d.water_l));
      if (exact.length) u.releases = exact.map(d => ({t: Number(d.time_s), litres: d.water_l, aircraft_index: u.index}));
      else {
        let previous = u.initialWater;
        for (const d of u.decisions) {
          const after = Number(d.aircraft_water_l), before = d.observation?.aircraft?.water_l ?? previous;
          if (orderType(d) === 'DROP' && finite(after) && finite(before)) {
            u.releases.push({t: Number(d.completed_at_s ?? d.time_s), litres: Math.max(0, before - after), aircraft_index: u.index});
          }
          if (finite(after)) previous = after;
        }
      }
    }
    return [...units.values()].sort((a, b) => a.index - b.index);
  }
  function positionAt(track, time) {
    if (!track.length) return null;
    if (time <= track[0].t) return track[0].p.slice();
    for (let i = 1; i < track.length; i++) {
      const a = track[i - 1], b = track[i];
      if (time <= b.t) { const f = b.t > a.t ? (time - a.t) / (b.t - a.t) : 1; return a.p.map((v, k) => v + (b.p[k] - v) * f); }
    }
    return track.at(-1).p.slice();
  }
  function at(unit, time) {
    const position = positionAt(unit.track, time); if (!position) return null;
    let heading = [1, 0];
    for (let i = 1; i < unit.track.length; i++) {
      const a = unit.track[i - 1], b = unit.track[i], delta = b.p.map((v, k) => v - a.p[k]), length = Math.hypot(...delta);
      if (length > 1e-6 && (a.t <= time || i === 1)) heading = delta.map(v => v / length);
      if (a.t <= time && b.t >= time) break;
    }
    const issued = unit.decisions.filter(d => Number(d.time_s) <= time).at(-1);
    const completed = unit.decisions.filter(d => Number(d.completed_at_s ?? d.time_s) <= time).at(-1);
    const active = issued && time < Number(issued.completed_at_s ?? issued.time_s);
    const type = issued && orderType(issued);
    const phase = !issued ? 'En espera' : type === 'DROP' ? (active ? 'Hacia descarga' : 'Disponible') : type === 'RETURN_TO_REFILL' ? (active ? 'Hacia recarga / recargando' : 'Disponible') : 'En espera';
    const water = completed && finite(Number(completed.aircraft_water_l)) ? Number(completed.aircraft_water_l) : unit.initialWater;
    const accent = ['#7dd3fc', '#c4b5fd', '#fbbf24', '#6ee7b7'][unit.index % 4];
    const callsign = `NJ-${String(unit.index + 1).padStart(2, '0')}`;
    const trail = unit.track.filter(p => p.t <= time).map(p => p.p); trail.push(position);
    return {index: unit.index, position, heading, phase, water, capacity: unit.capacity, accent, callsign, trail,
      labelLines: [`${callsign} · CL-415 ref.`, `${phase}${finite(water) ? ` · ${Math.round(water)} L` : ''}`]};
  }
  const api = {derive, at, positionAt};
  root.NightjarFleet = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
