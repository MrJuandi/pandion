(function (root) {
  "use strict";

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }
  function matrixShape(matrix, label) {
    if (matrix == null) return null;
    if (
      !Array.isArray(matrix) ||
      matrix.length === 0 ||
      !Array.isArray(matrix[0]) ||
      matrix[0].length === 0
    )
      throw new Error(label + " must be a non-empty rectangular matrix");
    const cols = matrix[0].length;
    if (matrix.some((row) => !Array.isArray(row) || row.length !== cols))
      throw new Error(
        label + " must be rectangular (all rows must have the same length)",
      );
    return { rows: matrix.length, cols };
  }
  function validateReplay(replay) {
    if (!replay || typeof replay !== "object" || Array.isArray(replay))
      throw new Error("Replay must be a JSON object");
    if (
      replay.horizon_s != null &&
      (!finite(replay.horizon_s) || replay.horizon_s < 0)
    )
      throw new Error("horizon_s must be finite and nonnegative");
    if (replay.grid != null) {
      if (typeof replay.grid !== "object" || Array.isArray(replay.grid))
        throw new Error("grid must be an object when present");
      for (const key of ["rows", "cols"])
        if (
          replay.grid[key] != null &&
          (!Number.isInteger(replay.grid[key]) || replay.grid[key] <= 0)
        )
          throw new Error("grid." + key + " must be a positive integer");
      if (
        replay.grid.cell_size_m != null &&
        (!finite(replay.grid.cell_size_m) || replay.grid.cell_size_m <= 0)
      )
        throw new Error("grid.cell_size_m must be finite and positive");
    }
    for (const key of ["arrival_time_s"]) {
      const shape = matrixShape(replay[key], key);
      if (
        shape &&
        replay[key].some((row) =>
          row.some((v) => v != null && (!finite(v) || v < 0)),
        )
      )
        throw new Error(
          key + " values must be finite and nonnegative, or null",
        );
    }
    const decisions = replay.decisions == null ? [] : replay.decisions;
    if (!Array.isArray(decisions))
      throw new Error("decisions must be an array when present");
    decisions.forEach((decision, i) => {
      if (
        !decision ||
        typeof decision !== "object" ||
        !finite(decision.time_s) ||
        decision.time_s < 0
      )
        throw new Error("decision " + i + " needs a finite nonnegative time_s");
      const fuel = decision.observation && decision.observation.fuel_model;
      const fireCells =
        decision.observation &&
        decision.observation.fire &&
        decision.observation.fire.cells;
      const cellSize = decision.observation && decision.observation.cell_size_m;
      if (cellSize != null && (!finite(cellSize) || cellSize <= 0))
        throw new Error(
          "decision " + i + " cell_size_m must be finite and positive",
        );
      if (fuel != null) matrixShape(fuel, "decision " + i + " fuel_model");
      if (fireCells != null)
        matrixShape(fireCells, "decision " + i + " observation.fire.cells");
    });
    if (replay.drops != null && !Array.isArray(replay.drops))
      throw new Error("drops must be an array when present");
    (replay.drops || []).forEach((drop, i) => {
      if (
        !drop ||
        typeof drop !== "object" ||
        !finite(drop.time_s) ||
        drop.time_s < 0
      )
        throw new Error("drop " + i + " needs a finite nonnegative time_s");
    });
    const grid = gridGeometry(replay);
    const declaredCell = replay.grid && replay.grid.cell_size_m;
    const observedCellSizes = (replay.decisions || [])
      .map((d) => d.observation && d.observation.cell_size_m)
      .filter((v) => v != null);
    if (
      declaredCell != null &&
      observedCellSizes.some((v) => v !== declaredCell)
    )
      throw new Error(
        "Recorded observation cell_size_m does not match grid.cell_size_m",
      );
    if (
      observedCellSizes.length &&
      observedCellSizes.some((v) => v !== observedCellSizes[0])
    )
      throw new Error(
        "Recorded observations use inconsistent cell_size_m values",
      );
    if (grid && replay.grid) {
      if (replay.grid.rows != null && replay.grid.rows !== grid.rows)
        throw new Error(
          "Declared grid.rows does not match recorded matrix dimensions",
        );
      if (replay.grid.cols != null && replay.grid.cols !== grid.cols)
        throw new Error(
          "Declared grid.cols does not match recorded matrix dimensions",
        );
    }
    const matrices = [replay.arrival_time_s];
    (replay.decisions || []).forEach((d) => {
      if (d.observation)
        matrices.push(
          d.observation.fuel_model,
          d.observation.fire && d.observation.fire.cells,
        );
    });
    if (grid)
      matrices.filter(Boolean).forEach((matrix) => {
        if (matrix.length !== grid.rows || matrix[0].length !== grid.cols)
          throw new Error(
            "Replay grid dimensions do not match recorded matrix dimensions",
          );
      });
    return replay;
  }
  function gridGeometry(replay) {
    if (!replay || typeof replay !== "object") return null;
    const arrival = matrixShape(replay.arrival_time_s, "arrival_time_s");
    const decisions = Array.isArray(replay.decisions) ? replay.decisions : [];
    const firstWithFuel = decisions.find(
      (d) => d && d.observation && Array.isArray(d.observation.fuel_model),
    );
    const fuel = firstWithFuel
      ? matrixShape(firstWithFuel.observation.fuel_model, "fuel_model")
      : null;
    const g = replay.grid || {};
    const rows = arrival
      ? arrival.rows
      : fuel
        ? fuel.rows
        : Number.isInteger(g.rows) && g.rows > 0
          ? g.rows
          : null;
    const cols = arrival
      ? arrival.cols
      : fuel
        ? fuel.cols
        : Number.isInteger(g.cols) && g.cols > 0
          ? g.cols
          : null;
    if (!rows || !cols) return null;
    let cellSize =
      finite(g.cell_size_m) && g.cell_size_m > 0 ? g.cell_size_m : null;
    for (const d of decisions)
      if (
        !cellSize &&
        d &&
        d.observation &&
        finite(d.observation.cell_size_m) &&
        d.observation.cell_size_m > 0
      )
        cellSize = d.observation.cell_size_m;
    return { rows, cols, cellSize };
  }
  function positionOnCanvas(position, geometry, width, height) {
    if (
      !Array.isArray(position) ||
      position.length < 2 ||
      !finite(position[0]) ||
      !finite(position[1]) ||
      !geometry ||
      !geometry.cellSize
    )
      return null;
    const x = ((position[0] / geometry.cellSize + 0.5) / geometry.cols) * width;
    const y =
      ((geometry.rows - position[1] / geometry.cellSize - 0.5) /
        geometry.rows) *
      height;
    if (x < 0 || y < 0 || x > width || y > height) return null;
    return { x, y };
  }
  function cellAtPoint(x, y, geometry, width, height) {
    if (
      !geometry ||
      width <= 0 ||
      height <= 0 ||
      x < 0 ||
      y < 0 ||
      x >= width ||
      y >= height
    )
      return null;
    return {
      row:
        geometry.rows -
        1 -
        Math.min(geometry.rows - 1, Math.floor((y / height) * geometry.rows)),
      col: Math.min(geometry.cols - 1, Math.floor((x / width) * geometry.cols)),
    };
  }
  function latestDecisionAt(replay, time) {
    time = replayTime(replay, time);
    let best = null;
    (replay && Array.isArray(replay.decisions) ? replay.decisions : []).forEach(
      (d, index) => {
        if (
          d &&
          finite(d.time_s) &&
          d.time_s <= time &&
          (!best ||
            d.time_s > best.decision.time_s ||
            (d.time_s === best.decision.time_s && index > best.index))
        )
          best = { decision: d, index };
      },
    );
    return best ? best.decision : null;
  }
  function observationAt(replay, time) {
    const d = latestDecisionAt(replay, time);
    return d && d.observation && typeof d.observation === "object"
      ? d.observation
      : null;
  }
  function reachedAreaAt(replay, time) {
    time = replayTime(replay, time);
    if (!replay || !Array.isArray(replay.arrival_time_s)) return null;
    const geom = gridGeometry(replay);
    if (!geom || !geom.cellSize) return null;
    let count = 0;
    replay.arrival_time_s.forEach((row) =>
      row.forEach((t) => {
        if (finite(t) && t <= time) count += 1;
      }),
    );
    return (count * geom.cellSize * geom.cellSize) / 10000;
  }
  function compatibleComparison(replays) {
    if (!Array.isArray(replays) || replays.length < 2)
      return { paired: false, reason: "Load at least two runs to compare." };
    const first = replays[0],
      hash = first && first.scenario_sha256,
      horizon = first && first.horizon_s,
      geom = gridGeometry(first);
    const same =
      !!hash &&
      finite(horizon) &&
      !!geom &&
      finite(geom.cellSize) &&
      replays.every((r) => {
        const g = gridGeometry(r);
        return (
          r &&
          r.scenario_sha256 === hash &&
          r.horizon_s === horizon &&
          g &&
          g.rows === geom.rows &&
          g.cols === geom.cols &&
          g.cellSize === geom.cellSize
        );
      });
    return same
      ? {
          paired: true,
          reason:
            "Matching scenario hash, horizon, and grid; final outcomes shown without ranking.",
        }
      : {
          paired: false,
          reason:
            "Exploratory comparison: runs are unmatched; no ranking or statistical claim is shown.",
        };
  }
  function comparisonRows(candidate, reference) {
    const sameRun = candidate === reference;
    const compatibility = compatibleComparison([reference, candidate]);
    const compatible = compatibility.paired && !sameRun;
    const area = (r) =>
      finite(r && r.affected_area_ha)
        ? r.affected_area_ha
        : finite(r && r.outcomes && r.outcomes.final_burned_ha)
          ? r.outcomes.final_burned_ha
          : null;
    const value = (r, keys) => {
      for (const key of keys)
        if (finite(r && r.aircraft_metrics && r.aircraft_metrics[key]))
          return r.aircraft_metrics[key];
      return null;
    };
    const pair = (key, label, unit, c, ref) => ({
      key,
      label,
      unit,
      candidate: c,
      reference: ref,
      delta: compatible && finite(c) && finite(ref) ? c - ref : null,
    });
    const flightTimeCandidate = value(candidate, ["flight_s"]);
    const flightTimeReference = value(reference, ["flight_s"]);
    const flightDistanceCandidate = value(candidate, [
      "total_flight_distance_km",
    ]);
    const flightDistanceReference = value(reference, [
      "total_flight_distance_km",
    ]);
    const flight =
      flightTimeCandidate != null || flightTimeReference != null
        ? pair(
            "flight_s",
            "Flight time",
            "s",
            flightTimeCandidate,
            flightTimeReference,
          )
        : pair(
            "flight_distance_km",
            "Flight distance",
            "km",
            flightDistanceCandidate,
            flightDistanceReference,
          );
    const rows = [
      pair(
        "area",
        "Final reached area",
        "ha",
        area(candidate),
        area(reference),
      ),
      pair(
        "water",
        "Water dropped",
        "L",
        value(candidate, ["water_dropped_l", "total_water_dropped_l"]),
        value(reference, ["water_dropped_l", "total_water_dropped_l"]),
      ),
      flight,
      pair(
        "drops",
        "Drops recorded",
        "",
        recordedDropCount(candidate),
        recordedDropCount(reference),
      ),
      pair(
        "rejections",
        "Rejected orders",
        "",
        value(candidate, ["rejected_orders", "rejected_orders_count"]),
        value(reference, ["rejected_orders", "rejected_orders_count"]),
      ),
    ];
    return {
      compatible,
      sameRun,
      reason: sameRun
        ? "Choose two different runs. Same-run selection has no comparative delta."
        : compatibility.reason,
      rows,
    };
  }
  function recordedDropCount(replay) {
    const recorded = replay && replay.aircraft_metrics;
    if (finite(recorded && recorded.drops_count)) return recorded.drops_count;
    return replay && Array.isArray(replay.drops) ? replay.drops.length : null;
  }
  function areaSeries(replay) {
    const unavailable = (reason) => ({
      available: false,
      reason,
      unit: "ha",
      horizon_s: null,
      points: [],
    });
    const geometry = gridGeometry(replay);
    if (!replay || !Array.isArray(replay.arrival_time_s))
      return unavailable("Arrival-time truth is unavailable.");
    if (!geometry || !finite(geometry.cellSize))
      return unavailable("Recorded grid resolution is unavailable.");
    if (!finite(replay.horizon_s) || replay.horizon_s < 0)
      return unavailable("Replay horizon is unavailable.");
    const perCellHa = (geometry.cellSize * geometry.cellSize) / 10000;
    const counts = new Map();
    replay.arrival_time_s.forEach((row) =>
      row.forEach((time) => {
        if (finite(time) && time >= 0 && time <= replay.horizon_s)
          counts.set(time, (counts.get(time) || 0) + 1);
      }),
    );
    const points = [{ time_s: 0, area_ha: (counts.get(0) || 0) * perCellHa }];
    let count = counts.get(0) || 0;
    Array.from(counts.keys())
      .filter((time) => time > 0)
      .sort((a, b) => a - b)
      .forEach((time) => {
        count += counts.get(time);
        points.push({ time_s: time, area_ha: count * perCellHa });
      });
    if (points[points.length - 1].time_s < replay.horizon_s)
      points.push({ time_s: replay.horizon_s, area_ha: count * perCellHa });
    return {
      available: true,
      reason: null,
      unit: "ha",
      horizon_s: replay.horizon_s,
      points,
    };
  }
  function replayTime(replay, time) {
    return replay && finite(replay.horizon_s)
      ? Math.max(0, Math.min(time, replay.horizon_s))
      : time;
  }
  function fitGridToFrame(geometry, width, height) {
    if (!geometry || ![geometry.rows, geometry.cols, width, height].every(value => finite(value) && value > 0)) return null;
    const cell = Math.min(width / geometry.cols, height / geometry.rows);
    return { width: cell * geometry.cols, height: cell * geometry.rows };
  }
  const api = {
    fitGridToFrame,
    latestDecisionAt,
    gridGeometry,
    positionOnCanvas,
    cellAtPoint,
    reachedAreaAt,
    compatibleComparison,
    observationAt,
    validateReplay,
    replayTime,
    comparisonRows,
    areaSeries,
    validateModelRecord,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  const $ = (s) => document.querySelector(s),
    $$ = (s) => Array.from(document.querySelectorAll(s));
  const dom = {
    runList: $("#runList"),
    scenarioSelect: $("#scenarioSelect"),
    scenarioContext: $("#scenarioContext"),
    runCount: $("#runCount"),
    emptyRuns: $("#emptyRuns"),
    metadata: $("#runMetadata"),
    mapViews: $("#mapViews"),
    mapTitle: $("#mapTitle"),
    mapSubtitle: $("#mapSubtitle"),
    runKicker: $("#runKicker"),
    truthStatus: $("#truthStatus"),
    scale: $("#scaleMark"),
    coord: $("#coordinateNote"),
    legend: $("#legend"),
    timeDisplay: $("#timeDisplay"),
    timeContext: $("#timeContext"),
    timeRange: $("#timeRange"),
    ticks: $("#timelineTicks"),
    play: $("#playButton"),
    eventRows: $("#eventRows"),
    eventCount: $("#eventCount"),
    emptyEvents: $("#emptyEvents"),
    summary: $("#currentSummary"),
    cell: $("#cellInspector"),
    outcomes: $("#outcomeRows"),
    comparison: $("#comparisonNote"),
    notice: $("#notice"),
    file: $("#fileInput"),
    tabButtons: $$(".tab-button"),
    tabPages: $$(".tab-page"),
    mapBlock: $("#mapBlock"),
    mapHostReplay: $("#mapHostReplay"),
    mapHostCompare: $("#mapHostCompare"),
    timelinePanel: $("#timelinePanel"),
    candidateRun: $("#candidateRun"),
    referenceRun: $("#referenceRun"),
    chart: $("#areaChart"),
    chartLegend: $("#chartLegend"),
    controllerSelect: $("#controllerSelect"),
    physicalSelect: $("#physicalSelect"),
    controllerModel: $("#controllerModel"),
    physicalModel: $("#physicalModel"),
    architecture: $("#architectureDiagram"),
    recordRunSelect: $("#recordRunSelect"),
    recordOutcomes: $("#recordOutcomes"),
    layerDisclosure: $(".layer-disclosure"),
    compareLayerHost: $("#compareLayerHost"),
    controllerLink: $("#controllerLink"),
    physicalLink: $("#physicalLink"),
  };
  const state = {
    runs: [],
    selected: 0,
    renderedSelected: -1,
    renderedRunCount: -1,
    compare: false,
    activeTab: "replay",
    candidateRun: 0,
    referenceRun: 1,
    models: [],
    selectedController: 0,
    selectedPhysical: 0,
    time: 0,
    playing: false,
    speed: 1,
    selectedCell: null,
    layers: {
      truth: true,
      observations: true,
      candidates: true,
      aircraft: true,
      grid: true,
    },
    timer: null,
    noticeTimer: null,
  };
  function escText(value) {
    return value == null || value === "" ? "Unavailable" : String(value);
  }
  function shortName(r, index) {
    const names = {ppo: "PPO", heuristic: "Downwind heuristic",
      containment_heuristic: "Containment heuristic", no_intervention: "No intervention", random_valid: "Random valid"};
    return String(
      names[r.policy] || r.policy ||
        r.scenario_id ||
        r.replay_schema_version ||
        r.schema_version ||
        "Replay " + (index + 1),
    );
  }
  function scenarioKey(r, index) {
    return r.scenario_id || "Imported episode " + (index + 1);
  }
  function scenarioLabel(r, index) {
    return String(scenarioKey(r, index)).replace(/^discrimination-(development|validation|reserved)-(\d+)$/, (_, split, number) =>
      "Discrimination · " + split[0].toUpperCase() + split.slice(1) + " " + number);
  }
  function metric(r, integratedKey, geographicKey) {
    const a = r.aircraft_metrics || {};
    return a[integratedKey] != null
      ? a[integratedKey]
      : a[geographicKey] != null
        ? a[geographicKey]
        : null;
  }
  function duration(seconds) {
    if (!finite(seconds)) return "Unavailable";
    const s = Math.max(0, Math.floor(seconds));
    const h = Math.floor(s / 3600),
      m = Math.floor((s % 3600) / 60),
      sec = s % 60;
    return h
      ? String(h).padStart(2, "0") +
          ":" +
          String(m).padStart(2, "0") +
          ":" +
          String(sec).padStart(2, "0")
      : String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
  }
  function currentReplay() {
    const index = state.compare ? state.candidateRun : state.selected;
    return state.runs[index] || null;
  }
  function maxTime() {
    const eligible = state.compare
      ? [state.runs[state.candidateRun], state.runs[state.referenceRun]]
      : [currentReplay()];
    return eligible.reduce(
      (m, r) => (r && finite(r.horizon_s) ? Math.max(m, r.horizon_s) : m),
      0,
    );
  }
  function showNotice(text, error) {
    dom.notice.textContent = text;
    dom.notice.classList.toggle("error", !!error);
    dom.notice.classList.add("visible");
    clearTimeout(state.noticeTimer);
    state.noticeTimer = setTimeout(
      () => dom.notice.classList.remove("visible"),
      3800,
    );
  }
  function renderRuns() {
    dom.runList.replaceChildren();
    dom.scenarioSelect.replaceChildren();
    dom.runCount.textContent = state.runs.length;
    dom.emptyRuns.hidden = state.runs.length > 0;
    const selected = state.runs[state.selected];
    const selectedKey = selected && scenarioKey(selected, state.selected);
    const scenarios = new Set();
    state.runs.forEach((r, i) => {
      const key = scenarioKey(r, i);
      if (!scenarios.has(key)) {
        scenarios.add(key);
        const option = document.createElement("option");
        option.value = key;
        option.textContent = scenarioLabel(r, i);
        dom.scenarioSelect.append(option);
      }
      if (key === selectedKey) {
        const option = document.createElement("option");
        option.className = "run-option";
        option.value = String(i);
        option.textContent = shortName(r, i);
        dom.runList.append(option);
      }
    });
    if (selected) {
      dom.scenarioSelect.value = selectedKey;
      dom.runList.value = String(state.selected);
    }
    dom.scenarioSelect.disabled = dom.runList.disabled = !state.runs.length;
    [dom.candidateRun, dom.referenceRun, dom.recordRunSelect].forEach(
      (select) => {
        if (!select) return;
        select.replaceChildren();
        state.runs.forEach((r, i) => {
          const option = document.createElement("option");
          option.value = String(i);
          option.textContent = shortName(r, i) + " · " + scenarioLabel(r, i);
          select.append(option);
        });
        const preferred =
          select === dom.candidateRun
            ? state.candidateRun
            : select === dom.referenceRun
              ? state.referenceRun
              : state.selected;
        if (state.runs.length)
          select.value = String(
            Math.max(0, Math.min(preferred, state.runs.length - 1)),
          );
        select.disabled = state.runs.length === 0;
      },
    );
  }
  function addMeta(dl, name, value) {
    const row = document.createElement("div"),
      dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = name;
    dd.textContent = escText(value);
    row.append(dt, dd);
    dl.append(row);
  }
  function renderMetadata(r) {
    dom.metadata.replaceChildren();
    dom.recordOutcomes.replaceChildren();
    if (!r) return;
    const g = gridGeometry(r);
    const obs = observationAt(r, state.time);
    const size = (g && g.cellSize) || (obs && obs.cell_size_m);
    addMeta(
      dom.metadata,
      "Replay schema",
      r.replay_schema_version || r.schema_version,
    );
    addMeta(dom.metadata, "Scenario", r.scenario_id);
    addMeta(dom.metadata, "Policy", r.policy);
    addMeta(dom.metadata, "Seed", r.seed);
    addMeta(dom.metadata, "Policy seed", r.policy_seed);
    addMeta(dom.metadata, "Provenance", r.provenance);
    addMeta(dom.metadata, "Recorded controller ID", r.controller_id);
    addMeta(dom.metadata, "Checkpoint SHA-256", r.checkpoint_sha256);
    addMeta(dom.metadata, "Physical model ID", r.physical_model_id);
    addMeta(dom.metadata, "Scenario SHA-256", r.scenario_sha256);
    addMeta(dom.metadata, "CRS", r.crs);
    addMeta(
      dom.metadata,
      "Grid",
      g ? g.rows + " rows × " + g.cols + " cols" : null,
    );
    addMeta(dom.metadata, "Cell size", size ? size + " m" : null);
    addMeta(dom.metadata, "Time origin", r.time_origin_utc);
    addMeta(
      dom.metadata,
      "Units",
      r.units
        ? Object.entries(r.units)
            .map(([k, v]) => k + ": " + v)
            .join(", ")
        : null,
    );
    if (r) {
      const area = finite(r.affected_area_ha) ? r.affected_area_ha : finite(r.outcomes && r.outcomes.final_burned_ha) ? r.outcomes.final_burned_ha : null;
      const flight = finite(metric(r, "flight_s", "flight_s"))
        ? [metric(r, "flight_s", "flight_s"), "s"]
        : finite(r.aircraft_metrics && r.aircraft_metrics.total_flight_distance_km)
          ? [r.aircraft_metrics.total_flight_distance_km, "km"]
          : [null, ""];
      const entries = [
        ["Final reached area", area, "ha"],
        ["Water dropped", metric(r, "water_dropped_l", "total_water_dropped_l"), "L"],
        [flight[1] === "s" ? "Flight time" : flight[1] === "km" ? "Flight distance" : "Flight", flight[0], flight[1]],
        ["Drops recorded", recordedDropCount(r), ""],
        ["Rejected orders", metric(r, "rejected_orders", "rejected_orders_count"), ""],
      ];
      entries.forEach(([label, value, unit]) => {
        const item = document.createElement("div");
        const title = document.createElement("span");
        const result = document.createElement("strong");
        item.className = "record-summary-item";
        title.textContent = label;
        result.textContent = finite(value)
          ? fmt(value) + (unit ? " " + unit : "")
          : "Unavailable";
        item.append(title, result);
        dom.recordOutcomes.append(item);
      });
    }
  }
  function validateModelRecord(record) {
    if (!record || typeof record !== "object" || Array.isArray(record))
      throw new Error("Model record must be a JSON object");
    if (record.schema_version !== "nightjar-viewer-model-0.1")
      throw new Error(
        "Model record schema_version must be nightjar-viewer-model-0.1",
      );
    if (typeof record.id !== "string" || !record.id)
      throw new Error("Model record id is required");
    if (!["controller", "physical"].includes(record.role))
      throw new Error("Model record role must be controller or physical");
    for (const key of ["config", "checkpoint", "training_config"]) {
      if (record[key] != null && (typeof record[key] !== "object" || Array.isArray(record[key]))) throw new Error("Model record " + key + " must be an object when present");
    }
    for (const key of ["details", "source_files"]) {
      if (record[key] != null && !Array.isArray(record[key])) throw new Error("Model record " + key + " must be an array when present");
    }
    (record.details || []).forEach((item, index) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error("Model details item " + index + " must be an object");
      if (item.label != null && typeof item.label !== "string") throw new Error("Model detail label must be a string");
    });
    (record.source_files || []).forEach((item, index) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error("Model source_files item " + index + " must be an object");
      for (const key of ["file", "sha256"]) if (item[key] != null && typeof item[key] !== "string") throw new Error("Model source file " + key + " must be a string");
    });
    if (record.checkpoint) {
      if (record.checkpoint.filename != null && typeof record.checkpoint.filename !== "string") throw new Error("Checkpoint filename must be a string");
      if (record.checkpoint.sha256 != null && typeof record.checkpoint.sha256 !== "string") throw new Error("Checkpoint sha256 must be a string");
      if (record.checkpoint.bytes != null && (!finite(record.checkpoint.bytes) || record.checkpoint.bytes < 0)) throw new Error("Checkpoint bytes must be finite and nonnegative");
    }
    for (const key of ["parameter_count", "tensor_count"]) {
      if (
        record[key] != null &&
        (!finite(record[key]) || record[key] < 0)
      ) {
        throw new Error(
          "Model " + key + " must be finite and nonnegative",
        );
      }
    }
    if (record.config) {
      for (const key of ["grid_rows", "grid_cols", "in_channels", "vector_dim", "hidden_dim", "action_dim", "lstm_hidden_dim"]) {
        if (
          record.config[key] != null &&
          (!Number.isInteger(record.config[key]) || record.config[key] <= 0)
        ) {
          throw new Error("Model config " + key + " must be a positive integer");
        }
      }
      if (record.config.recurrent != null && typeof record.config.recurrent !== "boolean") throw new Error("Model config recurrent must be boolean");
    }
    return record;
  }
  function addModel(record) {
    validateModelRecord(record);
    state.models.push(record);
  }
  function modelOptions(select, role, selected) {
    select.replaceChildren();
    const records = state.models
      .map((model, index) => ({ model, index }))
      .filter(({ model }) => model.role === role);
    records.forEach(({ model, index }) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = model.name || model.id;
      select.append(option);
    });
    select.disabled = records.length === 0;
    if (records.length)
      select.value = String(
        records.some(({ index }) => index === selected)
          ? selected
          : records[0].index,
      );
    return records;
  }
  function appendRecordLine(container, label, value) {
    const row = document.createElement("div");
    row.className = "model-record-row";
    const key = document.createElement("dt"),
      val = document.createElement("dd");
    key.textContent = label;
    const text =
      value == null || value === ""
        ? "Unavailable"
        : typeof value === "string"
          ? value
          : JSON.stringify(value);
    if (text.length > 180) {
      const disclosure = document.createElement("details");
      disclosure.className = "model-value-disclosure";
      const summary = document.createElement("summary");
      summary.textContent = "Expand recorded value";
      const pre = document.createElement("pre");
      pre.textContent = text;
      disclosure.append(summary, pre);
      val.append(disclosure);
    } else {
      val.textContent = text;
    }
    row.append(key, val);
    container.append(row);
  }
  function renderModelDetails(container, record, role) {
    container.replaceChildren();
    if (!record) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent =
        role === "controller"
          ? "No checkpoint record is available. Import its JSON inspection record to review architecture and provenance."
          : "No physical simulation record is loaded. Import a viewer model record or generate a workspace with a recorded scenario.";
      container.append(empty);
      return;
    }
    const heading = document.createElement("h3");
    heading.textContent = record.name || record.id;
    container.append(heading);
    const config = record.config || {};
    const status = document.createElement("p");
    status.className = "model-status";
    status.textContent = role === "controller"
      ? [record.architecture || "Architecture unavailable", config.recurrent === true ? "Recurrent" : config.recurrent === false ? "Feedforward" : "Memory unavailable",
          config.action_mode === "tactical" ? "Tactical actions" : config.action_mode === "spatial" ? "Spatial actions" : "Actions unavailable"].join(" · ")
      : record.status || "Status unavailable";
    container.append(status);
    const fields = document.createElement("dl");
    fields.className = "model-fields";
    if (role === "controller") {
      appendRecordLine(fields, "Checkpoint", record.checkpoint && record.checkpoint.filename);
      appendRecordLine(fields, "Parameters", finite(record.parameter_count) ? record.parameter_count.toLocaleString() : null);
      appendRecordLine(fields, "Input grid", finite(config.grid_rows) && finite(config.grid_cols) ? config.grid_rows + " × " + config.grid_cols + " cells" : null);
      appendRecordLine(fields, "Raster channels", config.in_channels);
      appendRecordLine(fields, "Vector features", config.vector_dim);
      appendRecordLine(fields, "Action space", finite(config.action_dim) ? config.action_dim + " · " + escText(config.action_mode) : null);
      appendRecordLine(fields, "Selected seed", record.selected_seed);
    } else {
      appendRecordLine(fields, "Grid", finite(config.grid_rows) && finite(config.grid_cols) ? config.grid_rows + " × " + config.grid_cols + " cells" : null);
      appendRecordLine(fields, "Cell resolution", finite(config.cell_size_m) ? fmt(config.cell_size_m) + " m" : null);
      appendRecordLine(fields, "Episode horizon", finite(config.horizon_s) ? duration(config.horizon_s) : null);
      const inputLabels = new Map([
        ["Fuel codes", "Surface fuel"], ["Midflame wind to [east, north] (m/s)", "Wind E / N (m/s)"],
        ["Dead-fuel moisture fraction range", "Fuel moisture (fraction)"], ["Recorded elevation range (m)", "Elevation range (m)"],
        ["Decision interval (s)", "Decision interval (s)"], ["Water-effect radius (m)", "Water radius (m)"],
        ["Water-effect assumption", "Suppression assumption"],
      ]);
      (record.details || []).filter(detail => inputLabels.has(detail.label)).forEach(detail =>
        appendRecordLine(fields, inputLabels.get(detail.label), Array.isArray(detail.value)
          ? detail.value.map(value => typeof value === "number" ? fmt(value) : String(value)).join(" / ")
          : detail.value));
    }
    container.append(fields);
    const provenance = document.createElement("details");
    provenance.className = "model-provenance";
    const summary = document.createElement("summary");
    summary.textContent = "Provenance & configuration";
    const metadata = document.createElement("dl");
    metadata.className = "model-fields";
    appendRecordLine(metadata, "Record schema", record.schema_version);
    appendRecordLine(metadata, "Record ID", record.id);
    appendRecordLine(metadata, "Created at UTC", record.created_at_utc);
    if (record.scenario_sha256) appendRecordLine(metadata, "Scenario SHA-256", record.scenario_sha256);
    if (role === "controller") {
      appendRecordLine(metadata, "Checkpoint SHA-256", record.checkpoint && record.checkpoint.sha256);
      appendRecordLine(metadata, "Checkpoint bytes", record.checkpoint && record.checkpoint.bytes);
      appendRecordLine(metadata, "Tensors", record.tensor_count);
      appendRecordLine(metadata, "Loader status", record.status);
    }
    Object.keys(config).sort().forEach(key => appendRecordLine(metadata, "Config · " + key, config[key]));
    if (record.training_config != null) appendRecordLine(metadata, "Training config", record.training_config);
    provenance.append(summary, metadata);
    if (Array.isArray(record.details) && record.details.length) {
      const details = document.createElement("dl");
      details.className = "model-fields model-extra-fields";
      const disclosure = document.createElement("details");
      disclosure.className = "model-extra-disclosure";
      const detailsHeading = document.createElement("summary");
      detailsHeading.textContent = "Recorded details · " + record.details.length;
      disclosure.append(detailsHeading, details);
      record.details.forEach(detail => appendRecordLine(details, detail.label || "Detail", detail.value));
      provenance.append(disclosure);
    }
    if (Array.isArray(record.source_files) && record.source_files.length) {
      const sources = document.createElement("details");
      sources.className = "source-fingerprints";
      const label = document.createElement("summary");
      label.textContent = "Source fingerprints · " + record.source_files.length;
      sources.append(label);
      record.source_files.forEach(source => {
        const line = document.createElement("p");
        line.textContent = (source.file || "File unavailable") + " · SHA-256 " + (source.sha256 || "Unavailable");
        sources.append(line);
      });
      provenance.append(sources);
    }
    container.append(provenance);
  }

  function renderModels() {
    const controllers = modelOptions(
      dom.controllerSelect,
      "controller",
      state.selectedController,
    );
    const physical = modelOptions(
      dom.physicalSelect,
      "physical",
      state.selectedPhysical,
    );
    state.selectedController = controllers.length
      ? Number(dom.controllerSelect.value)
      : -1;
    state.selectedPhysical = physical.length
      ? Number(dom.physicalSelect.value)
      : -1;
    const controller = state.models[state.selectedController] || null;
    const physicalRecord = state.models[state.selectedPhysical] || null;
    const replay = state.runs[state.selected] || null;
    const controllerLinked = !!(
      replay &&
      controller &&
      ((replay.controller_id && replay.controller_id === controller.id) ||
        (replay.checkpoint_sha256 &&
          controller.checkpoint &&
          replay.checkpoint_sha256 === controller.checkpoint.sha256))
    );
    const physicalLinked = !!(
      replay &&
      physicalRecord &&
      ((replay.physical_model_id &&
        replay.physical_model_id === physicalRecord.id) ||
        (replay.scenario_sha256 &&
          replay.scenario_sha256 === physicalRecord.scenario_sha256))
    );
    dom.controllerLink.textContent = controller
      ? controllerLinked
        ? "Recorded controller for selected replay."
        : "Inspected checkpoint · not used in the selected replay."
      : "No controller association available.";
    dom.physicalLink.textContent = physicalRecord
      ? physicalLinked
        ? "Scenario inputs for the selected replay."
        : "Physical inputs belong to another scenario."
      : "No physical-model association available.";
    renderModelDetails(dom.controllerModel, controller, "controller");
    renderModelDetails(dom.physicalModel, physicalRecord, "physical");
    dom.architecture.replaceChildren();
    if (!controller || controller.architecture !== "SpatialActorCritic") return;
    const config = controller.config || {};
    dom.architecture.setAttribute("aria-label", "SpatialActorCritic with parallel raster and observed-vector encoders");
    const branches = document.createElement("div");
    branches.className = "architecture-branches";
    const branchNode = (title, detail) => {
      const block = document.createElement("div"), heading = document.createElement("strong"), body = document.createElement("span");
      block.className = "architecture-node"; heading.textContent = title; body.textContent = detail; block.append(heading, body); return block;
    };
    const raster = document.createElement("div"), vector = document.createElement("div");
    raster.className = "architecture-branch"; vector.className = "architecture-branch";
    raster.append(
      branchNode(
        "Raster CNN",
        finite(config.grid_rows) &&
          finite(config.grid_cols) &&
          finite(config.in_channels)
          ? config.grid_rows +
              " × " +
              config.grid_cols +
              " × " +
              config.in_channels +
              " channels"
          : "Input dimensions unavailable",
      ),
    );
    vector.append(branchNode("Observed-vector MLP", finite(config.vector_dim) ? config.vector_dim + " features" : "Input dimension unavailable"));
    branches.append(raster, vector); dom.architecture.append(branches);
    const join = document.createElement("p"); join.className = "architecture-join"; join.textContent = "Parallel encoders concatenate"; dom.architecture.append(join);
    dom.architecture.append(branchNode("Fusion", finite(config.hidden_dim) ? config.hidden_dim + " hidden units" : "Width unavailable"));
    dom.architecture.append(
      branchNode(
        config.recurrent === true
          ? "LSTM"
          : config.recurrent === false
            ? "Feedforward path"
            : "Memory",
        config.recurrent === true
          ? escText(config.lstm_hidden_dim) + " hidden units"
          : config.recurrent === false
            ? "No recurrent layer"
            : "Configuration unavailable",
      ),
    );
    dom.architecture.append(branchNode("Actor + critic", finite(config.action_dim) ? config.action_dim + " actions · masked policy output" : "Action dimension unavailable"));
    dom.controllerModel.insertBefore(dom.architecture, dom.controllerModel.querySelector(".model-fields"));
  }
  function exportModelRecord(role) {
    const index =
      role === "controller" ? state.selectedController : state.selectedPhysical;
    const record = state.models[index];
    if (!record) {
      showNotice("No " + role + " record is selected.", true);
      return;
    }
    const name = String(record.id || role)
      .replace(/[^a-z0-9_-]+/gi, "-")
      .toLowerCase();
    download(
      new Blob([JSON.stringify(record, null, 2) + "\n"], {
        type: "application/json",
      }),
      name + ".json",
    );
  }
  function cellSet(items) {
    const set = new Set();
    if (!Array.isArray(items)) return set;
    items.forEach((v) => {
      if (Array.isArray(v) && Number.isInteger(v[0]) && Number.isInteger(v[1]))
        set.add(v[0] + "," + v[1]);
    });
    return set;
  }
  function observationCellAt(obs, row, col, time) {
    const report =
      obs && obs.fire && obs.fire.cells && obs.fire.cells[row]
        ? obs.fire.cells[row][col]
        : null;
    return report &&
      (!finite(report.observed_at_s) || report.observed_at_s <= time)
      ? report
      : null;
  }
  function makeCanvas(r, index, comparison, paneIndex) {
    const geom = gridGeometry(r);
    if (!geom) return null;
    const pane = document.createElement("div");
    pane.className = "map-pane";
    const title = document.createElement("div");
    title.className = "pane-title";
    title.textContent =
      shortName(r, index) +
      (comparison && finite(r.horizon_s) && state.time >= r.horizon_s
        ? " · complete"
        : "");
    const wrap = document.createElement("div");
    wrap.className = "map-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "map-canvas";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.setAttribute(
      "aria-label",
      shortName(r, index) + " grid map; use arrow keys to inspect cells",
    );
    canvas.style.setProperty("--grid-ratio", geom.cols + " / " + geom.rows);
    canvas.addEventListener("click", (ev) => {
      const rect = canvas.getBoundingClientRect();
      const c = cellAtPoint(
        ev.clientX - rect.left,
        ev.clientY - rect.top,
        geom,
        rect.width,
        rect.height,
      );
      if (c) {
        state.selected = index;
        if (state.compare) {
          if (paneIndex === 0) state.candidateRun = index;
          else state.referenceRun = index;
        }
        state.selectedCell = c;
        renderEverything();
      }
    });
    canvas.addEventListener("keydown", (ev) => {
      if (!state.selectedCell)
        state.selectedCell = {
          row: Math.floor(geom.rows / 2),
          col: Math.floor(geom.cols / 2),
        };
      let { row, col } = state.selectedCell;
      if (ev.key === "ArrowUp") row++;
      else if (ev.key === "ArrowDown") row--;
      else if (ev.key === "ArrowLeft") col--;
      else if (ev.key === "ArrowRight") col++;
      else return;
      ev.preventDefault();
      ev.stopPropagation();
      state.selected = index;
      if (state.compare) {
        if (paneIndex === 0) state.candidateRun = index;
        else state.referenceRun = index;
      }
      state.selectedCell = {
        row: Math.max(0, Math.min(geom.rows - 1, row)),
        col: Math.max(0, Math.min(geom.cols - 1, col)),
      };
      renderEverything();
      requestAnimationFrame(() => {
        const canvases = $$(".map-canvas");
        const target = canvases[paneIndex || 0];
        if (target) target.focus();
      });
    });
    wrap.append(canvas);
    pane.append(title, wrap);
    return pane;
  }
  function sizeCanvas(canvas, r, index) {
    const geom = gridGeometry(r),
      wrap = canvas.parentElement;
    if (!geom || !wrap) return;
    const inset = getComputedStyle(wrap);
    const cssW = Math.max(1, wrap.clientWidth - parseFloat(inset.paddingLeft) - parseFloat(inset.paddingRight)),
      cssH = Math.max(1, wrap.clientHeight - parseFloat(inset.paddingTop) - parseFloat(inset.paddingBottom));
    const { width: w, height: h } = fitGridToFrame(geom, cssW, cssH);
    const dpr = Math.min(root.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawGrid(ctx, r, geom, w, h, index);
  }
  function drawGrid(ctx, r, g, width, height, runIndex) {
    const currentTime = replayTime(r, state.time);
    const cw = width / g.cols,
      ch = height / g.rows;
    ctx.fillStyle = "#f7f6f0";
    ctx.fillRect(0, 0, width, height);
    const obs = observationAt(r, currentTime);
    const fuel = obs && obs.fuel_model;
    const arrivals = r.arrival_time_s;
    const drops = (r.drops || []).filter((d) => d.time_s <= currentTime);
    const truth = state.layers.truth ? arrivals : null;
    const observations =
      state.layers.observations && obs && obs.fire && obs.fire.cells
        ? obs.fire.cells
        : null;
    const candidates = state.layers.candidates
      ? cellSet(latestDecisionAt(r, currentTime)?.candidate_cells)
      : new Set();
    const treated = new Set();
    drops.forEach((d) => {
      const cells = d.newly_blocked_cells || d.treated_cells;
      if (Array.isArray(cells)) cellSet(cells).forEach((v) => treated.add(v));
    });
    for (let row = 0; row < g.rows; row++)
      for (let col = 0; col < g.cols; col++) {
        const x = col * cw,
          y = height - (row + 1) * ch,
          key = row + "," + col;
        let fill = "#f7f6f0";
        const f = fuel && fuel[row] ? fuel[row][col] : null;
        if (
          f != null &&
          (f === 0 ||
            f === "0" ||
            String(f).toUpperCase() === "NB" ||
            String(f).toUpperCase() === "NONE" ||
            String(f).toLowerCase().includes("nonburnable"))
        )
          fill = "#e1e3dc";
        if (
          truth &&
          truth[row] &&
          finite(truth[row][col]) &&
          truth[row][col] <= currentTime
        )
          fill = "#e8c6b8";
        if (treated.has(key)) fill = "#b7d7e4";
        ctx.fillStyle = fill;
        ctx.fillRect(
          x + 0.35,
          y + 0.35,
          Math.max(0, cw - 0.7),
          Math.max(0, ch - 0.7),
        );
        const delivered = state.layers.observations
          ? observationCellAt(obs, row, col, currentTime)
          : null;
        if (delivered && delivered.fire_detected === true) {
          ctx.strokeStyle = "#237d72";
          ctx.lineWidth = Math.max(1, Math.min(2, cw * 0.08, ch * 0.08));
          ctx.strokeRect(
            x + ctx.lineWidth,
            y + ctx.lineWidth,
            Math.max(0, cw - ctx.lineWidth * 2),
            Math.max(0, ch - ctx.lineWidth * 2),
          );
        }
        if (candidates.has(key)) {
          ctx.save();
          ctx.strokeStyle = "#b88731";
          ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 2]);
          ctx.strokeRect(
            x + 1.5,
            y + 1.5,
            Math.max(0, cw - 3),
            Math.max(0, ch - 3),
          );
          ctx.restore();
        }
        if (
          state.selectedCell &&
          state.selectedCell.row === row &&
          state.selectedCell.col === col
        ) {
          ctx.strokeStyle = "#273b31";
          ctx.lineWidth = 2;
          ctx.strokeRect(
            x + 1,
            y + 1,
            Math.max(0, cw - 2),
            Math.max(0, ch - 2),
          );
        }
      }
    if (state.layers.grid && cw >= 3 && ch >= 3) {
      ctx.strokeStyle = "rgba(95,110,99,.34)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (let c = 0; c <= g.cols; c++) {
        const x = c * cw;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let row = 0; row <= g.rows; row++) {
        const y = row * ch;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();
    }
    if (state.layers.aircraft) {
      const seen = [];
      (r.decisions || [])
        .filter((d) => d.time_s <= currentTime)
        .slice()
        .sort((a, b) => a.time_s - b.time_s)
        .forEach((d) => {
          const p =
            d.observation &&
            d.observation.aircraft &&
            d.observation.aircraft.position_m;
          const xy = positionOnCanvas(p, g, width, height);
          if (xy) seen.push({ point: xy, time: d.time_s });
        });
      // Each recorded point is drawn independently; no path is implied between snapshots.
      const latest = seen.length ? seen[seen.length - 1].time : null;
      seen.forEach(({ point: p, time }, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, time === latest ? 4 : 2.7, 0, Math.PI * 2);
        ctx.fillStyle = time === latest ? "#3777a5" : "#7599ad";
        ctx.fill();
        ctx.strokeStyle = "#f7f6f0";
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }
    const source = positionOnCanvas(
      obs && obs.water_source_m,
      g,
      width,
      height,
    );
    if (source) {
      ctx.fillStyle = "#3777a5";
      ctx.fillRect(source.x - 4, source.y - 4, 8, 8);
      ctx.strokeStyle = "#fffefa";
      ctx.lineWidth = 1;
      ctx.strokeRect(source.x - 4, source.y - 4, 8, 8);
    }
  }
  function renderMap() {
    const r = currentReplay();
    dom.mapViews.replaceChildren();
    dom.mapViews.classList.toggle(
      "comparing",
      state.compare && state.runs.length > 1,
    );
    if (!r) {
      const p = document.createElement("div");
      p.className = "map-placeholder";
      p.textContent = "Import or select a replay to inspect its grid.";
      dom.mapViews.append(p);
      return;
    }
    const list = state.compare
      ? [
          [state.runs[state.candidateRun], state.candidateRun],
          [state.runs[state.referenceRun], state.referenceRun],
        ].filter(([item]) => item)
      : [[r, state.selected]];
    const pendingCanvases = [];
    list.forEach(([item, i], paneIndex) => {
      const pane = makeCanvas(item, i, list.length > 1, paneIndex);
      if (pane) {
        dom.mapViews.append(pane);
        pendingCanvases.push([pane.querySelector("canvas"), item, i]);
      }
    });
    // Measure after both panels exist; adding the reference changes candidate width.
    pendingCanvases.forEach(([canvas, item, i]) => sizeCanvas(canvas, item, i));
    if (!dom.mapViews.childElementCount) {
      const placeholder = document.createElement("p");
      placeholder.className = "map-placeholder";
      placeholder.textContent =
        "Grid unavailable. Inspect the recorded metadata and final outcomes.";
      dom.mapViews.append(placeholder);
    }
    const g = gridGeometry(r);
    const truthAvailable = Array.isArray(r.arrival_time_s);
    dom.truthStatus.textContent = truthAvailable
      ? "Retrospective simulator truth: cells reached by the selected time."
      : "Truth layer unavailable in this replay schema.";
    dom.truthStatus.classList.toggle("truth-unavailable", !truthAvailable);
    dom.scale.replaceChildren();
    dom.scale.hidden = state.compare;
    if (state.compare && state.runs.length > 1) {
      dom.scale.textContent = "";
    } else if (g && g.cellSize) {
      const canvas = dom.mapViews.querySelector("canvas");
      const cellPx = canvas ? canvas.clientWidth / g.cols : 0;
      const span =
        cellPx > 0 ? Math.max(1, Math.min(g.cols, Math.floor(72 / cellPx))) : 0;
      if (span) {
        const mark = document.createElement("span");
        mark.className = "scale-line";
        mark.style.width = span * cellPx + "px";
        const label = document.createElement("span");
        label.textContent = span * g.cellSize + " m";
        dom.scale.append(mark, label);
      } else dom.scale.textContent = "Scale unavailable";
    } else dom.scale.textContent = "Scale unavailable";
    dom.coord.textContent = g
      ? "North-up cell grid · aircraft points are recorded local position_m cell-anchor proxies."
      : "Grid dimensions unavailable in this replay.";
    const entries = [];
    if (state.layers.truth && truthAvailable)
      entries.push(["Reached area", "#e8c6b8"]);
    if (state.layers.observations)
      entries.push(["Delivered fire report", "#91c9b8"]);
    if (state.layers.candidates) entries.push(["Candidate cell", "#f2e6c7"]);
    if (state.layers.aircraft)
      entries.push(["Recorded aircraft point", "#3777a5"]);
    if ((r.drops || []).some((d) => d.time_s <= state.time))
      entries.push(["Recorded treated cells", "#b7d7e4"]);
    const observation = observationAt(r, state.time);
    if (
      positionOnCanvas(observation && observation.water_source_m, g, 100, 100)
    )
      entries.push(["Recorded water source", "#3777a5"]);
    if (
      observation &&
      Array.isArray(observation.fuel_model) &&
      observation.fuel_model
        .flat()
        .some((f) => f === 0 || f === "0" || f === "NONE")
    )
      entries.push(["Fuel code 0 / NONE", "#e1e3dc"]);
    dom.legend.replaceChildren();
    entries.forEach(([label, color]) => {
      const el = document.createElement("span");
      el.className = "legend-item";
      const chip = document.createElement("i");
      chip.className = "legend-chip";
      chip.style.background = color;
      if (label === "Delivered fire report") {
        chip.style.background = "transparent";
        chip.style.border = "2px solid #237d72";
      } else if (label === "Candidate cell") {
        chip.style.background = "transparent";
        chip.style.border = "1px dashed #b88731";
      } else if (label === "Recorded aircraft point") {
        chip.style.borderRadius = "50%";
      }
      const text = document.createElement("span");
      text.textContent = label;
      el.append(chip, text);
      dom.legend.append(el);
    });
  }
  function addSummary(label, value, unit) {
    const box = document.createElement("div");
    box.className = "summary-item";
    const l = document.createElement("span");
    l.className = "summary-label";
    l.textContent = label;
    const v = document.createElement("strong");
    v.className = "summary-value";
    v.textContent = value == null ? "Unavailable" : String(value);
    if (unit) {
      const u = document.createElement("span");
      u.className = "summary-unit";
      u.textContent = " " + unit;
      v.append(u);
    }
    box.append(l, v);
    dom.summary.append(box);
  }
  function renderSummary(r) {
    dom.summary.replaceChildren();
    if (!r) return;
    const currentTime = replayTime(r, state.time);
    const area = state.layers.truth ? reachedAreaAt(r, currentTime) : null;
    const obs = observationAt(r, currentTime);
    const decision = latestDecisionAt(r, currentTime);
    $("#sampleContext").textContent = decision
      ? "Controller observation recorded at " +
        duration(decision.time_s) +
        ". Aircraft state is a recorded sample."
      : "No controller observation recorded at or before this time.";
    const pos = obs && obs.aircraft && obs.aircraft.position_m;
    const drops = Array.isArray(r.drops)
      ? r.drops.filter((d) => d.time_s <= currentTime).length
      : null;
    const water = obs && obs.aircraft && obs.aircraft.water_l;
    const phase = obs && obs.aircraft && obs.aircraft.phase;
    let detected = null;
    if (obs && obs.fire && Array.isArray(obs.fire.cells))
      detected = obs.fire.cells.reduce(
        (n, line, row) =>
          n +
          line.reduce(
            (m, _cell, col) =>
              m +
              (observationCellAt(obs, row, col, currentTime)?.fire_detected ===
              true
                ? 1
                : 0),
            0,
          ),
        0,
      );
    addSummary("Reached area", area == null ? null : area.toFixed(3), "ha");
    addSummary("Delivered fire cells", detected, "");
    addSummary("Recorded drops", drops, "");
    addSummary(
      "Aircraft position",
      Array.isArray(pos) && pos.length === 2 && pos.every(finite)
        ? pos.map((v) => Number(v).toFixed(0)).join(", ")
        : null,
      "m local",
    );
    addSummary("Aircraft water", finite(water) ? fmt(water) : null, "L");
    addSummary("Aircraft phase", phase || null, "");
  }
  function renderScenarioContext(r) {
    dom.scenarioContext.replaceChildren();
    if (!r) return;
    const g = gridGeometry(r), observation = observationAt(r, state.time);
    const wind = observation && observation.midflame_wind_to_m_s;
    addMeta(dom.scenarioContext, "Grid", g ? g.rows + " × " + g.cols + " cells" : null);
    addMeta(dom.scenarioContext, "Resolution", g && finite(g.cellSize) ? fmt(g.cellSize) + " m" : null);
    addMeta(dom.scenarioContext, "Wind E / N", Array.isArray(wind) && wind.length === 2 && wind.every(finite) ? wind.map(v => fmt(v)).join(" / ") + " m/s" : null);
    addMeta(dom.scenarioContext, "Scenario seed", r.seed);
    addMeta(dom.scenarioContext, "Coordinates", r.crs === "LOCAL_METERS" ? "Local metres" : r.crs);
  }
  function renderCellInspector() {
    dom.cell.replaceChildren();
    const r = currentReplay(),
      cell = state.selectedCell;
    if (!r || !cell) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "No cell selected.";
      dom.cell.append(p);
      return;
    }
    const currentTime = replayTime(r, state.time);
    const g = gridGeometry(r),
      obs = observationAt(r, currentTime),
      row = cell.row,
      col = cell.col;
    const rows = [];
    rows.push(["Grid cell", row + ", " + col, "cell-coord"]);
    const truth =
      state.layers.truth && Array.isArray(r.arrival_time_s)
        ? r.arrival_time_s[row]?.[col]
        : undefined;
    rows.push([
      "Fire arrival",
      truth === undefined
        ? null
        : truth === null
          ? "Not reached"
          : duration(truth),
      null,
    ]);
    const fuel =
      obs && obs.fuel_model && obs.fuel_model[row]
        ? obs.fuel_model[row][col]
        : null;
    rows.push(["Fuel model", fuel, null]);
    const report = state.layers.observations
      ? observationCellAt(obs, row, col, currentTime)
      : null;
    rows.push([
      "Delivered observation",
      report && typeof report.fire_detected === "boolean"
        ? report.fire_detected
          ? "Fire detected"
          : "No fire detected"
        : null,
      null,
    ]);
    const capture =
      report &&
      finite(report.observed_at_s) &&
      report.observed_at_s <= currentTime
        ? report.observed_at_s
        : null;
    rows.push([
      "Observed at",
      capture == null ? null : duration(capture),
      null,
    ]);
    rows.push([
      "Observation age",
      capture == null ? null : Math.max(0, currentTime - capture) + " s",
      null,
    ]);
    rows.push([
      "Source ID",
      report && capture != null ? report.source_id : null,
      null,
    ]);
    const candidates = latestDecisionAt(r, currentTime)?.candidate_cells;
    const cand = cellSet(candidates).has(row + "," + col);
    rows.push([
      "Current candidate",
      state.layers.candidates && Array.isArray(candidates)
        ? cand
          ? "Yes"
          : "No"
        : null,
      null,
    ]);
    rows.push([
      "Known treated",
      obs && Array.isArray(obs.known_treated_cells)
        ? cellSet(obs.known_treated_cells).has(row + "," + col)
          ? "Yes"
          : "No"
        : null,
      null,
    ]);
    const dl = document.createElement("dl");
    rows.forEach(([label, value, cls]) => {
      const div = document.createElement("div"),
        dt = document.createElement("dt"),
        dd = document.createElement("dd");
      if (cls) dd.className = cls;
      dt.textContent = label;
      dd.textContent = value == null ? "Unavailable" : String(value);
      div.append(dt, dd);
      dl.append(div);
    });
    dom.cell.append(dl);
  }
  function renderTimeline(r) {
    const horizon = maxTime();
    dom.timeRange.max = horizon;
    dom.timeRange.value = Math.min(state.time, horizon);
    dom.timeDisplay.textContent = duration(state.time);
    dom.timeContext.textContent =
      state.compare && state.runs.length > 1
        ? "of " + duration(horizon) + " · shorter runs complete"
        : "of " + duration(r && r.horizon_s);
    const decisions = (r && Array.isArray(r.decisions) ? r.decisions : [])
      .slice()
      .sort((a, b) => a.time_s - b.time_s);
    dom.ticks.replaceChildren();
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const s = document.createElement("span");
      s.textContent = duration(horizon * f);
      dom.ticks.append(s);
    });
    if (state.activeTab !== "records") return;
    dom.eventRows.replaceChildren();
    dom.eventCount.textContent =
      decisions.length + " event" + (decisions.length === 1 ? "" : "s");
    dom.emptyEvents.hidden = decisions.length > 0;
    decisions.forEach((d) => {
      const tr = document.createElement("tr");
      if (state.activeTab === "replay" && d === latestDecisionAt(r, state.time))
        tr.className = "current-event";
      const t = document.createElement("td");
      t.className = "event-time";
      t.textContent = duration(d.time_s);
      const action = document.createElement("td");
      const decisionText = d.order
        ? d.order.type || d.order.action || JSON.stringify(d.order)
        : d.action || "Unavailable";
      const target =
        d.order && (d.order.target_m || d.order.target_cell || d.order.cell);
      action.textContent = target
        ? decisionText + " · target " + JSON.stringify(target)
        : decisionText;
      const result = document.createElement("td");
      result.className =
        "event-result " +
        (d.accepted === true
          ? "accepted"
          : d.accepted === false
            ? "rejected"
            : "");
      result.textContent =
        d.accepted === true
          ? "Accepted"
          : d.accepted === false
            ? "Rejected"
            : "Recorded";
      const drop = (r.drops || []).find((x) => x.time_s === d.time_s);
      const candidateCount = Array.isArray(d.candidate_cells)
        ? d.candidate_cells.length
        : null;
      const candidates = document.createElement("td");
      candidates.textContent = candidateCount == null ? "Unavailable" : String(candidateCount);
      const effects = document.createElement("td");
      const blockedCount = Array.isArray(
        drop && (drop.newly_blocked_cells || drop.treated_cells),
      )
        ? (drop.newly_blocked_cells || drop.treated_cells).length
        : null;
      const effectParts = [];
      if (drop)
        effectParts.push(
          "drop recorded" +
            (blockedCount == null
              ? ""
              : ", " + blockedCount + " cells treated"),
        );
      effects.textContent = effectParts.length
        ? effectParts.join(" · ")
        : "Unavailable";
      tr.append(t, action, result, candidates, effects);
      tr.tabIndex = 0;
      tr.addEventListener("click", () => setTime(d.time_s));
      tr.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") {
          ev.preventDefault();
          setTime(d.time_s);
        }
      });
      dom.eventRows.append(tr);
    });
  }
  function fmt(v, digits) {
    return finite(v)
      ? v.toLocaleString(undefined, {
          maximumFractionDigits: digits == null ? 2 : digits,
        })
      : "Unavailable";
  }
  function metricValue(rowValue, unit) {
    if (!finite(rowValue)) return "Unavailable";
    return fmt(rowValue) + (unit ? " " + unit : "");
  }
  function renderOutcomes() {
    dom.outcomes.replaceChildren();
    const candidate = state.runs[state.candidateRun] || null;
    const reference = state.runs[state.referenceRun] || null;
    const result = comparisonRows(candidate, reference);
    dom.comparison.textContent =
      result.reason +
      " Δ = candidate − reference. Descriptive episode comparison.";
    result.rows.forEach((row) => {
      const tr = document.createElement("tr");
      const cells = [
        row.label,
        metricValue(row.reference, row.unit),
        metricValue(row.candidate, row.unit),
        result.compatible && row.delta != null
          ? (row.delta > 0 ? "+" : "") + metricValue(row.delta, row.unit)
          : "Unavailable",
      ];
      cells.forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      });
      dom.outcomes.append(tr);
    });
    renderAreaChart(candidate, reference);
  }
  function renderAreaChart(candidate, reference) {
    const svg = dom.chart;
    svg.replaceChildren();
    dom.chartLegend.replaceChildren();
    const series = [
      {
        run: candidate,
        label: candidate
          ? shortName(candidate, state.candidateRun)
          : "Candidate",
        color: "#247d70",
      },
      {
        run: reference,
        label: reference
          ? shortName(reference, state.referenceRun)
          : "Reference",
        color: "#b95b40",
      },
    ].map((item) => ({ ...item, data: areaSeries(item.run) }));
    const ns = "http://www.w3.org/2000/svg";
    const add = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs || {}).forEach(([key, value]) =>
        node.setAttribute(key, String(value)),
      );
      if (text != null) node.textContent = String(text);
      svg.append(node);
      return node;
    };
    svg.setAttribute("viewBox", "0 0 820 300");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute(
      "aria-label",
      "Retrospective reached-area curves derived from recorded arrival-time grids",
    );
    const valid = series.filter((item) => item.data.available);
    if (!valid.length) {
      add(
        "text",
        { x: 410, y: 145, class: "chart-empty", "text-anchor": "middle" },
        series
          .map((item) => item.data.reason)
          .filter(Boolean)
          .join(" ") || "Select replay records to plot available truth data.",
      );
    } else {
      const maxX = Math.max(...valid.map((item) => item.data.horizon_s));
      const maxY = Math.max(
        0,
        ...valid.flatMap((item) =>
          item.data.points.map((point) => point.area_ha),
        ),
      );
      const left = 60,
        right = 18,
        top = 16,
        bottom = 48,
        width = 820 - left - right,
        height = 300 - top - bottom;
      const x = (t) => left + (maxX > 0 ? t / maxX : 0) * width;
      const y = (v) => top + height - (maxY > 0 ? v / maxY : 0) * height;
      add("line", {
        x1: left,
        y1: top,
        x2: left,
        y2: top + height,
        class: "chart-axis",
      });
      add("line", {
        x1: left,
        y1: top + height,
        x2: left + width,
        y2: top + height,
        class: "chart-axis",
      });
      for (let i = 0; i <= 4; i++) {
        const fraction = i / 4;
        const tx = left + fraction * width;
        const ty = top + height - fraction * height;
        add("line", {
          x1: tx,
          y1: top,
          x2: tx,
          y2: top + height,
          class: "chart-gridline",
        });
        add("line", {
          x1: left,
          y1: ty,
          x2: left + width,
          y2: ty,
          class: "chart-gridline",
        });
        add(
          "text",
          {
            x: tx,
            y: top + height + 19,
            class: "chart-tick",
            "text-anchor": "middle",
          },
          fmt(maxX * fraction, 0),
        );
        add(
          "text",
          { x: left - 8, y: ty + 4, class: "chart-tick", "text-anchor": "end" },
          fmt(maxY * fraction, 2),
        );
      }
      add(
        "text",
        {
          x: left + width / 2,
          y: 291,
          class: "chart-axis-label",
          "text-anchor": "middle",
        },
        "Simulation time (s)",
      );
      add(
        "text",
        {
          x: 15,
          y: top + height / 2,
          class: "chart-axis-label",
          transform: `rotate(-90 15 ${top + height / 2})`,
          "text-anchor": "middle",
        },
        "Reached area (ha)",
      );
      series.forEach((item) => {
        if (!item.data.available) return;
        const points = item.data.points;
        if (!points.length) return;
        let d = `M ${x(points[0].time_s)} ${y(points[0].area_ha)}`;
        for (let i = 1; i < points.length; i++)
          d += ` H ${x(points[i].time_s)} V ${y(points[i].area_ha)}`;
        add("path", {
          d,
          class: "area-curve",
          stroke: item.color,
          fill: "none",
        });
      });
      const cursor = Math.min(state.time, maxX);
      add("line", {
        x1: x(cursor),
        y1: top,
        x2: x(cursor),
        y2: top + height,
        class: "chart-cursor",
      });
    }
    series.forEach((item) => {
      const label = document.createElement("span");
      label.className = "chart-legend-item";
      const swatch = document.createElement("i");
      swatch.style.borderColor = item.color;
      label.append(
        swatch,
        document.createTextNode(
          item.label +
            (item.data.available ? "" : " · unavailable: " + item.data.reason),
        ),
      );
      dom.chartLegend.append(label);
    });
  }
  function renderEverything() {
    const r = currentReplay();
    const recordRun =
      state.activeTab === "records" ? state.runs[state.selected] : r;
    if (
      state.renderedRunCount !== state.runs.length ||
      state.renderedSelected !== state.selected
    ) {
      renderRuns();
      state.renderedRunCount = state.runs.length;
      state.renderedSelected = state.selected;
    }
    if (state.activeTab === "records") renderMetadata(recordRun);
    if (state.activeTab === "models") renderModels();
    renderMap();
    renderTimeline(state.activeTab === "records" ? recordRun : r);
    renderSummary(r);
    renderScenarioContext(r);
    renderCellInspector();
    if (state.activeTab === "compare") renderOutcomes();
    const currentIndex = state.compare ? state.candidateRun : state.selected;
    dom.mapTitle.textContent = r
      ? scenarioLabel(r, currentIndex)
      : "Select a run";
    dom.runKicker.textContent = r
      ? shortName(r, currentIndex)
      : "Replay map";
    const g = r && gridGeometry(r);
    dom.mapSubtitle.textContent = r
      ? [
          shortName(r, currentIndex),
          g
            ? g.rows + " × " + g.cols + " cells"
            : "grid dimensions unavailable",
          r.crs || "CRS unavailable",
        ].join(" · ")
      : "The map shows recorded grid data only.";
    dom.play.textContent = state.playing ? "Pause" : "Play";
    dom.play.setAttribute("aria-label", state.playing ? "Pause" : "Play");
  }
  function selectRun(index) {
    if (index < 0 || index >= state.runs.length) return;
    state.selected = index;
    state.time = Math.min(state.time, maxTime());
    state.selectedCell = null;
    renderEverything();
  }
  function activateTab(name) {
    state.activeTab = name;
    state.compare = name === "compare";
    dom.tabButtons.forEach((button) => {
      const active = button.dataset.tab === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    dom.tabPages.forEach((page) => {
      const active = page.dataset.page === name;
      page.hidden = !active;
      page.classList.toggle("active", active);
    });
    if (state.compare)
      dom.mapHostCompare.append(dom.mapBlock, dom.timelinePanel);
    else {
      dom.mapHostReplay.append(dom.mapBlock);
      $("#replayLayerHost").append(dom.layerDisclosure);
      $(".replay-stage").append(dom.timelinePanel);
    }
    if (state.compare) dom.compareLayerHost.append(dom.layerDisclosure);
    dom.layerDisclosure.open = !state.compare;
    state.time = Math.min(state.time, maxTime());
    renderEverything();
    requestAnimationFrame(() => renderMap());
  }
  function setTime(value) {
    const h = maxTime();
    state.time = Math.max(0, Math.min(h, Number(value) || 0));
    renderEverything();
  }
  function decisionsSorted() {
    const replaySet = state.compare
      ? [state.runs[state.candidateRun], state.runs[state.referenceRun]]
      : [currentReplay()];
    return replaySet
      .flatMap((r) => (r && Array.isArray(r.decisions) ? r.decisions : []))
      .map((d) => d.time_s)
      .filter(finite)
      .sort((a, b) => a - b);
  }
  function jumpDecision(direction) {
    const ts = decisionsSorted();
    if (!ts.length) return;
    let idx = direction > 0 ? ts.findIndex((t) => t > state.time + 1e-9) : -1;
    if (direction < 0) {
      for (let i = ts.length - 1; i >= 0; i--)
        if (ts[i] < state.time - 1e-9) {
          idx = i;
          break;
        }
    }
    if (idx < 0) idx = direction > 0 ? ts.length - 1 : 0;
    setTime(ts[idx]);
  }
  function stopPlayback() {
    state.playing = false;
    clearInterval(state.timer);
    state.timer = null;
    renderEverything();
  }
  function startPlayback() {
    if (!currentReplay() || !finite(currentReplay().horizon_s)) return;
    if (state.time >= maxTime()) state.time = 0;
    state.playing = true;
    renderEverything();
    state.timer = setInterval(() => {
      const step = 0.1 * (Number($("#speedSelect").value) || 1);
      const next = state.time + step;
      if (next >= maxTime()) {
        state.time = maxTime();
        stopPlayback();
      } else setTime(next);
    }, 100);
  }
  function addReplay(r) {
    validateReplay(r);
    state.runs.push(r);
  }
  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  function safeFileName(r) {
    return (
      String(shortName(r, state.selected))
        .replace(/[^a-z0-9_-]+/gi, "-")
        .replace(/^-+|-+$/g, "")
        .toLowerCase() || "nightjar-replay"
    );
  }
  function exportMap() {
    const r = currentReplay(),
      g = r && gridGeometry(r);
    if (!r || !g) {
      showNotice(
        "A replay with grid dimensions is required to export a map.",
        true,
      );
      return;
    }
    const factor = state.compare ? 2 : 1;
    const w = factor > 1 ? 600 : 1200,
      h = factor > 1 ? 650 : 900;
    const columns = Math.min(factor, 3);
    const out = document.createElement("canvas");
    out.width = w * columns;
    out.height = h * Math.ceil(factor / columns);
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#f5f4ef";
    ctx.fillRect(0, 0, out.width, out.height);
    const replays = state.compare
      ? [
          [state.runs[state.candidateRun], state.candidateRun],
          [state.runs[state.referenceRun], state.referenceRun],
        ].filter(([item]) => item)
      : [[r, state.selected]];
    replays.forEach(([item, index], i) => {
      const gg = gridGeometry(item);
      if (!gg) return;
      const panelW = w;
      const ratio = gg.cols / gg.rows;
      let cw = panelW - 80,
        ch = h - 90;
      if (cw / ch > ratio) cw = ch * ratio;
      else ch = cw / ratio;
      const panelX = (i % columns) * w;
      const panelY = Math.floor(i / columns) * h;
      const x = panelX + (w - cw) / 2,
        y = panelY + (h - ch) / 2;
      ctx.fillStyle = "#40534a";
      ctx.font = "16px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(shortName(item, index), panelX + w / 2, panelY + 23, w - 30);
      ctx.font = "12px system-ui";
      ctx.fillText(
        "Episode " +
          duration(replayTime(item, state.time)) +
          " · " +
          escText(item.crs) +
          " · truth " +
          (state.layers.truth && item.arrival_time_s
            ? "on"
            : "off/unavailable"),
        panelX + w / 2,
        panelY + h - 20,
        w - 30,
      );
      const dprctx = ctx;
      dprctx.save();
      dprctx.translate(x, y);
      drawGrid(dprctx, item, gg, cw, ch, index);
      dprctx.restore();
    });
    out.toBlob((blob) => {
      if (blob) {
        download(
          blob,
          safeFileName(r) + (factor > 1 ? "-comparison" : "") + ".png",
        );
        showNotice("Map PNG download requested.");
      } else showNotice("Map PNG export failed.", true);
    }, "image/png");
  }
  function bind() {
    dom.runList.addEventListener("change", () => selectRun(Number(dom.runList.value)));
    dom.scenarioSelect.addEventListener("change", () => {
      const index = state.runs.findIndex((r, i) => scenarioKey(r, i) === dom.scenarioSelect.value);
      if (index >= 0) selectRun(index);
    });
    $("#importButton").addEventListener("click", () => dom.file.click());
    dom.file.addEventListener("change", async (ev) => {
      const files = Array.from(ev.target.files || []);
      let addedRuns = 0,
        addedModels = 0;
      const errors = [];
      for (const f of files) {
        try {
          const parsed = JSON.parse(await f.text());
          const batch = Array.isArray(parsed) ? parsed : [parsed];
          const models = batch.map(
            (record) =>
              record && record.schema_version === "nightjar-viewer-model-0.1",
          );
          batch.forEach((record, i) =>
            models[i] ? validateModelRecord(record) : validateReplay(record),
          );
          batch.forEach((record, i) => {
            if (models[i]) {
              state.models.push(record);
              addedModels += 1;
            } else {
              state.runs.push(record);
              addedRuns += 1;
            }
          });
        } catch (e) {
          errors.push(
            f.name +
              ": " +
              (e && e.message ? e.message : "invalid replay JSON"),
          );
        }
      }
      if (addedRuns || addedModels) {
        clearInterval(state.timer);
        state.timer = null;
        state.playing = false;
        if (addedRuns) {
          state.selected = state.runs.length - addedRuns;
          state.candidateRun = state.selected;
          if (
            state.runs.length > 1 &&
            state.referenceRun === state.candidateRun
          )
            state.referenceRun = (state.candidateRun + 1) % state.runs.length;
          state.time = 0;
          state.selectedCell = null;
        }
        renderEverything();
      }
      if (errors.length)
        showNotice(
          (addedRuns || addedModels
            ? addedRuns +
              " replay and " +
              addedModels +
              " model record(s) imported. "
            : "") + errors.join(" | "),
          true,
        );
      else if (addedRuns || addedModels)
        showNotice(
          addedRuns +
            " replay and " +
            addedModels +
            " model record(s) imported.",
        );
      ev.target.value = "";
    });
    $$(".layer-row input").forEach((el) =>
      el.addEventListener("change", () => {
        state.layers[el.dataset.layer] = el.checked;
        renderEverything();
      }),
    );
    dom.tabButtons.forEach((button) =>
      button.addEventListener("click", () => activateTab(button.dataset.tab)),
    );
    dom.tabButtons.forEach((button, i) =>
      button.addEventListener("keydown", (ev) => {
        const next =
          ev.key === "ArrowRight"
            ? (i + 1) % dom.tabButtons.length
            : ev.key === "ArrowLeft"
              ? (i + dom.tabButtons.length - 1) % dom.tabButtons.length
              : ev.key === "Home"
                ? 0
                : ev.key === "End"
                  ? dom.tabButtons.length - 1
                  : -1;
        if (next >= 0) {
          ev.preventDefault();
          dom.tabButtons[next].focus();
          activateTab(dom.tabButtons[next].dataset.tab);
        }
      }),
    );
    $("#compareButton").addEventListener("click", () => activateTab("compare"));
    dom.candidateRun.addEventListener("change", () => {
      state.candidateRun = Number(dom.candidateRun.value);
      state.selected = state.candidateRun;
      state.time = Math.min(state.time, maxTime());
      renderEverything();
    });
    dom.referenceRun.addEventListener("change", () => {
      state.referenceRun = Number(dom.referenceRun.value);
      state.time = Math.min(state.time, maxTime());
      renderEverything();
    });
    dom.recordRunSelect.addEventListener("change", () =>
      selectRun(Number(dom.recordRunSelect.value)),
    );
    dom.controllerSelect.addEventListener("change", () => {
      state.selectedController = Number(dom.controllerSelect.value);
      renderModels();
    });
    dom.physicalSelect.addEventListener("change", () => {
      state.selectedPhysical = Number(dom.physicalSelect.value);
      renderModels();
    });
    $("#exportControllerButton").addEventListener("click", () =>
      exportModelRecord("controller"),
    );
    $("#exportPhysicalButton").addEventListener("click", () =>
      exportModelRecord("physical"),
    );
    $("#exportJsonButton").addEventListener("click", () => {
      const r = state.runs[state.selected];
      if (r)
        download(
          new Blob([JSON.stringify(r, null, 2) + "\n"], {
            type: "application/json",
          }),
          safeFileName(r) + ".json",
        );
    });
    $("#exportPngButton").addEventListener("click", exportMap);
    $("#exportComparePngButton").addEventListener("click", exportMap);
    dom.timeRange.addEventListener("input", (ev) => {
      state.time = Number(ev.target.value);
      renderEverything();
    });
    dom.play.addEventListener("click", () =>
      state.playing ? stopPlayback() : startPlayback(),
    );
    $("#previousDecision").addEventListener("click", () => jumpDecision(-1));
    $("#nextDecision").addEventListener("click", () => jumpDecision(1));
    $("#resetButton").addEventListener("click", () => {
      stopPlayback();
      setTime(0);
    });
    $("#speedSelect").addEventListener("change", () => {
      if (state.playing) {
        stopPlayback();
        startPlayback();
      }
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.defaultPrevented) return;
      const target = ev.target;
      const editable =
        target &&
        (target.isContentEditable ||
          /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(target.tagName));
      if (editable) return;
      if (ev.code === "Space") {
        ev.preventDefault();
        state.playing ? stopPlayback() : startPlayback();
      } else if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        jumpDecision(-1);
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        jumpDecision(1);
      }
    });
    root.addEventListener("resize", () => renderMap());
  }
  try {
    (Array.isArray(INITIAL_REPLAYS) ? INITIAL_REPLAYS : []).forEach(addReplay);
    (Array.isArray(INITIAL_MODELS) ? INITIAL_MODELS : []).forEach(addModel);
    if (state.runs.length > 1 && state.referenceRun === state.candidateRun)
      state.referenceRun = 1;
  } catch (e) {
    showNotice(
      "An embedded workspace record could not be loaded: " + e.message,
      true,
    );
  }
  bind();
  activateTab("replay");
})(typeof globalThis !== "undefined" ? globalThis : this);
