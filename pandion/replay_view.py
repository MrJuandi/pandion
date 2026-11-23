"""Presentation adapters. Never feed these audit layers into a controller."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


def _raster(value: Any, rows: int, cols: int, name: str, default: Any) -> list:
    if value is None:
        return [[default for _ in range(cols)] for _ in range(rows)]
    if not isinstance(value, list) or len(value) != rows or any(not isinstance(line, list) or len(line) != cols for line in value):
        raise ValueError(f"{name} must match the {rows}x{cols} grid")
    return deepcopy(value)


def normalize_replay(replay: dict) -> dict:
    """Adapt synthetic, geographic, and static map artifacts without inventing outcomes."""
    out = deepcopy(replay)
    observations = [d.get("observation", {}) for d in out.get("decisions", [])]
    obs = observations[0] if observations else {}
    arrival = out.get("arrival_time_s")
    if arrival is None:
        raise ValueError("replay has no arrival_time_s; regenerate it with a current simulator")
    rows, cols = len(arrival), len(arrival[0]) if arrival else 0
    if not rows or not cols:
        raise ValueError("replay grid must be nonempty")
    grid = out.get("grid", {})
    if grid.get("rows", rows) != rows or grid.get("cols", cols) != cols:
        raise ValueError("replay dimensions disagree with arrival_time_s")
    size = grid.get("cell_size_m", out.get("cell_size_m", obs.get("cell_size_m")))
    # Old replay files without a recorded resolution are ambiguous, not 50 m maps.
    if not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
        raise ValueError("replay must record a finite positive cell_size_m")
    out["grid"] = {"rows": rows, "cols": cols, "cell_size_m": size}
    out["arrival_time_s"] = _raster(arrival, rows, cols, "arrival_time_s", None)
    for row in arrival:
        if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0) for v in row):
            raise ValueError("arrival_time_s must contain nonnegative finite seconds or null")
    horizon = out.get("horizon_s", 0)
    if not isinstance(horizon, (int, float)) or not math.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon_s must be nonnegative finite seconds")
    out["horizon_s"] = horizon
    out["fuel_grid"] = _raster(out.get("fuel_grid", obs.get("fuel_model")), rows, cols, "fuel_grid", None)
    out["elevation_m"] = _raster(out.get("display_elevation_m", out.get("elevation_m", obs.get("elevation_m"))), rows, cols, "elevation_m", None)
    for row in out["elevation_m"]:
        if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in row):
            raise ValueError("elevation_m must contain finite metres or null")
    out["water_mask"] = _raster(out.get("water_mask"), rows, cols, "water_mask", False)
    if any(not isinstance(v, bool) for row in out["water_mask"] for v in row):
        raise ValueError("water_mask must contain booleans")
    crs = out.get("crs", "UNKNOWN")
    out["crs"] = crs.get("code", "UNKNOWN") if isinstance(crs, dict) else crs
    out["origin_m"] = list(out.get("origin_m", out.get("origin_utm", [0, 0])))
    if len(out["origin_m"]) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in out["origin_m"]):
        raise ValueError("origin_m must contain two finite metre coordinates")
    # An explicit malformed identity must never silently merge two trajectories.
    for field in ("aircraft_track", "decisions", "drops", "aircraft_initial_state"):
        for record in out.get(field, []):
            key = "index" if field == "aircraft_initial_state" else "aircraft_index"
            index = record.get(key, 0)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(f"{field}: {key} must be a nonnegative integer")
    for decision in out.get("decisions", []):
        decision["completed_at_s"] = decision["time_s"]
        decision["time_s"] = decision.get("decision_time_s", decision["time_s"])
        observation = decision.get("observation", {})
        if "fire_reports" not in observation:
            cells = observation.get("fire", {}).get("cells", [])
            observation["fire_reports"] = [[r, c] for r, row in enumerate(cells) for c, cell in enumerate(row)
                                           if isinstance(cell, dict) and cell.get("fire_detected") is True]
        # Static rasters already live at the replay level. Keep delivered detections,
        # not dozens of copies of the same landscape, in the browser payload.
        for key in ("fire", "fuel_model", "elevation_m"):
            observation.pop(key, None)
    for drop in out.get("drops", []):
        drop["position_m"] = drop.get("position_m", drop.get("target_local_m"))
        drop["newly_blocked_cells"] = drop.get("newly_blocked_cells", drop.get("treated_cells", []))
    provenance = out.get("provenance", {})
    provenance = provenance if isinstance(provenance, dict) else {"source": provenance}
    meta = {
        "name": out.get("name", out.get("scenario_id", "Pandion")),
        "source_kind": out.get("source_kind", provenance.get("source_kind", "unspecified")),
        "source": provenance.get("source", "Source not recorded"),
        "split": out.get("split", "unspecified"), "seed": out.get("seed"),
        "scenario_hash": out.get("scenario_hash", out.get("scenario_sha256")),
        "acquired_at_utc": out.get("time_origin_utc", provenance.get("image_acquired_at_utc")),
        "assumptions": provenance.get("assumptions", {}),
        "terrain": {"label": "Simulator terrain" if any(v is not None for row in out["elevation_m"] for v in row) else "No elevations recorded",
                    "source": "", "display_only": False},
    }
    meta.update(out.get("metadata", {}))
    out["metadata"] = meta
    out.setdefault("scenario_id", "Pandion")
    out.setdefault("policy", "No policy recorded")
    out.setdefault("affected_area_ha", None)
    out.setdefault("aircraft_metrics", {})
    out.setdefault("decisions", [])
    out.setdefault("drops", [])
    out.pop("pending_observation", None)
    out.pop("display_elevation_m", None)
    return out
