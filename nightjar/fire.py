"""Deterministic synthetic surface-fire arrival times on a local metre grid.

The directional rate formula here is a test model, not Rothermel or a
validated wildfire model. See docs/PHYSICAL_MODEL_V0.md.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from heapq import heappop, heappush
import json
from math import exp, hypot, isfinite
from pathlib import Path


NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _nonnegative(value: object, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _grid(value: object, name: str, rows: int, cols: int, *, signed: bool = False) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows or any(not isinstance(row, list) or len(row) != cols for row in value):
        raise ValueError(f"{name} must have shape [{rows}, {cols}]")
    if signed:
        grid = [[float(v) for v in row] for row in value]
        if not all(isfinite(v) for row in grid for v in row):
            raise ValueError(f"{name} must contain finite values")
        return grid
    return [[_nonnegative(v, name) for v in row] for row in value]


def validate_fire_scenario(scenario: dict) -> dict:
    if scenario.get("schema_version") != "fire-synthetic-0.1":
        raise ValueError("schema_version must be 'fire-synthetic-0.1'")
    if scenario.get("provenance") != "synthetic" or scenario.get("crs") != "LOCAL_METERS":
        raise ValueError("v0.1 requires synthetic provenance and LOCAL_METERS CRS")
    if not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("scenario_id must be a nonempty string")
    if not isinstance(scenario.get("seed"), int) or isinstance(scenario["seed"], bool):
        raise ValueError("seed must be an integer")
    rows = scenario.get("rows")
    cols = scenario.get("cols")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1 or not isinstance(cols, int) or isinstance(cols, bool) or cols < 1:
        raise ValueError("rows and cols must be positive integers")
    cell_size = _nonnegative(scenario["cell_size_m"], "cell_size_m", positive=True)
    horizon = _nonnegative(scenario["horizon_s"], "horizon_s", positive=True)
    ros = _grid(scenario["base_ros_m_s"], "base_ros_m_s", rows, cols)
    elevation = _grid(scenario["elevation_m"], "elevation_m", rows, cols, signed=True)
    wind = scenario.get("wind_to_m_s")
    if not isinstance(wind, list) or len(wind) != 2:
        raise ValueError("wind_to_m_s must be [east_m_s, north_m_s]")
    wind = [float(v) for v in wind]
    if not all(isfinite(v) for v in wind):
        raise ValueError("wind_to_m_s must be finite")
    wind_coefficient = _nonnegative(scenario["wind_coefficient_s_m"], "wind_coefficient_s_m")
    slope_coefficient = _nonnegative(scenario["slope_coefficient"], "slope_coefficient")
    ignitions = scenario.get("ignitions")
    if not isinstance(ignitions, list) or not ignitions:
        raise ValueError("ignitions must be a nonempty list")
    normalized_ignitions = []
    for item in ignitions:
        if not isinstance(item, list) or len(item) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in item):
            raise ValueError("each ignition must be [row, col]")
        r, c = item
        if not 0 <= r < rows or not 0 <= c < cols or ros[r][c] == 0:
            raise ValueError("ignition must be within burnable cells")
        normalized_ignitions.append([r, c])
    return {
        "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
        "rows": rows, "cols": cols, "cell_size_m": cell_size,
        "horizon_s": horizon, "base_ros_m_s": ros, "elevation_m": elevation,
        "wind_to_m_s": wind, "wind_coefficient_s_m": wind_coefficient,
        "slope_coefficient": slope_coefficient, "ignitions": normalized_ignitions,
    }


def run_fire(scenario: dict) -> dict:
    """Compute first arrival at each cell centre using directed 8-neighbour edges.

    For an edge, rate at each endpoint is base_ros * exp(k_w * wind projection
    + k_s * signed rise/run). Travel time is half an edge at each endpoint rate.
    An unburnable cell (base_ros=0) blocks traversal.
    """
    cfg = validate_fire_scenario(scenario)
    rows, cols = cfg["rows"], cfg["cols"]
    arrival = [[float("inf") for _ in range(cols)] for _ in range(rows)]
    queue: list[tuple[float, int, int]] = []
    for r, c in cfg["ignitions"]:
        arrival[r][c] = 0.0
        heappush(queue, (0.0, r, c))
    while queue:
        current_time, r, c = heappop(queue)
        if current_time != arrival[r][c]:
            continue
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            source_rate = cfg["base_ros_m_s"][r][c]
            target_rate = cfg["base_ros_m_s"][nr][nc]
            if source_rate == 0 or target_rate == 0:
                continue
            if dr and dc and (cfg["base_ros_m_s"][r + dr][c] == 0 or cfg["base_ros_m_s"][r][c + dc] == 0):
                continue
            distance = cfg["cell_size_m"] * hypot(dc, dr)
            east, north = dc / hypot(dc, dr), dr / hypot(dc, dr)
            wind_projection = cfg["wind_to_m_s"][0] * east + cfg["wind_to_m_s"][1] * north
            rise_over_run = (cfg["elevation_m"][nr][nc] - cfg["elevation_m"][r][c]) / distance
            exponent = cfg["wind_coefficient_s_m"] * wind_projection + cfg["slope_coefficient"] * rise_over_run
            if abs(exponent) > 20:
                raise ValueError("directional multiplier exceeds the synthetic model's tested range")
            multiplier = exp(exponent)
            edge_time = distance / (2 * source_rate * multiplier) + distance / (2 * target_rate * multiplier)
            candidate = current_time + edge_time
            if candidate < arrival[nr][nc]:
                arrival[nr][nc] = candidate
                heappush(queue, (candidate, nr, nc))
    horizon = cfg["horizon_s"]
    reached = sum(t <= horizon for row in arrival for t in row)
    return {
        "replay_schema_version": "fire-synthetic-0.1",
        "scenario_id": cfg["scenario_id"], "seed": cfg["seed"],
        "scenario_sha256": sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "crs": "LOCAL_METERS", "provenance": "synthetic",
        "units": {"arrival": "s", "affected_area": "ha"},
        "arrival_time_s": [[round(t, 6) if isfinite(t) else None for t in row] for row in arrival],
        "affected_area_ha": reached * cfg["cell_size_m"] ** 2 / 10_000,
        "horizon_s": horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_fire(json.loads(args.scenario.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
