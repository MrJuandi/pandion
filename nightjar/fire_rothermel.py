"""FM1 Rothermel surface spread coupled to a directed grid-arrival solver."""

from __future__ import annotations

import argparse
from hashlib import sha256
from heapq import heappop, heappush
import json
from math import hypot, isfinite
from pathlib import Path

from .fire import NEIGHBORS
from .rothermel import FM1_MOISTURE_EXTINCTION, fm1_directional_rate


def validate_scenario(scenario: dict) -> dict:
    if scenario.get("schema_version") != "rothermel-fm1-0.1":
        raise ValueError("schema_version must be 'rothermel-fm1-0.1'")
    if scenario.get("provenance") != "synthetic" or scenario.get("crs") != "LOCAL_METERS":
        raise ValueError("this scenario must declare synthetic provenance and LOCAL_METERS CRS")
    if not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("scenario_id must be a nonempty string")
    if not isinstance(scenario.get("seed"), int) or isinstance(scenario["seed"], bool):
        raise ValueError("seed must be an integer")
    rows, cols = scenario.get("rows"), scenario.get("cols")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (rows, cols)):
        raise ValueError("rows and cols must be positive integers")
    cell_size = float(scenario["cell_size_m"])
    horizon = float(scenario["horizon_s"])
    if not all(isfinite(v) and v > 0 for v in (cell_size, horizon)):
        raise ValueError("cell_size_m and horizon_s must be positive finite values")
    fuel = scenario["fuel_model"]
    moisture = scenario["dead_fuel_moisture_fraction"]
    elevation = scenario["elevation_m"]
    for name, grid in (("fuel_model", fuel), ("dead_fuel_moisture_fraction", moisture), ("elevation_m", elevation)):
        if not isinstance(grid, list) or len(grid) != rows or any(not isinstance(row, list) or len(row) != cols for row in grid):
            raise ValueError(f"{name} must have shape [{rows}, {cols}]")
    if any(v not in ("FM1", "NONE") for row in fuel for v in row):
        raise ValueError("fuel_model entries must be FM1 or NONE")
    moisture = [[float(v) for v in row] for row in moisture]
    elevation = [[float(v) for v in row] for row in elevation]
    if any(not isfinite(v) or not 0 <= v <= 1 for row in moisture for v in row):
        raise ValueError("dead-fuel moisture must be a finite fraction in [0, 1]")
    if any(not isfinite(v) for row in elevation for v in row):
        raise ValueError("elevation must be finite")
    wind = scenario["midflame_wind_to_m_s"]
    if not isinstance(wind, list) or len(wind) != 2:
        raise ValueError("midflame_wind_to_m_s must be [east, north]")
    wind = tuple(float(v) for v in wind)
    if not all(isfinite(v) for v in wind):
        raise ValueError("midflame wind must be finite")
    ignitions = scenario["ignitions"]
    if not isinstance(ignitions, list) or not ignitions:
        raise ValueError("ignitions must be a nonempty list")
    for item in ignitions:
        if not isinstance(item, list) or len(item) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in item):
            raise ValueError("each ignition must be [row, col]")
        r, c = item
        if not (0 <= r < rows and 0 <= c < cols) or fuel[r][c] != "FM1" or moisture[r][c] >= FM1_MOISTURE_EXTINCTION:
            raise ValueError("ignitions must be within burnable, non-extinct FM1 cells")
    return {
        "rows": rows, "cols": cols, "cell_size_m": cell_size, "horizon_s": horizon,
        "fuel_model": fuel, "moisture": moisture, "elevation": elevation,
        "wind": wind, "ignitions": ignitions,
    }


def run_fire_rothermel(scenario: dict) -> dict:
    cfg = validate_scenario(scenario)
    rows, cols, cell = cfg["rows"], cfg["cols"], cfg["cell_size_m"]
    fuel, moisture, elevation = cfg["fuel_model"], cfg["moisture"], cfg["elevation"]

    def burnable(r: int, c: int) -> bool:
        return fuel[r][c] == "FM1" and moisture[r][c] < FM1_MOISTURE_EXTINCTION

    def gradient(r: int, c: int) -> tuple[float, float]:
        west, east = max(c - 1, 0), min(c + 1, cols - 1)
        south, north = max(r - 1, 0), min(r + 1, rows - 1)
        gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * cell) if east > west else 0.0
        gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * cell) if north > south else 0.0
        return gx, gy

    gradients = [[gradient(r, c) for c in range(cols)] for r in range(rows)]
    arrival = [[float("inf") for _ in range(cols)] for _ in range(rows)]
    queue: list[tuple[float, int, int]] = []
    for r, c in cfg["ignitions"]:
        arrival[r][c] = 0.0
        heappush(queue, (0.0, r, c))
    while queue:
        time, r, c = heappop(queue)
        if time != arrival[r][c]:
            continue
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not burnable(nr, nc):
                continue
            if dr and dc and (not burnable(r + dr, c) or not burnable(r, c + dc)):
                continue
            length = hypot(dc, dr) * cell
            direction = (dc, dr)
            source_rate = fm1_directional_rate(moisture[r][c], cfg["wind"], gradients[r][c], direction)
            target_rate = fm1_directional_rate(moisture[nr][nc], cfg["wind"], gradients[nr][nc], direction)
            candidate = time + length / (2 * source_rate) + length / (2 * target_rate)
            if candidate < arrival[nr][nc]:
                arrival[nr][nc] = candidate
                heappush(queue, (candidate, nr, nc))
    reached = sum(t <= cfg["horizon_s"] for row in arrival for t in row)
    return {
        "replay_schema_version": "rothermel-fm1-0.1",
        "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
        "scenario_sha256": sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "provenance": "synthetic", "crs": "LOCAL_METERS",
        "units": {"arrival": "s", "affected_area": "ha"},
        "horizon_s": cfg["horizon_s"],
        "arrival_time_s": [[round(v, 6) if isfinite(v) else None for v in row] for row in arrival],
        "affected_area_ha": reached * cell * cell / 10_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_fire_rothermel(json.loads(args.scenario.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
