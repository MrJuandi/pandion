"""Synthetic grid-arrival fire using standard fuel models via Pyretechnics."""

from __future__ import annotations

import argparse
from hashlib import sha256
from heapq import heappop, heappush
from importlib.metadata import version
import json
from math import atan2, degrees, hypot, isfinite
from pathlib import Path

from .fire import NEIGHBORS


def run_fire_standard(scenario: dict) -> dict:
    try:
        from pyretechnics import fuel_models, surface_fire
    except ImportError as exc:
        raise RuntimeError("Install requirements-standard.txt to run standard-fuel scenarios") from exc

    if scenario.get("schema_version") != "standard-fuel-0.1":
        raise ValueError("schema_version must be 'standard-fuel-0.1'")
    if scenario.get("provenance") != "synthetic" or scenario.get("crs") != "LOCAL_METERS":
        raise ValueError("standard-fuel-0.1 requires synthetic provenance and LOCAL_METERS")
    if not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("scenario_id must be a nonempty string")
    seed = scenario.get("seed")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rows, cols = scenario["rows"], scenario["cols"]
    if type(rows) is not int or type(cols) is not int or min(rows, cols) < 1:
        raise ValueError("rows and cols must be positive integers")
    cell, horizon = float(scenario["cell_size_m"]), float(scenario["horizon_s"])
    if not all(isfinite(x) and x > 0 for x in (cell, horizon)):
        raise ValueError("cell size and horizon must be positive and finite")
    fuel, elevation = scenario["fuel_model"], scenario["elevation_m"]
    for name, grid in (("fuel_model", fuel), ("elevation_m", elevation)):
        if not isinstance(grid, list) or len(grid) != rows or any(not isinstance(row, list) or len(row) != cols for row in grid):
            raise ValueError(f"{name} must have shape [{rows}, {cols}]")
    supported = set(fuel_models.list_fuel_model_numbers())
    if any(type(code) is not int or (code != 0 and code not in supported) for row in fuel for code in row):
        raise ValueError("fuel codes must be 0 (nonburnable) or a supported standard model number")
    elevation = [[float(v) for v in row] for row in elevation]
    if any(not isfinite(v) for row in elevation for v in row):
        raise ValueError("elevation must be finite")
    moisture = scenario["fuel_moisture_fraction"]
    fields = ("dead_1h", "dead_10h", "dead_100h", "dead_herbaceous", "live_herbaceous", "live_woody")
    if not isinstance(moisture, dict) or set(moisture) != set(fields):
        raise ValueError("fuel_moisture_fraction must contain six fuel classes")
    moisture_values = tuple(float(moisture[name]) for name in fields)
    if any(not isfinite(v) or v < 0 or v > 3 for v in moisture_values):
        raise ValueError("fuel moisture must be finite fractions in [0, 3]")
    wind = scenario["midflame_wind_to_m_s"]
    if not isinstance(wind, list) or len(wind) != 2:
        raise ValueError("midflame_wind_to_m_s must be [east, north]")
    wind = tuple(float(v) for v in wind)
    if any(not isfinite(v) for v in wind):
        raise ValueError("wind must be finite")
    ignitions = scenario["ignitions"]
    if not isinstance(ignitions, list) or not ignitions:
        raise ValueError("ignitions must be a nonempty list")
    for item in ignitions:
        if not isinstance(item, list) or len(item) != 2 or any(type(v) is not int for v in item):
            raise ValueError("ignitions must be [row, col] pairs")
        r, c = item
        if not (0 <= r < rows and 0 <= c < cols) or fuel[r][c] == 0:
            raise ValueError("ignitions must be inside a burnable fuel cell")
    if len({tuple(item) for item in ignitions}) != len(ignitions):
        raise ValueError("ignitions must be unique")

    gradients = []
    for r in range(rows):
        row = []
        for c in range(cols):
            west, east = max(c - 1, 0), min(c + 1, cols - 1)
            south, north = max(r - 1, 0), min(r + 1, rows - 1)
            gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * cell) if east > west else 0.0
            gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * cell) if north > south else 0.0
            row.append((gx, gy))
        gradients.append(row)

    upwind = degrees(atan2(-wind[0], -wind[1])) % 360 if wind != (0.0, 0.0) else 0.0
    behavior = []
    for r in range(rows):
        row = []
        for c in range(cols):
            code = fuel[r][c]
            if code == 0:
                row.append(None)
                continue
            model = fuel_models.moisturize(fuel_models.get_fuel_model(code), moisture_values)
            if not model["burnable"]:
                raise ValueError(f"fuel code {code} is nonburnable; use 0 instead")
            base = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
            gx, gy = gradients[r][c]
            aspect = degrees(atan2(-gx, -gy)) % 360 if gx or gy else 0.0
            row.append(surface_fire.calc_surface_fire_behavior_max(base, hypot(*wind) * 60, upwind, hypot(gx, gy), aspect))
        behavior.append(row)

    def rate(r: int, c: int, dr: int, dc: int) -> float:
        gx, gy = gradients[r][c]
        norm = hypot(dc, dr, gx * dc + gy * dr)
        direction = (dc / norm, dr / norm, (gx * dc + gy * dr) / norm)
        return surface_fire.calc_surface_fire_behavior_in_direction(behavior[r][c], direction)["spread_rate"] / 60

    arrival = [[float("inf") for _ in range(cols)] for _ in range(rows)]
    queue = []
    for r, c in ignitions:
        arrival[r][c] = 0.0
        heappush(queue, (0.0, r, c))
    while queue:
        time, r, c = heappop(queue)
        if time != arrival[r][c] or time > horizon:
            continue
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or behavior[nr][nc] is None:
                continue
            if dr and dc and (behavior[r + dr][c] is None or behavior[r][c + dc] is None):
                continue
            source_rate, target_rate = rate(r, c, dr, dc), rate(nr, nc, dr, dc)
            if source_rate <= 0 or target_rate <= 0:
                continue
            length = hypot(dc, dr) * cell
            candidate = time + length / (2 * source_rate) + length / (2 * target_rate)
            if candidate < arrival[nr][nc]:
                arrival[nr][nc] = candidate
                heappush(queue, (candidate, nr, nc))
    reached = sum(t <= horizon for row in arrival for t in row)
    return {
        "replay_schema_version": "standard-fuel-0.1",
        "scenario_id": scenario["scenario_id"], "seed": seed,
        "scenario_sha256": sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "provenance": "synthetic", "crs": "LOCAL_METERS",
        "engine": "pyretechnics", "engine_version": version("pyretechnics"), "units": {"arrival": "s", "affected_area": "ha"},
        "horizon_s": horizon,
        "arrival_time_s": [[round(v, 6) if isfinite(v) else None for v in row] for row in arrival],
        "affected_area_ha": reached * cell * cell / 10_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_fire_standard(json.loads(args.scenario.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
