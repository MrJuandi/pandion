"""Physical surface-fire spread simulation and satellite validation for EMSR578 AOI01.

This module simulates Rothermel surface fire spread over the georeferenced EMSR578 AOI01
scenario using Pyretechnics and Scott & Burgan 40 fuel models, propagating via shortest-arrival
Dijkstra from the initial perimeter, and evaluating spatial overlap against Copernicus MONIT01.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from heapq import heappop, heappush
from importlib.metadata import version
import json
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from pyretechnics import fuel_models, surface_fire

from .pilot_emsr578 import evaluate_spatial_overlap, extract_observed_mask

# 8-neighbor grid connectivity: (drow, dcol) where row increases northward, col eastward
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

# Documented Lleida June 2022 heatwave 97th percentile fuel moisture fractions
DEFAULT_MOISTURE = {
    "dead_1h": 0.04,
    "dead_10h": 0.05,
    "dead_100h": 0.07,
    "dead_herbaceous": 0.04,
    "live_herbaceous": 0.30,
    "live_woody": 0.60,
}

# Synoptic midflame wind pushing towards NNE (approx 2.0 m/s midflame)
DEFAULT_WIND_TO_M_S = (0.8, 1.8)

# Total elapsed seconds between DEL (2022-06-17 10:06 UTC) and MONIT01 (2022-06-18 10:47 UTC)
DEFAULT_HORIZON_S = 88860.0

SIMULATION_SCHEMA_VERSION = "nightjar-pilot-simulation-0.1"


def simulate_pilot_spread(
    scenario: dict[str, Any],
    horizon_s: float = DEFAULT_HORIZON_S,
    wind_to_m_s: Sequence[float] = DEFAULT_WIND_TO_M_S,
    moisture: dict[str, float] | None = None,
    elevation_m: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Simulate surface-fire spread over an EMSR578 pilot scenario.

    Parameters:
    - scenario: dict matching schema 'nightjar-emsr578-pilot-0.1'.
    - horizon_s: simulation horizon in seconds (default 88,860 s = 24.68 h).
    - wind_to_m_s: midflame wind vector [east, north] in m/s.
    - moisture: dict of 6 moisture fractions.
    - elevation_m: 2D elevation grid (defaults to constant 350.0m if None).
    """
    if horizon_s <= 0 or not isfinite(horizon_s):
        raise ValueError("horizon_s must be a positive finite number")
    if len(wind_to_m_s) != 2 or any(not isfinite(v) for v in wind_to_m_s):
        raise ValueError("wind_to_m_s must be a 2-element sequence of finite numbers")

    rows = scenario["grid"]["rows"]
    cols = scenario["grid"]["cols"]
    cell_size_m = float(scenario["grid"]["cell_size_m"])
    cell_area_ha = float(scenario["grid"]["cell_area_ha"])
    fuel = scenario["fuel_model"]
    init_mask = scenario["initial_burn_mask"]

    if moisture is None:
        moisture = deepcopy(DEFAULT_MOISTURE)
    moisture_fields = ("dead_1h", "dead_10h", "dead_100h", "dead_herbaceous", "live_herbaceous", "live_woody")
    if set(moisture) != set(moisture_fields):
        raise ValueError(f"moisture must define {moisture_fields}")
    moisture_values = tuple(float(moisture[k]) for k in moisture_fields)

    # Elevation & gradient calculation
    if elevation_m is None:
        elevation = [[350.0 for _ in range(cols)] for _ in range(rows)]
    else:
        elevation = [[float(v) for v in row] for row in elevation_m]
        if len(elevation) != rows or any(len(r) != cols for r in elevation):
            raise ValueError(f"elevation_m must have shape ({rows}, {cols})")

    t_start = time.perf_counter()

    gradients = []
    for r in range(rows):
        row = []
        for c in range(cols):
            west, east = max(c - 1, 0), min(c + 1, cols - 1)
            south, north = max(r - 1, 0), min(r + 1, rows - 1)
            gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * cell_size_m) if east > west else 0.0
            gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * cell_size_m) if north > south else 0.0
            row.append((gx, gy))
        gradients.append(row)

    wind = (float(wind_to_m_s[0]), float(wind_to_m_s[1]))
    upwind = degrees(atan2(-wind[0], -wind[1])) % 360 if wind != (0.0, 0.0) else 0.0
    wind_speed_m_min = hypot(*wind) * 60.0

    # Cache base moisturized fuel behavior for each unique model code
    unique_codes = {int(code) for row in fuel for code in row if code != 0}
    base_cache: dict[int, Any] = {}
    for code in unique_codes:
        model = fuel_models.moisturize(fuel_models.get_fuel_model(code), moisture_values)
        if model["burnable"]:
            base_cache[code] = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)

    # Compute cell behavior
    behavior = []
    for r in range(rows):
        row = []
        for c in range(cols):
            code = fuel[r][c]
            if code == 0 or code not in base_cache:
                row.append(None)
            else:
                base = base_cache[code]
                gx, gy = gradients[r][c]
                slope_tan = hypot(gx, gy)
                aspect = degrees(atan2(-gx, -gy)) % 360 if slope_tan > 0 else 0.0
                row.append(surface_fire.calc_surface_fire_behavior_max(base, wind_speed_m_min, upwind, slope_tan, aspect))
        behavior.append(row)

    def rate(r: int, c: int, dr: int, dc: int) -> float:
        gx, gy = gradients[r][c]
        norm = hypot(dc, dr, gx * dc + gy * dr)
        direction = (dc / norm, dr / norm, (gx * dc + gy * dr) / norm)
        return surface_fire.calc_surface_fire_behavior_in_direction(behavior[r][c], direction)["spread_rate"] / 60.0

    # Initialize arrival grid with initial burned perimeter
    arrival = [[float("inf") for _ in range(cols)] for _ in range(rows)]
    queue: list[tuple[float, int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if init_mask[r][c]:
                arrival[r][c] = 0.0
                if behavior[r][c] is not None:
                    heappush(queue, (0.0, r, c))

    # Priority queue propagation (Dijkstra shortest arrival)
    while queue:
        t_arr, r, c = heappop(queue)
        if t_arr != arrival[r][c] or t_arr > horizon_s:
            continue
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or behavior[nr][nc] is None:
                continue
            # Block diagonal traversal between two nonburnables
            if dr and dc and (behavior[r + dr][c] is None or behavior[r][c + dc] is None):
                continue
            sr = rate(r, c, dr, dc)
            tr = rate(nr, nc, dr, dc)
            if sr <= 0.0 or tr <= 0.0:
                continue
            length = hypot(dc, dr) * cell_size_m
            candidate = t_arr + length / (2.0 * sr) + length / (2.0 * tr)
            if candidate < arrival[nr][nc]:
                arrival[nr][nc] = candidate
                heappush(queue, (candidate, nr, nc))

    wall_time_s = time.perf_counter() - t_start

    simulated_mask = [[(v <= horizon_s) for v in row] for row in arrival]
    simulated_cells = sum(sum(row) for row in simulated_mask)
    simulated_area_ha = simulated_cells * cell_area_ha

    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "scenario_id": scenario.get("scenario_id", "unknown"),
        "scenario_crs": scenario.get("spatial_reference", {}).get("crs", "EPSG:25831"),
        "grid": {
            "rows": rows,
            "cols": cols,
            "cell_size_m": cell_size_m,
            "cell_area_ha": cell_area_ha,
        },
        "engine": {
            "name": "pyretechnics",
            "version": version("pyretechnics"),
            "model_family": "Scott & Burgan 40",
            "propagation": "8-neighbor shortest arrival Dijkstra",
        },
        "parameters": {
            "horizon_s": horizon_s,
            "horizon_h": round(horizon_s / 3600.0, 2),
            "midflame_wind_to_m_s": list(wind),
            "wind_speed_m_s": round(hypot(*wind), 2),
            "wind_direction_upwind_deg": round(upwind, 1),
            "fuel_moisture_fraction": moisture,
        },
        "initial_fire_cells": int(sum(sum(row) for row in init_mask)),
        "initial_fire_ha": round(int(sum(sum(row) for row in init_mask)) * cell_area_ha, 2),
        "simulated_fire_cells": simulated_cells,
        "simulated_fire_ha": round(simulated_area_ha, 2),
        "net_growth_ha": round(simulated_area_ha - int(sum(sum(row) for row in init_mask)) * cell_area_ha, 2),
        "computation_time_s": round(wall_time_s, 4),
        "arrival_time_s": [[round(v, 2) if isfinite(v) else None for v in row] for row in arrival],
        "simulated_burn_mask": simulated_mask,
    }


def evaluate_simulation_vs_satellite(
    simulation: dict[str, Any],
    comparison_zip: Path,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate simulated burn mask against a Copernicus vector ZIP product (e.g. monit01.zip)."""
    sim_mask = np.array(simulation["simulated_burn_mask"], dtype=bool)
    rows = simulation["grid"]["rows"]
    cols = simulation["grid"]["cols"]
    cell_size = simulation["grid"]["cell_size_m"]
    cell_area_ha = simulation["grid"]["cell_area_ha"]

    if scenario is not None:
        bounds = scenario["spatial_reference"]["bounds_utm31n"]
    else:
        # Default bounds of EMSR578 AOI01
        bounds = {"min_x": 326636.4, "max_x": 342209.4, "min_y": 4636071.6, "max_y": 4651196.9}

    obs_mask = extract_observed_mask(comparison_zip, bounds, rows, cols, cell_size)
    metrics = evaluate_spatial_overlap(sim_mask, obs_mask, cell_area_ha)

    metrics["simulation_parameters"] = simulation.get("parameters", {})
    metrics["simulation_engine"] = simulation.get("engine", {})
    metrics["comparison_product"] = str(comparison_zip)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    parser_run = subparsers.add_parser("run", help="Run physical spread simulation on scenario manifest")
    parser_run.add_argument("--scenario", type=Path, required=True)
    parser_run.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    parser_run.add_argument("--wind-east", type=float, default=DEFAULT_WIND_TO_M_S[0])
    parser_run.add_argument("--wind-north", type=float, default=DEFAULT_WIND_TO_M_S[1])
    parser_run.add_argument("--output", type=Path)

    # Subcommand: evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate simulation against satellite observed perimeter")
    parser_eval.add_argument("--simulation", type=Path, required=True)
    parser_eval.add_argument("--comparison-zip", type=Path, default=Path("data/raw/emsr578/monit01.zip"))
    parser_eval.add_argument("--scenario", type=Path)
    parser_eval.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "run":
        scen = json.loads(args.scenario.read_text())
        sim_res = simulate_pilot_spread(
            scenario=scen,
            horizon_s=args.horizon_s,
            wind_to_m_s=(args.wind_east, args.wind_north),
        )
        content = json.dumps(sim_res, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
            print(f"Simulation saved to {args.output} ({sim_res['simulated_fire_ha']} ha in {sim_res['computation_time_s']}s)")
        else:
            print(f"Simulated {sim_res['simulated_fire_ha']} ha ({sim_res['simulated_fire_cells']} cells) in {sim_res['computation_time_s']}s")

    elif args.command == "evaluate":
        sim_data = json.loads(args.simulation.read_text())
        scen_data = json.loads(args.scenario.read_text()) if args.scenario else None
        eval_res = evaluate_simulation_vs_satellite(sim_data, args.comparison_zip, scen_data)
        content = json.dumps(eval_res, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
            print(f"Evaluation saved to {args.output}")
        print(f"Spatial Overlap Metrics vs {args.comparison_zip.name}:")
        print(f"  IoU (Jaccard): {eval_res['iou']}")
        print(f"  Dice:          {eval_res['dice']}")
        print(f"  Simulated:     {eval_res['predicted_area_ha']} ha")
        print(f"  Observed:      {eval_res['observed_area_ha']} ha")
        print(f"  Commission:    {eval_res['commission_rate']:.4f}")
        print(f"  Omission:      {eval_res['omission_rate']:.4f}")


if __name__ == "__main__":
    main()
