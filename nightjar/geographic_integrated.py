"""Geographic integrated fire and aircraft decision environment.

This module couples real-world geographic scenarios (Scott & Burgan 40 fuels,
georeferenced waterbodies, satellite initial perimeters) with CL-415 aerial
operations and physical Rothermel/Pyretechnics fire spread.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from heapq import heappop, heappush
import json
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
from random import Random
import time
from typing import Any, Sequence

import numpy as np
from pyretechnics import fuel_models, surface_fire

from .observation import build_reports, observe
from .policies import candidate_cells, choose_order

REPLAY_SCHEMA_VERSION = "nightjar-geographic-replay-0.1"

# 8-neighbor grid connectivity: (drow, dcol)
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

# Default CL-415 operational kinematics
DEFAULT_CRUISE_SPEED_M_S = 80.0   # ~155 KIAS
DEFAULT_DROP_SPEED_M_S = 55.0     # ~107 KIAS
DEFAULT_REFILL_SERVICE_S = 180.0  # 3 min approach, scoop, climbout
DEFAULT_CAPACITY_L = 6137


def run_geographic_integrated(
    scenario: dict[str, Any],
    policy: str,
    *,
    policy_seed: int | None = None,
    _actions: list[dict] | None = None,
    _stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Run an event-driven decision episode on a geographic scenario.

    Parameters:
    - scenario: dict matching 'nightjar-geographic-scenario-0.1' or 'nightjar-cems-scenario-0.1'.
    - policy: 'no_intervention', 'random_valid', 'heuristic', or 'containment_heuristic'.
    - policy_seed: optional integer seed for randomized policies.
    - _actions: optional pre-scripted orders list for testing.
    - _stop_after_steps: optional step cutoff for smoke testing.
    """
    if policy not in {"no_intervention", "random_valid", "heuristic", "containment_heuristic"}:
        raise ValueError(f"Unknown policy: {policy}")

    # Extract grid and fire parameters
    grid_meta = scenario["grid"]
    rows = int(grid_meta["rows"])
    cols = int(grid_meta["cols"])
    cell_size = float(grid_meta["cell_size_m"])
    cell_area_ha = (cell_size**2) / 10000.0

    # Fuel grid: integer codes (Scott & Burgan)
    fire_dict = scenario.get("fire", scenario)
    fuel_grid = fire_dict["fuel_grid"]
    init_mask = fire_dict.get("burn_mask", fire_dict.get("initial_burn_mask"))
    horizon_s = float(fire_dict.get("horizon_s", scenario.get("horizon_s", 3600.0)))
    wind_to = fire_dict.get("wind_to_m_s", (1.5, 1.5))
    moisture = fire_dict.get("moisture", {
        "dead_1h": 0.04, "dead_10h": 0.05, "dead_100h": 0.07,
        "dead_herbaceous": 0.04, "live_herbaceous": 0.30, "live_woody": 0.60,
    })

    # Elevation & gradient calculation (flat default if omitted)
    elevation_input = fire_dict.get("elevation_m")
    if elevation_input is None:
        elevation = [[350.0 for _ in range(cols)] for _ in range(rows)]
    else:
        elevation = elevation_input

    gradients = []
    for r in range(rows):
        line = []
        for c in range(cols):
            west, east = max(c - 1, 0), min(c + 1, cols - 1)
            south, north = max(r - 1, 0), min(r + 1, rows - 1)
            gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * cell_size) if east > west else 0.0
            gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * cell_size) if north > south else 0.0
            line.append((gx, gy))
        gradients.append(line)

    wind = (float(wind_to[0]), float(wind_to[1]))
    upwind = degrees(atan2(-wind[0], -wind[1])) % 360 if wind != (0.0, 0.0) else 0.0
    wind_speed_m_min = hypot(*wind) * 60.0

    # Pyretechnics base behavior calculation
    moisture_values = (
        float(moisture["dead_1h"]), float(moisture["dead_10h"]), float(moisture["dead_100h"]),
        float(moisture["dead_herbaceous"]), float(moisture["live_herbaceous"]), float(moisture["live_woody"]),
    )
    unique_codes = {int(code) for row in fuel_grid for code in row if code != 0}
    base_cache: dict[int, Any] = {}
    for code in unique_codes:
        try:
            model = fuel_models.moisturize(fuel_models.get_fuel_model(code), moisture_values)
            if model.get("burnable", True):
                base_cache[code] = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
        except Exception:
            continue

    # Precompute cell directional fire behavior
    cell_behavior = []
    for r in range(rows):
        row_behav = []
        for c in range(cols):
            code = int(fuel_grid[r][c])
            if code == 0 or code not in base_cache:
                row_behav.append(None)
            else:
                base = base_cache[code]
                gx, gy = gradients[r][c]
                slope_tan = hypot(gx, gy)
                aspect = degrees(atan2(-gx, -gy)) % 360 if slope_tan > 0 else 0.0
                row_behav.append(surface_fire.calc_surface_fire_behavior_max(base, wind_speed_m_min, upwind, slope_tan, aspect))
        cell_behavior.append(row_behav)

    def spread_rate(r: int, c: int, dr: int, dc: int) -> float:
        """Return Rothermel forward spread rate in m/s between adjacent cells."""
        if cell_behavior[r][c] is None:
            return 0.0
        gx, gy = gradients[r][c]
        norm = hypot(dc, dr, gx * dc + gy * dr)
        if norm == 0.0:
            return 0.0
        direction = (dc / norm, dr / norm, (gx * dc + gy * dr) / norm)
        try:
            res = surface_fire.calc_surface_fire_behavior_in_direction(cell_behavior[r][c], direction)
            return max(0.001, float(res["spread_rate"]) / 60.0)
        except Exception:
            return 0.001

    def is_burnable(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and int(fuel_grid[r][c]) > 0 and cell_behavior[r][c] is not None

    # Aircraft initialization
    ac_config = scenario.get("aircraft", {})
    capacity_l = int(ac_config.get("tank_capacity_l", DEFAULT_CAPACITY_L))
    cruise_speed = float(ac_config.get("cruise_speed_m_s", DEFAULT_CRUISE_SPEED_M_S))
    refill_service_s = float(ac_config.get("refill_service_time_s", DEFAULT_REFILL_SERVICE_S))

    # Refill location in grid local meters [x, y]
    # If coordinates given in UTM, translate relative to origin
    origin_x = float(grid_meta.get("origin_utm_easting_m", 0.0))
    origin_y = float(grid_meta.get("origin_utm_northing_m", 0.0))

    if "water_scooping_utm" in ac_config:
        refill_x = float(ac_config["water_scooping_utm"][0]) - origin_x
        refill_y = float(ac_config["water_scooping_utm"][1]) - origin_y
    elif "water_waypoints" in ac_config and ac_config["water_waypoints"]:
        wp = ac_config["water_waypoints"][0]
        refill_x = float(wp.get("utm_easting_m", origin_x)) - origin_x
        refill_y = float(wp.get("utm_northing_m", origin_y)) - origin_y
    else:
        refill_x, refill_y = 0.0, 0.0

    if "initial_position_utm" in ac_config:
        init_ac_x = float(ac_config["initial_position_utm"][0]) - origin_x
        init_ac_y = float(ac_config["initial_position_utm"][1]) - origin_y
    else:
        init_ac_x, init_ac_y = cols * cell_size * 0.8, rows * cell_size * 0.8

    # State containers
    arrival: list[list[float | None]] = [[None] * cols for _ in range(rows)]
    barrier = [[False] * cols for _ in range(rows)]
    best = [[float("inf")] * cols for _ in range(rows)]
    queue: list[tuple[float, int, int]] = []

    # Initialize queue with initial burn perimeters
    initial_burned_count = 0
    for r in range(rows):
        for c in range(cols):
            if init_mask[r][c]:
                arrival[r][c] = 0.0
                best[r][c] = 0.0
                initial_burned_count += 1
                if is_burnable(r, c):
                    heappush(queue, (0.0, r, c))

    def advance_fire(until_s: float) -> None:
        """Advance physical Rothermel fire propagation up to until_s."""
        while queue and queue[0][0] <= until_s + 1e-9:
            cur_time, r, c = heappop(queue)
            if cur_time != best[r][c] or barrier[r][c] or not is_burnable(r, c):
                continue
            arrival[r][c] = cur_time

            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not is_burnable(nr, nc) or barrier[nr][nc]:
                    continue
                if dr and dc and (barrier[r + dr][c] or barrier[r][c + dc] or not is_burnable(r + dr, c) or not is_burnable(r, c + dc)):
                    continue

                r_source = spread_rate(r, c, dr, dc)
                r_target = spread_rate(nr, nc, dr, dc)
                if r_source <= 0.0 or r_target <= 0.0:
                    continue

                dist = hypot(dc * cell_size, dr * cell_size)
                edge_dt = (dist / (2.0 * r_source)) + (dist / (2.0 * r_target))
                candidate_time = cur_time + edge_dt

                if candidate_time < best[nr][nc]:
                    best[nr][nc] = candidate_time
                    heappush(queue, (candidate_time, nr, nc))

    # Water drop execution on physical landscape
    drops_log: list[dict[str, Any]] = []

    def apply_water_drop(drop_time: float, target_x_m: float, target_y_m: float) -> list[list[int]]:
        """Apply water drop to cell and immediate neighborhood, establishing a physical barrier."""
        target_c = int(round(target_x_m / cell_size))
        target_r = int(round(target_y_m / cell_size))
        treated = []
        # Target cell plus 4-neighbors within drop footprint
        drop_footprint = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in drop_footprint:
            nr, nc = target_r + dr, target_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if is_burnable(nr, nc) and arrival[nr][nc] is None and not barrier[nr][nc]:
                    barrier[nr][nc] = True
                    treated.append([nr, nc])
        drops_log.append({
            "time_s": round(drop_time, 2),
            "target_local_m": [round(target_x_m, 2), round(target_y_m, 2)],
            "target_utm_m": [round(target_x_m + origin_x, 2), round(target_y_m + origin_y, 2)],
            "treated_cells_count": len(treated),
            "treated_cells": treated,
        })
        return treated

    # Observation schedule
    raw_schedule = scenario.get("observation_schedule", [])
    schedule = []
    for idx, item in enumerate(raw_schedule):
        obs_t = float(item["observed_at_s"])
        avail_t = float(item.get("available_at_s", item.get("delivered_at_s", obs_t)))
        src_id = str(item.get("source_id", f"survey-{idx}"))
        schedule.append({
            "source_id": src_id,
            "observed_at_s": obs_t,
            "available_at_s": avail_t,
            "delivered_at_s": avail_t,
            "coverage_cells": [[r, c] for r in range(rows) for c in range(cols)],
        })
    schedule = sorted(schedule, key=lambda x: x["observed_at_s"])

    # Discrete aircraft simulation state
    cur_sim_time = 0.0
    aircraft_pos = [init_ac_x, init_ac_y]
    aircraft_water = capacity_l
    aircraft_phase = "idle"
    total_flight_dist_m = 0.0
    total_water_dropped_l = 0
    refills_count = 0
    drops_count = 0

    known_treated_cells: list[list[int]] = []
    reports: list[dict] = []
    delivered_reports: list[dict] = []
    pending_delivery: list[dict] = []
    schedule_idx = 0

    rng = Random(scenario.get("seed", 42) if policy_seed is None else policy_seed)
    decisions_log: list[dict] = []
    step_count = 0

    def sync_observations(to_time: float) -> None:
        """Capture observations and deliver ready reports up to to_time."""
        nonlocal schedule_idx
        # 1. Capture observations when observed_at_s occurs
        while schedule_idx < len(schedule) and schedule[schedule_idx]["observed_at_s"] <= to_time + 1e-9:
            item = schedule[schedule_idx]
            advance_fire(item["observed_at_s"])
            captured = build_reports(arrival, [item])
            pending_delivery.extend(captured)
            reports.extend(captured)
            schedule_idx += 1

        # 2. Deliver reports when available_at_s occurs
        delivered_now = [r for r in pending_delivery if r["available_at_s"] <= to_time + 1e-9]
        for r in delivered_now:
            if r not in delivered_reports:
                delivered_reports.append(r)

    # Main decision and event progression loop
    while cur_sim_time < horizon_s:
        if _stop_after_steps is not None and step_count >= _stop_after_steps:
            break

        sync_observations(cur_sim_time)
        advance_fire(cur_sim_time)

        # Build current observation for the policy (STRICT ZERO-LEAKAGE)
        fire_view = observe(delivered_reports, cur_sim_time, rows, cols)
        current_obs = {
            "time_s": cur_sim_time,
            "fire": fire_view,
            "aircraft": {
                "position_m": [round(aircraft_pos[0], 2), round(aircraft_pos[1], 2)],
                "water_l": aircraft_water,
                "phase": aircraft_phase,
                "target_m": None,
                "refill_remaining_s": 0.0,
            },
            "fuel_model": [[str(val) for val in row] for row in fuel_grid],
            "water_sources": [[refill_x, refill_y]],
            "water_source_m": [refill_x, refill_y],
            "midflame_wind_to_m_s": list(wind),
            "cell_size_m": cell_size,
            "elevation_m": [[0.0] * cols for _ in range(rows)] if scenario.get("elevation_m") is None else scenario["elevation_m"],
            "known_treated_cells": known_treated_cells,
        }

        # Policy order selection
        if _actions is not None and step_count < len(_actions):
            order = _actions[step_count]
        else:
            candidates = candidate_cells(current_obs)
            order = choose_order(policy, current_obs, candidates=candidates, rng=rng)

        order_type = order["type"]
        step_count += 1

        if order_type == "HOLD":
            # Advance time by decision interval (e.g. 60s)
            advance_dt = min(60.0, horizon_s - cur_sim_time)
            cur_sim_time += advance_dt
            decisions_log.append({
                "time_s": round(cur_sim_time, 2),
                "order": order,
                "aircraft_water_l": aircraft_water,
            })

        elif order_type == "RETURN_TO_REFILL":
            # Fly to refill waterbody
            dist = hypot(refill_x - aircraft_pos[0], refill_y - aircraft_pos[1])
            transit_time = dist / cruise_speed
            total_time = transit_time + refill_service_s

            target_time = min(horizon_s, cur_sim_time + total_time)
            # Advance fire to arrival and refill completion
            advance_fire(cur_sim_time + transit_time)
            advance_fire(target_time)

            total_flight_dist_m += dist * 2.0  # roundtrip transit consideration
            aircraft_pos = [refill_x, refill_y]
            aircraft_water = capacity_l
            refills_count += 1
            cur_sim_time = target_time

            decisions_log.append({
                "time_s": round(cur_sim_time, 2),
                "order": order,
                "refill_transit_s": round(transit_time, 2),
                "aircraft_water_l": aircraft_water,
            })

        elif order_type == "DROP":
            target_x, target_y = order["target_m"]
            dist = hypot(target_x - aircraft_pos[0], target_y - aircraft_pos[1])
            transit_time = dist / cruise_speed
            drop_time = cur_sim_time + transit_time

            if drop_time <= horizon_s:
                advance_fire(drop_time)
                # Execute water drop
                treated = apply_water_drop(drop_time, target_x, target_y)
                known_treated_cells.extend(treated)
                aircraft_water = max(0, aircraft_water - capacity_l)
                drops_count += 1
                total_water_dropped_l += capacity_l
                total_flight_dist_m += dist
                aircraft_pos = [target_x, target_y]
                cur_sim_time = drop_time
            else:
                # Horizon reached during transit
                cur_sim_time = horizon_s

            decisions_log.append({
                "time_s": round(cur_sim_time, 2),
                "order": order,
                "aircraft_water_l": aircraft_water,
            })

    # Final fire advancement to horizon
    advance_fire(horizon_s)
    sync_observations(horizon_s)

    # Compute final metrics
    burned_cells_final = sum(1 for r in range(rows) for c in range(cols) if arrival[r][c] is not None and arrival[r][c] <= horizon_s)
    final_burned_ha = round(burned_cells_final * cell_area_ha, 4)
    initial_burned_ha = round(initial_burned_count * cell_area_ha, 4)

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "scenario_id": scenario.get("scenario_id", "geographic-scenario"),
        "policy": policy,
        "policy_seed": policy_seed,
        "horizon_s": horizon_s,
        "grid": {
            "rows": rows,
            "cols": cols,
            "cell_size_m": cell_size,
            "extent_ha": round(rows * cols * cell_area_ha, 2),
        },
        "outcomes": {
            "initial_burned_ha": initial_burned_ha,
            "final_burned_ha": final_burned_ha,
            "net_expansion_ha": round(final_burned_ha - initial_burned_ha, 4),
            "burned_cells_count": burned_cells_final,
        },
        "aircraft_metrics": {
            "drops_count": drops_count,
            "refills_count": refills_count,
            "total_water_dropped_l": total_water_dropped_l,
            "total_flight_distance_km": round(total_flight_dist_m / 1000.0, 3),
            "rejected_orders_count": 0,
        },
        "decisions": decisions_log,
        "drops": drops_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Path to geographic scenario JSON")
    parser.add_argument("--policy", choices=("no_intervention", "random_valid", "heuristic", "containment_heuristic"), default="heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenario_data = json.loads(args.scenario.read_text())
    result = run_geographic_integrated(scenario_data, policy=args.policy, policy_seed=args.seed)

    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote replay artifact -> {args.output}")
    else:
        print(payload)

    print("\n--- Summary ---")
    print(f"Policy: {result['policy']}")
    print(f"Initial Burned Area: {result['outcomes']['initial_burned_ha']} ha")
    print(f"Final Burned Area:   {result['outcomes']['final_burned_ha']} ha (+{result['outcomes']['net_expansion_ha']} ha)")
    print(f"Drops: {result['aircraft_metrics']['drops_count']} | Refills: {result['aircraft_metrics']['refills_count']} | Water: {result['aircraft_metrics']['total_water_dropped_l']} L")


if __name__ == "__main__":
    main()
