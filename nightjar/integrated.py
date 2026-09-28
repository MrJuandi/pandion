"""Synthetic event-driven fire and aircraft episode with a provisional water barrier."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from heapq import heappop, heappush
import json
from math import hypot, isfinite
from pathlib import Path
from random import Random

from .aircraft import CAPACITY_L, run_episode as run_aircraft, validate_scenario as validate_aircraft
from .fire import NEIGHBORS
from .fire_rothermel import validate_scenario as validate_fire
from .observation import build_reports, observe
from .policies import candidate_cells, choose_order
from .rothermel import FM1_MOISTURE_EXTINCTION, fm1_directional_rate


def run_integrated(
    scenario: dict,
    policy: str,
    *,
    policy_seed: int | None = None,
    _actions: list[dict] | None = None,
    _stop_after_steps: int | None = None,
) -> dict:
    if scenario.get("schema_version") != "integrated-synthetic-0.1":
        raise ValueError("schema_version must be integrated-synthetic-0.1")
    fire_input, aircraft_input = scenario["fire"], scenario["aircraft"]
    fire, aircraft = validate_fire(fire_input), validate_aircraft(aircraft_input)
    for key in ("scenario_id", "seed", "crs", "provenance"):
        if fire_input[key] != aircraft_input[key] or fire_input[key] != scenario.get(key):
            raise ValueError(f"fire, aircraft and integrated {key} must agree")
    if fire["horizon_s"] != aircraft["horizon_s"]:
        raise ValueError("fire and aircraft horizons must agree")
    if aircraft["aircraft"]["initial_water_l"] != CAPACITY_L:
        raise ValueError("integrated experiment requires one full initial load")
    radius = float(scenario["water_effect"]["radius_m"])
    if not isfinite(radius) or radius < 0:
        raise ValueError("water-effect radius_m must be finite and nonnegative")
    rows, cols, cell = fire["rows"], fire["cols"], fire["cell_size_m"]
    raw_schedule = scenario["observation_schedule"]
    if not isinstance(raw_schedule, list):
        raise ValueError("observation_schedule must be a list")
    schedule = []
    for item in raw_schedule:
        if not isinstance(item, dict):
            raise ValueError("each observation schedule entry must be an object")
        if item.get("coverage") == "all":
            if "coverage_cells" in item:
                raise ValueError("use coverage='all' or coverage_cells, not both")
            schedule.append({**item, "coverage_cells": [[r, c] for r in range(rows) for c in range(cols)]})
        else:
            schedule.append(item)
    # The shared observation validator checks times, coverage and duplicates.
    build_reports([[None] * cols for _ in range(rows)], schedule)
    schedule = sorted(schedule, key=lambda item: item["observed_at_s"])
    rng = Random(scenario["seed"] if policy_seed is None else policy_seed)

    def burnable(r: int, c: int) -> bool:
        return fire["fuel_model"][r][c] == "FM1" and fire["moisture"][r][c] < FM1_MOISTURE_EXTINCTION

    elevation = fire["elevation"]
    gradients = []
    for r in range(rows):
        line = []
        for c in range(cols):
            west, east = max(c - 1, 0), min(c + 1, cols - 1)
            south, north = max(r - 1, 0), min(r + 1, rows - 1)
            gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * cell) if east > west else 0.0
            gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * cell) if north > south else 0.0
            line.append((gx, gy))
        gradients.append(line)

    arrival: list[list[float | None]] = [[None] * cols for _ in range(rows)]
    barrier = [[False] * cols for _ in range(rows)]
    queue: list[tuple[float, int, int]] = [(0.0, r, c) for r, c in fire["ignitions"]]
    best = [[float("inf")] * cols for _ in range(rows)]
    for r, c in fire["ignitions"]:
        best[r][c] = 0.0
    reports: list[dict] = []
    drops: list[dict] = []
    decisions: list[dict] = []
    orders: list[dict] = []
    aircraft_state = {
        "position_m": aircraft["aircraft"]["initial_position_m"].copy(),
        "water_l": CAPACITY_L, "phase": "idle", "target_m": None, "refill_remaining_s": 0.0,
    }
    latest_aircraft = None
    processed_drops = 0
    schedule_index = 0
    pending_observation = None

    def advance_fire(until_s: float) -> None:
        while queue and queue[0][0] <= until_s + 1e-9:
            time, r, c = heappop(queue)
            if time != best[r][c] or arrival[r][c] is not None or barrier[r][c] or not burnable(r, c):
                continue
            arrival[r][c] = time
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not burnable(nr, nc) or barrier[nr][nc]:
                    continue
                if dr and dc and (not burnable(r + dr, c) or not burnable(r, c + dc) or barrier[r + dr][c] or barrier[r][c + dc]):
                    continue
                direction = (dc, dr)
                source_rate = fm1_directional_rate(fire["moisture"][r][c], fire["wind"], gradients[r][c], direction)
                target_rate = fm1_directional_rate(fire["moisture"][nr][nc], fire["wind"], gradients[nr][nc], direction)
                distance = hypot(dc, dr) * cell
                candidate = time + distance / (2 * source_rate) + distance / (2 * target_rate)
                if candidate < best[nr][nc]:
                    best[nr][nc] = candidate
                    heappush(queue, (candidate, nr, nc))

    def apply_drop(event: dict) -> None:
        x, y = event["position_m"]
        newly_blocked = []
        for r in range(rows):
            for c in range(cols):
                if burnable(r, c) and arrival[r][c] is None and not barrier[r][c] and hypot(c * cell - x, r * cell - y) <= radius + 1e-9:
                    barrier[r][c] = True
                    newly_blocked.append([r, c])
        drops.append({**event, "newly_blocked_cells": newly_blocked})

    def advance_timeline(until_s: float, new_drops: list[dict]) -> None:
        nonlocal schedule_index
        events = []
        while schedule_index < len(schedule) and float(schedule[schedule_index]["observed_at_s"]) <= until_s + 1e-9:
            item = schedule[schedule_index]
            events.append((float(item["observed_at_s"]), 0, item))
            schedule_index += 1
        events.extend((event["time_s"], 1, event) for event in new_drops)
        for time, kind, item in sorted(events, key=lambda value: (value[0], value[1])):
            advance_fire(time)
            if kind == 0:
                reports.extend(build_reports(arrival, [item]))
            else:
                apply_drop(item)
        advance_fire(until_s)

    advance_timeline(0.0, [])
    time_s = 0.0
    horizon = fire["horizon_s"]
    interval = aircraft["decision_interval_s"]
    while time_s < horizon - 1e-9:
        fire_view = observe(reports, time_s, rows, cols)
        known_treated = [[r, c] for r in range(rows) for c in range(cols)
                         if any(hypot(c * cell - drop["position_m"][0], r * cell - drop["position_m"][1]) <= radius + 1e-9
                                for drop in drops)]
        observation = {
            "time_s": time_s, "fire": fire_view, "aircraft": deepcopy(aircraft_state),
            "fuel_model": deepcopy(fire["fuel_model"]), "midflame_wind_to_m_s": list(fire["wind"]),
            "cell_size_m": cell, "known_treated_cells": known_treated,
        }
        candidates = candidate_cells(observation)
        if _stop_after_steps is not None and len(decisions) >= _stop_after_steps:
            pending_observation = observation
            break
        if _actions is not None and len(decisions) < len(_actions):
            order = deepcopy(_actions[len(decisions)])
        else:
            order = choose_order(policy, observation, candidates, rng)
        orders.append(order)
        next_time = min(time_s + interval, horizon)
        stepped_aircraft = dict(aircraft_input)
        stepped_aircraft["horizon_s"] = next_time
        latest_aircraft = run_aircraft(stepped_aircraft, orders)
        aircraft_state = latest_aircraft["final_state"]
        all_drops = [event for event in latest_aircraft["events"] if event["kind"] == "drop"]
        advance_timeline(next_time, all_drops[processed_drops:])
        processed_drops = len(all_drops)
        decisions.append({"time_s": time_s, "observation": observation,
                          "candidate_cells": candidates, "order": order,
                          "accepted": latest_aircraft["decisions"][-1]["accepted"]})
        time_s = next_time

    affected = sum(value is not None for line in arrival for value in line) * cell * cell / 10_000
    result = {
        "replay_schema_version": "integrated-synthetic-0.1",
        "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
        "policy": "external_actions" if _actions is not None else policy,
        "policy_seed": scenario["seed"] if policy_seed is None else policy_seed,
        "scenario_sha256": sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "provenance": "synthetic", "crs": "LOCAL_METERS",
        "units": {"time": "s", "distance": "m", "water": "L", "area": "ha"},
        "horizon_s": horizon, "affected_area_ha": affected,
        "arrival_time_s": [[round(value, 6) if value is not None else None for value in line] for line in arrival],
        "water_barrier_cells": [[r, c] for r in range(rows) for c in range(cols) if barrier[r][c]],
        "decisions": decisions, "drops": drops,
        "aircraft_metrics": latest_aircraft["metrics"] if latest_aircraft else {"flight_s": 0.0, "water_dropped_l": 0.0, "rejected_orders": 0},
        "aircraft_final_state": aircraft_state,
    }
    if _stop_after_steps is not None:
        result["pending_observation"] = pending_observation
    if "time_origin_utc" in scenario:
        result["time_origin_utc"] = scenario["time_origin_utc"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--policy", choices=("no_intervention", "random_valid", "heuristic"), required=True)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_integrated(json.loads(args.scenario.read_text()), args.policy, policy_seed=args.policy_seed)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
