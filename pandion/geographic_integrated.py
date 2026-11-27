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
from math import atan2, degrees, hypot, isfinite, log1p
from pathlib import Path
from random import Random
import time
from typing import Any, Sequence

import numpy as np
from pyretechnics import fuel_models, surface_fire

from .environment import MOISTURE_KEYS, delivered_weather, edge_arrival, validate_forcing
from .observation import build_reports, observe
from .policies import MAX_STACKED_DROPS, candidate_cells, choose_order

REPLAY_SCHEMA_VERSION = "nightjar-geographic-replay-0.4"
OBSERVATION_MODES = ("realtime", "scheduled")

# 8-neighbor grid connectivity: (drow, dcol)
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

# Default CL-415 operational kinematics
DEFAULT_CRUISE_SPEED_M_S = 80.0   # ~155 KIAS
DEFAULT_DROP_SPEED_M_S = 55.0     # ~107 KIAS
DEFAULT_REFILL_SERVICE_S = 180.0  # 3 min approach, scoop, climbout
DEFAULT_CAPACITY_L = 6137

# CL-415 water drop effect (D-74, D-76). One drop treats the target cell: an
# effective CL-415 drop holds about 100 m of line (McFayden et al. 2023, Table 8-3
# and A8.5), one cell edge. A treated cell stops a front only if the fireline
# intensity of the spread reaching it is within what its stacked drops hold in the
# open grass fuel type (A8.5 O-1a): one drop up to 1,000 kW/m, three up to
# 3,200 kW/m; stronger fronts breach whatever the water. Up to MAX_STACKED_DROPS
# drops can be stacked on a cell.
DROP_HOLD_LIMITS_KW_M = ((1, 1000.0), (3, 3200.0))  # (drops needed, intensity held)
# A front that breaches a treated cell is delayed by the time it needs to heat the
# cell's water to boiling and evaporate it (D-78): delay = water mass x energy per kg
# / (fireline intensity x cell width). All the water is assumed to reach the fuel, so
# this is an upper bound on the slowing; no further empirical factor is applied.
WATER_HEATING_KJ_KG = 4.186 * (100.0 - 20.0)  # liquid water from 20 C to boiling
WATER_EVAPORATION_KJ_KG = 2257.0
DROP_HOLD_MAX_INTENSITY_KW_M = DROP_HOLD_LIMITS_KW_M[0][1]


def drops_needed(intensity_kw_m: float) -> float:
    """Stacked drops needed to hold a front of this intensity; inf when none can."""
    for drops, limit in DROP_HOLD_LIMITS_KW_M:
        if intensity_kw_m <= limit:
            return drops
    return float("inf")


# Thermal sensor (D-94): a burning cell is read as one brightness byte, log-scaled heat of the cell as a
# whole (its largest fireline intensity in any direction) from 76 (0 kW/m) to 255 (20,000 kW/m and above),
# clear of burned ground and halo (at most 51) and spread over most of the byte (D-96). It is what a camera could show, not an intensity per direction.
THERMAL_SCALE_KW_M = 50.0
THERMAL_MAX_KW_M = 20000.0
THERMAL_FLOOR = 0.3


def thermal_byte(intensity_kw_m: float) -> int:
    """Brightness (0-255) of a burning cell with this heat."""
    level = min(log1p(max(intensity_kw_m, 0.0) / THERMAL_SCALE_KW_M) / log1p(THERMAL_MAX_KW_M / THERMAL_SCALE_KW_M), 1.0)
    return round(255 * (THERMAL_FLOOR + (1.0 - THERMAL_FLOOR) * level))


POLICIES = ("no_intervention", "random_valid", "heuristic", "containment_heuristic", "hold_aware_heuristic",
            "blind_hold_aware_heuristic")


def run_geographic_integrated(
    scenario: dict[str, Any],
    policy: str,
    *,
    policy_seed: int | None = None,
    observation_mode: str | None = None,
    _actions: list[dict] | None = None,
    _stop_after_steps: int | None = None,
    _record_observations: bool = True,
) -> dict[str, Any]:
    """Run an event-driven decision episode on a geographic scenario.

    Parameters:
    - scenario: dict matching 'nightjar-geographic-scenario-0.1/0.2' or 'nightjar-cems-scenario-0.1'.
    - policy: one of POLICIES ('no_intervention', 'random_valid', 'heuristic',
      'containment_heuristic', 'hold_aware_heuristic' or 'blind_hold_aware_heuristic').
    - policy_seed: optional integer seed for randomized policies.
    - observation_mode: 'realtime' (default; full grid and current weather at
      every decision, no delivery delay) or 'scheduled' (legacy delayed
      observation_schedule). Falls back to scenario['observation_mode'].
    - _actions: optional pre-scripted orders list for testing.
    - _stop_after_steps: optional step cutoff for smoke testing.
    - _record_observations: copy each decision observation into the replay
      audit log (default). Training prefixes disable it: the copies grow
      quadratically with episode length and nothing in training reads them.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    episode = geographic_episode(scenario, policy, policy_seed=policy_seed, observation_mode=observation_mode,
                                 truncated=_stop_after_steps is not None,
                                 record_observations=_record_observations)
    rng = Random(scenario.get("seed", 42) if policy_seed is None else policy_seed)
    behavior = None
    if policy == "hold_aware_heuristic":
        from .action_features import ObservedFireBehavior  # lazy: action_features imports this module
        behavior = ObservedFireBehavior()
    step = 0
    try:
        observation, _, _ = next(episode)
        while True:
            if _stop_after_steps is not None and step >= _stop_after_steps:
                order = None
            elif _actions is not None and step < len(_actions):
                order = _actions[step]
            else:
                order = choose_order(policy, observation, candidates=candidate_cells(observation), rng=rng,
                                     behavior=behavior)
            step += 1
            observation, _, _ = episode.send(order)
    except StopIteration as finished:
        return finished.value


def geographic_episode(
    scenario: dict[str, Any],
    policy: str,
    *,
    policy_seed: int | None = None,
    observation_mode: str | None = None,
    truncated: bool = False,
    record_observations: bool = True,
):
    """Step a geographic episode as a generator; simulator state persists between decisions.

    Yields ``(observation, affected_area_ha, drops_log)`` at each decision and
    receives the order to execute via ``send``. Sending ``None`` stops the
    episode at that decision (only when ``truncated``); the replay is the
    generator's return value. Truncated episodes are scored at the stop time
    and record ``pending_observation``; full episodes advance to the horizon.
    ``policy`` and ``policy_seed`` are recorded in the replay only: the caller
    chooses orders. ``run_geographic_integrated`` is the batch driver and
    ``GeographicTrainingEnv`` the step driver.
    """
    observation_mode = observation_mode or scenario.get("observation_mode", "realtime")
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"Unknown observation_mode: {observation_mode}")
    water_model = scenario.get("water_model", {})
    water_on_burning = water_model.get("burning_cells", False)
    burning_only = water_on_burning and water_model.get("candidates") == "burning_only"

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

    if len(elevation) != rows or any(len(row) != cols for row in elevation):
        raise ValueError("elevation must match geographic grid dimensions")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for row in elevation for v in row):
        raise ValueError("geographic elevation must contain finite metres; gaps are not substituted")

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
    forcing = validate_forcing(scenario.get("environment"), horizon_s)
    physical_segments = forcing or [{"start_s": 0.0, "end_s": horizon_s,
                                     "midflame_wind_to_m_s": list(wind), "moisture": moisture}]
    unique_codes = {int(code) for row in fuel_grid for code in row if code != 0}
    behavior_cache: dict[int, list] = {}

    def behaviors(index: int) -> list:
        if index in behavior_cache:
            return behavior_cache[index]
        segment = physical_segments[index]
        swind = tuple(float(v) for v in segment["midflame_wind_to_m_s"])
        upwind = degrees(atan2(-swind[0], -swind[1])) % 360 if swind != (0.0, 0.0) else 0.0
        wind_speed = hypot(*swind) * 60.0
        moisture_values = tuple(float(segment["moisture"][k]) for k in MOISTURE_KEYS)
        base_cache = {}
        for code in unique_codes:
            model = fuel_models.moisturize(fuel_models.get_fuel_model(code), moisture_values)
            if model.get("burnable", True):
                base_cache[code] = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
        cells = []
        for r in range(rows):
            row_behav = []
            for c in range(cols):
                code = int(fuel_grid[r][c])
                if code == 0 or code not in base_cache:
                    row_behav.append(None)
                else:
                    gx, gy = gradients[r][c]
                    slope_tan = hypot(gx, gy)
                    aspect = degrees(atan2(-gx, -gy)) % 360 if slope_tan > 0 else 0.0
                    row_behav.append(surface_fire.calc_surface_fire_behavior_max(
                        base_cache[code], wind_speed, upwind, slope_tan, aspect))
            cells.append(row_behav)
        behavior_cache[index] = cells
        return cells

    cell_behavior = behaviors(0)
    # Directed rates are cached per forcing interval and grid edge. The entire
    # retrospective weather series is simulator state, never policy input.
    rate_cache: dict[tuple, float] = {}

    intensity_cache: dict[tuple, float] = {}

    def directional_behavior(r: int, c: int, dr: int, dc: int, index: int) -> None:
        key = (index, r, c, dr, dc)
        behavior = behaviors(index)[r][c]
        if behavior is None:
            rate_cache[key], intensity_cache[key] = 0.0, 0.0
            return
        gx, gy = gradients[r][c]
        norm = hypot(dc, dr, gx * dc + gy * dr)
        direction = (dc / norm, dr / norm, (gx * dc + gy * dr) / norm)
        res = surface_fire.calc_surface_fire_behavior_in_direction(behavior, direction)
        rate_cache[key] = max(0.001, float(res["spread_rate"]) / 60.0)
        intensity_cache[key] = float(res["fireline_intensity"])

    def spread_rate(r: int, c: int, dr: int, dc: int, index: int = 0) -> float:
        key = (index, r, c, dr, dc)
        if key not in rate_cache:
            directional_behavior(r, c, dr, dc, index)
        return rate_cache[key]

    def fireline_intensity(r: int, c: int, dr: int, dc: int, index: int = 0) -> float:
        """Fireline intensity (kW/m) of spread leaving (r, c) towards (dr, dc)."""
        key = (index, r, c, dr, dc)
        if key not in intensity_cache:
            directional_behavior(r, c, dr, dc, index)
        return intensity_cache[key]

    def segment_index(time_s: float) -> int:
        for index, segment in enumerate(physical_segments):
            if time_s < float(segment["end_s"]):
                return index
        return len(physical_segments) - 1

    residence_cache: dict[tuple[int, int], float] = {}

    def anderson_residence_s(code: int, index: int) -> float:
        """Anderson (1969) flaming residence t_r = 384 / sigma minutes, sigma in 1/ft.

        sigma is the Rothermel characteristic surface-area-to-volume ratio,
        weighted by the moisturized dead/live and size-class fractions.
        """
        key = (code, index)
        if key not in residence_cache:
            moisture_values = tuple(float(physical_segments[index]["moisture"][k]) for k in MOISTURE_KEYS)
            model = fuel_models.moisturize(fuel_models.get_fuel_model(code), moisture_values)
            sigma = sum(
                f_i * sum(f_ij * s for f_ij, s in zip(model["f_ij"][lo:hi], model["sigma"][lo:hi]))
                for f_i, (lo, hi) in zip(model["f_i"], ((0, 4), (4, 6)))
            )
            residence_cache[key] = 384.0 / sigma * 60.0 if sigma > 0 else 0.0
        return residence_cache[key]

    def burning_duration_s(r: int, c: int, ignition_s: float) -> float:
        """Time a cell stays actively burning after the front reaches it.

        Front crossing time (cell size / local head-fire rate) plus Anderson
        particle residence. Cells are tens of metres wide, so the crossing term
        dominates; residence alone (seconds) would make burning cells invisible
        at decision cadence. Uses the forcing interval active at ignition.
        """
        index = segment_index(ignition_s)
        behavior = behaviors(index)[r][c]
        if behavior is None:
            return 0.0
        head_rate_m_s = max(0.001, float(behavior["max_spread_rate"]) / 60.0)
        return cell_size / head_rate_m_s + anderson_residence_s(int(fuel_grid[r][c]), index)

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
    burnout: list[list[float | None]] = [[None] * cols for _ in range(rows)]
    drop_times: dict[tuple[int, int], list[float]] = {}  # times of the drops stacked on each treated cell
    burning_drop_times: dict[tuple[int, int], list[float]] = {}
    best = [[float("inf")] * cols for _ in range(rows)]
    queue: list[tuple[float, int, int, int, int]] = []  # (time, row, col, source row, source col)

    # Initialize queue with initial burn perimeters
    initial_burned_count = 0
    for r in range(rows):
        for c in range(cols):
            if init_mask[r][c]:
                arrival[r][c] = 0.0
                burnout[r][c] = 0.0
                best[r][c] = 0.0
                initial_burned_count += 1
                if is_burnable(r, c):
                    heappush(queue, (0.0, r, c, -1, -1))

    def linked(r: int, c: int, dr: int, dc: int) -> bool:
        """Fuel allows spread from (r, c) to its neighbour (dr, dc); diagonals need both orthogonal cells."""
        return is_burnable(r + dr, c + dc) and not (dr and dc and (not is_burnable(r + dr, c) or not is_burnable(r, c + dc)))

    def blockers(r: int, c: int, dr: int, dc: int) -> tuple[tuple[int, int], ...]:
        """Cells whose water barrier stops spread along this edge: the target and, for diagonals, both orthogonals."""
        return ((r + dr, c + dc), (r + dr, c), (r, c + dc)) if dr and dc else ((r + dr, c + dc),)

    def edge_needed_drops(r: int, c: int, dr: int, dc: int) -> float:
        """Drops needed to stop the front leaving reached cell (r, c) along this edge.

        The front's intensity is taken in the forcing interval in which (r, c)
        was reached, so whether an edge holds never depends on later forcing.
        """
        index = segment_index(arrival[r][c] or 0.0)
        return drops_needed(fireline_intensity(r, c, dr, dc, index))

    def edge_closed_at(r: int, c: int, dr: int, dc: int) -> float:
        """Time water on this edge started holding the front from (r, c); inf if it does not (D-74, D-76)."""
        stacked = [drop_times[cell] for cell in blockers(r, c, dr, dc) if cell in drop_times]
        if (r, c) in burning_drop_times:
            stacked.append(burning_drop_times[r, c])
        if not stacked:
            return float("inf")
        needed = edge_needed_drops(r, c, dr, dc)
        return min((times[int(needed) - 1] for times in stacked if len(times) >= needed), default=float("inf"))

    def edge_held(r: int, c: int, dr: int, dc: int) -> bool:
        return edge_closed_at(r, c, dr, dc) < float("inf")

    def can_spread(r: int, c: int, dr: int, dc: int) -> bool:
        return linked(r, c, dr, dc) and not edge_held(r, c, dr, dc)

    def water_delay_s(r: int, c: int, dr: int, dc: int) -> float:
        """Time a breaching front from (r, c) needs to boil off the target cell's water (D-78)."""
        times = drop_times.get((r + dr, c + dc))
        if not times:
            return 0.0
        intensity = fireline_intensity(r, c, dr, dc, segment_index(arrival[r][c] or 0.0))
        energy_kj = len(times) * capacity_l * (WATER_HEATING_KJ_KG + WATER_EVAPORATION_KJ_KG)
        return energy_kj / (max(intensity, 1e-6) * cell_size)

    def edge_time(r: int, c: int, dr: int, dc: int) -> float | None:
        """Arrival at (r + dr, c + dc) through the edge from reached cell (r, c); None if it cannot spread."""
        nr, nc = r + dr, c + dc
        cur_time = arrival[r][c]
        dist = hypot(dc * cell_size, dr * cell_size)
        if forcing:
            def speed_at(index: int) -> float:
                source = spread_rate(r, c, dr, dc, index)
                target = spread_rate(nr, nc, dr, dc, index)
                return 2.0 / (1.0/source + 1.0/target) if source > 0 and target > 0 else 0.0
            candidate_time = edge_arrival(cur_time, dist, forcing, speed_at)
        else:
            r_source = spread_rate(r, c, dr, dc)
            r_target = spread_rate(nr, nc, dr, dc)
            if r_source <= 0.0 or r_target <= 0.0:
                return None
            candidate_time = cur_time + dist/(2.0*r_source) + dist/(2.0*r_target)
        return candidate_time + water_delay_s(r, c, dr, dc)

    def relax(r: int, c: int, dr: int, dc: int) -> None:
        """Queue the arrival at (r + dr, c + dc) through the edge from reached cell (r, c)."""
        nr, nc = r + dr, c + dc
        candidate_time = edge_time(r, c, dr, dc)
        if candidate_time is None:
            return
        if candidate_time < best[nr][nc]:
            best[nr][nc] = candidate_time
            heappush(queue, (candidate_time, nr, nc, r, c))

    def advance_fire(until_s: float) -> None:
        """Advance physical Rothermel fire propagation up to until_s."""
        while queue and queue[0][0] <= until_s + 1e-9:
            cur_time, r, c, sr, sc = heappop(queue)
            if cur_time != best[r][c] or not is_burnable(r, c):
                continue
            if sr >= 0 and (edge_held(sr, sc, r - sr, c - sc) or (
                    (r, c) in drop_times
                    and (edge_time(sr, sc, r - sr, c - sc) or float("inf")) > cur_time + 1e-6)):
                # Water added after this arrival was queued holds or delays that
                # front (D-76, D-78); re-evaluate every edge into the cell.
                best[r][c] = float("inf")
                for dr, dc in NEIGHBORS:
                    pr, pc = r - dr, c - dc
                    if (0 <= pr < rows and 0 <= pc < cols and arrival[pr][pc] is not None
                            and linked(pr, pc, dr, dc) and not edge_held(pr, pc, dr, dc)):
                        relax(pr, pc, dr, dc)
                continue
            arrival[r][c] = cur_time
            if init_mask[r][c] and cur_time == 0.0:
                # Mapped initial perimeter: only cells bordering unburned
                # burnable fuel are an active front; the interior burned
                # before the episode starts.
                on_front = any(is_burnable(r + dr, c + dc) and not init_mask[r + dr][c + dc]
                               for dr, dc in NEIGHBORS)
                burnout[r][c] = burning_duration_s(r, c, 0.0) if on_front else 0.0
            else:
                burnout[r][c] = cur_time + burning_duration_s(r, c, cur_time)

            for dr, dc in NEIGHBORS:
                if can_spread(r, c, dr, dc) and arrival[r + dr][c + dc] is None:
                    relax(r, c, dr, dc)

    # Water drop execution on physical landscape
    drops_log: list[dict[str, Any]] = []
    known_burning_drops: list[list[int]] = []

    def apply_water_drop(drop_time: float, target_x_m: float, target_y_m: float, aircraft_index: int) -> list[list[int]]:
        """Add one CL-415 drop to the target cell (D-74, D-76); it may stack on earlier drops."""
        target_c = int(round(target_x_m / cell_size))
        target_r = int(round(target_y_m / cell_size))
        treated = []
        burning_drop = False
        r, c = target_r, target_c
        if is_burnable(r, c):
            reached_at = arrival[r][c]
            if reached_at is None:
                times = drop_times.setdefault((r, c), [])
                if len(times) < MAX_STACKED_DROPS:
                    times.append(drop_time)
                    treated.append([r, c])
            elif water_on_burning and reached_at <= drop_time and is_burning(r, c, drop_time):
                times = burning_drop_times.setdefault((r, c), [])
                if len(times) < MAX_STACKED_DROPS:
                    times.append(drop_time)
                    known_burning_drops.append([r, c])
                    burning_drop = True
        drops_log.append({
            "time_s": round(drop_time, 2),
            "aircraft_index": aircraft_index, "water_l": capacity_l,
            "target_local_m": [round(target_x_m, 2), round(target_y_m, 2)],
            "target_utm_m": [round(target_x_m + origin_x, 2), round(target_y_m + origin_y, 2)],
            "treated_cells_count": len(treated) + int(burning_drop),
            "treated_cells": [[r, c]] if burning_drop else treated,
        })
        if water_on_burning:
            drops_log[-1]["burning_drop"] = burning_drop
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
            # Explicit coverage is kept (scheduled mode); absent or "all" means the whole grid.
            "coverage_cells": ([[int(r), int(c)] for r, c in item["coverage_cells"]]
                               if item.get("coverage_cells") is not None and item.get("coverage") != "all"
                               else [[r, c] for r in range(rows) for c in range(cols)]),
        })
    schedule = sorted(schedule, key=lambda x: x["observed_at_s"])

    # Discrete aircraft simulation state: a fleet of identical aircraft. Each has
    # a time at which it is free again; decisions are taken for the lowest-index
    # free aircraft, and drops take effect when that aircraft reaches the target.
    # With one aircraft this reproduces the original sequential loop exactly.
    fleet_size = int(ac_config.get("fleet_size", 1))
    if fleet_size < 1:
        raise ValueError("aircraft.fleet_size must be at least 1")
    extra_starts = ac_config.get("fleet_initial_positions_utm") or []
    starts = [(init_ac_x, init_ac_y)]
    for i in range(1, fleet_size):
        if i - 1 < len(extra_starts):
            starts.append((float(extra_starts[i - 1][0]) - origin_x, float(extra_starts[i - 1][1]) - origin_y))
        else:
            starts.append((refill_x, refill_y))  # additional aircraft start at the water source
    fleet = [{"pos": [x, y], "water": capacity_l, "free_at": 0.0, "task": None} for x, y in starts]
    pending_effects: list[tuple[float, int, int, str, Any]] = []
    effect_seq = 0
    cur_sim_time = 0.0
    total_flight_dist_m = 0.0
    total_water_dropped_l = 0
    refills_count = 0
    drops_count = 0

    known_treated_cells: list[list[int]] = []
    reports: list[dict] = []
    delivered_reports: list[dict] = []
    pending_delivery: list[dict] = []
    schedule_idx = 0

    decisions_log: list[dict] = []
    aircraft_track = [{"time_s": 0.0, "position_m": list(fleet[0]["pos"])}] if fleet_size == 1 else [
        {"time_s": 0.0, "position_m": list(a["pos"]), "aircraft_index": i} for i, a in enumerate(fleet)]

    def apply_effects(until_s: float) -> None:
        """Apply scheduled drop/refill completions up to ``until_s`` in time order."""
        nonlocal refills_count, total_flight_dist_m
        pending_effects.sort(key=lambda e: (e[0], e[1]))
        while pending_effects and pending_effects[0][0] <= until_s + 1e-9:
            effect_time, _, index, kind, payload = pending_effects.pop(0)
            advance_fire(effect_time)
            craft = fleet[index]
            if kind == "drop":
                treated = apply_water_drop(effect_time, payload[0], payload[1], index)
                known_treated_cells.extend(treated)
                craft["pos"] = [payload[0], payload[1]]
                craft["water"] = max(0, craft["water"] - capacity_l)
            elif kind == "refill":
                craft["pos"] = [refill_x, refill_y]
                craft["water"] = capacity_l
                refills_count += 1
                total_flight_dist_m += payload[0]
            craft["task"] = None

    def fleet_view(time_s: float) -> list[dict]:
        """Every aircraft's state at ``time_s``; positions interpolated along transits."""
        view = []
        for index, craft in enumerate(fleet):
            task = craft["task"] if craft["free_at"] > time_s + 1e-9 else None
            position, phase, target = list(craft["pos"]), "idle", None
            if task is not None:
                phase, target = task["phase"], list(task["to"])
                span = task["arrive"] - task["t0"]
                frac = 1.0 if span <= 0 else min(max((time_s - task["t0"]) / span, 0.0), 1.0)
                position = [task["from"][k] + frac * (task["to"][k] - task["from"][k]) for k in range(2)]
            view.append({"index": index, "position_m": [round(position[0], 2), round(position[1], 2)],
                         "water_l": craft["water"], "capacity_l": capacity_l, "phase": phase, "target_m": target,
                         "busy_remaining_s": round(max(craft["free_at"] - time_s, 0.0), 2)})
        return view

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

    def front_active(r: int, c: int) -> bool:
        """True while fire can still advance from (r, c) into a neighbour (D-67).

        A neighbour the fire can spread into (burnable, not yet reached, not
        protected by a water barrier) means the front has not left this cell,
        whatever its nominal burning duration. Only the present state is used:
        whether the scheduled arrival is finite depends on future forcing and
        the horizon, so it must not decide what the controller observes.
        """
        return is_burnable(r, c) and any(
            can_spread(r, c, dr, dc) and arrival[r + dr][c + dc] is None for dr, dc in NEIGHBORS)

    def front_open(r: int, c: int) -> bool:
        """Like ``front_active`` but water plays no part: whether a drop will hold a front is the
        simulator's knowledge and must not show in what the camera sees (D-95)."""
        return is_burnable(r, c) and any(
            linked(r, c, dr, dc) and arrival[r + dr][c + dc] is None for dr, dc in NEIGHBORS)

    def is_burning(r: int, c: int, time_s: float) -> bool:
        """Whether a reached cell is still burning at the given time."""
        return (burnout[r][c] is not None and burnout[r][c] > time_s + 1e-9) or front_active(r, c)

    def direct_attack_knocks_down(r: int, c: int) -> bool:
        times = burning_drop_times.get((r, c))
        if not times:
            return False
        open_edges = [(dr, dc) for dr, dc in NEIGHBORS
                      if linked(r, c, dr, dc) and arrival[r + dr][c + dc] is None]
        return bool(open_edges) and all(len(times) >= edge_needed_drops(r, c, dr, dc) for dr, dc in open_edges)

    def thermal_reading(r: int, c: int, time_s: float) -> int:
        """Brightness a thermal camera shows on burning cell (r, c) now (D-94): the heat of the cell
        as a whole, its largest intensity in any direction, so it depends neither on where the fire
        can still go nor on water."""
        index = segment_index(time_s)
        return thermal_byte(max(fireline_intensity(r, c, dr, dc, index) for dr, dc in NEIGHBORS))

    def realtime_fire_view(time_s: float) -> dict:
        """Full-grid fire state at time_s: perfect, instantaneous observation.

        A reached cell is ``burning`` during its nominal burning duration (front
        crossing at head-fire rate plus Anderson residence) and for as long as the
        front is still advancing from it into a neighbour (D-67); otherwise
        ``burned``. Slow flank and rear fronts therefore stay visible as burning.
        ``visible_state`` ignores pre-wetting (D-95) and can show direct knockdown.
        """
        cells = []
        for r in range(rows):
            line = []
            for c in range(cols):
                reached = arrival[r][c] is not None and arrival[r][c] <= time_s + 1e-9
                if not reached:
                    state = "unburned"
                elif is_burning(r, c, time_s):
                    state = "burning"
                else:
                    state = "burned"
                visible = state
                # A camera can see direct knockdown (owner decision pending (O-20)).
                if water_on_burning and reached and direct_attack_knocks_down(r, c):
                    visible = "burned"
                elif state == "burned" and front_open(r, c):
                    visible = "burning"  # stays alight whatever the water in front of it
                line.append({
                    "fire_detected": reached,
                    "fire_state": state,
                    "visible_state": visible,
                    "thermal_byte": thermal_reading(r, c, time_s) if visible == "burning" else None,
                    "arrival_time_s": round(arrival[r][c], 6) if reached else None,
                    "observed_at_s": time_s,
                    "age_s": 0.0,
                    "source_id": "realtime",
                })
            cells.append(line)
        return {"time_s": time_s, "cells": cells}

    pending_observation = None

    # Main decision and event progression loop
    while True:
        apply_effects(cur_sim_time)
        if cur_sim_time >= horizon_s:
            break
        idle = [i for i, craft in enumerate(fleet) if craft["free_at"] <= cur_sim_time + 1e-9]
        if not idle:
            cur_sim_time = min(horizon_s, min(craft["free_at"] for craft in fleet))
            continue
        ego = idle[0]
        aircraft_pos, aircraft_water = fleet[ego]["pos"], fleet[ego]["water"]
        advance_fire(cur_sim_time)

        # Build current observation for the policy. Real-time mode exposes the
        # present state only; scheduled mode exposes delivered reports only.
        # Neither mode exposes future arrivals or future weather.
        if observation_mode == "realtime":
            fire_view = realtime_fire_view(cur_sim_time)
            weather_report = deepcopy(forcing[segment_index(cur_sim_time)]) if forcing else None
            reported_wind = weather_report["midflame_wind_to_m_s"] if weather_report else list(wind)
            weather_status = ("modeled_constant" if not weather_report else "realtime_station_interval"
                              if "station_weather" in weather_report else "realtime_synthetic_interval")
            fuel_moisture = deepcopy(physical_segments[segment_index(cur_sim_time)]["moisture"])
        else:
            sync_observations(cur_sim_time)
            fire_view = observe(delivered_reports, cur_sim_time, rows, cols)
            weather_report = delivered_weather(forcing, cur_sim_time)
            reported_wind = weather_report["midflame_wind_to_m_s"] if weather_report else ([0.0, 0.0] if forcing else list(wind))
            weather_status = "delivered_station_interval" if weather_report else ("not_yet_delivered" if forcing else "modeled_constant")
            fuel_moisture = deepcopy(weather_report["moisture"]) if weather_report else (None if forcing else deepcopy(moisture))
        current_obs = {
            "time_s": cur_sim_time,
            "fire": fire_view,
            "aircraft": {
                "position_m": [round(aircraft_pos[0], 2), round(aircraft_pos[1], 2)],
                "water_l": aircraft_water,
                "capacity_l": capacity_l,
                "phase": "idle",
                "target_m": None,
                "refill_remaining_s": 0.0,
            },
            "fuel_model": [[str(val) for val in row] for row in fuel_grid],
            "water_sources": [[refill_x, refill_y]],
            "water_source_m": [refill_x, refill_y],
            "midflame_wind_to_m_s": list(reported_wind),
            "environment_weather": weather_report,
            "weather_status": weather_status,
            "fuel_moisture": fuel_moisture,
            "observation_mode": observation_mode,
            "cell_size_m": cell_size,
            "elevation_m": deepcopy(elevation_input) if elevation_input is not None else deepcopy(scenario.get("elevation_m", [[0.0] * cols for _ in range(rows)])),
            "known_treated_cells": known_treated_cells,
            "max_stacked_drops": MAX_STACKED_DROPS,
        }
        if water_on_burning:
            current_obs["water_on_burning"] = True
            current_obs["known_burning_drops"] = known_burning_drops
            if burning_only:
                current_obs["water_on_burning_candidates"] = "burning_only"
        if fleet_size > 1:
            current_obs["aircraft_index"] = ego
            current_obs["fleet"] = fleet_view(cur_sim_time)

        burned_now = sum(1 for r in range(rows) for c in range(cols)
                         if arrival[r][c] is not None and arrival[r][c] <= cur_sim_time + 1e-9)
        order = yield current_obs, round(burned_now * cell_area_ha, 4), drops_log
        if order is None:
            if not truncated:
                raise ValueError("an order is required; only truncated episodes can stop early")
            pending_observation = current_obs
            break

        order_type = order["type"]
        craft = fleet[ego]
        decided_at = cur_sim_time

        if order_type == "HOLD":
            # Hold for one decision interval (60 s)
            craft["free_at"] = decided_at + min(60.0, horizon_s - decided_at)
            craft["task"] = {"phase": "holding", "t0": decided_at, "from": list(aircraft_pos),
                             "to": list(aircraft_pos), "arrive": craft["free_at"]}
            decisions_log.append({
                "time_s": round(craft["free_at"], 2),
                "order": order,
                "aircraft_water_l": aircraft_water,
            })
            track_position = list(aircraft_pos)

        elif order_type == "RETURN_TO_REFILL":
            # Fly to the refill waterbody, then scoop
            dist = hypot(refill_x - aircraft_pos[0], refill_y - aircraft_pos[1])
            transit_time = dist / cruise_speed
            total_time = transit_time + refill_service_s
            craft["free_at"] = decided_at + total_time
            craft["task"] = {"phase": "refill", "t0": decided_at, "from": list(aircraft_pos),
                             "to": [refill_x, refill_y], "arrive": decided_at + transit_time}
            pending_effects.append((craft["free_at"], effect_seq, ego, "refill",
                                    (dist, decided_at, transit_time)))
            effect_seq += 1
            decisions_log.append({
                "time_s": round(craft["free_at"], 2),
                "order": order,
                "refill_transit_s": round(transit_time, 2),
                "aircraft_water_l": capacity_l,
            })
            track_position = [refill_x, refill_y]
            # Arriving at the source ends the transit; service keeps the aircraft there.
            arrival_time = decided_at + transit_time
            if arrival_time <= horizon_s + 1e-9:
                aircraft_track.append({"time_s": round(arrival_time, 2),
                                       "position_m": [refill_x, refill_y],
                                       **({"aircraft_index": ego} if fleet_size > 1 else {})})

        elif order_type == "DROP":
            if aircraft_water < capacity_l:
                # The executor enforces the tank, whatever the caller filtered (one full load per drop).
                raise ValueError(f"DROP ordered for aircraft {ego} with {aircraft_water} L; a drop needs {capacity_l} L")
            target_x, target_y = order["target_m"]
            dist = hypot(target_x - aircraft_pos[0], target_y - aircraft_pos[1])
            drop_time = decided_at + dist / cruise_speed
            if drop_time <= horizon_s:
                craft["free_at"] = drop_time
                pending_effects.append((drop_time, effect_seq, ego, "drop", (target_x, target_y)))
                effect_seq += 1
                drops_count += 1
                total_water_dropped_l += capacity_l
                total_flight_dist_m += dist
                water_after = max(0, aircraft_water - capacity_l)
                track_position = [target_x, target_y]
            else:
                # Horizon reached during transit
                craft["free_at"] = drop_time
                water_after = aircraft_water
                fraction = 0.0 if dist <= 0 else min(1.0, max(0.0, (horizon_s - decided_at) / (drop_time - decided_at)))
                track_position = [aircraft_pos[0] + fraction * (target_x - aircraft_pos[0]),
                                  aircraft_pos[1] + fraction * (target_y - aircraft_pos[1])]
                total_flight_dist_m += dist * fraction
            craft["task"] = {"phase": "transit_to_drop", "t0": decided_at, "from": list(aircraft_pos),
                             "to": [target_x, target_y], "arrive": drop_time}
            decisions_log.append({
                "time_s": round(craft["free_at"], 2),
                "order": order,
                "aircraft_water_l": water_after,
            })
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        # Audit output only: the controller still receives current_obs alone.
        decisions_log[-1]["completed_at_s"] = round(craft["free_at"], 2)
        decisions_log[-1]["decision_time_s"] = current_obs["time_s"]
        if fleet_size > 1:
            decisions_log[-1]["aircraft_index"] = ego
        if record_observations:
            decisions_log[-1]["observation"] = deepcopy(current_obs)
        track_time = craft["free_at"]
        if track_time > horizon_s:
            track_time = horizon_s
            if order_type == "RETURN_TO_REFILL":
                elapsed = max(0.0, horizon_s - decided_at)
                fraction = 1.0 if transit_time <= 0 else min(1.0, elapsed / transit_time)
                track_position = [aircraft_pos[0] + fraction * (refill_x - aircraft_pos[0]),
                                  aircraft_pos[1] + fraction * (refill_y - aircraft_pos[1])]
        track_entry = {"time_s": round(track_time, 2), "position_m": track_position}
        if fleet_size > 1:
            track_entry["aircraft_index"] = ego
        aircraft_track.append(track_entry)

    if not truncated or pending_observation is None:
        # The episode reached the horizon (truncated or not): score it there.
        # Fire must be advanced explicitly; the last decision may precede the
        # horizon by a HOLD interval or a transit, and only pending effects
        # advance it inside the loop.
        advance_fire(horizon_s)
        if observation_mode == "scheduled":
            sync_observations(horizon_s)
        eval_time = horizon_s
    else:
        eval_time = cur_sim_time

    # Unfinished refill transit contributes only the outbound distance flown by eval_time.
    for completion_time, _, _, kind, payload in pending_effects:
        if kind == "refill":
            dist, decided_at, transit_time = payload
            fraction = 1.0 if transit_time <= 0 else min(1.0, max(0.0, (eval_time - decided_at) / transit_time))
            total_flight_dist_m += dist * fraction

    def front_end_times(end_s: float) -> list[list[float | None]]:
        """Audit raster: when each reached cell stopped burning (D-67).

        The later of its nominal burnout and, for each neighbour the fire could
        still spread into when this cell was reached, the time that edge closed
        (the neighbour ignited, or a barrier blocked it); ``end_s`` when an edge
        was still open at the end. Matches ``front_active`` at every decision.
        """
        out: list[list[float | None]] = []
        for r in range(rows):
            line: list[float | None] = []
            for c in range(cols):
                if arrival[r][c] is None:
                    line.append(None)
                    continue
                end = burnout[r][c] if burnout[r][c] is not None else arrival[r][c]
                if not is_burnable(r, c):
                    line.append(round(end, 6))
                    continue
                for dr, dc in NEIGHBORS:
                    if not linked(r, c, dr, dc):
                        continue
                    nr, nc = r + dr, c + dc
                    closed = min([arrival[nr][nc] if arrival[nr][nc] is not None else float("inf")]
                                 + [edge_closed_at(r, c, dr, dc)])
                    if closed > arrival[r][c]:
                        end = max(end, min(closed, end_s))
                line.append(round(end, 6))
            out.append(line)
        return out

    # Compute metrics
    burned_cells_final = sum(1 for r in range(rows) for c in range(cols) if arrival[r][c] is not None and arrival[r][c] <= eval_time + 1e-9)
    final_burned_ha = round(burned_cells_final * cell_area_ha, 4)
    initial_burned_ha = round(initial_burned_count * cell_area_ha, 4)

    result = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "scenario_id": scenario.get("scenario_id", "geographic-scenario"),
        "policy": policy,
        "policy_seed": policy_seed,
        "seed": scenario.get("seed"),
        "scenario_hash": scenario.get("scenario_hash"),
        "time_origin_utc": scenario.get("provenance", {}).get("image_acquired_at_utc"),
        "environment": deepcopy(scenario.get("environment")),
        "units": {"time": "s", "distance": "m", "water": "L", "area": "ha", "elevation": "m"},
        "crs": deepcopy(scenario.get("crs", "LOCAL_METERS")),
        "origin_utm": [origin_x, origin_y],
        "provenance": deepcopy(scenario.get("provenance", {})),
        "observation_mode": observation_mode,
        "arrival_time_s": [[round(v, 6) if v is not None else None for v in line] for line in arrival],
        "burnout_time_s": front_end_times(eval_time),
        "burning_model": "burning while within front crossing (cell_size / head-fire rate) + Anderson 1969 residence 384/sigma min, or while it borders a neighbour the fire can still spread into (burnable, unreached, no barrier; D-67, D-69); initial perimeter: cells bordering unburned burnable fuel ignited at t=0, interior cells already burned",
        "fuel_grid": deepcopy(fuel_grid),
        "water_mask": deepcopy(fire_dict.get("water_mask", [[False] * cols for _ in range(rows)])),
        "elevation_m": deepcopy(elevation),
        "aircraft_track": aircraft_track,
        "aircraft_initial_state": [{"index": i, "position_m": list(pos), "water_l": capacity_l,
                                    "capacity_l": capacity_l} for i, pos in enumerate(starts)],
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
        "affected_area_ha": final_burned_ha,
        "aircraft_metrics": {
            "drops_count": drops_count,
            "refills_count": refills_count,
            "total_water_dropped_l": total_water_dropped_l,
            "total_flight_distance_km": round(total_flight_dist_m / 1000.0, 3),
            "rejected_orders_count": 0,
            "rejected_orders": 0,
        },
        "decisions": decisions_log,
        "drops": drops_log,
    }
    if truncated:
        result["pending_observation"] = pending_observation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Path to geographic scenario JSON")
    parser.add_argument("--policy", choices=POLICIES, default="heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenario_data = json.loads(args.scenario.read_text())
    result = run_geographic_integrated(scenario_data, policy=args.policy, policy_seed=args.seed,
                                       observation_mode=args.observation_mode)

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
