"""Observation-only candidate generation and policies for the synthetic loop."""

from __future__ import annotations

import random
from math import hypot
from typing import Any


def _is_burnable_fuel(val: Any) -> bool:
    if isinstance(val, (int, float)):
        return val > 0
    return str(val).upper() not in ("NONE", "0", "")


# Cells wetted by one drop: the target and its 4-neighbours (same footprint as the
# geographic executor). Two such footprints overlap iff their targets are within
# Manhattan distance 2.
_DROP_FOOTPRINT = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_OVERLAP_RADIUS = 2
# A geographic cell may receive up to this many stacked drops (D-76; McFayden et al.
# 2023 A8.5: three CL-415 drops hold open-fuel line up to 3,200 kW/m). Observations
# announce their limit as ``max_stacked_drops``; without it a treated cell is never
# targeted again, as in the synthetic integrated episode.
MAX_STACKED_DROPS = 3


def stack_limit(observation: dict) -> int:
    return int(observation.get("max_stacked_drops", 1))


def full_cells(observation: dict) -> set[tuple[int, int]]:
    """Treated cells that cannot take another drop."""
    limit = stack_limit(observation)
    return {cell for cell, n in treated_counts(observation).items() if n >= limit}


def treated_counts(observation: dict) -> dict[tuple[int, int], int]:
    """Drops received per cell; ``known_treated_cells`` lists a cell once per drop."""
    counts: dict[tuple[int, int], int] = {}
    for cell in observation.get("known_treated_cells", []):
        key = (int(cell[0]), int(cell[1]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def claimed_cells(observation: dict) -> set[tuple[int, int]]:
    """Drop targets forbidden to the ego aircraft (D-63).

    A target is forbidden when its footprint would overlap the footprint of
    another aircraft already flying to a drop (Manhattan distance <= 2 between
    targets). Read from the observed ``fleet`` only; empty with one aircraft.
    """
    fleet = observation.get("fleet")
    if not fleet:
        return set()
    ego = observation.get("aircraft_index", 0)
    cell_size = float(observation["cell_size_m"])
    claimed = set()
    for other in fleet:
        if other["index"] == ego or other.get("phase") != "transit_to_drop" or other.get("target_m") is None:
            continue
        r = int(round(other["target_m"][1] / cell_size))
        c = int(round(other["target_m"][0] / cell_size))
        claimed.update((r + dr, c + dc)
                       for dr in range(-_OVERLAP_RADIUS, _OVERLAP_RADIUS + 1)
                       for dc in range(-_OVERLAP_RADIUS, _OVERLAP_RADIUS + 1)
                       if abs(dr) + abs(dc) <= _OVERLAP_RADIUS)
    return claimed


def candidate_cells(observation: dict) -> list[list[int]]:
    """Return observed unburned burnable cells adjacent to an observed active fire.

    Cells holding the observation's ``max_stacked_drops`` are excluded (one drop
    when absent); a cell with fewer can be targeted again to stack water (D-76).

    Only delivered cell records in ``observation['fire']['cells']`` are read;
    unknown cells never count as either burning or unburned. With several
    aircraft, targets whose drop footprint would overlap another aircraft's
    pending drop are excluded (D-63), for every policy and action mapper.
    """
    fire = observation["fire"]
    cells = fire["cells"]
    fuel = observation["fuel_model"]
    if cells is None:
        return []
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    if len(fuel) != rows or any(len(row) != cols for row in cells) or any(len(row) != cols for row in fuel):
        raise ValueError("fire.cells and fuel_model must have matching rectangular shapes")
    burning = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if isinstance(cells[r][c], dict) and cells[r][c].get("fire_detected") is True
    }
    full = full_cells(observation)
    forbidden = claimed_cells(observation)
    candidates = set()
    for r, c in burning:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            record = cells[nr][nc]
            if (nr, nc) not in full and (nr, nc) not in forbidden and _is_burnable_fuel(fuel[nr][nc]) and isinstance(record, dict) and record.get("fire_detected") is False:
                candidates.add((nr, nc))
    return [[r, c] for r, c in sorted(candidates)]


def choose_order(
    policy: str,
    observation: dict,
    candidates: list[list[int]] | None = None,
    rng: random.Random | None = None,
    behavior: Any = None,
) -> dict:
    """Choose HOLD, RETURN_TO_REFILL, or a DROP order from current observation.

    ``policy`` is one of ``no_intervention``, ``random_valid``, ``heuristic``,
    ``containment_heuristic``, or ``hold_aware_heuristic``. The non-random
    policies use only observed cells and static fuel/aircraft data.
    ``hold_aware_heuristic`` takes an ``ObservedFireBehavior`` as ``behavior``
    (created here when omitted).

    With several aircraft, overlapping drops are excluded upstream by
    ``candidate_cells``; with no admissible target the aircraft holds.
    """
    if policy not in {"no_intervention", "random_valid", "heuristic", "containment_heuristic", "hold_aware_heuristic"}:
        raise ValueError("policy must be no_intervention, random_valid, heuristic, containment_heuristic, or hold_aware_heuristic")
    if policy == "no_intervention":
        return {"type": "HOLD"}
    aircraft = observation["aircraft"]
    if aircraft["phase"] != "idle":
        return {"type": "HOLD"}
    if aircraft["water_l"] <= 0:
        return {"type": "RETURN_TO_REFILL"}
    candidates = candidate_cells(observation) if candidates is None else candidates
    if not candidates:
        return {"type": "HOLD"}
    if policy == "random_valid":
        if rng is None:
            raise ValueError("random_valid requires a random.Random instance")
        target = rng.choice(candidates)
    elif policy == "containment_heuristic":
        target = _containment_target(observation, candidates)
    elif policy == "hold_aware_heuristic":
        if observation.get("fuel_moisture") is None:
            # Delayed observation before the first weather report: no fire behaviour yet.
            target = _heuristic_target(observation, candidates)
        else:
            if behavior is None:
                from .action_features import ObservedFireBehavior
                behavior = ObservedFireBehavior()
            target = _hold_aware_target(observation, candidates, behavior)
            if target is None:
                return {"type": "HOLD"}
    else:
        target = _heuristic_target(observation, candidates)
    cell_size = float(observation["cell_size_m"])
    return {"type": "DROP", "target_m": [target[1] * cell_size, target[0] * cell_size]}


def _downwind_rank(observation: dict, candidate: list[int]) -> tuple[float, float, int, int]:
    """Sort key of a candidate: maximal downwind projection, then proximity, then coordinate order."""
    cells = observation["fire"]["cells"]
    east, north = observation["midflame_wind_to_m_s"]
    position_x, position_y = observation["aircraft"]["position_m"]
    cell_size = float(observation["cell_size_m"])
    r, c = candidate
    sources = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(cells) and 0 <= nc < len(cells[0]):
            record = cells[nr][nc]
            if isinstance(record, dict) and record.get("fire_detected") is True:
                # Geographic displacement from burning source to target.
                # Neighbor offsets point from target back to source; invert
                # them to project source-to-target displacement.
                sources.append((-dc * cell_size * east - dr * cell_size * north) / cell_size)
    downwind = max(sources) if sources else float("-inf")
    x, y = c * cell_size, r * cell_size
    distance = hypot(x - position_x, y - position_y)
    return (-downwind, distance, r, c)


def _heuristic_target(observation: dict, candidates: list[list[int]]) -> list[int]:
    """Maximize downwind projection, then proximity, then coordinate order."""
    return min(candidates, key=lambda candidate: _downwind_rank(observation, candidate))


def _hold_aware_target(observation: dict, candidates: list[list[int]], behavior: Any) -> list[int] | None:
    """Downwind target with two corrections from the stacked drops a front needs.

    Candidates whose treated water already holds the front are dropped, and
    the rest are tiered: fronts that stacked drops can hold, then fronts no
    water holds that have none yet, then those that already have some. Ties
    follow the downwind rank. The hold class is the D-74/D-76 limits applied to
    Rothermel fireline intensity from observed fuel, terrain, wind and moisture.
    """
    counts = treated_counts(observation)
    ranked = []
    for r, c in candidates:
        need = behavior.needed_at(observation, r, c)
        have = counts.get((r, c), 0)
        holdable = need != float("inf")
        if holdable and have >= need:
            continue  # the water already there holds this front
        tier = 0 if holdable else (1 if have == 0 else 2)
        ranked.append((tier,) + _downwind_rank(observation, [r, c]))
    if not ranked:
        return None
    best = min(ranked)
    return [best[3], best[4]]


def _containment_target(observation: dict, candidates: list[list[int]]) -> list[int]:
    """Prioritize downwind head-attack, downstream fuel protection, bottleneck containment, and cycle efficiency."""
    cells = observation["fire"]["cells"]
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    fuel = observation["fuel_model"]
    treated = {tuple(c) for c in observation.get("known_treated_cells", [])}
    east, north = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
    position_x, position_y = observation["aircraft"]["position_m"]
    water_pos = observation.get("water_source_m", [position_x, position_y])
    cell_size = float(observation["cell_size_m"])

    burning = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if isinstance(cells[r][c], dict) and cells[r][c].get("fire_detected") is True
    }

    step_r = 1 if north > 0.5 else (-1 if north < -0.5 else 0)
    step_c = 1 if east > 0.5 else (-1 if east < -0.5 else 0)

    def rank(candidate: list[int]) -> tuple[float, float, int, int]:
        r, c = candidate
        sources = []
        burning_neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                record = cells[nr][nc]
                if isinstance(record, dict) and record.get("fire_detected") is True:
                    burning_neighbors += 1
                    sources.append((-dc * cell_size * east - dr * cell_size * north) / cell_size)
        downwind = max(sources) if sources else float("-inf")

        downstream_fuel = 0
        if step_r != 0 or step_c != 0:
            for step in range(1, 5):
                nr, nc = r + step * step_r, c + step * step_c
                if 0 <= nr < rows and 0 <= nc < cols:
                    if _is_burnable_fuel(fuel[nr][nc]) and (nr, nc) not in burning and (nr, nc) not in treated:
                        downstream_fuel += 1
                    elif not _is_burnable_fuel(fuel[nr][nc]):
                        break

        x, y = c * cell_size, r * cell_size
        d_ac = hypot(x - position_x, y - position_y)
        d_water = hypot(x - water_pos[0], y - water_pos[1])
        cycle_dist = d_ac + 0.4 * d_water

        score = -downwind - 0.20 * downstream_fuel - 0.20 * (burning_neighbors - 1)
        return (score, cycle_dist, r, c)

    return min(candidates, key=rank)



