"""Observation-only candidate generation and policies for the synthetic loop."""

from __future__ import annotations

import random
from math import hypot


def candidate_cells(observation: dict) -> list[list[int]]:
    """Return observed unburned FM1 cells adjacent to an observed active fire.

    Only delivered cell records in ``observation['fire']['cells']`` are read;
    unknown cells never count as either burning or unburned.
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
    treated = {tuple(cell) for cell in observation.get("known_treated_cells", [])}
    candidates = set()
    for r, c in burning:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            record = cells[nr][nc]
            if (nr, nc) not in treated and fuel[nr][nc] == "FM1" and isinstance(record, dict) and record.get("fire_detected") is False:
                candidates.add((nr, nc))
    return [[r, c] for r, c in sorted(candidates)]


def choose_order(
    policy: str,
    observation: dict,
    candidates: list[list[int]] | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Choose HOLD, RETURN_TO_REFILL, or a DROP order from current observation.

    ``policy`` is one of ``no_intervention``, ``random_valid``,
    ``heuristic``, or ``containment_heuristic``. The latter three use only
    observed cells and static fuel/aircraft data.
    """
    if policy not in {"no_intervention", "random_valid", "heuristic", "containment_heuristic"}:
        raise ValueError("policy must be no_intervention, random_valid, heuristic, or containment_heuristic")
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
    else:
        target = _heuristic_target(observation, candidates)
    cell_size = float(observation["cell_size_m"])
    return {"type": "DROP", "target_m": [target[1] * cell_size, target[0] * cell_size]}


def _heuristic_target(observation: dict, candidates: list[list[int]]) -> list[int]:
    """Maximize downwind projection, then proximity, then coordinate order."""
    cells = observation["fire"]["cells"]
    east, north = observation["midflame_wind_to_m_s"]
    position_x, position_y = observation["aircraft"]["position_m"]
    cell_size = float(observation["cell_size_m"])

    def rank(candidate: list[int]) -> tuple[float, float, int, int]:
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

    return min(candidates, key=rank)


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
                    if fuel[nr][nc] == "FM1" and (nr, nc) not in burning and (nr, nc) not in treated:
                        downstream_fuel += 1
                    elif fuel[nr][nc] == "NONE":
                        break

        x, y = c * cell_size, r * cell_size
        d_ac = hypot(x - position_x, y - position_y)
        d_water = hypot(x - water_pos[0], y - water_pos[1])
        cycle_dist = d_ac + 0.4 * d_water

        score = -downwind - 0.20 * downstream_fuel - 0.20 * (burning_neighbors - 1)
        return (score, cycle_dist, r, c)

    return min(candidates, key=rank)



