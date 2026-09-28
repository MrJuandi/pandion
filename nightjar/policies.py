"""Observation-only candidate generation and policies for the synthetic loop."""

from __future__ import annotations

import random
from math import hypot


def candidate_cells(observation: dict) -> list[list[int]]:
    """Return observed unburned fuel cells adjacent to an observed active fire.

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
            if (nr, nc) not in treated and fuel[nr][nc] in ("FM1", "CURED_GRASS") and isinstance(record, dict) and record.get("fire_detected") is False:
                candidates.add((nr, nc))
    return [[r, c] for r, c in sorted(candidates)]


def choose_order(
    policy: str,
    observation: dict,
    candidates: list[list[int]] | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Choose HOLD, RETURN_TO_REFILL, or a DROP order from current observation.

    ``policy`` is one of ``no_intervention``, ``random_valid``, or
    ``heuristic``. The latter two use only observed cells and static fuel data.
    """
    if policy not in {"no_intervention", "random_valid", "heuristic"}:
        raise ValueError("policy must be no_intervention, random_valid, or heuristic")
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
