"""Per-action inputs for the parametric head, from the controller observation only (D-82)."""

from __future__ import annotations

from math import atan2, degrees, hypot, isfinite
from typing import Any

import numpy as np
from pyretechnics import fuel_models, surface_fire

from .environment import MOISTURE_KEYS
from .geographic_integrated import NEIGHBORS, drops_needed
from .parametric_actions import ParametricActionMapper
from .policies import stack_limit, treated_counts

ACTION_FEATURE_NAMES = ("valid", "stacked", "need_one", "need_three", "breach", "remaining",
                        "sufficient", "reached_neighbours", "aircraft_distance", "water_distance",
                        "downwind", "offset")
NUM_ACTION_FEATURES = len(ACTION_FEATURE_NAMES)
WATER_DISTANCE_SCALE_M = 20000.0


class ObservedFireBehavior:
    """Directional fireline intensity (kW/m) from observed fuel, terrain, wind and moisture.

    Mirrors the simulator's Rothermel/Pyretechnics calls in ``geographic_integrated``
    but reads only observation fields. Gradients are computed once per instance;
    behaviour and intensity are cached per (wind, moisture) pair.
    """

    def __init__(self) -> None:
        self._gradients: list[list[tuple[float, float]]] | None = None
        self._cache: dict[tuple, dict[str, Any]] = {}

    def _grid_gradients(self, observation: dict) -> list[list[tuple[float, float]]]:
        if self._gradients is None:
            elevation = observation["elevation_m"]
            rows, cols = len(elevation), len(elevation[0])
            size = float(observation["cell_size_m"])
            grads = []
            for r in range(rows):
                line = []
                for c in range(cols):
                    west, east = max(c - 1, 0), min(c + 1, cols - 1)
                    south, north = max(r - 1, 0), min(r + 1, rows - 1)
                    gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * size) if east > west else 0.0
                    gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * size) if north > south else 0.0
                    line.append((gx, gy))
                grads.append(line)
            self._gradients = grads
        return self._gradients

    def _state(self, observation: dict) -> dict[str, Any]:
        moisture = observation.get("fuel_moisture")
        if moisture is None:
            raise ValueError("action features need fuel_moisture in the observation")
        wind = tuple(float(v) for v in observation["midflame_wind_to_m_s"])
        moisture_values = tuple(float(moisture[k]) for k in MOISTURE_KEYS)
        key = (wind, moisture_values)
        if key not in self._cache:
            upwind = degrees(atan2(-wind[0], -wind[1])) % 360 if wind != (0.0, 0.0) else 0.0
            self._cache[key] = {"upwind": upwind, "wind_speed": hypot(*wind) * 60.0,
                                "moisture": moisture_values, "base": {}, "cells": {}, "intensity": {}}
        return self._cache[key]

    def _cell_behavior(self, observation: dict, state: dict[str, Any], r: int, c: int) -> Any:
        """Max-spread behaviour of cell (r, c), or None when it is nonburnable."""
        if (r, c) not in state["cells"]:
            code = int(observation["fuel_model"][r][c])
            behavior = None
            if code != 0:
                if code not in state["base"]:
                    model = fuel_models.moisturize(fuel_models.get_fuel_model(code), state["moisture"])
                    state["base"][code] = (surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
                                           if model.get("burnable", True) else None)
                base = state["base"][code]
                if base is not None:
                    gx, gy = self._grid_gradients(observation)[r][c]
                    slope_tan = hypot(gx, gy)
                    aspect = degrees(atan2(-gx, -gy)) % 360 if slope_tan > 0 else 0.0
                    behavior = surface_fire.calc_surface_fire_behavior_max(
                        base, state["wind_speed"], state["upwind"], slope_tan, aspect)
            state["cells"][(r, c)] = behavior
        return state["cells"][(r, c)]

    def intensity(self, observation: dict, r: int, c: int, dr: int, dc: int) -> float:
        """Fireline intensity (kW/m) of spread leaving cell (r, c) towards offset (dr, dc)."""
        state = self._state(observation)
        key = (r, c, dr, dc)
        if key not in state["intensity"]:
            behavior = self._cell_behavior(observation, state, r, c)
            if behavior is None:
                value = 0.0
            else:
                gx, gy = self._grid_gradients(observation)[r][c]
                norm = hypot(dc, dr, gx * dc + gy * dr)
                direction = (dc / norm, dr / norm, (gx * dc + gy * dr) / norm)
                value = float(surface_fire.calc_surface_fire_behavior_in_direction(behavior, direction)["fireline_intensity"])
            state["intensity"][key] = value
        return state["intensity"][key]

    def needed_at(self, observation: dict, r: int, c: int) -> float:
        """Stacked drops needed to hold the fire now bordering unburned cell (r, c); 0 if no neighbour burns."""
        cells = observation["fire"]["cells"]
        rows, cols = len(cells), len(cells[0])
        best, reached = 0.0, False
        for dr, dc in NEIGHBORS:
            sr, sc = r - dr, c - dc
            if 0 <= sr < rows and 0 <= sc < cols:
                cell = cells[sr][sc]
                if isinstance(cell, dict) and cell.get("fire_detected") is True:
                    best = max(best, self.intensity(observation, sr, sc, dr, dc))
                    reached = True
        return drops_needed(best) if reached else 0.0

    def needed_from(self, observation: dict, sr: int, sc: int, dr: int, dc: int) -> float:
        """Stacked drops needed to hold spread leaving (sr, sc) towards (dr, dc)."""
        return drops_needed(self.intensity(observation, sr, sc, dr, dc))


def action_feature_array(observation: dict, mapper: ParametricActionMapper,
                         behavior: ObservedFireBehavior) -> np.ndarray:
    """float32 [mapper.num_actions - 2, 1 + NUM_ACTION_FEATURES]; row i describes action i + 2.

    Column 0 is the flat target cell index; the rest follow ``ACTION_FEATURE_NAMES``.
    Rows of invalid actions are zero, matching ``mapper.compute_mask``.
    """
    out = np.zeros((mapper.num_actions - 2, 1 + NUM_ACTION_FEATURES), dtype=np.float32)
    aircraft = observation["aircraft"]
    if aircraft["phase"] != "idle" or aircraft["water_l"] <= 0:
        return out
    cells = observation["fire"]["cells"]
    rows, cols = len(cells), len(cells[0])
    size = mapper.cell_size_m
    reached = {(r, c) for r, line in enumerate(cells) for c, cell in enumerate(line)
               if isinstance(cell, dict) and cell.get("fire_detected") is True}
    counts = treated_counts(observation)
    limit = stack_limit(observation)
    ax, ay = (float(v) for v in aircraft["position_m"])
    wx, wy = (float(v) for v in observation.get("water_source_m", [ax, ay]))
    diagonal = hypot(rows * size, cols * size)
    east, north = (float(v) for v in observation.get("midflame_wind_to_m_s", [0.0, 0.0]))
    wind_len = hypot(east, north)
    for index, (r, c, orow, ocol, d) in mapper.target_details(observation).items():
        row = out[index - 2]
        have = counts.get((r, c), 0)
        if d == 0:
            need = behavior.needed_at(observation, r, c)
        else:
            need = behavior.needed_from(observation, r - orow, c - ocol, orow, ocol)
        neighbours = sum(1 for dr, dc in NEIGHBORS if (r + dr, c + dc) in reached)
        row[0] = r * cols + c
        row[1] = 1.0  # valid
        row[2] = have / limit
        row[3] = 1.0 if need == 1 else 0.0
        row[4] = 1.0 if need == 3 else 0.0
        row[5] = 1.0 if need == float("inf") else 0.0
        row[6] = 1.0 if need == float("inf") else min(max(need - have, 0.0), 3.0) / 3.0
        row[7] = 1.0 if (isfinite(need) and need > 0 and have >= need) else 0.0
        row[8] = neighbours / 8.0
        row[9] = min(hypot(c * size - ax, r * size - ay) / diagonal, 1.5)
        row[10] = min(hypot(c * size - wx, r * size - wy) / WATER_DISTANCE_SCALE_M, 1.5)
        row[11] = (ocol * east + orow * north) / wind_len if wind_len > 1e-6 else 0.0
        row[12] = d / 2.0
    return out
