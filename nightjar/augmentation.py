"""Geometric training augmentation for geographic scenarios (D-65).

With six training maps the image policy learned map-specific habits: it beat
the heuristic on training maps but not on the held-out map. The fire physics
here has no preferred orientation, so a scenario rotated by a multiple of 90°
and/or mirrored, with its wind, terrain, fuel, imagery and aircraft positions
transformed consistently, is another valid training scenario.

Transforms are the eight symmetries of the square (dihedral group D4), indexed
0..7 as ``t = 4 * flip + quarter_turns``: first an optional east-west mirror,
then ``quarter_turns`` counter-clockwise quarter turns about the grid centre.
Odd quarter turns need a square grid. Augmented scenarios are training data
only: their UTM coordinates are virtual and ``provenance.augmentation`` says so.
"""

from __future__ import annotations

from copy import deepcopy
from math import atan2, cos, degrees, radians, sin
from typing import Any

import numpy as np

NUM_TRANSFORMS = 8


def _parts(t: int) -> tuple[bool, int]:
    if not 0 <= t < NUM_TRANSFORMS:
        raise ValueError("transform index must be in 0..7")
    return t >= 4, t % 4


def transform_vector(vector: tuple[float, float], t: int) -> tuple[float, float]:
    """Apply the linear part (mirror, then quarter turns) to an (east, north) vector."""
    flip, turns = _parts(t)
    x, y = float(vector[0]), float(vector[1])
    if flip:
        x = -x
    for _ in range(turns):
        x, y = -y, x
    return x, y


def transform_point(x: float, y: float, t: int, width_m: float, height_m: float) -> tuple[float, float]:
    """Transform a local (east, north) point about the centre of [0, width_m] x [0, height_m].

    The simulator places cell (r, c) at (c, r) * cell_size and maps points to
    cells by rounding, so the grid centre is ((cols - 1) / 2, (rows - 1) / 2) *
    cell_size: pass width_m = (cols - 1) * cell_size, height_m = (rows - 1) * cell_size.
    """
    cx, cy = width_m / 2.0, height_m / 2.0
    dx, dy = transform_vector((x - cx, y - cy), t)
    return cx + dx, cy + dy


def transform_grid(values: Any, t: int) -> np.ndarray:
    """Transform a [rows][cols] array stored row 0 = south, column 0 = west."""
    array = np.asarray(values)
    flip, turns = _parts(t)
    if turns % 2 and array.shape[0] != array.shape[1]:
        raise ValueError("odd quarter turns need a square grid")
    # Arrays are indexed [north, east]; mirroring east-west reverses columns and a
    # counter-clockwise turn in the (east, north) plane is np.rot90 with axes (1, 0).
    if flip:
        array = array[:, ::-1]
    return np.ascontiguousarray(np.rot90(array, k=turns, axes=(1, 0)))


def _wind_from_deg(degrees_from: float, t: int) -> float:
    """Transform a meteorological 'from' direction (clockwise from north)."""
    theta = radians(degrees_from)
    to_east, to_north = -sin(theta), -cos(theta)
    e, n = transform_vector((to_east, to_north), t)
    return round(degrees(atan2(-e, -n)) % 360.0, 6)


def augment_scenario(scenario: dict[str, Any], t: int) -> dict[str, Any]:
    """Return scenario ``t`` of the D4 family; ``t == 0`` returns the input unchanged."""
    if t == 0:
        return scenario
    out = deepcopy(scenario)
    grid = out["grid"]
    rows, cols, size = int(grid["rows"]), int(grid["cols"]), float(grid["cell_size_m"])
    # Span of cell centres (cell (r, c) sits at (c, r) * size), not the raster extent.
    width, height = (cols - 1) * size, (rows - 1) * size
    x0 = float(grid.get("origin_utm_easting_m", 0.0))
    y0 = float(grid.get("origin_utm_northing_m", 0.0))
    fire = out["fire"]
    for key in ("fuel_grid", "water_mask", "burn_mask", "elevation_m"):
        if key in fire:
            fire[key] = transform_grid(fire[key], t).tolist()
    fire["wind_to_m_s"] = list(transform_vector(fire["wind_to_m_s"], t))
    environment = out.get("environment") or {}
    for segment in environment.get("forcing", []) or []:
        segment["midflame_wind_to_m_s"] = list(transform_vector(segment["midflame_wind_to_m_s"], t))
        station = segment.get("station_weather")
        if station and station.get("wind_from_deg") is not None:
            station["wind_from_deg"] = _wind_from_deg(float(station["wind_from_deg"]), t)

    def utm(point: list[float]) -> list[float]:
        x, y = transform_point(float(point[0]) - x0, float(point[1]) - y0, t, width, height)
        return [round(x + x0, 3), round(y + y0, 3)]

    aircraft = out.get("aircraft", {})
    for key in ("initial_position_utm", "water_scooping_utm"):
        if key in aircraft:
            aircraft[key] = utm(aircraft[key])
    if aircraft.get("fleet_initial_positions_utm"):
        aircraft["fleet_initial_positions_utm"] = [utm(p) for p in aircraft["fleet_initial_positions_utm"]]
    for waypoint in aircraft.get("water_waypoints", []) or []:
        waypoint["utm_easting_m"], waypoint["utm_northing_m"] = utm([waypoint["utm_easting_m"],
                                                                     waypoint["utm_northing_m"]])
        waypoint["lon_deg"] = waypoint["lat_deg"] = None  # no longer a real location

    out["scenario_id"] = f"{scenario['scenario_id']}-aug{t}"
    out["augmentation"] = {"transform": t}
    out.setdefault("provenance", {})["augmentation"] = {
        "transform": t, "group": "D4 (mirror east-west, then quarter turns counter-clockwise)",
        "source_scenario_hash": scenario.get("scenario_hash"),
        "note": "training-only transformed copy; UTM coordinates are virtual",
    }
    return out


def augment_image(image: np.ndarray, t: int) -> np.ndarray:
    """Transform a [H, W, C] grid-oriented image (row 0 south) like the scenario grids."""
    return transform_grid(image, t)
