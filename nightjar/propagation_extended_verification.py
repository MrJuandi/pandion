"""Frozen numerical benchmark for wind, slope, and planar fuel transitions.

This module measures numerical arrival error against independently computed
local fire-behavior ellipses. It does not establish physical validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
import json
from math import isfinite, sin, cos, radians, sqrt
import os
from pathlib import Path
import platform

from . import fire_standard
from .fire_standard import run_fire_standard

SCHEMA_VERSION = "nightjar-propagation-extended-0.1"
CELLS_M = (4.0, 2.0, 1.0)
TARGETS = ((40.0, 0.0), (40.0, 20.0), (40.0, 40.0), (80.0, 0.0))
HETERO_TARGETS = ((20.0, 0.0), (60.0, 0.0), (80.0, 0.0))
SEED = 0
MOISTURE = {"dead_1h": 0.04, "dead_10h": 0.05, "dead_100h": 0.07,
            "dead_herbaceous": 0.04, "live_herbaceous": 0.30, "live_woody": 0.60}
TOL = 0.05
DOMAIN_TOL = 0.01
HORIZON_MULTIPLIER = 1.25
PINNED_ENGINE = "2026.8.10"
PROTOCOL = Path(__file__).resolve().parents[1] / "docs" / "PROPAGATION_EXTENDED_PROTOCOL.md"
CASES = (
    {"id": "calm", "wind_u10_kmh": 0.0, "upwind_deg": 0.0, "slope": 0.0, "aspect_deg": 0.0},
    {"id": "wind-east", "wind_u10_kmh": 12.0, "upwind_deg": 270.0, "slope": 0.0, "aspect_deg": 0.0},
    {"id": "wind-northeast", "wind_u10_kmh": 12.0, "upwind_deg": 225.0, "slope": 0.0, "aspect_deg": 0.0},
    {"id": "slope-east", "wind_u10_kmh": 0.0, "upwind_deg": 0.0, "slope": 0.5, "aspect_deg": 270.0},
    {"id": "combined-crosswind", "wind_u10_kmh": 12.0, "upwind_deg": 180.0, "slope": 0.5, "aspect_deg": 270.0},
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _passes(value: float | None, threshold: float = TOL) -> bool:
    return value is not None and isfinite(value) and value < threshold


def _source_paths() -> dict[str, Path]:
    names = ("burn_cells", "surface_fire", "eulerian_level_set", "conversion", "fuel_models", "space_time_cube")
    paths = {}
    for name in names:
        imported_path = Path(import_module(f"pyretechnics.{name}").__file__)
        source_path = imported_path.parent / f"{name}.py"
        paths[f"pyretechnics/{imported_path.name}"] = imported_path
        if source_path.exists():
            paths[f"pyretechnics/{source_path.name}"] = source_path
    paths["nightjar/fire_standard.py"] = Path(fire_standard.__file__)
    paths["nightjar/propagation_extended_verification.py"] = Path(__file__)
    paths["docs/PROPAGATION_EXTENDED_PROTOCOL.md"] = PROTOCOL
    return paths


def _hashes() -> dict[str, str]:
    return {name: _hash(path) for name, path in _source_paths().items()}


def _config() -> dict:
    raw = {
        "cell_sizes_m": CELLS_M,
        "targets_xy_m": TARGETS,
        "domain_half_extent_m": 120,
        "domain_probe_half_extent_m": 160,
        "horizon_safety_multiplier": HORIZON_MULTIPLIER,
        "fuel_model_number": 101,
        "fuel_moisture_fraction": MOISTURE,
        "canopy": {"cover": 0.0, "height_m": 0.0, "base_height_m": 0.0, "bulk_density_kg_m3": 0.0},
        "spotting": False,
        "max_cells_per_timestep": 0.4,
        "buffer_width": 3,
        "use_wind_limit": True,
        "surface_lw_ratio_model": "behave",
        "point_cases": CASES,
        "heterogeneous_cases": ["hetero-east", "hetero-west"],
        "heterogeneous_interface_x_m": 40.0,
        "heterogeneous_targets_xy_m": HETERO_TARGETS,
        "heterogeneous_fuels": {"left": 101, "right": 104},
        "acceptance": {"analytic_relative_error_strict_lt": TOL,
                       "adjacent_grid_change_over_fine_analytic_strict_lt": TOL,
                       "domain_probe_arrival_change_over_same_analytic_strict_lt": DOMAIN_TOL},
    }
    # Freeze and JSON-loaded configuration must have the same list/number types.
    return json.loads(json.dumps(raw, sort_keys=True))


def _versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("pyretechnics", "numpy", "scipy")}


def _exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def freeze(path: Path) -> dict:
    """Write a fixed, hash-bound protocol before any simulation is run."""
    versions = _versions()
    if versions["pyretechnics"] != PINNED_ENGINE:
        raise RuntimeError(f"requires pyretechnics=={PINNED_ENGINE}; found {versions['pyretechnics']}")
    value = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "synthetic numerical verification",
        "crs": "LOCAL_METERS",
        "synthetic_time_origin_utc": "2000-01-01T00:00:00Z",
        "seed": SEED,
        "units": {"coordinates": "m", "arrival": "s", "spread_rate": "m/min and m/s as labeled",
                  "wind_input": "km/h at 10m", "wind_native": "m/min at 20ft", "wind_scenario": "m/s midflame"},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "engine_versions": versions},
        "code_sha256": _hashes(),
        "configuration": _config(),
    }
    _exclusive_json(path, value)
    return value


def _validate_freeze(path: Path) -> dict:
    frozen = json.loads(path.read_text())
    if frozen.get("schema_version") != SCHEMA_VERSION or frozen.get("seed") != SEED:
        raise ValueError("freeze schema or seed does not match this benchmark")
    if frozen.get("provenance") != "synthetic numerical verification" or frozen.get("crs") != "LOCAL_METERS":
        raise ValueError("freeze provenance or CRS was changed")
    if frozen.get("units") != {
        "coordinates": "m", "arrival": "s", "spread_rate": "m/min and m/s as labeled",
        "wind_input": "km/h at 10m", "wind_native": "m/min at 20ft", "wind_scenario": "m/s midflame",
    }:
        raise ValueError("freeze units were changed")
    try:
        created = datetime.fromisoformat(frozen["created_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("freeze timestamp is invalid") from exc
    if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
        raise ValueError("freeze timestamp must be timezone-aware UTC")
    if created > datetime.now(timezone.utc):
        raise ValueError("freeze timestamp cannot be in the future")
    if frozen.get("synthetic_time_origin_utc") != "2000-01-01T00:00:00Z":
        raise ValueError("freeze synthetic time origin was changed")
    if frozen.get("configuration") != _config():
        raise ValueError("frozen configuration was changed")
    if frozen.get("code_sha256") != _hashes():
        raise ValueError("source or protocol changed after freeze")
    expected_runtime = {"python": platform.python_version(), "platform": platform.platform(), "engine_versions": _versions()}
    if frozen.get("runtime") != expected_runtime:
        raise ValueError("frozen runtime differs from the current runtime")
    if expected_runtime["engine_versions"]["pyretechnics"] != PINNED_ENGINE:
        raise ValueError(f"requires pyretechnics=={PINNED_ENGINE}")
    return frozen


def ellipse_arrival(p: float, perpendicular_squared: float, a: float, b: float,
                    eccentricity: float, c: float) -> float:
    """Positive point-source arrival from the exact translated ellipse equation."""
    if not all(isfinite(v) for v in (p, perpendicular_squared, a, b, eccentricity, c)):
        raise ValueError("ellipse arguments must be finite")
    if a <= 0 or b <= 0 or perpendicular_squared < 0 or not 0 <= eccentricity < 1:
        raise ValueError("invalid ellipse axes, distance, or eccentricity")
    if (abs(c - a * eccentricity) > 1e-10 * max(1.0, a)
            or abs(b * b - a * a * (1.0 - eccentricity * eccentricity)) > 1e-10 * max(1.0, a * a)):
        raise ValueError("ellipse axes and focus displacement are inconsistent")
    A = 1.0 - eccentricity * eccentricity
    B = 2.0 * p * c / (a * a)
    C = p * p / (a * a) + perpendicular_squared / (b * b)
    if C == 0.0:
        return 0.0
    discriminant = B * B + 4.0 * A * C
    if discriminant < 0.0:
        raise ValueError("invalid ellipse discriminant")
    root = sqrt(discriminant)
    return 2.0 * C / (root + B) if B >= 0 else (-B + root) / (2.0 * A)


def _behavior(fuel: int, case: dict) -> dict:
    from pyretechnics import conversion, fuel_models, surface_fire

    moisture = tuple(MOISTURE[key] for key in (
        "dead_1h", "dead_10h", "dead_100h", "dead_herbaceous", "live_herbaceous", "live_woody"
    ))
    model = fuel_models.moisturize(fuel_models.get_fuel_model(fuel), moisture)
    base = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
    u20_kmh = conversion.wind_speed_10m_to_wind_speed_20ft(case["wind_u10_kmh"])
    u20_m_min = conversion.km_hr_to_m_min(u20_kmh)
    midflame_m_min = surface_fire.calc_midflame_wind_speed(u20_m_min, float(model["delta"]), 0.0, 0.0)
    behavior = surface_fire.calc_surface_fire_behavior_max(
        base, midflame_m_min, case["upwind_deg"], case["slope"], case["aspect_deg"], True, "behave"
    )
    head_m_min = float(behavior["max_spread_rate"])
    length_width = float(behavior["length_to_width_ratio"])
    eccentricity = sqrt(max(0.0, 1.0 - 1.0 / (length_width * length_width)))
    a = head_m_min / (1.0 + eccentricity)
    b = a / length_width
    c = a * eccentricity
    direction = tuple(float(value) for value in behavior["max_spread_direction"])
    return {
        "behavior": behavior, "head_m_min": head_m_min, "length_width": length_width,
        "eccentricity": eccentricity, "a_m_min": a, "b_m_min": b, "c_m_min": c,
        "direction_xyz": direction, "midflame_m_min": midflame_m_min,
        "midflame_m_s": midflame_m_min / 60.0,
        "wind_u10_kmh": case["wind_u10_kmh"], "wind_u20_kmh": u20_kmh,
        "wind_u20_m_min": u20_m_min,
    }


def _scenario(cell: float, case: dict, half: int = 120, heterogeneous: bool = False,
              reverse: bool = False, horizon_s: float | None = None) -> tuple[dict, dict]:
    n = round(half / cell)
    rows = cols = 2 * n + 1
    origin = n
    gx = -case["slope"] * sin(radians(case["aspect_deg"]))
    gy = -case["slope"] * cos(radians(case["aspect_deg"]))
    wind_speed = _behavior(101, case)["midflame_m_s"]
    wind_angle = radians(case["upwind_deg"] + 180.0)
    wind = [wind_speed * sin(wind_angle), wind_speed * cos(wind_angle)] if wind_speed else [0.0, 0.0]
    fuel = []
    for row in range(rows):
        y = (row - origin) * cell
        fuel_row = []
        for col in range(cols):
            x = (col - origin) * cell
            if heterogeneous:
                code = (104 if x < 40 else 101) if reverse else (101 if x < 40 else 104)
            else:
                code = 101
            fuel_row.append(code)
        fuel.append(fuel_row)
    ignitions = [[row, origin] for row in range(rows)] if heterogeneous else [[origin, origin]]
    scenario = {
        "schema_version": "standard-fuel-0.1", "scenario_id": f"extended-{case['id']}-{cell:g}m",
        "provenance": "synthetic", "crs": "LOCAL_METERS", "seed": SEED,
        "rows": rows, "cols": cols, "cell_size_m": cell,
        "horizon_s": horizon_s if horizon_s is not None else 1.0,
        "fuel_model": fuel,
        "elevation_m": [[gx * ((col - origin) * cell) + gy * ((row - origin) * cell)
                         for col in range(cols)] for row in range(rows)],
        "fuel_moisture_fraction": MOISTURE, "midflame_wind_to_m_s": wind, "ignitions": ignitions,
    }
    return scenario, {"gx": gx, "gy": gy, "wind": wind}


def _make_levelset_cubes(scenario: dict, case: dict) -> dict:
    import numpy as np
    from pyretechnics.space_time_cube import SpaceTimeCube

    rows, cols = scenario["rows"], scenario["cols"]
    cell = scenario["cell_size_m"]
    horizon_min = scenario["horizon_s"] / 60.0
    shape = (1, rows, cols)
    moisture = scenario["fuel_moisture_fraction"]

    def scalar_cube(value):
        return SpaceTimeCube(shape, float(value))

    fuel_grid = np.asarray([scenario["fuel_model"]], dtype=np.float32)
    fuel_cube = SpaceTimeCube(shape, fuel_grid)
    cubes = {
        "slope": scalar_cube(case["slope"]), "aspect": scalar_cube(case["aspect_deg"]),
        "fuel_model": fuel_cube, "canopy_cover": scalar_cube(0.0), "canopy_height": scalar_cube(0.0),
        "canopy_base_height": scalar_cube(0.0), "canopy_bulk_density": scalar_cube(0.0),
        "wind_speed_10m": scalar_cube(case["wind_u10_kmh"]),
        "upwind_direction": scalar_cube(case["upwind_deg"]),
        "fuel_moisture_dead_1hr": scalar_cube(moisture["dead_1h"]),
        "fuel_moisture_dead_10hr": scalar_cube(moisture["dead_10h"]),
        "fuel_moisture_dead_100hr": scalar_cube(moisture["dead_100h"]),
        "fuel_moisture_live_herbaceous": scalar_cube(moisture["live_herbaceous"]),
        "fuel_moisture_live_woody": scalar_cube(moisture["live_woody"]),
        "foliar_moisture": scalar_cube(0.60), "fuel_spread_adjustment": scalar_cube(1.0),
        "weather_spread_adjustment": scalar_cube(1.0),
    }
    return cubes


def _levelset(scenario: dict, case: dict, targets: tuple[tuple[float, float], ...]) -> dict:
    from pyretechnics.eulerian_level_set import SpreadState, spread_fire_with_phi_field

    rows, cols = scenario["rows"], scenario["cols"]
    cell = scenario["cell_size_m"]
    horizon_min = scenario["horizon_s"] / 60.0
    shape = (1, rows, cols)
    cubes = _make_levelset_cubes(scenario, case)
    state = SpreadState(shape).set_resolution((horizon_min + 1.0, cell, cell)).set_start_time(0.0)
    for row, col in scenario["ignitions"]:
        state.ignite_cell((row, col))
    result = spread_fire_with_phi_field(
        cubes, state, start_time=0.0, max_duration=horizon_min,
        max_cells_per_timestep=0.4, buffer_width=3, use_wind_limit=True,
        spot_ignitions={}, spot_config=None,
    )
    matrix = result["spread_state"].get_full_matrices(["time_of_arrival"])["time_of_arrival"]
    origin = (rows - 1) // 2
    arrivals = {}
    for x, y in targets:
        value = float(matrix[origin + round(y / cell), origin + round(x / cell)])
        arrivals[(x, y)] = value * 60.0 if isfinite(value) else None
    return arrivals


def _native_head_rate(fuel: int, case: dict) -> float:
    """Ask Pyretechnics' input-only burn-cell API for its head rate in m/min."""
    from pyretechnics.burn_cells import burn_cell_as_head_fire
    from pyretechnics.space_time_cube import SpaceTimeCube

    shape = (1, 1, 1)
    def cube(value):
        return SpaceTimeCube(shape, float(value))
    cubes = {
        "slope": cube(case["slope"]), "aspect": cube(case["aspect_deg"]), "fuel_model": cube(fuel),
        "canopy_cover": cube(0), "canopy_height": cube(0), "canopy_base_height": cube(0),
        "canopy_bulk_density": cube(0), "wind_speed_10m": cube(case["wind_u10_kmh"]),
        "upwind_direction": cube(case["upwind_deg"]),
        "fuel_moisture_dead_1hr": cube(MOISTURE["dead_1h"]),
        "fuel_moisture_dead_10hr": cube(MOISTURE["dead_10h"]),
        "fuel_moisture_dead_100hr": cube(MOISTURE["dead_100h"]),
        "fuel_moisture_live_herbaceous": cube(MOISTURE["live_herbaceous"]),
        "fuel_moisture_live_woody": cube(MOISTURE["live_woody"]),
        "foliar_moisture": cube(0.60), "fuel_spread_adjustment": cube(1),
        "weather_spread_adjustment": cube(1),
    }
    result = burn_cell_as_head_fire(cubes, (0, 0, 0), True, "behave")
    return float(result["spread_rate"])


def _validate_native_head_rates() -> None:
    for case in CASES:
        calculated = _behavior(101, case)["head_m_min"]
        native = _native_head_rate(101, case)
        if abs(calculated - native) / native >= 1e-5:
            raise RuntimeError(f"input-only head-rate API disagreement for {case['id']}: {calculated} vs {native}")


def _point_reference(case: dict, xy: tuple[float, float]) -> tuple[float, dict]:
    behavior = _behavior(101, case)
    x, y = xy
    gx = -case["slope"] * sin(radians(case["aspect_deg"]))
    gy = -case["slope"] * cos(radians(case["aspect_deg"]))
    dx, dy, dz = behavior["direction_xyz"]
    norm = sqrt(dx * dx + dy * dy + dz * dz)
    direction = (dx / norm, dy / norm, dz / norm)
    q = (x, y, gx * x + gy * y)
    p = sum(qi * di for qi, di in zip(q, direction))
    perpendicular_squared = max(0.0, sum(value * value for value in q) - p * p)
    arrival_minutes = ellipse_arrival(
        p, perpendicular_squared, behavior["a_m_min"], behavior["b_m_min"],
        behavior["eccentricity"], behavior["c_m_min"],
    )
    return arrival_minutes * 60.0, {
        "reference_head_m_min": behavior["head_m_min"],
        "length_width_ratio": behavior["length_width"],
        "eccentricity": behavior["eccentricity"],
        "ellipse_a_m_min": behavior["a_m_min"],
        "ellipse_b_m_min": behavior["b_m_min"],
        "ellipse_c_m_min": behavior["c_m_min"],
        "direction_xyz": behavior["direction_xyz"],
        "slope_gradient": [gx, gy],
        "wind_u10_kmh": behavior["wind_u10_kmh"],
        "wind_u20_kmh": behavior["wind_u20_kmh"],
        "wind_u20_m_min": behavior["wind_u20_m_min"],
        "midflame_wind_m_min": behavior["midflame_m_min"],
        "midflame_wind_m_s": behavior["midflame_m_s"],
    }


def _hetero_reference(reverse: bool) -> dict:
    still_air = {"wind_u10_kmh": 0.0, "upwind_deg": 0.0, "slope": 0.0, "aspect_deg": 0.0}
    left_fuel, right_fuel = (104, 101) if reverse else (101, 104)
    left_rate = _behavior(left_fuel, still_air)["head_m_min"]
    right_rate = _behavior(right_fuel, still_air)["head_m_min"]
    return {"left_rate_m_min": left_rate, "right_rate_m_min": right_rate}


def _hetero_arrival_s(x: float, rates: dict) -> float:
    return (min(x, 40.0) / rates["left_rate_m_min"] +
            max(x - 40.0, 0.0) / rates["right_rate_m_min"]) * 60.0


def _dijkstra(scenario: dict, targets: tuple[tuple[float, float], ...]) -> dict:
    replay = run_fire_standard(scenario)
    origin = (scenario["rows"] - 1) // 2
    cell = scenario["cell_size_m"]
    return {(x, y): replay["arrival_time_s"][origin + round(y / cell)][origin + round(x / cell)]
            for x, y in targets}


def _failure_records(case_id: str, cell: float, targets: tuple[tuple[float, float], ...],
                     engines: tuple[str, ...], error: Exception, analytic: dict | None = None) -> list[dict]:
    message = f"{type(error).__name__}: {error}"
    records = []
    for xy in targets:
        for engine in engines:
            entry = {"case_id": case_id, "grid_cell_size_m": cell, "target_xy_m": list(xy),
                     "engine": engine, "analytic_arrival_s": None, "arrival_s": None,
                     "relative_error": None, "pass": False, "error": message}
            if analytic and xy in analytic:
                entry["analytic_arrival_s"] = analytic[xy]
            records.append(entry)
    return records


def _execution_error(errors: list[dict], case_id: str, cell: float | None,
                    engine: str, error: Exception) -> None:
    errors.append({"case_id": case_id, "grid_cell_size_m": cell, "engine": engine,
                   "error": f"{type(error).__name__}: {error}"})


def _domain_failure_records(case_id: str, targets: tuple[tuple[float, float], ...],
                            engines: tuple[str, ...], error: Exception, analytic: dict) -> list[dict]:
    message = f"{type(error).__name__}: {error}"
    return [{"case_id": case_id, "target_xy_m": list(xy), "engine": engine,
             "half_extents_m": [120, 160], "analytic_arrival_s": analytic[xy],
             "arrival_change_fraction": None, "pass": False, "error": message}
            for xy in targets for engine in engines]


def _grid_refinement(groups: tuple[list[dict], ...]) -> list[dict]:
    refinement = []
    for group in groups:
        keys = sorted({(row["case_id"], tuple(row["target_xy_m"]), row["engine"]) for row in group})
        for case_id, xy, engine in keys:
            values = {row["grid_cell_size_m"]: row for row in group
                      if row["case_id"] == case_id and tuple(row["target_xy_m"]) == xy and row["engine"] == engine}
            for coarse, fine in zip(CELLS_M, CELLS_M[1:]):
                coarse_row, fine_row = values.get(coarse), values.get(fine)
                valid = (coarse_row and fine_row and coarse_row.get("arrival_s") is not None
                         and fine_row.get("arrival_s") is not None and fine_row.get("analytic_arrival_s"))
                change = (abs(coarse_row["arrival_s"] - fine_row["arrival_s"]) /
                          fine_row["analytic_arrival_s"]) if valid else None
                refinement.append({"case_id": case_id, "engine": engine, "target_xy_m": list(xy),
                                   "coarse_cell_size_m": coarse, "fine_cell_size_m": fine,
                                   "change_fraction": change, "pass": _passes(change)})
    return refinement


def _domain_probe(case_id: str, case: dict, targets: tuple[tuple[float, float], ...],
                  analytic: dict, errors: list[dict], heterogeneous: bool = False,
                  reverse: bool = False) -> list[dict]:
    engines = ("dijkstra", "levelset")
    arrivals = {half: {} for half in (120, 160)}
    failure_text = {}
    horizon = max(analytic.values()) * HORIZON_MULTIPLIER
    for half in (120, 160):
        try:
            scenario, _ = _scenario(1.0, case, half, heterogeneous, reverse, horizon)
        except Exception as exc:
            _execution_error(errors, f"{case_id}-domain", half, "scenario-setup", exc)
            for engine in engines:
                arrivals[half][engine] = {xy: None for xy in targets}
                failure_text[(half, engine)] = f"{type(exc).__name__}: {exc}"
            continue
        for engine, execute in (("dijkstra", lambda: _dijkstra(scenario, targets)),
                                ("levelset", lambda: _levelset(scenario, case, targets))):
            try:
                arrivals[half][engine] = execute()
            except Exception as exc:
                _execution_error(errors, f"{case_id}-domain", half, engine, exc)
                arrivals[half][engine] = {xy: None for xy in targets}
                failure_text[(half, engine)] = f"{type(exc).__name__}: {exc}"
    output = []
    for xy in targets:
        for engine in engines:
            small = arrivals[120][engine][xy]
            large = arrivals[160][engine][xy]
            change = abs(small - large) / analytic[xy] if small is not None and large is not None else None
            entry = {"case_id": case_id, "target_xy_m": list(xy), "engine": engine,
                     "half_extents_m": [120, 160], "analytic_arrival_s": analytic[xy],
                     "arrival_120m_s": small, "arrival_160m_s": large,
                     "arrival_change_fraction": change, "pass": _passes(change, DOMAIN_TOL)}
            if change is None:
                entry["error"] = failure_text.get((120, engine), failure_text.get((160, engine), "target was unreachable"))
            output.append(entry)
    return output


def run(freeze_path: Path, report_path: Path) -> dict:
    frozen = _validate_freeze(freeze_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    _validate_native_head_rates()
    point_records: list[dict] = []
    hetero_records: list[dict] = []
    errors: list[dict] = []

    for case in CASES:
        analytic = {xy: _point_reference(case, xy) for xy in TARGETS}
        horizon = max(value[0] for value in analytic.values()) * HORIZON_MULTIPLIER
        for cell in CELLS_M:
            try:
                scenario, geometry = _scenario(cell, case, horizon_s=horizon)
            except Exception as exc:
                _execution_error(errors, case["id"], cell, "scenario-setup", exc)
                for xy in TARGETS:
                    exact, reference = analytic[xy]
                    point_records.extend(_failure_records(case["id"], cell, (xy,),
                        ("dijkstra", "levelset"), exc, {xy: exact}))
                continue
            for engine, execute in (("dijkstra", lambda: _dijkstra(scenario, TARGETS)),
                                    ("levelset", lambda: _levelset(scenario, case, TARGETS))):
                try:
                    arrivals = execute()
                except Exception as exc:
                    _execution_error(errors, case["id"], cell, engine, exc)
                    for xy in TARGETS:
                        exact, _ = analytic[xy]
                        point_records.extend(_failure_records(case["id"], cell, (xy,), (engine,), exc, {xy: exact}))
                    continue
                for xy in TARGETS:
                    exact, reference = analytic[xy]
                    arrival = arrivals.get(xy)
                    relative_error = abs(arrival - exact) / exact if arrival is not None and exact > 0 else None
                    point_records.append({"case_id": case["id"], "grid_cell_size_m": cell,
                        "target_xy_m": list(xy), "engine": engine, "analytic_arrival_s": exact,
                        "arrival_s": arrival, "relative_error": relative_error,
                        "pass": _passes(relative_error), "reference": reference,
                        "grid_geometry": geometry, "simulation_horizon_s": horizon})

    hetero_analytic: dict[str, dict] = {}
    for reverse in (False, True):
        case_id = "hetero-west" if reverse else "hetero-east"
        case = {"id": case_id, "wind_u10_kmh": 0.0, "upwind_deg": 0.0, "slope": 0.0, "aspect_deg": 0.0}
        rates = _hetero_reference(reverse)
        exact = {xy: _hetero_arrival_s(xy[0], rates) for xy in HETERO_TARGETS}
        hetero_analytic[case_id] = exact
        horizon = max(exact.values()) * HORIZON_MULTIPLIER
        for cell in CELLS_M:
            try:
                scenario, _ = _scenario(cell, case, heterogeneous=True, reverse=reverse, horizon_s=horizon)
            except Exception as exc:
                _execution_error(errors, case_id, cell, "scenario-setup", exc)
                for xy in HETERO_TARGETS:
                    hetero_records.extend(_failure_records(case_id, cell, (xy,),
                        ("dijkstra", "levelset"), exc, {xy: exact[xy]}))
                continue
            for engine, execute in (("dijkstra", lambda: _dijkstra(scenario, HETERO_TARGETS)),
                                    ("levelset", lambda: _levelset(scenario, case, HETERO_TARGETS))):
                try:
                    arrivals = execute()
                except Exception as exc:
                    _execution_error(errors, case_id, cell, engine, exc)
                    for xy in HETERO_TARGETS:
                        hetero_records.extend(_failure_records(case_id, cell, (xy,), (engine,), exc, exact))
                    continue
                for xy in HETERO_TARGETS:
                    arrival = arrivals.get(xy)
                    relative_error = abs(arrival - exact[xy]) / exact[xy] if arrival is not None else None
                    hetero_records.append({"case_id": case_id, "grid_cell_size_m": cell,
                        "target_xy_m": list(xy), "engine": engine, "interface_x_m": 40.0,
                        "interface_sample_coordinate_m": 40.0 - cell / 2.0,
                        "analytic_arrival_s": exact[xy], "arrival_s": arrival,
                        "relative_error": relative_error, "pass": _passes(relative_error),
                        "fuel_rates_m_min": rates, "simulation_horizon_s": horizon})

    refinement = _grid_refinement((point_records, hetero_records))
    pair_acceptance = []
    for case_id in [case["id"] for case in CASES] + ["hetero-east", "hetero-west"]:
        records = point_records if case_id in {case["id"] for case in CASES} else hetero_records
        targets = TARGETS if records is point_records else HETERO_TARGETS
        for engine in ("dijkstra", "levelset"):
            for coarse, fine in zip(CELLS_M, CELLS_M[1:]):
                analytical_pass = all(any(row["case_id"] == case_id and row["engine"] == engine
                    and row["grid_cell_size_m"] == grid and tuple(row["target_xy_m"]) == xy
                    and row.get("pass") is True for row in records) for grid in (coarse, fine) for xy in targets)
                shifts = [row for row in refinement if row["case_id"] == case_id and row["engine"] == engine
                          and row["coarse_cell_size_m"] == coarse and row["fine_cell_size_m"] == fine]
                refinement_pass = len(shifts) == len(targets) and all(row["pass"] for row in shifts)
                pair_acceptance.append({"case_id": case_id, "engine": engine,
                    "coarse_cell_size_m": coarse, "fine_cell_size_m": fine,
                    "all_targets_analytic_at_both_resolutions": analytical_pass,
                    "all_target_refinement_pass": refinement_pass,
                    "pair_pass": analytical_pass and refinement_pass})

    domain = []
    wind_case = next(case for case in CASES if case["id"] == "wind-east")
    domain.extend(_domain_probe("wind-east", wind_case, TARGETS,
                                {xy: _point_reference(wind_case, xy)[0] for xy in TARGETS}, errors))
    still_air = {"id": "heterogeneous-domain", "wind_u10_kmh": 0.0, "upwind_deg": 0.0,
                 "slope": 0.0, "aspect_deg": 0.0}
    for reverse in (False, True):
        case_id = "hetero-west" if reverse else "hetero-east"
        domain.extend(_domain_probe(case_id, still_air, HETERO_TARGETS, hetero_analytic[case_id], errors,
                                    heterogeneous=True, reverse=reverse))

    point_pass = bool(point_records) and all(row.get("pass") is True for row in point_records)
    hetero_pass = bool(hetero_records) and all(row.get("pass") is True for row in hetero_records)
    refinement_pass = bool(refinement) and all(row.get("pass") is True for row in refinement)
    domain_pass = bool(domain) and all(row.get("pass") is True for row in domain)
    report = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": frozen["provenance"], "crs": frozen["crs"], "seed": SEED,
        "synthetic_time_origin_utc": frozen["synthetic_time_origin_utc"], "units": frozen["units"],
        "runtime": frozen["runtime"], "frozen_protocol_sha256": _hash(freeze_path),
        "code_sha256": _hashes(), "configuration": frozen["configuration"],
        "point_cases": point_records, "heterogeneous_cases": hetero_records,
        "grid_refinement": refinement, "resolution_pair_acceptance": pair_acceptance,
        "domain_probe": domain, "case_errors": errors, "physical_validation_pass": None,
        "checks": {"point_arrival_error_lt_5pct": point_pass,
                   "heterogeneous_arrival_error_lt_5pct": hetero_pass,
                   "adjacent_grid_change_lt_5pct": refinement_pass,
                   "domain_probe_arrival_change_lt_1pct": domain_pass,
                   "execution_errors_none": not errors,
                   "all_targets_reached": all(row.get("arrival_s") is not None for row in point_records + hetero_records)},
        "overall_numerical_pass": point_pass and hetero_pass and refinement_pass and domain_pass and not errors,
        "limitations": [
            "Numerical solver verification only; no physical validation or field data inference.",
            "One-cell point ignition is compared directly to a point source; initialization extent remains in error.",
            "Heterogeneous interface sampling and finite ignited-column extent remain in error against the nominal continuous problem.",
        ],
    }
    _validate_freeze(freeze_path)
    _exclusive_json(report_path, report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", required=True, type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.protocol)
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
