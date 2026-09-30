"""Reproducible numerical checks for homogeneous Pyretechnics fire propagation.

This is an offline verification benchmark. It does not change or provide an
interface to Nightjar's production simulator or controller.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from math import hypot, isfinite
from pathlib import Path
import platform

from .fire_standard import run_fire_standard
from . import fire_standard as fire_standard_module

SCHEMA_VERSION = "nightjar-propagation-verification-0.1"
CELL_SIZES_M = (10.0, 5.0, 2.0, 1.0)
TARGETS_M = ((40.0, 10.0), (40.0, 20.0), (40.0, 40.0), (80.0, 0.0))
SEED = 0
FUEL_MODEL = 101
MOISTURE = {
    "dead_1h": 0.04,
    "dead_10h": 0.05,
    "dead_100h": 0.07,
    "dead_herbaceous": 0.04,
    "live_herbaceous": 0.30,
    "live_woody": 0.60,
}
ANALYTICAL_TOLERANCE = 0.05
REFINEMENT_TOLERANCE = 0.05


def _configuration_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _passes(metric: float, threshold: float) -> bool:
    """Apply the benchmark's declared strict upper-bound classification."""
    return isfinite(metric) and metric < threshold


def _scenario(cell_size_m: float, target_m: tuple[float, float]) -> tuple[dict, tuple[int, int]]:
    """Create a centered local grid with all target coordinates exactly aligned."""
    half_extent = 120.0
    cells_each_side = round(half_extent / cell_size_m)
    rows = cols = 2 * cells_each_side + 1
    origin = cells_each_side
    x_m, y_m = target_m
    if not all((v / cell_size_m).is_integer() for v in (x_m, y_m)):
        raise ValueError("benchmark target coordinates must align exactly to each grid")
    target = (origin + int(y_m / cell_size_m), origin + int(x_m / cell_size_m))
    scenario = {
        "schema_version": "standard-fuel-0.1",
        "scenario_id": f"calm-point-{cell_size_m:g}m-{x_m:g}-{y_m:g}",
        "provenance": "synthetic",
        "crs": "LOCAL_METERS",
        "seed": SEED,
        "rows": rows,
        "cols": cols,
        "cell_size_m": cell_size_m,
        "horizon_s": 100_000.0,
        "fuel_model": [[FUEL_MODEL] * cols for _ in range(rows)],
        "elevation_m": [[0.0] * cols for _ in range(rows)],
        "fuel_moisture_fraction": MOISTURE,
        "midflame_wind_to_m_s": [0.0, 0.0],
        "ignitions": [[origin, origin]],
    }
    return scenario, target


def _levelset_arrivals_s(scenario: dict, targets: dict[tuple[float, float], tuple[int, int]], rate_m_s: float) -> tuple[dict, float]:
    """Run pinned Pyretechnics level set from the same single-cell point ignition."""
    from pyretechnics.eulerian_level_set import SpreadState, spread_fire_with_phi_field
    from pyretechnics.space_time_cube import SpaceTimeCube

    rows, cols = scenario["rows"], scenario["cols"]
    cell = scenario["cell_size_m"]
    horizon_min = scenario["horizon_s"] / 60.0
    bands = 1
    shape = (bands, rows, cols)
    moisture = scenario["fuel_moisture_fraction"]
    # SpaceTimeCube fuel_model uses the same Scott & Burgan model number.
    cubes = {
        "slope": SpaceTimeCube(shape, 0.0),
        "aspect": SpaceTimeCube(shape, 0.0),
        "fuel_model": SpaceTimeCube(shape, float(FUEL_MODEL)),
        "canopy_cover": SpaceTimeCube(shape, 0.0),
        "canopy_height": SpaceTimeCube(shape, 0.0),
        "canopy_base_height": SpaceTimeCube(shape, 0.0),
        "canopy_bulk_density": SpaceTimeCube(shape, 0.0),
        "wind_speed_10m": SpaceTimeCube(shape, 0.0),
        "upwind_direction": SpaceTimeCube(shape, 0.0),
        "fuel_moisture_dead_1hr": SpaceTimeCube(shape, moisture["dead_1h"]),
        "fuel_moisture_dead_10hr": SpaceTimeCube(shape, moisture["dead_10h"]),
        "fuel_moisture_dead_100hr": SpaceTimeCube(shape, moisture["dead_100h"]),
        "fuel_moisture_live_herbaceous": SpaceTimeCube(shape, moisture["live_herbaceous"]),
        "fuel_moisture_live_woody": SpaceTimeCube(shape, moisture["live_woody"]),
        "foliar_moisture": SpaceTimeCube(shape, 0.60),
        "fuel_spread_adjustment": SpaceTimeCube(shape, 1.0),
        "weather_spread_adjustment": SpaceTimeCube(shape, 1.0),
    }
    state = SpreadState(shape).set_resolution((horizon_min + 1.0, cell, cell)).set_start_time(0.0)
    row0, col0 = scenario["ignitions"][0]
    state.ignite_cell((row0, col0))
    result = spread_fire_with_phi_field(
        cubes,
        state,
        start_time=0.0,
        max_duration=horizon_min,
        max_cells_per_timestep=0.4,
        buffer_width=3,
        use_wind_limit=True,
        spot_ignitions={},
        spot_config=None,
    )
    matrices = result["spread_state"].get_full_matrices(["time_of_arrival", "spread_rate"])
    arrivals = {}
    for target_m, target in targets.items():
        minutes = float(matrices["time_of_arrival"][target[0], target[1]])
        if not isfinite(minutes):
            raise RuntimeError(f"level-set did not reach target {target_m}")
        arrivals[target_m] = minutes * 60.0
    # Calibrate/check that both engines use the same still-air physical rate.
    # The level-set output is only used if its measured cell behavior agrees;
    # its own arrival remains an independent numerical result.
    rate_values = matrices["spread_rate"]
    positive_rates = rate_values[rate_values > 0]
    levelset_rate = float(positive_rates.max()) / 60.0 if positive_rates.size else 0.0
    if levelset_rate <= 0 or abs(levelset_rate - rate_m_s) / rate_m_s > 0.01:
        raise RuntimeError(f"level-set and Dijkstra still-air rates differ by more than 1% ({levelset_rate} vs {rate_m_s})")
    return arrivals, levelset_rate


def _target_index(cell_size_m: float, xy_m: tuple[float, float], rows: int) -> tuple[int, int]:
    origin = (rows - 1) // 2
    x_m, y_m = xy_m
    return origin + int(y_m / cell_size_m), origin + int(x_m / cell_size_m)


def _seed_sensitivity(cell_size_m: float, target_m: tuple[float, float]) -> dict:
    """Verify whether the stored seed metadata affects this nonstochastic path."""
    scenario, target = _scenario(cell_size_m, target_m)
    alternate = deepcopy(scenario)
    alternate["seed"] = SEED + 1
    first = run_fire_standard(scenario)["arrival_time_s"][target[0]][target[1]]
    second = run_fire_standard(alternate)["arrival_time_s"][target[0]][target[1]]
    change = abs(first - second) if first is not None and second is not None else float("inf")
    return {
        "cell_size_m": cell_size_m,
        "target_xy_m": list(target_m),
        "seeds": [SEED, SEED + 1],
        "arrival_absolute_change_s": change,
        "seed_sensitive": change != 0.0,
        "interpretation": "seed is metadata only; propagation and level-set contain no stochastic process",
    }


def run_verification(*, include_levelset: bool = True) -> dict:
    """Run point-arrival and physical-coordinate grid-refinement checks."""
    records = []
    per_grid: dict[float, dict[tuple[float, float], dict]] = {}
    for cell in CELL_SIZES_M:
        per_grid[cell] = {}
        scenario, _ = _scenario(cell, TARGETS_M[0])
        target_indices = {target_m: _target_index(cell, target_m, scenario["rows"]) for target_m in TARGETS_M}
        replay = run_fire_standard(scenario)
        # This homogeneous still-air path is exactly isotropic; derive its
        # rate from the axial arrival in the same grid, four cells from origin.
        origin = (scenario["rows"] - 1) // 2
        rate_m_s = 4 * cell / replay["arrival_time_s"][origin][origin + 4]
        levelset_arrivals = {}
        levelset_rate = None
        if include_levelset:
            levelset_arrivals, levelset_rate = _levelset_arrivals_s(scenario, target_indices, rate_m_s)
        for target_m in TARGETS_M:
            target = target_indices[target_m]
            arrival = replay["arrival_time_s"][target[0]][target[1]]
            if arrival is None:
                raise RuntimeError(f"Dijkstra did not reach {target_m} m at cell size {cell} m")
            exact_s = hypot(*target_m) / rate_m_s
            error = abs(arrival - exact_s) / exact_s
            levelset_arrival = None
            levelset_error = None
            if include_levelset:
                levelset_arrival = levelset_arrivals[target_m]
                levelset_error = abs(levelset_arrival - exact_s) / exact_s
            record = {
                "case_id": f"calm-point-{cell:g}m-{target_m[0]:g}-{target_m[1]:g}",
                "grid_cell_size_m": cell,
                "target_xy_m": list(target_m),
                "target_distance_m": hypot(*target_m),
                "target_distance_over_cell_size": hypot(*target_m) / cell,
                "fuel_model_number": FUEL_MODEL,
                "midflame_wind_to_m_s": [0.0, 0.0],
                "canopy_and_crown": "disabled (zero canopy inputs)",
                "spotting": "disabled",
                "initialization": "one ignited grid cell centered at (0, 0); exact reference is a point source; seed-cell extent is not subtracted",
                "still_air_rate_m_s": rate_m_s,
                "analytic_point_arrival_s": exact_s,
                "dijkstra_arrival_s": arrival,
                "dijkstra_relative_error": error,
                "dijkstra_analytical_pass": _passes(error, ANALYTICAL_TOLERANCE),
                "levelset_arrival_s": levelset_arrival,
                "levelset_rate_m_s": levelset_rate,
                "levelset_relative_error": levelset_error,
                "levelset_analytical_pass": None if levelset_error is None else _passes(levelset_error, ANALYTICAL_TOLERANCE),
            }
            records.append(record)
            per_grid[cell][target_m] = record

    refinement = []
    engine_names = ("dijkstra", "levelset") if include_levelset else ("dijkstra",)
    # Adjacent grid pairs at identical physical target coordinates.
    for coarse, fine in zip(CELL_SIZES_M, CELL_SIZES_M[1:]):
        for target_m in TARGETS_M:
            coarse_record, fine_record = per_grid[coarse][target_m], per_grid[fine][target_m]
            for engine in engine_names:
                coarse_arrival = coarse_record[f"{engine}_arrival_s"]
                fine_arrival = fine_record[f"{engine}_arrival_s"]
                change = abs(coarse_arrival - fine_arrival) / fine_record["analytic_point_arrival_s"]
                refinement.append({
                    "engine": engine,
                    "target_xy_m": list(target_m),
                    "coarse_cell_size_m": coarse,
                    "fine_cell_size_m": fine,
                    "arrival_metric_change_fraction": change,
                    "pass": _passes(change, REFINEMENT_TOLERANCE),
                })

    resolution_pair_acceptance = []
    for coarse, fine in zip(CELL_SIZES_M, CELL_SIZES_M[1:]):
        for engine in engine_names:
            coarse_key = f"{engine}_analytical_pass"
            pair_analytical_pass = all(
                per_grid[resolution][target_m][coarse_key]
                for resolution in (coarse, fine)
                for target_m in TARGETS_M
            )
            pair_refinement = [
                item for item in refinement
                if item["engine"] == engine and item["coarse_cell_size_m"] == coarse and item["fine_cell_size_m"] == fine
            ]
            max_change = max(item["arrival_metric_change_fraction"] for item in pair_refinement)
            resolution_pair_acceptance.append({
                "engine": engine,
                "coarse_cell_size_m": coarse,
                "fine_cell_size_m": fine,
                "analytical_pass_at_both_resolutions": pair_analytical_pass,
                "max_refinement_metric_change_fraction": max_change,
                "refinement_pass": _passes(max_change, REFINEMENT_TOLERANCE),
                "pair_pass": pair_analytical_pass and _passes(max_change, REFINEMENT_TOLERANCE),
            })

    checks = {
        "dijkstra_analytical_arrival_lt_5pct": all(r["dijkstra_analytical_pass"] for r in records),
        "dijkstra_grid_refinement_metric_change_lt_5pct": all(r["pass"] for r in refinement if r["engine"] == "dijkstra"),
        "levelset_grid_refinement_metric_change_lt_5pct": None if not include_levelset else all(r["pass"] for r in refinement if r["engine"] == "levelset"),
        "levelset_analytical_arrival_lt_5pct": None if not include_levelset else all(r["levelset_analytical_pass"] for r in records),
    }
    package_versions = {name: version(name) for name in ("pyretechnics", "numpy", "scipy")}
    configuration = {
        "cell_sizes_m": CELL_SIZES_M,
        "targets_xy_m": TARGETS_M,
        "seed": SEED,
        "fuel_model_number": FUEL_MODEL,
        "fuel_moisture_fraction": MOISTURE,
        "midflame_wind_to_m_s": (0.0, 0.0),
        "canopy_cover_height_base_density": (0.0, 0.0, 0.0),
        "spotting": False,
        "tolerances": {
            "analytical_arrival_relative_error": ANALYTICAL_TOLERANCE,
            "grid_refinement_metric_change": REFINEMENT_TOLERANCE,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "synthetic numerical verification",
        "crs": "LOCAL_METERS",
        "units": {"coordinates": "m", "arrival": "s", "rate": "m/s", "wind": "m/s"},
        "seed": SEED,
        "synthetic_time_origin_utc": "2000-01-01T00:00:00Z",
        "seed_policy": "fixed deterministic synthetic inputs; no stochastic process is used",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "engine_versions": package_versions},
        "code_sha256": {
            "nightjar/fire_standard.py": sha256(Path(fire_standard_module.__file__).read_bytes()).hexdigest(),
            "nightjar/propagation_verification.py": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "configuration_sha256": _configuration_hash(configuration),
        "configuration": configuration,
        "acceptance": {
            "analytical_arrival_relative_error": {"threshold": ANALYTICAL_TOLERANCE, "comparison": "strictly less than"},
            "grid_refinement_metric_change": {"threshold": REFINEMENT_TOLERANCE, "comparison": "strictly less than", "metric": "absolute arrival-time change divided by fine-grid analytical arrival"},
            "empirical_validation": {"thresholds": {"nMAE": 0.30, "absolute_bias": 0.20}, "status": "not implemented in this numeric verification benchmark"},
        },
        "checks": checks,
        "overall_pass": all(value is not False for value in checks.values()),
        "cases": records,
        "grid_refinement": refinement,
        "resolution_pair_acceptance": resolution_pair_acceptance,
        "seed_sensitivity": _seed_sensitivity(2.0, (40.0, 20.0)),
        "limitations": [
            "Calm homogeneous point-source cases isolate spatial propagation error; no RxCADRE observations or empirical validation metrics are computed.",
            "The Dijkstra benchmark starts from one burned cell center while the analytic reference is a point source; no seed-cell radius correction is applied.",
            "Only coordinates exactly aligned to every tested grid are included; no rotated, sloped, nonuniform-fuel, or wind-driven cases are evaluated.",
            "The native level-set comparison uses the same calm fuel and moisture inputs and checks still-air ROS agreement within 1%; it is a numerical cross-check, not a validation against truth.",
            "The seed-sensitivity probe uses two finite metadata seeds; since the benchmark has no stochastic process, it is expected to leave arrivals identical and does not measure uncertainty in physical inputs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--skip-levelset", action="store_true", help="run only the Dijkstra and analytical checks")
    args = parser.parse_args()
    result = run_verification(include_levelset=not args.skip_levelset)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
