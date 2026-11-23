"""Compare FM1 grid-arrival area across cell sizes on one fixed local domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fire_rothermel import run_fire_rothermel


def compare_resolution(
    cell_sizes_m: list[int] | tuple[int, ...] = (100, 50, 25, 10, 5),
    *,
    moisture_fraction: float = 0.06,
    wind_to_m_s: tuple[float, float] = (2, 0),
    elevation_gradient: tuple[float, float] = (0, 0),
) -> dict:
    if not cell_sizes_m or any(not isinstance(size, int) or size <= 0 or 500 % size for size in cell_sizes_m):
        raise ValueError("cell sizes must be positive integer divisors of 500 m")
    if tuple(cell_sizes_m) != tuple(sorted(set(cell_sizes_m), reverse=True)):
        raise ValueError("cell sizes must be unique and ordered coarse to fine")
    results = []
    for size in cell_sizes_m:
        count = 1000 // size + 1
        scenario = {
            "schema_version": "rothermel-fm1-0.1", "scenario_id": f"fm1-resolution-{size}m",
            "provenance": "synthetic", "seed": 42, "crs": "LOCAL_METERS",
            "rows": count, "cols": count, "cell_size_m": size, "horizon_s": 600,
            "fuel_model": [["FM1"] * count for _ in range(count)],
            "dead_fuel_moisture_fraction": [[moisture_fraction] * count for _ in range(count)],
            "elevation_m": [[elevation_gradient[0] * (c - count // 2) * size + elevation_gradient[1] * (r - count // 2) * size for c in range(count)] for r in range(count)],
            "midflame_wind_to_m_s": list(wind_to_m_s), "ignitions": [[count // 2, count // 2]],
        }
        replay = run_fire_rothermel(scenario)
        area = replay["affected_area_ha"]
        previous = results[-1]["affected_area_ha"] if results else None
        results.append({
            "cell_size_m": size, "grid_cells_per_side": count,
            "scenario_sha256": replay["scenario_sha256"],
            "affected_area_ha": area,
            "change_from_previous_percent": round(100 * (area - previous) / previous, 3) if previous else None,
        })
    return {
        "comparison_schema_version": "fm1-resolution-0.1",
        "fire_schema_version": "rothermel-fm1-0.1",
        "seed": 42, "crs": "LOCAL_METERS", "provenance": "synthetic",
        "units": {"distance": "m", "time": "s", "area": "ha", "wind": "m/s"},
        "fixed_center_extent_m": 1000, "horizon_s": 600,
        "wind_to_m_s": list(wind_to_m_s), "dead_fuel_moisture_fraction": moisture_fraction,
        "elevation_gradient": list(elevation_gradient),
        "results": results,
    }


def compare_varied_resolution() -> dict:
    """Check several homogeneous conditions; report differences, not a passing grid."""
    profiles = (
        ("calm", 0.06, (0, 0), (0, 0)),
        ("east_wind", 0.06, (2, 0), (0, 0)),
        ("west_wind", 0.06, (-2, 0), (0, 0)),
        ("dry_east_wind", 0.03, (2, 0), (0, 0)),
        ("moist_east_wind", 0.09, (2, 0), (0, 0)),
        ("east_upslope", 0.06, (2, 0), (0.1, 0)),
        ("east_wind_west_upslope", 0.06, (2, 0), (-0.1, 0)),
    )
    cases = []
    for name, moisture, wind, gradient in profiles:
        result = compare_resolution((25, 10, 5), moisture_fraction=moisture, wind_to_m_s=wind, elevation_gradient=gradient)
        coarse = result["results"][0]["affected_area_ha"]
        fine = result["results"][-1]["affected_area_ha"]
        cases.append({
            "profile": name, "dead_fuel_moisture_fraction": moisture,
            "wind_to_m_s": list(wind), "elevation_gradient": list(gradient),
            "results": result["results"],
            "difference_25m_vs_5m_percent_of_5m": round(100 * (coarse - fine) / fine, 3),
        })
    return {
        "comparison_schema_version": "fm1-varied-resolution-0.1",
        "fire_schema_version": "rothermel-fm1-0.1",
        "seed": 42, "crs": "LOCAL_METERS", "provenance": "synthetic",
        "units": {"distance": "m", "time": "s", "area": "ha", "wind": "m/s"},
        "fixed_center_extent_m": 1000, "horizon_s": 600, "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--varied", action="store_true", help="Compare wind, moisture, and slope cases")
    args = parser.parse_args()
    result = compare_varied_resolution() if args.varied else compare_resolution()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
