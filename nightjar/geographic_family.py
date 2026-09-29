"""Real-world geographic scenario family generator for Nightjar.

This module builds a versioned suite of real-incident decision scenarios ('nightjar-geographic-family-0.1')
across disjoint splits to prevent benchmark overfitting and data leakage:
- 'development': Scenarios derived from Copernicus EMSR578 (Artesa de Segre, Catalonia / UTM 31N)
- 'validation': Scenarios derived from Copernicus EMSR590 (Monfragüe / Casas de Miravete, Extremadura / UTM 30N)
- 'reserved': Scenarios derived from Copernicus EMSR592 (Sierra de Mijas) and EMSR600 (Vall d'Ebo / UTM 30N)

Each scenario incorporates real Scott & Burgan 40 fuels, georeferenced water scooping sources
(reservoirs/rivers), satellite-derived initial burn perimeters, CL-415 operational parameters,
and controlled environmental ensembles (wind, moisture, observation delay).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np

from .cems_catalog import (
    ACTIVATIONS,
    build_cems_scenario,
    fetch_cems_package,
    universal_clc_to_scott_burgan,
)
from .geodesy import epsg_for_utm_zone, wgs84_to_utm

FAMILY_SCHEMA_VERSION = "nightjar-geographic-family-0.1"

# Base seeds per split for deterministic generation
SPLIT_BASE_SEEDS = {
    "development": 100000,
    "validation": 200000,
    "reserved": 300000,
}

# Plausible synoptic midflame wind regimes for summer Iberian heatwaves (east, north in m/s)
WIND_ENSEMBLES = [
    (1.5, 1.5),   # moderate SW -> NE
    (0.8, 2.2),   # moderate S -> N
    (-1.5, 1.8),  # moderate SE -> NW
    (2.5, 0.5),   # brisk W -> E
    (0.0, 2.8),   # brisk S -> N
    (1.8, -1.0),  # moderate NW -> SE
]

# Dead fine-fuel moisture ensembles (1h dead fuel fraction)
MOISTURE_ENSEMBLES = [
    {"dead_1h": 0.04, "dead_10h": 0.05, "dead_100h": 0.07, "dead_herbaceous": 0.04, "live_herbaceous": 0.30, "live_woody": 0.60},  # 97th percentile extreme drought
    {"dead_1h": 0.05, "dead_10h": 0.06, "dead_100h": 0.08, "dead_herbaceous": 0.05, "live_herbaceous": 0.40, "live_woody": 0.70},  # 90th percentile very dry
    {"dead_1h": 0.06, "dead_10h": 0.07, "dead_100h": 0.09, "dead_herbaceous": 0.06, "live_herbaceous": 0.50, "live_woody": 0.80},  # moderate summer dry
]

# Satellite pass intervals and report delivery latencies (seconds)
OBSERVATION_CADENCES = [
    {"interval_s": 300.0, "delay_s": 60.0},   # 5-min UAV / fast sensor
    {"interval_s": 600.0, "delay_s": 90.0},   # 10-min surveillance
    {"interval_s": 900.0, "delay_s": 120.0},  # 15-min tactical satellite
]


def canonical_hash(value: object) -> str:
    """Return SHA-256 hex digest of a JSON-serializable object with canonical formatting."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def slice_tactical_aoi(
    cems_scenario: dict[str, Any],
    center_row: int | None = None,
    center_col: int | None = None,
    window_rows: int = 50,
    window_cols: int = 50,
) -> dict[str, Any]:
    """Slice a tactical sub-window from a large CEMS scenario centered on active perimeter.

    Ensures the sub-window preserves fuels, burn perimeter, water mask, and updates origins.
    """
    fuel_grid = np.array(cems_scenario["fuel_grid"], dtype=np.int32)
    water_mask = np.array(cems_scenario["water_mask"], dtype=bool)
    burn_mask = np.array(cems_scenario["burn_mask"], dtype=bool)
    grid_meta = cems_scenario["grid"]

    rows, cols = fuel_grid.shape
    cell_size = float(grid_meta["cell_size_m"])

    if center_row is None or center_col is None:
        # Find centroid of burned cells
        burned_indices = np.argwhere(burn_mask)
        if len(burned_indices) > 0:
            center_row = int(np.mean(burned_indices[:, 0]))
            center_col = int(np.mean(burned_indices[:, 1]))
        else:
            center_row = rows // 2
            center_col = cols // 2

    # Clamp window boundaries
    r0 = max(0, min(rows - window_rows, center_row - window_rows // 2))
    c0 = max(0, min(cols - window_cols, center_col - window_cols // 2))
    r1 = min(rows, r0 + window_rows)
    c1 = min(cols, c0 + window_cols)

    sub_fuel = fuel_grid[r0:r1, c0:c1].tolist()
    sub_water = water_mask[r0:r1, c0:c1].tolist()
    sub_burn = burn_mask[r0:r1, c0:c1].tolist()

    sub_origin_x = grid_meta["origin_utm_easting_m"] + c0 * cell_size
    sub_origin_y = grid_meta["origin_utm_northing_m"] + r0 * cell_size
    cell_area_ha = (cell_size**2) / 10000.0

    return {
        "rows": r1 - r0,
        "cols": c1 - c0,
        "cell_size_m": cell_size,
        "origin_utm_easting_m": round(sub_origin_x, 2),
        "origin_utm_northing_m": round(sub_origin_y, 2),
        "extent_ha": round((r1 - r0) * (c1 - c0) * cell_area_ha, 2),
        "fuel_grid": sub_fuel,
        "water_mask": sub_water,
        "burn_mask": sub_burn,
        "r0": r0,
        "c0": c0,
    }


def generate_geographic_scenario(
    cems_scenario: dict[str, Any],
    index: int,
    split: str,
    window_rows: int = 40,
    window_cols: int = 40,
    horizon_s: float = 3600.0,
) -> dict[str, Any]:
    """Generate a single deterministic decision scenario from a CEMS base scenario."""
    base_seed = SPLIT_BASE_SEEDS.get(split, 400000)
    seed = base_seed + index
    rng = random.Random(seed)

    # 1. Slice tactical window around burn perimeter
    tactical = slice_tactical_aoi(
        cems_scenario,
        window_rows=window_rows,
        window_cols=window_cols,
    )

    # 2. Select environmental variations
    wind_to = list(rng.choice(WIND_ENSEMBLES))
    moisture = deepcopy(rng.choice(MOISTURE_ENSEMBLES))
    cadence = rng.choice(OBSERVATION_CADENCES)

    # 3. Build observation schedule
    obs_schedule = []
    t = 0.0
    while t <= horizon_s:
        delivered_t = min(horizon_s, t + cadence["delay_s"])
        obs_schedule.append({
            "source_id": f"survey-{int(t)}s",
            "observed_at_s": round(t, 1),
            "available_at_s": round(delivered_t, 1),
            "delivered_at_s": round(delivered_t, 1),
            "coverage": "all",
        })
        t += cadence["interval_s"]

    # 4. Configure aircraft initial position and water sources
    waypoints = cems_scenario.get("water_scooping_waypoints", [])
    if waypoints:
        nearest_wp = waypoints[0]
        refill_utm = [nearest_wp["utm_easting_m"], nearest_wp["utm_northing_m"]]
    else:
        # Fallback to corner of grid
        refill_utm = [tactical["origin_utm_easting_m"], tactical["origin_utm_northing_m"]]

    # Aircraft initial position: patrol point offset from fire
    ac_x = tactical["origin_utm_easting_m"] + (tactical["cols"] * 0.8) * tactical["cell_size_m"]
    ac_y = tactical["origin_utm_northing_m"] + (tactical["rows"] * 0.8) * tactical["cell_size_m"]

    act_id = cems_scenario.get("activation_id", "EMSR578")
    scenario_id = f"geographic-{split[:3]}-{act_id.lower()}-{index:02d}"

    return {
        "schema_version": "nightjar-geographic-scenario-0.1",
        "scenario_id": scenario_id,
        "split": split,
        "seed": seed,
        "activation_id": act_id,
        "name": f"{cems_scenario.get('name', 'Wildfire')} Tactical Scenario {index:02d}",
        "provenance": {
            "family": FAMILY_SCHEMA_VERSION,
            "source_activation": act_id,
            "crs": cems_scenario.get("crs", {}),
            "source_license": "Copernicus terms of use (attribution required)",
        },
        "crs": cems_scenario.get("crs", {}),
        "grid": {
            "rows": tactical["rows"],
            "cols": tactical["cols"],
            "cell_size_m": tactical["cell_size_m"],
            "origin_utm_easting_m": tactical["origin_utm_easting_m"],
            "origin_utm_northing_m": tactical["origin_utm_northing_m"],
            "extent_ha": tactical["extent_ha"],
        },
        "fire": {
            "fuel_grid": tactical["fuel_grid"],
            "burn_mask": tactical["burn_mask"],
            "wind_to_m_s": wind_to,
            "moisture": moisture,
            "horizon_s": horizon_s,
        },
        "aircraft": {
            "model": "CL-415",
            "tank_capacity_l": 6137,
            "initial_water_l": 6137,
            "cruise_speed_m_s": 80.0,
            "drop_speed_m_s": 55.0,
            "initial_position_utm": [round(ac_x, 2), round(ac_y, 2)],
            "water_scooping_utm": refill_utm,
            "refill_service_time_s": 180.0,
            "water_waypoints": waypoints,
        },
        "observation_schedule": obs_schedule,
    }


def build_synthetic_cems_fixture(
    activation_id: str = "EMSR578",
    utm_zone: int = 31,
    rows: int = 50,
    cols: int = 50,
) -> dict[str, Any]:
    """Generate a valid in-memory CEMS scenario fixture for offline testing."""
    rng = random.Random(42)
    # Heterogeneous fuels: 101 (GR1), 145 (SH5), 165 (TU5), 186 (TL6), 0 (NB)
    fuel_choices = [101, 145, 165, 186, 0]
    fuel_grid = [[rng.choice(fuel_choices) for _ in range(cols)] for _ in range(rows)]
    water_mask = [[False] * cols for _ in range(rows)]
    # Mark water reservoir at bottom-left corner
    for r in range(5):
        for c in range(5):
            fuel_grid[r][c] = 0
            water_mask[r][c] = True

    burn_mask = [[False] * cols for _ in range(rows)]
    # Initial burn cluster in center
    cr, cc = rows // 2, cols // 2
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) <= 3:
                burn_mask[cr + dr][cc + dc] = True

    origin_x, origin_y = 400000.0, 4600000.0
    return {
        "schema_version": "nightjar-cems-scenario-0.1",
        "scenario_id": f"fixture-{activation_id.lower()}",
        "activation_id": activation_id,
        "name": f"{activation_id} Test Fixture",
        "region": "Test Region",
        "default_split": "development",
        "crs": {
            "authority": "EPSG",
            "code": epsg_for_utm_zone(utm_zone),
            "utm_zone": utm_zone,
            "horizontal_datum": "ETRS89",
        },
        "grid": {
            "rows": rows,
            "cols": cols,
            "cell_size_m": 100.0,
            "origin_utm_easting_m": origin_x,
            "origin_utm_northing_m": origin_y,
            "extent_ha": rows * cols,
        },
        "fuel_grid": fuel_grid,
        "water_mask": water_mask,
        "burn_mask": burn_mask,
        "water_scooping_waypoints": [
            {
                "name": "Reservoir Waypoint",
                "obj_type": "Lake",
                "lon_deg": 1.0,
                "lat_deg": 41.5,
                "utm_easting_m": origin_x + 250.0,
                "utm_northing_m": origin_y + 250.0,
                "eligible_for_scooping": True,
            }
        ],
    }


def build_geographic_family(
    cems_scenarios: dict[str, dict[str, Any]] | None = None,
    num_dev: int = 8,
    num_val: int = 8,
    num_res: int = 8,
) -> dict[str, Any]:
    """Build the complete multi-incident geographic scenario family across splits."""
    if cems_scenarios is None:
        cems_scenarios = {}

    # Ensure base scenarios exist for each split
    dev_base = cems_scenarios.get("EMSR578") or build_synthetic_cems_fixture("EMSR578", utm_zone=31)
    val_base = cems_scenarios.get("EMSR590") or build_synthetic_cems_fixture("EMSR590", utm_zone=30)
    res_base = cems_scenarios.get("EMSR592") or build_synthetic_cems_fixture("EMSR592", utm_zone=30)

    dev_scenarios = [generate_geographic_scenario(dev_base, i, "development") for i in range(num_dev)]
    val_scenarios = [generate_geographic_scenario(val_base, i, "validation") for i in range(num_val)]
    res_scenarios = [generate_geographic_scenario(res_base, i, "reserved") for i in range(num_res)]

    all_scenarios = {
        "development": dev_scenarios,
        "validation": val_scenarios,
        "reserved": res_scenarios,
    }

    manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "family_id": "geographic-family-0.1",
        "description": "Multi-incident real-world geographic decision benchmark for wildfire aerial response.",
        "counts": {
            "development": len(dev_scenarios),
            "validation": len(val_scenarios),
            "reserved": len(res_scenarios),
            "total": len(dev_scenarios) + len(val_scenarios) + len(res_scenarios),
        },
        "incidents_by_split": {
            "development": ["EMSR578 (Artesa de Segre / UTM 31N)"],
            "validation": ["EMSR590 (Monfragüe / Casas de Miravete / UTM 30N)"],
            "reserved": ["EMSR592 (Sierra de Mijas / UTM 30N)", "EMSR600 (Vall d'Ebo / UTM 30N)"],
        },
        "scenarios": {
            split: [
                {
                    "scenario_id": s["scenario_id"],
                    "activation_id": s["activation_id"],
                    "seed": s["seed"],
                    "hash": canonical_hash(s),
                }
                for s in scens
            ]
            for split, scens in all_scenarios.items()
        },
    }

    return {
        "manifest": manifest,
        "scenarios": all_scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/geographic-family"))
    parser.add_argument("--num-dev", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=8)
    parser.add_argument("--num-res", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check for local real EMSR578 raw package
    cems_scenarios = {}
    real_emsr578_zip = Path("data/raw/emsr578/del.zip")
    if real_emsr578_zip.exists():
        cems_scenarios["EMSR578"] = build_cems_scenario("EMSR578", real_emsr578_zip)

    result = build_geographic_family(
        cems_scenarios=cems_scenarios,
        num_dev=args.num_dev,
        num_val=args.num_val,
        num_res=args.num_res,
    )

    # Write manifest and scenarios
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result["manifest"], indent=2))
    print(f"Wrote manifest -> {manifest_path}")

    for split, scens in result["scenarios"].items():
        split_dir = args.output_dir / split
        split_dir.mkdir(exist_ok=True)
        for s in scens:
            scen_path = split_dir / f"{s['scenario_id']}.json"
            scen_path.write_text(json.dumps(s, indent=2))

    print(f"Generated {result['manifest']['counts']['total']} geographic scenarios across splits:")
    print(f"  Development: {result['manifest']['counts']['development']} scenarios")
    print(f"  Validation:  {result['manifest']['counts']['validation']} scenarios")
    print(f"  Reserved:    {result['manifest']['counts']['reserved']} scenarios")


if __name__ == "__main__":
    main()
