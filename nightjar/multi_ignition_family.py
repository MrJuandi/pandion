"""Derive a multi-focus, shifting-wind family from a geographic environment family.

The single-ignition family (``ignition_family``, D-56) gives one small fire
under constant wind, where attacking the head is almost always right, so a
tactical policy can only learn one good default (D-59). This family keeps each
parent's map, terrain, water, aircraft and imagery grid and changes the
problem the controller faces:

- 2-4 simultaneous foci (3x3 or 5x5 cells, burnable cells only), sampled
  inside the CEMS mapped burned area at least ``MIN_SEPARATION_CELLS`` apart,
  so the aircraft must choose which fire to attack;
- wind that shifts every ``WIND_SEGMENT_S``: direction follows a random walk
  (normal steps, ``DIRECTION_STEP_DEG`` standard deviation) and speed is the
  parent speed times a factor in ``SPEED_FACTOR``, stored as synthetic
  piecewise forcing known when each interval starts. Parents that already carry
  measured station intervals (EMSR578) keep them unchanged.

Foci and wind are synthetic: they are not reported ignition origins or observed
weather. Splits and incidents are inherited; reserved scenarios are written but
must stay unopened.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import atan2, cos, hypot, radians, sin
from pathlib import Path
import random
from typing import Any

from .environment import MOISTURE_KEYS, SCHEMA_VERSION as ENVIRONMENT_SCHEMA_VERSION
from .identity import content_hash, without_generation_times
from .geographic_family import canonical_hash
from .geographic_ppo import load_geographic_split
from .ignition_family import SPLITS, ignition_candidates

SCHEMA_VERSION = "nightjar-geographic-multi-ignition-family-0.2"
FOCI_RANGE = (2, 4)
FOCUS_RADII_CELLS = (1, 2)
MIN_SEPARATION_CELLS = 6
WIND_SEGMENT_S = 900.0
DIRECTION_STEP_DEG = 45.0
SPEED_FACTOR = (0.7, 1.4)
SPEED_LIMITS_M_S = (0.5, 6.0)
DEFAULT_SCENARIOS = {"development": 4, "validation": 3, "reserved": 3}


def sample_foci(candidates: list[tuple[int, int]], count: int, rng: random.Random,
                min_separation: int = MIN_SEPARATION_CELLS) -> list[tuple[int, int]]:
    """Pick up to ``count`` centres at least ``min_separation`` cells apart (Chebyshev)."""
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if all(max(abs(r - r2), abs(c - c2)) >= min_separation for r2, c2 in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                break
    return chosen


def shifting_wind(base_wind: tuple[float, float], horizon_s: float, moisture: dict[str, float],
                  rng: random.Random) -> list[dict[str, Any]]:
    """Synthetic piecewise wind: random-walk direction, scaled speed, constant moisture."""
    speed0 = max(hypot(*base_wind), SPEED_LIMITS_M_S[0])
    direction = atan2(base_wind[1], base_wind[0]) if hypot(*base_wind) > 0 else rng.uniform(-3.14159, 3.14159)
    forcing, start = [], 0.0
    while start < horizon_s - 1e-9:
        end = min(horizon_s, start + WIND_SEGMENT_S)
        if forcing:
            direction += radians(rng.gauss(0.0, DIRECTION_STEP_DEG))
        speed = min(max(speed0 * rng.uniform(*SPEED_FACTOR), SPEED_LIMITS_M_S[0]), SPEED_LIMITS_M_S[1])
        forcing.append({
            "start_s": start, "end_s": end, "available_at_s": start,
            "midflame_wind_to_m_s": [round(speed * cos(direction), 4), round(speed * sin(direction), 4)],
            "moisture": {k: float(moisture[k]) for k in MOISTURE_KEYS},
        })
        start = end
    return forcing


def derive_scenario(parent: dict[str, Any], index: int, *, seed: int = 0) -> dict[str, Any]:
    """Child scenario with several foci and, unless the parent has station forcing, shifting wind."""
    rng = random.Random(f"{content_hash(parent)}:multi:{seed}:{index}")
    child = without_generation_times(parent)
    fire = child["fire"]
    fuel = fire["fuel_grid"]
    rows, cols = len(fuel), len(fuel[0])
    candidates = ignition_candidates(parent["fire"])
    foci = sample_foci(candidates, rng.randint(*FOCI_RANGE), rng)
    if len(foci) < FOCI_RANGE[0]:
        raise ValueError(f"{parent['scenario_id']}: cannot place {FOCI_RANGE[0]} separated foci")
    radii = [rng.choice(FOCUS_RADII_CELLS) for _ in foci]
    fire["burn_mask"] = [
        [int(fuel[r][c]) > 0 and any(abs(r - r0) <= k and abs(c - c0) <= k for (r0, c0), k in zip(foci, radii))
         for c in range(cols)]
        for r in range(rows)
    ]

    environment = child.get("environment") or {}
    if environment.get("weather_mode") == "station_intervals":
        wind_note = "parent measured station intervals kept"
    else:
        moisture = fire.get("moisture")
        if not moisture or any(k not in moisture for k in MOISTURE_KEYS):
            raise ValueError(f"{parent['scenario_id']}: parent moisture lacks {MOISTURE_KEYS}")
        forcing = shifting_wind(tuple(float(v) for v in fire["wind_to_m_s"]), float(fire["horizon_s"]), moisture, rng)
        child["environment"] = {
            **{k: v for k, v in environment.items() if k not in ("forcing", "weather_mode")},
            "schema_version": environment.get("schema_version", ENVIRONMENT_SCHEMA_VERSION),
            "weather_mode": "synthetic_intervals",
            "forcing": forcing,
            "synthetic_wind": {
                "segment_s": WIND_SEGMENT_S, "direction_step_std_deg": DIRECTION_STEP_DEG,
                "speed_factor_range": list(SPEED_FACTOR), "speed_limits_m_s": list(SPEED_LIMITS_M_S),
                "base_wind_to_m_s": list(fire["wind_to_m_s"]),
                "note": "synthetic random-walk wind; not observed weather",
            },
        }
        fire["wind_to_m_s"] = list(forcing[0]["midflame_wind_to_m_s"])
        wind_note = "synthetic shifting wind"

    child["scenario_id"] = f"{parent['scenario_id']}-mf{index:02d}"
    child["seed"] = int(parent["seed"]) * 100 + 50 + index
    child["name"] = f"{parent.get('name', 'Wildfire')} multi-focus {index:02d}"
    provenance = child["provenance"]
    provenance["multi_ignition_family"] = SCHEMA_VERSION
    provenance["experimental_seed"] = seed
    provenance["parent_content_sha256"] = content_hash(parent)
    provenance["multi_ignition_parent_scenario_hash"] = parent["scenario_hash"]
    provenance["cems_burn_mask_sha256"] = canonical_hash(parent["fire"]["burn_mask"])
    provenance["foci"] = [
        {"centre_row_col": [r, c], "radius_cells": k} for (r, c), k in zip(foci, radii)
    ]
    provenance["foci_sampling"] = ("burnable cells inside the CEMS mapped burned area, at least "
                                   f"{MIN_SEPARATION_CELLS} cells apart; not reported ignition origins")
    provenance["initial_burned_cells"] = sum(map(sum, fire["burn_mask"]))
    assumptions = provenance.setdefault("assumptions", {})
    assumptions["initial_state"] = "synthetic_multi_focus_inside_cems_burned_area"
    assumptions["wind"] = wind_note if wind_note == "synthetic shifting wind" else assumptions.get("wind", wind_note)
    child.pop("scenario_hash", None)
    child["scenario_hash"] = canonical_hash(child)
    return child


def build_multi_ignition_family(
    scenarios_dir: Path | str,
    output_dir: Path | str,
    counts: dict[str, int] | None = None,
    *,
    synthetic: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Derive and save the family from a manifest-verified geographic environment family."""
    scenarios_dir, output_dir = Path(scenarios_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix with existing data in {output_dir}; choose a new empty directory")
    counts = {**DEFAULT_SCENARIOS, **(counts or {})}
    if any(counts[split] <= 0 for split in SPLITS):
        raise ValueError("scenarios per parent must be positive for every split")
    parent_manifest = json.loads((scenarios_dir / "manifest.json").read_text(encoding="utf-8"))
    derived = {
        split: [derive_scenario(parent, i, seed=seed)
                for parent in load_geographic_split(split, scenarios_dir, synthetic=synthetic)
                for i in range(counts[split])]
        for split in SPLITS
    }
    manifest = deepcopy(parent_manifest)
    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "family_id": "geographic-multi-ignition-family-0.2",
        "description": "Initial-attack scenarios with 2-4 simultaneous foci and shifting synthetic wind on the parent family's maps.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "parent_family": {"scenarios_dir": str(scenarios_dir), "schema_version": parent_manifest.get("schema_version"),
                          "manifest_sha256": canonical_hash(parent_manifest)},
        "multi_ignition": {
            "experimental_seed": seed,
            "foci_range": list(FOCI_RANGE), "focus_radii_cells": list(FOCUS_RADII_CELLS),
            "min_separation_cells": MIN_SEPARATION_CELLS, "wind_segment_s": WIND_SEGMENT_S,
            "direction_step_std_deg": DIRECTION_STEP_DEG, "speed_factor_range": list(SPEED_FACTOR),
            "scenarios_per_parent": counts,
        },
        "counts": {**{split: len(items) for split, items in derived.items()},
                   "total": sum(len(items) for items in derived.values())},
        "scenarios": {
            split: [{"scenario_id": s["scenario_id"], "hash": s["scenario_hash"], "source_kind": s["source_kind"],
                     "activation_id": s["activation_id"], "seed": s["seed"],
                     "source_zip_sha256": s["provenance"].get("source_zip_sha256")} for s in items]
            for split, items in derived.items()
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for split, items in derived.items():
        (output_dir / split).mkdir()
        for scenario in items:
            (output_dir / split / f"{scenario['scenario_id']}.json").write_text(
                json.dumps(scenario, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "scenarios": derived}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios-dir", type=Path, required=True, help="Parent geographic environment family")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    for split in SPLITS:
        parser.add_argument(f"--{split}-scenarios", type=int, default=DEFAULT_SCENARIOS[split],
                            help=f"Scenarios per {split} parent")
    parser.add_argument("--seed", type=int, default=0, help="Explicit experimental seed (independent of generation time)")
    parser.add_argument("--synthetic", action="store_true", help="Parent is an explicitly generated synthetic family")
    args = parser.parse_args()
    result = build_multi_ignition_family(args.scenarios_dir, args.output_dir,
                                         {split: getattr(args, f"{split}_scenarios") for split in SPLITS},
                                         synthetic=args.synthetic, seed=args.seed)
    print(json.dumps(result["manifest"]["counts"]))


if __name__ == "__main__":
    main()
