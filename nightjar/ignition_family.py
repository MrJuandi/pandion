"""Derive an initial-attack family: small ignitions inside each CEMS burned area.

The geographic families start every episode from the Copernicus delineation
polygon, which already covers most of the tactical window (EMSR578: 1146 of
1600 cells), so the aircraft can change only a few percent of the outcome and
all variants of one incident share the same map. This module keeps each parent
scenario's fuels, terrain, water, weather forcing and observation schedule and
replaces the initial burn with a small square ignition (5x5 or 7x7 cells,
burnable cells only). Ignition centres are sampled among burnable cells inside
the CEMS mapped burned area, so they lie where fire actually spread; they are
*not* reported ignition origins, and the CEMS polygon is no longer a state.

Each parent yields several ignitions, so one incident provides many distinct
tactical problems. Splits and incidents are inherited unchanged from the parent
family, and reserved scenarios are written but must stay unopened.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

from .geographic_family import canonical_hash
from .geographic_ppo import load_geographic_split

SCHEMA_VERSION = "nightjar-geographic-ignition-family-0.1"
SPLITS = ("development", "validation", "reserved")
IGNITION_RADII_CELLS = (2, 3)
EDGE_MARGIN_CELLS = 8
DEFAULT_IGNITIONS = {"development": 6, "validation": 3, "reserved": 3}


def ignition_candidates(fire: dict[str, Any], margin: int = EDGE_MARGIN_CELLS) -> list[tuple[int, int]]:
    """Cells inside the CEMS burned area whose 3x3 neighbourhood is burnable, away from window edges."""
    fuel, burned = fire["fuel_grid"], fire["burn_mask"]
    rows, cols = len(fuel), len(fuel[0])
    return [
        (r, c)
        for r in range(margin, rows - margin)
        for c in range(margin, cols - margin)
        if burned[r][c] and all(int(fuel[r + dr][c + dc]) > 0 for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    ]


def ignite(parent: dict[str, Any], index: int, centre: tuple[int, int], radius: int) -> dict[str, Any]:
    """Return a child scenario whose initial burn is a square ignition around ``centre``."""
    child = deepcopy(parent)
    fire = child["fire"]
    fuel = fire["fuel_grid"]
    r0, c0 = centre
    fire["burn_mask"] = [
        [abs(r - r0) <= radius and abs(c - c0) <= radius and int(fuel[r][c]) > 0 for c in range(len(fuel[0]))]
        for r in range(len(fuel))
    ]
    child["scenario_id"] = f"{parent['scenario_id']}-ig{index:02d}"
    child["seed"] = int(parent["seed"]) * 100 + index
    child["name"] = f"{parent.get('name', 'Wildfire')} ignition {index:02d}"
    provenance = child["provenance"]
    provenance["ignition_family"] = SCHEMA_VERSION
    provenance["ignition_parent_scenario_hash"] = parent["scenario_hash"]
    provenance["cems_burn_mask_sha256"] = canonical_hash(parent["fire"]["burn_mask"])
    provenance["ignition"] = {
        "centre_row_col": [r0, c0],
        "radius_cells": radius,
        "initial_burned_cells": sum(map(sum, fire["burn_mask"])),
        "sampling": "burnable cell inside the CEMS mapped burned area; not a reported ignition origin",
    }
    provenance.setdefault("assumptions", {})["initial_state"] = "synthetic_ignition_inside_cems_burned_area"
    child.pop("scenario_hash", None)
    child["scenario_hash"] = canonical_hash(child)
    return child


def derive_split(parents: list[dict[str, Any]], ignitions_per_parent: int) -> list[dict[str, Any]]:
    children = []
    for parent in parents:
        candidates = ignition_candidates(parent["fire"])
        if len(candidates) < ignitions_per_parent:
            raise ValueError(f"{parent['scenario_id']}: only {len(candidates)} ignition candidates")
        rng = random.Random(f"{parent['scenario_hash']}:ignitions")
        for index, centre in enumerate(rng.sample(candidates, ignitions_per_parent)):
            children.append(ignite(parent, index, centre, rng.choice(IGNITION_RADII_CELLS)))
    return children


def build_ignition_family(
    scenarios_dir: Path | str,
    output_dir: Path | str,
    ignitions: dict[str, int] | None = None,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Derive and save the ignition family from a manifest-verified geographic family."""
    scenarios_dir, output_dir = Path(scenarios_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix with existing data in {output_dir}; choose a new empty directory")
    counts = {**DEFAULT_IGNITIONS, **(ignitions or {})}
    if any(counts[split] <= 0 for split in SPLITS):
        raise ValueError("ignitions per parent must be positive for every split")

    parent_manifest = json.loads((scenarios_dir / "manifest.json").read_text(encoding="utf-8"))
    derived = {split: derive_split(load_geographic_split(split, scenarios_dir, synthetic=synthetic), counts[split]) for split in SPLITS}

    manifest = deepcopy(parent_manifest)
    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "family_id": "geographic-ignition-family-0.1",
        "description": "Initial-attack scenarios: small ignitions inside CEMS burned areas on the parent family's maps.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "parent_family": {"scenarios_dir": str(scenarios_dir), "schema_version": parent_manifest.get("schema_version"),
                          "manifest_sha256": canonical_hash(parent_manifest)},
        "ignition": {"radii_cells": list(IGNITION_RADII_CELLS), "edge_margin_cells": EDGE_MARGIN_CELLS,
                     "ignitions_per_parent": counts},
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
    parser.add_argument("--scenarios-dir", type=Path, required=True, help="Parent geographic family directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    for split in SPLITS:
        parser.add_argument(f"--{split}-ignitions", type=int, default=DEFAULT_IGNITIONS[split],
                            help=f"Ignitions per {split} parent scenario")
    parser.add_argument("--synthetic", action="store_true", help="Parent is an explicitly generated synthetic family")
    args = parser.parse_args()
    result = build_ignition_family(args.scenarios_dir, args.output_dir,
                                   {split: getattr(args, f"{split}_ignitions") for split in SPLITS}, synthetic=args.synthetic)
    print(json.dumps(result["manifest"]["counts"]))


if __name__ == "__main__":
    main()
