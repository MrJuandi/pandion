"""Sensitivity of reference-policy outcomes to the water-effect parameters (D-74, D-76, D-78).

Runs the no-intervention, heuristic and containment policies on a seeded sample
of geographic scenarios while the drop hold limits and the evaporation delay of
``nightjar.geographic_integrated`` are varied. It measures how much the
heuristic's benefit depends on those parameters; it does not validate them.
No training is involved and nothing is read from validation or reserved splits
unless the caller points ``--scenarios-dir`` at them.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from typing import Any

SCHEMA_VERSION = "nightjar-water-sensitivity-0.1"
POLICIES = ("heuristic", "containment_heuristic")
BASE_LIMITS = ((1, 1000.0), (3, 3200.0))  # (drops needed, intensity held in kW/m), D-74 and D-76
BIG = 1e12

# (name, hold limits, evaporation-delay scale)
VARIANTS: tuple[tuple[str, tuple[tuple[int, float], ...], float], ...] = (
    ("baseline", BASE_LIMITS, 1.0),
    ("hold x0.5", ((1, 500.0), (3, 1600.0)), 1.0),
    ("hold x2", ((1, 2000.0), (3, 6400.0)), 1.0),
    ("hold infinite", ((1, BIG), (3, BIG)), 1.0),
    ("no hold (delay only)", ((1, 0.0), (3, 0.0)), 1.0),
    ("delay x0", BASE_LIMITS, 0.0),
    ("delay x0.25", BASE_LIMITS, 0.25),
    ("hold x0.5, delay x0", ((1, 500.0), (3, 1600.0)), 0.0),
    ("hold x2, delay x0", ((1, 2000.0), (3, 6400.0)), 0.0),
)


def apply_variant(limits: tuple[tuple[int, float], ...], delay_scale: float) -> None:
    """Set the module-level water parameters of the simulator (process-local)."""
    from . import geographic_integrated as gi

    gi.DROP_HOLD_LIMITS_KW_M = limits
    gi.WATER_HEATING_KJ_KG = 4.186 * 80.0 * delay_scale
    gi.WATER_EVAPORATION_KJ_KG = 2257.0 * delay_scale


def _run_task(task: tuple[str, str, tuple, float, str, int]) -> tuple[str, str, str, float]:
    path, name, limits, scale, policy, fleet_size = task
    from .geographic_integrated import run_geographic_integrated

    apply_variant(limits, scale)
    scenario = json.loads(Path(path).read_text())
    scenario["aircraft"]["fleet_size"] = fleet_size
    result = run_geographic_integrated(scenario, policy=policy, _record_observations=False)
    return Path(path).name, name, policy, float(result["outcomes"]["final_burned_ha"])


def summarize(rows: list[tuple[str, str, str, float]]) -> list[dict[str, Any]]:
    """Per variant: mean burned area per policy, growth prevented by the heuristic, and its gap to containment."""
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for scenario, variant, policy, burned_ha in rows:
        by_key[(variant, policy)][scenario] = burned_ha
    unmitigated = by_key[("baseline", "no_intervention")]
    files = sorted(unmitigated)
    base = mean(unmitigated[f] for f in files)
    out = []
    for variant in dict.fromkeys(v for v, p in by_key if p == "heuristic"):
        heuristic = by_key[(variant, "heuristic")]
        containment = by_key[(variant, "containment_heuristic")]
        h, c = mean(heuristic[f] for f in files), mean(containment[f] for f in files)
        out.append({
            "variant": variant,
            "scenarios": len(files),
            "no_intervention_ha": round(base, 2),
            "heuristic_ha": round(h, 2),
            "containment_ha": round(c, 2),
            "growth_prevented_by_heuristic_pct": round(100.0 * (base - h) / base, 1),
            "heuristic_minus_containment_ha": round(mean(heuristic[f] - containment[f] for f in files), 2),
        })
    return out


def run_sensitivity(scenarios_dir: Path, count: int, seed: int, workers: int, fleet_size: int = 2) -> dict[str, Any]:
    files = sorted(scenarios_dir.glob("*.json"))
    random.Random(seed).shuffle(files)
    files = sorted(files[:count])
    tasks = [(str(f), "baseline", BASE_LIMITS, 1.0, "no_intervention", fleet_size) for f in files]
    for name, limits, scale in VARIANTS:
        for policy in POLICIES:
            tasks += [(str(f), name, limits, scale, policy, fleet_size) for f in files]
    started = time.monotonic()
    with get_context("spawn").Pool(workers) as pool:
        rows = list(pool.imap_unordered(_run_task, tasks, chunksize=2))
    return {
        "schema_version": SCHEMA_VERSION,
        "scenarios_dir": str(scenarios_dir),
        "sample": {"count": len(files), "seed": seed, "fleet_size": fleet_size},
        "variants": [{"name": n, "hold_limits_kw_m": list(map(list, l)), "delay_scale": s} for n, l, s in VARIANTS],
        "seconds": round(time.monotonic() - started, 1),
        "summary": summarize(rows),
        "rows": [list(r) for r in sorted(rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios_dir", type=Path, help="directory of geographic scenario JSON files (use development only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=64, help="size of the seeded random sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fleet-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    report = run_sensitivity(args.scenarios_dir, args.scenarios, args.seed, args.workers, args.fleet_size)
    args.output.write_text(json.dumps(report, indent=1))
    for row in report["summary"]:
        print(f"{row['variant']:22s} heuristic {row['heuristic_ha']:6.1f} containment {row['containment_ha']:6.1f} "
              f"prevented {row['growth_prevented_by_heuristic_pct']:5.1f}%  heuristic-containment "
              f"{row['heuristic_minus_containment_ha']:+.2f} ha")


if __name__ == "__main__":
    main()
