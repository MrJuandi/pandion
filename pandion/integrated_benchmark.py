"""Compare three observation-matched policies in one synthetic integrated case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .integrated import run_integrated


def compare_policies(scenario: dict, random_seeds: tuple[int, ...] = tuple(range(10))) -> dict:
    if not random_seeds or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must be nonempty and unique")
    no_intervention = run_integrated(scenario, "no_intervention")
    heuristic = run_integrated(scenario, "heuristic")
    random_runs = [run_integrated(scenario, "random_valid", policy_seed=seed) for seed in random_seeds]
    baseline_area = no_intervention["affected_area_ha"]

    def entry(replay: dict) -> dict:
        return {"policy_seed": replay["policy_seed"], "affected_area_ha": replay["affected_area_ha"],
                "difference_from_no_intervention_ha": replay["affected_area_ha"] - baseline_area,
                "drops": len(replay["drops"]), "water_dropped_l": replay["aircraft_metrics"]["water_dropped_l"],
                "flight_s": replay["aircraft_metrics"]["flight_s"],
                "rejected_orders": replay["aircraft_metrics"]["rejected_orders"]}

    return {"comparison_schema_version": "integrated-synthetic-comparison-0.1",
            "scenario_id": scenario["scenario_id"], "scenario_sha256": no_intervention["scenario_sha256"],
            "provenance": "synthetic", "units": {"area": "ha", "water": "L", "flight": "s"},
            "no_intervention": entry(no_intervention), "heuristic": entry(heuristic),
            "random_valid": {"runs": [entry(run) for run in random_runs],
                             "mean_affected_area_ha": mean(run["affected_area_ha"] for run in random_runs)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--random-runs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.random_runs < 1:
        parser.error("--random-runs must be positive")
    result = compare_policies(json.loads(args.scenario.read_text()), tuple(range(args.random_runs)))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
