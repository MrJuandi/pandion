"""Benchmark comparison across policies on real geographic scenarios.

Evaluates no_intervention, random_valid, heuristic, containment_heuristic, and
hold_aware_heuristic, measuring final burned area, hectares saved, and aircraft operational efficiency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .geographic_integrated import run_geographic_integrated

BENCHMARK_SCHEMA_VERSION = "nightjar-geographic-benchmark-0.1"


def run_geographic_benchmark(
    scenario: dict[str, Any],
    random_seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> dict[str, Any]:
    """Run all reference policies on a geographic scenario and compare outcomes."""
    # 1. No intervention (spread baseline)
    no_int = run_geographic_integrated(scenario, policy="no_intervention")
    unmitigated_ha = float(no_int["outcomes"]["final_burned_ha"])

    # 2. Downwind heuristic
    heur = run_geographic_integrated(scenario, policy="heuristic")
    heur_ha = float(heur["outcomes"]["final_burned_ha"])

    # 3. Containment heuristic
    cont = run_geographic_integrated(scenario, policy="containment_heuristic")
    cont_ha = float(cont["outcomes"]["final_burned_ha"])

    # 4. Hold-aware heuristic (stacked-drop corrections)
    hold = run_geographic_integrated(scenario, policy="hold_aware_heuristic")
    hold_ha = float(hold["outcomes"]["final_burned_ha"])

    # 5. Random valid policy (averaged across seeds)
    random_runs = [
        run_geographic_integrated(scenario, policy="random_valid", policy_seed=s)
        for s in random_seeds
    ]
    random_areas = [float(r["outcomes"]["final_burned_ha"]) for r in random_runs]
    random_mean_ha = round(mean(random_areas), 4)

    policy_results = {
        "no_intervention": {
            "final_burned_ha": unmitigated_ha,
            "area_saved_ha": 0.0,
            "drops_count": 0,
            "refills_count": 0,
            "water_l": 0,
            "flight_km": 0.0,
        },
        "random_valid_mean": {
            "final_burned_ha": random_mean_ha,
            "area_saved_ha": round(max(0.0, unmitigated_ha - random_mean_ha), 4),
            "drops_count": round(mean(r["aircraft_metrics"]["drops_count"] for r in random_runs), 1),
            "refills_count": round(mean(r["aircraft_metrics"]["refills_count"] for r in random_runs), 1),
            "water_l": round(mean(r["aircraft_metrics"]["total_water_dropped_l"] for r in random_runs), 1),
            "flight_km": round(mean(r["aircraft_metrics"]["total_flight_distance_km"] for r in random_runs), 2),
        },
        "heuristic": {
            "final_burned_ha": heur_ha,
            "area_saved_ha": round(max(0.0, unmitigated_ha - heur_ha), 4),
            "drops_count": heur["aircraft_metrics"]["drops_count"],
            "refills_count": heur["aircraft_metrics"]["refills_count"],
            "water_l": heur["aircraft_metrics"]["total_water_dropped_l"],
            "flight_km": heur["aircraft_metrics"]["total_flight_distance_km"],
        },
        "containment_heuristic": {
            "final_burned_ha": cont_ha,
            "area_saved_ha": round(max(0.0, unmitigated_ha - cont_ha), 4),
            "drops_count": cont["aircraft_metrics"]["drops_count"],
            "refills_count": cont["aircraft_metrics"]["refills_count"],
            "water_l": cont["aircraft_metrics"]["total_water_dropped_l"],
            "flight_km": cont["aircraft_metrics"]["total_flight_distance_km"],
        },
        "hold_aware_heuristic": {
            "final_burned_ha": hold_ha,
            "area_saved_ha": round(max(0.0, unmitigated_ha - hold_ha), 4),
            "drops_count": hold["aircraft_metrics"]["drops_count"],
            "refills_count": hold["aircraft_metrics"]["refills_count"],
            "water_l": hold["aircraft_metrics"]["total_water_dropped_l"],
            "flight_km": hold["aircraft_metrics"]["total_flight_distance_km"],
        },
    }

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "scenario_id": scenario.get("scenario_id", "geographic-scenario"),
        "horizon_s": no_int["horizon_s"],
        "unmitigated_ha": unmitigated_ha,
        "policies": policy_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Path to geographic scenario JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenario_data = json.loads(args.scenario.read_text())
    result = run_geographic_benchmark(scenario_data)

    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote benchmark results -> {args.output}")
    else:
        print(payload)

    print("\n================== Geographic Benchmark Summary ==================")
    print(f"Scenario: {result['scenario_id']} (Horizon: {result['horizon_s']}s)")
    print(f"{'Policy':<24} {'Burned (ha)':<14} {'Saved (ha)':<12} {'Drops':<8} {'Flight (km)':<12}")
    print("-" * 72)
    for p_name, data in result["policies"].items():
        print(f"{p_name:<24} {data['final_burned_ha']:<14.2f} {data['area_saved_ha']:<12.2f} {data['drops_count']:<8} {data['flight_km']:<12.1f}")


if __name__ == "__main__":
    main()
