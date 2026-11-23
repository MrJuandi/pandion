"""Build and evaluate a synthetic observation-delay challenge family."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean

from .aircraft import validate_scenario as validate_aircraft
from .fire_rothermel import validate_scenario as validate_fire
from .integrated import run_integrated

SOURCE_PATH = Path(__file__).resolve().parent.parent / "examples" / "synthetic_integrated.json"
TIME_ORIGIN_UTC = "2026-01-01T00:00:00Z"
UNITS = {"time": "s", "distance": "m", "water": "L", "area": "ha", "wind_speed": "m/s"}
DEVELOPMENT_INDICES = (0, 1, 6, 9)
RESERVED_INDICES = (0, 1, 2, 3)
WIND_CHOICES = [[2, 0], [-2, 0], [0, 2], [0, -2], [3, 1], [-3, -1]]
POSITION_CHOICES = [[0, 0], [0, 500], [500, 0], [500, 500], [250, 250]]


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _one(base: dict, index: int, split: str) -> dict:
    rng = random.Random((100000 if split == "development" else 200000) + index)
    ignition = [rng.randrange(3, 8), rng.randrange(3, 8)]
    wind = rng.choice(WIND_CHOICES)
    position = rng.choice(POSITION_CHOICES)
    probability = rng.choice([.05, .12, .2])
    scenario = deepcopy(base)
    fuel = scenario["fire"]["fuel_model"]
    for row in range(11):
        for col in range(11):
            if rng.random() < probability and [row, col] != ignition:
                fuel[row][col] = "NONE"
    delay = rng.choice([30, 90, 150])
    sid = f"challenge-{split}-{index:02d}"
    scenario["scenario_id"] = sid
    scenario["fire"]["scenario_id"] = scenario["aircraft"]["scenario_id"] = sid
    seed = (100000 if split == "development" else 200000) + index
    scenario["seed"] = scenario["fire"]["seed"] = scenario["aircraft"]["seed"] = seed
    for document in (scenario, scenario["fire"], scenario["aircraft"]):
        document.update(provenance="synthetic", crs="LOCAL_METERS", time_origin_utc=TIME_ORIGIN_UTC,
                        units=deepcopy(UNITS))
    scenario["fire"].update(rows=11, cols=11, horizon_s=1200, ignitions=[ignition],
                            midflame_wind_to_m_s=wind)
    scenario["aircraft"].update(horizon_s=1200)
    scenario["aircraft"]["aircraft"]["initial_position_m"] = position
    scenario["water_effect"]["radius_m"] = 35
    scenario["observation_schedule"] = [
        {"source_id": "synthetic-survey", "observed_at_s": t,
         "available_at_s": 0 if t == 0 else t + delay, "coverage": "all"}
        for t in range(0, 1201, 60)
    ]
    scenario["generation"] = {
        "generator_version": "challenge-family-0.1",
        "generator_seed": seed,
        "blocked_cell_probability": probability,
        "blocked_cell_count": sum(value == "NONE" for row in fuel for value in row),
        "report_delay_s": delay,
    }
    validate_fire(scenario["fire"])
    validate_aircraft(scenario["aircraft"])
    if any(scenario[k] != scenario[part][k] for part in ("fire", "aircraft")
           for k in ("scenario_id", "seed", "provenance", "crs")):
        raise ValueError("integrated and component metadata must agree")
    return scenario


def build_scenarios(base: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Return development and reserved scenarios from independent indexed streams."""
    if base is None:
        base = json.loads(SOURCE_PATH.read_text())
    original = deepcopy(base)
    development = [_one(base, i, "development") for i in DEVELOPMENT_INDICES]
    reserved = [_one(base, i, "reserved") for i in RESERVED_INDICES]
    if base != original:
        raise ValueError("scenario generation mutated source input")
    return development, reserved


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metrics(replay: dict) -> dict:
    return {"affected_area_ha": replay["affected_area_ha"], "drops": len(replay["drops"]),
            "water_dropped_l": replay["aircraft_metrics"]["water_dropped_l"],
            "flight_s": replay["aircraft_metrics"]["flight_s"],
            "rejected_orders": replay["aircraft_metrics"]["rejected_orders"]}


def write_family(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    base = json.loads(SOURCE_PATH.read_text())
    source_hash = sha256(SOURCE_PATH.read_bytes()).hexdigest()
    development, reserved = build_scenarios(base)
    specs = []
    for split, scenarios, indices in (("development", development, DEVELOPMENT_INDICES),
                                       ("reserved", reserved, RESERVED_INDICES)):
        for index, scenario in zip(indices, scenarios):
            rel = Path("scenarios") / scenario["scenario_id"]
            for component in ("fire", "aircraft"):
                _write(output_dir / rel / f"{component}.json", scenario[component])
            _write(output_dir / rel / "integrated.json", scenario)
            specs.append({"scenario_id": scenario["scenario_id"], "split": split, "index": index,
                          "seed": scenario["seed"], "generation": scenario["generation"],
                          "ignition": scenario["fire"]["ignitions"][0],
                          "wind_to_m_s": scenario["fire"]["midflame_wind_to_m_s"],
                          "aircraft_position_m": scenario["aircraft"]["aircraft"]["initial_position_m"],
                          "sha256": canonical_hash(scenario),
                          "integrated_file": str(rel / "integrated.json"),
                          "fire_file": str(rel / "fire.json"), "aircraft_file": str(rel / "aircraft.json")})
    manifest = {"family_schema_version": "nightjar-challenge-family-0.1",
                "source_scenario": SOURCE_PATH.name, "source_scenario_sha256": source_hash,
                "provenance": "synthetic", "crs": "LOCAL_METERS", "time_origin_utc": TIME_ORIGIN_UTC,
                "units": deepcopy(UNITS), "grid": {"rows": 11, "cols": 11, "cell_size_m": 50},
                "horizon_s": 1200, "water_radius_m": 35,
                "development_indices": list(DEVELOPMENT_INDICES), "reserved_indices": list(RESERVED_INDICES),
                "selection_note": "Development indices were selected after exploratory inspection; development results are descriptive. Reserved cases were generated independently and their outcomes were not opened.",
                "policy_seeds": {"random_valid": list(range(10)), "no_intervention": None, "heuristic": None},
                "scenarios": specs}
    _write(output_dir / "manifest.json", manifest)
    runs = []
    for scenario in development:
        sid = scenario["scenario_id"]
        for policy in ("no_intervention", "heuristic", "random_valid"):
            seeds = range(10) if policy == "random_valid" else (None,)
            for seed in seeds:
                replay = run_integrated(scenario, policy, policy_seed=seed)
                label = f"seed-{seed:02d}" if seed is not None else "deterministic"
                rel = Path("replays") / sid / policy / f"{label}.json"
                _write(output_dir / rel, replay)
                runs.append({"scenario_id": sid, "policy": policy, "policy_seed": seed,
                             **_metrics(replay), "scenario_sha256": replay["scenario_sha256"],
                             "replay_file": str(rel)})
    report = {"report_schema_version": "nightjar-challenge-report-0.1",
              "family_schema_version": manifest["family_schema_version"],
              "provenance": "synthetic", "crs": "LOCAL_METERS", "time_origin_utc": TIME_ORIGIN_UTC,
              "units": deepcopy(UNITS),
              "interpretation": "Descriptive synthetic development results only; development cases were selected after exploratory inspection. Reserved cases are omitted and unopened.",
              "runs": runs}
    _write(output_dir / "comparison_report.json", report)
    manifest["replay_count"] = len(runs)
    manifest["comparison_report_sha256"] = sha256((output_dir / "comparison_report.json").read_bytes()).hexdigest()
    _write(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/challenge-family"))
    args = parser.parse_args()
    manifest = write_family(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "scenarios": len(manifest["scenarios"]),
                      "replays": manifest["replay_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
