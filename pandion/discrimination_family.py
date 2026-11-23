"""Build and evaluate a synthetic decision-discrimination challenge family.

This family introduces multi-foci ignitions, asymmetric water source transit,
dynamic reporting delay, and guaranteed non-trivial fire spread to distinguish
spatial aircraft dispatch decisions from simple greedy baselines.
"""
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
FAMILY_SCHEMA_VERSION = "nightjar-discrimination-family-0.1"
REPORT_SCHEMA_VERSION = "nightjar-discrimination-report-0.1"

DEVELOPMENT_INDICES = tuple(range(8))
VALIDATION_INDICES = tuple(range(32))
RESERVED_INDICES = tuple(range(8))
RESERVED_V2_INDICES = tuple(range(32))

SPLIT_BASE_SEEDS = {
    "development": 500000,
    "validation": 600000,
    "reserved": 700000,
    "reserved_v2": 800000,
}

WIND_CHOICES = [[2, 0], [-2, 0], [0, 2], [0, -2], [3, 1], [-3, -1], [2, -2], [-2, 2]]
WATER_SOURCE_CHOICES = [[0, 250], [500, 250], [250, 0], [250, 500]]
AIRCRAFT_POSITION_CHOICES = [[0, 0], [500, 500], [0, 500], [500, 0]]

RANDOMIZED_WATER_SOURCES = [
    [0, 250], [500, 250], [250, 0], [250, 500],
    [0, 0], [500, 500], [0, 500], [500, 0],
]
RANDOMIZED_AIRCRAFT_POSITIONS = [
    [0, 0], [500, 500], [0, 500], [500, 0], [250, 250],
]
RANDOMIZED_WIND_CHOICES = [
    [2, 0], [-2, 0], [0, 2], [0, -2],
    [3, 1], [-3, -1], [2, -2], [-2, 2],
    [1, 2], [-1, -2], [3, 0], [-3, 0],
    [0, 3], [0, -3], [1, 1], [-1, -1],
]
RANDOMIZED_PROBABILITIES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
RANDOMIZED_DELAYS = [45, 60, 90, 120, 150]


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _one(base: dict, index: int, split: str) -> dict:
    seed = SPLIT_BASE_SEEDS[split] + index
    rng = random.Random(seed)

    # Multi-foci: two ignitions in separate quadrants with Manhattan distance >= 4
    ign1 = [rng.randrange(2, 6), rng.randrange(2, 6)]
    ign2 = [rng.randrange(6, 10), rng.randrange(6, 10)]
    ignitions = [ign1, ign2]

    wind = rng.choice(WIND_CHOICES)
    water_source = rng.choice(WATER_SOURCE_CHOICES)
    position = rng.choice(AIRCRAFT_POSITION_CHOICES)
    prob = rng.choice([0.05, 0.10, 0.15])
    delay = rng.choice([60, 90, 120])

    scenario = deepcopy(base)
    fuel = scenario["fire"]["fuel_model"]

    # Protect immediate 4-neighborhood of ignitions so spread is never suffocated at t=0
    protected = set()
    for r, c in ignitions:
        protected.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            protected.add((r + dr, c + dc))

    for row in range(11):
        for col in range(11):
            if (row, col) not in protected and rng.random() < prob:
                fuel[row][col] = "NONE"

    split_slug = split.replace("_", "-")
    sid = f"discrimination-{split_slug}-{index:02d}"
    scenario["scenario_id"] = sid
    scenario["fire"]["scenario_id"] = sid
    scenario["aircraft"]["scenario_id"] = sid
    scenario["seed"] = scenario["fire"]["seed"] = scenario["aircraft"]["seed"] = seed

    for document in (scenario, scenario["fire"], scenario["aircraft"]):
        document.update(
            provenance="synthetic",
            crs="LOCAL_METERS",
            time_origin_utc=TIME_ORIGIN_UTC,
            units=deepcopy(UNITS),
        )

    scenario["fire"].update(
        rows=11,
        cols=11,
        horizon_s=1200,
        ignitions=ignitions,
        midflame_wind_to_m_s=wind,
    )
    scenario["aircraft"].update(
        horizon_s=1200,
        water_source_m=water_source,
    )
    scenario["aircraft"]["aircraft"]["initial_position_m"] = position
    scenario["water_effect"]["radius_m"] = 35
    scenario["observation_schedule"] = [
        {
            "source_id": "synthetic-survey",
            "observed_at_s": t,
            "available_at_s": 0 if t == 0 else t + delay,
            "coverage": "all",
        }
        for t in range(0, 1201, 60)
    ]
    scenario["generation"] = {
        "generator_version": "discrimination-family-0.2" if "v2" in split else "discrimination-family-0.1",
        "generator_seed": seed,
        "blocked_cell_probability": prob,
        "blocked_cell_count": sum(value == "NONE" for row in fuel for value in row),
        "report_delay_s": delay,
        "ignitions": ignitions,
        "water_source_m": water_source,
    }

    validate_fire(scenario["fire"])
    validate_aircraft(scenario["aircraft"])

    for part in ("fire", "aircraft"):
        for key in ("scenario_id", "seed", "provenance", "crs", "time_origin_utc", "units"):
            if scenario[key] != scenario[part][key]:
                raise ValueError(f"integrated and {part} metadata mismatch for {key}")

    return scenario


def generate_domain_randomized_scenario(seed: int, base: dict | None = None) -> dict:
    """Generate a procedurally randomized training scenario with diverse initial conditions."""
    if base is None:
        base = json.loads(SOURCE_PATH.read_text())
    rng = random.Random(seed)

    # Multi-foci: two ignitions with Manhattan distance >= 4, placed in interior cells [1, 9]
    ign1 = [rng.randrange(1, 10), rng.randrange(1, 10)]
    while True:
        ign2 = [rng.randrange(1, 10), rng.randrange(1, 10)]
        if abs(ign1[0] - ign2[0]) + abs(ign1[1] - ign2[1]) >= 4:
            break
    ignitions = [ign1, ign2]

    wind = rng.choice(RANDOMIZED_WIND_CHOICES)
    water_source = rng.choice(RANDOMIZED_WATER_SOURCES)
    position = rng.choice(RANDOMIZED_AIRCRAFT_POSITIONS)
    prob = rng.choice(RANDOMIZED_PROBABILITIES)
    delay = rng.choice(RANDOMIZED_DELAYS)

    scenario = deepcopy(base)
    fuel = scenario["fire"]["fuel_model"]

    # Protect immediate 4-neighborhood of ignitions so spread is never suffocated at t=0
    protected = set()
    for r, c in ignitions:
        protected.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            protected.add((r + dr, c + dc))

    for row in range(11):
        for col in range(11):
            if (row, col) not in protected and rng.random() < prob:
                fuel[row][col] = "NONE"

    sid = f"discrimination-random-{seed}"
    scenario["scenario_id"] = sid
    scenario["fire"]["scenario_id"] = sid
    scenario["aircraft"]["scenario_id"] = sid
    scenario["seed"] = scenario["fire"]["seed"] = scenario["aircraft"]["seed"] = seed

    for document in (scenario, scenario["fire"], scenario["aircraft"]):
        document.update(
            provenance="synthetic",
            crs="LOCAL_METERS",
            time_origin_utc=TIME_ORIGIN_UTC,
            units=deepcopy(UNITS),
        )

    scenario["fire"].update(
        rows=11,
        cols=11,
        horizon_s=1200,
        ignitions=ignitions,
        midflame_wind_to_m_s=wind,
    )
    scenario["aircraft"].update(
        horizon_s=1200,
        water_source_m=water_source,
    )
    scenario["aircraft"]["aircraft"]["initial_position_m"] = position
    scenario["water_effect"]["radius_m"] = 35
    scenario["observation_schedule"] = [
        {
            "source_id": "synthetic-survey",
            "observed_at_s": t,
            "available_at_s": 0 if t == 0 else t + delay,
            "coverage": "all",
        }
        for t in range(0, 1201, 60)
    ]
    scenario["generation"] = {
        "generator_version": "discrimination-domain-random-0.1",
        "generator_seed": seed,
        "blocked_cell_probability": prob,
        "blocked_cell_count": sum(value == "NONE" for row in fuel for value in row),
        "report_delay_s": delay,
        "ignitions": ignitions,
        "water_source_m": water_source,
    }

    validate_fire(scenario["fire"])
    validate_aircraft(scenario["aircraft"])

    for part in ("fire", "aircraft"):
        for key in ("scenario_id", "seed", "provenance", "crs", "time_origin_utc", "units"):
            if scenario[key] != scenario[part][key]:
                raise ValueError(f"integrated and {part} metadata mismatch for {key}")

    return scenario


def build_scenarios(base: dict | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Return development, validation, and reserved scenarios from disjoint seed streams."""
    if base is None:
        base = json.loads(SOURCE_PATH.read_text())
    original = deepcopy(base)

    dev = [_one(base, i, "development") for i in DEVELOPMENT_INDICES]
    val = [_one(base, i, "validation") for i in VALIDATION_INDICES]
    res = [_one(base, i, "reserved") for i in RESERVED_INDICES]

    if base != original:
        raise ValueError("scenario generation mutated source input")

    all_scenarios = dev + val + res
    if len({s["scenario_id"] for s in all_scenarios}) != len(all_scenarios):
        raise ValueError("scenario IDs must be unique across splits")
    if {s["seed"] for s in dev} & {s["seed"] for s in val} or {s["seed"] for s in dev} & {s["seed"] for s in res}:
        raise ValueError("split seeds must be disjoint")
    if len({canonical_hash(s) for s in all_scenarios}) != len(all_scenarios):
        raise ValueError("scenario hashes must be unique")

    return dev, val, res


def build_reserved_v2_scenarios(base: dict | None = None) -> list[dict]:
    """Return 32 fresh, untouched reserved scenarios (v2) from disjoint seed stream (800000..800031)."""
    if base is None:
        base = json.loads(SOURCE_PATH.read_text())
    original = deepcopy(base)

    res_v2 = [_one(base, i, "reserved_v2") for i in RESERVED_V2_INDICES]

    if base != original:
        raise ValueError("scenario generation mutated source input")

    if len({s["scenario_id"] for s in res_v2}) != len(res_v2):
        raise ValueError("scenario IDs must be unique across reserved_v2")
    if len({canonical_hash(s) for s in res_v2}) != len(res_v2):
        raise ValueError("scenario hashes must be unique")

    dev, val, res_v1 = build_scenarios(base)
    existing_seeds = {s["seed"] for s in dev + val + res_v1}
    if existing_seeds & {s["seed"] for s in res_v2}:
        raise ValueError("reserved_v2 seeds must be disjoint from dev, val, and reserved_v1")

    return res_v2


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metrics(replay: dict) -> dict:
    return {
        "affected_area_ha": replay["affected_area_ha"],
        "drops": len(replay["drops"]),
        "water_dropped_l": replay["aircraft_metrics"]["water_dropped_l"],
        "flight_s": replay["aircraft_metrics"]["flight_s"],
        "rejected_orders": replay["aircraft_metrics"]["rejected_orders"],
    }


def write_family(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    base = json.loads(SOURCE_PATH.read_text())
    source_hash = sha256(SOURCE_PATH.read_bytes()).hexdigest()

    development, validation, reserved = build_scenarios(base)
    specs = []

    splits_data = [
        ("development", development, DEVELOPMENT_INDICES),
        ("validation", validation, VALIDATION_INDICES),
        ("reserved", reserved, RESERVED_INDICES),
    ]

    for split, scenarios, indices in splits_data:
        for index, scenario in zip(indices, scenarios):
            rel = Path("scenarios") / scenario["scenario_id"]
            for component in ("fire", "aircraft"):
                _write(output_dir / rel / f"{component}.json", scenario[component])
            _write(output_dir / rel / "integrated.json", scenario)
            specs.append({
                "scenario_id": scenario["scenario_id"],
                "split": split,
                "index": index,
                "seed": scenario["seed"],
                "generation": scenario["generation"],
                "ignitions": scenario["fire"]["ignitions"],
                "wind_to_m_s": scenario["fire"]["midflame_wind_to_m_s"],
                "aircraft_position_m": scenario["aircraft"]["aircraft"]["initial_position_m"],
                "water_source_m": scenario["aircraft"]["water_source_m"],
                "sha256": canonical_hash(scenario),
                "integrated_file": str(rel / "integrated.json"),
                "fire_file": str(rel / "fire.json"),
                "aircraft_file": str(rel / "aircraft.json"),
            })

    manifest = {
        "family_schema_version": FAMILY_SCHEMA_VERSION,
        "source_scenario": SOURCE_PATH.name,
        "source_scenario_sha256": source_hash,
        "provenance": "synthetic",
        "crs": "LOCAL_METERS",
        "time_origin_utc": TIME_ORIGIN_UTC,
        "units": deepcopy(UNITS),
        "grid": {"rows": 11, "cols": 11, "cell_size_m": 50},
        "horizon_s": 1200,
        "water_radius_m": 35,
        "development_indices": list(DEVELOPMENT_INDICES),
        "validation_indices": list(VALIDATION_INDICES),
        "reserved_indices": list(RESERVED_INDICES),
        "selection_note": (
            "Development split evaluates baseline references. Validation scenarios are held out for "
            "checkpoint selection. Reserved scenarios are generated independently and their outcomes "
            "remain strictly unopened and unexecuted."
        ),
        "policy_seeds": {
            "random_valid": list(range(10)),
            "no_intervention": None,
            "heuristic": None,
            "containment_heuristic": None,
        },
        "scenarios": specs,
    }
    _write(output_dir / "manifest.json", manifest)

    # Evaluate development split only
    runs = []
    policies = ("no_intervention", "heuristic", "containment_heuristic", "random_valid")

    for scenario in development:
        sid = scenario["scenario_id"]
        for policy in policies:
            seeds = range(10) if policy == "random_valid" else (None,)
            for seed in seeds:
                replay = run_integrated(scenario, policy, policy_seed=seed)
                label = f"seed-{seed:02d}" if seed is not None else "deterministic"
                rel = Path("replays") / sid / policy / f"{label}.json"
                _write(output_dir / rel, replay)
                runs.append({
                    "scenario_id": sid,
                    "policy": policy,
                    "policy_seed": seed,
                    **_metrics(replay),
                    "scenario_sha256": replay["scenario_sha256"],
                    "replay_file": str(rel),
                })

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "family_schema_version": manifest["family_schema_version"],
        "provenance": "synthetic",
        "crs": "LOCAL_METERS",
        "time_origin_utc": TIME_ORIGIN_UTC,
        "units": deepcopy(UNITS),
        "interpretation": (
            "Descriptive synthetic development results comparing no intervention, greedy heuristic, "
            "informed containment heuristic, and valid random choice across multi-foci scenarios. "
            "Validation and reserved outcomes are omitted and unopened."
        ),
        "runs": runs,
    }
    _write(output_dir / "comparison_report.json", report)

    manifest["replay_count"] = len(runs)
    manifest["comparison_report_sha256"] = sha256((output_dir / "comparison_report.json").read_bytes()).hexdigest()
    _write(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/discrimination-family"))
    args = parser.parse_args()
    manifest = write_family(args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "scenarios": len(manifest["scenarios"]),
        "development_replays": manifest["replay_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
