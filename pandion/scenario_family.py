"""Generate and descriptively evaluate a fixed synthetic integrated scenario family."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .aircraft import validate_scenario as validate_aircraft
from .fire_rothermel import validate_scenario as validate_fire
from .integrated import run_integrated

FAMILY_SCHEMA_VERSION = "nightjar-scenario-family-0.1"
SOURCE_PATH = Path(__file__).resolve().parent.parent / "examples" / "synthetic_integrated.json"
POLICIES = ("no_intervention", "heuristic", "random_valid")
POLICY_SEEDS = tuple(range(10))
TIME_ORIGIN_UTC = "2026-01-01T00:00:00Z"
SCENARIO_UNITS = {"time": "s", "distance": "m", "water": "L", "area": "ha", "wind_speed": "m/s"}
# Versioned diagnostic design; each base case is paired across water radii.
VARIANT_SPECS = (
    {"variant_id": "dev-01-r35", "pair_id": "dev-01", "split": "development", "ignition": [2, 2], "wind": [2, 0], "aircraft_position_m": [0, 0], "water_radius_m": 35},
    {"variant_id": "dev-01-r55", "pair_id": "dev-01", "split": "development", "ignition": [2, 2], "wind": [2, 0], "aircraft_position_m": [0, 0], "water_radius_m": 55},
    {"variant_id": "dev-02-r35", "pair_id": "dev-02", "split": "development", "ignition": [8, 8], "wind": [0, -2], "aircraft_position_m": [500, 0], "water_radius_m": 35},
    {"variant_id": "dev-02-r55", "pair_id": "dev-02", "split": "development", "ignition": [8, 8], "wind": [0, -2], "aircraft_position_m": [500, 0], "water_radius_m": 55},
    {"variant_id": "reserved-01-r35", "pair_id": "reserved-01", "split": "reserved", "ignition": [3, 5], "wind": [0, 2], "aircraft_position_m": [0, 0], "water_radius_m": 35},
    {"variant_id": "reserved-01-r55", "pair_id": "reserved-01", "split": "reserved", "ignition": [3, 5], "wind": [0, 2], "aircraft_position_m": [0, 0], "water_radius_m": 55},
    {"variant_id": "reserved-02-r35", "pair_id": "reserved-02", "split": "reserved", "ignition": [5, 3], "wind": [-2, 0], "aircraft_position_m": [500, 500], "water_radius_m": 35},
    {"variant_id": "reserved-02-r55", "pair_id": "reserved-02", "split": "reserved", "ignition": [5, 3], "wind": [-2, 0], "aircraft_position_m": [500, 500], "water_radius_m": 55},
)


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_scenarios(base: dict | None = None) -> list[dict]:
    """Return validated independent scenario copies for the predeclared eight variants."""
    if base is None:
        base = json.loads(SOURCE_PATH.read_text())
    if len(VARIANT_SPECS) != 8 or len({spec["variant_id"] for spec in VARIANT_SPECS}) != 8:
        raise ValueError("family must contain exactly eight unique variants")
    result = []
    for spec in VARIANT_SPECS:
        scenario = deepcopy(base)
        scenario_id = f"synthetic-family-{spec['variant_id']}"
        scenario["scenario_id"] = scenario_id
        scenario["fire"]["scenario_id"] = scenario_id
        scenario["aircraft"]["scenario_id"] = scenario_id
        # Pair members share simulator seed; only the water radius changes.
        seed = 4200 + len(result) // 2
        scenario["seed"] = scenario["fire"]["seed"] = scenario["aircraft"]["seed"] = seed
        scenario["provenance"] = scenario["fire"]["provenance"] = scenario["aircraft"]["provenance"] = "synthetic"
        scenario["crs"] = scenario["fire"]["crs"] = scenario["aircraft"]["crs"] = "LOCAL_METERS"
        for document in (scenario, scenario["fire"], scenario["aircraft"]):
            document["time_origin_utc"] = TIME_ORIGIN_UTC
            document["units"] = deepcopy(SCENARIO_UNITS)
        scenario["fire"]["rows"] = scenario["fire"]["cols"] = 11
        scenario["fire"]["ignitions"] = [list(spec["ignition"])]
        scenario["fire"]["midflame_wind_to_m_s"] = list(spec["wind"])
        scenario["aircraft"]["aircraft"]["initial_position_m"] = list(spec["aircraft_position_m"])
        scenario["water_effect"]["radius_m"] = spec["water_radius_m"]
        validate_fire(scenario["fire"])
        validate_aircraft(scenario["aircraft"])
        if any(scenario[k] != scenario[part][k] for part in ("fire", "aircraft") for k in ("scenario_id", "seed", "provenance", "crs")):
            raise ValueError("integrated, fire and aircraft metadata must agree")
        result.append(scenario)
    ids = [s["scenario_id"] for s in result]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metrics(replay: dict) -> dict:
    return {
        "policy_seed": replay["policy_seed"],
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
    original = deepcopy(base)
    scenarios = build_scenarios(base)
    if base != original:
        raise ValueError("scenario generation mutated source input")
    scenario_rows = []
    for spec, scenario in zip(VARIANT_SPECS, scenarios):
        sid = scenario["scenario_id"]
        subdir = output_dir / "scenarios" / sid
        _write_json(subdir / "integrated.json", scenario)
        _write_json(subdir / "fire.json", scenario["fire"])
        _write_json(subdir / "aircraft.json", scenario["aircraft"])
        scenario_rows.append({**spec, "scenario_id": sid, "seed": scenario["seed"],
                              "sha256": canonical_hash(scenario), "fire_sha256": canonical_hash(scenario["fire"]),
                              "aircraft_sha256": canonical_hash(scenario["aircraft"]),
                              "integrated_file": str((subdir / "integrated.json").relative_to(output_dir)),
                              "fire_file": str((subdir / "fire.json").relative_to(output_dir)),
                              "aircraft_file": str((subdir / "aircraft.json").relative_to(output_dir))})
    manifest = {
        "family_schema_version": FAMILY_SCHEMA_VERSION,
        "source_scenario": str(SOURCE_PATH.name), "source_scenario_sha256": source_hash,
        "provenance": "synthetic", "crs": "LOCAL_METERS", "time_origin_utc": TIME_ORIGIN_UTC,
        "units": deepcopy(SCENARIO_UNITS),
        "grid": {"rows": 11, "cols": 11, "cell_size_m": 50},
        "policies": list(POLICIES), "policy_seeds": {"random_valid": list(POLICY_SEEDS),
                                                          "no_intervention": None, "heuristic": None},
        "interpretation": "Descriptive synthetic results only; eight scenarios do not support statistical inference. The paired design followed preliminary baseline inspection, and reserved outcomes are opened diagnostics, not an untouched policy test.",
        "variants": scenario_rows,
    }
    # Persist specs and hashes before this benchmark execution.
    _write_json(output_dir / "manifest.json", manifest)
    all_runs = []
    split_values: dict[str, list[dict]] = {"development": [], "reserved": []}
    for spec, scenario in zip(VARIANT_SPECS, scenarios):
        sid = scenario["scenario_id"]
        for policy in POLICIES:
            seeds = POLICY_SEEDS if policy == "random_valid" else (None,)
            for seed in seeds:
                replay = run_integrated(scenario, policy, policy_seed=seed)
                seed_label = f"seed-{seed:02d}" if seed is not None else "deterministic"
                relative = Path("replays") / sid / policy / f"{seed_label}.json"
                _write_json(output_dir / relative, replay)
                metrics = _metrics(replay)
                metrics["policy_seed"] = seed
                row = {"scenario_id": sid, "split": spec["split"], "policy": policy, **metrics,
                       "replay_file": str(relative)}
                all_runs.append(row)
                split_values[spec["split"]].append(row)
    def summarize(rows: list[dict]) -> dict:
        keys = ("affected_area_ha", "drops", "water_dropped_l", "flight_s", "rejected_orders")
        return {policy: {key: {"mean": mean(r[key] for r in rows if r["policy"] == policy),
                              "min": min(r[key] for r in rows if r["policy"] == policy),
                              "max": max(r[key] for r in rows if r["policy"] == policy)}
                         for key in keys} for policy in POLICIES}
    report = {"report_schema_version": "nightjar-scenario-family-report-0.1", "family_schema_version": FAMILY_SCHEMA_VERSION,
              "time_origin_utc": TIME_ORIGIN_UTC, "units": deepcopy(SCENARIO_UNITS),
              "interpretation": manifest["interpretation"], "runs": all_runs,
              "split_summaries": {split: summarize(rows) for split, rows in split_values.items()}}
    _write_json(output_dir / "comparison_report.json", report)
    manifest["comparison_report_sha256"] = sha256((output_dir / "comparison_report.json").read_bytes()).hexdigest()
    manifest["replay_count"] = len(all_runs)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/scenario-family"))
    args = parser.parse_args()
    manifest = write_family(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "variants": len(manifest["variants"]),
                      "replays": manifest["replay_count"], "manifest": str(args.output_dir / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
