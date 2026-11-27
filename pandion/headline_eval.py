"""Evaluate geographic checkpoints and summarize paired multi-seed results."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from multiprocessing import get_context
from pathlib import Path, PureWindowsPath
import random
import tomllib

import numpy as np

from .identity import partition_holdout
from .policy_artifact import file_hash
from .viewer import strip_local_paths

SETS = ("holdout", "development_sample")
REFERENCES = ("heuristic", "hold_aware_heuristic")
BASELINES = (*REFERENCES, "no_intervention")
_MODELS: dict = {}


def bootstrap_ci(diff: np.ndarray) -> list[float]:
    """Return the seeded percentile interval for the mean paired difference."""
    means = np.random.default_rng(0).choice(diff, size=(10000, len(diff))).mean(axis=1)
    return [round(float(value), 2) for value in np.percentile(means, [2.5, 97.5])]


def _rounded_mean(values: np.ndarray) -> float:
    return round(float(values.mean()), 2)


def _better_equal_worse(diff: np.ndarray) -> list[int]:
    return [int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum())]


def _package_version() -> str:
    try:
        return version("pandion")
    except PackageNotFoundError:
        with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as source:
            return tomllib.load(source)["project"]["version"]


def _evaluate_row(item: tuple) -> dict:
    import torch

    from .geographic_integrated import run_geographic_integrated
    from .geographic_ppo import build_policy_model, rollout_geographic

    torch.set_num_threads(1)
    report, imagery_dir, checkpoints, scenario = item
    options = {key: report[key] for key in ("input_mode", "include_ros", "include_thermal")}
    options["imagery_dir"] = imagery_dir
    row = {key: scenario[key] for key in ("scenario_id", "activation_id")}
    for name in BASELINES:
        row[name] = run_geographic_integrated(scenario, name, _record_observations=False)["affected_area_ha"]
    for label, path in checkpoints.items():
        if path not in _MODELS:
            model = build_policy_model(
                scenario, action_mode=report["action_mode"], recurrent=report["recurrent"],
                slot_features=report["slot_features"], drop_distances=tuple(report.get("drop_distances", (0,))),
                action_features=bool(report.get("action_features", False)), **options,
            )
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            _MODELS[path] = model.eval()
        with torch.no_grad():
            _, metrics = rollout_geographic(
                _MODELS[path], scenario, deterministic=True, inference_only=True,
                skip_forced=report["skip_forced"], **options,
            )
        row[label] = metrics["final_burned_ha"]
    return row


def _file_info(path: Path) -> dict:
    return {"filename": path.name, "sha256": file_hash(path)}


def _evaluate(args: argparse.Namespace) -> None:
    from .geographic_ppo import load_geographic_split, with_fleet

    checkpoints = {label: Path(path) for label, path in args.checkpoint.items()}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    development = with_fleet(
        load_geographic_split("development", args.scenarios_dir, synthetic=args.synthetic), report["fleet_size"],
    )
    train, held = partition_holdout(development, report["holdout_activation"])
    random.Random(args.sample_seed).shuffle(train)
    sets = {"holdout": held, "development_sample": train[:args.sample]}
    result = {
        "report": _file_info(args.report),
        "checkpoints": {label: _file_info(path) for label, path in checkpoints.items()},
        "package_version": _package_version(),
        "sample": args.sample,
        "sample_seed": args.sample_seed,
        "synthetic": args.synthetic,
    }
    with get_context("spawn").Pool(args.workers) as pool:
        for name, scenarios in sets.items():
            rows = pool.map(
                _evaluate_row, [(report, args.imagery_dir, checkpoints, scenario) for scenario in scenarios], chunksize=2,
            )
            columns = {key: np.array([row[key] for row in rows]) for key in (*BASELINES, *checkpoints)}
            summary = {"n": len(rows), **{ref: _rounded_mean(columns[ref]) for ref in BASELINES}}
            for label in checkpoints:
                policy = columns[label]
                entry = {"policy": _rounded_mean(policy)}
                for ref in REFERENCES:
                    diff = policy - columns[ref]
                    entry[f"minus_{ref}"] = _rounded_mean(diff)
                    entry[f"ci95_{ref}"] = bootstrap_ci(diff)
                    entry[f"better_equal_worse_{ref}"] = _better_equal_worse(diff)
                summary[label] = entry
            result[name] = {"summary": summary, "rows": rows}
            args.output.write_text(json.dumps(strip_local_paths(result), indent=1), encoding="utf-8")
            print(name, json.dumps(summary), flush=True)


def _merge_rows(evaluations: list[tuple[Path, dict]], split: str) -> list[dict]:
    merged = {}
    for index, (path, evaluation) in enumerate(evaluations):
        rows = evaluation.get(split, {}).get("rows", [])
        by_id = {row["scenario_id"]: row for row in rows}
        if not by_id or len(by_id) != len(rows):
            raise ValueError(f"{path.name}: {split} must contain unique, nonempty scenario IDs")
        if index == 0:
            merged = by_id
            continue
        if by_id.keys() != merged.keys():
            raise ValueError(f"{path.name}: mismatched scenario IDs for {split}")
        for scenario_id, row in by_id.items():
            target = merged[scenario_id]
            for key, value in row.items():
                if key in target and value != target[key]:
                    raise ValueError(f"{path.name}: conflicting {key!r} for {split} scenario {scenario_id!r}")
                target[key] = value
    return [merged[scenario_id] for scenario_id in sorted(merged)]


def _summarize(args: argparse.Namespace) -> None:
    policies = {name: labels.split(",") for name, labels in args.policy.items()}
    for name, labels in policies.items():
        if any(not label or label in (*BASELINES, "scenario_id", "activation_id") for label in labels):
            raise ValueError(f"{name}: supply checkpoint labels as a comma-separated list")
        if len(set(labels)) != len(labels):
            raise ValueError(f"{name}: checkpoint labels must be unique within each policy")
    if args.pair:
        left, right = args.pair
        if left not in policies or right not in policies or len(policies[left]) != len(policies[right]):
            raise ValueError("--pair requires two named policies with the same number of seeds")
        pair_name = f"{left}_minus_{right}"
        if pair_name in policies:
            raise ValueError(f"pair name conflicts with policy {pair_name!r}")

    evaluations = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.evaluation]
    result = {}
    for split in SETS:
        rows = _merge_rows(evaluations, split)
        labels = {label for names in policies.values() for label in names}
        for label in sorted(labels):
            if any(label not in row for row in rows):
                raise ValueError(f"missing checkpoint label {label!r} in {split}")
        columns = {key: np.array([row[key] for row in rows]) for key in (*REFERENCES, *labels)}
        summary = {"n": len(rows)}
        for name, seeds in policies.items():
            policy = np.mean([columns[label] for label in seeds], axis=0)
            entry = {"policy": _rounded_mean(policy)}
            for ref in REFERENCES:
                reference = columns[ref]
                diff = policy - reference
                entry[f"minus_{ref}"] = _rounded_mean(diff)
                entry[f"ci95_{ref}"] = bootstrap_ci(diff)
                entry[f"per_seed_minus_{ref}"] = {label: _rounded_mean(columns[label] - reference) for label in seeds}
                entry[f"bew_{ref}"] = _better_equal_worse(diff)
            summary[name] = entry
        if args.pair:
            per_seed = {a: columns[a] - columns[b] for a, b in zip(policies[left], policies[right], strict=True)}
            diff = np.mean(list(per_seed.values()), axis=0)
            summary[pair_name] = {
                "mean": _rounded_mean(diff),
                "ci95": bootstrap_ci(diff),
                "per_seed": {label: _rounded_mean(values) for label, values in per_seed.items()},
            }
        result[split] = summary
    serialized = json.dumps(result, indent=1)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


def _named_values(values: list[str], reserved: tuple[str, ...]) -> dict[str, str]:
    parsed = {}
    for value in values:
        label, _, item = value.partition("=")
        if not label or not item or label in reserved or Path(label).is_absolute() or PureWindowsPath(label).is_absolute():
            raise ValueError(f"expected a non-reserved label followed by '=VALUE': {value!r}")
        if label in parsed:
            raise ValueError(f"duplicate label {label!r}")
        parsed[label] = item
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate checkpoints on holdout and sampled development cases")
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument("--scenarios-dir", type=Path, required=True)
    evaluate.add_argument("--imagery-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--sample", type=int, default=120)
    evaluate.add_argument("--sample-seed", type=int, default=0)
    evaluate.add_argument("--synthetic", action="store_true", help="Use a manifest-verified synthetic family")
    evaluate.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Aggregate paired results across policy seeds")
    summarize.add_argument("--evaluation", action="append", type=Path, required=True)
    summarize.add_argument("--policy", action="append", required=True, metavar="NAME=L1,L2,L3")
    summarize.add_argument("--pair", nargs=2, metavar=("NAME_A", "NAME_B"), help="Pair seeds in the supplied label order")
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.workers < 1 or args.sample < 1:
            evaluate.error("--workers and --sample must be positive")
        try:
            args.checkpoint = _named_values(args.checkpoint, (*BASELINES, "n", "scenario_id", "activation_id"))
        except ValueError as exc:
            evaluate.error(str(exc))
        _evaluate(args)
    else:
        try:
            args.policy = _named_values(args.policy, ("n",))
        except ValueError as exc:
            summarize.error(str(exc))
        _summarize(args)


if __name__ == "__main__":
    main()
