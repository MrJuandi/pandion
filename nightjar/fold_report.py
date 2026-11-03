"""Summarise leave-one-incident-out selection reports against the baselines (D-71).

Each ``geographic_ppo --select --holdout-activation A`` run writes a selection
report whose ``holdout`` block scores the selected policy and the baselines on
incident A, which was used neither for training nor for selecting the weights
(D-68). This module collects such reports, one per fold and seed, and reports
per fold and pooled over folds the paired difference policy - baseline in
final burned area with a bootstrap confidence interval.

With several seeds for a fold the policy result of each scenario is first
averaged over seeds, so the interval covers scenario variation for the mean
policy; the spread of the per-seed means is reported separately. The pooled
interval resamples scenarios within each fold (stratified), so every fold keeps
its weight. Reports must share the training configuration; only the held-out
incident and the seed may differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .identity import audit_independence, content_hash, incident_group

SCHEMA_VERSION = "nightjar-fold-report-0.2"
SELECTION_SCHEMA = "nightjar-geographic-ppo-selection-0.1"
BASELINES = ("heuristic", "containment_heuristic", "no_intervention")
# Fields that define the experiment; reports that differ in any of them are not comparable.
CONFIG_FIELDS = ("recurrent", "ppo", "input_mode", "action_mode", "include_ros", "include_thermal",
                 "fleet_size", "imagery_dir", "candidates_count", "slot_features", "skip_forced", "updates", "episodes_per_update",
                 "training_settings", "device")


def load_reports(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Selection reports with a holdout block, from files or directories of ``*.json``."""
    files: list[Path] = []
    for path in map(Path, paths):
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    reports = []
    for file in files:
        report = json.loads(file.read_text())
        if report.get("schema_version") != SELECTION_SCHEMA or not report.get("holdout"):
            continue
        report["_path"] = str(file)
        reports.append(report)
    if not reports:
        raise ValueError("no selection reports with a holdout block found")
    return reports


def _config(report: dict[str, Any]) -> dict[str, Any]:
    config = {field: report.get(field) for field in CONFIG_FIELDS}
    dataset = report.get("dataset", {})
    config["scenarios_dir"] = dataset.get("scenarios_dir")
    splits = dataset.get("splits", {})
    if splits and all(e.get("scenario_hash") for entries in splits.values() for e in entries):
        config["dataset_universe_sha256"] = content_hash(sorted(
            {(e["scenario_id"], e["scenario_hash"]) for entries in splits.values() for e in entries}))
        config.pop("scenarios_dir")
        config["selection_sha256"] = content_hash(sorted(
            (e["scenario_id"], e["scenario_hash"]) for e in splits.get("validation", [])))
    if dataset.get("imagery_content_sha256"):
        config["imagery_content_sha256"] = dataset["imagery_content_sha256"]
        config.pop("imagery_dir")
    execution = report.get("execution", {})
    config["source_sha256"] = execution.get("source_sha256")
    config["runtime"] = {k: execution.get(k) for k in ("python", "packages")}
    return config


def check_comparable(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """The shared configuration; raises if reports differ in anything but fold and seed."""
    reference = _config(reports[0])
    for report in reports[1:]:
        config = _config(report)
        differing = sorted(key for key in reference if config[key] != reference[key])
        if differing:
            raise ValueError(f"{report['_path']} differs from {reports[0]['_path']} in: {', '.join(differing)}")
    seen: set[tuple[str, int]] = set()
    for report in reports:
        key = (incident_group(report["holdout_activation"]), report["seed"])
        if key in seen:
            raise ValueError(f"duplicate report for fold {key[0]} seed {key[1]}")
        seen.add(key)
    return reference


def bootstrap_ci(strata: list[np.ndarray], *, resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile interval for the mean of the pooled values, resampling within each stratum."""
    rng = np.random.default_rng(seed)
    total = sum(len(s) for s in strata)
    sums = np.zeros(resamples)
    for values in strata:
        idx = rng.integers(0, len(values), size=(resamples, len(values)))
        sums += values[idx].sum(axis=1)
    means = np.sort(sums / total)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _comparison(policy: np.ndarray, baseline: np.ndarray, strata_sizes: list[int], **ci: Any) -> dict[str, Any]:
    diff = policy - baseline
    bounds = np.cumsum([0] + strata_sizes)
    low, high = bootstrap_ci([diff[a:b] for a, b in zip(bounds[:-1], bounds[1:])], **ci)
    base_mean = float(baseline.mean())
    return {
        "mean_difference_ha": round(float(diff.mean()), 4),
        "ci95_ha": [round(low, 4), round(high, 4)],
        "relative_difference": round(float(diff.mean()) / base_mean, 4) if base_mean else None,
        "policy_better": int((diff < 0).sum()),
        "equal": int((diff == 0).sum()),
        "policy_worse": int((diff > 0).sum()),
    }


def _fold_rows(reports: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], list[float]]:
    """Scenario ids, seed-averaged policy area, baseline areas and per-seed policy means."""
    for report in reports:
        rows = report["holdout"]["rows"]
        if not rows or len({row["scenario_id"] for row in rows}) != len(rows):
            raise ValueError("empty or duplicate holdout scenario rows")
        if any(not np.isfinite(row.get(key, np.nan)) or row[key] < 0 for row in rows
               for key in ("policy_ha", *(f"{n}_ha" for n in BASELINES))):
            raise ValueError("holdout areas must be finite and nonnegative")
    by_seed = [{row["scenario_id"]: row for row in r["holdout"]["rows"]} for r in reports]
    ids = sorted(by_seed[0])
    for rows in by_seed[1:]:
        if sorted(rows) != ids:
            raise ValueError(f"fold {reports[0]['holdout_activation']}: seeds were scored on different scenarios")
    for rows in by_seed[1:]:
        for sid in ids:
            if rows[sid].get("scenario_hash") != by_seed[0][sid].get("scenario_hash"):
                raise ValueError(f"scenario content differs between seeds: {sid}")
    policy = np.array([np.mean([rows[i]["policy_ha"] for rows in by_seed]) for i in ids])
    baselines = {}
    for name in BASELINES:
        values = np.array([by_seed[0][i][f"{name}_ha"] for i in ids], dtype=float)
        for rows in by_seed[1:]:
            if not np.array_equal(values, [rows[i][f"{name}_ha"] for i in ids]):
                raise ValueError(f"fold {reports[0]['holdout_activation']}: {name} differs between seeds")
        baselines[name] = values
    seed_means = [float(np.mean([rows[i]["policy_ha"] for i in ids])) for rows in by_seed]
    return ids, policy, baselines, seed_means


def summarise(reports: list[dict[str, Any]], *, resamples: int = 10000, seed: int = 0, strict: bool = False) -> dict[str, Any]:
    if not reports or resamples <= 0:
        raise ValueError("reports and positive bootstrap resamples required")
    warnings = []
    for report in reports:
        splits = report.get("dataset", {}).get("splits", {})
        issues = audit_independence(splits)
        if issues:
            raise ValueError("partitions are not independent: " + "; ".join(issues))
        if not splits or any(not e.get("incident_group") or not e.get("content_sha256")
                             for entries in splits.values() for e in entries):
            warnings.append("Legacy dataset metadata: incident/content independence cannot be verified.")
        execution, artifact = report.get("execution", {}), report.get("artifact", {})
        if not execution.get("source_sha256") or not artifact.get("checkpoint_sha256"):
            warnings.append("Legacy report: simulator version or checkpoint association is unverified.")
        if not execution.get("python") or not execution.get("packages"):
            warnings.append("Runtime versions are unverified.")
        if any(report.get(key) is None for key in ("slot_features", "skip_forced", "updates", "episodes_per_update")):
            warnings.append("Some execution settings are absent from the legacy report.")
        if not report.get("training_settings"):
            warnings.append("Training budget/reward/selection settings are incomplete.")
        holdout_entries = {e["scenario_id"]: e for e in splits.get("holdout", [])}
        if holdout_entries:
            rows = report["holdout"]["rows"]
            if {r["scenario_id"] for r in rows} != set(holdout_entries):
                raise ValueError("holdout rows do not match dataset membership")
            for row in rows:
                entry = holdout_entries[row["scenario_id"]]
                if not row.get("scenario_hash"):
                    warnings.append("Legacy holdout rows have no scenario hashes; row-to-dataset identity is unverified.")
                elif row["scenario_hash"] != entry.get("scenario_hash"):
                    raise ValueError("holdout row hash does not match dataset")
                if entry.get("incident_group") and entry["incident_group"] != incident_group(report["holdout_activation"]):
                    raise ValueError("holdout rows belong to another incident group")
        else:
            warnings.append("Holdout membership cannot be verified from legacy report.")
        if report.get("input_mode") == "image" and not report.get("dataset", {}).get("imagery_content_sha256"):
            warnings.append("Imagery content is unverified.")
        if strict and warnings:
            raise ValueError("strict comparison requires complete provenance: " + "; ".join(sorted(set(warnings))))
    config = check_comparable(reports)
    folds: dict[str, list[dict[str, Any]]] = {}
    for report in sorted(reports, key=lambda r: (r["holdout_activation"], r["seed"])):
        folds.setdefault(incident_group(report["holdout_activation"]), []).append(report)
    ci = {"resamples": resamples, "seed": seed}
    fold_summaries, pooled_policy, pooled_base, sizes = [], [], {n: [] for n in BASELINES}, []
    for activation, fold in folds.items():
        ids, policy, baselines, seed_means = _fold_rows(fold)
        fold_summaries.append({
            "holdout_activation": activation,
            "seeds": [r["seed"] for r in fold],
            "scenarios": len(ids),
            "policy_mean_ha": round(float(policy.mean()), 4),
            "policy_seed_means_ha": [round(m, 4) for m in seed_means],
            "policy_seed_std_ha": round(float(np.std(seed_means, ddof=1)), 4) if len(seed_means) > 1 else None,
            "baselines_mean_ha": {n: round(float(v.mean()), 4) for n, v in baselines.items()},
            "versus": {n: _comparison(policy, baselines[n], [len(ids)], **ci) for n in BASELINES[:2]},
            "reports": [r["_path"] for r in fold],
        })
        pooled_policy.append(policy)
        for n in BASELINES:
            pooled_base[n].append(baselines[n])
        sizes.append(len(ids))
    policy = np.concatenate(pooled_policy)
    pooled = {
        "folds": len(fold_summaries),
        "scenarios": int(policy.size),
        "policy_mean_ha": round(float(policy.mean()), 4),
        "baselines_mean_ha": {n: round(float(np.concatenate(v).mean()), 4) for n, v in pooled_base.items()},
        "versus": {n: _comparison(policy, np.concatenate(pooled_base[n]), sizes, **ci) for n in BASELINES[:2]},
        "folds_policy_better_than_heuristic": sum(f["versus"]["heuristic"]["mean_difference_ha"] < 0
                                                  for f in fold_summaries),
    }
    # Resample independent incident groups as well as scenarios. This interval
    # describes variation across observed incidents, not across trained seeds.
    rng = np.random.default_rng(seed)
    diffs = [p - b for p, b in zip(pooled_policy, pooled_base["heuristic"])]
    draws = []
    for _ in range(resamples):
        selected = rng.integers(0, len(diffs), size=len(diffs))
        values = [diffs[i][rng.integers(0, len(diffs[i]), size=len(diffs[i]))] for i in selected]
        draws.append(float(np.concatenate(values).mean()))
    pooled["incident_bootstrap_ci95_ha"] = [round(float(v), 4) for v in np.quantile(draws, [0.025, 0.975])]
    return {"schema_version": SCHEMA_VERSION, "config": config,
            "provenance_status": "complete_metadata" if not warnings else "legacy_incomplete",
            "warnings": sorted(set(warnings)),
            "uncertainty_scope": "scenario bootstrap conditional on observed folds and mean over available seeds; incident bootstrap separately resamples groups; seed spread is descriptive",
            "bootstrap": {"resamples": resamples, "seed": seed, "confidence": 0.95,
                          "pooled_resampling": "scenarios resampled within each fold"},
            "folds": fold_summaries, "pooled": pooled}


def format_table(summary: dict[str, Any]) -> str:
    """Markdown table: one row per fold and a pooled row."""
    lines = ["| Held-out incident | Seeds | Policy ha | Heuristic ha | Containment ha | "
             "Policy - heuristic (95% CI) | Better / equal / worse |",
             "|---|---|---|---|---|---|---|"]

    def row(name: str, seeds: str, entry: dict[str, Any]) -> str:
        h = entry["versus"]["heuristic"]
        relative = f"{h['relative_difference']:+.1%}" if h["relative_difference"] is not None else "n/a"
        return (f"| {name} | {seeds} | {entry['policy_mean_ha']:.1f} | {entry['baselines_mean_ha']['heuristic']:.1f} | "
                f"{entry['baselines_mean_ha']['containment_heuristic']:.1f} | {h['mean_difference_ha']:+.2f} "
                f"[{h['ci95_ha'][0]:+.2f}, {h['ci95_ha'][1]:+.2f}] ({relative}) | "
                f"{h['policy_better']} / {h['equal']} / {h['policy_worse']} |")

    for fold in summary["folds"]:
        lines.append(row(fold["holdout_activation"], ",".join(map(str, fold["seeds"])), fold))
    pooled = summary["pooled"]
    lines.append(row(f"**Pooled ({pooled['folds']} folds)**", "", pooled))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="selection report files or directories of them")
    parser.add_argument("--output", type=Path, help="write the summary JSON here")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap random seed")
    parser.add_argument("--strict", action="store_true", help="Require content, incident and code/checkpoint provenance")
    args = parser.parse_args()
    summary = summarise(load_reports(args.paths), resamples=args.resamples, seed=args.seed, strict=args.strict)
    print(f"Provenance: {summary['provenance_status']}")
    for warning in summary["warnings"]:
        print(f"  {warning}")
    print(format_table(summary))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote fold report -> {args.output}")


if __name__ == "__main__":
    main()
