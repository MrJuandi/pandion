"""One-time evaluation of selected PPO checkpoint on reserved discrimination scenarios.

Evaluates reserved discrimination scenarios strictly once against observation-matched
baselines (no_intervention, greedy heuristic, containment heuristic, and valid random choice)
under the predeclared improvement criteria from docs/EVALUATION.md.
Supports both the original 8-scenario opened split (v1) for comparative diagnostic and
the fresh 32-scenario held-out split (v2) for Stage 1 Gate closure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .discrimination_family import build_reserved_v2_scenarios, build_scenarios, canonical_hash
from .integrated import run_integrated
from .model import SpatialActorCritic, TACTICAL_ACTION_NAMES
from .training_env import IntegratedTrainingEnv

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SCHEMA_VERSION_V1 = "nightjar-reserved-evaluation-0.1"
SCHEMA_VERSION_V2 = "nightjar-reserved-evaluation-0.2"
DEFAULT_CHECKPOINT = Path("runs/ppo-tactical-v2.pt")
DEFAULT_OUTPUT_V1 = Path("runs/reserved-evaluation/report.json")
DEFAULT_REPLAYS_DIR_V1 = Path("runs/reserved-evaluation/replays")
DEFAULT_OUTPUT_V2 = Path("runs/reserved-v2-evaluation/report.json")
DEFAULT_REPLAYS_DIR_V2 = Path("runs/reserved-v2-evaluation/replays")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rollout_ppo_with_replay(model: Any, scenario: dict, device: Any = None) -> dict:
    """Run one deterministic PPO episode and return its complete replay dictionary."""
    action_mode = getattr(model, "action_mode", "tactical" if getattr(model, "action_dim", None) in (8, len(TACTICAL_ACTION_NAMES)) else "spatial")
    env = IntegratedTrainingEnv(scenario, action_mode=action_mode)
    observation = env.reset(seed=scenario["seed"])
    mapper = env.action_mapper()
    tensorizer = env.observation_tensorizer()
    done = False
    state = None
    is_recurrent = getattr(model, "recurrent", False)
    while not done:
        if is_recurrent:
            action_idx, next_state = model.select_action(
                observation, tensorizer, mapper, deterministic=True, state=state, device=device
            )
            state = next_state
        else:
            action_idx = model.select_action(
                observation, tensorizer, mapper, deterministic=True, device=device
            )
        observation, _reward, done, _info = env.step_discrete(action_idx)
    return env.replay


def _extract_metrics(replay: dict) -> dict:
    return {
        "affected_area_ha": replay["affected_area_ha"],
        "drops": len(replay["drops"]),
        "water_dropped_l": replay["aircraft_metrics"]["water_dropped_l"],
        "flight_s": replay["aircraft_metrics"]["flight_s"],
        "rejected_orders": replay["aircraft_metrics"]["rejected_orders"],
    }


def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Calculate non-parametric bootstrap confidence interval for the sample mean."""
    if not values:
        return {"low": 0.0, "high": 0.0, "alpha": alpha, "n_resamples": n_resamples}
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    low_idx = max(0, min(int((alpha / 2.0) * n_resamples), n_resamples - 1))
    high_idx = max(0, min(int((1.0 - alpha / 2.0) * n_resamples), n_resamples - 1))
    return {
        "low": float(boot_means[low_idx]),
        "high": float(boot_means[high_idx]),
        "alpha": alpha,
        "n_resamples": n_resamples,
    }


def _paired_comparison(ppo_list: list[dict], baseline_list: list[dict]) -> dict:
    differences = [p["affected_area_ha"] - b["affected_area_ha"] for p, b in zip(ppo_list, baseline_list)]
    reductions = [-d for d in differences]
    wins = sum(d < 0 for d in differences)
    losses = sum(d > 0 for d in differences)
    ties = sum(d == 0 for d in differences)
    count = len(differences)
    all_zero_rejected = all(p["rejected_orders"] == 0 and b["rejected_orders"] == 0
                            for p, b in zip(ppo_list, baseline_list))
    ci_reduction = _bootstrap_ci(reductions)
    ci_difference = _bootstrap_ci(differences)
    return {
        "paired_count": count,
        "mean_ppo_minus_baseline_ha": float(mean(differences)),
        "mean_reduction_ha": float(mean(reductions)),
        "reduction_95_ci": [ci_reduction["low"], ci_reduction["high"]],
        "difference_95_ci": [ci_difference["low"], ci_difference["high"]],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / count),
        "loss_rate": float(losses / count),
        "tie_rate": float(ties / count),
        "all_zero_rejected_orders": all_zero_rejected,
    }


def run_reserved_evaluation(
    checkpoint_path: Path,
    output_path: Path | None = None,
    replays_dir: Path | None = None,
    device: Any = None,
    split_version: str = "v2",
) -> dict:
    """Evaluate checkpoint and baselines on reserved scenarios (v1: 8 scenarios, v2: 32 scenarios)."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for evaluating PPO checkpoints")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint_sha = _file_sha256(checkpoint_path)
    checkpoint_data = torch.load(checkpoint_path, map_location=device or "cpu")
    model_config = checkpoint_data.get("model_config", {"grid_rows": 11, "grid_cols": 11})
    action_mode = model_config.get("action_mode", "tactical" if model_config.get("action_dim") in (8, len(TACTICAL_ACTION_NAMES)) else "spatial")
    action_dim = model_config.get("action_dim", len(TACTICAL_ACTION_NAMES) if action_mode == "tactical" else (2 + model_config["grid_rows"] * model_config["grid_cols"]))

    model = SpatialActorCritic(
        grid_rows=model_config["grid_rows"],
        grid_cols=model_config["grid_cols"],
        recurrent=model_config.get("recurrent", False),
        action_dim=action_dim,
    )
    setattr(model, "action_mode", action_mode)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    if device is not None:
        model.to(device)
    model.eval()

    if split_version == "v2":
        reserved = build_reserved_v2_scenarios()
        if len(reserved) != 32:
            raise ValueError(f"expected 32 reserved v2 scenarios, found {len(reserved)}")
        schema_version = SCHEMA_VERSION_V2
        split_name = "reserved_v2"
        default_out = DEFAULT_OUTPUT_V2
        default_rep = DEFAULT_REPLAYS_DIR_V2
    elif split_version == "v1":
        _dev, _val, reserved = build_scenarios()
        if len(reserved) != 8:
            raise ValueError(f"expected 8 reserved v1 scenarios, found {len(reserved)}")
        schema_version = SCHEMA_VERSION_V1
        split_name = "reserved"
        default_out = DEFAULT_OUTPUT_V1
        default_rep = DEFAULT_REPLAYS_DIR_V1
    else:
        raise ValueError(f"unknown split_version: {split_version}")

    final_output_path = output_path if output_path is not None else default_out
    final_replays_dir = replays_dir if replays_dir is not None else default_rep

    timestamp = datetime.now(timezone.utc).isoformat()
    scenario_provenance = [
        {"scenario_id": s["scenario_id"], "sha256": canonical_hash(s), "seed": s["seed"]}
        for s in reserved
    ]

    per_scenario_results = []
    ppo_metrics_list = []
    heuristic_metrics_list = []
    containment_metrics_list = []
    random_mean_metrics_list = []
    no_intervention_metrics_list = []

    for scenario in reserved:
        sid = scenario["scenario_id"]
        scenario_record = {"scenario_id": sid, "seed": scenario["seed"]}

        # 1. PPO deterministic evaluation
        ppo_replay = _rollout_ppo_with_replay(model, scenario, device=device)
        ppo_metrics = _extract_metrics(ppo_replay)
        ppo_metrics_list.append(ppo_metrics)
        scenario_record["ppo"] = ppo_metrics

        if final_replays_dir is not None:
            ppo_rel = Path("replays") / "ppo" / f"{sid}.json"
            full_path = final_replays_dir.parent / ppo_rel if final_replays_dir.name == "replays" else final_replays_dir / "ppo" / f"{sid}.json"
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(json.dumps(ppo_replay, indent=2, sort_keys=True) + "\n")
            scenario_record["ppo_replay"] = str(ppo_rel)

        # 2. No intervention
        no_int_replay = run_integrated(scenario, "no_intervention")
        no_int_metrics = _extract_metrics(no_int_replay)
        no_intervention_metrics_list.append(no_int_metrics)
        scenario_record["no_intervention"] = no_int_metrics

        if final_replays_dir is not None:
            no_int_rel = Path("replays") / "no_intervention" / f"{sid}.json"
            full_path = final_replays_dir.parent / no_int_rel if final_replays_dir.name == "replays" else final_replays_dir / "no_intervention" / f"{sid}.json"
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(json.dumps(no_int_replay, indent=2, sort_keys=True) + "\n")
            scenario_record["no_intervention_replay"] = str(no_int_rel)

        # 3. Downwind greedy heuristic
        heur_replay = run_integrated(scenario, "heuristic")
        heur_metrics = _extract_metrics(heur_replay)
        heuristic_metrics_list.append(heur_metrics)
        scenario_record["heuristic"] = heur_metrics

        if final_replays_dir is not None:
            heur_rel = Path("replays") / "heuristic" / f"{sid}.json"
            full_path = final_replays_dir.parent / heur_rel if final_replays_dir.name == "replays" else final_replays_dir / "heuristic" / f"{sid}.json"
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(json.dumps(heur_replay, indent=2, sort_keys=True) + "\n")
            scenario_record["heuristic_replay"] = str(heur_rel)

        # 4. Containment heuristic
        cont_replay = run_integrated(scenario, "containment_heuristic")
        cont_metrics = _extract_metrics(cont_replay)
        containment_metrics_list.append(cont_metrics)
        scenario_record["containment_heuristic"] = cont_metrics

        if final_replays_dir is not None:
            cont_rel = Path("replays") / "containment_heuristic" / f"{sid}.json"
            full_path = final_replays_dir.parent / cont_rel if final_replays_dir.name == "replays" else final_replays_dir / "containment_heuristic" / f"{sid}.json"
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(json.dumps(cont_replay, indent=2, sort_keys=True) + "\n")
            scenario_record["containment_heuristic_replay"] = str(cont_rel)

        # 5. Random valid (10 choice seeds)
        random_runs = []
        for r_seed in range(10):
            rand_replay = run_integrated(scenario, "random_valid", policy_seed=r_seed)
            rand_metrics = _extract_metrics(rand_replay)
            random_runs.append(rand_metrics)
            if final_replays_dir is not None:
                rand_rel = Path("replays") / "random_valid" / f"seed-{r_seed:02d}" / f"{sid}.json"
                full_path = final_replays_dir.parent / rand_rel if final_replays_dir.name == "replays" else final_replays_dir / "random_valid" / f"seed-{r_seed:02d}" / f"{sid}.json"
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(json.dumps(rand_replay, indent=2, sort_keys=True) + "\n")

        rand_summary = {
            "mean_affected_area_ha": float(mean(r["affected_area_ha"] for r in random_runs)),
            "min_affected_area_ha": min(r["affected_area_ha"] for r in random_runs),
            "max_affected_area_ha": max(r["affected_area_ha"] for r in random_runs),
            "mean_flight_s": float(mean(r["flight_s"] for r in random_runs)),
            "mean_water_dropped_l": float(mean(r["water_dropped_l"] for r in random_runs)),
            "rejected_orders": sum(r["rejected_orders"] for r in random_runs),
            "runs": 10,
        }
        random_mean_metrics_list.append({"affected_area_ha": rand_summary["mean_affected_area_ha"],
                                         "rejected_orders": rand_summary["rejected_orders"]})
        scenario_record["random_valid"] = rand_summary

        per_scenario_results.append(scenario_record)

    paired_heuristic = _paired_comparison(ppo_metrics_list, heuristic_metrics_list)
    paired_containment = _paired_comparison(ppo_metrics_list, containment_metrics_list)
    paired_random = _paired_comparison(ppo_metrics_list, random_mean_metrics_list)
    paired_no_int = _paired_comparison(ppo_metrics_list, no_intervention_metrics_list)

    predeclared_criteria = {
        "reduction_vs_heuristic_at_least_0_50ha": paired_heuristic["mean_reduction_ha"] >= 0.50,
        "non_inferior_vs_containment": paired_containment["mean_ppo_minus_baseline_ha"] <= 0.0,
        "heuristic_wins_at_least_50_percent": paired_heuristic["win_rate"] >= 0.50,
        "heuristic_losses_at_most_25_percent": paired_heuristic["loss_rate"] <= 0.25,
        "zero_rejected_orders": paired_heuristic["all_zero_rejected_orders"] and paired_containment["all_zero_rejected_orders"],
    }
    all_passed = all(predeclared_criteria.values())
    predeclared_criteria["all_criteria_met"] = all_passed

    report = {
        "schema_version": schema_version,
        "provenance": "synthetic",
        "split": split_name,
        "split_version": split_version,
        "synthetic_no_physical_claim": True,
        "timestamp_utc": timestamp,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "training_config": checkpoint_data.get("training_config"),
            "selected_seed": checkpoint_data.get("selected_seed"),
            "selected_candidate_index": checkpoint_data.get("selected_candidate_index"),
        },
        "scenario_provenance": scenario_provenance,
        "predeclared_criteria": predeclared_criteria,
        "paired_comparisons": {
            "heuristic": paired_heuristic,
            "containment_heuristic": paired_containment,
            "random_valid_mean": paired_random,
            "no_intervention": paired_no_int,
        },
        "per_scenario_results": per_scenario_results,
    }

    if final_output_path is not None:
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["output_report"] = str(final_output_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--replays-dir", type=Path, default=None)
    parser.add_argument("--split-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    report = run_reserved_evaluation(
        args.checkpoint,
        output_path=args.output,
        replays_dir=args.replays_dir,
        split_version=args.split_version,
    )
    rendered = json.dumps({
        "schema_version": report["schema_version"],
        "checkpoint": report["checkpoint"]["path"],
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "split": report["split"],
        "split_version": report["split_version"],
        "predeclared_criteria": report["predeclared_criteria"],
        "mean_reduction_vs_heuristic_ha": report["paired_comparisons"]["heuristic"]["mean_reduction_ha"],
        "heuristic_reduction_95_ci": report["paired_comparisons"]["heuristic"]["reduction_95_ci"],
        "mean_reduction_vs_containment_ha": report["paired_comparisons"]["containment_heuristic"]["mean_reduction_ha"],
        "containment_reduction_95_ci": report["paired_comparisons"]["containment_heuristic"]["reduction_95_ci"],
        "heuristic_win_rate": report["paired_comparisons"]["heuristic"]["win_rate"],
        "output_report": report.get("output_report"),
    }, indent=2, sort_keys=True)
    print(rendered)


if __name__ == "__main__":
    main()
