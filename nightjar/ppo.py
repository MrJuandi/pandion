"""Minimal PPO training path for the synthetic discrimination benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from .model import SpatialActorCritic
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised by optional-dependency environments
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    SpatialActorCritic = None  # type: ignore[assignment,misc]
    HAS_TORCH = False

from .discrimination_family import build_scenarios
from .integrated import run_integrated
from .training_env import IntegratedTrainingEnv

SCHEMA_VERSION = "nightjar-ppo-smoke-0.1"
SELECTION_SCHEMA_VERSION = "nightjar-ppo-selection-0.1"


def _stable_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _selection_key(metrics: list[dict], seed: int, index: int) -> tuple[float, int, int]:
    mean_area = sum(item["affected_area_ha"] for item in metrics) / len(metrics)
    return mean_area, seed, index


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for PPO; install requirements-rl.txt (torch>=2.0.0).")


def returns_and_advantages(rewards: list[float], values: list[float], gamma: float = 0.99) -> tuple[Any, Any]:
    """Compute discounted returns and value advantages for one terminal rollout."""
    _require_torch()
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    result = torch.tensor(returns, dtype=torch.float32)
    advantage = result - torch.tensor(values, dtype=torch.float32)
    return result, (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)


def _rollout(model: Any, scenario: dict, seed: int, device: Any) -> tuple[list[dict], dict]:
    env = IntegratedTrainingEnv(scenario)
    observation = env.reset(seed=scenario["seed"])
    mapper = env.action_mapper()
    tensorizer = env.observation_tensorizer()
    records: list[dict] = []
    rewards: list[float] = []
    values: list[float] = []
    done = False
    while not done:
        raster, vector = tensorizer.to_torch(observation, device=device)
        mask = mapper.to_mask_torch(observation, device=device)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action(raster, vector, mask)
        next_observation, reward, done, _ = env.step_discrete(int(action.item()))
        records.append({"raster": raster.squeeze(0), "vector": vector.squeeze(0), "mask": mask.squeeze(0),
                        "action": action.squeeze(0), "log_prob": log_prob.squeeze(0)})
        rewards.append(reward)
        values.append(float(value.item()))
        observation = next_observation
    returns, advantages = returns_and_advantages(rewards, values)
    for record, ret, adv in zip(records, returns, advantages):
        record["return"] = ret.to(device)
        record["advantage"] = adv.to(device)
    replay = env.replay
    metrics = {
        "affected_area_ha": replay["affected_area_ha"],
        "rejected_orders": replay["aircraft_metrics"]["rejected_orders"],
        "total_reward": float(sum(rewards)),
        "action_count": len(records),
    }
    return records, metrics


def ppo_update(model: Any, optimizer: Any, records: list[dict], clip_ratio: float = 0.2) -> float:
    """Apply one full-batch clipped PPO update and return its scalar loss."""
    _require_torch()
    raster = torch.stack([r["raster"] for r in records])
    vector = torch.stack([r["vector"] for r in records])
    masks = torch.stack([r["mask"] for r in records])
    actions = torch.stack([r["action"] for r in records])
    old_log_probs = torch.stack([r["log_prob"] for r in records]).detach()
    returns = torch.stack([r["return"] for r in records]).detach()
    advantages = torch.stack([r["advantage"] for r in records]).detach()
    log_probs, entropy, values = model.evaluate_actions(raster, vector, masks, actions)
    ratio = (log_probs - old_log_probs).exp()
    policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(model: Any, scenarios: list[dict], seed: int = 0, device: Any = None) -> list[dict]:
    """Deterministically evaluate scenarios using only the delivered observation mask."""
    _require_torch()
    model.eval()
    results = []
    with torch.no_grad():
        for index, scenario in enumerate(scenarios):
            _, metrics = _rollout_deterministic(model, scenario, seed + index, device)
            results.append({"scenario_id": scenario["scenario_id"], **metrics})
    model.train()
    return results


def _rollout_deterministic(model: Any, scenario: dict, seed: int, device: Any) -> tuple[None, dict]:
    env = IntegratedTrainingEnv(scenario)
    observation = env.reset(seed=scenario["seed"])
    mapper, tensorizer = env.action_mapper(), env.observation_tensorizer()
    rewards, action_count = [], 0
    done = False
    while not done:
        action = model.select_action(observation, tensorizer, mapper, deterministic=True, device=device)
        observation, reward, done, _ = env.step_discrete(action)
        rewards.append(reward)
        action_count += 1
    replay = env.replay
    return None, {"affected_area_ha": replay["affected_area_ha"],
                  "rejected_orders": replay["aircraft_metrics"]["rejected_orders"],
                  "total_reward": float(sum(rewards)), "action_count": action_count}


def _baseline_metrics(scenarios: list[dict]) -> list[dict]:
    """Run observation-matched baselines on exactly the supplied scenarios."""
    results = []
    for scenario in scenarios:
        case = {"scenario_id": scenario["scenario_id"]}
        for policy in ("heuristic", "containment_heuristic"):
            replay = run_integrated(scenario, policy)
            if replay.get("scenario_id") != scenario["scenario_id"]:
                raise ValueError("baseline replay scenario_id does not match validation scenario")
            case[policy] = {"affected_area_ha": replay["affected_area_ha"],
                            "rejected_orders": replay["aircraft_metrics"]["rejected_orders"]}
        results.append(case)
    return results


def _paired_summary(metrics: list[dict], baselines: list[dict], policy: str) -> dict:
    baseline_by_id = {item["scenario_id"]: item[policy] for item in baselines}
    if len(baseline_by_id) != len(baselines):
        raise ValueError("duplicate scenario_id in baseline metrics")
    deltas = []
    wins = losses = zero_rejected = 0
    for ppo in metrics:
        scenario_id = ppo["scenario_id"]
        if scenario_id not in baseline_by_id:
            raise ValueError("PPO and baseline validation scenario_ids are not aligned")
        baseline = baseline_by_id[scenario_id]
        deltas.append(ppo["affected_area_ha"] - baseline["affected_area_ha"])
        wins += ppo["affected_area_ha"] < baseline["affected_area_ha"]
        losses += ppo["affected_area_ha"] > baseline["affected_area_ha"]
        zero_rejected += ppo["rejected_orders"] == 0
    if len(deltas) != len(baselines):
        raise ValueError("PPO and baseline validation scenario sets differ")
    mean_delta = sum(deltas) / len(deltas)
    return {"paired_count": len(deltas), "mean_ppo_minus_baseline_ha": mean_delta,
            "mean_reduction_ha": -mean_delta, "wins": wins, "losses": losses,
            "win_rate": wins / len(deltas), "loss_rate": losses / len(deltas),
            "zero_rejected_orders": zero_rejected,
            "all_zero_rejected_orders": zero_rejected == len(deltas)}


def _train_candidate(*, development: list[dict], updates: int, episodes_per_update: int, seed: int) -> tuple[Any, list[float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    fuel = development[0]["fire"]["fuel_model"]
    model = SpatialActorCritic(grid_rows=len(fuel), grid_cols=len(fuel[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for update in range(updates):
        records = []
        for episode in range(episodes_per_update):
            scenario = development[(update * episodes_per_update + episode) % len(development)]
            episode_records, _ = _rollout(model, scenario, seed + episode, None)
            records.extend(episode_records)
        losses.append(ppo_update(model, optimizer, records))
    return model, losses


def train_smoke(*, updates: int = 1, episodes_per_update: int = 1, seed: int = 0) -> dict:
    """Train a small model on development scenarios only and return JSON-safe results."""
    _require_torch()
    development, _, _ = build_scenarios()
    model, losses = _train_candidate(development=development, updates=updates,
                                     episodes_per_update=episodes_per_update, seed=seed)
    metrics = evaluate(model, [development[0]], seed=seed)
    return {"schema_version": SCHEMA_VERSION, "provenance": "synthetic", "split": "development",
            "synthetic_no_physical_claim": True, "seeds": {"training": seed},
            "config": {"updates": updates, "episodes_per_update": episodes_per_update, "clip_ratio": 0.2},
            "metrics": metrics, "losses": losses}


def train_select(*, candidates: int = 2, updates: int = 1, episodes_per_update: int = 1,
                 seed: int = 0, checkpoint: Path | None = None) -> dict:
    """Train on development and choose one checkpoint using validation only.

    Reserved scenarios are intentionally unpacked but never passed to training or evaluation.
    Ties use validation mean affected area, then candidate seed, then candidate index.
    """
    _require_torch()
    if candidates < 1:
        raise ValueError("candidates must be at least 1")
    if updates < 1 or episodes_per_update < 1:
        raise ValueError("updates and episodes_per_update must be at least 1")
    development, validation, _reserved = build_scenarios()
    baseline_metrics = _baseline_metrics(validation)
    candidate_results = []
    selected_model = None
    selected_key = None
    for index in range(candidates):
        candidate_seed = seed + index
        model, losses = _train_candidate(development=development, updates=updates,
                                         episodes_per_update=episodes_per_update, seed=candidate_seed)
        metrics = evaluate(model, validation, seed=candidate_seed)
        key = _selection_key(metrics, candidate_seed, index)
        by_id = {item["scenario_id"]: item for item in metrics}
        if set(by_id) != {item["scenario_id"] for item in baseline_metrics}:
            raise ValueError("PPO and baseline validation scenario_ids are not aligned")
        matched_metrics = []
        for baseline in baseline_metrics:
            ppo = by_id[baseline["scenario_id"]]
            matched_metrics.append({"scenario_id": baseline["scenario_id"],
                                    "ppo": ppo,
                                    "heuristic": baseline["heuristic"],
                                    "containment_heuristic": baseline["containment_heuristic"]})
        candidate_results.append({"candidate_index": index, "seed": candidate_seed,
                                  "losses": losses, "validation_metrics": matched_metrics,
                                  "validation_mean_affected_area_ha": key[0]})
        if selected_key is None or key < selected_key:
            selected_key, selected_model = key, model
    result = {"schema_version": SELECTION_SCHEMA_VERSION, "provenance": "synthetic",
              "synthetic_no_physical_claim": True, "trained_split": "development",
              "selection_split": "validation", "reserved_split": "not_evaluated",
              "seeds": {"candidate_start": seed, "selected": selected_key[1]},
              "config": {"candidates": candidates, "updates": updates,
                         "episodes_per_update": episodes_per_update, "clip_ratio": 0.2},
              "scenario_provenance": {
                  "development": [{"scenario_id": s["scenario_id"], "sha256": _stable_sha256(s)} for s in development],
                  "validation": [{"scenario_id": s["scenario_id"], "sha256": _stable_sha256(s)} for s in validation],
              },
              "selection_rule": "lowest validation mean affected_area_ha; ties by seed then candidate_index",
              "baseline_validation_metrics": baseline_metrics,
              "selected_candidate_index": selected_key[2], "candidates": candidate_results}
    selected_metrics = next(candidate["validation_metrics"] for candidate in candidate_results
                            if candidate["candidate_index"] == selected_key[2])
    selected_ppo = [item["ppo"] for item in selected_metrics]
    result["selected_paired_comparison"] = {
        "heuristic": _paired_summary(selected_ppo, baseline_metrics, "heuristic"),
        "containment_heuristic": _paired_summary(selected_ppo, baseline_metrics, "containment_heuristic"),
    }
    comparison = result["selected_paired_comparison"]
    result["predeclared_criteria"] = {
        "reduction_vs_heuristic_at_least_0_50ha": comparison["heuristic"]["mean_reduction_ha"] >= 0.50,
        "non_inferior_vs_containment": comparison["containment_heuristic"]["mean_ppo_minus_baseline_ha"] <= 0.0,
        "heuristic_wins_at_least_50_percent": comparison["heuristic"]["win_rate"] >= 0.50,
        "heuristic_losses_at_most_25_percent": comparison["heuristic"]["loss_rate"] <= 0.25,
        "zero_rejected_orders": comparison["heuristic"]["all_zero_rejected_orders"] and
                               comparison["containment_heuristic"]["all_zero_rejected_orders"],
    }
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": SELECTION_SCHEMA_VERSION,
                    "model_state_dict": selected_model.state_dict(),
                    "model_config": {"grid_rows": len(development[0]["fire"]["fuel_model"]),
                                     "grid_cols": len(development[0]["fire"]["fuel_model"][0])},
                    "training_config": result["config"],
                    "selected_candidate_index": result["selected_candidate_index"],
                    "selected_seed": result["seeds"]["selected"]}, checkpoint)
        result["checkpoint"] = str(checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select", action="store_true", help="train candidates and select on validation")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        if args.select:
            if args.output is None and args.checkpoint is None:
                parser.error("--select requires --output or --checkpoint so the selected checkpoint is saved")
            checkpoint = args.checkpoint
            if checkpoint is None and args.output:
                checkpoint = args.output.with_suffix(".pt")
            result = train_select(candidates=args.candidates, updates=args.updates,
                                  episodes_per_update=args.episodes_per_update,
                                  seed=args.seed, checkpoint=checkpoint)
        else:
            result = train_smoke(updates=args.updates, episodes_per_update=args.episodes_per_update, seed=args.seed)
    except ImportError as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
