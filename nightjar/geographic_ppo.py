"""Reinforcement learning pipeline for real-world geographic wildfire suppression.

Implements PPO training, model selection, and evaluation for the Canadair CL-415
using Tactical Action Abstraction (9 macro-actions), Recurrent LSTM memory,
and real-world Copernicus CEMS scenarios (Scott & Burgan 40 fuels, georeferenced waterbodies).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .aircraft import CAPACITY_L
from .geographic_benchmark import run_geographic_benchmark
from .geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from .geographic_training_env import GeographicTrainingEnv
from .model import (
    HAS_TORCH,
    SpatialActorCritic,
    TACTICAL_ACTION_NAMES,
)

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None

SMOKE_SCHEMA_VERSION = "nightjar-geographic-ppo-smoke-0.1"
SELECTION_SCHEMA_VERSION = "nightjar-geographic-ppo-selection-0.1"
EVAL_SCHEMA_VERSION = "nightjar-geographic-ppo-eval-0.1"


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for Geographic PPO; install requirements-rl.txt (torch>=2.0.0).")


def load_geographic_split(split: str, scenarios_dir: Path | str = "runs/geographic-family") -> list[dict]:
    """Load scenarios for a split from disk if available, or generate reproducible fallback fixtures."""
    scenarios_dir = Path(scenarios_dir)
    split_dir = scenarios_dir / split
    if split_dir.exists():
        scenarios = []
        for p in sorted(split_dir.glob("*.json")):
            try:
                scenarios.append(json.loads(p.read_text()))
            except Exception:
                continue
        if scenarios:
            return scenarios

    # Fallback to generating scenarios on-the-fly using reviewed CEMS activation fixtures
    act_name = "EMSR578" if split == "development" else ("EMSR590" if split == "validation" else "EMSR592")
    zone = 31 if act_name == "EMSR578" else 30
    fixture = build_synthetic_cems_fixture(act_name, utm_zone=zone, rows=35, cols=35)
    return [
        generate_geographic_scenario(
            fixture,
            index=i,
            split=split,
            window_rows=30,
            window_cols=30,
            horizon_s=1800.0,
        )
        for i in range(4)
    ]


def returns_and_advantages(rewards: list[float], values: list[float], gamma: float = 0.99) -> tuple[Any, Any]:
    """Compute discounted returns and normalized advantages for one rollout."""
    _require_torch()
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    result = torch.tensor(returns, dtype=torch.float32)
    advantage = result - torch.tensor(values, dtype=torch.float32)
    std = advantage.std(unbiased=False)
    norm_advantage = (advantage - advantage.mean()) / (std + 1e-8) if std > 1e-6 else (advantage - advantage.mean())
    return result, norm_advantage


def rollout_geographic(
    model: Any,
    scenario: dict[str, Any],
    *,
    device: Any = None,
    deterministic: bool = False,
    drop_bonus_per_cell: float = 0.25,
) -> tuple[list[dict], dict[str, Any]]:
    """Execute a single rollout of the CL-415 in a geographic scenario."""
    _require_torch()
    env = GeographicTrainingEnv(scenario, drop_bonus_per_cell=drop_bonus_per_cell, action_mode="tactical")
    observation = env.reset(seed=scenario.get("seed"))
    mapper = env.action_mapper()
    tensorizer = env.observation_tensorizer()

    records: list[dict] = []
    rewards: list[float] = []
    values: list[float] = []
    done = False
    state = None
    is_recurrent = getattr(model, "recurrent", False)

    while not done:
        raster, vector = tensorizer.to_torch(observation, device=device)
        mask = mapper.to_mask_torch(observation, device=device)

        with torch.no_grad():
            if is_recurrent:
                action, log_prob, _, value, next_state = model.get_action(
                    raster, vector, mask, deterministic=deterministic, state=state
                )
                state = next_state
            else:
                action, log_prob, _, value = model.get_action(
                    raster, vector, mask, deterministic=deterministic
                )

        next_obs, reward, done, _ = env.step_discrete(int(action.item()))

        records.append({
            "raster": raster.squeeze(0),
            "vector": vector.squeeze(0),
            "mask": mask.squeeze(0),
            "action": action.squeeze(0),
            "log_prob": log_prob.squeeze(0),
        })
        rewards.append(reward)
        values.append(float(value.item()))
        observation = next_obs

    returns, advantages = returns_and_advantages(rewards, values)
    for record, ret, adv in zip(records, returns, advantages):
        record["return"] = ret.to(device)
        record["advantage"] = adv.to(device)

    replay = env.replay
    metrics = {
        "scenario_id": scenario.get("scenario_id", "geo-scenario"),
        "initial_burned_ha": replay["outcomes"]["initial_burned_ha"],
        "final_burned_ha": replay["outcomes"]["final_burned_ha"],
        "net_expansion_ha": replay["outcomes"]["net_expansion_ha"],
        "affected_area_ha": replay["affected_area_ha"],
        "drops_count": replay["aircraft_metrics"]["drops_count"],
        "refills_count": replay["aircraft_metrics"]["refills_count"],
        "water_dropped_l": replay["aircraft_metrics"]["total_water_dropped_l"],
        "flight_distance_km": replay["aircraft_metrics"]["total_flight_distance_km"],
        "rejected_orders": replay["aircraft_metrics"]["rejected_orders"],
        "total_reward": round(float(sum(rewards)), 4),
        "steps_count": len(records),
    }
    return records, metrics


def ppo_update_geographic(
    model: Any,
    optimizer: Any,
    records: list[dict] | list[list[dict]],
    clip_ratio: float = 0.2,
) -> float:
    """Apply one clipped PPO update over rollout episodes with BPTT if recurrent."""
    _require_torch()
    if not records:
        return 0.0

    episodes = [records] if isinstance(records[0], dict) else records
    is_recurrent = getattr(model, "recurrent", False)

    if not is_recurrent:
        flat = [r for ep in episodes for r in ep]
        raster = torch.stack([r["raster"] for r in flat])
        vector = torch.stack([r["vector"] for r in flat])
        masks = torch.stack([r["mask"] for r in flat])
        actions = torch.stack([r["action"] for r in flat])
        old_log_probs = torch.stack([r["log_prob"] for r in flat]).detach()
        returns = torch.stack([r["return"] for r in flat]).detach()
        advantages = torch.stack([r["advantage"] for r in flat]).detach()

        log_probs, entropy, values = model.evaluate_actions(raster, vector, masks, actions)
    else:
        all_lps, all_ents, all_vals, all_old_lps, all_rets, all_advs = [], [], [], [], [], []
        for ep in episodes:
            if not ep:
                continue
            r_seq = torch.stack([r["raster"] for r in ep])
            v_seq = torch.stack([r["vector"] for r in ep])
            m_seq = torch.stack([r["mask"] for r in ep])
            a_seq = torch.stack([r["action"] for r in ep])

            lp, ent, val = model.evaluate_actions(r_seq, v_seq, m_seq, a_seq)
            all_lps.append(lp)
            all_ents.append(ent)
            all_vals.append(val)
            all_old_lps.append(torch.stack([r["log_prob"] for r in ep]).detach())
            all_rets.append(torch.stack([r["return"] for r in ep]).detach())
            all_advs.append(torch.stack([r["advantage"] for r in ep]).detach())

        log_probs = torch.cat(all_lps)
        entropy = torch.cat(all_ents)
        values = torch.cat(all_vals)
        old_log_probs = torch.cat(all_old_lps)
        returns = torch.cat(all_rets)
        advantages = torch.cat(all_advs)

    ratio = (log_probs - old_log_probs).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    return float(loss.item())


def evaluate_geographic_policy(
    model: Any,
    scenarios: list[dict[str, Any]],
    *,
    device: Any = None,
) -> list[dict[str, Any]]:
    """Deterministically evaluate the model policy across a set of scenarios."""
    _require_torch()
    model.eval()
    results = []
    with torch.no_grad():
        for sc in scenarios:
            _, metrics = rollout_geographic(model, sc, device=device, deterministic=True)
            results.append(metrics)
    model.train()
    return results


def train_geographic_ppo(
    dev_scenarios: list[dict[str, Any]],
    *,
    updates: int = 1,
    episodes_per_update: int = 1,
    lr: float = 3e-4,
    recurrent: bool = True,
    device: Any = None,
) -> tuple[Any, list[float]]:
    """Train a SpatialActorCritic policy on geographic development scenarios."""
    _require_torch()
    sample_grid = dev_scenarios[0]["grid"]
    rows = int(sample_grid["rows"])
    cols = int(sample_grid["cols"])

    model = SpatialActorCritic(
        grid_rows=rows,
        grid_cols=cols,
        action_dim=len(TACTICAL_ACTION_NAMES),
        recurrent=recurrent,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for update_idx in range(updates):
        batch_records = []
        for ep in range(episodes_per_update):
            scenario = random.choice(dev_scenarios)
            records, _ = rollout_geographic(model, scenario, device=device, deterministic=False)
            batch_records.append(records)

        loss = ppo_update_geographic(model, optimizer, batch_records)
        losses.append(loss)

    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a fast 1-update smoke check")
    parser.add_argument("--train", action="store_true", help="Train PPO on development scenarios")
    parser.add_argument("--select", action="store_true", help="Train candidates and select on validation")
    parser.add_argument("--eval", action="store_true", help="Evaluate checkpoint against baselines")
    parser.add_argument("--scenarios-dir", type=Path, default=Path("runs/geographic-family"))
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feedforward", action="store_true", help="Use feedforward instead of LSTM")
    parser.add_argument("--eval-scenarios", type=int, default=None, help="Limit number of validation scenarios to evaluate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _require_torch()
    device = torch.device("cpu")
    recurrent = not args.feedforward

    dev_scenarios = load_geographic_split("development", args.scenarios_dir)

    if args.smoke or (not args.train and not args.select and not args.eval):
        print(f"Running Geographic PPO smoke test (recurrent={recurrent})...")
        model, losses = train_geographic_ppo(
            dev_scenarios[:2],
            updates=1,
            episodes_per_update=1,
            recurrent=recurrent,
            device=device,
        )
        _, metrics = rollout_geographic(model, dev_scenarios[0], device=device, deterministic=True)
        report = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "recurrent": recurrent,
            "loss": losses[0],
            "evaluation_metrics": metrics,
        }
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
            print(f"Wrote smoke artifact -> {args.output}")
        else:
            print(rendered)
        return

    if args.select:
        val_scenarios = load_geographic_split("validation", args.scenarios_dir)
        if args.eval_scenarios is not None:
            val_scenarios = val_scenarios[:args.eval_scenarios]
        print(f"Training {args.candidates} candidates on {len(dev_scenarios)} development scenarios...")
        best_candidate = None
        best_mean_burned = float("inf")
        candidate_reports = []

        for c_idx in range(args.candidates):
            c_seed = 42 + c_idx
            random.seed(c_seed)
            torch.manual_seed(c_seed)

            cand_model, c_losses = train_geographic_ppo(
                dev_scenarios,
                updates=args.updates,
                episodes_per_update=args.episodes_per_update,
                lr=args.lr,
                recurrent=recurrent,
                device=device,
            )

            # Evaluate candidate on validation scenarios
            val_results = evaluate_geographic_policy(cand_model, val_scenarios, device=device)
            mean_burned = sum(r["final_burned_ha"] for r in val_results) / max(len(val_results), 1)
            total_drops = sum(r["drops_count"] for r in val_results)

            candidate_reports.append({
                "candidate_index": c_idx,
                "seed": c_seed,
                "mean_burned_ha": round(mean_burned, 2),
                "total_drops": total_drops,
                "losses": c_losses,
            })

            if mean_burned < best_mean_burned:
                best_mean_burned = mean_burned
                best_candidate = cand_model

        if args.checkpoint and best_candidate is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_candidate.state_dict(), args.checkpoint)
            print(f"Saved best candidate checkpoint -> {args.checkpoint}")

        report = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "recurrent": recurrent,
            "candidates_count": args.candidates,
            "best_mean_burned_ha": round(best_mean_burned, 2),
            "candidates": candidate_reports,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(f"Wrote selection report -> {args.output}")
        return


if __name__ == "__main__":
    main()
