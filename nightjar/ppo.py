"""Minimal PPO training path for the synthetic discrimination benchmark."""

from __future__ import annotations

import argparse
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
from .training_env import IntegratedTrainingEnv

SCHEMA_VERSION = "nightjar-ppo-smoke-0.1"


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


def train_smoke(*, updates: int = 1, episodes_per_update: int = 1, seed: int = 0) -> dict:
    """Train a small model on development scenarios only and return JSON-safe results."""
    _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    development, _, _ = build_scenarios()
    scenario = development[0]
    fuel = scenario["fire"]["fuel_model"]
    model = SpatialActorCritic(grid_rows=len(fuel), grid_cols=len(fuel[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for update in range(updates):
        records = []
        for episode in range(episodes_per_update):
            episode_records, _ = _rollout(model, development[(update * episodes_per_update + episode) % len(development)], seed + episode, None)
            records.extend(episode_records)
        losses.append(ppo_update(model, optimizer, records))
    metrics = evaluate(model, [scenario], seed=seed)
    return {"schema_version": SCHEMA_VERSION, "provenance": "synthetic", "split": "development",
            "synthetic_no_physical_claim": True, "seeds": {"training": seed},
            "config": {"updates": updates, "episodes_per_update": episodes_per_update, "clip_ratio": 0.2},
            "metrics": metrics, "losses": losses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
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
