"""Reinforcement learning pipeline for real-world geographic wildfire suppression.

Implements PPO training, model selection, and evaluation for the Canadair CL-415
using Tactical Action Abstraction (9 macro-actions), Recurrent LSTM memory,
and real-world Copernicus CEMS scenarios (Scott & Burgan 40 fuels, georeferenced waterbodies).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from .identity import scenario_entry, partition_holdout, content_hash
from .policy_artifact import write_report
from .aircraft import CAPACITY_L
from .geographic_benchmark import run_geographic_benchmark
from .geographic_family import canonical_hash
from .cems_catalog import ACTIVATIONS, incidents_for_split
from .geographic_training_env import REWARD_REFERENCES, GeographicTrainingEnv
from .vine import choose_vine_decisions, vine_episode
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
INPUT_MODES = ("legacy", "image", "cells")
_IMAGE_CACHE: dict[tuple[str, str], Any] = {}


def make_input_tensorizer(
    scenario: dict[str, Any],
    *,
    input_mode: str = "legacy",
    imagery_dir: Path | str | None = None,
    include_ros: bool = False,
    include_thermal: bool = False,
    focus_slots: bool = False,
    action_features: bool = False,
    drop_distances: Any = (0,),
) -> Any:
    """Tensorizer for a scenario: legacy rasters, D-49 image inputs, or the no-image
    ablation ``cells`` (same per-cell/vector inputs, fire state as explicit channels).
    Fleet features are added when the scenario has more than one aircraft.
    ``action_features`` (D-82) and ``drop_distances`` apply to image and cells inputs only."""
    fleet_features = int(scenario.get("aircraft", {}).get("fleet_size", 1)) > 1
    if input_mode == "legacy":
        return GeographicTrainingEnv(scenario).observation_tensorizer()
    if input_mode == "cells":
        from .policy_inputs import GeographicInputTensorizer
        return GeographicInputTensorizer(scenario, None, include_ros=include_ros, explicit_fire=True,
                                         focus_slots=focus_slots, fleet_features=fleet_features,
                                         action_features=action_features, drop_distances=tuple(drop_distances))
    if input_mode != "image":
        raise ValueError(f"Unknown input_mode: {input_mode}")
    if imagery_dir is None:
        raise ValueError("image inputs need imagery_dir (output of pandion.imagery_acquire)")
    from .imagery_acquire import load_activation_image
    from .policy_inputs import GeographicInputTensorizer

    transform = (scenario.get("augmentation") or {}).get("transform", 0)
    key = (str(imagery_dir), scenario["activation_id"], transform)
    if key not in _IMAGE_CACHE:
        image = load_activation_image(imagery_dir, scenario)
        if transform:
            from .augmentation import augment_image
            image = augment_image(image, transform)
        _IMAGE_CACHE[key] = image
    return GeographicInputTensorizer(scenario, _IMAGE_CACHE[key], include_ros=include_ros,
                                     include_thermal=include_thermal, focus_slots=focus_slots,
                                     fleet_features=fleet_features, action_features=action_features,
                                     drop_distances=tuple(drop_distances))


def build_policy_model(
    scenario: dict[str, Any],
    *,
    input_mode: str = "legacy",
    action_mode: str = "tactical",
    recurrent: bool = True,
    imagery_dir: Path | str | None = None,
    include_ros: bool = False,
    include_thermal: bool = False,
    slot_features: bool = False,
    drop_distances: Any = (0,),
    action_features: bool = False,
) -> Any:
    """Construct the actor-critic matching an input/action configuration."""
    _require_torch()
    grid = scenario["grid"]
    rows, cols = int(grid["rows"]), int(grid["cols"])
    if input_mode == "legacy":
        if action_mode != "tactical":
            raise ValueError("legacy geographic inputs support tactical actions only")
        return SpatialActorCritic(grid_rows=rows, grid_cols=cols,
                                  action_dim=len(TACTICAL_ACTION_NAMES), recurrent=recurrent)
    from .image_model import ImageActorCritic

    tensorizer = make_input_tensorizer(scenario, input_mode=input_mode, imagery_dir=imagery_dir,
                                       include_ros=include_ros, include_thermal=include_thermal,
                                       focus_slots=action_mode == "parametric",
                                       action_features=action_features and action_mode == "parametric",
                                       drop_distances=drop_distances)
    return ImageActorCritic(cell_channels=tensorizer.num_cell_channels,
                            vector_dim=tensorizer.num_vector_features,
                            px_per_cell=max(tensorizer.px_per_cell, 1), grid_rows=rows, grid_cols=cols,
                            action_mode=action_mode, recurrent=recurrent,
                            use_image=input_mode == "image",
                            image_channels=getattr(tensorizer, "image_channels", 3),
                            slot_raw_features=slot_features and action_mode == "parametric",
                            drop_distances=tuple(drop_distances) if action_mode == "parametric" else (0,),
                            action_feature_dim=tensorizer.num_action_features)


def with_fleet(scenarios: list[dict[str, Any]], fleet_size: int) -> list[dict[str, Any]]:
    """Copies of ``scenarios`` flown by ``fleet_size`` identical aircraft (rehashed).

    Additional aircraft start at the water source unless the scenario lists
    ``aircraft.fleet_initial_positions_utm``. Size 1 returns the scenarios unchanged.
    """
    if fleet_size == 1:
        return scenarios
    if fleet_size < 1:
        raise ValueError("fleet_size must be at least 1")
    out = []
    for scenario in scenarios:
        copy = deepcopy(scenario)
        copy["aircraft"]["fleet_size"] = fleet_size
        copy.setdefault("provenance", {})["fleet_override"] = {
            "fleet_size": fleet_size, "original_scenario_hash": scenario.get("scenario_hash")}
        copy.pop("scenario_hash", None)
        copy["scenario_hash"] = canonical_hash(copy)
        out.append(copy)
    return out


def resolve_device(name: str = "cpu") -> str:
    """Map a --device choice to an available torch device string."""
    _require_torch()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available on this machine")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine")
    return name


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for Geographic PPO; install requirements-rl.txt (torch>=2.0.0).")


def load_geographic_split(
    split: str, scenarios_dir: Path | str = "runs/geographic-family-real-v2", *, synthetic: bool = False
) -> list[dict]:
    """Load a manifest-verified split; synthetic fixtures must already exist and be explicitly selected."""
    if split not in ("development", "validation", "reserved"):
        raise ValueError(f"Unknown geographic split: {split}")
    scenarios_dir = Path(scenarios_dir)
    split_dir = scenarios_dir / split
    manifest_path = scenarios_dir / "manifest.json"
    if not split_dir.is_dir():
        if synthetic:
            raise FileNotFoundError(f"Synthetic mode requires an existing explicitly generated split directory: {split_dir}")
        raise FileNotFoundError(f"Geographic split directory not found: {split_dir}")
    paths = sorted(split_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"Geographic split contains no scenario JSON files: {split_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Geographic family manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid geographic family manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid geographic family manifest object: {manifest_path}")
    manifest_scenarios = manifest.get("scenarios")
    if not isinstance(manifest_scenarios, dict):
        raise ValueError(f"Manifest has no valid scenario split mapping: {manifest_path}")
    split_entries = manifest_scenarios.get(split)
    if not isinstance(split_entries, list) or any(not isinstance(entry, dict) or "scenario_id" not in entry for entry in split_entries):
        raise ValueError(f"Invalid or missing manifest entries for split '{split}' in {manifest_path}")
    expected = {entry["scenario_id"]: entry for entry in split_entries}
    if len(expected) != len(split_entries):
        raise ValueError(f"Duplicate scenario IDs in manifest split '{split}': {manifest_path}")
    designated = set(incidents_for_split(split))
    scenarios = []
    seen_ids = set()
    for path in paths:
        try:
            scenario = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid scenario JSON {path}: {exc}") from exc
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario JSON must contain an object: {path}")
        sid = scenario.get("scenario_id")
        entry = expected.get(sid)
        if entry is None:
            raise ValueError(f"Scenario is absent from manifest: {path}")
        if sid in seen_ids:
            raise ValueError(f"Duplicate scenario ID in split directory: {sid!r} ({path})")
        seen_ids.add(sid)
        activation_id = scenario.get("activation_id")
        if (not isinstance(activation_id, str) or scenario.get("split") != split
                or activation_id != entry.get("activation_id")
                or activation_id not in designated or entry.get("activation_id") not in designated):
            raise ValueError(f"Scenario split/activation does not match requested '{split}' partition: {path}")
        actual_hash = canonical_hash({k: v for k, v in scenario.items() if k != "scenario_hash"})
        if scenario.get("scenario_hash") != actual_hash or entry.get("hash") != actual_hash:
            raise ValueError(f"Scenario embedded or manifest hash mismatch: {path}")
        kind = scenario.get("source_kind")
        provenance = scenario.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"Scenario has invalid provenance object: {path}")
        source_hash = provenance.get("source_zip_sha256")
        stage = provenance.get("source_stage")
        product = ACTIVATIONS[activation_id]["products"].get(stage) if isinstance(stage, str) else None
        valid_source = (kind == "synthetic_fixture" and source_hash is None) or (
            kind == "copernicus_cems" and product is not None
            and source_hash == product["sha256"]
            and provenance.get("source_product_code") == product["product_code"]
            and provenance.get("source_product_filename") == product["vector_filename"]
            and provenance.get("image_acquired_at_utc") == product["acquired_at_utc"]
        )
        if (kind != manifest.get("source_kind") or kind != entry.get("source_kind")
                or source_hash != entry.get("source_zip_sha256") or not valid_source
                or kind not in (("synthetic_fixture",) if synthetic else ("copernicus_cems",))):
            raise ValueError(f"Unexpected or unregistered source provenance in {path}: {kind!r}")
        scenarios.append(scenario)
    if set(expected) != {scenario["scenario_id"] for scenario in scenarios}:
        raise ValueError(f"Scenario membership does not match manifest for {split_dir}")
    return scenarios


def _dataset_report(scenarios_dir: Path | str, splits: dict[str, list[dict]]) -> dict[str, Any]:
    return {
        "scenarios_dir": str(Path(scenarios_dir)),
        "splits": {
            split: [scenario_entry(scenario) for scenario in scenarios]
            for split, scenarios in splits.items()
        },
    }


def returns_and_advantages(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    *,
    gae_lambda: float | None = None,
    normalize: bool = True,
) -> tuple[Any, Any]:
    """Returns and advantages for one terminated episode.

    ``gae_lambda=None`` (default) uses Monte Carlo discounted returns and
    advantage = return - value. Otherwise generalized advantage estimation:
    delta_t = r_t + gamma * V_{t+1} - V_t (V = 0 after termination),
    A_t = sum_k (gamma * lambda)^k delta_{t+k}, and the value target is A_t + V_t.
    ``normalize`` standardizes advantages within the episode (legacy behaviour);
    batch-level normalization is done in ``ppo_update_geographic`` instead.
    """
    _require_torch()
    value_tensor = torch.tensor(values, dtype=torch.float32)
    if gae_lambda is None:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        result = torch.tensor(returns, dtype=torch.float32)
        advantage = result - value_tensor
    else:
        advantages = [0.0] * len(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            running = delta + gamma * gae_lambda * running
            advantages[t] = running
        advantage = torch.tensor(advantages, dtype=torch.float32)
        result = advantage + value_tensor
    if not normalize or advantage.numel() == 0:
        return result, advantage
    std = advantage.std(unbiased=False)
    norm_advantage = (advantage - advantage.mean()) / (std + 1e-8) if std > 1e-6 else (advantage - advantage.mean())
    return result, norm_advantage


class RewardScaler:
    """Divide rewards by the running standard deviation of discounted returns.

    Same idea as Gymnasium's ``NormalizeReward``: keeps value targets near unit
    scale so the critic loss does not swamp the policy term. The trainer updates
    the statistics with each collected batch before computing its targets.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        self.gamma, self.epsilon = gamma, epsilon
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    @property
    def scale(self) -> float:
        if self.count < 2:
            return 1.0
        return 1.0 / ((self.m2 / self.count) ** 0.5 + self.epsilon)

    def update(self, rewards: list[float]) -> None:
        running = 0.0
        for reward in rewards:
            running = running * self.gamma + reward
            self.count += 1
            delta = running - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (running - self.mean)


def rollout_geographic(
    model: Any,
    scenario: dict[str, Any],
    *,
    device: Any = None,
    deterministic: bool = False,
    drop_bonus_per_cell: float = 0.25,
    input_mode: str = "legacy",
    imagery_dir: Path | str | None = None,
    include_ros: bool = False,
    include_thermal: bool = False,
    gamma: float = 0.99,
    gae_lambda: float | None = None,
    reward_scale: float = 1.0,
    normalize_advantages: bool = True,
    skip_forced: bool = False,
    inference_only: bool = False,
    replay_out: list[dict] | None = None,
    reward_reference: str | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Execute a single rollout of the CL-415 in a geographic scenario.

    With ``reward_reference`` (D-81) each step's reward also adds the reference
    policy's ("heuristic" or "no_intervention") burned-area growth over the same
    interval, so the episode's rewards sum to reference minus policy final area.
    With ``skip_forced`` (D-64), steps whose mask allows a single action are
    executed without querying the model and are not training records; their
    reward is added to the preceding decision (or the first one), so returns
    are unchanged while forward passes and BPTT length shrink. The recurrent
    state then advances only at real decisions, in training and evaluation alike.

    Value targets and advantages use rewards multiplied by ``reward_scale``;
    ``metrics["total_reward"]`` and each record's ``reward`` stay unscaled.

    The model's ``action_mode`` (tactical by default) selects the action mapper;
    ``input_mode`` selects legacy rasters or D-49 image inputs.
    """
    _require_torch()
    action_mode = getattr(model, "action_mode", "tactical")
    env = GeographicTrainingEnv(scenario, drop_bonus_per_cell=drop_bonus_per_cell, action_mode=action_mode,
                                drop_distances=getattr(model, "drop_distances", (0,)),
                                reward_reference=reward_reference)
    observation = env.reset(seed=scenario.get("seed"))
    mapper = env.action_mapper()
    tensorizer = make_input_tensorizer(scenario, input_mode=input_mode, imagery_dir=imagery_dir,
                                       include_ros=include_ros, include_thermal=include_thermal,
                                       focus_slots=action_mode == "parametric",
                                       action_features=getattr(model, "action_feature_dim", 0) > 0,
                                       drop_distances=getattr(model, "drop_distances", (0,)))

    records: list[dict] = []
    rewards: list[float] = []
    value_tensors: list[Any] = []
    total_reward = 0.0
    carried = 0.0  # forced-step reward seen before the first decision
    done = False
    state = None
    is_recurrent = getattr(model, "recurrent", False)

    while not done:
        if skip_forced:
            valid = mapper.compute_mask(observation)
            if sum(valid) == 1:
                observation, reward, done, _ = env.step_discrete(valid.index(True))
                total_reward += reward
                if rewards:
                    rewards[-1] += reward
                else:
                    carried += reward
                continue
        inputs = tuple(tensorizer.to_torch(observation, device=device))
        mask = mapper.to_mask_torch(observation, device=device)

        with torch.no_grad():
            if is_recurrent:
                action, log_prob, _, value, next_state = model.get_action(
                    *inputs, mask, deterministic=deterministic, state=state
                )
                state = next_state
            else:
                action, log_prob, _, value = model.get_action(
                    *inputs, mask, deterministic=deterministic
                )

        next_obs, reward, done, _ = env.step_discrete(int(action.item()))

        if not inference_only:
            records.append({
                "inputs": tuple(t.squeeze(0) for t in inputs),
                "mask": mask.squeeze(0),
                "action": action.squeeze(0),
                "log_prob": log_prob.squeeze(0),
            })
        rewards.append(reward + carried)
        carried = 0.0
        total_reward += reward
        value_tensors.append(value.squeeze(0))
        observation = next_obs

    if not inference_only:
        values = torch.stack(value_tensors).cpu().tolist() if value_tensors else []
        returns, advantages = returns_and_advantages([r * reward_scale for r in rewards], values, gamma,
                                                     gae_lambda=gae_lambda, normalize=normalize_advantages)
        returns, advantages = returns.to(device), advantages.to(device)
        for record, ret, adv, reward in zip(records, returns, advantages, rewards):
            record["return"] = ret
            record["advantage"] = adv
            record["reward"] = reward
        for record, value in zip(records, values):
            record["value"] = value

    replay = env.replay
    if replay_out is not None:
        replay_out.append(replay)
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
        "total_reward": round(float(total_reward), 4),
        "steps_count": len(records),
    }
    return records, metrics


def _stack_inputs(records: list[dict]) -> tuple:
    return tuple(torch.stack(parts) for parts in zip(*(r["inputs"] for r in records)))


def _pack_episode(records: list[dict]) -> dict:
    """Prepare immutable rollout tensors once, reused across PPO epochs."""
    packed = {"inputs": _stack_inputs(records),
              **{key: torch.stack([r[key] for r in records]).detach()
                 for key in ("mask", "action", "log_prob", "return", "advantage")}}
    branched = [(t, r["vine"]) for t, r in enumerate(records) if "vine" in r]
    if branched:
        device = packed["action"].device
        packed["vine_index"] = torch.tensor([t for t, _ in branched], dtype=torch.long, device=device)
        for key in ("action", "log_prob", "advantage"):
            packed[f"vine_{key}"] = torch.stack([v[key] for _, v in branched]).to(device).detach()
    return packed


def ppo_update_geographic(
    model: Any,
    optimizer: Any,
    records: list[dict] | list[list[dict]],
    clip_ratio: float = 0.2,
    *,
    epochs: int = 1,
    minibatches: int = 1,
    normalize_advantages: bool = False,
    target_kl: float | None = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> float:
    """Clipped PPO over rollout episodes, with BPTT if recurrent.

    Each optimizer step minimizes the mean over its steps of the clipped policy
    term + ``value_coef`` × squared value error − ``entropy_coef`` × entropy.
    Gradients are accumulated one episode at a time (sums divided by the
    minibatch step count), equal to one batched backward with one episode's
    activations in memory. Minibatches are groups of whole episodes, so
    recurrent sequences stay intact; ``epochs`` passes reuse the same rollouts.
    ``normalize_advantages`` standardizes advantages over the whole batch.
    ``target_kl`` stops further epochs once the mean approximate KL of an epoch
    exceeds 1.5 × target. The defaults (1 epoch, 1 minibatch) are the original
    single-step update. Returns the mean loss over optimizer steps.

    Branched mode (D-85): when any record carries ``"vine"`` (sampled actions, their
    log-probabilities at collection and branched advantages), the policy term is the
    clipped surrogate over those branched samples only, averaged over the samples of
    the minibatch; the value and entropy terms still cover every step.
    """
    _require_torch()
    if not records:
        return 0.0

    episodes = [ep for ep in ([records] if isinstance(records[0], dict) else records) if ep]
    if not episodes:
        return 0.0
    branched = any("vine" in r for ep in episodes for r in ep)
    adv_key = "vine" if branched else "advantage"
    adv_mean, adv_std = 0.0, 1.0
    if normalize_advantages:
        all_adv = torch.cat([r["vine"]["advantage"] if branched else r["advantage"].reshape(1)
                             for ep in episodes for r in ep if adv_key in r])
        adv_mean, adv_std = float(all_adv.mean()), float(all_adv.std(unbiased=False)) + 1e-8
    minibatches = max(1, min(minibatches, len(episodes)))
    packed = [_pack_episode(ep) for ep in episodes]
    losses = []
    for _ in range(max(1, epochs)):
        order = torch.randperm(len(episodes)).tolist() if minibatches > 1 else list(range(len(episodes)))
        groups = [order[i::minibatches] for i in range(minibatches)]
        epoch_kl = []
        for group in groups:
            group_steps = sum(len(episodes[i]) for i in group)
            group_samples = (sum(packed[i]["vine_action"].numel() for i in group if "vine_action" in packed[i])
                             if branched else group_steps)
            optimizer.zero_grad()
            group_loss = 0.0
            for index in group:
                ep = episodes[index]
                tensors = packed[index]
                inputs, masks, actions = tensors["inputs"], tensors["mask"], tensors["action"]
                old_log_probs, returns = tensors["log_prob"], tensors["return"]

                if branched:
                    all_log_probs, entropy, values = model.evaluate_log_probs(*inputs, masks)
                    if "vine_index" in tensors:
                        rows = all_log_probs[tensors["vine_index"]]
                        log_probs = rows.gather(1, tensors["vine_action"])
                        old_log_probs = tensors["vine_log_prob"]
                        advantages = (tensors["vine_advantage"] - adv_mean) / adv_std
                    else:
                        log_probs = old_log_probs = advantages = torch.zeros(0, device=all_log_probs.device)
                else:
                    log_probs, entropy, values = model.evaluate_actions(*inputs, masks, actions)
                    advantages = (tensors["advantage"] - adv_mean) / adv_std
                log_ratio = log_probs - old_log_probs
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
                policy_sum = -torch.minimum(ratio * advantages, clipped_ratio * advantages).sum()
                value_sum = ((values - returns) ** 2).sum()
                loss = (policy_sum / max(group_samples, 1)
                        + (value_coef * value_sum - entropy_coef * entropy.sum()) / group_steps)
                loss.backward()
                group_loss += float(loss.item())
                if log_ratio.numel():
                    with torch.no_grad():
                        epoch_kl.append(float(((ratio - 1.0) - log_ratio).mean()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            losses.append(group_loss)
        if target_kl is not None and epoch_kl and sum(epoch_kl) / len(epoch_kl) > 1.5 * target_kl:
            break
    return sum(losses) / len(losses)


def holdout_evaluation(
    model: Any,
    scenarios: list[dict[str, Any]],
    *,
    device: Any = None,
    **rollout_options: Any,
) -> dict[str, Any]:
    """Score a selected policy and the baselines on scenarios used neither for
    training nor for selection (D-68)."""
    policy = evaluate_geographic_policy(model, scenarios, device=device, **rollout_options)
    names = ("no_intervention", "heuristic", "containment_heuristic", "hold_aware_heuristic", "random_valid_mean")
    rows = []
    for scenario, metrics in zip(scenarios, policy):
        bench = run_geographic_benchmark(scenario, random_seeds=(42, 43))["policies"]
        rows.append({**scenario_entry(scenario), "policy_ha": metrics["final_burned_ha"],
                     **{f"{name}_ha": bench[name]["final_burned_ha"] for name in names}})
    count = max(len(rows), 1)
    return {
        "activation_ids": sorted({s["activation_id"] for s in scenarios}),
        "scenarios": len(rows),
        "policy_mean_ha": round(sum(r["policy_ha"] for r in rows) / count, 4),
        "baselines_mean_ha": {name: round(sum(r[f"{name}_ha"] for r in rows) / count, 4) for name in names},
        "policy_better_than_heuristic": sum(r["policy_ha"] < r["heuristic_ha"] for r in rows),
        "policy_equal_to_heuristic": sum(r["policy_ha"] == r["heuristic_ha"] for r in rows),
        "rows": rows,
    }


def evaluate_geographic_policy(
    model: Any,
    scenarios: list[dict[str, Any]],
    *,
    device: Any = None,
    **input_options: Any,
) -> list[dict[str, Any]]:
    """Deterministically evaluate the model policy across a set of scenarios."""
    _require_torch()
    model.eval()
    results = []
    with torch.no_grad():
        for sc in scenarios:
            _, metrics = rollout_geographic(model, sc, device=device, deterministic=True, inference_only=True, **input_options)
            results.append(metrics)
    model.train()
    return results


_WORKER_MODELS: dict[str, Any] = {}


def _rollout_task(task: tuple) -> tuple[list[dict], dict[str, Any]]:
    """Worker-process rollout on CPU with the current weights (D-64)."""
    build_kwargs, state_dict, scenario, rollout_kwargs, seed = task
    torch.set_num_threads(1)
    key = repr(sorted(build_kwargs.items()))
    model = _WORKER_MODELS.get(key)
    if model is None:
        model = build_policy_model(scenario, **build_kwargs)
        _WORKER_MODELS[key] = model
    model.load_state_dict(state_dict)
    torch.manual_seed(seed)
    kwargs = {"deterministic": False, **rollout_kwargs}
    with torch.no_grad():
        return rollout_geographic(model, scenario, device=None, **kwargs)


def _vine_task(task: tuple) -> list[dict[str, Any]]:
    """Worker-process branching of one collected episode on CPU (D-85)."""
    build_kwargs, state_dict, scenario, decision_actions, chosen, vine_kwargs = task
    torch.set_num_threads(1)
    key = repr(sorted(build_kwargs.items()))
    model = _WORKER_MODELS.get(key)
    if model is None:
        model = build_policy_model(scenario, **build_kwargs)
        _WORKER_MODELS[key] = model
    model.load_state_dict(state_dict)
    model.eval()
    return vine_episode(model, scenario, decision_actions, chosen, **vine_kwargs)


def attach_vine_samples(records: list[dict], branches: list[dict[str, Any]], device: Any = None) -> int:
    """Store branched samples on the decision records they belong to; returns the sample count.

    Each branched decision gets ``record["vine"] = {"action", "log_prob", "advantage"}``
    with the leave-one-out advantage of every sampled action, in hectares (D-85)."""
    from .vine import leave_one_out_advantages
    count = 0
    for branch in branches:
        records[branch["decision"]]["vine"] = {
            "action": torch.tensor(branch["actions"], dtype=torch.long, device=device),
            "log_prob": torch.tensor(branch["log_probs"], dtype=torch.float32, device=device),
            "advantage": torch.tensor(leave_one_out_advantages(branch["returns"]), dtype=torch.float32,
                                      device=device),
        }
        count += len(branch["actions"])
    return count


def _to_device(records: list[dict], device: Any) -> list[dict]:
    for record in records:
        for key, value in list(record.items()):
            if key == "inputs":
                record[key] = tuple(t.to(device) for t in value)
            elif torch.is_tensor(value):
                record[key] = value.to(device)
    return records


def train_geographic_ppo(
    dev_scenarios: list[dict[str, Any]],
    *,
    updates: int = 1,
    episodes_per_update: int = 1,
    lr: float = 3e-4,
    recurrent: bool = True,
    device: Any = None,
    resume_checkpoint: Path | str | None = None,
    input_mode: str = "legacy",
    action_mode: str = "tactical",
    imagery_dir: Path | str | None = None,
    include_ros: bool = False,
    include_thermal: bool = False,
    checkpoint_path: Path | str | None = None,
    checkpoint_every: int = 0,
    ppo_epochs: int = 1,
    minibatches: int = 1,
    gae_lambda: float | None = None,
    reward_scaling: bool = False,
    target_kl: float | None = None,
    entropy_coef: float = 0.01,
    entropy_coef_final: float | None = None,
    skip_forced: bool = False,
    drop_bonus_per_cell: float = 0.25,
    rollout_workers: int = 1,
    image_checkpoint_chunk: int | None = None,
    augment: bool = False,
    slot_features: bool = False,
    drop_distances: Any = (0,),
    selection_scenarios: list[dict[str, Any]] | None = None,
    eval_every: int = 0,
    history: list[dict[str, Any]] | None = None,
    rollout_envs: int = 1,
    gamma: float = 1.0,
    reward_reference: str | None = None,
    action_features: bool = False,
    vine_states: int = 0,
    vine_actions: int = 4,
) -> tuple[Any, list[float]]:
    """Train a geographic policy (legacy SpatialActorCritic or D-49 ImageActorCritic).

    With ``checkpoint_every > 0`` the weights are written to ``<checkpoint_path>.partial``
    every that many updates, so an interrupted run can continue with ``--resume``.
    With ``eval_every > 0`` and ``selection_scenarios`` (D-66) the policy is
    evaluated deterministically every ``eval_every`` updates and at the end; the
    weights with the lowest mean final burned area are restored before
    returning and saved to ``<checkpoint_path>.best``. Each evaluation is appended
    to ``history`` when given. With ``augment`` (D-65) each episode uses one of the eight rotations/mirrors
    of its scenario, chosen at random. With ``rollout_envs == 1`` and
    ``rollout_workers > 1`` the existing CPU model-copy collector is retained.
    With ``rollout_envs > 1`` only the parent owns the model/device; workers own
    simulation and tensorization. The update itself runs on ``device``. The entropy bonus decays linearly from ``entropy_coef`` to
    ``entropy_coef_final`` over the updates (constant when the latter is None).
    ``rollout_envs`` batches policy inference over independent episodes;
    ``rollout_workers`` optionally moves CPU environments into persistent
    spawned processes. Policy weights stay fixed throughout collection.
    ``gamma`` discounts returns per decision (D-80); 1.0, the default, optimises the final burned area.
    Serial collection remains the default for seeded trajectory compatibility.
    ``reward_reference`` (D-81) trains on hectares saved relative to a reference
    policy in the same scenario; it applies to training collection only, not selection.
    ``action_features`` (D-82) gives the parametric head features of each action's target cell.
    With ``vine_states > 0`` (D-85) up to that many decisions of every collected episode
    are branched: ``vine_actions`` actions are sampled from the policy there and each is
    followed to the end with common random numbers; the policy term then uses these
    branched samples and their leave-one-out advantages instead of Monte Carlo ones.
    """
    _require_torch()
    if rollout_envs < 1 or rollout_workers < 0:
        raise ValueError("rollout_envs must be positive and rollout_workers nonnegative")
    if vine_states < 0 or (vine_states and vine_actions < 2):
        raise ValueError("vine_states must be nonnegative and vine_actions at least 2")
    if vine_states and (rollout_envs != 1 or gamma != 1.0 or input_mode not in ("image", "cells")):
        raise ValueError("vine sampling needs serial or worker collection (rollout_envs 1), gamma 1 "
                         "and image or cells inputs")
    if episodes_per_update < 1 or updates < 0:
        raise ValueError("episodes_per_update must be positive and updates nonnegative")
    input_options = {"input_mode": input_mode, "imagery_dir": imagery_dir, "include_ros": include_ros,
                     "include_thermal": include_thermal}
    build_kwargs = {"action_mode": action_mode, "recurrent": recurrent, "slot_features": slot_features,
                    "drop_distances": tuple(drop_distances), "action_features": action_features, **input_options}
    augmented: dict[tuple[str, int], dict[str, Any]] = {}
    model = build_policy_model(dev_scenarios[0], **build_kwargs).to(device)
    if hasattr(model, "IMAGE_CHUNK"):
        model.IMAGE_CHUNK = image_checkpoint_chunk
    pool = None

    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint file not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        print(f"Resumed model weights from {resume_checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    partial_path = Path(f"{checkpoint_path}.partial") if checkpoint_path and checkpoint_every > 0 else None
    best_state: dict[str, Any] | None = None
    best_score = float("inf")

    def select(update_number: int) -> None:
        nonlocal best_state, best_score
        eval_kwargs = {"deterministic": True, "skip_forced": skip_forced, "drop_bonus_per_cell": drop_bonus_per_cell,
                       **input_options}
        if pool is not None:
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            tasks = [(build_kwargs, weights, scenario, eval_kwargs, 0) for scenario in selection_scenarios]
            results = [metrics for _, metrics in pool.map(_rollout_task, tasks)]
        else:
            model.eval()
            try:
                if backend is not None:
                    _, results = rollout_geographic_batch(
                        model, selection_scenarios, device=device, rollout_envs=rollout_envs,
                        backend=backend, **eval_kwargs)
                else:
                    with torch.no_grad():
                        results = [rollout_geographic(model, scenario, device=device, **eval_kwargs)[1]
                                   for scenario in selection_scenarios]
            finally:
                model.train()
        score = sum(r["final_burned_ha"] for r in results) / len(results)
        improved = score < best_score
        if improved:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, Path(f"{checkpoint_path}.best"))
        if history is not None:
            history.append({"update": update_number, "mean_final_burned_ha": round(score, 4), "best": improved})
        print(f"  [Select @ {update_number}] mean burned {score:.2f} ha{' (best)' if improved else ''}", flush=True)

    scaler = RewardScaler(gamma) if reward_scaling else None
    # Legacy defaults keep per-episode advantage normalization; the improved
    # update normalizes over the batch instead.
    batch_normalized = ppo_epochs > 1 or minibatches > 1 or gae_lambda is not None or reward_scaling
    from .geographic_rollouts import RolloutBackend, rollout_geographic_batch
    with ExitStack() as resources:
        backend = None
        if updates and rollout_envs > 1:
            backend = resources.enter_context(RolloutBackend(
                action_mode=action_mode, input_options=input_options,
                workers=min(rollout_workers, rollout_envs, episodes_per_update),
                skip_forced=skip_forced, drop_bonus_per_cell=drop_bonus_per_cell,
                drop_distances=tuple(drop_distances),
                action_features=action_features and action_mode == "parametric",
                reward_reference=reward_reference))
        elif updates and rollout_workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            pool = resources.enter_context(ProcessPoolExecutor(
                rollout_workers, mp_context=multiprocessing.get_context("spawn")))
        for update_idx in range(updates):
            started = time.monotonic()
            batch_records = []
            rollout_kwargs = {"gae_lambda": gae_lambda, "normalize_advantages": not batch_normalized, "gamma": gamma,
                              "skip_forced": skip_forced, "drop_bonus_per_cell": drop_bonus_per_cell,
                              "reward_reference": reward_reference, **input_options}
            chosen = [random.choice(dev_scenarios) for _ in range(episodes_per_update)]
            if augment:
                from .augmentation import NUM_TRANSFORMS, augment_scenario
                picks = [random.randrange(NUM_TRANSFORMS) for _ in chosen]
                chosen = [augmented.setdefault((sc["scenario_id"], t), augment_scenario(sc, t))
                          for sc, t in zip(chosen, picks)]
            if backend is not None:
                batch_records, _ = rollout_geographic_batch(
                    model, chosen, device=device, rollout_envs=rollout_envs,
                    backend=backend, **rollout_kwargs)
            elif pool is not None:
                state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                tasks = [(build_kwargs, state_dict, sc, rollout_kwargs, random.randrange(2**31))
                         for sc in chosen]
                batch_records = [_to_device(records, device) for records, _ in pool.map(_rollout_task, tasks)]
            else:
                for scenario in chosen:
                    records, _ = rollout_geographic(model, scenario, device=device, deterministic=False,
                                                    **rollout_kwargs)
                    batch_records.append(records)
            if device is not None and str(device).startswith("mps"):
                torch.mps.synchronize()
            elif device is not None and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            rollout_seconds = time.monotonic() - started
            if scaler is not None:
                for records in batch_records:
                    scaler.update([r["reward"] for r in records])
                for records in batch_records:
                    returns, advantages = returns_and_advantages(
                        [r["reward"] * scaler.scale for r in records], [r["value"] for r in records], gamma,
                        gae_lambda=gae_lambda, normalize=False)
                    returns, advantages = returns.to(device), advantages.to(device)
                    for record, ret, adv in zip(records, returns, advantages):
                        record["return"] = ret
                        record["advantage"] = adv
            vine_seconds = 0.0
            if vine_states:
                vine_started = time.monotonic()
                vine_kwargs = {"branch_actions": vine_actions, "skip_forced": skip_forced,
                               "drop_bonus_per_cell": drop_bonus_per_cell, "reward_reference": reward_reference,
                               **input_options}
                jobs = [(scenario, [int(r["action"]) for r in records],
                         choose_vine_decisions(len(records), vine_states, random), random.randrange(2 ** 31))
                        for scenario, records in zip(chosen, batch_records)]
                if pool is not None:
                    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    branches = list(pool.map(_vine_task, [
                        (build_kwargs, weights, scenario, actions, picked, {**vine_kwargs, "seed": seed})
                        for scenario, actions, picked, seed in jobs]))
                else:
                    with torch.no_grad():
                        branches = [vine_episode(model, scenario, actions, picked, seed=seed, device=device,
                                                 **vine_kwargs)
                                    for scenario, actions, picked, seed in jobs]
                for records, episode_branches in zip(batch_records, branches):
                    attach_vine_samples(records, episode_branches, device)
                vine_seconds = time.monotonic() - vine_started
            progress = update_idx / max(updates - 1, 1)
            current_entropy = (entropy_coef if entropy_coef_final is None
                               else entropy_coef + (entropy_coef_final - entropy_coef) * progress)
            loss = ppo_update_geographic(model, optimizer, batch_records, epochs=ppo_epochs,
                                         minibatches=minibatches, normalize_advantages=batch_normalized,
                                         target_kl=target_kl, entropy_coef=current_entropy)
            losses.append(loss)
            elapsed = time.monotonic() - started
            print(f"  [Update {update_idx + 1:2d}/{updates}] PPO Loss: {loss:9.4f} "
                  f"(episodes: {len(batch_records)}, {elapsed:.1f} s) "
                  f"[rollout {rollout_seconds:.2f} s; "
                  f"{f'vine {vine_seconds:.2f} s; ' if vine_states else ''}"
                  f"update {elapsed - rollout_seconds - vine_seconds:.2f} s]",
                  flush=True)
            if partial_path is not None and (update_idx + 1) % checkpoint_every == 0:
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), partial_path)
            if eval_every > 0 and selection_scenarios and (
                    (update_idx + 1) % eval_every == 0 or update_idx + 1 == updates):
                select(update_idx + 1)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a fast 1-update smoke check")
    parser.add_argument("--train", action="store_true", help="Train PPO on development scenarios")
    parser.add_argument("--select", action="store_true", help="Train candidates and select on validation")
    parser.add_argument("--eval", action="store_true", help="Evaluate checkpoint against baselines")
    parser.add_argument("--split", choices=("development", "validation", "reserved"), default="development")
    parser.add_argument("--scenarios-dir", type=Path, default=Path("runs/geographic-family-real-v2"))
    parser.add_argument("--synthetic", action="store_true", help="load an explicitly generated synthetic family")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--rollout-envs", type=int, default=1,
                        help="active episodes per policy inference batch (default: serial)")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42, help="Seed Python and PyTorch before training")
    parser.add_argument("--feedforward", action="store_true", help="Use feedforward instead of LSTM")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="legacy",
                        help="legacy rasters, D-49 image inputs (needs --imagery-dir), or the no-image ablation 'cells'")
    parser.add_argument("--action-mode", choices=("tactical", "spatial", "parametric"), default="tactical",
                        help="tactical macro-actions, one drop action per cell, or focus x aspect "
                             "parameterised actions (D-61); the latter two need image or cells inputs")
    parser.add_argument("--imagery-dir", type=Path, help="output directory of pandion.imagery_acquire")
    parser.add_argument("--ros", action="store_true", help="add the Rothermel ROS channel (image inputs)")
    parser.add_argument("--thermal", action="store_true",
                        help="add a simulated thermal band as a 4th image channel (image inputs, D-58)")
    parser.add_argument("--eval-scenarios", type=int, default=None, help="Limit number of validation scenarios to evaluate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-report", type=Path, help="Load evaluation architecture and verified weights binding from this report")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume weights from")
    parser.add_argument("--fleet-size", type=int, default=1,
                        help="number of identical aircraft (D-62); extra aircraft start at the water source")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu",
                        help="torch device; auto picks cuda, then Apple mps, then cpu")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="passes over each batch of rollouts")
    parser.add_argument("--minibatches", type=int, default=1, help="episode groups per epoch")
    parser.add_argument("--gae-lambda", type=float, help="GAE lambda (default: Monte Carlo returns)")
    parser.add_argument("--reward-scaling", action="store_true",
                        help="divide rewards by the running std of discounted returns")
    parser.add_argument("--target-kl", type=float, help="stop epochs when approx KL exceeds 1.5x this")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy bonus coefficient")
    parser.add_argument("--drop-bonus", type=float, default=0.25,
                        help="training reward per newly treated cell (0 = burned-area change only)")
    parser.add_argument("--image-checkpoint-chunk", type=int,
                        help="checkpoint the image encoder in chunks of N frames to save memory (slower)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="with --select: evaluate on the selection scenarios every N updates and keep the "
                             "best weights (D-66)")
    parser.add_argument("--augment", action="store_true",
                        help="train on random rotations/mirrors of each scenario (D-65)")
    parser.add_argument("--drop-distances", type=int, nargs="+", default=[0],
                        help="parametric drops may aim these many cells beyond the front (D-79); default 0 only")
    parser.add_argument("--slot-features", action="store_true",
                        help="parametric head also sees each focus slot's raw input averages and area (D-65)")
    parser.add_argument("--holdout-activation",
                        help="leave this development incident out of training; after selection on the validation "
                             "split, report the selected policy and the baselines on it (clean holdout, D-68)")
    parser.add_argument("--rollout-workers", type=int, default=1,
                        help="CPU workers; with --rollout-envs > 1 they own environments with central device inference")
    parser.add_argument("--skip-forced", action="store_true",
                        help="do not query or train on single-action steps (D-64); applies to evaluation too")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="discount per decision (D-80); 1.0 optimises the final burned area")
    parser.add_argument("--reward-reference", choices=REWARD_REFERENCES, default=None,
                        help="train on hectares saved relative to this policy in the same scenario (D-81)")
    parser.add_argument("--selection-activations", nargs="+", metavar="ACTIVATION",
                        help="with --select: leave these development incidents out of training and choose the "
                             "weights on them instead of on the validation split (D-84)")
    parser.add_argument("--action-features", action="store_true",
                        help="parametric head also sees features of each action's target cell (D-82)")
    parser.add_argument("--vine-states", type=int, default=0,
                        help="branch this many decisions of each training episode and train the policy on "
                             "branched advantages with common random numbers (D-85); 0 = Monte Carlo/GAE")
    parser.add_argument("--vine-actions", type=int, default=4,
                        help="actions sampled from the policy at each branched decision (D-85)")
    parser.add_argument("--entropy-coef-final", type=float,
                        help="decay the entropy bonus linearly to this value over the updates")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="also save weights to <checkpoint>.partial every N updates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rollout_envs < 1 or args.rollout_workers < 0:
        parser.error("--rollout-envs must be positive and --rollout-workers nonnegative")

    _require_torch()
    device = torch.device(resolve_device(args.device))
    recurrent = not args.feedforward
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev_scenarios = with_fleet(load_geographic_split("development", args.scenarios_dir, synthetic=args.synthetic),
                               args.fleet_size)
    input_options = {"input_mode": args.input_mode, "imagery_dir": args.imagery_dir, "include_ros": args.ros,
                     "include_thermal": args.thermal}
    model_options = {"action_mode": args.action_mode, "slot_features": args.slot_features,
                     "drop_distances": tuple(args.drop_distances), "action_features": args.action_features,
                     **input_options}
    holdout_scenarios = []
    if args.holdout_activation:
        dev_scenarios, holdout_scenarios = partition_holdout(dev_scenarios, args.holdout_activation)
    selection_scenarios: list[dict[str, Any]] = []
    for activation in args.selection_activations or []:
        # D-84: selection incidents are development incidents used neither for training nor as the holdout.
        dev_scenarios, chosen = partition_holdout(dev_scenarios, activation)
        selection_scenarios += chosen
    rollout_options = {**input_options, "skip_forced": args.skip_forced}
    ppo_options = {"ppo_epochs": args.ppo_epochs, "minibatches": args.minibatches, "gae_lambda": args.gae_lambda,
                   "reward_scaling": args.reward_scaling, "target_kl": args.target_kl,
                   "entropy_coef": args.entropy_coef, "entropy_coef_final": args.entropy_coef_final,
                   "skip_forced": args.skip_forced, "drop_bonus_per_cell": args.drop_bonus,
                   "rollout_workers": args.rollout_workers, "augment": args.augment,
                   "slot_features": args.slot_features, "eval_every": args.eval_every,
                   "image_checkpoint_chunk": args.image_checkpoint_chunk, "gamma": args.gamma,
                   "reward_reference": args.reward_reference,
                   "vine_states": args.vine_states, "vine_actions": args.vine_actions}
    train_options = {**model_options, **ppo_options,
                     "checkpoint_path": args.checkpoint, "checkpoint_every": args.checkpoint_every,
                     "rollout_envs": args.rollout_envs, "rollout_workers": args.rollout_workers}
    config_report = {"ppo": {"epochs": args.ppo_epochs, "minibatches": args.minibatches,
                             "gae_lambda": args.gae_lambda, "reward_scaling": args.reward_scaling,
                             "target_kl": args.target_kl, "entropy_coef": args.entropy_coef,
                             "entropy_coef_final": args.entropy_coef_final, "gamma": args.gamma,
                             "reward_reference": args.reward_reference,
                             "vine_states": args.vine_states, "vine_actions": args.vine_actions}, "input_mode": args.input_mode, "action_mode": args.action_mode, "include_ros": args.ros, "include_thermal": args.thermal, "fleet_size": args.fleet_size, "holdout_activation": args.holdout_activation,
                     "drop_distances": list(args.drop_distances), "action_features": args.action_features,
                     "selection_activations": args.selection_activations,
                     "imagery_dir": str(args.imagery_dir) if args.imagery_dir else None,
                     "collection": {"rollout_envs": args.rollout_envs, "rollout_workers": args.rollout_workers},
                     "slot_features": args.slot_features, "skip_forced": args.skip_forced,
                     "updates": args.updates, "episodes_per_update": args.episodes_per_update,
                     "training_settings": {"learning_rate": args.lr, "augment": args.augment,
                         "drop_bonus_per_cell": args.drop_bonus, "eval_every": args.eval_every,
                         "image_checkpoint_chunk": args.image_checkpoint_chunk,
                         "resume_checkpoint_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None}}

    if args.smoke or (not args.train and not args.select and not args.eval):
        print(f"Running Geographic PPO smoke test (recurrent={recurrent})...")
        smoke_scenarios = dev_scenarios[:2]
        model, losses = train_geographic_ppo(
            smoke_scenarios,
            updates=1,
            episodes_per_update=1,
            recurrent=recurrent,
            device=device,
            resume_checkpoint=args.resume,
            **train_options,
        )
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved trained checkpoint -> {args.checkpoint}")
        _, metrics = rollout_geographic(model, dev_scenarios[0], device=device, deterministic=True,
                                        **rollout_options)
        report = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "dataset": _dataset_report(args.scenarios_dir, {"development": smoke_scenarios}),
            "seed": args.seed,
            "recurrent": recurrent,
            **config_report, "device": str(device),
            "updates": 1, "episodes_per_update": 1,
            "loss": losses[0],
            "evaluation_metrics": metrics,
        }
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_report(report, args.output, args.checkpoint)
            print(f"Wrote smoke artifact -> {args.output}")
        else:
            print(rendered)
        return

    if args.train:
        print(f"Training Geographic PPO (updates={args.updates}, episodes_per_update={args.episodes_per_update}, recurrent={recurrent})...")
        model, losses = train_geographic_ppo(
            dev_scenarios,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            lr=args.lr,
            recurrent=recurrent,
            device=device,
            resume_checkpoint=args.resume,
            **train_options,
        )
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved trained checkpoint -> {args.checkpoint}")

        _, eval_m = rollout_geographic(model, dev_scenarios[0], device=device, deterministic=True,
                                        **rollout_options)
        report = {
            "schema_version": "nightjar-geographic-ppo-train-0.1",
            "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios}),
            "seed": args.seed,
            "recurrent": recurrent,
            **config_report, "device": str(device),
            "updates": args.updates,
            "final_loss": losses[-1] if losses else 0.0,
            "evaluation": eval_m,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_report(report, args.output, args.checkpoint)
            print(f"Wrote training report -> {args.output}")
        return

    if args.select:
        selection_split = "selection" if selection_scenarios else "validation"
        val_scenarios = selection_scenarios or with_fleet(
            load_geographic_split("validation", args.scenarios_dir, synthetic=args.synthetic), args.fleet_size)
        if args.eval_scenarios is not None:
            val_scenarios = val_scenarios[:args.eval_scenarios]
        print(f"Training {args.candidates} candidates on {len(dev_scenarios)} development scenarios...")
        best_candidate = None
        best_mean_burned = float("inf")
        selected_seed = None
        candidate_reports = []

        for c_idx in range(args.candidates):
            c_seed = args.seed + c_idx
            random.seed(c_seed)
            torch.manual_seed(c_seed)
            print(f"\n--- Candidate {c_idx + 1}/{args.candidates} (seed={c_seed}) ---")

            selection_history: list[dict[str, Any]] = []
            cand_model, c_losses = train_geographic_ppo(
                dev_scenarios,
                updates=args.updates,
                episodes_per_update=args.episodes_per_update,
                lr=args.lr,
                recurrent=recurrent,
                device=device,
                selection_scenarios=val_scenarios,
                history=selection_history,
                **train_options,
            )

            # Evaluate candidate on validation scenarios
            val_results = evaluate_geographic_policy(cand_model, val_scenarios, device=device, **rollout_options)
            mean_burned = sum(r["final_burned_ha"] for r in val_results) / max(len(val_results), 1)
            total_drops = sum(r["drops_count"] for r in val_results)

            candidate_reports.append({
                "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios, selection_split: val_scenarios}),
                "candidate_index": c_idx,
                "seed": c_seed,
                "mean_burned_ha": round(mean_burned, 2),
                "total_drops": total_drops,
                "losses": c_losses,
                "selection_history": selection_history,
            })

            print(f"Candidate {c_idx + 1} {selection_split} mean burned: {mean_burned:.2f} ha (total drops: {total_drops})")

            if mean_burned < best_mean_burned:
                best_mean_burned = mean_burned
                best_candidate = cand_model
                selected_seed = c_seed

        if args.checkpoint and best_candidate is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_candidate.state_dict(), args.checkpoint)
            print(f"\nSaved best candidate checkpoint -> {args.checkpoint}")

        holdout_report = None
        if holdout_scenarios and best_candidate is not None:
            holdout_report = holdout_evaluation(best_candidate, holdout_scenarios, device=device, **rollout_options)
            print(f"Holdout {args.holdout_activation}: policy {holdout_report['policy_mean_ha']:.2f} ha | "
                  f"heuristic {holdout_report['baselines_mean_ha']['heuristic']:.2f} | "
                  f"containment {holdout_report['baselines_mean_ha']['containment_heuristic']:.2f} | "
                  f"hold-aware {holdout_report['baselines_mean_ha']['hold_aware_heuristic']:.2f} | "
                  f"no intervention {holdout_report['baselines_mean_ha']['no_intervention']:.2f}")

        report = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "holdout": holdout_report,
            "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios, selection_split: val_scenarios,
                                                            **({"holdout": holdout_scenarios} if holdout_scenarios else {})}),
            "seed": args.seed,
            "recurrent": recurrent,
            **config_report, "device": str(device),
            "candidates_count": args.candidates, "selected_seed": selected_seed,
            "best_mean_burned_ha": round(best_mean_burned, 2),
            "candidates": candidate_reports,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_report(report, args.output, args.checkpoint)
            print(f"Wrote selection report -> {args.output}")
        return

    if args.eval:
        if not args.checkpoint or not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file required for --eval: {args.checkpoint}")
        target_scenarios = with_fleet(load_geographic_split(args.split, args.scenarios_dir, synthetic=args.synthetic),
                                      args.fleet_size)
        if args.eval_scenarios is not None:
            target_scenarios = target_scenarios[:args.eval_scenarios]

        if args.training_report:
            from .policy_artifact import load_policy_artifact
            model, artifact_info = load_policy_artifact(args.checkpoint, args.training_report,
                load_geographic_split("development", args.scenarios_dir, synthetic=args.synthetic),
                imagery_dir=args.imagery_dir, device=str(device), skip_forced=True if args.skip_forced else None)
            config = model._pandion_inference
            target_scenarios = with_fleet(load_geographic_split(args.split, args.scenarios_dir, synthetic=args.synthetic), config["fleet_size"])
            if args.eval_scenarios is not None:
                target_scenarios = target_scenarios[:args.eval_scenarios]
            rollout_options = {key: config[key] for key in ("input_mode", "imagery_dir", "include_ros", "include_thermal", "skip_forced")}
            config_report.update({key: config[key] for key in ("input_mode", "action_mode", "recurrent", "include_ros", "include_thermal", "fleet_size", "slot_features", "skip_forced")})
            config_report["imagery_dir"] = str(config["imagery_dir"]) if config["imagery_dir"] else None
            config_report["action_features"] = bool(config.get("action_features", False))
            config_report["device"] = str(device)
            config_report["training_artifact"] = artifact_info
        else:
            model = build_policy_model(target_scenarios[0], recurrent=recurrent, **model_options).to(device)
            model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=device))
            model.eval()

        print(f"Evaluating {args.checkpoint} against baselines on split '{args.split}' ({len(target_scenarios)} scenarios)...")
        eval_reports = []
        for sc in target_scenarios:
            bm = run_geographic_benchmark(sc, random_seeds=(42, 43))
            _, rl_metrics = rollout_geographic(model, sc, device=device, deterministic=True, inference_only=True, **rollout_options)
            no_int_ha = bm["policies"]["no_intervention"]["final_burned_ha"]
            heur_ha = bm["policies"]["heuristic"]["final_burned_ha"]
            cont_ha = bm["policies"]["containment_heuristic"]["final_burned_ha"]
            hold_ha = bm["policies"]["hold_aware_heuristic"]["final_burned_ha"]
            rand_ha = bm["policies"]["random_valid_mean"]["final_burned_ha"]
            rl_ha = rl_metrics["final_burned_ha"]
            area_saved = round(no_int_ha - rl_ha, 2)
            eval_reports.append({
                "scenario_id": sc["scenario_id"],
                "no_intervention_ha": no_int_ha,
                "heuristic_ha": heur_ha,
                "containment_ha": cont_ha,
                "hold_aware_ha": hold_ha,
                "random_valid_ha": rand_ha,
                "ppo_rl_ha": rl_ha,
                "area_saved_ha": area_saved,
                "drops_count": rl_metrics["drops_count"],
                "flight_km": rl_metrics["flight_distance_km"],
                "rejected_orders": rl_metrics["rejected_orders"],
            })

        print("\n========================= Geographic Evaluation Summary =========================")
        print(f"{'Scenario':<28} {'No Interv':<12} {'Heuristic':<12} {'Contain':<12} {'PPO (RL)':<12} {'Saved (ha)':<10}")
        print("-" * 88)
        for r in eval_reports:
            print(f"{r['scenario_id']:<28} {r['no_intervention_ha']:<12.1f} {r['heuristic_ha']:<12.1f} {r['containment_ha']:<12.1f} {r['ppo_rl_ha']:<12.1f} {r['area_saved_ha']:<10.1f}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_report({"schema_version": EVAL_SCHEMA_VERSION, "recurrent": recurrent, **config_report,
                          "dataset": _dataset_report(args.scenarios_dir, {args.split: target_scenarios}),
                          "reports": eval_reports}, args.output, args.checkpoint, attach_sidecar=False)
            print(f"Wrote evaluation report -> {args.output}")
        return



if __name__ == "__main__":
    main()
