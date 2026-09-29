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
from .geographic_family import canonical_hash
from .cems_catalog import ACTIVATIONS
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
    designated = {"development": {"EMSR578"}, "validation": {"EMSR590"},
                  "reserved": {"EMSR592", "EMSR600"}}[split]
    scenarios = []
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
        valid_source_hash = (kind == "synthetic_fixture" and source_hash is None) or (
            kind == "copernicus_cems" and source_hash in {
                product["sha256"] for product in ACTIVATIONS[activation_id]["products"].values()
            }
        )
        if (kind != manifest.get("source_kind") or kind != entry.get("source_kind")
                or source_hash != entry.get("source_zip_sha256") or not valid_source_hash
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
            split: [{"scenario_id": scenario["scenario_id"],
                     "scenario_hash": canonical_hash({k: v for k, v in scenario.items() if k != "scenario_hash"}),
                     "source_kind": scenario["source_kind"]} for scenario in scenarios]
            for split, scenarios in splits.items()
        },
    }


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
    resume_checkpoint: Path | str | None = None,
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

    if resume_checkpoint and Path(resume_checkpoint).exists():
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        print(f"Resumed model weights from {resume_checkpoint}")

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
        print(f"  [Update {update_idx + 1:2d}/{updates}] PPO Loss: {loss:9.4f} (episodes: {len(batch_records)})")

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
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feedforward", action="store_true", help="Use feedforward instead of LSTM")
    parser.add_argument("--eval-scenarios", type=int, default=None, help="Limit number of validation scenarios to evaluate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume weights from")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _require_torch()
    device = torch.device("cpu")
    recurrent = not args.feedforward

    dev_scenarios = load_geographic_split("development", args.scenarios_dir, synthetic=args.synthetic)

    if args.smoke or (not args.train and not args.select and not args.eval):
        print(f"Running Geographic PPO smoke test (recurrent={recurrent})...")
        model, losses = train_geographic_ppo(
            dev_scenarios[:2],
            updates=1,
            episodes_per_update=1,
            recurrent=recurrent,
            device=device,
            resume_checkpoint=args.resume,
        )
        _, metrics = rollout_geographic(model, dev_scenarios[0], device=device, deterministic=True)
        report = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios}),
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
        )
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved trained checkpoint -> {args.checkpoint}")

        _, eval_m = rollout_geographic(model, dev_scenarios[0], device=device, deterministic=True)
        report = {
            "schema_version": "nightjar-geographic-ppo-train-0.1",
            "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios}),
            "recurrent": recurrent,
            "updates": args.updates,
            "final_loss": losses[-1] if losses else 0.0,
            "evaluation": eval_m,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(f"Wrote training report -> {args.output}")
        return

    if args.select:
        val_scenarios = load_geographic_split("validation", args.scenarios_dir, synthetic=args.synthetic)
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
            print(f"\n--- Candidate {c_idx + 1}/{args.candidates} (seed={c_seed}) ---")

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
                "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios, "validation": val_scenarios}),
                "candidate_index": c_idx,
                "seed": c_seed,
                "mean_burned_ha": round(mean_burned, 2),
                "total_drops": total_drops,
                "losses": c_losses,
            })

            print(f"Candidate {c_idx + 1} validation mean burned: {mean_burned:.2f} ha (total drops: {total_drops})")

            if mean_burned < best_mean_burned:
                best_mean_burned = mean_burned
                best_candidate = cand_model

        if args.checkpoint and best_candidate is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_candidate.state_dict(), args.checkpoint)
            print(f"\nSaved best candidate checkpoint -> {args.checkpoint}")

        report = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "dataset": _dataset_report(args.scenarios_dir, {"development": dev_scenarios, "validation": val_scenarios}),
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

    if args.eval:
        if not args.checkpoint or not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file required for --eval: {args.checkpoint}")
        target_scenarios = load_geographic_split(args.split, args.scenarios_dir, synthetic=args.synthetic)
        if args.eval_scenarios is not None:
            target_scenarios = target_scenarios[:args.eval_scenarios]

        sample_grid = target_scenarios[0]["grid"]
        rows, cols = int(sample_grid["rows"]), int(sample_grid["cols"])
        model = SpatialActorCritic(
            grid_rows=rows,
            grid_cols=cols,
            action_dim=len(TACTICAL_ACTION_NAMES),
            recurrent=recurrent,
        ).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        print(f"Evaluating {args.checkpoint} against baselines on split '{args.split}' ({len(target_scenarios)} scenarios)...")
        eval_reports = []
        for sc in target_scenarios:
            bm = run_geographic_benchmark(sc, random_seeds=(42, 43))
            _, rl_metrics = rollout_geographic(model, sc, device=device, deterministic=True)
            no_int_ha = bm["policies"]["no_intervention"]["final_burned_ha"]
            heur_ha = bm["policies"]["heuristic"]["final_burned_ha"]
            cont_ha = bm["policies"]["containment_heuristic"]["final_burned_ha"]
            rand_ha = bm["policies"]["random_valid_mean"]["final_burned_ha"]
            rl_ha = rl_metrics["final_burned_ha"]
            area_saved = round(no_int_ha - rl_ha, 2)
            eval_reports.append({
                "scenario_id": sc["scenario_id"],
                "no_intervention_ha": no_int_ha,
                "heuristic_ha": heur_ha,
                "containment_ha": cont_ha,
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
            args.output.write_text(json.dumps({"schema_version": EVAL_SCHEMA_VERSION,
                                               "dataset": _dataset_report(args.scenarios_dir, {args.split: target_scenarios}),
                                               "reports": eval_reports}, indent=2))
            print(f"Wrote evaluation report -> {args.output}")
        return



if __name__ == "__main__":
    main()
