"""Batched, on-policy collection with optional persistent CPU workers.

Only the parent process owns the policy/device. Workers own independent
episodes and exchange typed input arrays, masks and scalar outcomes.
"""
from __future__ import annotations

from contextlib import nullcontext
import multiprocessing as mp
import traceback
from typing import Any

import numpy as np


def rollout_metrics(scenario: dict, replay: dict, rewards: list[float]) -> dict:
    """The same episode endpoints used by the serial geographic rollout."""
    return {
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
        "steps_count": len(rewards),
    }


class _Episode:
    def __init__(self, scenario: dict, options: dict):
        from .geographic_ppo import make_input_tensorizer
        from .geographic_training_env import GeographicTrainingEnv

        self.options = options
        self.scenario = scenario
        self.env = GeographicTrainingEnv(scenario, action_mode=options["action_mode"],
                                         drop_bonus_per_cell=options["drop_bonus_per_cell"])
        self.mapper = self.env.action_mapper()
        self.tensorizer = make_input_tensorizer(
            scenario, **options["inputs"], focus_slots=options["action_mode"] == "parametric")
        self.rewards: list[float] = []
        self.decisions = 0
        self.observation = self.env.reset(seed=scenario.get("seed"))

    def packet(self, reward: float = 0.0, done: bool = False) -> dict:
        if done:
            metrics = rollout_metrics(self.scenario, self.env.replay, self.rewards)
            metrics["steps_count"] = self.decisions
            return {"reward": reward, "done": True, "metrics": metrics}
        mask = np.asarray(self.mapper.compute_mask(self.observation), dtype=np.bool_)
        if self.options.get("skip_forced", False) and int(np.count_nonzero(mask)) == 1:
            return {"mask": mask, "reward": reward, "done": False}
        if hasattr(self.tensorizer, "extract_arrays"):
            inputs = self.tensorizer.extract_arrays(self.observation)
        else:
            # Legacy tensorizer keeps its existing public interface.
            inputs = tuple(t.squeeze(0).numpy()
                           for t in self.tensorizer.to_torch(self.observation, device="cpu"))
        return {"inputs": inputs, "mask": mask, "reward": reward, "done": False}

    def step(self, action: int, *, decision: bool = True) -> dict:
        self.observation, reward, done, _ = self.env.step_discrete(action)
        self.rewards.append(reward)
        if decision:
            self.decisions += 1
        return self.packet(reward, done)


class _LocalBackend:
    def __init__(self, options: dict):
        self.options = options
        self.episodes: dict[int, _Episode] = {}

    def reset(self, scenarios: list[tuple[int, dict]]) -> dict:
        self.episodes = {slot: _Episode(scenario, self.options) for slot, scenario in scenarios}
        return {slot: ep.packet() for slot, ep in self.episodes.items()}

    def step(self, actions: list[tuple[int, int] | tuple[int, int, bool]]) -> dict:
        results = {}
        for item in actions:
            slot, action, *decision = item
            results[slot] = self.episodes[slot].step(action, decision=decision[0] if decision else True)
        return results

    def close(self) -> None:
        self.episodes.clear()


def _worker(connection: Any, options: dict) -> None:
    import torch
    torch.set_num_threads(1)
    backend = _LocalBackend(options)
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            try:
                result = getattr(backend, command)(payload)
                connection.send(("ok", result))
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        backend.close()
        connection.close()


class RolloutBackend:
    """Reusable CPU episode owners; close releases all workers after a run.

    Workers use spawn (safe with a parent accelerator) and one torch CPU
    thread each. Arrays travel through pipes; full observation/replay dicts
    stay with their environment owner. No weights or GPU tensors are sent.
    """
    def __init__(self, *, action_mode: str, input_options: dict,
                 workers: int = 0, drop_bonus_per_cell: float = 0.25,
                 skip_forced: bool = False):
        if workers < 0:
            raise ValueError("rollout_workers must be nonnegative")
        options = {"action_mode": action_mode, "inputs": input_options,
                   "drop_bonus_per_cell": drop_bonus_per_cell,
                   "skip_forced": skip_forced}
        self.local = _LocalBackend(options) if workers == 0 else None
        self.connections: list[Any] = []
        self.processes: list[Any] = []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, options), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def _call(self, command: str, items: list[tuple[int, Any]]) -> dict:
        if self.closed:
            raise RuntimeError("rollout backend is closed")
        if self.local is not None:
            return getattr(self.local, command)(items)
        groups = [[] for _ in self.connections]
        for item in items:
            groups[item[0] % len(groups)].append(item)
        # Send all work first, then receive; workers step independently.
        pending = []
        for connection, group in zip(self.connections, groups):
            if group or command == "reset":
                connection.send((command, group))
                pending.append(connection)
        result = {}
        for connection in pending:
            try:
                status, payload = connection.recv()
            except EOFError as exc:
                raise RuntimeError("rollout worker exited without a result") from exc
            if status != "ok":
                raise RuntimeError(f"rollout worker failed: {payload}")
            result.update(payload)
        return result

    def reset(self, scenarios: list[tuple[int, dict]]) -> dict:
        return self._call("reset", scenarios)

    def step(self, actions: list[tuple[int, int]]) -> dict:
        return self._call("step", actions)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.local is not None:
            self.local.close()
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def __enter__(self) -> RolloutBackend:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def rollout_geographic_batch(
    model: Any, scenarios: list[dict], *, device: Any = None,
    rollout_envs: int = 8, rollout_workers: int = 0,
    deterministic: bool = False, gamma: float = 0.99,
    gae_lambda: float | None = None, reward_scale: float = 1.0,
    normalize_advantages: bool = True, drop_bonus_per_cell: float = 0.25,
    skip_forced: bool = False,
    input_mode: str = "legacy", imagery_dir: Any = None,
    include_ros: bool = False, include_thermal: bool = False,
    backend: RolloutBackend | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    """Collect complete episodes with frozen weights and independent RNN states.

    At most ``rollout_envs`` episodes are active at once. Results retain input
    scenario order even when episode lengths differ. Batched stochastic
    sampling consumes RNG differently from serial collection, so the same
    seed does not promise the same actions as serial mode. Active episodes
    whose input or mask shapes differ are evaluated in separate forward
    passes. With a ``backend``, its own ``skip_forced`` and input options apply.
    """
    import torch
    from .geographic_ppo import returns_and_advantages

    if rollout_envs < 1:
        raise ValueError("rollout_envs must be positive")
    if rollout_workers < 0:
        raise ValueError("rollout_workers must be nonnegative")
    if not scenarios:
        return [], []
    options = {"input_mode": input_mode, "imagery_dir": imagery_dir,
               "include_ros": include_ros, "include_thermal": include_thermal}
    context = (nullcontext(backend) if backend is not None else RolloutBackend(
        action_mode=getattr(model, "action_mode", "tactical"), input_options=options,
        workers=min(rollout_workers, rollout_envs, len(scenarios)),
        drop_bonus_per_cell=drop_bonus_per_cell, skip_forced=skip_forced))
    all_records: list[list[dict]] = [[] for _ in scenarios]
    metrics: list[Any] = [None for _ in scenarios]
    with context as owner:
        for start in range(0, len(scenarios), rollout_envs):
            indices = list(range(start, min(start + rollout_envs, len(scenarios))))
            packets = owner.reset([(i, scenarios[i]) for i in indices])
            rewards = {i: [] for i in indices}
            values = {i: [] for i in indices}
            carried = {i: 0.0 for i in indices}
            active = indices
            states: dict[int, Any] = {i: None for i in indices}
            while active:
                # The episode owner decides which steps are forced: it omits
                # inputs when its skip_forced option leaves a single action.
                forced = {i: "inputs" not in packets[i] for i in active}
                decisions = [i for i in active if not forced[i]]
                chosen: dict[int, int] = {}
                decided: dict[int, dict] = {}
                # Scenarios of different window sizes cannot share a tensor;
                # each shape group gets its own forward pass.
                groups: dict[tuple, list[int]] = {}
                for i in decisions:
                    key = tuple(a.shape for a in packets[i]["inputs"]) + (packets[i]["mask"].shape,)
                    groups.setdefault(key, []).append(i)
                for group in groups.values():
                    inputs = tuple(torch.from_numpy(np.stack(arrays)).to(device)
                                   for arrays in zip(*(packets[i]["inputs"] for i in group)))
                    masks = torch.from_numpy(np.stack([packets[i]["mask"] for i in group])).to(device)
                    next_state = None
                    with torch.no_grad():
                        if getattr(model, "recurrent", False):
                            prior_state = None
                            if any(states[i] is not None for i in group):
                                state_parts = [states[i] for i in group]
                                zero_state = model.initial_state(batch_size=len(group), device=inputs[0].device)
                                prior_state = tuple(torch.cat([
                                    part[j] if part is not None else zero_state[j][row:row + 1]
                                    for row, part in enumerate(state_parts)], dim=0)
                                    for j in range(len(zero_state)))
                            actions, log_probs, _, estimates, next_state = model.get_action(
                                *inputs, masks, deterministic=deterministic, state=prior_state)
                        else:
                            actions, log_probs, _, estimates = model.get_action(
                                *inputs, masks, deterministic=deterministic)
                    action_list = actions.cpu().tolist()
                    for row, index in enumerate(group):
                        chosen[index] = int(action_list[row])
                        decided[index] = {"inputs": tuple(t[row] for t in inputs), "mask": masks[row],
                                          "action": actions[row], "log_prob": log_probs[row],
                                          "value": estimates[row]}
                        if next_state is not None:
                            states[index] = tuple(t[row:row + 1] for t in next_state)
                for index in active:
                    if forced[index]:
                        chosen[index] = int(np.flatnonzero(packets[index]["mask"])[0])
                packets = owner.step([(i, chosen[i], not forced[i]) for i in active])
                for index in active:
                    packet = packets[index]
                    if forced[index]:
                        if rewards[index]:
                            rewards[index][-1] += packet["reward"]
                        else:
                            carried[index] += packet["reward"]
                    else:
                        record = decided[index]
                        values[index].append(record.pop("value"))
                        all_records[index].append(record)
                        rewards[index].append(carried[index] + packet["reward"])
                        carried[index] = 0.0
                    if packet["done"]:
                        metrics[index] = packet["metrics"]
                active = [i for i in active if not packets[i]["done"]]
            for index in indices:
                if not rewards[index]:
                    continue
                # Values and targets cross devices once per episode, in bulk.
                estimates = torch.stack(values[index]).cpu().tolist()
                ret, adv = returns_and_advantages(
                    [r * reward_scale for r in rewards[index]], estimates, gamma,
                    gae_lambda=gae_lambda, normalize=normalize_advantages)
                ret, adv = ret.to(device), adv.to(device)
                for record, rt, ad, reward, estimate in zip(
                        all_records[index], ret, adv, rewards[index], estimates):
                    record.update({"return": rt, "advantage": ad,
                                   "reward": reward, "value": estimate})
    return all_records, metrics
