"""Branched ("vine") advantage estimates for PPO (D-85).

A Monte Carlo return credits every decision of an episode with the outcome of
the whole episode, and in this environment the spread caused by the policy's
own later choices is larger than the effect of most single decisions. Vine
sampling (Schulman et al., 2015, "Trust Region Policy Optimization", section
5.2) instead branches the simulator at a few decisions of each collected
episode: several actions are sampled from the current policy at that state and
each is followed to the end of the episode by the same policy, with common
random numbers so that every branch draws the same sequence of random numbers
for its later choices. The difference between branch returns is then due to
the branched action and the simulator, not to sampling noise in the
continuation.

The simulator cannot be copied mid-episode, so each branch replays the prefix:
recorded decision actions in order, with single-action steps taken as forced
when ``skip_forced`` is set (exactly as in collection). The recurrent state
after a decision does not depend on the action taken there, so all branches
continue from the same state. Only the controller observation is used; the
branch returns are training targets, never policy inputs.
"""

from __future__ import annotations

import random
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is required by the callers
    torch = None


def choose_vine_decisions(num_decisions: int, states: int, rng: random.Random) -> list[int]:
    """Indices of up to ``states`` decisions of an episode, uniformly without replacement, sorted."""
    if states <= 0 or num_decisions <= 0:
        return []
    return sorted(rng.sample(range(num_decisions), min(states, num_decisions)))


def leave_one_out_advantages(returns: list[float]) -> list[float]:
    """A_i = R_i - mean of the other branches' returns; zeros for a single branch."""
    count = len(returns)
    if count < 2:
        return [0.0] * count
    total = sum(returns)
    return [r - (total - r) / (count - 1) for r in returns]


def _mix(seed: int, decision: int, stream: int) -> int:
    """Deterministic generator seed (Python's hash of strings varies between processes)."""
    return (int(seed) * 1_000_003 + decision * 2 + stream) % (2 ** 63)


def _forward(model: Any, inputs: tuple, state: Any) -> tuple[Any, Any]:
    """Masked-free logits and next recurrent state (None for feedforward models)."""
    if getattr(model, "recurrent", False):
        logits, _, next_state = model.forward(*inputs, state=state)
        return logits, next_state
    logits, _ = model.forward(*inputs)
    return logits, None


def _new_env(scenario: dict, model: Any, options: dict) -> tuple[Any, dict, Any, Any]:
    from .geographic_ppo import make_input_tensorizer
    from .geographic_training_env import GeographicTrainingEnv
    action_mode = getattr(model, "action_mode", "tactical")
    drop_distances = getattr(model, "drop_distances", (0,))
    env = GeographicTrainingEnv(scenario, drop_bonus_per_cell=options["drop_bonus_per_cell"],
                                action_mode=action_mode, drop_distances=drop_distances,
                                reward_reference=options.get("reward_reference"))
    observation = env.reset(seed=scenario.get("seed"))
    tensorizer = make_input_tensorizer(scenario, input_mode=options.get("input_mode", "legacy"),
                                       imagery_dir=options.get("imagery_dir"),
                                       include_ros=options.get("include_ros", False),
                                       include_thermal=options.get("include_thermal", False),
                                       focus_slots=action_mode == "parametric",
                                       action_features=getattr(model, "action_feature_dim", 0) > 0,
                                       drop_distances=drop_distances)
    return env, observation, env.action_mapper(), tensorizer


def _is_forced(mapper: Any, observation: dict, skip_forced: bool) -> int | None:
    if not skip_forced:
        return None
    valid = mapper.compute_mask(observation)
    return valid.index(True) if sum(valid) == 1 else None


def vine_episode(
    model: Any,
    scenario: dict,
    decision_actions: list[int],
    chosen: list[int],
    *,
    branch_actions: int,
    seed: int,
    skip_forced: bool,
    drop_bonus_per_cell: float,
    reward_reference: str | None = None,
    device: Any = None,
    **input_options: Any,
) -> list[dict[str, Any]]:
    """Branch one collected episode at the decisions ``chosen`` (indices into ``decision_actions``).

    Returns one entry per chosen decision: ``{"decision", "actions", "log_probs", "returns"}``
    with ``branch_actions`` actions sampled with replacement from the policy at that
    decision, their log-probabilities, and each branch's summed reward from that
    decision to the end of the episode (gamma 1). Identical actions share one
    simulation, since a branch is deterministic given its action and random numbers.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("vine sampling requires PyTorch")
    if not chosen:
        return []
    options = {"drop_bonus_per_cell": drop_bonus_per_cell, "reward_reference": reward_reference, **input_options}

    # Pass 1: replay the episode up to the last chosen decision to get the policy and
    # recurrent state at each chosen decision.
    env, observation, mapper, tensorizer = _new_env(scenario, model, options)
    targets: dict[int, dict[str, Any]] = {}
    state, decision, last = None, 0, max(chosen)
    with torch.no_grad():
        while decision <= last:
            forced = _is_forced(mapper, observation, skip_forced)
            if forced is not None:
                observation, _, done, _ = env.step_discrete(forced)
                if done:
                    raise RuntimeError("episode ended before a recorded decision; replay diverged")
                continue
            inputs = tuple(tensorizer.to_torch(observation, device=device))
            mask = mapper.to_mask_torch(observation, device=device)
            logits, next_state = _forward(model, inputs, state)
            if decision in chosen:
                log_probs = torch.log_softmax(model._masked(logits, mask), dim=-1)[0]
                generator = torch.Generator().manual_seed(_mix(seed, decision, 0))
                sampled = torch.multinomial(log_probs.exp().cpu(), branch_actions, replacement=True,
                                            generator=generator).tolist()
                targets[decision] = {"actions": sampled, "log_probs": [float(log_probs[a]) for a in sampled],
                                     "next_state": next_state}
            if decision == last:
                break
            observation, _, done, _ = env.step_discrete(decision_actions[decision])
            if done:
                raise RuntimeError("episode ended before a recorded decision; replay diverged")
            state, decision = next_state, decision + 1

    # Pass 2: one simulation per distinct action, with common random numbers.
    results = []
    for decision in chosen:
        target = targets[decision]
        crn_seed = _mix(seed, decision, 1)
        returns_by_action = {a: _branch_return(model, scenario, decision_actions[:decision], a,
                                               target["next_state"], crn_seed, skip_forced, options, device)
                             for a in dict.fromkeys(target["actions"])}
        results.append({"decision": decision, "actions": target["actions"], "log_probs": target["log_probs"],
                        "returns": [returns_by_action[a] for a in target["actions"]]})
    return results


def _branch_return(model: Any, scenario: dict, prefix: list[int], action: int, state: Any, crn_seed: int,
                   skip_forced: bool, options: dict, device: Any) -> float:
    env, observation, mapper, tensorizer = _new_env(scenario, model, options)
    remaining = list(prefix)
    with torch.no_grad():
        while True:  # replay the prefix up to the branched decision
            forced = _is_forced(mapper, observation, skip_forced)
            if forced is not None:
                observation, _, _, _ = env.step_discrete(forced)
                continue
            if not remaining:
                break
            observation, _, _, _ = env.step_discrete(remaining.pop(0))
        observation, total, done, _ = env.step_discrete(action)
        generator = torch.Generator().manual_seed(crn_seed)
        while not done:
            forced = _is_forced(mapper, observation, skip_forced)
            if forced is not None:
                observation, reward, done, _ = env.step_discrete(forced)
                total += reward
                continue
            inputs = tuple(tensorizer.to_torch(observation, device=device))
            mask = mapper.to_mask_torch(observation, device=device)
            logits, state = _forward(model, inputs, state)
            probs = torch.softmax(model._masked(logits, mask), dim=-1)[0].cpu()
            choice = int(torch.multinomial(probs, 1, generator=generator).item())
            observation, reward, done, _ = env.step_discrete(choice)
            total += reward
    return float(total)
