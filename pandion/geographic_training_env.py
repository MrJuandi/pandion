"""Headless step/reset interface for real-world geographic wildfire scenarios.

Couples real-incident CEMS scenarios (Scott & Burgan 40 fuels, georeferenced waterbodies)
with CL-415 aerial operations and continuous-time Rothermel fire spread.
"""

from __future__ import annotations

from copy import deepcopy
import bisect
import pickle
from typing import Any, Callable

from .aircraft import tank_capacity_l
from .geographic_integrated import geographic_episode, run_geographic_integrated
from .parametric_actions import MAX_BURNING_FRONTIER, FrontierActionMapper
from .policies import candidate_cells

REWARD_REFERENCES = ("heuristic", "no_intervention")
_REFERENCE_CACHE: dict[tuple[str, str], tuple[list[float], float]] = {}


def reference_area_curve(scenario: dict, policy: str) -> Callable[[float], float]:
    """Burned area (ha) of ``policy`` in this scenario as a function of time.
    The reference episode never depends on the learner; cached per scenario hash."""
    if policy not in REWARD_REFERENCES:
        raise ValueError(f"reward_reference must be one of {REWARD_REFERENCES}, got {policy!r}")
    key = (str(scenario.get("scenario_hash") or scenario.get("scenario_id")), policy)
    if key not in _REFERENCE_CACHE:
        if len(_REFERENCE_CACHE) > 4096:
            _REFERENCE_CACHE.clear()
        replay = run_geographic_integrated(scenario, policy=policy, _record_observations=False)
        times = sorted(t for row in replay["arrival_time_s"] for t in row if t is not None)
        cell_ha = float(scenario["grid"]["cell_size_m"]) ** 2 / 10000.0
        _REFERENCE_CACHE[key] = (times, cell_ha)
    times, cell_ha = _REFERENCE_CACHE[key]
    return lambda t: bisect.bisect_right(times, t + 1e-9) * cell_ha


def _copy(value: Any) -> Any:
    """Defensive deep copy; pickle is several times faster than deepcopy for
    full-grid observations of plain dicts, lists and numbers."""
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def valid_geographic_actions(observation: dict) -> list[dict]:
    """List valid simulator orders available from delivered information and aircraft state."""
    aircraft = observation["aircraft"]
    if aircraft["phase"] != "idle":
        return [{"type": "HOLD"}]
    if aircraft["water_l"] <= 0:
        return [{"type": "RETURN_TO_REFILL"}]
    actions = [{"type": "HOLD"}]
    if aircraft["water_l"] < tank_capacity_l(aircraft):
        actions.append({"type": "RETURN_TO_REFILL"})
    cell_size = float(observation["cell_size_m"])
    actions.extend(
        {"type": "DROP", "target_m": [c * cell_size, r * cell_size]}
        for r, c in candidate_cells(observation)
    )
    return actions


class GeographicTrainingEnv:
    """A deterministic, observation-only training environment over a real geographic scenario.

    Reward is the negative change in cumulative burned area, in hectares,
    between consecutive decision boundaries, plus an optional drop bonus
    for newly blocked cells on the active fire perimeter. With
    ``reward_reference`` (D-81) the reward also adds the reference policy's
    area growth over the same interval, so a full episode sums to the
    reference's final area minus the policy's final area.
    """

    reward_units = "ha"
    reward_definition = "-(cumulative_burned_area_t_ha - cumulative_burned_area_previous_ha)"

    def __init__(
        self,
        scenario: dict[str, Any],
        *,
        drop_bonus_per_cell: float = 0.25,
        action_mode: str = "tactical",
        drop_distances: Any = (0,),
        reward_reference: str | None = None,
    ):
        if reward_reference is not None and reward_reference not in REWARD_REFERENCES:
            raise ValueError(f"reward_reference must be one of {REWARD_REFERENCES}, got {reward_reference!r}")
        self._scenario = deepcopy(scenario)
        self._drop_distances = tuple(drop_distances)
        self._drop_bonus_per_cell = float(drop_bonus_per_cell)
        self._action_mode = action_mode
        self._reward_reference = reward_reference
        self._burning_frontier = action_mode == "frontier" and bool(scenario.get("water_model", {}).get("burning_cells"))
        self.frontier_truncated_decisions = 0
        self._reference: Callable[[float], float] | None = None
        self._reference_time = 0.0
        if reward_reference is not None:
            self.reward_definition += f" + {reward_reference}_policy_area_growth_over_the_same_interval_ha"
        self._episode: Any = None
        self._observation: dict | None = None
        self._replay: dict | None = None
        self._area_ha = 0.0
        self._drop_count = 0
        self._terminated = False

    def _advance(self, order: dict | None) -> tuple[dict | None, float, list[dict]]:
        """Advance the persistent episode; returns (observation or None, area_ha, drops).

        The simulator state is kept between decisions, so a step costs one
        decision interval instead of a replay of the whole prefix. Results are
        identical to replaying the prefix with ``run_geographic_integrated``.
        """
        try:
            observation, area_ha, drops = (next(self._episode) if order is None
                                           else self._episode.send(order))
            return _copy(observation), area_ha, drops
        except StopIteration as finished:
            self._replay = finished.value
            return None, self._replay["affected_area_ha"], self._replay.get("drops", [])

    def _mask(self, observation: dict) -> list[dict]:
        valid = valid_geographic_actions(observation)
        if self._action_mode == "parametric" and self._drop_distances != (0,):
            # D-79: drops beyond the front aim at cells that are not fire-adjacent candidates.
            mapper = self.action_mapper()._tactical
            if observation["aircraft"]["phase"] == "idle" and observation["aircraft"]["water_l"] > 0:
                extra = [{"type": "DROP", "target_m": [c * mapper.cell_size_m, r * mapper.cell_size_m]}
                         for r, c in mapper.targets(observation).values()]
                valid += [order for order in extra if order not in valid]
        return valid

    @property
    def action_mask(self) -> list[dict]:
        """Valid actions for the observation most recently returned."""
        if self._observation is None:
            raise RuntimeError("call reset() before reading action_mask")
        if self._terminated:
            return []
        return self._mask(self._observation)

    @property
    def replay(self) -> dict:
        """Return a defensive copy of the raw simulator replay after termination."""
        if not self._terminated or self._replay is None:
            raise RuntimeError("replay is available after the episode terminates")
        return deepcopy(self._replay)

    def reset(self, *, seed: int | None = None) -> dict:
        """Reset to the scenario seed; a supplied seed must match its metadata."""
        if seed is not None and seed != self._scenario.get("seed"):
            raise ValueError("reset seed must match scenario seed for reproducible replay")
        self._terminated = False
        self._replay = None
        self.frontier_truncated_decisions = 0
        self._episode = geographic_episode(self._scenario, "heuristic", truncated=True,
                                           record_observations=False)
        observation, self._area_ha, drops = self._advance(None)
        self._drop_count = len(drops)
        self._terminated = observation is None
        self._observation = observation
        self._reference = (reference_area_curve(self._scenario, self._reward_reference)
                           if self._reward_reference else None)
        self._reference_time = float(observation["time_s"]) if observation is not None else 0.0
        return _copy(self._observation)

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Apply an action from current mask and return (obs, reward, done, info)."""
        if self._observation is None:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before stepping again")
        valid = self._mask(self._observation)
        if not isinstance(action, dict) or action not in valid:
            raise ValueError(f"invalid action; choose one of the actions in info['action_mask']: {valid!r}")
        if self._burning_frontier:
            self.frontier_truncated_decisions += sum(order["type"] == "DROP" for order in valid) > MAX_BURNING_FRONTIER

        previous_observation = self._observation
        next_observation, new_area, new_drops = self._advance(deepcopy(action))
        reward = -(new_area - self._area_ha)
        if self._reference is not None:
            end = (float(next_observation["time_s"]) if next_observation is not None
                   else float(self._scenario["fire"]["horizon_s"]))
            reward += self._reference(end) - self._reference(self._reference_time)
            self._reference_time = end
        self._area_ha = new_area

        if self._drop_bonus_per_cell > 0 and len(new_drops) > self._drop_count:
            for drop_record in new_drops[self._drop_count:]:
                treated = len(drop_record.get("treated_cells", []))
                reward += treated * self._drop_bonus_per_cell
        self._drop_count = len(new_drops)

        self._terminated = next_observation is None
        if next_observation is None:
            next_observation = previous_observation
        self._observation = next_observation  # already a private copy from _advance

        info = {
            "action_mask": self._mask(self._observation) if not self._terminated else [],
            "time_s": self._scenario["fire"]["horizon_s"] if self._terminated else self._observation["time_s"],
            "reward_units": self.reward_units,
            "reward_definition": self.reward_definition,
        }
        return _copy(self._observation), float(reward), self._terminated, info

    def action_mapper(self) -> Any:
        """Return an ActionMapper matched to this geographic scenario's dimensions."""
        from .model import ActionMapper
        grid = self._scenario["grid"]
        if self._burning_frontier:
            return FrontierActionMapper(grid["cell_size_m"], self._drop_distances, ceiling=MAX_BURNING_FRONTIER)
        return ActionMapper(
            rows=int(grid["rows"]),
            cols=int(grid["cols"]),
            cell_size_m=float(grid["cell_size_m"]),
            action_mode=self._action_mode,
            drop_distances=self._drop_distances,
        )

    def observation_tensorizer(self) -> Any:
        """Return an ObservationTensorizer matched to this geographic scenario."""
        from .model import ObservationTensorizer
        grid = self._scenario["grid"]
        fire = self._scenario["fire"]
        return ObservationTensorizer(
            rows=int(grid["rows"]),
            cols=int(grid["cols"]),
            cell_size_m=float(grid["cell_size_m"]),
            horizon_s=float(fire["horizon_s"]),
        )

    def step_discrete(self, action_index: int) -> tuple[dict, float, bool, dict]:
        """Apply a discrete action index from ActionMapper and advance one step."""
        mapper = self.action_mapper()
        order = mapper.to_order(action_index, observation=self._observation)
        return self.step(order)
