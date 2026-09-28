"""Headless step/reset interface for the provisional synthetic episode.

Transitions are reconstructed by deterministic replay of the episode's action
prefix. Policies receive only the observation and action mask returned here.
"""

from __future__ import annotations

from copy import deepcopy

from .aircraft import CAPACITY_L
from .integrated import run_integrated
from .policies import candidate_cells


def valid_actions(observation: dict) -> list[dict]:
    """List orders available from delivered information and aircraft state."""
    actions = [{"type": "HOLD"}]
    aircraft = observation["aircraft"]
    if aircraft["phase"] == "idle":
        if aircraft["water_l"] < CAPACITY_L:
            actions.append({"type": "RETURN_TO_REFILL"})
        if aircraft["water_l"] > 0:
            cell_size = observation["cell_size_m"]
            actions.extend(
                {"type": "DROP", "target_m": [c * cell_size, r * cell_size]}
                for r, c in candidate_cells(observation)
            )
    return actions


class IntegratedTrainingEnv:
    """A deterministic, observation-only environment over an integrated scenario.

    Reward is the negative change in cumulative burned area, in hectares,
    between consecutive decision boundaries. Thus zero means no newly burned
    area in that interval and larger (less negative) rewards mean less spread.
    """

    reward_units = "ha"
    reward_definition = "-(cumulative_burned_area_t_ha - cumulative_burned_area_previous_ha)"

    def __init__(self, scenario: dict):
        self._scenario = deepcopy(scenario)
        self._actions: list[dict] = []
        self._observation: dict | None = None
        self._replay: dict | None = None
        self._area_ha = 0.0
        self._terminated = False

    def _run_prefix(self) -> dict:
        return run_integrated(
            self._scenario,
            "no_intervention",
            _actions=self._actions,
            _stop_after_steps=len(self._actions),
        )

    def _mask(self, observation: dict) -> list[dict]:
        return valid_actions(observation)

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
        if seed is not None and seed != self._scenario["seed"]:
            raise ValueError("reset seed must match the scenario seed for reproducible replay")
        self._actions = []
        self._terminated = False
        self._replay = self._run_prefix()
        self._observation = deepcopy(self._replay["pending_observation"])
        self._area_ha = self._replay["affected_area_ha"]
        return deepcopy(self._observation)

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Apply an action from the current mask and return obs, reward, done, info."""
        if self._observation is None:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before stepping again")
        valid = self._mask(self._observation)
        if not isinstance(action, dict) or action not in valid:
            raise ValueError(f"invalid action; choose one of the actions in info['action_mask']: {valid!r}")

        previous_observation = self._observation
        self._actions.append(deepcopy(action))
        self._replay = self._run_prefix()
        new_area = self._replay["affected_area_ha"]
        reward = -(new_area - self._area_ha)
        self._area_ha = new_area
        next_observation = self._replay.get("pending_observation")
        self._terminated = next_observation is None
        if next_observation is None:
            # At the horizon there is no subsequent decision; return the last
            # delivered observation as the terminal observation.
            next_observation = previous_observation
        self._observation = deepcopy(next_observation)
        info = {
            "action_mask": self._mask(self._observation) if not self._terminated else [],
            "time_s": self._scenario["fire"]["horizon_s"] if self._terminated else self._observation["time_s"],
            "reward_units": self.reward_units,
            "reward_definition": self.reward_definition,
        }
        return deepcopy(self._observation), float(reward), self._terminated, info
