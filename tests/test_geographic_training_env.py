"""Unit and integration tests for GeographicTrainingEnv."""

from copy import deepcopy
import unittest

from pandion.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from pandion.geographic_integrated import geographic_episode, run_geographic_integrated
from pandion.geographic_training_env import GeographicTrainingEnv, valid_geographic_actions
from pandion.model import ActionMapper
from pandion.parametric_actions import ParametricActionMapper
from pandion.policies import choose_order


class TestGeographicTrainingEnv(unittest.TestCase):
    def setUp(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=35, cols=35)
        self.scenario = generate_geographic_scenario(
            fixture,
            index=0,
            split="development",
            window_rows=25,
            window_cols=25,
            horizon_s=1200.0,
        )
        self.env = GeographicTrainingEnv(self.scenario, action_mode="tactical")

    def test_reset_initializes_observation_and_mask(self):
        obs = self.env.reset()
        self.assertEqual(obs["time_s"], 0.0)
        self.assertEqual(len(obs["fire"]["cells"]), 25)
        self.assertEqual(len(obs["fire"]["cells"][0]), 25)
        mask = self.env.action_mask
        self.assertGreater(len(mask), 0)
        # ActionMapper and ObservationTensorizer factories work
        mapper = self.env.action_mapper()
        self.assertEqual(mapper.action_mode, "tactical")
        tensorizer = self.env.observation_tensorizer()
        raster, vec = tensorizer.extract_features(obs)
        self.assertEqual(len(raster), 10)
        self.assertEqual(len(vec), 10)

    def test_step_hold_advances_simulation(self):
        self.env.reset()
        obs, reward, done, info = self.env.step({"type": "HOLD"})
        self.assertFalse(done)
        self.assertGreater(obs["time_s"], 0.0)
        self.assertLessEqual(reward, 0.0)
        self.assertIn("action_mask", info)

    def test_step_discrete_tactical_macro_action(self):
        obs = self.env.reset()
        mapper = self.env.action_mapper()
        mask = mapper.compute_mask(obs)
        # Check if tactical drop (e.g. ATTACK_HEAD_NEAREST = 2) is valid
        if mask[2]:
            next_obs, reward, done, info = self.env.step_discrete(2)
            self.assertGreater(next_obs["time_s"], 0.0)
            self.assertEqual(next_obs["aircraft"]["water_l"], 0)  # dropped full load

    def test_invalid_action_raises_value_error(self):
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step({"type": "INVALID_ACTION"})

    def test_reset_seed_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.env.reset(seed=99999999)

    def test_incremental_steps_match_prefix_replays(self):
        """Persistent stepping equals replaying the action prefix from scratch."""
        import random
        rng = random.Random(5)
        obs = self.env.reset()
        actions, done, area = [], False, None
        while not done:
            action = rng.choice(self.env.action_mask)
            actions.append(action)
            obs, reward, done, _ = self.env.step(action)
            prefix = run_geographic_integrated(self.scenario, "heuristic", _actions=actions,
                                               _stop_after_steps=len(actions), _record_observations=False)
            if not done:
                self.assertEqual(obs, prefix["pending_observation"])
            if area is not None:
                drops = prefix["drops"][-1]["treated_cells_count"] if actions[-1]["type"] == "DROP" else 0
                self.assertAlmostEqual(reward, -(prefix["affected_area_ha"] - area) + 0.25 * drops, places=9)
            area = prefix["affected_area_ha"]
        self.assertEqual(self.env.replay, prefix)

    def test_returned_observations_are_independent_copies(self):
        obs = self.env.reset()
        obs["fire"]["cells"][0][0]["fire_state"] = "tampered"
        obs["known_treated_cells"].append([0, 0])
        self.assertNotEqual(self.env._observation["fire"]["cells"][0][0]["fire_state"], "tampered")
        self.assertEqual(self.env._observation["known_treated_cells"], [])

    def test_full_episodes_cannot_stop_early(self):
        episode = geographic_episode(self.scenario, "heuristic")
        next(episode)
        with self.assertRaises(ValueError):
            episode.send(None)


    def test_masks_use_the_scenario_aircraft_capacity(self):
        """A full smaller tank (e.g. AT-802F, 3100 L) cannot be refilled; a drop empties it."""
        scenario = deepcopy(self.scenario)
        scenario["aircraft"]["tank_capacity_l"] = 3100
        env = GeographicTrainingEnv(scenario, action_mode="tactical")
        obs = env.reset()
        self.assertEqual(obs["aircraft"]["capacity_l"], 3100)
        self.assertEqual(obs["aircraft"]["water_l"], 3100)
        self.assertNotIn({"type": "RETURN_TO_REFILL"}, valid_geographic_actions(obs))
        size = float(scenario["grid"]["cell_size_m"])
        rows, cols = int(scenario["grid"]["rows"]), int(scenario["grid"]["cols"])
        for mapper in (ActionMapper(rows=rows, cols=cols, cell_size_m=size, action_mode="tactical"),
                       ActionMapper(rows=rows, cols=cols, cell_size_m=size, action_mode="spatial"),
                       ParametricActionMapper(size)):
            self.assertFalse(mapper.compute_mask(obs)[1], type(mapper).__name__)
        drop = next(a for a in valid_geographic_actions(obs) if a["type"] == "DROP")
        obs, _, _, info = env.step(drop)
        while obs["aircraft"]["water_l"] > 0:
            obs, _, _, info = env.step({"type": "HOLD"})
        self.assertEqual(info["action_mask"], [{"type": "RETURN_TO_REFILL"}])

    def _run_policy(self, env, policy="heuristic"):
        """Step an episode with ``policy``; returns the per-step rewards."""
        obs = env.reset()
        rewards, done = [], False
        while not done:
            obs, reward, done, _ = env.step(choose_order(policy, obs))
            rewards.append(reward)
        return rewards

    def spreading_scenario(self):
        """Scenario whose fire spreads within the horizon, so reference areas differ
        (the 35x35 fixture above burns a constant 21 ha under every policy)."""
        fixture = build_synthetic_cems_fixture("EMSR578", rows=40, cols=40)
        return generate_geographic_scenario(fixture, index=0, split="development",
                                            window_rows=12, window_cols=12, horizon_s=3600.0)

    def test_reward_reference_heuristic_totals_zero_when_following_heuristic(self):
        scenario = self.spreading_scenario()
        env = GeographicTrainingEnv(scenario, drop_bonus_per_cell=0, reward_reference="heuristic")
        self.assertIn("heuristic_policy_area_growth", env.reward_definition)
        self.assertAlmostEqual(sum(self._run_policy(env)), 0.0, delta=1e-6)

    def test_reward_reference_no_intervention_totals_area_saved(self):
        scenario = self.spreading_scenario()
        env = GeographicTrainingEnv(scenario, drop_bonus_per_cell=0, reward_reference="no_intervention")
        total = sum(self._run_policy(env))
        no_intervention = run_geographic_integrated(scenario, "no_intervention")["affected_area_ha"]
        self.assertGreater(abs(no_intervention - env.replay["affected_area_ha"]), 0.5)
        self.assertAlmostEqual(total, no_intervention - env.replay["affected_area_ha"], delta=1e-6)

    def test_default_reward_is_unchanged_burned_area_change(self):
        env = GeographicTrainingEnv(self.scenario, drop_bonus_per_cell=0)
        self.assertEqual(env.reward_definition, GeographicTrainingEnv.reward_definition)
        self.env.reset()
        initial = self.env._area_ha
        rewards = self._run_policy(env)
        self.assertAlmostEqual(sum(rewards), -(env.replay["affected_area_ha"] - initial), delta=1e-6)

    def test_invalid_reward_reference_raises_value_error(self):
        from pandion.geographic_training_env import reference_area_curve
        with self.assertRaises(ValueError):
            GeographicTrainingEnv(self.scenario, reward_reference="random_valid")
        with self.assertRaises(ValueError):
            reference_area_curve(self.scenario, "random_valid")

if __name__ == "__main__":
    unittest.main()
