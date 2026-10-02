"""Unit and integration tests for GeographicTrainingEnv."""

import unittest

from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from nightjar.geographic_integrated import geographic_episode, run_geographic_integrated
from nightjar.geographic_training_env import GeographicTrainingEnv


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


if __name__ == "__main__":
    unittest.main()
