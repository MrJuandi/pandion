import copy
import json
from pathlib import Path
import unittest

from pandion.training_env import IntegratedTrainingEnv
from pandion.integrated import run_integrated


SCENARIO = Path(__file__).resolve().parents[1] / "examples" / "synthetic_integrated.json"


class IntegratedTrainingEnvTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads(SCENARIO.read_text())

    def test_reset_and_action_prefix_replay_are_deterministic(self):
        env = IntegratedTrainingEnv(self.scenario)
        observation = env.reset(seed=42)
        mask = env.action_mask
        self.assertIn({"type": "HOLD"}, mask)
        drop = next(action for action in mask if action["type"] == "DROP")
        first = env.step(drop)

        repeated = IntegratedTrainingEnv(copy.deepcopy(self.scenario))
        self.assertEqual(repeated.reset(seed=42), observation)
        self.assertEqual(repeated.step(drop), first)
        self.assertEqual(first[3]["reward_units"], "ha")
        self.assertEqual(
            first[3]["reward_definition"],
            "-(cumulative_burned_area_t_ha - cumulative_burned_area_previous_ha)",
        )

    def test_invalid_or_stale_action_fails(self):
        env = IntegratedTrainingEnv(self.scenario)
        observation = env.reset()
        action_mask = env.action_mask
        with self.assertRaisesRegex(ValueError, "action_mask"):
            env.step({"type": "DROP", "target_m": [0, 0]})
        self.assertEqual(env.action_mask, action_mask)
        self.assertEqual(env._observation, observation)
        self.assertEqual(env._actions, [])
        with self.assertRaisesRegex(ValueError, "scenario seed"):
            env.reset(seed=43)

    def test_hold_rewards_and_final_replay_match_full_integrated_run(self):
        env = IntegratedTrainingEnv(self.scenario)
        env.reset()
        initial_area = len(self.scenario["fire"]["ignitions"]) * self.scenario["fire"]["cell_size_m"] ** 2 / 10_000
        rewards = []
        while True:
            observation, reward, terminated, info = env.step({"type": "HOLD"})
            rewards.append(reward)
            self.assert_no_latent_area(observation)
            self.assert_no_latent_area(info)
            if terminated:
                break
        self.assertEqual(info["action_mask"], [])
        self.assertEqual(info["time_s"], self.scenario["fire"]["horizon_s"])
        full = run_integrated(self.scenario, "no_intervention")
        replay = env.replay
        self.assertEqual(replay["affected_area_ha"], full["affected_area_ha"])
        self.assertEqual(replay["decisions"], full["decisions"])
        self.assertAlmostEqual(sum(rewards), -(full["affected_area_ha"] - initial_area))
        replay["decisions"].clear()
        self.assertTrue(env.replay["decisions"])
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            env.step({"type": "HOLD"})

    def test_observation_info_and_mask_exclude_latent_fire_area(self):
        env = IntegratedTrainingEnv(self.scenario)
        observation = env.reset()
        self.assert_no_latent_area(observation)
        self.assert_no_latent_area(env.action_mask)
        _, _, _, info = env.step({"type": "HOLD"})
        self.assert_no_latent_area(info)
        with self.assertRaisesRegex(RuntimeError, "after the episode terminates"):
            _ = env.replay

    def assert_no_latent_area(self, value):
        forbidden = {"affected_area_ha", "arrival_time_s", "cumulative_burned_area_ha", "water_barrier_cells"}
        if isinstance(value, dict):
            self.assertTrue(forbidden.isdisjoint(value))
            for nested in value.values():
                self.assert_no_latent_area(nested)
        elif isinstance(value, list):
            for nested in value:
                self.assert_no_latent_area(nested)

    def test_empty_idle_aircraft_mask_requires_refill(self):
        env = IntegratedTrainingEnv(self.scenario)
        obs = env.reset()
        obs["aircraft"]["phase"] = "idle"
        obs["aircraft"]["water_l"] = 0.0
        mask = env._mask(obs)
        self.assertEqual(mask, [{"type": "RETURN_TO_REFILL"}])

    def test_drop_bonus_per_cell(self):
        env_standard = IntegratedTrainingEnv(self.scenario, drop_bonus_per_cell=0.0)
        env_bonus = IntegratedTrainingEnv(self.scenario, drop_bonus_per_cell=1.0)
        env_standard.reset(seed=42)
        env_bonus.reset(seed=42)
        drop = next(action for action in env_standard.action_mask if action["type"] == "DROP")
        _, r_std, _, _ = env_standard.step(drop)
        _, r_bon, _, _ = env_bonus.step(drop)
        # Drop successfully creates 1 barrier cell in the synthetic example
        self.assertAlmostEqual(r_bon, r_std + 1.0)


if __name__ == "__main__":
    unittest.main()
