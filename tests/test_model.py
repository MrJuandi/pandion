"""Unit and invariant tests for ObservationTensorizer, ActionMapper, and SpatialActorCritic."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from nightjar.aircraft import CAPACITY_L
from nightjar.integrated import run_integrated
from nightjar.model import ActionMapper, HAS_TORCH, ObservationTensorizer, SpatialActorCritic
from nightjar.policies import candidate_cells
from nightjar.training_env import IntegratedTrainingEnv, valid_actions

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "synthetic_integrated.json"


class ModelTests(unittest.TestCase):
    def setUp(self):
        with open(EXAMPLE_PATH) as f:
            self.scenario = json.load(f)
        self.env = IntegratedTrainingEnv(self.scenario)
        self.obs = self.env.reset()
        self.fuel = self.scenario["fire"]["fuel_model"]
        self.rows = len(self.fuel)
        self.cols = len(self.fuel[0])
        self.cell_size_m = float(self.scenario["fire"]["cell_size_m"])
        self.horizon_s = float(self.scenario["fire"]["horizon_s"])
        self.tensorizer = ObservationTensorizer(
            self.rows, self.cols, self.cell_size_m, self.horizon_s
        )
        self.mapper = ActionMapper(self.rows, self.cols, self.cell_size_m)

    def test_tensorizer_shapes_and_ranges(self):
        raster, vector = self.tensorizer.extract_features(self.obs)
        self.assertEqual(len(raster), 9)
        self.assertEqual(len(raster[0]), self.rows)
        self.assertEqual(len(raster[0][0]), self.cols)
        self.assertEqual(len(vector), ObservationTensorizer.NUM_VECTOR_FEATURES)

        # Check values are finite and non-negative
        for c in range(len(raster)):
            for r in range(self.rows):
                for col in range(self.cols):
                    val = raster[c][r][col]
                    self.assertTrue(0.0 <= val <= 1.0, f"raster channel {c} val {val} out of range [0, 1]")

        for i, val in enumerate(vector):
            self.assertTrue(-1.0 <= val <= 1.0 or val >= 0.0, f"vector idx {i} val {val} invalid")

    def test_tensorizer_invariance_to_hidden_state_leakage(self):
        first_obs = deepcopy(self.obs)
        second_obs = deepcopy(self.obs)

        # Add mock latent fields that should never be read by the tensorizer
        second_obs["_latent_arrival_times"] = [[0.0] * self.cols for _ in range(self.rows)]
        second_obs["_true_fire_state"] = "SECRET_FUTURE"

        r1, v1 = self.tensorizer.extract_features(first_obs)
        r2, v2 = self.tensorizer.extract_features(second_obs)
        self.assertEqual(r1, r2)
        self.assertEqual(v1, v2)

    def test_tensorizer_channel_semantics(self):
        # Candidate channel must match candidate_cells(observation)
        expected_candidates = {tuple(p) for p in candidate_cells(self.obs)}
        raster, _ = self.tensorizer.extract_features(self.obs)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in expected_candidates:
                    self.assertEqual(raster[5][r][c], 1.0)
                else:
                    self.assertEqual(raster[5][r][c], 0.0)


    def test_integrated_observation_exposes_public_elevation_copy(self):
        replay = run_integrated(self.scenario, "no_intervention", _stop_after_steps=0)
        observation = replay["pending_observation"]
        self.assertEqual(observation["elevation_m"], self.scenario["fire"]["elevation_m"])
        observation["elevation_m"][0][0] = 999.0
        self.assertNotEqual(observation["elevation_m"], self.scenario["fire"]["elevation_m"])

    def test_topographic_channels_flat_grid_are_zero(self):
        obs = deepcopy(self.obs)
        obs["elevation_m"] = [[5.0] * self.cols for _ in range(self.rows)]
        raster, _ = self.tensorizer.extract_features(obs)
        self.assertTrue(all(value == 0.0 for row in raster[7] for value in row))
        self.assertTrue(all(value == 0.0 for row in raster[8] for value in row))

    def test_topographic_channels_ramp_edges_and_range(self):
        obs = deepcopy(self.obs)
        obs["elevation_m"] = [[float(c * self.cell_size_m) for c in range(self.cols)] for _ in range(self.rows)]
        raster, _ = self.tensorizer.extract_features(obs)
        self.assertEqual(raster[7][0][0], 0.0)
        self.assertEqual(raster[7][self.rows - 1][self.cols - 1], 1.0)
        for r in range(self.rows):
            for c in range(self.cols):
                self.assertTrue(0.0 <= raster[7][r][c] <= 1.0)
                self.assertEqual(raster[8][r][c], 1.0)

    def test_topographic_channels_non_linear_grid_use_centered_interior_and_one_sided_edges(self):
        obs = deepcopy(self.obs)
        obs["cell_size_m"] = 100.0
        obs["elevation_m"] = [
            [0.0, 10.0, 40.0],
            [5.0, 25.0, 65.0],
            [20.0, 50.0, 100.0],
        ]
        tensorizer = ObservationTensorizer(3, 3, 100.0, self.horizon_s)
        raster, _ = tensorizer.extract_features(obs)

        self.assertEqual(raster[7], [
            [0.0, 0.1, 0.4],
            [0.05, 0.25, 0.65],
            [0.2, 0.5, 1.0],
        ])
        self.assertAlmostEqual(raster[8][1][1], math.hypot(0.3, 0.2))
        self.assertAlmostEqual(raster[8][0][0], math.hypot(0.1, 0.05))
        self.assertAlmostEqual(raster[8][0][1], math.hypot(0.2, 0.15))
        self.assertAlmostEqual(raster[8][1][0], math.hypot(0.2, 0.1))

    def test_topographic_slope_handles_degenerate_dimensions(self):
        obs = deepcopy(self.obs)
        obs["elevation_m"] = [[0.0, 50.0, 100.0]]
        tensorizer = ObservationTensorizer(1, 3, 50.0, self.horizon_s)
        raster, _ = tensorizer.extract_features(obs)
        self.assertEqual(raster[8][0], [1.0, 1.0, 1.0])

        obs = deepcopy(self.obs)
        obs["elevation_m"] = [[0.0], [25.0], [50.0]]
        tensorizer = ObservationTensorizer(3, 1, 50.0, self.horizon_s)
        raster, _ = tensorizer.extract_features(obs)
        self.assertEqual([row[0] for row in raster[8]], [0.5, 0.5, 0.5])

    def test_topographic_channels_do_not_read_latent_elevation(self):
        first_obs = deepcopy(self.obs)
        second_obs = deepcopy(self.obs)
        second_obs["_latent_elevation_m"] = [[9999.0] * self.cols for _ in range(self.rows)]
        r1, v1 = self.tensorizer.extract_features(first_obs)
        r2, v2 = self.tensorizer.extract_features(second_obs)
        self.assertEqual(r1, r2)
        self.assertEqual(v1, v2)

    def test_action_mapper_bijection_and_bounds(self):
        self.assertEqual(self.mapper.to_order(0), {"type": "HOLD"})
        self.assertEqual(self.mapper.to_action_index({"type": "HOLD"}), 0)

        self.assertEqual(self.mapper.to_order(1), {"type": "RETURN_TO_REFILL"})
        self.assertEqual(self.mapper.to_action_index({"type": "RETURN_TO_REFILL"}), 1)

        # Test grid drop bijection
        for r in range(self.rows):
            for c in range(self.cols):
                action_idx = 2 + r * self.cols + c
                order = self.mapper.to_order(action_idx)
                self.assertEqual(order["type"], "DROP")
                self.assertAlmostEqual(order["target_m"][0], c * self.cell_size_m)
                self.assertAlmostEqual(order["target_m"][1], r * self.cell_size_m)
                self.assertEqual(self.mapper.to_action_index(order), action_idx)

        # Out of bounds checks
        with self.assertRaises(IndexError):
            self.mapper.to_order(-1)
        with self.assertRaises(IndexError):
            self.mapper.to_order(self.mapper.num_actions)
        with self.assertRaises(ValueError):
            self.mapper.to_action_index({"type": "UNKNOWN"})
        with self.assertRaises(ValueError):
            self.mapper.to_action_index({"type": "DROP", "target_m": [99999.0, 99999.0]})

    def test_action_mask_exact_agreement_with_valid_actions(self):
        mask = self.mapper.compute_mask(self.obs)
        valid = valid_actions(self.obs)
        valid_indices = {self.mapper.to_action_index(a) for a in valid}

        # Check mask has True exactly where action is in valid_actions
        for a_idx, is_valid in enumerate(mask):
            if is_valid:
                self.assertIn(a_idx, valid_indices)
                self.assertIn(self.mapper.to_order(a_idx), valid)
            else:
                self.assertNotIn(a_idx, valid_indices)

    def test_env_discrete_stepping_and_convenience_methods(self):
        env = IntegratedTrainingEnv(self.scenario)
        env.reset()
        mapper = env.action_mapper()
        tensorizer = env.observation_tensorizer()
        self.assertIsInstance(mapper, ActionMapper)
        self.assertIsInstance(tensorizer, ObservationTensorizer)

        # Step HOLD (discrete index 0)
        next_obs, reward, done, info = env.step_discrete(0)
        self.assertIn("action_mask", info)
        self.assertFalse(done)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for SpatialActorCritic tests")
    def test_spatial_actor_critic_forward_and_masking(self):
        import torch
        model = SpatialActorCritic(
            grid_rows=self.rows,
            grid_cols=self.cols,
        )

        raster_t, vector_t = self.tensorizer.to_torch(self.obs)
        mask_t = self.mapper.to_mask_torch(self.obs)

        self.assertEqual(raster_t.shape, (1, ObservationTensorizer.NUM_RASTER_CHANNELS, self.rows, self.cols))
        self.assertEqual(vector_t.shape, (1, ObservationTensorizer.NUM_VECTOR_FEATURES))
        self.assertEqual(mask_t.shape, (1, self.mapper.num_actions))

        # Forward pass
        logits, value = model(raster_t, vector_t)
        self.assertEqual(logits.shape, (1, self.mapper.num_actions))
        self.assertEqual(value.shape, (1,))

        # Action sampling with masking
        action, log_prob, entropy, val_est = model.get_action(raster_t, vector_t, mask_t, deterministic=False)
        action_idx = int(action.item())
        self.assertTrue(mask_t[0, action_idx].item(), "Sampled action must be valid according to mask")

        # Deterministic action with masking
        det_action, _, _, _ = model.get_action(raster_t, vector_t, mask_t, deterministic=True)
        det_action_idx = int(det_action.item())
        self.assertTrue(mask_t[0, det_action_idx].item(), "Deterministic action must be valid according to mask")

        # Action evaluation
        eval_log_prob, eval_entropy, eval_val = model.evaluate_actions(raster_t, vector_t, mask_t, action)
        self.assertEqual(eval_log_prob.shape, (1,))
        self.assertEqual(eval_entropy.shape, (1,))
        self.assertEqual(eval_val.shape, (1,))

        # Gradient flow test
        loss = -eval_log_prob.mean() + eval_val.mean()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for SpatialActorCritic tests")
    def test_end_to_end_episode_with_neural_policy(self):
        import torch
        torch.manual_seed(42)
        model = SpatialActorCritic(grid_rows=self.rows, grid_cols=self.cols)
        env = IntegratedTrainingEnv(self.scenario)
        obs = env.reset()

        step_count = 0
        done = False
        while not done:
            action_idx = model.select_action(obs, self.tensorizer, self.mapper, deterministic=True)
            obs, reward, done, info = env.step_discrete(action_idx)
            step_count += 1

        self.assertTrue(done)
        self.assertGreater(step_count, 0)
        replay = env.replay
        self.assertIn("affected_area_ha", replay)
        self.assertEqual(replay["aircraft_metrics"]["rejected_orders"], 0)


if __name__ == "__main__":
    unittest.main()
