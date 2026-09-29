"""Unit and invariant tests for ObservationTensorizer, ActionMapper, and SpatialActorCritic."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from nightjar.aircraft import CAPACITY_L
from nightjar.integrated import run_integrated
from nightjar.model import ActionMapper, HAS_TORCH, ObservationTensorizer, SpatialActorCritic, TacticalActionMapper, TACTICAL_ACTION_NAMES
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
        self.assertEqual(len(raster), 10)
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

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for SpatialActorCritic tests")
    def test_recurrent_spatial_actor_critic(self):
        import torch
        model = SpatialActorCritic(
            grid_rows=self.rows,
            grid_cols=self.cols,
            recurrent=True,
            lstm_hidden_dim=64,
        )
        self.assertTrue(model.recurrent)

        # Initial state
        state = model.initial_state(batch_size=1)
        self.assertIsNotNone(state)
        h, c = state
        self.assertEqual(h.shape, (1, 64))
        self.assertEqual(c.shape, (1, 64))
        self.assertTrue(torch.all(h == 0.0))
        self.assertTrue(torch.all(c == 0.0))

        raster_t, vector_t = self.tensorizer.to_torch(self.obs)
        mask_t = self.mapper.to_mask_torch(self.obs)

        # Forward pass
        logits, value, next_state = model(raster_t, vector_t, state=state)
        self.assertEqual(logits.shape, (1, self.mapper.num_actions))
        self.assertEqual(value.shape, (1,))
        h_next, c_next = next_state
        self.assertEqual(h_next.shape, (1, 64))
        self.assertFalse(torch.all(h_next == 0.0))

        # Action sampling with masking
        action, log_prob, entropy, val_est, state_sample = model.get_action(
            raster_t, vector_t, mask_t, deterministic=False, state=state
        )
        action_idx = int(action.item())
        self.assertTrue(mask_t[0, action_idx].item())
        self.assertEqual(state_sample[0].shape, (1, 64))

        # Deterministic action with masking
        det_action, _, _, _, state_det = model.get_action(
            raster_t, vector_t, mask_t, deterministic=True, state=state
        )
        det_action_idx = int(det_action.item())
        self.assertTrue(mask_t[0, det_action_idx].item())

        # Sequence BPTT gradient test over 3 timesteps
        raster_seq = torch.cat([raster_t, raster_t, raster_t], dim=0)
        vector_seq = torch.cat([vector_t, vector_t, vector_t], dim=0)
        mask_seq = torch.cat([mask_t, mask_t, mask_t], dim=0)
        actions_seq = torch.tensor([0, 0, 0], dtype=torch.long)

        log_probs, entropies, values = model.evaluate_actions(
            raster_seq, vector_seq, mask_seq, actions_seq
        )
        self.assertEqual(log_probs.shape, (3,))
        self.assertEqual(entropies.shape, (3,))
        self.assertEqual(values.shape, (3,))

        loss = -log_probs.sum() + values.sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for SpatialActorCritic tests")
    def test_end_to_end_episode_with_recurrent_policy(self):
        import torch
        torch.manual_seed(42)
        model = SpatialActorCritic(
            grid_rows=self.rows,
            grid_cols=self.cols,
            recurrent=True,
            lstm_hidden_dim=64,
        )
        env = IntegratedTrainingEnv(self.scenario)
        obs = env.reset()

        step_count = 0
        done = False
        state = None
        while not done:
            action_idx, state = model.select_action(
                obs, self.tensorizer, self.mapper, deterministic=True, state=state
            )
            obs, reward, done, info = env.step_discrete(action_idx)
            step_count += 1

        self.assertTrue(done)
        self.assertGreater(step_count, 0)
        replay = env.replay
        self.assertIn("affected_area_ha", replay)
        self.assertEqual(replay["aircraft_metrics"]["rejected_orders"], 0)

    def test_tactical_action_mapper_and_action_mode(self):
        tactical_mapper = TacticalActionMapper(self.cell_size_m)
        self.assertEqual(tactical_mapper.num_actions, 9)
        self.assertEqual(tactical_mapper.action_names, TACTICAL_ACTION_NAMES)

        # ActionMapper with action_mode="tactical"
        mapper = ActionMapper(self.rows, self.cols, self.cell_size_m, action_mode="tactical")
        self.assertEqual(mapper.num_actions, 9)
        self.assertEqual(mapper.action_mode, "tactical")

        # compute_mask when idle with water
        mask = mapper.compute_mask(self.obs)
        self.assertEqual(len(mask), 9)
        self.assertTrue(mask[0])  # HOLD
        self.assertTrue(mask[2])  # ATTACK_HEAD_NEAREST (candidates exist in example)
        self.assertTrue(mask[4])  # ATTACK_CONTAINMENT_NEAREST

        # Resolution of tactical orders
        hold_order = mapper.to_order(0, observation=self.obs)
        self.assertEqual(hold_order, {"type": "HOLD"})

        refill_order = mapper.to_order(1, observation=self.obs)
        self.assertEqual(refill_order, {"type": "RETURN_TO_REFILL"})

        for act_idx in range(2, 9):
            drop_order = mapper.to_order(act_idx, observation=self.obs)
            self.assertEqual(drop_order["type"], "DROP")
            self.assertIn("target_m", drop_order)
            self.assertEqual(len(drop_order["target_m"]), 2)

        # Busy aircraft mask
        busy_obs = deepcopy(self.obs)
        busy_obs["aircraft"]["phase"] = "transit_to_drop"
        busy_mask = mapper.compute_mask(busy_obs)
        self.assertEqual(busy_mask, [True] + [False] * 8)

        # Empty water mask
        empty_obs = deepcopy(self.obs)
        empty_obs["aircraft"]["water_l"] = 0
        empty_mask = mapper.compute_mask(empty_obs)
        self.assertEqual(empty_mask, [False, True] + [False] * 7)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for SpatialActorCritic tests")
    def test_end_to_end_episode_with_tactical_recurrent_policy(self):
        import torch
        torch.manual_seed(42)
        model = SpatialActorCritic(
            grid_rows=self.rows,
            grid_cols=self.cols,
            recurrent=True,
            lstm_hidden_dim=64,
            action_dim=len(TACTICAL_ACTION_NAMES),
        )
        setattr(model, "action_mode", "tactical")
        env = IntegratedTrainingEnv(self.scenario, action_mode="tactical")
        obs = env.reset()
        mapper = env.action_mapper()

        step_count = 0
        done = False
        state = None
        while not done:
            action_idx, state = model.select_action(
                obs, self.tensorizer, mapper, deterministic=True, state=state
            )
            obs, reward, done, info = env.step_discrete(action_idx)
            step_count += 1

        self.assertTrue(done)
        self.assertGreater(step_count, 0)
        replay = env.replay
        self.assertIn("affected_area_ha", replay)
        self.assertEqual(replay["aircraft_metrics"]["rejected_orders"], 0)


if __name__ == "__main__":
    unittest.main()
