"""Tests for D-49 image policy inputs, the image actor-critic and image PPO rollouts."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nightjar.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario
from nightjar.geographic_integrated import run_geographic_integrated
from nightjar.imagery_acquire import SCHEMA_VERSION
from nightjar.model import HAS_TORCH, ActionMapper
from nightjar.policy_inputs import FUEL_FAMILIES, GeographicInputTensorizer, elevation_gradient, fuel_family

PX = 4


def make_scenario():
    fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=40, cols=40)
    return generate_geographic_scenario(fixture, index=0, split="development",
                                        window_rows=12, window_cols=12, horizon_s=900.0)


def write_imagery(directory: Path, scenario: dict, image: np.ndarray) -> None:
    name = scenario["activation_id"].lower() + ".npy"
    np.save(directory / name, image)
    g = scenario["grid"]
    manifest = {"schema_version": SCHEMA_VERSION, "activations": {scenario["activation_id"]: {
        "crs": scenario["crs"]["code"], "px_per_cell": PX, "array": name,
        "array_sha256": sha256((directory / name).read_bytes()).hexdigest(),
        "grid": {k: g[k] for k in ("rows", "cols", "cell_size_m", "origin_utm_easting_m", "origin_utm_northing_m")}}}}
    (directory / "sources.json").write_text(json.dumps(manifest))


class PolicyInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = make_scenario()
        cls.rows = int(cls.scenario["grid"]["rows"])
        cls.replay = run_geographic_integrated(cls.scenario, policy="heuristic")
        cls.base = np.full((cls.rows * PX, cls.rows * PX, 3), 80, dtype=np.uint8)

    def test_fuel_families_cover_scott_burgan_and_anderson(self):
        self.assertEqual(FUEL_FAMILIES[fuel_family(0)], "nonburnable")
        self.assertEqual(FUEL_FAMILIES[fuel_family(98)], "nonburnable")
        self.assertEqual(FUEL_FAMILIES[fuel_family(102)], "grass")
        self.assertEqual(FUEL_FAMILIES[fuel_family(122)], "grass_shrub")
        self.assertEqual(FUEL_FAMILIES[fuel_family(145)], "shrub")
        self.assertEqual(FUEL_FAMILIES[fuel_family(165)], "timber_understory")
        self.assertEqual(FUEL_FAMILIES[fuel_family(183)], "timber_litter")
        self.assertEqual(FUEL_FAMILIES[fuel_family(203)], "slash_blowdown")
        self.assertEqual(FUEL_FAMILIES[fuel_family(1)], "grass")

    def test_gradient_points_east_and_north_in_grid_convention(self):
        elevation = np.add.outer(np.arange(3) * 10.0, np.arange(4) * 5.0)  # rises north and east
        gx, gy = elevation_gradient(elevation, 100.0)
        np.testing.assert_allclose(gx, 0.05)
        np.testing.assert_allclose(gy, 0.10)

    def test_shapes_and_channel_contents(self):
        obs = self.replay["decisions"][-1]["observation"]
        tz = GeographicInputTensorizer(self.scenario, self.base)
        rgb, cells, vector = tz.extract(obs)
        self.assertEqual(rgb.shape, (3, self.rows * PX, self.rows * PX))
        self.assertEqual(cells.shape, (tz.num_cell_channels, self.rows, self.rows))
        self.assertEqual(vector.shape, (tz.NUM_VECTOR_FEATURES,))
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        fuel_one_hot = cells[3:3 + len(FUEL_FAMILIES)]
        np.testing.assert_allclose(fuel_one_hot.sum(axis=0), 1.0)
        treated = cells[3 + len(FUEL_FAMILIES)]
        for r, c in obs["known_treated_cells"]:
            self.assertEqual(treated[r, c], 1.0)
        self.assertEqual(treated.sum(), len(obs["known_treated_cells"]))
        distance = cells[4 + len(FUEL_FAMILIES)]
        x, y = obs["aircraft"]["position_m"]
        size = self.scenario["grid"]["cell_size_m"]
        nearest = np.unravel_index(np.argmin(distance), distance.shape)
        self.assertEqual(nearest, (min(max(round(y / size), 0), self.rows - 1), min(max(round(x / size), 0), self.rows - 1)))
        self.assertAlmostEqual(vector[0] * 10.0, obs["midflame_wind_to_m_s"][0], places=5)

    def test_fire_state_reaches_inputs_only_through_rgb(self):
        obs = deepcopy(self.replay["decisions"][-1]["observation"])
        tz = GeographicInputTensorizer(self.scenario, self.base)
        rgb, cells, vector = tz.extract(obs)
        for row in obs["fire"]["cells"]:
            for cell in row:
                cell["fire_state"] = "unburned"
        rgb2, cells2, vector2 = tz.extract(obs)
        self.assertFalse(np.array_equal(rgb, rgb2))
        np.testing.assert_array_equal(cells, cells2)
        np.testing.assert_array_equal(vector, vector2)

    def test_optional_ros_channel_is_zero_on_nonburnable(self):
        obs = self.replay["decisions"][0]["observation"]
        tz = GeographicInputTensorizer(self.scenario, self.base, include_ros=True)
        _, cells, _ = tz.extract(obs)
        self.assertEqual(cells.shape[0], GeographicInputTensorizer.BASE_CELL_CHANNELS + 1)
        ros = cells[-1]
        nonburnable = cells[3] == 1.0
        self.assertTrue((ros[nonburnable] == 0).all())
        self.assertTrue((ros[~nonburnable] > 0).any())
        self.assertTrue((ros <= 1.0).all())

    def test_explicit_fire_ablation_has_no_image_and_fire_channels(self):
        obs = self.replay["decisions"][-1]["observation"]
        tz = GeographicInputTensorizer(self.scenario, None, explicit_fire=True)
        out = tz.extract(obs)
        self.assertEqual(len(out), 2)
        cells, vector = out
        self.assertEqual(cells.shape[0], GeographicInputTensorizer.BASE_CELL_CHANNELS + 3)
        burning, burned, age = cells[-3], cells[-2], cells[-1]
        for r, row in enumerate(obs["fire"]["cells"]):
            for c, cell in enumerate(row):
                self.assertEqual(burning[r, c], float(cell["fire_state"] == "burning"))
                self.assertEqual(burned[r, c], float(cell["fire_state"] == "burned"))
                if cell["fire_state"] == "unburned":
                    self.assertEqual(age[r, c], 0.0)
        self.assertTrue((age >= 0).all() and (age <= 1).all())
        image_tz = GeographicInputTensorizer(self.scenario, self.base)
        np.testing.assert_array_equal(cells[:-3], image_tz.extract(obs)[1])
        np.testing.assert_array_equal(vector, image_tz.extract(obs)[2])
        with self.assertRaises(ValueError):
            GeographicInputTensorizer(self.scenario, None)

    def test_thermal_band_is_a_fourth_image_channel(self):
        obs = self.replay["decisions"][-1]["observation"]
        tz = GeographicInputTensorizer(self.scenario, self.base, include_thermal=True)
        rgb, cells, vector = tz.extract(obs)
        self.assertEqual(rgb.shape[0], 4)
        plain = GeographicInputTensorizer(self.scenario, self.base).extract(obs)
        np.testing.assert_array_equal(rgb[:3], plain[0])
        np.testing.assert_array_equal(cells, plain[1])
        burning = np.repeat(np.repeat(tz.fire_channels(obs)[0] if tz.explicit_fire else
                                      np.array([[c["fire_state"] == "burning" for c in row]
                                                for row in obs["fire"]["cells"]]), PX, 0), PX, 1)
        self.assertTrue((rgb[3][burning.astype(bool)] == 1.0).all())
        with self.assertRaises(ValueError):
            GeographicInputTensorizer(self.scenario, None, explicit_fire=True, include_thermal=True)

    def test_rejects_delayed_observations_and_bad_base(self):
        delayed = run_geographic_integrated(self.scenario, policy="no_intervention",
                                            observation_mode="scheduled", _stop_after_steps=1)
        tz = GeographicInputTensorizer(self.scenario, self.base)
        with self.assertRaises(ValueError):
            tz.extract(delayed["pending_observation"])
        with self.assertRaises(ValueError):
            GeographicInputTensorizer(self.scenario, self.base[:-1])


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class ImageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.scenario = make_scenario()
        cls.rows = int(cls.scenario["grid"]["rows"])
        replay = run_geographic_integrated(cls.scenario, policy="heuristic")
        cls.obs = replay["decisions"][-1]["observation"]
        cls.tz = GeographicInputTensorizer(cls.scenario, np.full((cls.rows * PX, cls.rows * PX, 3), 80, np.uint8))

    def model(self, action_mode, recurrent):
        from nightjar.image_model import ImageActorCritic
        self.torch.manual_seed(0)
        return ImageActorCritic(cell_channels=self.tz.num_cell_channels, vector_dim=self.tz.NUM_VECTOR_FEATURES,
                                px_per_cell=PX, grid_rows=self.rows, grid_cols=self.rows,
                                action_mode=action_mode, recurrent=recurrent, width=8, hidden_dim=16,
                                lstm_hidden_dim=16)

    def test_actions_respect_mask_in_both_action_modes(self):
        inputs = self.tz.to_torch(self.obs)
        for mode in ("tactical", "spatial"):
            for recurrent in (False, True):
                model = self.model(mode, recurrent)
                mask = ActionMapper(self.rows, self.rows, self.scenario["grid"]["cell_size_m"],
                                    action_mode=mode).to_mask_torch(self.obs)
                self.assertEqual(mask.shape[-1], model.action_dim)
                for _ in range(5):
                    out = model.get_action(*inputs, mask)
                    self.assertTrue(bool(mask[0, out[0].item()]))
                self.assertEqual(len(out), 5 if recurrent else 4)

    def test_spatial_cell_logits_follow_row_major_grid(self):
        model = self.model("spatial", False)
        rgb, cells, vector = self.tz.to_torch(self.obs)
        with self.torch.no_grad():
            model.cell_actor_head.weight.zero_()
            model.cell_actor_head.bias.zero_()
            model.cell_condition.weight.zero_()
            model.cell_condition.bias.zero_()
            # Make the target cell's feature dominate through the stem output.
            features, latent = model.encode(rgb, cells, vector)
            features = self.torch.zeros_like(features)
            features[0, 0, 2, 5] = 10.0
            model.cell_actor_head.weight[0, 0] = 1.0
            logits, _ = model._heads(features, latent)
        self.assertEqual(int(logits[0, 2:].argmax()), 2 * self.rows + 5)

    def test_recurrent_evaluation_matches_stepwise_rollout(self):
        torch = self.torch
        model = self.model("tactical", True)
        inputs = self.tz.to_torch(self.obs)
        mask = ActionMapper(self.rows, self.rows, 100.0, action_mode="tactical").to_mask_torch(self.obs)
        state, log_probs, actions = None, [], []
        with torch.no_grad():
            for _ in range(3):
                action, lp, _, _, state = model.get_action(*inputs, mask, deterministic=True, state=state)
                actions.append(action)
                log_probs.append(lp)
            seq = tuple(t.repeat(3, *([1] * (t.dim() - 1))) for t in inputs)
            lp_seq, _, _ = model.evaluate_actions(*seq, mask.repeat(3, 1), torch.cat(actions))
        torch.testing.assert_close(lp_seq, torch.cat(log_probs))

    def test_image_ppo_rollout_and_update(self):
        torch = self.torch
        from nightjar.geographic_ppo import build_policy_model, ppo_update_geographic, rollout_geographic
        with tempfile.TemporaryDirectory() as tmp:
            write_imagery(Path(tmp), self.scenario, np.full((self.rows * PX, self.rows * PX, 3), 80, np.uint8))
            for input_mode, mode, thermal in (("image", "tactical", False), ("image", "spatial", False),
                                              ("cells", "tactical", False), ("image", "tactical", True),
                                              ("image", "parametric", True)):
                options = {"input_mode": input_mode, "imagery_dir": tmp, "include_ros": False,
                           "include_thermal": thermal}
                model = build_policy_model(self.scenario, action_mode=mode, recurrent=True, **options)
                records, metrics = rollout_geographic(model, self.scenario, **options)
                self.assertGreater(len(records), 0)
                self.assertEqual(len(records[0]["inputs"]), 3 if input_mode == "image" else 2)
                self.assertEqual(hasattr(model, "rgb_encoder"), input_mode == "image")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss = ppo_update_geographic(model, optimizer, [records])
                self.assertTrue(np.isfinite(loss))
                self.assertEqual(metrics["rejected_orders"], 0)


if __name__ == "__main__":
    unittest.main()
