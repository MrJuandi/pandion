"""Thermal-brightness inputs (D-94) and the learned speed channel (D-92)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pandion.action_features import ObservedFireBehavior
from pandion.geographic_integrated import NEIGHBORS, run_geographic_integrated, thermal_byte
from pandion.imagery_composite import thermal_view
from pandion.learned_spread import cached_head_speed, learned_head_speed
from tests.test_geographic_integrated import growing_fire_scenario, with_weather


def _observation(intensities):
    """3x3 grid: centre burning with the given intensity, the rest unburned."""
    cells = [[{"fire_detected": False, "fire_state": "unburned", "visible_state": "unburned"} for _ in range(3)] for _ in range(3)]
    for (r, c), value in intensities.items():
        cells[r][c] = {"fire_detected": True, "fire_state": "burning", "visible_state": "burning", "thermal_byte": thermal_byte(value)}
    return {"time_s": 0.0, "fire": {"cells": cells}, "cell_size_m": 100.0}


class SensorTests(unittest.TestCase):
    def test_the_observation_carries_brightness_and_no_intensity(self):
        # D-94: one byte per burning cell, equal to the cell's largest directional intensity on the thermal scale.
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        replay = run_geographic_integrated(scenario, policy="heuristic")
        formula, checked = ObservedFireBehavior(), 0
        for decision in replay["decisions"]:
            obs = decision["observation"]
            for r, line in enumerate(obs["fire"]["cells"]):
                for c, cell in enumerate(line):
                    self.assertNotIn("intensity_kw_m", cell)
                    if cell["visible_state"] != "burning":
                        self.assertIsNone(cell["thermal_byte"])
                        continue
                    head = max(formula.intensity(obs, r, c, dr, dc) for dr, dc in NEIGHBORS)
                    self.assertEqual(cell["thermal_byte"], thermal_byte(head))
                    checked += 1
        self.assertGreater(checked, 0)


class BlindRuleTests(unittest.TestCase):
    def test_the_visible_fire_does_not_depend_on_water(self):
        # D-95: a drop changes nothing the camera shows until the fire itself behaves differently.
        scenario = with_weather(growing_fire_scenario(), (0.08, 0.09, 0.10, 0.08, 0.6, 0.9), (0.7, 0.7))
        replay = run_geographic_integrated(scenario, policy="heuristic")
        differ = 0
        for decision in replay["decisions"]:
            obs = decision["observation"]
            for r, line in enumerate(obs["fire"]["cells"]):
                for c, cell in enumerate(line):
                    if cell["fire_state"] == "burning":
                        self.assertEqual(cell["visible_state"], "burning")
                    differ += cell["fire_state"] != cell["visible_state"]
        self.assertGreater(differ, 0)

    def test_the_rule_never_skips_a_treated_cell_and_prefers_untreated_ones(self):
        from pandion.policies import candidate_cells, choose_order
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        obs = run_geographic_integrated(scenario, policy="heuristic")["decisions"][0]["observation"]
        options = candidate_cells(obs)
        first = choose_order("blind_hold_aware_heuristic", obs, candidates=options)["target_m"]
        cell = [round(first[1] / 100), round(first[0] / 100)]
        self.assertIn(cell, options)
        second = choose_order("blind_hold_aware_heuristic", {**obs, "known_treated_cells": [cell]}, candidates=options)
        self.assertNotEqual(second["target_m"], first)
        everywhere = {**obs, "known_treated_cells": [list(o) for o in options]}
        self.assertEqual(choose_order("blind_hold_aware_heuristic", everywhere, candidates=options)["type"], "DROP")


class BlindOrderTests(unittest.TestCase):
    def test_untreated_cells_come_first_and_then_the_dimmer_fire(self):
        from pandion.policies import choose_order
        obs = _observation({(0, 0): 8000.0, (2, 2): 500.0})
        obs.update(fuel_model=[["102"] * 3 for _ in range(3)], midflame_wind_to_m_s=[0.0, 0.0], known_treated_cells=[],
                   aircraft={"phase": "idle", "water_l": 100.0, "position_m": [0.0, 0.0]})
        # cells (0, 1) and (1, 0) border the strong fire; (2, 1) and (1, 2) the weak one.
        first = choose_order("blind_hold_aware_heuristic", obs)["target_m"]
        self.assertIn(first, ([100.0, 200.0], [200.0, 100.0]))
        treated = {**obs, "known_treated_cells": [[2, 1], [1, 2]], "max_stacked_drops": 3}
        second = choose_order("blind_hold_aware_heuristic", treated)["target_m"]
        self.assertIn(second, ([100.0, 0.0], [0.0, 100.0]))


class ThermalImageTests(unittest.TestCase):
    def test_brightness_separates_weak_from_strong_and_both_from_ground_that_is_not_burning(self):
        values = [thermal_byte(v) for v in (0.0, 100.0, 1000.0, 3200.0, 10000.0, 20000.0, 90000.0)]
        self.assertEqual(values, sorted(values))
        self.assertGreaterEqual(values[0], 76)
        self.assertEqual(values[-1], 255)
        self.assertGreaterEqual(thermal_byte(3200.0) - thermal_byte(1000.0), 30)
        weak = thermal_view(_observation({(1, 1): 100.0}), 4, 100.0, measured=True)
        strong = thermal_view(_observation({(1, 1): 10000.0}), 4, 100.0, measured=True)
        self.assertLess(weak[4:8, 4:8].max(), strong[4:8, 4:8].max())
        self.assertGreaterEqual(weak[4:8, 4:8].min(), 0.295)
        outside = np.ones(weak.shape, bool)
        outside[4:8, 4:8] = False
        self.assertLessEqual(float(strong[outside].max()), 0.2 + 1e-6)
        plain = thermal_view(_observation({(1, 1): 100.0}), 4, 100.0)
        self.assertAlmostEqual(float(plain[4:8, 4:8].max()), 1.0)

    def test_the_targets_carry_heat_terrain_and_fuel_and_no_limit(self):
        from pandion.action_features import TARGET_FEATURE_NAMES, action_feature_array, target_feature_array
        from pandion.parametric_actions import ParametricActionMapper
        from pandion.policy_inputs import fuel_family
        import pandion.geographic_integrated as g
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        obs = run_geographic_integrated(scenario, policy="heuristic")["decisions"][6]["observation"]
        mapper = ParametricActionMapper(float(obs["cell_size_m"]), (0, 1, 2))
        table = target_feature_array(obs, mapper, fuel_family)
        col = {n: 1 + i for i, n in enumerate(TARGET_FEATURE_NAMES)}
        valid = table[:, col["valid"]] > 0
        self.assertGreater(valid.sum(), 0)
        self.assertFalse(table[~valid].any())
        self.assertGreater(table[valid, col["heat_max"]].max(), 0.3)
        self.assertTrue((table[valid][:, [col[f"fuel_{k}"] for k in range(7)]].sum(1) == 1).all())
        keep = g.DROP_HOLD_LIMITS_KW_M
        try:
            g.DROP_HOLD_LIMITS_KW_M = ((1, 1.0), (3, 2.0))
            np.testing.assert_array_equal(table, target_feature_array(obs, mapper, fuel_family))
        finally:
            g.DROP_HOLD_LIMITS_KW_M = keep
        names = set(TARGET_FEATURE_NAMES)
        self.assertFalse(names & {"need_one", "need_three", "breach", "remaining", "sufficient"})
        self.assertGreater(float(np.abs(action_feature_array(obs, mapper, ObservedFireBehavior())[:, 4:9]).sum()), 0.0)


class LearnedHoldTests(unittest.TestCase):
    def _weights(self, folder):
        from pandion.learned_hold import FEATURES
        rng, n = np.random.default_rng(0), len(FEATURES)
        path = Path(folder) / "hold.npz"
        np.savez(path, mu=np.zeros(n), sd=np.ones(n), **{"0/w": rng.normal(size=(4, n)), "0/b": np.zeros(4),
                 "1/w": rng.normal(size=(4, 4)), "1/b": np.zeros(4), "2/w": rng.normal(size=(1, 4)), "2/b": np.zeros(1)})
        return path

    def test_risk_columns_are_probabilities_and_follow_the_drop_count(self):
        from pandion.action_features import NUM_TARGET_FEATURES, TARGET_FEATURE_NAMES, target_feature_array
        from pandion.learned_hold import FEATURES, risk_before_and_after
        from pandion.parametric_actions import ParametricActionMapper
        from pandion.policy_inputs import fuel_family
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        obs = run_geographic_integrated(scenario, policy="heuristic")["decisions"][6]["observation"]
        mapper = ParametricActionMapper(float(obs["cell_size_m"]), (0, 1, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = self._weights(folder)
            table = target_feature_array(obs, mapper, fuel_family, path)
            self.assertEqual(table.shape[1], 1 + NUM_TARGET_FEATURES + 4)
            risk = table[:, 1 + NUM_TARGET_FEATURES:3 + NUM_TARGET_FEATURES]
            self.assertTrue(((risk >= 0) & (risk <= 1)).all())
            self.assertGreater(float(risk.max()), 0.0)
            np.testing.assert_array_equal(table[:, :1 + NUM_TARGET_FEATURES], target_feature_array(obs, mapper, fuel_family))
            col = {n: 1 + i for i, n in enumerate(TARGET_FEATURE_NAMES)}
            ahead, front = table[table[:, col["offset"]] > 0], table[(table[:, col["offset"]] == 0) & (table[:, 1] > 0)]
            self.assertFalse(ahead[:, 1 + NUM_TARGET_FEATURES:].any())  # no prediction where it was not fitted
            self.assertTrue((front[:, -1] == 1).any())
            np.testing.assert_allclose(table[:, -2], table[:, -4] - table[:, -3], atol=1e-5)
            rows = np.zeros((1, len(FEATURES)), dtype=np.float32)
            rows[0, FEATURES.index("drops")] = 1.0
            before, after = risk_before_and_after(rows, path)  # three drops already: one more changes nothing
            self.assertAlmostEqual(float(before[0]), float(after[0]))


class LearnedBurnTests(unittest.TestCase):
    def test_the_map_is_a_probability_with_reached_and_nonburnable_cells_fixed(self):
        from pandion.learned_burn import NUM_CHANNELS, burn_channels, burn_map, load_net
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        scenario["fire"]["fuel_grid"][0][0] = 0
        obs = run_geographic_integrated(scenario, policy="heuristic")["decisions"][4]["observation"]
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "burn.npz"
            # the module builds its own network, so make weights of the right shapes by loading a random one
            names = {"inp.weight": (32, NUM_CHANNELS, 3, 3), "inp.bias": (32,), "out.weight": (1, 32, 1, 1), "out.bias": (1,)}
            for i in range(7):
                for j in (0, 2):
                    names[f"blocks.{i}.{j}.weight"], names[f"blocks.{i}.{j}.bias"] = (32, 32, 3, 3), (32,)
            np.savez(path, mu=np.zeros(NUM_CHANNELS, np.float32), sd=np.ones(NUM_CHANNELS, np.float32),
                     **{"net/" + n: (rng.normal(size=sh) * 0.1).astype(np.float32) for n, sh in names.items()})
            load_net.cache_clear()
            prob = burn_map(obs, 3600.0, path)
            x = burn_channels(obs, 3600.0)
        self.assertEqual(prob.shape, x.shape[1:])
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())
        self.assertTrue((prob[x[0] > 0] == 1).all())
        self.assertEqual(float(prob[(x[0] == 0) & (x[24] == 0)].max(initial=0.0)), 0.0)


class ArrivalTests(unittest.TestCase):
    def test_arrival_outputs_are_ordered_and_feed_two_target_columns(self):
        from pandion.action_features import NUM_TARGET_FEATURES, target_feature_array
        from pandion.learned_burn import NUM_CHANNELS, burn_outputs, load_net
        from pandion.parametric_actions import ParametricActionMapper
        from pandion.policy_inputs import fuel_family
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        obs = run_geographic_integrated(scenario, policy="heuristic")["decisions"][4]["observation"]
        rng = np.random.default_rng(1)
        names = {"inp.weight": (32, NUM_CHANNELS, 3, 3), "inp.bias": (32,), "out.weight": (5, 32, 1, 1), "out.bias": (5,)}
        for i in range(7):
            for j in (0, 2):
                names[f"blocks.{i}.{j}.weight"], names[f"blocks.{i}.{j}.bias"] = (32, 32, 3, 3), (32,)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arrival.npz"
            np.savez(path, mu=np.zeros(NUM_CHANNELS, np.float32), sd=np.ones(NUM_CHANNELS, np.float32),
                     minutes=np.array([1, 3, 10, 30, 1e9], np.float32),
                     **{"net/" + n: (rng.normal(size=sh) * 0.1).astype(np.float32) for n, sh in names.items()})
            load_net.cache_clear()
            probs, minutes = burn_outputs(obs, 3600.0, path)
            mapper = ParametricActionMapper(float(obs["cell_size_m"]), (0, 1, 2))
            table = target_feature_array(obs, mapper, fuel_family, None, 3600.0, (probs, minutes, 80.0))
        self.assertEqual(probs.shape[0], 5)
        self.assertTrue((np.diff(probs, axis=0) >= 0).all())  # reached within more minutes is never less likely
        self.assertEqual(table.shape[1], 1 + NUM_TARGET_FEATURES + 2)
        valid = table[:, 1] > 0
        self.assertTrue(((table[valid, -2:] >= 0) & (table[valid, -2:] <= 1)).all())
        self.assertTrue((table[valid, -2] <= table[valid, -1] + 1e-6).all())  # flights here last under ten minutes


class ActionSkipTests(unittest.TestCase):
    def test_the_linear_path_starts_at_zero_and_loads_onto_a_model_without_it(self):
        import torch
        from pandion.image_model import ImageActorCritic
        kw = dict(cell_channels=12 + 5, vector_dim=15, px_per_cell=1, grid_rows=6, grid_cols=6, action_mode="parametric",
                  use_image=False, drop_distances=(0, 1, 2), action_feature_dim=22)
        torch.manual_seed(0)
        plain = ImageActorCritic(**kw).eval()
        skip = ImageActorCritic(**kw, action_skip=True).eval()
        missing, unexpected = skip.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertEqual(sorted(missing), ["target_skip.bias", "target_skip.weight"])
        cells, vector = torch.rand(1, 17, 6, 6), torch.rand(1, 15)
        actions = torch.zeros(1, plain.action_dim - 2, 23)
        actions[0, :5, 0] = torch.arange(5).float()
        actions[0, :5, 1] = 1.0
        actions[0, :5, 2:] = torch.rand(5, 21)
        with torch.no_grad():
            a, _, _ = plain(cells, vector, actions, state=None)
            b, _, _ = skip(cells, vector, actions, state=None)
            torch.testing.assert_close(a, b)
            skip.target_skip.weight.fill_(1.0)
            c, _, _ = skip(cells, vector, actions, state=None)
        self.assertGreater(float((c - a).abs().max()), 0.0)


class LearnedSpeedTests(unittest.TestCase):
    def _weights(self, folder):
        rng = np.random.default_rng(0)
        n = 8 + 8 + 11 + 6
        arrays = {"0/idx": np.zeros(300, int), "0/emb": rng.normal(size=(2, 8)),
                  "0/w0": rng.normal(size=(4, n)), "0/b0": np.zeros(4), "0/w1": rng.normal(size=(4, 4)), "0/b1": np.zeros(4),
                  "0/w2": rng.normal(size=(1, 4)), "0/b2": np.array([-3.0])}
        path = Path(folder) / "w.npz"
        np.savez(path, **arrays)
        return path

    def test_speed_is_nonnegative_zero_on_bare_cells_and_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._weights(folder)
            fuel = [["101"] * 5 for _ in range(5)]
            fuel[2][2] = "0"
            obs = {"fuel_model": fuel, "elevation_m": np.arange(25.0).reshape(5, 5).tolist(), "cell_size_m": 100.0,
                   "midflame_wind_to_m_s": [2.0, 1.0],
                   "fuel_moisture": {k: 0.1 for k in ("dead_1h", "dead_10h", "dead_100h", "dead_herbaceous",
                                                      "live_herbaceous", "live_woody")}}
            speed = learned_head_speed(obs, path)
            self.assertEqual(speed.shape, (5, 5))
            self.assertEqual(speed[2, 2], 0.0)
            self.assertTrue((speed >= 0).all())
            np.testing.assert_allclose(cached_head_speed(obs, path), speed.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
