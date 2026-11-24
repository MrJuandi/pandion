"""Unit tests for branched ("vine") advantages and their PPO integration (D-85, D-88 to D-90)."""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pandion.geographic_ppo import HAS_TORCH

if not HAS_TORCH:
    raise unittest.SkipTest("PyTorch not installed")

import torch

from pandion.geographic_family import (
    build_geographic_family,
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from pandion.geographic_ppo import (
    attach_vine_samples,
    build_policy_model,
    main,
    ppo_update_geographic,
    rollout_geographic,
    train_geographic_ppo,
)
from pandion.vine import (
    allocate_vine_states,
    choose_vine_decisions,
    leave_one_out_advantages,
    remaining_time_weights,
    vine_episode,
)


def _scenario():
    fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=25, cols=25)
    return generate_geographic_scenario(fixture, index=0, split="development",
                                        window_rows=15, window_cols=15, horizon_s=900.0)


def _stack(records):
    inputs = tuple(torch.stack(parts) for parts in zip(*(r["inputs"] for r in records)))
    masks = torch.stack([r["mask"] for r in records])
    return inputs, masks


class TestLeaveOneOutAndChoice(unittest.TestCase):
    def test_leave_one_out_advantages_values(self):
        self.assertEqual(leave_one_out_advantages([1.0, 2.0, 3.0]), [-1.5, 0.0, 1.5])

    def test_leave_one_out_single_and_empty(self):
        self.assertEqual(leave_one_out_advantages([4.2]), [0.0])
        self.assertEqual(leave_one_out_advantages([]), [])

    def test_leave_one_out_sums_to_zero(self):
        rng = random.Random(3)
        for count in (2, 3, 4, 7):
            with self.subTest(count=count):
                returns = [rng.uniform(-50.0, 50.0) for _ in range(count)]
                self.assertAlmostEqual(sum(leave_one_out_advantages(returns)), 0.0, places=9)

    def test_choose_vine_decisions_is_sorted_unique_in_range(self):
        chosen = choose_vine_decisions(20, 5, random.Random(0))
        self.assertEqual(chosen, sorted(chosen))
        self.assertEqual(len(set(chosen)), len(chosen))
        self.assertTrue(all(0 <= i < 20 for i in chosen))
        self.assertEqual(len(chosen), 5)

    def test_choose_vine_decisions_caps_at_episode_length(self):
        self.assertEqual(choose_vine_decisions(3, 10, random.Random(0)), [0, 1, 2])

    def test_choose_vine_decisions_empty_cases(self):
        self.assertEqual(choose_vine_decisions(10, 0, random.Random(0)), [])
        self.assertEqual(choose_vine_decisions(0, 3, random.Random(0)), [])

    def test_choose_vine_decisions_same_seed_same_result(self):
        first = choose_vine_decisions(50, 6, random.Random(0))
        second = choose_vine_decisions(50, 6, random.Random(0))
        self.assertEqual(first, second)


class TestWeightedChoice(unittest.TestCase):
    def test_weighted_choice_is_sorted_unique_and_capped(self):
        chosen = choose_vine_decisions(20, 5, random.Random(0), weights=[float(20 - i) for i in range(20)])
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertEqual(len(chosen), 5)
        self.assertEqual(choose_vine_decisions(3, 10, random.Random(0), weights=[1.0, 2.0, 3.0]), [0, 1, 2])

    def test_weighted_choice_follows_weights(self):
        rng = random.Random(1)
        weights = [9.0, 1.0, 1.0, 1.0]
        hits = sum(0 in choose_vine_decisions(4, 1, rng, weights=weights) for _ in range(4000))
        self.assertAlmostEqual(hits / 4000, 0.75, delta=0.03)

    def test_zero_weights_only_when_positive_run_out(self):
        weights = [0.0, 0.0, 5.0, 1.0, 0.0]
        for seed in range(50):
            self.assertEqual(choose_vine_decisions(5, 2, random.Random(seed), weights=weights), [2, 3])
            self.assertEqual(len(set(choose_vine_decisions(5, 3, random.Random(seed), weights=weights)) & {2, 3}), 2)

    def test_weighted_choice_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            choose_vine_decisions(3, 1, random.Random(0), weights=[1.0, 2.0])
        with self.assertRaises(ValueError):
            choose_vine_decisions(2, 1, random.Random(0), weights=[1.0, -1.0])

    def test_uniform_choice_unchanged_by_weighting_option(self):
        self.assertEqual(choose_vine_decisions(50, 6, random.Random(0)),
                         sorted(random.Random(0).sample(range(50), 6)))

    def test_remaining_time_weights(self):
        self.assertEqual(remaining_time_weights([0.0, 600.0, 3600.0, 4000.0], 3600.0), [3600.0, 3000.0, 0.0, 0.0])


class TestVineEpisode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        torch.manual_seed(42)
        cls.scenario = _scenario()
        cls.model = build_policy_model(cls.scenario, input_mode="cells", action_mode="parametric",
                                       recurrent=True, slot_features=True, drop_distances=(0, 1, 2),
                                       action_features=True)
        cls.model.eval()
        torch.manual_seed(0)
        cls.records, _ = rollout_geographic(cls.model, cls.scenario, deterministic=False, skip_forced=True,
                                            input_mode="cells", reward_reference=None,
                                            drop_bonus_per_cell=0.0)
        cls.actions = [int(r["action"]) for r in cls.records]
        assert len(cls.records) >= 3, "fixture episode too short for prefix checks"

    def test_records_carry_increasing_decision_times(self):
        times = [r["time_s"] for r in self.records]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[0], 0.0)
        self.assertLess(times[-1], self.scenario["fire"]["horizon_s"])

    def _branch(self, chosen, seed=7, branch_actions=3):
        return vine_episode(self.model, self.scenario, self.actions, chosen, branch_actions=branch_actions,
                            seed=seed, skip_forced=True, drop_bonus_per_cell=0.0, input_mode="cells")

    def test_vine_episode_is_deterministic(self):
        chosen = [0, len(self.actions) // 2]
        first = self._branch(chosen)
        second = self._branch(chosen)
        self.assertEqual(first, second)
        self.assertEqual([b["decision"] for b in first], chosen)
        for branch in first:
            self.assertEqual(len(branch["actions"]), 3)
            self.assertEqual(len(branch["log_probs"]), 3)
            self.assertEqual(len(branch["returns"]), 3)

    @unittest.skipUnless(hasattr(__import__("os"), "fork"), "needs os.fork")
    def test_snapshot_branching_matches_replay(self):
        n = len(self.actions)
        for chosen in ([0], [0, n // 2, n - 1], [n - 1]):
            with self.subTest(chosen=chosen):
                replayed = self._branch(chosen, seed=5, branch_actions=4)
                forked = vine_episode(self.model, self.scenario, self.actions, chosen, branch_actions=4, seed=5,
                                      skip_forced=True, drop_bonus_per_cell=0.0, input_mode="cells",
                                      snapshot=True)
                self.assertEqual(forked, replayed)

    @unittest.skipUnless(hasattr(__import__("os"), "fork"), "needs os.fork")
    def test_failed_forked_branch_raises(self):
        from pandion.vine import _forked_return
        self.assertEqual(_forked_return(lambda: 2.5), 2.5)
        with self.assertRaisesRegex(RuntimeError, "forked vine branch failed"):
            _forked_return(lambda: 1 / 0)

    def test_equal_actions_share_exactly_equal_returns(self):
        chosen = [0, len(self.actions) // 2, len(self.actions) - 1]
        branches = self._branch(chosen, seed=11, branch_actions=6)
        for branch in branches:
            pairs = list(zip(branch["actions"], branch["returns"]))
            for i, (action_i, return_i) in enumerate(pairs):
                for action_j, return_j in pairs[i + 1:]:
                    if action_i == action_j:
                        self.assertEqual(return_i, return_j)

    def test_prefix_replay_matches_collected_log_probs(self):
        n = len(self.actions)
        chosen = [0, n // 2, n - 1]
        branches = self._branch(chosen)
        inputs, masks = _stack(self.records)
        with torch.no_grad():
            table, _, _ = self.model.evaluate_log_probs(*inputs, masks)
        for t, record in enumerate(self.records):
            self.assertAlmostEqual(float(record["log_prob"]), float(table[t, self.actions[t]]), delta=1e-4)
        for branch in branches:
            d = branch["decision"]
            for action, log_prob in zip(branch["actions"], branch["log_probs"]):
                with self.subTest(decision=d, action=action):
                    self.assertAlmostEqual(log_prob, float(table[d, action]), delta=1e-4)

    def test_evaluate_log_probs_agrees_with_evaluate_actions(self):
        inputs, masks = _stack(self.records)
        actions = torch.tensor(self.actions, dtype=torch.long)
        with torch.no_grad():
            all_log_probs, all_entropy, all_values = self.model.evaluate_log_probs(*inputs, masks)
            log_prob, entropy, values = self.model.evaluate_actions(*inputs, masks, actions)
        self.assertEqual(all_log_probs.shape[0], len(self.actions))
        gathered = all_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(gathered, log_prob, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(all_entropy, entropy, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(all_values, values, rtol=1e-5, atol=1e-6)


class TestVinePPOUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        torch.manual_seed(42)
        cls.scenario = _scenario()

    def _fresh_model_and_records(self):
        torch.manual_seed(5)
        model = build_policy_model(self.scenario, input_mode="cells", action_mode="parametric",
                                   recurrent=True, slot_features=True, drop_distances=(0, 1, 2),
                                   action_features=True)
        model.eval()
        torch.manual_seed(0)
        records, _ = rollout_geographic(model, self.scenario, deterministic=False, skip_forced=True,
                                        input_mode="cells", reward_reference=None, drop_bonus_per_cell=0.0)
        actions = [int(r["action"]) for r in records]
        n = len(records)
        chosen = sorted({0, n // 2, n - 1})
        branches = vine_episode(model, self.scenario, actions, chosen, branch_actions=3, seed=13,
                                skip_forced=True, drop_bonus_per_cell=0.0, input_mode="cells")
        attach_vine_samples(records, branches)
        return model, records

    def test_branched_update_is_finite_and_moves_parameters(self):
        model, records = self._fresh_model_and_records()
        self.assertTrue(any("vine" in r for r in records))
        model.train()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = ppo_update_geographic(model, optimizer, records, epochs=2, minibatches=1,
                                     normalize_advantages=True)
        self.assertTrue(math.isfinite(loss))
        changed = [not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())]
        self.assertTrue(any(changed))

    def test_zero_branched_advantages_leave_parameters_unchanged(self):
        model, records = self._fresh_model_and_records()
        for record in records:
            if "vine" in record:
                record["vine"]["advantage"] = torch.zeros_like(record["vine"]["advantage"])
        model.train()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        ppo_update_geographic(model, optimizer, records, epochs=2, minibatches=1,
                              normalize_advantages=False, value_coef=0.0, entropy_coef=0.0)
        for b, p in zip(before, model.parameters()):
            self.assertTrue(torch.equal(b, p.detach()))


class TestVineAllocation(unittest.TestCase):
    def test_budget_floor_and_caps(self):
        alloc = allocate_vine_states([10, 10, 10, 10], [0.0, 0.0, 30.0, 10.0], 4)
        self.assertEqual(sum(alloc), 16)
        self.assertEqual(alloc[:2], [1, 1])
        self.assertGreater(alloc[2], alloc[3])
        alloc = allocate_vine_states([2, 20], [100.0, 0.0], 4)
        self.assertEqual(alloc, [2, 6])  # capped at two decisions, the rest goes to the other episode

    def test_zero_weights_share_equally(self):
        self.assertEqual(allocate_vine_states([10, 10, 10], [0.0, 0.0, 0.0], 4), [4, 4, 4])

    def test_short_batches_and_empty_episodes(self):
        self.assertEqual(allocate_vine_states([1, 0, 2], [5.0, 5.0, 0.0], 4), [1, 0, 2])
        self.assertEqual(allocate_vine_states([], [], 4), [])
        self.assertEqual(allocate_vine_states([5, 5], [1.0, 1.0], 0, floor=0), [0, 0])

    def test_fractional_shares_use_the_whole_budget(self):
        alloc = allocate_vine_states([30] * 7, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 4)
        self.assertEqual(sum(alloc), 28)
        self.assertEqual(alloc, sorted(alloc))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            allocate_vine_states([1, 2], [1.0], 4)
        with self.assertRaises(ValueError):
            allocate_vine_states([1], [1.0], -1)


class TestVineTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        torch.manual_seed(42)
        cls.scenario = _scenario()

    def test_train_with_vine_returns_one_loss(self):
        model, losses = train_geographic_ppo([self.scenario], updates=1, episodes_per_update=1,
                                             input_mode="cells", action_mode="parametric",
                                             rollout_workers=1, vine_states=1, vine_actions=2)
        self.assertEqual(len(losses), 1)
        self.assertTrue(math.isfinite(losses[0]))

    def test_train_with_growth_allocation_returns_one_loss(self):
        model, losses = train_geographic_ppo([self.scenario], updates=1, episodes_per_update=2,
                                             input_mode="cells", action_mode="parametric",
                                             rollout_workers=1, vine_states=1, vine_actions=2,
                                             vine_allocation="growth")
        self.assertEqual(len(losses), 1)
        self.assertTrue(math.isfinite(losses[0]))

    def test_train_with_remaining_time_weighting(self):
        _, losses = train_geographic_ppo([self.scenario], updates=1, episodes_per_update=1,
                                         input_mode="cells", action_mode="parametric",
                                         vine_states=2, vine_actions=2, vine_weighting="remaining")
        self.assertTrue(math.isfinite(losses[0]))

    def test_vine_argument_validation(self):
        base = dict(updates=1, episodes_per_update=1, input_mode="cells", action_mode="parametric")
        cases = {
            "vine_actions": dict(vine_states=1, vine_actions=1),
            "rollout_envs": dict(vine_states=1, vine_actions=2, rollout_envs=2),
            "gamma": dict(vine_states=1, vine_actions=2, gamma=0.99),
            "legacy": dict(vine_states=1, vine_actions=2, input_mode="legacy"),
            "allocation": dict(vine_states=1, vine_actions=2, vine_allocation="random"),
            "weighting": dict(vine_states=1, vine_actions=2, vine_weighting="late"),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                options = {**base, **overrides}
                if name == "legacy":
                    options["action_mode"] = "tactical"
                with self.assertRaisesRegex(ValueError, "vine"):
                    train_geographic_ppo([self.scenario], **options)


class TestVineSmokeCLI(unittest.TestCase):
    def test_smoke_cli_records_vine_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = build_geographic_family(num_dev=1, num_val=1, num_res=2, synthetic=True)
            (root / "manifest.json").write_text(json.dumps(family["manifest"]))
            (root / "development").mkdir()
            for scenario in family["scenarios"]["development"]:
                (root / "development" / f"{scenario['scenario_id']}.json").write_text(json.dumps(scenario))
            output = root / "smoke.json"
            with patch("sys.argv", ["geographic_ppo", "--smoke", "--synthetic",
                                    "--scenarios-dir", str(root), "--output", str(output),
                                    "--vine-states", "1", "--vine-actions", "2", "--vine-allocation", "growth",
                                    "--vine-weighting", "remaining",
                                    "--input-mode", "cells", "--action-mode", "parametric"]):
                random.seed(0)
                torch.manual_seed(0)
                main()
            report = json.loads(output.read_text())
            self.assertEqual(report["ppo"]["vine_states"], 1)
            self.assertEqual(report["ppo"]["vine_actions"], 2)
            self.assertEqual(report["ppo"]["vine_allocation"], "growth")
            self.assertEqual(report["ppo"]["vine_weighting"], "remaining")
            self.assertTrue(math.isfinite(report["loss"]))


if __name__ == "__main__":
    unittest.main()
