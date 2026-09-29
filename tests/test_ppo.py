import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from nightjar.ppo import _baseline_metrics, _selection_key, returns_and_advantages, train_select, train_smoke


@unittest.skipUnless(HAS_TORCH, "PyTorch is optional")
class PPOTests(unittest.TestCase):
    def test_returns_and_advantages(self):
        returns, advantages = returns_and_advantages([1.0, 2.0], [0.0, 0.0], gamma=1.0)
        self.assertEqual(returns.tolist(), [3.0, 2.0])
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)

    def test_smoke_uses_valid_actions_on_development(self):
        result = train_smoke(updates=1, episodes_per_update=1, seed=3)
        self.assertEqual(result["split"], "development")
        self.assertTrue(result["synthetic_no_physical_claim"])
        self.assertEqual(result["metrics"][0]["rejected_orders"], 0)
        self.assertGreater(result["metrics"][0]["action_count"], 0)

    def test_selection_rejects_empty_training_config(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            train_select(updates=0)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            train_select(episodes_per_update=0)

    def test_selection_tie_break_is_seed_then_index(self):
        self.assertLess(_selection_key([{"affected_area_ha": 1.0}], 2, 1),
                        _selection_key([{"affected_area_ha": 1.0}], 3, 0))
        self.assertLess(_selection_key([{"affected_area_ha": 1.0}], 2, 0),
                        _selection_key([{"affected_area_ha": 1.0}], 2, 1))

    def test_selection_records_validation_metrics_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.pt"
            result = train_select(candidates=2, updates=1, episodes_per_update=1,
                                  seed=3, checkpoint=checkpoint)
            self.assertTrue(checkpoint.exists())
        self.assertEqual(result["trained_split"], "development")
        self.assertEqual(result["selection_split"], "validation")
        self.assertEqual(result["reserved_split"], "not_evaluated")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertTrue(all(len(c["validation_metrics"]) == 32 for c in result["candidates"]))
        self.assertIn("selection_rule", result)
        self.assertEqual(len(result["scenario_provenance"]["development"]), 8)
        self.assertEqual(len(result["scenario_provenance"]["validation"]), 32)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in result["scenario_provenance"]["validation"]))

    def test_baselines_pair_each_validation_scenario_and_not_reserved(self):
        scenarios = [{"scenario_id": "validation-00"}, {"scenario_id": "validation-01"}]
        calls = []

        def replay(scenario, policy):
            calls.append((scenario["scenario_id"], policy))
            return {"scenario_id": scenario["scenario_id"], "affected_area_ha": 1.0,
                    "aircraft_metrics": {"rejected_orders": 0}}

        with patch("nightjar.ppo.run_integrated", side_effect=replay):
            result = _baseline_metrics(scenarios)
        self.assertEqual(calls, [("validation-00", "heuristic"),
                                 ("validation-00", "containment_heuristic"),
                                 ("validation-01", "heuristic"),
                                 ("validation-01", "containment_heuristic")])
        self.assertEqual([item["scenario_id"] for item in result],
                         ["validation-00", "validation-01"])
        self.assertNotIn("reserved", " ".join(item["scenario_id"] for item in result))

    def test_selection_never_evaluates_reserved(self):
        evaluated = []
        def record_evaluate(model, scenarios, seed=0, device=None):
            evaluated.extend(s["scenario_id"] for s in scenarios)
            return [{"scenario_id": s["scenario_id"], "affected_area_ha": 1.0,
                     "rejected_orders": 0} for s in scenarios]

        with patch("nightjar.ppo._train_candidate", return_value=(None, [])), \
                patch("nightjar.ppo.evaluate", side_effect=record_evaluate):
            result = train_select(candidates=1)
        self.assertTrue(all("reserved" not in scenario_id for scenario_id in evaluated))
        self.assertEqual(result["reserved_split"], "not_evaluated")

    def test_smoke_with_recurrent_and_domain_randomization(self):
        result = train_smoke(
            updates=1,
            episodes_per_update=1,
            seed=5,
            recurrent=True,
            domain_randomization=True,
        )
        self.assertEqual(result["split"], "domain_randomized")
        self.assertTrue(result["config"]["recurrent"])
        self.assertTrue(result["config"]["domain_randomization"])
        self.assertEqual(len(result["losses"]), 1)
        self.assertEqual(result["metrics"][0]["rejected_orders"], 0)
        self.assertGreater(result["metrics"][0]["action_count"], 0)

    def test_recurrent_selection_and_checkpoint_format(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "recurrent.pt"
            result = train_select(
                candidates=1,
                updates=1,
                episodes_per_update=1,
                seed=7,
                recurrent=True,
                domain_randomization=True,
                checkpoint=checkpoint,
            )
            self.assertTrue(checkpoint.exists())
            saved = torch.load(checkpoint, weights_only=False)
            self.assertTrue(saved["model_config"]["recurrent"])
            self.assertEqual(result["trained_split"], "domain_randomized")
            self.assertTrue(result["config"]["recurrent"])
            self.assertTrue(result["config"]["domain_randomization"])


if __name__ == "__main__":
    unittest.main()
