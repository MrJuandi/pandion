import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from nightjar.evaluate_reserved import (
    _extract_metrics,
    _paired_comparison,
    run_reserved_evaluation,
)


class EvaluateReservedTests(unittest.TestCase):
    def test_extract_metrics(self):
        replay = {
            "affected_area_ha": 4.5,
            "drops": [{}, {}],
            "aircraft_metrics": {
                "water_dropped_l": 12274.0,
                "flight_s": 120.0,
                "rejected_orders": 0,
            },
        }
        metrics = _extract_metrics(replay)
        self.assertEqual(metrics["affected_area_ha"], 4.5)
        self.assertEqual(metrics["drops"], 2)
        self.assertEqual(metrics["water_dropped_l"], 12274.0)
        self.assertEqual(metrics["flight_s"], 120.0)
        self.assertEqual(metrics["rejected_orders"], 0)

    def test_paired_comparison_computations(self):
        ppo = [
            {"affected_area_ha": 3.0, "rejected_orders": 0},
            {"affected_area_ha": 5.0, "rejected_orders": 0},
            {"affected_area_ha": 4.0, "rejected_orders": 0},
            {"affected_area_ha": 6.0, "rejected_orders": 0},
        ]
        baseline = [
            {"affected_area_ha": 4.0, "rejected_orders": 0},  # win (+1.0 reduction)
            {"affected_area_ha": 4.5, "rejected_orders": 0},  # loss (-0.5 reduction)
            {"affected_area_ha": 4.0, "rejected_orders": 0},  # tie (0.0)
            {"affected_area_ha": 7.0, "rejected_orders": 0},  # win (+1.0 reduction)
        ]
        result = _paired_comparison(ppo, baseline)
        self.assertEqual(result["paired_count"], 4)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["loss_rate"], 0.25)
        self.assertEqual(result["tie_rate"], 0.25)
        self.assertAlmostEqual(result["mean_reduction_ha"], 0.375)
        self.assertAlmostEqual(result["mean_ppo_minus_baseline_ha"], -0.375)
        self.assertTrue(result["all_zero_rejected_orders"])
        self.assertIn("reduction_95_ci", result)
        self.assertIn("difference_95_ci", result)
        self.assertEqual(len(result["reduction_95_ci"]), 2)
        self.assertLessEqual(result["reduction_95_ci"][0], result["reduction_95_ci"][1])

    def test_bootstrap_ci(self):
        from nightjar.evaluate_reserved import _bootstrap_ci
        # Deterministic with seed
        ci1 = _bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=1000, seed=42)
        ci2 = _bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=1000, seed=42)
        self.assertEqual(ci1, ci2)
        self.assertLessEqual(ci1["low"], 3.0)
        self.assertGreaterEqual(ci1["high"], 3.0)
        # Empty list
        empty_ci = _bootstrap_ci([])
        self.assertEqual(empty_ci["low"], 0.0)
        self.assertEqual(empty_ci["high"], 0.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required")
    def test_missing_checkpoint_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            run_reserved_evaluation(Path("non_existent_checkpoint.pt"))

    def test_missing_torch_raises_import_error(self):
        if not HAS_TORCH:
            with self.assertRaises(ImportError):
                run_reserved_evaluation(Path("non_existent_checkpoint.pt"))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required")
    def test_reserved_evaluation_evaluates_v1_and_v2(self):
        evaluated_scenarios = []

        def mock_rollout(model, scenario, device=None):
            evaluated_scenarios.append(scenario["scenario_id"])
            return {
                "affected_area_ha": 5.0,
                "drops": [],
                "aircraft_metrics": {"water_dropped_l": 0.0, "flight_s": 0.0, "rejected_orders": 0},
            }

        def mock_baseline(scenario, policy, policy_seed=None):
            return {
                "affected_area_ha": 6.0,
                "drops": [],
                "aircraft_metrics": {"water_dropped_l": 0.0, "flight_s": 0.0, "rejected_orders": 0},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "dummy.pt"
            torch.save(
                {
                    "schema_version": "dummy",
                    "model_state_dict": {},
                    "model_config": {"grid_rows": 11, "grid_cols": 11},
                },
                ckpt_path,
            )

            with patch("nightjar.evaluate_reserved.SpatialActorCritic") as mock_sac, \
                 patch("nightjar.evaluate_reserved._rollout_ppo_with_replay", side_effect=mock_rollout), \
                 patch("nightjar.evaluate_reserved.run_integrated", side_effect=mock_baseline):
                mock_model = MagicMock()
                mock_sac.return_value = mock_model

                # Test v1
                report_v1 = run_reserved_evaluation(ckpt_path, split_version="v1")
                self.assertEqual(len(evaluated_scenarios), 8)
                self.assertTrue(all(sid.startswith("discrimination-reserved-") for sid in evaluated_scenarios))
                self.assertEqual(report_v1["split"], "reserved")
                self.assertIn("predeclared_criteria", report_v1)

                # Test v2
                evaluated_scenarios.clear()
                report_v2 = run_reserved_evaluation(ckpt_path, split_version="v2")
                self.assertEqual(len(evaluated_scenarios), 32)
                self.assertTrue(all(sid.startswith("discrimination-reserved-v2-") for sid in evaluated_scenarios))
                self.assertEqual(report_v2["split"], "reserved_v2")
                self.assertEqual(report_v2["schema_version"], "nightjar-reserved-evaluation-0.2")
                self.assertIn("reduction_95_ci", report_v2["paired_comparisons"]["heuristic"])


if __name__ == "__main__":
    unittest.main()
