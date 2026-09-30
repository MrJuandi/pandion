import csv
import math
from pathlib import Path
import tempfile
import unittest

from nightjar.csiro_head_validation import (
    cheney_mell_rate, forecast_head, load_forcing, parse_front_curves,
    run_diagnostic, score_series,
)


class CSIROHeadDiagnosticTests(unittest.TestCase):
    def test_forecast_keeps_initial_anchor_and_does_not_reset_to_observations(self):
        result = score_series("C064", [(27, 15), (53, 46), (100, 100)], 0.5)
        self.assertEqual([r["predicted_position_m"] for r in result["records"]], [28, 51.5])
        self.assertAlmostEqual(result["records"][1]["predicted_arrival_s"], 197)
        self.assertFalse(result["restricted_rate_gate_pass"])
        self.assertAlmostEqual(result["position_mae_m"], (18 + 48.5) / 2)

    def test_wind_shift_tail_is_retained_but_cannot_improve_primary_scores(self):
        short = score_series("F19", [(56, 80), (86, 120)], 1)
        long = score_series("F19", [(56, 80), (86, 120), (138, 172)], 1)
        self.assertEqual(short["rate_nmae"], long["rate_nmae"])
        self.assertEqual(short["rate_normalized_bias"], long["rate_normalized_bias"])
        self.assertEqual(long["primary_window_count"], 1)
        self.assertFalse(long["records"][1]["eligible_primary_window"])
        self.assertEqual(len(long["records"]), 2)

    def test_invalid_observations_and_nonfinite_predictions_are_rejected(self):
        for samples in ([(27, 15)], [(27, 15), (27, 40)], [(27, 15), (53, 10)], [(27, 15), (53, math.nan)]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                score_series("C064", samples, 1)
        with self.assertRaises(ValueError):
            forecast_head(27, 15, math.inf, 53)
        with self.assertRaises(ValueError):
            score_series("F19", [(86, 120), (138, 200)], 1)

    def test_ragged_front_pairs_and_max_x_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "front.csv"
            path.write_text("27 x,27 y,53 x,53 y\n10,-5,30,-10\n15,0,40,0\n,,25,15\n")
            curves = parse_front_curves(path)
            self.assertEqual(len(curves[27]), 2)
            self.assertEqual(max(x for x, _ in curves[53]), 40)
            path.write_text("27 x,27 y\n10,\n")
            with self.assertRaises(ValueError):
                parse_front_curves(path)

    def test_input_extraction_ignores_reported_rates_and_future_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = root / "Plot_Average_Data.csv"
            periods = root / "Burn_Period_Data.csv"
            columns = ["FireID", "FuelLoad.kgm2", "FuelHt.m", "DFMC.%", "MidFlwind.ms", "2mWind.ms", "EffHFWidth.m", "Cure.%", "ROS.ms"]
            rows = [[b, .3, .2, 6, 2, 4, 50, 95, 999] for b in ("C064", "F19")]
            with plot.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows)
            periods.write_text("FireID,SAVR.cm,TimeSinceIgn,CumDist.m,ROS.ms\nC064,97.7,100,999,999\nF19,122.4,138,999,999\n")
            paths = {plot.name: plot, periods.name: periods}
            before = load_forcing(paths)
            plot.write_text(plot.read_text().replace("999", "123456"))
            periods.write_text(periods.read_text().replace("999", "123456"))
            self.assertEqual(before, load_forcing(paths))
            self.assertEqual(before["C064"]["sav_m_inv"], 9770)

    def test_grass_reference_percent_moisture_and_finite_width(self):
        wide = cheney_mell_rate(4.8, .058, 1e12)
        self.assertAlmostEqual(wide, (0.165 + 0.534 * 4.8) * math.exp(-0.108 * 5.8), places=9)
        self.assertLess(cheney_mell_rate(4.8, .058, 175), wide)
        with self.assertRaises(ValueError):
            cheney_mell_rate(4.8, .058, 0)

    def test_pinned_external_sources_end_to_end(self):
        root = Path(__file__).resolve().parents[1] / "data" / "raw"
        if not (root / "csiro-firemodels" / "Case_C064_exp.csv").exists() or not (root / "annaburroo" / "Plot_Average_Data.csv").exists():
            self.skipTest("optional pinned external CSIRO/Firemodels sources absent")
        report = run_diagnostic(root / "annaburroo", root / "csiro-firemodels")
        self.assertIsNone(report["physical_validation_pass"])
        self.assertIsNone(report["experiment_time_origin_utc"])
        for burn in ("C064", "F19"):
            self.assertEqual(set(report["scores"][burn]["fm1_original_inputs"]), {"digitized_max_x", "prepared_head", "original_cumulative_distance"})
            for result in report["scores"][burn]["cheney_mell_reference"].values():
                self.assertIsNone(result["restricted_rate_gate_pass"])
        self.assertEqual(report["scores"]["F19"]["fm1_original_inputs"]["digitized_max_x"]["primary_window_count"], 1)


if __name__ == "__main__":
    unittest.main()
