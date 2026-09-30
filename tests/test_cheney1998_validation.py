"""Deterministic synthetic contract tests; no new empirical outcomes here."""
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nightjar import cheney1998_validation as experiment
from nightjar.csiro_head_validation import canonical_hash, score_series


class CandidateInputTests(unittest.TestCase):
    def tables(self, root, missing_wind=False):
        plot_fields = ["FireID", "10mWind.ms", "2mWind.ms", "DFMC.%", "Cure.%", "FuelTreat", "EffHFWidth.m", "ROS.ms", "RSS.ms"]
        if missing_wind:
            plot_fields.remove("10mWind.ms")
        period_fields = ["FireID", "DFMC.%", "Cure.%", "FuelTreat", "ROS.ms"]
        paths = {name: root / name for name in ("Plot_Average_Data.csv", "Burn_Period_Data.csv")}
        for name, fields, rows in (
            ("Plot_Average_Data.csv", plot_fields, [
                dict(zip(["FireID", "10mWind.ms", "2mWind.ms", "DFMC.%", "Cure.%", "FuelTreat", "EffHFWidth.m", "ROS.ms", "RSS.ms"],
                         ["C064", "2", "99", "6", "100", "E2", "50", "forbidden", "forbidden"])),
                dict(zip(["FireID", "10mWind.ms", "2mWind.ms", "DFMC.%", "Cure.%", "FuelTreat", "EffHFWidth.m", "ROS.ms", "RSS.ms"],
                         ["F19", "3", "99", "7", "95", "T1", "175", "forbidden", "forbidden"])),
            ]),
            ("Burn_Period_Data.csv", period_fields, [
                {"FireID": "C064", "DFMC.%": "6", "Cure.%": "95", "FuelTreat": "E3", "ROS.ms": "forbidden"},
                {"FireID": "F19", "DFMC.%": "7", "Cure.%": "95", "FuelTreat": "T1", "ROS.ms": "forbidden"},
            ]),
        ):
            with paths[name].open("w", newline="") as f:
                writer = csv.DictWriter(f, fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        return paths

    def test_explicit_10m_column_and_percent_units_without_outcome_access(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, audit = experiment.load_candidate_inputs(self.tables(Path(directory)))
        self.assertAlmostEqual(inputs["C064"]["plot_table"]["open_wind_10m_km_h"], 7.2)
        self.assertEqual(inputs["C064"]["plot_table"]["dead_moisture_percent"], 6.0)
        self.assertEqual(inputs["C064"]["plot_table"]["pasture_condition"], "cut_grazed")
        self.assertEqual(inputs["C064"]["period_fuel_variant"]["curing_percent"], 95)
        self.assertEqual(audit["C064"]["period_treatment"], "E3")
        self.assertEqual(list(inputs["F19"]), ["plot_table"])
        self.assertEqual(inputs["F19"]["plot_table"]["pasture_condition"], "undisturbed")

    def test_missing_10m_input_never_falls_back_to_available_2m_column(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyError):
                experiment.load_candidate_inputs(self.tables(Path(directory), missing_wind=True))


class FrozenProtocolTests(unittest.TestCase):
    def definition(self):
        return {"reviewed_model": "fixed", "seed": 0}

    def test_freeze_binds_definition_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            with patch.object(experiment, "experiment_definition", return_value=self.definition()):
                with patch.object(experiment.cheney1998, "grass_rate", side_effect=AssertionError("freeze must not predict")):
                    value = experiment.freeze_protocol(path)
                self.assertEqual(experiment.checked_protocol(path), value)
                with self.assertRaises(FileExistsError):
                    experiment.freeze_protocol(path)
            changed = {"reviewed_model": "changed", "seed": 0}
            with patch.object(experiment, "experiment_definition", return_value=changed):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    experiment.checked_protocol(path)

    def test_corrupt_definition_hash_and_non_utc_timestamp_rejected(self):
        definition = self.definition()
        value = {"schema_version": experiment.PROTOCOL_VERSION, "definition": definition,
                 "definition_sha256": canonical_hash(definition), "frozen_at_utc": "2026-01-01T00:00:00+00:00"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for change in ({"definition_sha256": "invalid"}, {"frozen_at_utc": "2026-01-01T00:00:00"},
                           {"frozen_at_utc": "2026-01-01T00:00:00+02:00"}, {"frozen_at_utc": "2999-01-01T00:00:00+00:00"}):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**value, **change}))
                    with patch.object(experiment, "experiment_definition", return_value=definition):
                        with self.assertRaises(ValueError):
                            experiment.checked_protocol(path)

    def test_run_checks_protocol_before_any_prediction(self):
        with patch.object(experiment, "checked_protocol", side_effect=ValueError("mismatch")), \
             patch.object(experiment.cheney1998, "grass_rate") as predict:
            with self.assertRaises(ValueError):
                experiment.run_experiment(Path("unneeded"))
            predict.assert_not_called()

    def test_reports_never_overwrite_and_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            experiment.write_new_json(path, {"physical_validation_pass": None})
            with self.assertRaises(FileExistsError):
                experiment.write_new_json(path, {})
            self.assertIsNone(json.loads(path.read_text())["physical_validation_pass"])
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                experiment.write_new_json(invalid, {"bad": float("nan")})
            self.assertFalse(invalid.exists())


class ComparisonTests(unittest.TestCase):
    def test_zero_speed_retains_all_windows_and_undefined_arrivals(self):
        result = experiment.zero_rate_score("F19", [(56, 80), (86, 120), (138, 200)])
        self.assertEqual(result["rate_nmae"], 1)
        self.assertEqual(result["rate_normalized_bias"], -1)
        self.assertFalse(result["restricted_rate_gate_pass"])
        self.assertEqual(len(result["records"]), 2)
        self.assertFalse(result["records"][1]["eligible_primary_window"])
        self.assertIsNone(result["arrival_mae_s"])
        self.assertTrue(all(r["predicted_arrival_s"] is None for r in result["records"]))
        self.assertTrue(all(r["predicted_position_m"] == 80 for r in result["records"]))

    def synthetic_fixture(self):
        grid = {"wind_2m_m_s": [3, 4], "dead_moisture_percent": [9, 18.5], "curing_percent": [95, 100],
                "pasture_condition": ["undisturbed", "cut_grazed", "eaten_out"], "assumed_u2_over_u10": .81}
        values = {"open_wind_10m_km_h": 20, "dead_moisture_percent": 6, "curing_percent": 95, "pasture_condition": "undisturbed"}
        definition = {"inputs": {"C064": {"plot_table": values, "period_fuel_variant": {**values, "curing_percent": 100}},
                                  "F19": {"plot_table": values}}, "acceptance_exclusion": "synthetic test exclusion",
                      "fireflux": {"sensitivity_only": grid, "missing": ["open10m", "curing"]},
                      "other_ineligible": {"RxCADRE": "blocked"}, "units": {}, "crs": "LOCAL_METERS", "time_origin": "synthetic"}
        samples = {"C064": [(27, 0), (53, 26), (100, 73)], "F19": [(56, 0), (86, 30), (138, 82)]}
        observations = {burn: {source: points for source in ("digitized_max_x", "prepared_head", "original_cumulative_distance")}
                        for burn, points in samples.items()}
        previous = {"observations": observations, "scores": {burn: {model: {source: score_series(burn, points, 1.0)
                                                                          for source, points in observations[burn].items()}
                                                                   for model in experiment.csiro.MODELS} for burn in samples},
                    "runtime": {"python": "test", "pyretechnics": "test"}}
        return definition, previous

    def test_all_cases_baselines_failures_and_sensitivities_are_retained(self):
        definition, previous = self.synthetic_fixture()
        frozen = {"definition": definition, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        prior_fireflux = {"sentinel": "prior failure retained"}
        with patch.object(experiment, "checked_protocol", return_value=frozen), \
             patch.object(experiment, "digest", return_value="test"), \
             patch.object(experiment.csiro, "run_diagnostic", return_value=deepcopy(previous)), \
             patch.object(experiment.fireflux, "run_diagnostic", return_value=prior_fireflux):
            result = experiment.run_experiment(Path("unused"), annaburroo=Path("unused"), firemodels=Path("unused"), fireflux_sources=Path("unused"))
        self.assertEqual(result["previous_csiro_experiment"], previous)
        self.assertEqual(result["previous_fireflux_experiment"], prior_fireflux)
        self.assertEqual(len(result["fireflux_sensitivity_only"]), 24)
        self.assertTrue(all(not r["eligible_main_comparison"] for r in result["fireflux_sensitivity_only"]))
        self.assertEqual(result["fireflux_main_comparison"]["status"], "not_evaluable")
        self.assertIsNone(result["fireflux_main_comparison"]["candidate_rate_m_s"])
        self.assertFalse(result["restricted_improvement_across_burns_and_sources"])
        self.assertIsNone(result["physical_validation_pass"])
        self.assertEqual(len(result["candidate_scores"]["C064"]), 2)
        f19 = result["candidate_scores"]["F19"]["plot_table"]["digitized_max_x"]
        self.assertEqual(len(f19["records"]), 2)
        self.assertFalse(f19["records"][1]["eligible_primary_window"])


if __name__ == "__main__":
    unittest.main()
