import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nightjar import fireflux_validation as fv


ROOT = Path(__file__).resolve().parents[1]


def source_fixture(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, payload in (("clements2007.pdf", b"clements fixture"),
                          ("kochanski2013.pdf", b"kochanski fixture")):
        (root / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def protocol_fixture(source_hashes: dict[str, str]) -> dict:
    return {
        "schema_version": "nightjar-fireflux-protocol-0.1", "crs": "LOCAL_TRANSECT_METERS",
        "model": "Unchanged Mell et al. 2007 Eq.2 from CSIRO diagnostic",
        "forcing": {"wind_2m_m_s": 3.0, "dead_moisture_fraction": 0.09},
        "gates": {"nmae_lt": 0.3, "absolute_bias_lt": 0.2, "physical_validation_pass": None},
        "primary_metrics": ["single interval normalized rate error and signed bias", "distance below observed reported rate", "conditional minimum transit time300/R_upper"],
        "primary_observable": "Kochanski et al. 2013 section 5.1 observed between-tower rate, 1.61 m/s",
        "primary_geometry": "300m downwind tower separation, from section 2; no closed perimeter",
        "primary_test": "Upper bound at W approaching infinity; not a finite-width forecast",
        "initialization": "First observed photo head at152s,-45m for Clements trajectories; Kochanski rate scored directly and transit300/R conditional, without inventing ST timestamp",
        "secondary_metrics": ["head residual at190s and separate195s plume endpoint", "conditional minimum tower arrival152+45/R_upper"],
        "secondary_observable": "Clements et al. 2007 p1375 head about 45m upstream at12:46:02 CST to front passage at12:46:40 CST on p1377; elapsed152 to190s from12:43:30 CST",
        "secondary_source_variants": ["4.5m plume temperature peak at12:46:45 CST is a different endpoint, retain separately", "Kochanski2013 MT thermocouple peak219s is not substituted for Clements2007 front190s"],
        "sensitivity": {"finite_width_surrogate_m": 385.0, "moisture_fraction": [0.09, 0.185], "wind_2m_m_s": [3.0, 4.0]},
        "sensitivity_meaning": "3-4m/s comes from preliminary Clements2007 AMS overview;18.5% is measured mixed live+dead morning sample not dead fuel at burn;385m is ignition length not measured head width. No confidence bounds or eligible alternative fits.",
        "uncertainty_policy": "Retain unknown instrument/input/distance/time-convention errors as unknown. Quantify two-decimal rate rounding and1Hz sample discretization only as separate arithmetic sensitivities, never a confidence interval. Report required wind/moisture for matching rates instead of invented uncertainty distributions.",
        "seed": 0, "sources": source_hashes, "frozen_at_utc": "2006-01-01T00:00:00Z",
        "observation_access_before_freeze": ["already inspected", "no candidate predictions computed"],
        "independence": "scientifically independent campaign, not blind observations",
        "candidate_code_sha256": hashlib.sha256(
            Path(fv.__file__).with_name("csiro_head_validation.py").read_bytes()).hexdigest(),
    }


class FireFluxUpperBoundTests(unittest.TestCase):
    def test_analytic_limit_bounds_finite_width_forcing_variants(self):
        for wind in (2.0, 3.0, 4.0, 5.0):
            for moisture in (0.05, 0.09, 0.185):
                upper = fv.mell_rate_upper_bound(wind, moisture)
                rates = [fv.cheney_mell_rate(wind, moisture, width) for width in (50.0, 385.0, 1000.0)]
                self.assertTrue(all(rate < upper for rate in rates))
                self.assertLess(rates[0], rates[1])
                self.assertLess(rates[1], rates[2])

    def test_inverse_equations_recover_independently_evaluated_rate(self):
        wind, moisture = 4.2, 0.113
        expected = (0.165 + 0.534 * wind) * __import__("math").exp(-0.108 * moisture * 100)
        recovered_wind = fv.minimum_wind_for_rate(expected, moisture)
        recovered_moisture = fv.maximum_moisture_for_rate(expected, wind)
        self.assertAlmostEqual(recovered_wind, wind, places=12)
        self.assertAlmostEqual(recovered_moisture, moisture, places=12)

    def test_changed_source_is_rejected_before_report(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Path(directory)
            source_fixture(sources)
            with patch.object(fv, "SOURCE_HASHES", {name: "0" * 64 for name in fv.SOURCE_HASHES}):
                with self.assertRaisesRegex(ValueError, "source hash changed"):
                    fv.run_diagnostic(sources)

    def test_default_diagnostic_path_never_fetches(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Path(directory)
            fixture_hashes = source_fixture(sources)
            with patch.object(fv, "SOURCE_HASHES", fixture_hashes), \
                 patch.object(fv.subprocess, "run", side_effect=AssertionError("unexpected download")):
                report = fv.run_diagnostic(sources)
            self.assertEqual(report["schema_version"], fv.SCHEMA_VERSION)
            self.assertIsNone(report["physical_validation_pass"])
            self.assertEqual(report["sources"][0]["sha256"], fixture_hashes["clements2007.pdf"])

    def test_cached_fetch_records_verification_without_claiming_new_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = source_fixture(root)
            with patch.object(fv, "SOURCE_HASHES", hashes), \
                 patch.object(fv.subprocess, "run", side_effect=AssertionError("unexpected download")):
                fv.fetch_sources(root)
            manifest = json.loads((root / "source-manifest.json").read_text())
            self.assertEqual(manifest["downloaded_filenames_in_this_call"], [])
            self.assertIn("verified_at_utc", manifest)
            self.assertNotIn("acquired_at_utc", manifest)

    def test_protocol_bound_cli_contract_with_synthetic_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "sources"
            output = root / "report.json"
            fixture_hashes = source_fixture(source_dir)
            protocol = protocol_fixture(fixture_hashes)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            with patch.object(fv, "SOURCE_HASHES", fixture_hashes), patch("sys.argv", [
                "fireflux_validation", "--sources", str(source_dir), "--protocol", str(protocol_path), "--output", str(output)
            ]):
                fv.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["protocol"]["sha256"], hashlib.sha256(protocol_path.read_bytes()).hexdigest())
            self.assertIn("does not establish historical observation blindness", report["protocol"]["history_note"])
            self.assertEqual(report["protocol"]["historical_observation_access_before_freeze"],
                             ["already inspected", "no candidate predictions computed"])
            self.assertEqual(report["crs"], "LOCAL_TRANSECT_METERS")
            self.assertEqual(report["experiment_time_origin_utc"], "2006-02-23T18:43:30Z")
            self.assertIsNone(report["physical_validation_pass"])
            self.assertEqual(report["observation_facts_sha256"], hashlib.sha256(
                json.dumps({"observation_facts": report["observation_facts"]}, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def test_upper_bound_reports_conditional_rejection_and_distinct_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Path(directory)
            source_fixture(sources)
            with patch.object(fv, "SOURCE_HASHES", {name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
                                                      for name in fv.SOURCE_HASHES}):
                report = fv.run_diagnostic(sources)
            primary = report["primary"]
            self.assertLess(primary["rate_m_s"], fv.OBSERVED_RATE_M_S)
            self.assertAlmostEqual(primary["nmae"], abs(primary["signed_rate_residual_m_s"]) / 1.61)
            self.assertTrue(primary["nmae_gate_rejected_by_upper_bound"])
            self.assertTrue(primary["absolute_bias_gate_rejected_by_upper_bound"])
            self.assertIsNone(report["physical_validation_pass"])
            self.assertIn("actual fire-front wind", primary["configuration_rejection_scope"])
            self.assertEqual(report["secondary_photo_trajectory"]["plume_peak_endpoint_variant"]["endpoint_kind"],
                             "4.5 m plume temperature peak, not ground arrival")
            self.assertEqual(report["secondary_photo_trajectory"]["source_time_discrepancies_s"], {
                "kochanski_mt_peak_minus_clements_plume_peak": 24.0,
                "kochanski_mt_peak_minus_clements_front_passage": 29.0,
            })
            trajectory = report["secondary_photo_trajectory"]
            self.assertAlmostEqual(trajectory["front_passage_endpoint"]["predicted_head_max_x_m"],
                                   -45 + primary["rate_m_s"] * (190 - 152))
            self.assertAlmostEqual(trajectory["front_passage_endpoint"]["arrival_time_minimum_s"],
                                   152 + 45 / primary["rate_m_s"])
            self.assertAlmostEqual(trajectory["plume_peak_endpoint_variant"]["predicted_head_max_x_m"],
                                   -45 + primary["rate_m_s"] * (195 - 152))

    def test_changed_candidate_cannot_reuse_protocol_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = protocol_fixture(fv.SOURCE_HASHES)
            protocol["candidate_code_sha256"] = "0" * 64
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "candidate code differs"):
                fv._protocol_check(path)

    def test_even_a_perfectly_feasible_bound_never_sets_physical_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Path(directory)
            fixture_hashes = source_fixture(sources)
            with patch.object(fv, "SOURCE_HASHES", fixture_hashes), \
                 patch.object(fv, "mell_rate_upper_bound", return_value=fv.OBSERVED_RATE_M_S):
                report = fv.run_diagnostic(sources)
            self.assertFalse(report["primary"]["nmae_gate_rejected_by_upper_bound"])
            self.assertFalse(report["primary"]["absolute_bias_gate_rejected_by_upper_bound"])
            self.assertIsNone(report["physical_validation_pass"])

    def test_upper_bound_above_observed_has_zero_minimum_error_not_false_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Path(directory)
            fixture_hashes = source_fixture(sources)
            with patch.object(fv, "SOURCE_HASHES", fixture_hashes), \
                 patch.object(fv, "mell_rate_upper_bound", return_value=2.1):
                report = fv.run_diagnostic(sources)
            self.assertEqual(report["primary"]["minimum_possible_nmae_given_upper_bound"], 0)
            self.assertFalse(report["primary"]["nmae_gate_rejected_by_upper_bound"])
            self.assertIsNone(report["physical_validation_pass"])

    def test_pinned_pdf_hashes_and_end_to_end_cli_when_sources_are_available(self):
        sources = ROOT / "data" / "raw" / "fireflux"
        if not all((sources / name).is_file() for name in fv.SOURCE_HASHES):
            self.skipTest("optional pinned FireFlux source PDFs absent")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fireflux-report.json"
            with patch("sys.argv", ["fireflux_validation", "--sources", str(sources), "--output", str(output)]):
                fv.main()
            report = json.loads(output.read_text())
        self.assertIsNone(report["physical_validation_pass"])
        self.assertEqual({row["filename"]: row["sha256"] for row in report["sources"]}, fv.SOURCE_HASHES)
        self.assertNotIn("acquired_at_utc", report)


if __name__ == "__main__":
    unittest.main()
