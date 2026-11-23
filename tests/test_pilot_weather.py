from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pandion.pilot_weather import VARIABLES, audit_weather


class PilotWeatherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.scenario = {"scenario_id": "synthetic-weather-fixture", "observation_timestamps": {
            "t0_initial_utc": "2022-06-17T10:06:00Z", "t1_monitoring_utc": "2022-06-17T10:47:00Z",
            "t2_monitoring_utc": "2022-06-17T11:12:00Z"}}
        self.stations = [{"codi_estacio": "X6", "nom_estacio": "Synthetic station", "longitud": "1", "latitud": "42", "altitud": "300", "data_inici": "2000-01-01T00:00:00"}]
        self.variables = [{"codi_variable": code, "nom_variable": name, "unitat": unit} for code, (name, unit) in VARIABLES.items()]
        self.rows = [{"codi_estacio": "X6", "codi_variable": code, "data_lectura": instant, "valor_lectura": "1.0", "codi_estat": "V", "codi_base": "SH"}
                     for code in VARIABLES for instant in ["2022-06-17T10:00:00.000", "2022-06-17T10:30:00.000", "2022-06-17T11:00:00.000"]]

    def audit(self):
        paths = []
        for name, value in [("stations", self.stations), ("variables", self.variables), ("measurements", self.rows)]:
            path = self.directory / (name + ".json")
            path.write_text(json.dumps(value))
            paths.append(path)
        return audit_weather(self.scenario, *paths)

    def test_forward_intervals_cover_partial_endpoints_and_preserve_metadata(self):
        result = self.audit()
        self.assertTrue(result["complete_valid_coverage"])
        temperature = result["variables"]["air_temperature"]
        self.assertEqual(temperature["expected_intervals"], 3)
        self.assertEqual(temperature["records"][0]["interval_start_utc"], "2022-06-17T10:00:00Z")
        self.assertEqual(temperature["records"][0]["interval_end_utc"], "2022-06-17T10:30:00Z")
        self.assertEqual(result["window"]["t0_to_t1_s"], 2460)
        self.assertEqual(len(result["sources"]["exports"]["measurements"]["sha256"]), 64)

    def test_missing_and_unvalidated_records_are_not_filled(self):
        self.rows = self.rows[1:]
        self.rows[0]["codi_estat"] = "T"
        result = self.audit()
        self.assertFalse(result["complete_valid_coverage"])
        temperature = result["variables"]["air_temperature"]
        self.assertEqual(temperature["usable_intervals"], 1)
        self.assertEqual(len(temperature["missing_intervals_utc"]), 1)
        self.assertEqual(len(temperature["unusable_intervals_utc"]), 1)

    def test_duplicate_wrong_station_nonfinite_and_wrong_cadence_rejected(self):
        original = deepcopy(self.rows)
        for key, value in [("codi_estacio", "OTHER"), ("valor_lectura", "nan"), ("codi_base", "HO"), ("data_lectura", "2022-06-17T10:01:00")]:
            self.rows = deepcopy(original)
            self.rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit()
        self.rows = original + [original[0]]
        with self.assertRaises(ValueError):
            self.audit()

    def test_metadata_units_and_window_must_match(self):
        self.variables[0]["unitat"] = "K"
        with self.assertRaises(ValueError):
            self.audit()
        self.variables[0]["unitat"] = "°C"
        self.scenario["observation_timestamps"]["t2_monitoring_utc"] = "2022-06-17T10:00:00Z"
        with self.assertRaises(ValueError):
            self.audit()


if __name__ == "__main__":
    unittest.main()
