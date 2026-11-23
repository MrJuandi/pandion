"""Tests for the multi-focus, shifting-wind family and synthetic interval forcing."""

from math import hypot
import random
import unittest

from pandion.environment import validate_forcing
from pandion.geographic_family import build_synthetic_cems_fixture, canonical_hash, generate_geographic_scenario
from pandion.geographic_integrated import run_geographic_integrated
from pandion.multi_ignition_family import (
    FOCI_RANGE,
    MIN_SEPARATION_CELLS,
    SPEED_LIMITS_M_S,
    WIND_SEGMENT_S,
    derive_scenario,
    sample_foci,
    shifting_wind,
)

MOISTURE = {"dead_1h": 0.05, "dead_10h": 0.06, "dead_100h": 0.07, "dead_herbaceous": 0.05,
            "live_herbaceous": 0.3, "live_woody": 0.6}


def parent_scenario():
    fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=50, cols=50)
    scenario = generate_geographic_scenario(fixture, index=0, split="development",
                                            window_rows=40, window_cols=40, horizon_s=1800.0)
    scenario["fire"]["moisture"] = dict(MOISTURE)
    scenario["fire"]["burn_mask"] = [[8 <= r < 32 and 8 <= c < 32 for c in range(40)] for r in range(40)]
    scenario.pop("environment", None)
    scenario.pop("scenario_hash", None)
    scenario["scenario_hash"] = canonical_hash(scenario)
    return scenario


class MultiIgnitionFamilyTests(unittest.TestCase):
    def test_sample_foci_respects_separation(self):
        candidates = [(r, c) for r in range(30) for c in range(30)]
        foci = sample_foci(candidates, 4, random.Random(1))
        self.assertEqual(len(foci), 4)
        for i, (r, c) in enumerate(foci):
            for r2, c2 in foci[i + 1:]:
                self.assertGreaterEqual(max(abs(r - r2), abs(c - c2)), MIN_SEPARATION_CELLS)
        self.assertEqual(len(sample_foci([(0, 0), (1, 1)], 2, random.Random(0))), 1)

    def test_shifting_wind_is_valid_continuous_and_bounded(self):
        forcing = shifting_wind((1.5, 1.5), 3600.0, MOISTURE, random.Random(3))
        self.assertEqual(len(forcing), int(3600 / WIND_SEGMENT_S))
        self.assertEqual(validate_forcing({"schema_version": "nightjar-environment-0.1",
                                           "weather_mode": "synthetic_intervals", "forcing": forcing}, 3600.0), forcing)
        for segment in forcing:
            speed = hypot(*segment["midflame_wind_to_m_s"])
            self.assertTrue(SPEED_LIMITS_M_S[0] - 1e-6 <= speed <= SPEED_LIMITS_M_S[1] + 1e-6)
            self.assertEqual(segment["available_at_s"], segment["start_s"])
        directions = {tuple(s["midflame_wind_to_m_s"]) for s in forcing}
        self.assertGreater(len(directions), 1)

    def test_synthetic_forcing_rejects_gaps_and_late_availability(self):
        forcing = shifting_wind((1.0, 0.0), 1800.0, MOISTURE, random.Random(0))
        env = {"schema_version": "nightjar-environment-0.1", "weather_mode": "synthetic_intervals"}
        gap = [dict(s) for s in forcing]
        gap[1]["start_s"] += 10.0
        with self.assertRaises(ValueError):
            validate_forcing({**env, "forcing": gap}, 1800.0)
        late = [dict(s) for s in forcing]
        late[0]["available_at_s"] = late[0]["end_s"]
        with self.assertRaises(ValueError):
            validate_forcing({**env, "forcing": late}, 1800.0)
        with self.assertRaises(ValueError):
            validate_forcing({**env, "forcing": forcing}, 3600.0)

    def test_derived_scenario_has_separated_foci_shifting_wind_and_runs(self):
        parent = parent_scenario()
        child = derive_scenario(parent, 0)
        again = derive_scenario(parent, 0)
        self.assertEqual(child, again)  # deterministic
        self.assertNotEqual(child["scenario_hash"], parent["scenario_hash"])
        foci = child["provenance"]["foci"]
        self.assertTrue(FOCI_RANGE[0] <= len(foci) <= FOCI_RANGE[1])
        mask = child["fire"]["burn_mask"]
        self.assertEqual(child["provenance"]["initial_burned_cells"], sum(map(sum, mask)))
        for focus in foci:
            r, c = focus["centre_row_col"]
            self.assertTrue(mask[r][c])
            self.assertTrue(parent["fire"]["burn_mask"][r][c])  # inside the CEMS burned area
        self.assertEqual(child["environment"]["weather_mode"], "synthetic_intervals")
        replay = run_geographic_integrated(child, "heuristic")
        statuses = {d["observation"]["weather_status"] for d in replay["decisions"]}
        self.assertEqual(statuses, {"realtime_synthetic_interval"})
        winds = {tuple(d["observation"]["midflame_wind_to_m_s"]) for d in replay["decisions"]}
        self.assertGreater(len(winds), 1)
        # One-cell foci (D-74) in slow litter fuels need longer than the horizon to cross a
        # 100 m cell; check spread in uniform grass.
        child["fire"]["fuel_grid"] = [[102] * 40 for _ in range(40)]
        spread = run_geographic_integrated(child, "no_intervention")
        self.assertGreater(spread["outcomes"]["final_burned_ha"], spread["outcomes"]["initial_burned_ha"])

    def test_station_forcing_parents_keep_measured_weather(self):
        parent = parent_scenario()
        parent["environment"] = {"weather_mode": "station_intervals", "forcing": ["kept"]}
        parent.pop("scenario_hash")
        parent["scenario_hash"] = canonical_hash(parent)
        child = derive_scenario(parent, 1)
        self.assertEqual(child["environment"], parent["environment"])


if __name__ == "__main__":
    unittest.main()
