from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nightjar.annaburroo_temporal import compare


class AnnaburrooTemporalTests(unittest.TestCase):
    def test_audited_temporal_diagnostic(self):
        directory = Path("data/raw/annaburroo")
        plot = directory / "Plot_Average_Data.csv"
        period = directory / "Burn_Period_Data.csv"
        if not plot.exists() or not period.exists():
            self.skipTest("audited CSIRO tables unavailable; fetch with nightjar.annaburroo")
        result = compare(plot, period)
        self.assertEqual((result["included_fires"], result["included_periods"], result["excluded_periods"]),
                         (120, 576, 123))
        self.assertEqual(result["unmatched_plot_ignition_types"], {"point": 96, "line_or_other": 27})
        self.assertAlmostEqual(result["final_distance_mae_m"], 73.81050017772336)
        self.assertAlmostEqual(result["mean_fire_period_ros_mae_m_s"], 0.6096305849955312)
        self.assertEqual(result["fires_at_or_above_fm1_extinction"], ["B032", "B033", "B041", "B044"])
        self.assertTrue(all(fire["periods"][-1] == fire["final"] for fire in result["fires"]))

    def test_changed_source_is_rejected(self):
        with TemporaryDirectory() as directory:
            plot = Path(directory) / "Plot_Average_Data.csv"
            period = Path(directory) / "Burn_Period_Data.csv"
            plot.write_text("changed")
            period.write_text("changed")
            with self.assertRaisesRegex(ValueError, "differs from the audited"):
                compare(plot, period)


if __name__ == "__main__":
    unittest.main()
