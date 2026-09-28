from pathlib import Path
import unittest

from nightjar.annaburroo_rothermel_sensitivity import compare, growth_distance
from nightjar.rothermel import (FM1_DEPTH_FT, FM1_LOAD_LB_FT2, FM1_SAV_FT_INV,
                                FT_PER_M, LB_FT2_PER_KG_M2, fm1_components,
                                one_class_components)


class RothermelSensitivityTests(unittest.TestCase):
    def test_general_one_class_reproduces_fm1(self):
        for moisture, wind in ((0.03, 0), (0.06, 2), (0.12, 4)):
            reference = fm1_components(moisture, wind)
            general = one_class_components(
                moisture, wind, FM1_LOAD_LB_FT2 / LB_FT2_PER_KG_M2,
                FM1_DEPTH_FT / FT_PER_M, FM1_SAV_FT_INV / 30.48)
            self.assertAlmostEqual(general["r0_m_s"], reference["r0_m_s"], places=12)
            self.assertAlmostEqual(general["phi_w"], reference["phi_w"], places=12)

    def test_growth_integral(self):
        self.assertEqual(growth_distance(2, 10, 0), 20)
        self.assertEqual(growth_distance(2, 0, 30), 0)
        self.assertGreater(growth_distance(2, 10, 30), 0)
        self.assertLess(growth_distance(2, 10, 30), 20)
        with self.assertRaises(ValueError):
            growth_distance(2, 10, -1)

    def test_audited_sensitivity(self):
        directory = Path("data/raw/annaburroo")
        plot = directory / "Plot_Average_Data.csv"
        periods = directory / "Burn_Period_Data.csv"
        if not plot.exists() or not periods.exists():
            self.skipTest("audited CSIRO tables unavailable; fetch with nightjar.annaburroo")
        result = compare(plot, periods)
        self.assertEqual(result["included_fires"], 120)
        self.assertEqual(result["omitted_fires_without_fuel_bed"], [])
        variants = result["variants"]
        self.assertAlmostEqual(variants["FM1_tau_0s"]["final_distance_mae_m"], 73.81050017772336)
        self.assertAlmostEqual(variants["FM1_tau_60s"]["final_distance_mae_m"], 52.50142413808279)
        self.assertAlmostEqual(variants["measured_bed_all_dead_tau_0s"]["final_distance_mae_m"], 91.29566685348355)
        fully_cured = variants["measured_bed_all_dead_tau_0s"]["by_curing"]["fully_cured_100_percent"]
        partly_cured = variants["measured_bed_all_dead_tau_0s"]["by_curing"]["partly_cured_below_100_percent"]
        self.assertEqual((fully_cured["fires"], partly_cured["fires"]), (73, 47))
        self.assertAlmostEqual(fully_cured["final_distance_mae_m"], 56.737908597421544)
        self.assertAlmostEqual(partly_cured["final_distance_mae_m"], 144.97048286821817)


if __name__ == "__main__":
    unittest.main()
