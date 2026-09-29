"""Unit tests for universal geodesy module (ETRS89 / WGS84 and UTM)."""

import math
import unittest

from nightjar.geodesy import (
    epsg_for_utm_zone,
    utm_central_meridian_deg,
    utm_to_wgs84,
    utm_zone_from_lon,
    wgs84_to_utm,
    wgs84_to_utm29n,
    wgs84_to_utm30n,
    wgs84_to_utm31n,
    utm29n_to_wgs84,
    utm30n_to_wgs84,
    utm31n_to_wgs84,
)


class TestGeodesy(unittest.TestCase):
    def test_utm_central_meridian(self):
        self.assertEqual(utm_central_meridian_deg(29), -9.0)
        self.assertEqual(utm_central_meridian_deg(30), -3.0)
        self.assertEqual(utm_central_meridian_deg(31), 3.0)
        with self.assertRaises(ValueError):
            utm_central_meridian_deg(0)
        with self.assertRaises(ValueError):
            utm_central_meridian_deg(61)

    def test_utm_zone_from_lon(self):
        self.assertEqual(utm_zone_from_lon(2.5), 31)
        self.assertEqual(utm_zone_from_lon(-3.5), 30)
        self.assertEqual(utm_zone_from_lon(-8.5), 29)

    def test_epsg_for_utm_zone(self):
        self.assertEqual(epsg_for_utm_zone(29), "EPSG:25829")
        self.assertEqual(epsg_for_utm_zone(30), "EPSG:25830")
        self.assertEqual(epsg_for_utm_zone(31), "EPSG:25831")
        self.assertEqual(epsg_for_utm_zone(30, datum="WGS84"), "EPSG:32630")
        with self.assertRaises(ValueError):
            epsg_for_utm_zone(30, datum="NAD83")

    def test_roundtrip_submillimeter_closure(self):
        """Test forward + inverse transformation closure across zones."""
        test_points = [
            # (lon, lat, zone)
            (1.0, 41.5, 31),    # Catalonia (Artesa de Segre area)
            (2.17, 41.38, 31),  # Barcelona
            (-3.70, 40.41, 30), # Madrid
            (-5.75, 39.80, 30), # Monfragüe / Casas de Miravete
            (-4.62, 36.59, 30), # Sierra de Mijas / Málaga
            (-8.61, 41.15, 29), # Porto / NW Iberia
            (-9.13, 38.72, 29), # Lisbon
        ]
        for lon, lat, zone in test_points:
            x, y = wgs84_to_utm(lon, lat, zone=zone)
            lon_back, lat_back = utm_to_wgs84(x, y, zone=zone)
            # Check angular closure (< 1e-7 deg ~ 1 cm)
            self.assertAlmostEqual(lon, lon_back, places=6)
            self.assertAlmostEqual(lat, lat_back, places=6)

            # Roundtrip distance in meters should be < 0.001 m (1 mm)
            x_back, y_back = wgs84_to_utm(lon_back, lat_back, zone=zone)
            self.assertLess(math.hypot(x - x_back, y - y_back), 0.001)

    def test_backward_compatibility_aliases(self):
        lon, lat = 1.05, 41.92
        x31, y31 = wgs84_to_utm31n(lon, lat)
        x_gen, y_gen = wgs84_to_utm(lon, lat, zone=31)
        self.assertEqual(x31, x_gen)
        self.assertEqual(y31, y_gen)

        lon_back, lat_back = utm31n_to_wgs84(x31, y31)
        self.assertAlmostEqual(lon, lon_back, places=6)
        self.assertAlmostEqual(lat, lat_back, places=6)

        # Test Zone 30 aliases
        lon30, lat30 = -5.7, 39.8
        x30, y30 = wgs84_to_utm30n(lon30, lat30)
        self.assertAlmostEqual(x30, 268842.2, delta=1.0)
        lon_back30, lat_back30 = utm30n_to_wgs84(x30, y30)
        self.assertAlmostEqual(lon30, lon_back30, places=6)
        self.assertAlmostEqual(lat30, lat_back30, places=6)

        # Test Zone 29 aliases
        lon29, lat29 = -8.5, 42.0
        x29, y29 = wgs84_to_utm29n(lon29, lat29)
        lon_back29, lat_back29 = utm29n_to_wgs84(x29, y29)
        self.assertAlmostEqual(lon29, lon_back29, places=6)
        self.assertAlmostEqual(lat29, lat_back29, places=6)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            wgs84_to_utm(math.nan, 40.0, zone=30)
        with self.assertRaises(ValueError):
            wgs84_to_utm(0.0, 95.0, zone=30)
        with self.assertRaises(ValueError):
            utm_to_wgs84(math.inf, 4000000.0, zone=30)


if __name__ == "__main__":
    unittest.main()
