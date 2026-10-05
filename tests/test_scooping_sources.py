"""CL-415 scooping-source screening and selection."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from nightjar import cems_catalog
from nightjar.cems_catalog import (
    ACTIVATIONS,
    SCOOPING_SOURCES,
    _extract_water_waypoints,
    incidents_for_split,
    scooping_point,
)
from nightjar.geodesy import utm_to_wgs84
from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
    select_scooping_source,
)

ZONE = 30
# Incidents whose CEMS hydrography already contains an eligible body (D-70).
CEMS_SCOOPING_SOURCE = {"EMSR592", "EMSR600", "EMSR583", "EMSR625_AOI02", "EMSR602_AOI04"}
SOURCE_FIELDS = {"name", "utm_easting_m", "utm_northing_m", "area_ha", "extent_m", "shore_clearance_m",
                 "source", "reference"}
X0, Y0 = 500000.0, 4400000.0


def feature(name, obj_type, ring_utm):
    ring = [list(utm_to_wgs84(x, y, zone=ZONE)) for x, y in ring_utm]
    return {"properties": {"name": name, "obj_type": obj_type},
            "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}


def rectangle(x, y, width, height):
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def c_shape(x, y):
    """A curved 2.4 km reservoir arm whose vertex mean falls on dry land."""
    return [(x, y), (x + 2400, y), (x + 2400, y + 1200), (x, y + 1200), (x, y + 1000),
            (x + 2200, y + 1000), (x + 2200, y + 200), (x, y + 200)]


class ScoopingSourceTests(unittest.TestCase):
    def test_scooping_point_lies_on_open_water_even_when_the_vertex_mean_does_not(self):
        ring = np.array(c_shape(X0, Y0))
        x, y, clearance = scooping_point([[ring]])
        mean_x, mean_y = ring.mean(axis=0)
        self.assertTrue(200 < mean_y - Y0 < 1000)  # vertex mean: on the land between the arms
        self.assertTrue(cems_catalog._inside_rings(np.array([[x, y]]), [ring])[0])
        self.assertGreaterEqual(clearance, 90.0)  # arms are 200 m wide
        self.assertLessEqual(clearance, 145.0)  # inscribed circle at the corner of two 200 m arms

    def test_only_long_wide_open_water_is_eligible(self):
        waypoints = {w["name"]: w for w in _extract_water_waypoints([
            feature("pond", "BH080-Lake", rectangle(X0, Y0, 60, 40)),
            feature("narrow canal", "BH140-River", rectangle(X0, Y0 + 2000, 3000, 60)),
            feature("marsh", "BH090-Land Subject to Inundation", rectangle(X0, Y0 + 4000, 2000, 800)),
            feature("reservoir", "BH080-Lake", c_shape(X0, Y0 + 6000)),
        ], utm_zone=ZONE)}
        self.assertFalse(waypoints["pond"]["eligible_for_scooping"])
        self.assertFalse(waypoints["narrow canal"]["eligible_for_scooping"])  # 30 m from the shore at best
        self.assertFalse(waypoints["marsh"]["eligible_for_scooping"])
        reservoir = waypoints["reservoir"]
        self.assertTrue(reservoir["eligible_for_scooping"])
        self.assertAlmostEqual(reservoir["area_ha"], (2400 * 1200 - 2200 * 800) / 1e4, delta=0.5)
        self.assertGreaterEqual(reservoir["shore_clearance_m"], 90.0)

    def test_nearest_eligible_source_is_chosen_and_recorded(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=50, cols=50)
        near = {"name": "near", "utm_easting_m": 402000.0, "utm_northing_m": 4596000.0, "eligible_for_scooping": True}
        far = dict(near, name="far", utm_northing_m=4590000.0)
        tiny = dict(near, name="tiny pond", utm_northing_m=4602500.0, eligible_for_scooping=False)
        fixture["water_scooping_waypoints"] = [far, tiny, near]
        scenario = generate_geographic_scenario(fixture, 0, "development")
        self.assertEqual(scenario["provenance"]["water_source"]["name"], "near")
        self.assertEqual(scenario["aircraft"]["water_scooping_utm"], [402000.0, 4596000.0])

    def test_incident_without_an_eligible_source_is_rejected(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=50, cols=50)
        for waypoint in fixture["water_scooping_waypoints"]:
            waypoint["eligible_for_scooping"] = False
        with self.assertRaisesRegex(ValueError, "no water body suitable"):
            generate_geographic_scenario(fixture, 0, "development")

    def test_registered_external_source_serves_real_incidents_only(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=50, cols=50)
        fixture["water_scooping_waypoints"] = []
        external = {"name": "Reviewed reservoir", "utm_easting_m": 401000.0, "utm_northing_m": 4590000.0,
                    "area_ha": 300.0, "reference": "test"}
        with patch.dict(cems_catalog.SCOOPING_SOURCES, {"EMSR578": [external]}):
            with self.assertRaises(ValueError):
                select_scooping_source(fixture, (402500.0, 4602500.0))  # synthetic fixture: not used
            real = deepcopy(fixture)
            real["source_kind"] = "copernicus_cems"
            chosen = select_scooping_source(real, (402500.0, 4602500.0))
        self.assertEqual((chosen["name"], chosen["origin"]), ("Reviewed reservoir", "registered_external"))


class RegisteredScoopingSourceTests(unittest.TestCase):
    def test_every_split_incident_has_a_scooping_source(self):
        for split in ("development", "validation", "reserved"):
            for activation in incidents_for_split(split):
                with self.subTest(activation=activation):
                    self.assertTrue(activation in CEMS_SCOOPING_SOURCE or SCOOPING_SOURCES.get(activation))

    def test_registered_sources_are_complete_and_in_the_incident_zone(self):
        self.assertTrue(SCOOPING_SOURCES)
        for activation, sources in SCOOPING_SOURCES.items():
            info = ACTIVATIONS[activation]
            west, south, east, north = info["aoi_bounds_wgs84"]
            for source in sources:
                with self.subTest(activation=activation, source=source.get("name")):
                    self.assertEqual(set(source), SOURCE_FIELDS)
                    self.assertGreaterEqual(source["area_ha"], cems_catalog.SCOOPING_MIN_AREA_HA)
                    self.assertGreaterEqual(source["extent_m"], cems_catalog.SCOOPING_MIN_EXTENT_M)
                    self.assertGreaterEqual(source["shore_clearance_m"], cems_catalog.SCOOPING_MIN_CLEARANCE_M)
                    self.assertTrue(160000.0 < source["utm_easting_m"] < 840000.0)
                    lon, lat = utm_to_wgs84(source["utm_easting_m"], source["utm_northing_m"], zone=info["utm_zone"])
                    # Within ~30 km (0.4 degrees) of the AOI read in the incident's own UTM zone.
                    self.assertTrue(west - 0.4 <= lon <= east + 0.4 and south - 0.4 <= lat <= north + 0.4,
                                    (lon, lat))


if __name__ == "__main__":
    unittest.main()
