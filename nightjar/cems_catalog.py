"""Copernicus Emergency Management Service (CEMS) multi-incident catalog and scenario builder.

This module provides unified ingestion for multiple Copernicus wildfire activations:
- EMSR578: Artesa de Segre (Lleida, Catalonia / UTM Zone 31N)
- EMSR590: Las Hurdes / Sierra de Francia (Cáceres, Extremadura / UTM Zone 30N)
- EMSR592: Landiras / La Teste-de-Buch (Gironde / Landes, France / UTM Zone 30N)
- EMSR600: Massarosa (Lucca, Tuscany, Italy / UTM Zone 32N)

All activations are georeferenced to standard ETRS89 / UTM (Zones 29N, 30N, 31N, 32N)
with CORINE Land Cover crosswalked to Scott & Burgan 40 fuel models.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Sequence
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from .geodesy import epsg_for_utm_zone, utm_to_wgs84, wgs84_to_utm
from .pilot_emsr578 import (
    CORINE_TO_SCOTT_BURGAN as EMSR578_CORINE_CROSSWALK,
    _points_in_polygon,
    _rasterize_polygons,
)

SCHEMA_VERSION = "nightjar-cems-scenario-0.1"
BASE_URL_TEMPLATE = "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/{activation}/"

# Extended universal CORINE Land Cover -> Scott & Burgan 40 crosswalk
UNIVERSAL_CORINE_CROSSWALK: dict[str, dict[str, Any]] = deepcopy(EMSR578_CORINE_CROSSWALK)
UNIVERSAL_CORINE_CROSSWALK.update({
    "223": {
        "clc_code": "223",
        "clc_name": "Olive groves",
        "sb_model": "NB",
        "sb_code": 0,
        "description": "Nonburnable (tilled olive groves acting as fuelbreaks)",
        "confidence": "High",
        "rationale": "Commercial and traditional olive groves in Spain with tilled bare soil prevent surface spread.",
        "alternatives": [
            {"model": "GR1", "code": 101, "rationale": "Unweeded olive groves with cured grass understory"},
        ],
    },
    "231": {
        "clc_code": "231",
        "clc_name": "Pastures",
        "sb_model": "GR2",
        "sb_code": 102,
        "description": "Low-moderate load, dry climate grass",
        "confidence": "High",
        "rationale": "Herbaceous grazing pastures completely cured during summer heatwaves.",
        "alternatives": [
            {"model": "GR1", "code": 101, "rationale": "Intensively grazed sparse grass"},
        ],
    },
    "322": {
        "clc_code": "322",
        "clc_name": "Moors and heathland",
        "sb_model": "SH6",
        "sb_code": 146,
        "description": "High load, humid/montane shrub (dense heathland / brezales)",
        "confidence": "High",
        "rationale": "Dense Ericaceous heathland (Erica, Calluna, Genista) with high flammable shrub biomass.",
        "alternatives": [
            {"model": "SH5", "code": 145, "rationale": "Drier Mediterranean maquis"},
        ],
    },
    "333": {
        "clc_code": "333",
        "clc_name": "Sparsely vegetated areas",
        "sb_model": "GR1",
        "sb_code": 101,
        "description": "Very low load sparse grass on rocky substrate",
        "confidence": "Medium",
        "rationale": "Steep rocky slopes with scattered tufts of cured grass and dwarf scrub.",
        "alternatives": [
            {"model": "NB", "code": 0, "rationale": "Bare exposed bedrock without continuous fuel"},
        ],
    },
    "411": {
        "clc_code": "411",
        "clc_name": "Inland marshes",
        "sb_model": "NB",
        "sb_code": 0,
        "description": "Nonburnable (water-saturated marshland)",
        "confidence": "High",
        "rationale": "Wetland vegetation saturated with standing water acts as a nonburnable barrier.",
        "alternatives": [],
    },
})


def universal_clc_to_scott_burgan(clc_code: str) -> int:
    """Return Scott & Burgan fuel model integer code for a 3-digit CORINE Land Cover code."""
    clean_code = str(clc_code).split("-")[0].strip()
    entry = UNIVERSAL_CORINE_CROSSWALK.get(clean_code)
    if entry is None:
        return 0
    return int(entry["sb_code"])


# Reviewed CEMS Wildfire Activation Registry
ACTIVATIONS: dict[str, dict[str, Any]] = {
    "EMSR578": {
        "activation_id": "EMSR578",
        "name": "Artesa de Segre",
        "province": "Lleida",
        "region": "Catalonia",
        "country": "Spain",
        "utm_zone": 31,
        "epsg": "EPSG:25831",
        "aoi_bounds_wgs84": (0.5, 41.5, 1.5, 42.5),
        "default_split": "development",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR578/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-06-17T10:06:00Z",
                "sha256": "4ad2197f1a7ccd4b8da0e63a71f2dbc6aee53d58688c448f25ed9cdf0c01f026",
                "reported_area_ha": 1501.73,
                "vector_filename": "EMSR578_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
            "monit01": {
                "stage": "monit01",
                "product_code": "MONIT01",
                "acquired_at_utc": "2022-06-18T10:47:00Z",
                "sha256": "41f4fa2dac9e76d3034e244f9d24f3ae788b05c2557ed2b19a9aff7c8b6e6a9e",
                "reported_area_ha": 2111.21,
                "vector_filename": "EMSR578_AOI01_DEL_MONIT01_r1_RTP01_v1_vector.zip",
            },
            "monit02": {
                "stage": "monit02",
                "product_code": "MONIT02",
                "acquired_at_utc": "2022-06-19T10:42:00Z",
                "sha256": "92b11953975914332507e49f2ca6bb2192be07bcfbcb0e471c37152e68d831ac",
                "reported_area_ha": 2693.19,
                "vector_filename": "EMSR578_AOI01_DEL_MONIT02_r1_RTP01_v1_vector.zip",
            },
        },
    },
    "EMSR590": {
        "activation_id": "EMSR590",
        "name": "Las Hurdes / Sierra de Francia",
        "province": "Cáceres",
        "region": "Extremadura",
        "country": "Spain",
        "utm_zone": 30,
        "epsg": "EPSG:25830",
        "aoi_bounds_wgs84": (-6.6, 40.2, -6.0, 40.7),
        "default_split": "validation",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR590/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-07-14T10:48:00Z",
                "sha256": "46238c94a62a32144b5d9cb721ecb941bb8b5fa7e24bc7c4f2d8e56575871645",
                "reported_area_ha": 4637.40,
                "vector_filename": "EMSR590_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
        },
    },
    "EMSR592": {
        "activation_id": "EMSR592",
        "name": "Landiras / La Teste-de-Buch",
        "province": "Gironde / Landes",
        "region": "Nouvelle-Aquitaine",
        "country": "France",
        "utm_zone": 30,
        "epsg": "EPSG:25830",
        "aoi_bounds_wgs84": (-0.9, 44.2, -0.1, 44.9),
        "default_split": "reserved",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR592/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-07-17T10:26:00Z",
                "sha256": "8891a611dee603a431b67e82ab86dc6fcd53432e0c3342f359a269ccb48537a2",
                "reported_area_ha": 6726.49,
                "vector_filename": "EMSR592_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
        },
    },
    "EMSR600": {
        "activation_id": "EMSR600",
        "name": "Massarosa",
        "province": "Lucca",
        "region": "Tuscany",
        "country": "Italy",
        "utm_zone": 32,
        "epsg": "EPSG:25832",
        "aoi_bounds_wgs84": (10.0, 43.5, 10.7, 44.2),
        "default_split": "reserved",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR600/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-07-21T10:23:00Z",
                "sha256": "e64eb4fb7f2465888a02615d2adf38278726d7fba413c4e470dfb278728ac2a0",
                "reported_area_ha": 836.73,
                "vector_filename": "EMSR600_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
        },
    },
}



def _development_incident(key: str, cems_activation: str, aoi: str, name: str, province: str, region: str,
                          utm_zone: int, bounds: tuple[float, float, float, float], delivered_at_utc: str,
                          sha256_digest: str, burnt_area_ha: float, version: str = "v1") -> dict[str, Any]:
    """Registry entry for an additional Spanish development incident (one AOI of a CEMS activation)."""
    return {
        "activation_id": key,
        "cems_activation": cems_activation,
        "aoi": aoi,
        "name": name,
        "province": province,
        "region": region,
        "country": "Spain",
        "utm_zone": utm_zone,
        "epsg": f"EPSG:{25800 + utm_zone}",
        "aoi_bounds_wgs84": bounds,
        "default_split": "development",
        "source_page": f"https://mapping.emergency.copernicus.eu/activations/{cems_activation}/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                # Image acquisition time is printed only on the map PDF, which was not retrieved.
                "acquired_at_utc": None,
                "delivered_at_utc": delivered_at_utc,
                "sha256": sha256_digest,
                "reported_area_ha": burnt_area_ha,
                "vector_filename": f"{cems_activation}_{aoi}_DEL_PRODUCT_r1_RTP01_{version}_vector.zip",
            },
        },
    }


# Additional development-only incidents (D-57). Burnt area is the sum of the
# observedEventA ``area`` attributes; delivery times are from the activation page.
ACTIVATIONS.update({
    "EMSR583": _development_incident(
        "EMSR583", "EMSR583", "AOI01", "Castejón de Tornos", "Teruel", "Aragón", 30,
        (-1.6, 40.8, -1.1, 41.3), "2022-06-23T14:48:38Z",
        "59c39487e614c163d9572a827646b1e6e17f959d1933c9938173c31e57e94b06", 1496.74),
    "EMSR625_AOI01": _development_incident(
        "EMSR625_AOI01", "EMSR625", "AOI01", "Bejís", "Castellón", "Comunitat Valenciana", 30,
        (-1.0, 39.6, -0.4, 40.2), "2022-08-19T18:11:46Z",
        "65faa95e0aca6a252584b5ca5b18514ee18607176aeb549a51d00acb69d5fca3", 17525.87),
    "EMSR625_AOI02": _development_incident(
        "EMSR625_AOI02", "EMSR625", "AOI02", "Vall d'Ebo", "Alicante", "Comunitat Valenciana", 30,
        (-0.5, 38.5, 0.2, 39.0), "2022-08-19T17:08:55Z",
        "b19b8ec8b8bc6cec35933a7e2f90df6fb337cd96af3808925c6b8140161e6af2", 9332.3),
    "EMSR602_AOI02": _development_incident(
        "EMSR602_AOI02", "EMSR602", "AOI02", "San Esteban de Valdueza", "León", "Castilla y León", 29,
        (-6.9, 42.2, -6.3, 42.7), "2022-07-22T00:00:00Z",
        "ccd2e91ae04605336be7935dae6d583aeefa9d095a6345b214484d13c617f5cb", 1404.7),
    "EMSR602_AOI04": _development_incident(
        "EMSR602_AOI04", "EMSR602", "AOI04", "Cebreros", "Ávila", "Castilla y León", 30,
        (-4.7, 40.3, -4.2, 40.7), "2022-07-22T00:00:00Z",
        "7911b3c74855b1148e443945c792abe21314471ae3331297a05dfc9210237e1c", 4096.7),
})

# Further development-only incidents (D-71): every mainland-Spain wildfire
# delineation package on the CEMS mapping site (2019-2022) that carries the
# naturalLandUseA, observedEventA and areaOfInterestA layers. Later Rapid Mapping
# packages (2023 onwards) no longer include land use and are not usable as is.
# Burnt area is the observedEventA polygon area (EMSR365 and EMSR375 have no area
# attribute; elsewhere it matches the attribute sum within 2 ha). Delivery time is
# the earliest file timestamp in the package; no time zone is recorded there.
# EMSR457's AOI is labelled 'Patras' in the package, but its geometry lies in
# Almonaster la Real (Huelva), the activation's location.
ACTIVATIONS.update({
    "EMSR365": _development_incident(
        "EMSR365", "EMSR365", "AOI01", "Torre de l'Espanyol", "Tarragona", "Catalonia", 31,
        (0.3, 41.1, 0.9, 41.6), "2019-06-28T17:43:24Z",
        "7824a7ab7c43bd0ef2fd58abcfc9b02304cd07885f323fa42eca5fea07360aa6", 5043.9),
    "EMSR375": _development_incident(
        "EMSR375", "EMSR375", "AOI01", "Real Sitio de San Ildefonso", "Segovia", "Castilla y León", 30,
        (-4.1, 40.8, -3.8, 41.0), "2019-08-05T23:38:56Z",
        "3d053a7b5d110a8057e10dd8191bba1d1c873b69fc265bc4c26deb438bf3fded", 351.3),
    "EMSR457": _development_incident(
        "EMSR457", "EMSR457", "AOI01", "Almonaster la Real", "Huelva", "Andalucía", 29,
        (-6.9, 37.6, -6.6, 37.9), "2020-09-01T09:45:52Z",
        "3aa22ab1ab31ef7b5e7745cadcab2036e5722affa59bc78e9f6ebe040d38ca58", 8593.5, version="v2"),
    "EMSR458": _development_incident(
        "EMSR458", "EMSR458", "AOI01", "Cabezuela del Valle", "Cáceres", "Extremadura", 30,
        (-5.9, 40.0, -5.5, 40.3), "2020-08-29T21:15:26Z",
        "129366e52400a82874770d9f48265899ec8ed68fcb73d845270d65ab6f9b0376", 2568.0),
    "EMSR522": _development_incident(
        "EMSR522", "EMSR522", "AOI01", "Isso (Hellín)", "Albacete", "Castilla-La Mancha", 30,
        (-2.0, 38.3, -1.6, 38.7), "2021-07-26T19:57:34Z",
        "2d2d3471b86e75e57f28fd1ada3193116fe3068834e3fc0477d37dc2636a0e3f", 1082.2),
    "EMSR538": _development_incident(
        "EMSR538", "EMSR538", "AOI01", "Navalacruz", "Ávila", "Castilla y León", 30,
        (-5.2, 40.2, -4.3, 40.8), "2021-08-17T22:55:56Z",
        "27741cf44c1e5c4ce34668a5b532ac1aa15a4dd91fa6e5eb6fc5afac417b9644", 22723.3),
    "EMSR545": _development_incident(
        "EMSR545", "EMSR545", "AOI01", "Jubrique (Sierra Bermeja)", "Málaga", "Andalucía", 30,
        (-5.4, 36.3, -5.0, 36.7), "2021-09-10T23:26:50Z",
        "e1ff1466784cb3de2e174ef48f1ddaa3389b1d76748d3141d45b686cda7feca1", 3689.7),
    "EMSR581": _development_incident(
        "EMSR581", "EMSR581", "AOI01", "Nonaspe", "Zaragoza", "Aragón", 31,
        (0.0, 41.0, 0.6, 41.5), "2022-06-21T21:41:12Z",
        "5fe7e9b676d7a7d18e6ff3e0b54fc958ae9d2795101ca2007ad2b3922a97a18b", 2210.9),
    "EMSR597": _development_incident(
        "EMSR597", "EMSR597", "AOI01", "Moros (Ateca)", "Zaragoza", "Aragón", 30,
        (-2.1, 41.2, -1.6, 41.6), "2022-07-21T03:05:30Z",
        "70eea1b18af61b7add5e56d0a01822598f667fbe6195961af1132fc280e05992", 8463.1),
    "EMSR599_AOI01": _development_incident(
        "EMSR599_AOI01", "EMSR599", "AOI01", "Folgoso do Courel", "Lugo", "Galicia", 29,
        (-7.6, 42.4, -6.9, 42.8), "2022-07-22T03:28:12Z",
        "9375019a81f3e208f889ae66f3b535a72ff49e0c350968a7b95a2ce658f3ff14", 12714.6),
    "EMSR599_AOI02": _development_incident(
        "EMSR599_AOI02", "EMSR599", "AOI02", "O Barco de Valdeorras", "Ourense", "Galicia", 29,
        (-7.2, 42.1, -6.6, 42.6), "2022-07-22T02:16:46Z",
        "24b67e3400d2f0f764b221da3ee5e5e8540509a3f669c102b66460588d1b966b", 11419.4),
    "EMSR620": _development_incident(
        "EMSR620", "EMSR620", "AOI01", "Boca de Huérgano", "León", "Castilla y León", 30,
        (-5.0, 42.8, -4.7, 43.1), "2022-08-11T22:46:18Z",
        "a81babf8430ed42c9bd33a509ec5657642de44d8044100178abe04d0254aaf88", 1097.5),
    "EMSR623": _development_incident(
        "EMSR623", "EMSR623", "AOI01", "Añón de Moncayo", "Zaragoza", "Aragón", 30,
        (-1.9, 41.6, -1.4, 42.0), "2022-08-14T22:18:54Z",
        "5f8c2d0fdd092148dfa8bfdefcec3ed673874c00e2424e4be6b5a91acc539460", 4469.1),
    "EMSR632": _development_incident(
        "EMSR632", "EMSR632", "AOI01", "Los Guájares", "Granada", "Andalucía", 30,
        (-3.8, 36.7, -3.4, 37.0), "2022-09-11T17:52:28Z",
        "f992abde999a33b243d2d179284ca8cea74cac578ccbeb8a34596b5a26ad3ec9", 3769.2),
})

FIXED_SPLIT_INCIDENTS = {"validation": ("EMSR590",), "reserved": ("EMSR592", "EMSR600")}


def incidents_for_split(split: str) -> tuple[str, ...]:
    """Registered incidents designated for a split; validation and reserved stay fixed."""
    if split in FIXED_SPLIT_INCIDENTS:
        return FIXED_SPLIT_INCIDENTS[split]
    if split != "development":
        raise ValueError(f"Unknown split: {split}")
    return tuple(key for key, info in ACTIVATIONS.items() if info["default_split"] == "development")


def raw_package_name(activation_id: str, stage: str = "del") -> str:
    """Local file name for a cached package, e.g. ``emsr578_del.zip`` or ``emsr625-aoi01_del.zip``."""
    return f"{activation_id.lower().replace('_', '-')}_{stage}.zip"


def cems_product_url(activation_id: str, stage: str = "del") -> str:
    """Return the official download URL for a CEMS vector package."""
    activation = ACTIVATIONS.get(activation_id.upper())
    if not activation:
        raise ValueError(f"Unknown activation: {activation_id}")
    prod = activation["products"].get(stage)
    if not prod:
        raise ValueError(f"Unknown stage '{stage}' for activation {activation_id}")
    folder = activation.get("cems_activation", activation_id.upper())
    return BASE_URL_TEMPLATE.format(activation=folder) + prod["vector_filename"]


def fetch_cems_package(activation_id: str, stage: str, output_path: Path) -> Path:
    """Download and verify SHA-256 digest of a CEMS vector package."""
    activation = ACTIVATIONS.get(activation_id.upper())
    if not activation:
        raise ValueError(f"Unknown activation: {activation_id}")
    prod = activation["products"].get(stage)
    if not prod:
        raise ValueError(f"Unknown stage '{stage}' for activation {activation_id}")

    expected_sha = prod["sha256"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        current_sha = sha256(output_path.read_bytes()).hexdigest()
        if current_sha == expected_sha:
            return output_path

    url = cems_product_url(activation_id, stage)
    with urlopen(url, timeout=90) as response:
        content = response.read()

    actual_sha = sha256(content).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"{activation_id} {stage}: download hash ({actual_sha}) differs from reviewed hash ({expected_sha})"
        )

    output_path.write_bytes(content)
    return output_path


def _validate_aoi_geographic_bounds(
    activation_id: str,
    features: list[dict[str, Any]],
    bounds: Sequence[float],
) -> None:
    """Reject a correctly hashed package whose AOI belongs to another location.

    Bounds are broad WGS84 envelopes recorded for each registered product AOI,
    ordered west, south, east, north. A package digest establishes byte
    identity; this independent check guards the catalog's geographic identity.
    """
    west, south, east, north = map(float, bounds)
    coordinates: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if (isinstance(value, (list, tuple)) and len(value) >= 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))):
            coordinates.append((float(value[0]), float(value[1])))
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for feature in features:
        geom = feature.get("geometry") or {}
        if geom.get("type") in ("Polygon", "MultiPolygon"):
            collect(geom.get("coordinates", []))

    if not coordinates:
        raise ValueError(f"No polygon coordinates found in AOI for {activation_id}")

    outside = [(lon, lat) for lon, lat in coordinates
               if not (west <= lon <= east and south <= lat <= north)]
    if outside:
        lon, lat = outside[0]
        raise ValueError(
            f"{activation_id}: AOI coordinate ({lon:.6f}, {lat:.6f}) falls outside "
            f"registered WGS84 bounds ({west}, {south}, {east}, {north})"
        )


# CL-415 scooping needs a long, wide, open water surface (the manufacturer quotes
# roughly 1.3-1.4 km of water length and about 90 m width at sea level). CEMS
# hydrography also maps ponds, pools and marshes, so a water body is eligible
# only if it passes these geometric proxies; they are screening thresholds, not
# an operational assessment of depth, obstacles or approach paths.
SCOOPING_MIN_AREA_HA = 10.0
SCOOPING_MIN_EXTENT_M = 1300.0
SCOOPING_MIN_CLEARANCE_M = 45.0  # distance from the scooping point to the shore (half width)
NON_SCOOPABLE_TYPE_PREFIXES = ("BH090",)  # land subject to inundation (marsh, lagoons)

# Reviewed scooping sources outside the CEMS area of interest, per activation.
# CEMS maps hydrography only inside the AOI, so the reservoirs aircraft would
# actually use can be missing. Each entry: name, utm_easting_m, utm_northing_m
# (a point on open water, activation UTM zone), area_ha, source and reference.
#
# The entries below are OpenStreetMap natural=water / landuse=reservoir polygons
# within 30 km of the tactical window centre, retrieved from the Overpass API on
# 2026-10-05 and screened with _extract_water_waypoints (same thresholds and
# scooping_point as CEMS hydrography); the 1-3 nearest eligible bodies are kept.
# Excluded on review: Meandro El Melero (osm way/704298460, EMSR590), a river bend
# whose 1.4 km extent is the diagonal of a curve rather than a straight run.
# Data (c) OpenStreetMap contributors, available under the ODbL.
OSM_RETRIEVED_UTC = "2026-10-05"


def _osm_source(name: str, easting_m: float, northing_m: float, area_ha: float, extent_m: float,
                shore_clearance_m: float, osm_id: str) -> dict[str, Any]:
    return {"name": name, "utm_easting_m": easting_m, "utm_northing_m": northing_m, "area_ha": area_ha,
            "extent_m": extent_m, "shore_clearance_m": shore_clearance_m, "source": "OpenStreetMap",
            "reference": f"osm {osm_id}, retrieved {OSM_RETRIEVED_UTC}"}


SCOOPING_SOURCES: dict[str, list[dict[str, Any]]] = {
    "EMSR578": [
        _osm_source("Pantà de Camarasa", 322486.15, 4643939.15, 551.08, 14160.5, 285.8, "relation/5598197"),
        _osm_source("Pantà de Sant Llorenç de Montgai", 320647.84, 4636430.05, 126.26, 4676.4, 360.5, "relation/6004039"),
        _osm_source("Pantà de Terradets", 325406.12, 4659203.28, 276.94, 5154.8, 410.0, "relation/5598230"),
    ],
    "EMSR590": [
        _osm_source("Embalse de Águeda", 205364.89, 4491741.66, 121.77, 7744.2, 148.4, "way/48120464"),
        _osm_source("Unnamed reservoir", 199781.31, 4485847.04, 16.01, 2606.0, 53.0, "way/675416760"),
    ],
    "EMSR625_AOI01": [
        _osm_source("Pantano del Regajo", 711437.29, 4418791.29, 75.01, 3210.5, 250.1, "way/30016888"),
        _osm_source("Embalse de Arenoso", 708428.31, 4440890.81, 294.76, 6192.4, 438.9, "relation/2566648"),
        _osm_source("Embassament d'Algar", 724426.17, 4407558.45, 13.45, 1427.7, 56.0, "way/981657125"),
    ],
    "EMSR602_AOI02": [
        _osm_source("Embalse de la Fuente del Azufre", 699460.02, 4715769.74, 23.5, 2573.0, 96.5, "way/23609119"),
        _osm_source("Embalse de Campañana", 683931.75, 4708094.96, 92.94, 1935.2, 360.4, "way/47234341"),
        _osm_source("Encoro de Penarrubia", 679858.92, 4703635.5, 112.26, 5832.6, 190.1, "relation/18080576"),
    ],
    # D-71 incidents without an eligible CEMS body, same query and screen.
    # OSM water=river sections were excluded on review (river reaches, not
    # scoopable straight runs), e.g. near EMSR458 and EMSR623.
    "EMSR375": [
        _osm_source("Embalse del Pontón Alto", 413016.94, 4529840.51, 66.56, 2687.3, 184.1, "relation/20054369"),
        _osm_source("Embalse de Pinilla", 432416.68, 4532088.12, 430.23, 5599.5, 630.9, "way/30822229"),
        _osm_source("Embalse de Navacerrada", 414840.17, 4507923.17, 73.12, 1522.2, 344.1, "way/19104785"),
    ],
    "EMSR458": [
        _osm_source("Unnamed reservoir", 277714.11, 4438195.69, 45.81, 1598.1, 178.7, "way/321883985"),
        _osm_source("Embalse de Baños", 254087.16, 4463454.91, 201.85, 3939.0, 424.4, "way/41763249"),
        _osm_source("Embalse de Navamuño", 263949.3, 4469790.62, 67.12, 2151.7, 253.7, "way/41762577"),
    ],
    "EMSR538": [
        _osm_source("Embalse de El Burguillo", 363916.61, 4476397.63, 855.5, 12799.9, 466.4, "relation/3836574"),
        _osm_source("Embalse del Castro de Las Cogotas", 357239.99, 4508457.15, 326.02, 6617.2, 286.9, "way/24314675"),
        _osm_source("Embalse del Charco del Cura", 372327.42, 4475796.18, 30.92, 2208.4, 113.4, "way/30141667"),
    ],
    "EMSR597": [
        _osm_source("Pantano de la Tranquera", 601308.99, 4566922.94, 440.38, 6601.2, 288.7, "relation/3875391"),
        _osm_source("Embalse de Maidevera", 603395.37, 4603655.55, 104.68, 2991.4, 209.6, "way/26438927"),
        _osm_source("Embalse de Monteagudo de las Vicarías", 568462.39, 4582142.74, 96.15, 2347.0, 273.4, "way/414475700"),
    ],
    "EMSR620": [
        _osm_source("Embalse de Riaño", 336750.37, 4759052.7, 2048.97, 16684.1, 644.5, "relation/254780"),
        _osm_source("Embalse de Besande", 347759.33, 4748148.6, 20.25, 1427.9, 99.5, "way/55081817"),
        _osm_source("Embalse de Compuerto", 351443.86, 4747454.94, 369.16, 8449.5, 314.9, "way/24365335"),
    ],
    "EMSR623": [
        _osm_source("Embalse del Val", 600420.21, 4636730.16, 97.39, 3151.2, 179.5, "way/243010131"),
        _osm_source("La Loteta", 638785.82, 4630443.69, 1051.66, 6664.8, 1206.0, "relation/20353654"),
    ],
}


def _ring_area_m2(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2.0


def _inside_rings(points: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Even-odd test over all rings of one polygon (outer ring and holes)."""
    inside = np.zeros(len(points), dtype=bool)
    px, py = points[:, 0][:, None], points[:, 1][:, None]
    for ring in rings:
        x1, y1 = ring[:, 0][None, :], ring[:, 1][None, :]
        x2, y2 = np.roll(ring[:, 0], -1)[None, :], np.roll(ring[:, 1], -1)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = ((y1 > py) != (y2 > py)) & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
        inside ^= (crosses.sum(axis=1) % 2).astype(bool)
    return inside


def _shore_distance(points: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    a = np.concatenate([ring for ring in rings])
    b = np.concatenate([np.roll(ring, -1, axis=0) for ring in rings])
    ab = b - a
    length2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    best = np.full(len(points), np.inf)
    for chunk in range(0, len(points), 512):
        p = points[chunk:chunk + 512][:, None, :]
        t = np.clip(((p - a[None]) * ab[None]).sum(axis=2) / length2[None], 0.0, 1.0)
        nearest = a[None] + t[..., None] * ab[None]
        best[chunk:chunk + 512] = np.sqrt(((p - nearest) ** 2).sum(axis=2)).min(axis=1)
    return best


def scooping_point(polygons: list[list[np.ndarray]]) -> tuple[float, float, float]:
    """Point on open water farthest from the shore: (x, y, clearance_m).

    ``polygons`` are lists of UTM rings (outer ring first, then holes). A grid
    search is refined once around the best cell; the result always lies inside
    the water, unlike the mean of the polygon vertices.
    """
    best = (0.0, 0.0, -1.0)
    for rings in polygons:
        points = np.concatenate(rings)
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        step = max(5.0, math.sqrt(max((x1 - x0) * (y1 - y0), 1.0) / 2500.0))
        centre, span = ((x0 + x1) / 2, (y0 + y1) / 2), ((x1 - x0) / 2 + step, (y1 - y0) / 2 + step)
        for _ in range(2):
            gx = np.arange(centre[0] - span[0], centre[0] + span[0] + step, step)
            gy = np.arange(centre[1] - span[1], centre[1] + span[1] + step, step)
            grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
            grid = grid[_inside_rings(grid, rings)]
            if not len(grid):
                break
            distance = _shore_distance(grid, rings)
            k = int(np.argmax(distance))
            if distance[k] > best[2]:
                best = (float(grid[k, 0]), float(grid[k, 1]), float(distance[k]))
            centre, span, step = tuple(grid[k]), (2 * step, 2 * step), step / 5
    return best


def _extract_water_waypoints(features: list[dict], utm_zone: int) -> list[dict[str, Any]]:
    """Hydrography polygons with size, extent and a scooping point on open water.

    ``eligible_for_scooping`` applies the CL-415 screening thresholds above.
    The point of an eligible body is the open-water point farthest from the
    shore; other bodies keep the mean of their vertices (informative only).
    """
    waypoints = []
    for hf in features:
        props = hf.get("properties", {})
        geom = hf.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        if gtype == "Point":
            polygons_ll = []
            all_ll = [(float(coords[0]), float(coords[1]))]
        elif gtype in ("Polygon", "MultiPolygon"):
            polygons_ll = [coords] if gtype == "Polygon" else coords
            all_ll = [(float(pt[0]), float(pt[1])) for poly in polygons_ll for ring in poly for pt in ring]
        else:
            continue
        if not all_ll:
            continue
        polygons = [[np.array([wgs84_to_utm(float(pt[0]), float(pt[1]), zone=utm_zone) for pt in ring])
                     for ring in poly if len(ring) >= 3] for poly in polygons_ll]
        polygons = [rings for rings in polygons if rings]
        area_ha = sum(_ring_area_m2(rings[0]) - sum(_ring_area_m2(h) for h in rings[1:])
                      for rings in polygons) / 1e4
        utm_points = np.concatenate([ring for rings in polygons for ring in rings]) if polygons else None
        extent_m = float(np.hypot(*(utm_points.max(axis=0) - utm_points.min(axis=0)))) if utm_points is not None else 0.0
        obj_type = props.get("obj_type", "Waterbody")
        name = props.get("name", obj_type)
        if name == "Unknown":
            name = obj_type
        candidate = (area_ha >= SCOOPING_MIN_AREA_HA and extent_m >= SCOOPING_MIN_EXTENT_M
                     and not str(obj_type).startswith(NON_SCOOPABLE_TYPE_PREFIXES))
        clearance = None
        if candidate:
            ux, uy, clearance = scooping_point(polygons)
            point = "open-water point farthest from the shore"
        else:
            clon = sum(p[0] for p in all_ll) / len(all_ll)
            clat = sum(p[1] for p in all_ll) / len(all_ll)
            ux, uy = wgs84_to_utm(clon, clat, zone=utm_zone)
            point = "mean of polygon vertices (may lie outside the water)"
        eligible = bool(candidate and clearance is not None and clearance >= SCOOPING_MIN_CLEARANCE_M)
        lon, lat = utm_to_wgs84(ux, uy, zone=utm_zone)
        waypoints.append({
            "name": name,
            "obj_type": obj_type,
            "lon_deg": round(lon, 6),
            "lat_deg": round(lat, 6),
            "utm_easting_m": round(ux, 2),
            "utm_northing_m": round(uy, 2),
            "area_ha": round(area_ha, 2),
            "extent_m": round(extent_m, 1),
            "shore_clearance_m": round(clearance, 1) if clearance is not None else None,
            "point": point,
            "eligible_for_scooping": eligible,
        })
    return waypoints


def build_cems_scenario(
    activation_id: str,
    zip_path: Path,
    cell_size_m: float = 100.0,
    stage: str = "del",
) -> dict[str, Any]:
    """Build a georeferenced raster scenario from any reviewed Copernicus CEMS vector package.

    Parameters:
    - activation_id: reviewed activation code (e.g. 'EMSR578', 'EMSR590').
    - zip_path: path to vector zip archive.
    - cell_size_m: grid cell resolution in meters (default 100 m).
    - stage: product stage ('del', 'monit01', etc.).
    """
    if cell_size_m <= 0 or not math.isfinite(cell_size_m):
        raise ValueError("cell_size_m must be a positive finite number")

    act_info = ACTIVATIONS.get(activation_id.upper())
    if not act_info:
        raise ValueError(f"Unknown activation: {activation_id}")
    utm_zone = act_info["utm_zone"]
    epsg = act_info["epsg"]
    prod_info = act_info["products"].get(stage)
    if prod_info is None:
        raise ValueError(f"Unknown stage '{stage}' for activation {activation_id}")

    if not zip_path.exists():
        raise FileNotFoundError(f"Vector package not found: {zip_path}")
    actual_sha = sha256(zip_path.read_bytes()).hexdigest()
    if actual_sha != prod_info["sha256"]:
        raise ValueError(f"{activation_id} {stage}: source ZIP SHA-256 {actual_sha} differs from registered hash {prod_info['sha256']}")

    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        aoi_names = [n for n in names if "areaOfInterestA" in n and n.endswith(".json")]
        landuse_names = [n for n in names if "naturalLandUseA" in n and n.endswith(".json")]
        hydro_names = [n for n in names if "hydrographyA" in n and n.endswith(".json")]
        obs_names = [n for n in names if "observedEventA" in n and n.endswith(".json")]

        if not aoi_names or not landuse_names or not obs_names:
            raise ValueError(f"{zip_path}: missing required GeoJSON layers (aoi, naturalLandUse, observedEvent)")

        aoi_data = json.loads(archive.read(aoi_names[0]))
        landuse_data = json.loads(archive.read(landuse_names[0]))
        hydro_data = json.loads(archive.read(hydro_names[0])) if hydro_names else {"type": "FeatureCollection", "features": []}
        obs_data = json.loads(archive.read(obs_names[0]))

    _validate_aoi_geographic_bounds(
        activation_id.upper(),
        aoi_data.get("features", []),
        act_info["aoi_bounds_wgs84"],
    )

    # Compute bounding box in UTM coordinates from AOI polygon
    utm_coords: list[tuple[float, float]] = []
    for feat in aoi_data["features"]:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type")
        if gtype == "Polygon":
            polys = [coords]
        elif gtype == "MultiPolygon":
            polys = coords
        else:
            continue
        for poly in polys:
            for ring in poly:
                for lon, lat in ring:
                    x, y = wgs84_to_utm(float(lon), float(lat), zone=utm_zone)
                    utm_coords.append((x, y))

    if not utm_coords:
        raise ValueError(f"No coordinates found in AOI for {activation_id}")

    xs = [c[0] for c in utm_coords]
    ys = [c[1] for c in utm_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Grid dimensions aligned to cell_size_m
    cols = int(math.ceil((max_x - min_x) / cell_size_m))
    rows = int(math.ceil((max_y - min_y) / cell_size_m))

    # Grid cell center coordinates
    x_centers = min_x + (np.arange(cols) + 0.5) * cell_size_m
    y_centers = min_y + (np.arange(rows) + 0.5) * cell_size_m
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Helper projection function for ray-casting
    def _proj(lon: float, lat: float) -> tuple[float, float]:
        return wgs84_to_utm(lon, lat, zone=utm_zone)

    # 1. Rasterize naturalLandUseA into Scott & Burgan fuel model codes
    fuel_flat = np.zeros(len(grid_points), dtype=np.int32)

    def _lookup_fuel(props: dict) -> int:
        info = props.get("info", "")
        code = info.split("-")[0].strip() if info else ""
        return universal_clc_to_scott_burgan(code)

    _rasterize_cems_polygons(landuse_data["features"], grid_points, fuel_flat, _lookup_fuel, _proj)

    # 2. Rasterize hydrographyA as nonburnable (code 0) and record water mask
    water_flat = np.zeros(len(grid_points), dtype=bool)
    if hydro_data["features"]:
        _rasterize_cems_polygons(hydro_data["features"], grid_points, fuel_flat, 0, _proj)
        _rasterize_cems_polygons(hydro_data["features"], grid_points, water_flat, True, _proj)

    # 3. Rasterize observedEventA (burn mask at stage)
    burn_flat = np.zeros(len(grid_points), dtype=bool)
    _rasterize_cems_polygons(obs_data["features"], grid_points, burn_flat, True, _proj)

    # Reshape to 2D
    fuel_grid = fuel_flat.reshape((rows, cols))
    water_mask = water_flat.reshape((rows, cols))
    burn_mask = burn_flat.reshape((rows, cols))

    cell_area_ha = (cell_size_m**2) / 10000.0
    burn_cells = int(np.sum(burn_mask))
    burn_area_ha = round(burn_cells * cell_area_ha, 4)

    # Extract water scooping waypoints
    utm_waypoints = _extract_water_waypoints(hydro_data.get("features", []), utm_zone=utm_zone)


    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": f"cems-{activation_id.lower()}-{stage}-0.1",
        "activation_id": activation_id.upper(),
        "name": act_info["name"],
        "region": act_info["region"],
        "default_split": act_info["default_split"],
        "provenance": {
            "source": f"Copernicus Emergency Management Service On-Demand Mapping, {activation_id} ({act_info['name']})",
            "source_page": act_info["source_page"],
            "license": "Copernicus terms of use (open access, attribution required)",
            "source_zip_sha256": actual_sha,
            "source_stage": stage,
            "source_product_code": prod_info["product_code"],
            "source_product_filename": prod_info["vector_filename"],
            "source_url": cems_product_url(activation_id, stage),
            "image_acquired_at_utc": prod_info["acquired_at_utc"],
            "acquired_at_utc": prod_info["acquired_at_utc"],
            "units": {"coordinates": "m", "cell_size": "m", "burned_area": "ha"},
            "resolution_m": cell_size_m,
            "coordinate_contract": {
                "grid_origin": "southwest cell boundary in projected CRS meters",
                "array_axes": "row increases north; column increases east",
                "source_sampling": "cell centers at origin + (index + 0.5) * resolution",
                "fuel_coverage": "code 0 combines water, nonburnable, unmapped, and unavailable coverage; no independent source-coverage mask",
            },
        },
        "position_semantics": {
            "source_raster_samples": "cell centers",
            "discrete_drop_targets": "local cell-anchor proxies (column * resolution, row * resolution), not measured physical drop centers",
        },
        "crs": {
            "authority": "EPSG",
            "code": epsg,
            "utm_zone": utm_zone,
            "horizontal_datum": "ETRS89",
        },
        "grid": {
            "rows": rows,
            "cols": cols,
            "cell_size_m": cell_size_m,
            "origin_utm_easting_m": round(min_x, 3),
            "origin_utm_northing_m": round(min_y, 3),
            "origin_semantics": "southwest cell boundary",
            "extent_ha": round(rows * cols * cell_area_ha, 2),
        },
        "initial_perimeter": {
            "burned_cells_count": burn_cells,
            "rasterized_area_ha": burn_area_ha,
            "reported_satellite_area_ha": prod_info.get("reported_area_ha"),
            "discretization_error_ha": round(burn_area_ha - prod_info.get("reported_area_ha", burn_area_ha), 4),
        },
        "fuel_grid": fuel_grid.tolist(),
        "water_mask": water_mask.tolist(),
        "burn_mask": burn_mask.tolist(),
        "water_scooping_waypoints": utm_waypoints,
    }


def _rasterize_cems_polygons(
    features: list[dict],
    grid_points: np.ndarray,
    target_grid: np.ndarray,
    value_lookup: Any,
    proj_func: Any,
) -> None:
    """Rasterize features into target_grid using bounding box ray-casting with proj_func."""
    for feat in features:
        val = value_lookup(feat["properties"]) if callable(value_lookup) else value_lookup
        geom = feat.get("geometry")
        if not geom:
            continue
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            polys = [coords]
        elif gtype == "MultiPolygon":
            polys = coords
        else:
            continue

        for poly in polys:
            if not poly or not poly[0]:
                continue
            outer = np.array([proj_func(lon, lat) for lon, lat in poly[0]], dtype=float)
            bb_x0 = float(outer[:, 0].min())
            bb_x1 = float(outer[:, 0].max())
            bb_y0 = float(outer[:, 1].min())
            bb_y1 = float(outer[:, 1].max())

            sub_idx = np.where(
                (grid_points[:, 0] >= bb_x0) & (grid_points[:, 0] <= bb_x1) &
                (grid_points[:, 1] >= bb_y0) & (grid_points[:, 1] <= bb_y1)
            )[0]
            if len(sub_idx) == 0:
                continue

            in_poly = _points_in_polygon(grid_points[sub_idx], outer)
            for hole in poly[1:]:
                if not hole:
                    continue
                hole_pts = np.array([proj_func(lon, lat) for lon, lat in hole], dtype=float)
                in_hole = _points_in_polygon(grid_points[sub_idx], hole_pts)
                in_poly = in_poly & ~in_hole

            target_grid[sub_idx[in_poly]] = val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "fetch", "build", "report"))
    parser.add_argument("--activation", type=str, default="EMSR578")
    parser.add_argument("--stage", type=str, default="del")
    parser.add_argument("--directory", type=Path, default=Path("data/raw/cems"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.action == "list":
        print(f"{'Activation':<10} {'Name':<28} {'Region':<20} {'UTM Zone':<10} {'Split':<12}")
        print("-" * 82)
        for act_id, info in ACTIVATIONS.items():
            print(f"{act_id:<10} {info['name']:<28} {info['region']:<20} {info['utm_zone']:<10} {info['default_split']:<12}")

    elif args.action == "fetch":
        target = args.directory / f"{args.activation.lower()}_{args.stage}.zip"
        path = fetch_cems_package(args.activation, args.stage, target)
        print(f"Downloaded and verified {args.activation} {args.stage} -> {path}")

    elif args.action == "build":
        zip_path = args.directory / f"{args.activation.lower()}_{args.stage}.zip"
        scenario = build_cems_scenario(args.activation, zip_path, stage=args.stage)
        payload = json.dumps(scenario, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
            print(f"Built scenario for {args.activation} -> {args.output}")
        else:
            print(payload)


if __name__ == "__main__":
    main()
