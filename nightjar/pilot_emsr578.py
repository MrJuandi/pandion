"""Pilot data pipeline and fuel crosswalk for Copernicus EMSR578 AOI01 (Artesa de Segre).

This module implements the Phase 0 geographic pilot pipeline:
1. Exact geodetic transformation between WGS84 and ETRS89 / UTM Zone 31N (EPSG:25831).
2. Documented crosswalk from CORINE Land Cover classes to Scott & Burgan 40 fuel models.
3. Raster scenario generation from Copernicus vector packages.
4. Spatial validation metrics (IoU, Dice, commission, omission) for comparing simulated
   versus observed satellite fire perimeters.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZipFile

import numpy as np
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------------
# Geodetic Projection: WGS84 (EPSG:4326) <-> ETRS89 / UTM Zone 31N (EPSG:25831)
# ---------------------------------------------------------------------------
# GRS80 / WGS84 ellipsoid parameters
_A = 6378137.0  # semi-major axis (meters)
_F = 1.0 / 298.257223563  # flattening
_B = _A * (1.0 - _F)  # semi-minor axis
_E2 = (_A**2 - _B**2) / (_A**2)  # first eccentricity squared
_E_PRIME2 = (_A**2 - _B**2) / (_B**2)  # second eccentricity squared
_K0 = 0.9996  # UTM scale factor
_LON0_RAD = math.radians(3.0)  # UTM Zone 31N central meridian: 3 degrees East
_FALSE_EASTING = 500000.0  # False Easting (meters)
_FALSE_NORTHING = 0.0  # False Northing (meters)


def wgs84_to_utm31n(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Convert WGS84 geographic coordinates (degrees) to ETRS89 / UTM Zone 31N (meters).

    Uses Krüger's transverse Mercator series on the GRS80/WGS84 ellipsoid.
    Returns (easting_m, northing_m).
    """
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        raise ValueError("Coordinates must be finite numbers")
    if not (-180.0 <= lon_deg <= 180.0 and -80.0 <= lat_deg <= 84.0):
        raise ValueError(f"Coordinates ({lon_deg}, {lat_deg}) out of valid UTM range")

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n_val = _A / math.sqrt(1.0 - _E2 * sin_lat**2)
    t_val = tan_lat**2
    c_val = _E_PRIME2 * cos_lat**2
    a_val = (lon - _LON0_RAD) * cos_lat

    # Meridional distance M
    m_val = _A * (
        (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0) * lat
        - (3.0 * _E2 / 8.0 + 3.0 * _E2**2 / 32.0 + 45.0 * _E2**3 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * _E2**2 / 256.0 + 45.0 * _E2**3 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * _E2**3 / 3072.0) * math.sin(6.0 * lat)
    )

    x = _K0 * n_val * (
        a_val
        + (1.0 - t_val + c_val) * a_val**3 / 6.0
        + (5.0 - 18.0 * t_val + t_val**2 + 72.0 * c_val - 58.0 * _E_PRIME2) * a_val**5 / 120.0
    ) + _FALSE_EASTING

    y = _K0 * (
        m_val
        + n_val * tan_lat * (
            a_val**2 / 2.0
            + (5.0 - t_val + 9.0 * c_val + 4.0 * c_val**2) * a_val**4 / 24.0
            + (61.0 - 58.0 * t_val + t_val**2 + 600.0 * c_val - 330.0 * _E_PRIME2) * a_val**6 / 720.0
        )
    ) + _FALSE_NORTHING

    return x, y


def utm31n_to_wgs84(x_m: float, y_m: float) -> tuple[float, float]:
    """Convert ETRS89 / UTM Zone 31N coordinates (meters) to WGS84 geographic (degrees).

    Returns (lon_deg, lat_deg).
    """
    if not (math.isfinite(x_m) and math.isfinite(y_m)):
        raise ValueError("Coordinates must be finite numbers")

    x_adj = x_m - _FALSE_EASTING
    y_adj = y_m - _FALSE_NORTHING

    m_val = y_adj / _K0
    mu = m_val / (_A * (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - _E2)) / (1.0 + math.sqrt(1.0 - _E2))

    phi1 = mu + (
        (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    n1 = _A / math.sqrt(1.0 - _E2 * sin_phi1**2)
    t1 = tan_phi1**2
    c1 = _E_PRIME2 * cos_phi1**2
    r1 = _A * (1.0 - _E2) / ((1.0 - _E2 * sin_phi1**2)**1.5)
    d = x_adj / (n1 * _K0)

    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * _E_PRIME2) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * _E_PRIME2 - 3.0 * c1**2) * d**6 / 720.0
    )

    lon = _LON0_RAD + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * _E_PRIME2 + 24.0 * t1**2) * d**5 / 120.0
    ) / cos_phi1

    return math.degrees(lon), math.degrees(lat)


# ---------------------------------------------------------------------------
# Fuel Crosswalk: CORINE Land Cover -> Scott & Burgan 40 Fuel Models
# ---------------------------------------------------------------------------

CORINE_TO_SCOTT_BURGAN: dict[str, dict[str, Any]] = {
    "211": {
        "clc_code": "211",
        "clc_name": "Non-irrigated arable land",
        "sb_model": "GR1",
        "sb_code": 101,
        "description": "Short, sparse dry climate grass / cereal stubble",
        "confidence": "Medium",
        "rationale": "Winter cereals in Lleida/Noguera are ripe or harvested in mid-June; dry stubble carries fast low-intensity surface fire.",
        "alternatives": [
            {"model": "GR2", "code": 102, "rationale": "Dense post-harvest unploughed herbaceous load"},
            {"model": "NB", "code": 0, "rationale": "Clean ploughed fallow without stubble"},
        ],
    },
    "212": {
        "clc_code": "212",
        "clc_name": "Permanently irrigated land",
        "sb_model": "NB",
        "sb_code": 0,
        "description": "Nonburnable (active irrigation)",
        "confidence": "High",
        "rationale": "Irrigated valley crops maintain high moisture (>150%) and act as effective passive barriers during summer campaigns.",
        "alternatives": [
            {"model": "GR1", "code": 101, "rationale": "Dry unmanaged ditch banks"},
        ],
    },
    "221": {
        "clc_code": "221",
        "clc_name": "Vineyards",
        "sb_model": "NB",
        "sb_code": 0,
        "description": "Nonburnable (clean-tilled vineyard)",
        "confidence": "High",
        "rationale": "Traditional wide-spaced Mediterranean vineyards with bare tilled soil act as passive fuel breaks.",
        "alternatives": [
            {"model": "GR1", "code": 101, "rationale": "Unweeded vineyards with cured inter-row grass"},
        ],
    },
    "222": {
        "clc_code": "222",
        "clc_name": "Fruit trees and berry plantations",
        "sb_model": "NB",
        "sb_code": 0,
        "description": "Nonburnable (managed orchard)",
        "confidence": "High",
        "rationale": "Commercial fruit orchards in the Segre river basin are irrigated and weed-controlled.",
        "alternatives": [
            {"model": "TL1", "code": 181, "rationale": "Abandoned orchards with light dry litter"},
        ],
    },
    "242": {
        "clc_code": "242",
        "clc_name": "Complex cultivation patterns",
        "sb_model": "GR1",
        "sb_code": 101,
        "description": "Short, sparse grass (cultivation mosaic margins)",
        "confidence": "Medium",
        "rationale": "Fragmented smallholdings with dry field edges and cereal parcels.",
        "alternatives": [
            {"model": "NB", "code": 0, "rationale": "Intensively managed nonburnable plots"},
            {"model": "GS1", "code": 121, "rationale": "Plots intermixed with low scrub hedgerows"},
        ],
    },
    "243": {
        "clc_code": "243",
        "clc_name": "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "sb_model": "GS2",
        "sb_code": 122,
        "description": "Moderate load, dry climate grass-shrub mix",
        "confidence": "Medium",
        "rationale": "Marginal agroforestry interface: abandoned stone terraces with encroaching garrigue and cured grasses.",
        "alternatives": [
            {"model": "GS1", "code": 121, "rationale": "Low load grass-shrub interface"},
            {"model": "SH2", "code": 142, "rationale": "Predominantly shrub-encroached terraces"},
        ],
    },
    "311": {
        "clc_code": "311",
        "clc_name": "Broad-leaved forest",
        "sb_model": "TL6",
        "sb_code": 186,
        "description": "Moderate load broadleaf litter (Quercus ilex / gall oak)",
        "confidence": "Medium",
        "rationale": "Mediterranean sclerophyllous oak forests produce compact, slow-burning leaf litter with moderate depth.",
        "alternatives": [
            {"model": "TL2", "code": 182, "rationale": "Low load broadleaf litter in open stands"},
            {"model": "TU5", "code": 165, "rationale": "Oak forest with dense flammable understory"},
        ],
    },
    "312": {
        "clc_code": "312",
        "clc_name": "Coniferous forest",
        "sb_model": "TU5",
        "sb_code": 165,
        "description": "Very high load, dry climate timber-shrub (Pinus halepensis)",
        "confidence": "High",
        "rationale": "Aleppo pine stands in interior Catalonia are highly fire-prone with dense shrub understories acting as ladder fuels.",
        "alternatives": [
            {"model": "TL8", "code": 188, "rationale": "Long-needle pine litter under closed canopy without shrubs"},
            {"model": "TU1", "code": 161, "rationale": "Low load timber-grass in open woodlands"},
        ],
    },
    "313": {
        "clc_code": "313",
        "clc_name": "Mixed forest",
        "sb_model": "TU2",
        "sb_code": 162,
        "description": "Moderate load, dry climate timber-shrub / mixed litter",
        "confidence": "Medium",
        "rationale": "Transition forest between pine and deciduous oak with mixed needle/leaf litter and moderate shrub understory.",
        "alternatives": [
            {"model": "TL3", "code": 183, "rationale": "Moderate load mixed forest litter"},
            {"model": "TU5", "code": 165, "rationale": "Heavy understory mixed stand"},
        ],
    },
    "321": {
        "clc_code": "321",
        "clc_name": "Natural grassland",
        "sb_model": "GR2",
        "sb_code": 102,
        "description": "Low load, dry climate grass (cured summer pasture)",
        "confidence": "High",
        "rationale": "Natural montane and hill herbaceous pastures, completely cured in mid-June Mediterranean heatwaves.",
        "alternatives": [
            {"model": "GR1", "code": 101, "rationale": "Sparser or grazed grassland"},
            {"model": "GR4", "code": 104, "rationale": "Moderate load ungrazed tall grass"},
        ],
    },
    "323": {
        "clc_code": "323",
        "clc_name": "Sclerophyllous vegetation",
        "sb_model": "SH5",
        "sb_code": 145,
        "description": "High load, dry climate shrub (dense garrigue / maquis / coscojar)",
        "confidence": "High",
        "rationale": "Primary fuel type driving rapid rate of spread in Artesa de Segre: dense Kermes oak (Quercus coccifera), rosemary, and gorse 1-1.5m tall.",
        "alternatives": [
            {"model": "SH2", "code": 142, "rationale": "Moderate load / open garrigue"},
            {"model": "SH7", "code": 147, "rationale": "Very high load tall maquis"},
        ],
    },
    "324": {
        "clc_code": "324",
        "clc_name": "Transitional woodland shrub",
        "sb_model": "SH2",
        "sb_code": 142,
        "description": "Moderate load, dry climate shrub with scattered young trees",
        "confidence": "Medium",
        "rationale": "Regenerating scrub with scattered saplings following previous disturbances or pasture abandonment.",
        "alternatives": [
            {"model": "SH5", "code": 145, "rationale": "Dense mature transitional scrub"},
            {"model": "TU1", "code": 161, "rationale": "Woodland with open grassy understory"},
        ],
    },
    "998": {
        "clc_code": "998",
        "clc_name": "Other",
        "sb_model": "NB",
        "sb_code": 0,
        "description": "Nonburnable (rock, bare soil, quarries)",
        "confidence": "High",
        "rationale": "Unvegetated surfaces, rock outcrops, quarries, or unclassified non-fuel terrain.",
        "alternatives": [],
    },
}


def clc_to_scott_burgan(clc_code: str) -> int:
    """Return Scott & Burgan fuel model integer code for a 3-digit CORINE Land Cover code.

    Defaults to 0 (nonburnable) if code is unknown.
    """
    clean_code = str(clc_code).split("-")[0].strip()
    entry = CORINE_TO_SCOTT_BURGAN.get(clean_code)
    if entry is None:
        return 0
    return int(entry["sb_code"])


# ---------------------------------------------------------------------------
# Spatial Overlap Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_spatial_overlap(
    predicted_mask: Sequence[Sequence[bool]] | np.ndarray,
    observed_mask: Sequence[Sequence[bool]] | np.ndarray,
    cell_area_ha: float,
) -> dict[str, Any]:
    """Calculate spatial comparison metrics between a predicted and observed burn perimeter mask.

    Computes:
    - Jaccard Index / Intersection over Union (IoU)
    - Dice Similarity Coefficient
    - True positive, false positive (commission), false negative (omission), true negative counts
    - Commission rate: FP / (TP + FP)
    - Omission rate: FN / (TP + FN)
    - Predicted, observed, intersection, and union areas in hectares
    - Absolute and relative area differences
    """
    pred = np.asarray(predicted_mask, dtype=bool)
    obs = np.asarray(observed_mask, dtype=bool)

    if pred.shape != obs.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs observed {obs.shape}")
    if not (math.isfinite(cell_area_ha) and cell_area_ha > 0):
        raise ValueError("cell_area_ha must be a positive finite float")

    tp = int(np.logical_and(pred, obs).sum())
    fp = int(np.logical_and(pred, ~obs).sum())
    fn = int(np.logical_and(~pred, obs).sum())
    tn = int(np.logical_and(~pred, ~obs).sum())

    pred_cells = int(pred.sum())
    obs_cells = int(obs.sum())
    union_cells = tp + fp + fn

    iou = float(tp / union_cells) if union_cells > 0 else 1.0
    dice = float(2.0 * tp / (pred_cells + obs_cells)) if (pred_cells + obs_cells) > 0 else 1.0

    commission_rate = float(fp / pred_cells) if pred_cells > 0 else 0.0
    omission_rate = float(fn / obs_cells) if obs_cells > 0 else 0.0

    pred_ha = pred_cells * cell_area_ha
    obs_ha = obs_cells * cell_area_ha
    inter_ha = tp * cell_area_ha
    union_ha = union_cells * cell_area_ha
    area_error_ha = pred_ha - obs_ha
    area_error_pct = float((area_error_ha / obs_ha) * 100.0) if obs_ha > 0 else 0.0

    return {
        "iou": round(iou, 6),
        "dice": round(dice, 6),
        "commission_rate": round(commission_rate, 6),
        "omission_rate": round(omission_rate, 6),
        "predicted_area_ha": round(pred_ha, 4),
        "observed_area_ha": round(obs_ha, 4),
        "intersection_area_ha": round(inter_ha, 4),
        "union_area_ha": round(union_ha, 4),
        "area_error_ha": round(area_error_ha, 4),
        "area_error_pct": round(area_error_pct, 4),
        "counts": {
            "true_positive_cells": tp,
            "false_positive_cells": fp,
            "false_negative_cells": fn,
            "true_negative_cells": tn,
            "total_cells": pred.size,
        },
        "units": {
            "area": "ha",
            "cell_area": "ha",
        },
    }


# ---------------------------------------------------------------------------
# Polygon Rasterization Helpers
# ---------------------------------------------------------------------------

def _rasterize_polygons(
    features: list[dict],
    grid_points: np.ndarray,
    target_grid: np.ndarray,
    value_lookup: Any,
) -> None:
    """Rasterize a list of GeoJSON features onto target_grid using bounding box filtered MplPath.

    value_lookup can be a constant value or a callable taking feature['properties'].
    """
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
            outer = [wgs84_to_utm31n(lon, lat) for lon, lat in poly[0]]
            path = MplPath(outer)
            bb = path.get_extents()

            sub_idx = np.where(
                (grid_points[:, 0] >= bb.x0) & (grid_points[:, 0] <= bb.x1) &
                (grid_points[:, 1] >= bb.y0) & (grid_points[:, 1] <= bb.y1)
            )[0]
            if len(sub_idx) == 0:
                continue

            in_poly = path.contains_points(grid_points[sub_idx])

            # Handle interior holes
            for hole in poly[1:]:
                if not hole:
                    continue
                hole_path = MplPath([wgs84_to_utm31n(lon, lat) for lon, lat in hole])
                in_hole = hole_path.contains_points(grid_points[sub_idx])
                in_poly = in_poly & ~in_hole

            target_grid[sub_idx[in_poly]] = val


# ---------------------------------------------------------------------------
# Scenario Generation from Copernicus EMSR578 Vectors
# ---------------------------------------------------------------------------

def build_emsr578_scenario(
    directory: Path,
    cell_size_m: float = 100.0,
) -> dict[str, Any]:
    """Build a georeferenced raster scenario from Copernicus EMSR578 AOI01 del.zip package.

    Constructs:
    - Metric grid in ETRS89 / UTM Zone 31N (EPSG:25831)
    - Scott & Burgan fuel model grid from naturalLandUseA and hydrographyA
    - Initial burn perimeter mask from DEL observedEventA (2022-06-17 10:06 UTC)
    - Aircraft water scooping locations from hydrographyA (Rialb reservoir, Segre river)
    """
    if cell_size_m <= 0 or not math.isfinite(cell_size_m):
        raise ValueError("cell_size_m must be a positive finite number")

    del_zip = directory / "del.zip"
    if not del_zip.exists():
        raise FileNotFoundError(f"Missing {del_zip}; run 'python3 -m nightjar.emsr578 fetch' first")

    with ZipFile(del_zip) as archive:
        aoi_name = [n for n in archive.namelist() if n.endswith("areaOfInterestA_r1_v1.json")][0]
        landuse_name = [n for n in archive.namelist() if n.endswith("naturalLandUseA_r1_v1.json")][0]
        hydro_name = [n for n in archive.namelist() if n.endswith("hydrographyA_r1_v1.json")][0]
        obs_name = [n for n in archive.namelist() if n.endswith("observedEventA_r1_v1.json")][0]

        aoi_data = json.loads(archive.read(aoi_name))
        landuse_data = json.loads(archive.read(landuse_name))
        hydro_data = json.loads(archive.read(hydro_name))
        obs_data = json.loads(archive.read(obs_name))

    # Compute metric bounding box from AOI in UTM 31N
    aoi_feature = aoi_data["features"][0]
    aoi_coords = aoi_feature["geometry"]["coordinates"][0]
    aoi_utm = [wgs84_to_utm31n(lon, lat) for lon, lat in aoi_coords]

    min_x = min(p[0] for p in aoi_utm)
    max_x = max(p[0] for p in aoi_utm)
    min_y = min(p[1] for p in aoi_utm)
    max_y = max(p[1] for p in aoi_utm)

    cols = int(math.ceil((max_x - min_x) / cell_size_m))
    rows = int(math.ceil((max_y - min_y) / cell_size_m))

    # Grid cell center coordinates (rows increase northward, cols eastward)
    xs = min_x + (np.arange(cols) + 0.5) * cell_size_m
    ys = min_y + (np.arange(rows) + 0.5) * cell_size_m
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # 1. Rasterize naturalLandUseA into Scott & Burgan fuel model codes
    fuel_flat = np.zeros(len(grid_points), dtype=np.int32)

    def _lookup_clc_fuel(props: dict) -> int:
        info = props.get("info", "")
        code = info.split("-")[0].strip() if info else ""
        return clc_to_scott_burgan(code)

    _rasterize_polygons(landuse_data["features"], grid_points, fuel_flat, _lookup_clc_fuel)

    # 2. Rasterize hydrographyA as nonburnable (code 0) and record water mask
    water_flat = np.zeros(len(grid_points), dtype=bool)
    _rasterize_polygons(hydro_data["features"], grid_points, fuel_flat, 0)
    _rasterize_polygons(hydro_data["features"], grid_points, water_flat, True)

    # 3. Rasterize observedEventA (initial burn mask at t0)
    burn_flat = np.zeros(len(grid_points), dtype=bool)
    _rasterize_polygons(obs_data["features"], grid_points, burn_flat, True)

    # Reshape to 2D (rows, cols)
    fuel_grid = fuel_flat.reshape((rows, cols))
    water_mask = water_flat.reshape((rows, cols))
    burn_mask = burn_flat.reshape((rows, cols))

    # Compute summary statistics
    cell_area_ha = (cell_size_m**2) / 10000.0
    unique_fuels, fuel_counts = np.unique(fuel_grid, return_counts=True)
    fuel_area_summary = {
        int(k): round(float(c) * cell_area_ha, 2)
        for k, c in zip(unique_fuels, fuel_counts)
    }

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_lon, center_lat = utm31n_to_wgs84(center_x, center_y)

    # Extract water scooping sources from hydrography
    water_sources = []
    for hf in hydro_data["features"]:
        props = hf["properties"]
        geom = hf["geometry"]
        gtype = geom["type"]
        coords = geom["coordinates"]
        polys = [coords] if gtype == "Polygon" else coords
        # Collect vertices for centroid calculation
        all_pts = []
        for poly in polys:
            for ring in poly:
                all_pts.extend([wgs84_to_utm31n(lon, lat) for lon, lat in ring])
        if not all_pts:
            continue
        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)
        clon, clat = utm31n_to_wgs84(cx, cy)
        obj_type = props.get("obj_type", "Unknown")
        name = "Pantà de Rialb" if "BH130" in obj_type else ("Riu Segre" if "BH080" in obj_type else obj_type)
        water_sources.append({
            "name": name,
            "type": obj_type,
            "center_utm31n": [round(cx, 1), round(cy, 1)],
            "center_wgs84": [round(clon, 6), round(clat, 6)],
        })

    reported_area_ha = float(obs_data["features"][0]["properties"].get("area", 0.0))
    raster_burned_ha = round(float(burn_mask.sum()) * cell_area_ha, 2)

    return {
        "schema_version": "nightjar-emsr578-pilot-0.1",
        "scenario_id": "emsr578-artesa-de-segre-aoi01",
        "title": "Artesa de Segre wildfire pilot scenario (Copernicus EMSR578 AOI01)",
        "source": "Copernicus Emergency Management Service On-Demand Mapping EMSR578 AOI01",
        "spatial_reference": {
            "crs": "EPSG:25831",
            "datum": "ETRS89 / UTM Zone 31N",
            "bounds_utm31n": {
                "min_x": round(min_x, 1),
                "max_x": round(max_x, 1),
                "min_y": round(min_y, 1),
                "max_y": round(max_y, 1),
            },
            "center_wgs84": {
                "lon": round(center_lon, 6),
                "lat": round(center_lat, 6),
            },
        },
        "grid": {
            "rows": rows,
            "cols": cols,
            "cell_size_m": cell_size_m,
            "cell_area_ha": cell_area_ha,
            "total_area_ha": round(rows * cols * cell_area_ha, 2),
        },
        "observation_timestamps": {
            "t0_initial_utc": "2022-06-17T10:06:00Z",
            "t1_monitoring_utc": "2022-06-18T10:47:00Z",
            "t2_monitoring_utc": "2022-06-19T10:42:00Z",
        },
        "initial_fire": {
            "source_product": "EMSR578_AOI01_DEL_PRODUCT_observedEventA_r1_v1",
            "reported_burned_area_ha": reported_area_ha,
            "raster_burned_cells": int(burn_mask.sum()),
            "raster_burned_area_ha": raster_burned_ha,
            "discretization_error_ha": round(raster_burned_ha - reported_area_ha, 2),
            "discretization_error_pct": round(((raster_burned_ha - reported_area_ha) / reported_area_ha) * 100.0, 2) if reported_area_ha > 0 else 0.0,
        },
        "fuel_distribution_ha": fuel_area_summary,
        "water_sources": water_sources,
        "fuel_model": fuel_grid.tolist(),
        "water_mask": water_mask.tolist(),
        "initial_burn_mask": burn_mask.tolist(),
        "crosswalk_table": list(CORINE_TO_SCOTT_BURGAN.values()),
        "limitations": [
            "Fuel models are inferred from CORINE Land Cover (2018) via documented ecological crosswalk, not direct in-situ biomass measurements.",
            "Moisture fractions and localized wind vectors require atmospheric reconstruction (ERA5/Meteocat) before physical spread validation.",
            "Suppression actions by ground and aerial crews (Bombers de la Generalitat / MITECO) during 17-19 June 2022 are not yet represented.",
            "Perimeters represent cumulative photo-interpreted burned areas, not instantaneous flaming fronts.",
        ],
    }


def extract_observed_mask(
    zip_path: Path,
    bounds_utm31n: dict[str, float],
    rows: int,
    cols: int,
    cell_size_m: float,
) -> np.ndarray:
    """Extract and rasterize the observedEventA burned area mask from a Copernicus vector ZIP."""
    with ZipFile(zip_path) as archive:
        matches = [m for m in archive.namelist() if m.endswith("observedEventA_r1_v1.json")]
        if not matches:
            raise ValueError(f"No observedEventA found in {zip_path}")
        obs_data = json.loads(archive.read(matches[0]))

    min_x = bounds_utm31n["min_x"]
    min_y = bounds_utm31n["min_y"]
    xs = min_x + (np.arange(cols) + 0.5) * cell_size_m
    ys = min_y + (np.arange(rows) + 0.5) * cell_size_m
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    mask_flat = np.zeros(len(grid_points), dtype=bool)
    _rasterize_polygons(obs_data["features"], grid_points, mask_flat, True)
    return mask_flat.reshape((rows, cols))


# ---------------------------------------------------------------------------
# CLI Command Interface
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: crosswalk
    parser_cw = subparsers.add_parser("crosswalk", help="Export CORINE to Scott & Burgan fuel crosswalk")
    parser_cw.add_argument("--format", choices=("json", "markdown"), default="json")
    parser_cw.add_argument("--output", type=Path)

    # Subcommand: manifest
    parser_mf = subparsers.add_parser("manifest", help="Build EMSR578 pilot scenario manifest")
    parser_mf.add_argument("--directory", type=Path, default=Path("data/raw/emsr578"))
    parser_mf.add_argument("--cell-size", type=float, default=100.0, help="Grid cell size in meters (default 100m)")
    parser_mf.add_argument("--output", type=Path)

    # Subcommand: compare-perimeters
    parser_cp = subparsers.add_parser("compare-perimeters", help="Compare two Copernicus observed perimeters")
    parser_cp.add_argument("--scenario", type=Path, required=True, help="Path to scenario JSON generated by manifest")
    parser_cp.add_argument("--comparison-zip", type=Path, default=Path("data/raw/emsr578/monit01.zip"))
    parser_cp.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "crosswalk":
        if args.format == "json":
            content = json.dumps(list(CORINE_TO_SCOTT_BURGAN.values()), indent=2) + "\n"
        else:
            lines = [
                "| CLC Code | CLC Land Cover | Scott & Burgan Model | Code | Confidence | Botanical Rationale |",
                "|:---:|:---|:---:|:---:|:---:|:---|",
            ]
            for row in CORINE_TO_SCOTT_BURGAN.values():
                lines.append(
                    f"| `{row['clc_code']}` | {row['clc_name']} | **{row['sb_model']}** | `{row['sb_code']}` | {row['confidence']} | {row['rationale']} |"
                )
            content = "\n".join(lines) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
        else:
            print(content, end="")

    elif args.command == "manifest":
        manifest = build_emsr578_scenario(args.directory, cell_size_m=args.cell_size)
        content = json.dumps(manifest, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
        else:
            print(f"Generated scenario manifest: {manifest['grid']['rows']}x{manifest['grid']['cols']} cells, {manifest['grid']['total_area_ha']} ha")

    elif args.command == "compare-perimeters":
        scenario = json.loads(args.scenario.read_text())
        pred_mask = np.array(scenario["initial_burn_mask"], dtype=bool)
        bounds = scenario["spatial_reference"]["bounds_utm31n"]
        rows = scenario["grid"]["rows"]
        cols = scenario["grid"]["cols"]
        cell_size = scenario["grid"]["cell_size_m"]
        cell_area_ha = scenario["grid"]["cell_area_ha"]

        obs_mask = extract_observed_mask(args.comparison_zip, bounds, rows, cols, cell_size)
        metrics = evaluate_spatial_overlap(pred_mask, obs_mask, cell_area_ha)
        metrics["source_predicted"] = scenario["initial_fire"]["source_product"]
        metrics["source_observed"] = str(args.comparison_zip)

        content = json.dumps(metrics, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
        else:
            print(content, end="")


if __name__ == "__main__":
    main()
