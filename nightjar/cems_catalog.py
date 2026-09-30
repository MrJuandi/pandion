"""Copernicus Emergency Management Service (CEMS) multi-incident catalog and scenario builder.

This module provides unified ingestion for multiple Copernicus wildfire activations:
- EMSR578: Artesa de Segre (Lleida, Catalonia / UTM Zone 31N)
- EMSR590: Casas de Miravete / Monfragüe (Cáceres, Extremadura / UTM Zone 30N)
- EMSR592: Sierra de Mijas (Málaga, Andalusia / UTM Zone 30N)
- EMSR600: Vall d'Ebo (Alicante, Valencian Community / UTM Zone 30N)

All activations are georeferenced to standard ETRS89 / UTM (Zones 29N, 30N, 31N)
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
        "name": "Casas de Miravete / Monfragüe",
        "province": "Cáceres",
        "region": "Extremadura",
        "country": "Spain",
        "utm_zone": 30,
        "epsg": "EPSG:25830",
        "default_split": "validation",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR590/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-07-14T18:06:06Z",
                "sha256": "46238c94a62a32144b5d9cb721ecb941bb8b5fa7e24bc7c4f2d8e56575871645",
                "reported_area_ha": 4637.40,
                "vector_filename": "EMSR590_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
        },
    },
    "EMSR592": {
        "activation_id": "EMSR592",
        "name": "Sierra de Mijas",
        "province": "Málaga",
        "region": "Andalusia",
        "country": "Spain",
        "utm_zone": 30,
        "epsg": "EPSG:25830",
        "default_split": "reserved",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR592/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-07-15T12:00:00Z",
                "sha256": "8891a611dee603a431b67e82ab86dc6fcd53432e0c3342f359a269ccb48537a2",
                "reported_area_ha": 41.68,
                "vector_filename": "EMSR592_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
        },
    },
    "EMSR600": {
        "activation_id": "EMSR600",
        "name": "Vall d'Ebo",
        "province": "Alicante",
        "region": "Valencian Community",
        "country": "Spain",
        "utm_zone": 30,
        "epsg": "EPSG:25830",
        "default_split": "reserved",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR600/",
        "products": {
            "del": {
                "stage": "del",
                "product_code": "PRODUCT",
                "acquired_at_utc": "2022-08-14T12:00:00Z",
                "sha256": "e64eb4fb7f2465888a02615d2adf38278726d7fba413c4e470dfb278728ac2a0",
                "reported_area_ha": 836.73,
                "vector_filename": "EMSR600_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
            },
        },
    },
}


def cems_product_url(activation_id: str, stage: str = "del") -> str:
    """Return the official download URL for a CEMS vector package."""
    activation = ACTIVATIONS.get(activation_id.upper())
    if not activation:
        raise ValueError(f"Unknown activation: {activation_id}")
    prod = activation["products"].get(stage)
    if not prod:
        raise ValueError(f"Unknown stage '{stage}' for activation {activation_id}")
    return BASE_URL_TEMPLATE.format(activation=activation_id.upper()) + prod["vector_filename"]


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


def _extract_water_waypoints(features: list[dict], utm_zone: int) -> list[dict[str, Any]]:
    waypoints = []
    for hf in features:
        props = hf.get("properties", {})
        geom = hf.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        all_pts = []
        if gtype == "Point":
            all_pts.append((float(coords[0]), float(coords[1])))
        elif gtype in ("Polygon", "MultiPolygon"):
            polys = [coords] if gtype == "Polygon" else coords
            for poly in polys:
                for ring in poly:
                    for pt in ring:
                        all_pts.append((float(pt[0]), float(pt[1])))
        if not all_pts:
            continue
        clon = sum(p[0] for p in all_pts) / len(all_pts)
        clat = sum(p[1] for p in all_pts) / len(all_pts)
        ux, uy = wgs84_to_utm(clon, clat, zone=utm_zone)
        obj_type = props.get("obj_type", "Waterbody")
        name = props.get("name", obj_type)
        if name == "Unknown":
            name = obj_type
        waypoints.append({
            "name": name,
            "obj_type": obj_type,
            "lon_deg": round(clon, 6),
            "lat_deg": round(clat, 6),
            "utm_easting_m": round(ux, 2),
            "utm_northing_m": round(uy, 2),
            "eligible_for_scooping": True,
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
