"""Fetch and audit three CEMS EMSR578 AOI01 burned-area snapshots.

The snapshots are independent observations for a future validation protocol.
This module does not compare them to a Pandion simulation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


BASE = "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR578/"
PRODUCTS = (
    ("del", "PRODUCT", "2022-06-17T10:06:00Z", "4ad2197f1a7ccd4b8da0e63a71f2dbc6aee53d58688c448f25ed9cdf0c01f026"),
    ("monit01", "MONIT01", "2022-06-18T10:47:00Z", "41f4fa2dac9e76d3034e244f9d24f3ae788b05c2557ed2b19a9aff7c8b6e6a9e"),
    ("monit02", "MONIT02", "2022-06-19T10:42:00Z", "92b11953975914332507e49f2ca6bb2192be07bcfbcb0e471c37152e68d831ac"),
)


def product_url(stage: str) -> str:
    return BASE + f"EMSR578_AOI01_DEL_{stage}_r1_RTP01_v1_vector.zip"


def map_pdf_url(stage: str) -> str:
    return BASE + f"EMSR578_AOI01_DEL_{stage}_r1_RTP01_v1.pdf"


def fetch(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, stage, _, expected_hash in PRODUCTS:
        target = directory / f"{name}.zip"
        if target.exists() and sha256(target.read_bytes()).hexdigest() == expected_hash:
            continue
        with urlopen(product_url(stage), timeout=60) as response:
            content = response.read()
        if sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"{name}: download hash differs from the reviewed product; inspect the source before use")
        target.write_bytes(content)


def report(directory: Path) -> dict:
    observations = []
    for name, stage, acquired_at, expected_hash in PRODUCTS:
        target = directory / f"{name}.zip"
        content = target.read_bytes()
        digest = sha256(content).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"{target}: SHA-256 mismatch")
        with ZipFile(target) as archive:
            matches = [item for item in archive.namelist() if item.endswith("observedEventA_r1_v1.json")]
            if len(matches) != 1:
                raise ValueError(f"{target}: expected exactly one observedEventA GeoJSON")
            geojson = json.loads(archive.read(matches[0]))
        if geojson.get("type") != "FeatureCollection" or len(geojson.get("features", [])) != 1:
            raise ValueError(f"{target}: unexpected feature collection")
        feature = geojson["features"][0]
        properties = feature["properties"]
        if properties.get("notation") != "Burnt area" or feature["geometry"].get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"{target}: expected a burnt-area polygon")
        area = float(properties["area"])
        if area <= 0:
            raise ValueError(f"{target}: area must be positive")
        observations.append({
            "product": stage, "image_acquired_at_utc": acquired_at,
            "area_ha": area, "download_url": product_url(stage),
            "acquisition_time_source_pdf": map_pdf_url(stage),
            "source_zip_sha256": digest, "polygon_member": matches[0],
        })
    for previous, current in zip(observations, observations[1:]):
        start = datetime.fromisoformat(previous["image_acquired_at_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(current["image_acquired_at_utc"].replace("Z", "+00:00"))
        current["elapsed_h_since_previous"] = round((end - start).total_seconds() / 3600, 6)
        current["area_change_ha_since_previous"] = round(current["area_ha"] - previous["area_ha"], 6)
    return {
        "schema_version": "cems-emsr578-aoi01-0.1",
        "source": "Copernicus Emergency Management Service On-Demand Mapping, EMSR578 AOI01 Artesa de Segre",
        "source_page": "https://mapping.emergency.copernicus.eu/activations/EMSR578/",
        "units": {"area": "ha", "time": "UTC"},
        "observations": observations,
        "limitations": [
            "Cumulative photo-interpreted burned area, not an active-fire perimeter at every time step.",
            "Vegetation is not established as FM1 short grass; suppression and weather history are not represented here.",
            "This audit is not a physical validation of Pandion fire spread.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "report"))
    parser.add_argument("--directory", type=Path, default=Path("data/raw/emsr578"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.directory)
        return
    rendered = json.dumps(report(args.directory), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
