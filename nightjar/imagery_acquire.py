"""Acquire pre-fire Sentinel-2 true-colour imagery aligned to geographic scenario grids.

Each activation's grid receives one cloud-free Sentinel-2 L2A true-colour
(TCI, "visual" asset) crop sampled at ``px_per_cell`` pixels per cell. Scenes
must be acquired at least ``margin_days`` before the CEMS activation time, so
the base image precedes the fire; dynamic fire state is drawn over it later.

Scenes are found through the public Earth Search STAC API (Element84, AWS open
data) and screened by the L2A scene classification (SCL) inside the crop, not
by tile-wide cloud cover. Rasterio is optional until acquisition is requested.

Stored arrays use the scenario grid convention: row 0 is the southern edge and
rows increase north. ``load_activation_image`` returns that orientation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from .geodesy import utm_to_wgs84

SCHEMA_VERSION = "nightjar-imagery-sources-0.1"
STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
CEMS_ACTIVATION_URL = "https://mapping.emergency.copernicus.eu/activations/api/activations/{code}/"
COLLECTION = "sentinel-2-l2a"
DEFAULT_PX_PER_CELL = 8
DEFAULT_MARGIN_DAYS = 14
DEFAULT_WINDOW_DAYS = 75
MAX_SCENES_COMPOSITED = 10
# SCL classes that hide the surface: no data, saturated/defective, cloud
# shadow, cloud medium/high probability, thin cirrus, snow/ice. Unclassified
# (7) is kept: it persists on bright bare ground and quarries in every scene.
OBSCURED_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)
# SCL misses the diffuse halo around clouds and shadows; buffer the mask.
CLOUD_BUFFER_M = 100.0


def _utc(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cems_code(activation_id: str) -> str:
    """CEMS activation code without the AOI suffix (``EMSR602_AOI02`` -> ``EMSR602``)."""
    return activation_id.split("_", 1)[0]


def search_window(activation_time_utc: str, margin_days: int, window_days: int) -> tuple[datetime, datetime]:
    """Latest acceptable scene time and the start of the search window."""
    if margin_days < 0 or window_days <= 0:
        raise ValueError("margin_days must be nonnegative and window_days positive")
    end = _utc(activation_time_utc) - timedelta(days=margin_days)
    return end - timedelta(days=window_days), end


def target_grid(rows: int, cols: int, cell_size_m: float, x0: float, y0: float, px_per_cell: int) -> dict:
    """North-up raster grid covering the scenario cells at px_per_cell resolution."""
    if px_per_cell < 1:
        raise ValueError("px_per_cell must be positive")
    pixel = cell_size_m / px_per_cell
    return {
        "width": cols * px_per_cell,
        "height": rows * px_per_cell,
        "pixel_size_m": pixel,
        # GDAL affine (a, b, c, d, e, f): north-up, top-left corner origin.
        "transform": [pixel, 0.0, x0, 0.0, -pixel, y0 + rows * cell_size_m],
    }


def family_grid(scenarios: list[dict]) -> tuple[int, int, float, float, float, str]:
    """Shared (rows, cols, cell_size, x0, y0, crs) of one activation's variants."""
    def key(s: dict) -> tuple:
        g = s["grid"]
        return (int(g["rows"]), int(g["cols"]), float(g["cell_size_m"]),
                float(g["origin_utm_easting_m"]), float(g["origin_utm_northing_m"]), s["crs"]["code"])
    reference = key(scenarios[0])
    for scenario in scenarios[1:]:
        other = key(scenario)
        if other[:2] != reference[:2] or other[5] != reference[5] or any(
                abs(other[i] - reference[i]) > 1e-3 for i in (2, 3, 4)):
            raise ValueError(f"{scenario['activation_id']} variants disagree on grid; one image cannot serve all")
    return reference


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Square binary dilation by radius_px pixels (numpy only)."""
    if radius_px <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius_px)
    out = np.zeros_like(mask)
    for dy in range(2 * radius_px + 1):
        for dx in range(2 * radius_px + 1):
            out |= padded[dy:dy + height, dx:dx + width]
    return out


def obscured_mask(scl: np.ndarray, buffer_px: int = 0) -> np.ndarray:
    """Pixels whose surface is hidden or invalid, buffered by buffer_px."""
    if scl.size == 0:
        raise ValueError("empty SCL crop")
    return dilate(np.isin(scl, OBSCURED_SCL_CLASSES), buffer_px)


def obscured_fraction(scl: np.ndarray, buffer_px: int = 0) -> float:
    """Fraction of SCL pixels whose surface is hidden, invalid or within the buffer."""
    return float(obscured_mask(scl, buffer_px).mean())


def buffer_pixels(pixel_size_m: float) -> int:
    return int(np.ceil(CLOUD_BUFFER_M / pixel_size_m))


def north_up_to_grid(image: np.ndarray) -> np.ndarray:
    """Flip a north-up [H, W, 3] image so row 0 is south, matching scenario grids."""
    return np.ascontiguousarray(image[::-1])


def _http_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    with urlopen(Request(url, data=data, headers=headers), timeout=60) as response:
        return json.load(response)


def _bbox_wgs84(x0: float, y0: float, width_m: float, height_m: float, zone: int) -> list[float]:
    corners = [utm_to_wgs84(x, y, zone=zone) for x in (x0, x0 + width_m) for y in (y0, y0 + height_m)]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    return [min(lons), min(lats), max(lons), max(lats)]


def _read_aligned(href: str, crs: str, grid: dict, resampling_name: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    with rasterio.Env(GDAL_HTTP_TIMEOUT="60", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(href) as ds:
            with WarpedVRT(ds, crs=crs, transform=Affine(*grid["transform"]), width=grid["width"],
                           height=grid["height"], resampling=getattr(Resampling, resampling_name),
                           nodata=0) as vrt:
                return vrt.read()


def rank_scenes(items: list[dict], scl_reader, buffer_px: int = 0) -> list[tuple[dict, float]]:
    """Rank scenes by obscured fraction inside the crop (then newest first).

    Tile-wide eo:cloud_cover is a poor proxy for a 4 km crop, so every scene in
    the window is screened with its own SCL layer.
    """
    if not items:
        raise ValueError("no Sentinel-2 scenes in the search window")
    scored = [(item, obscured_fraction(scl_reader(item), buffer_px)) for item in items]
    return sorted(scored, key=lambda pair: (pair[1], -_utc(pair[0]["properties"]["datetime"]).timestamp()))


def match_to_reference(image: np.ndarray, clear: np.ndarray, reference: np.ndarray,
                       reference_clear: np.ndarray, min_pixels: int = 500) -> np.ndarray:
    """Per-channel mean/std match of image to reference over pixels clear in both.

    Reduces date-to-date illumination and haze differences at composite seams.
    Returns the image unchanged when the shared clear area is too small.
    """
    shared = clear & reference_clear
    if shared.sum() < min_pixels:
        return image
    out = image.astype(np.float32)
    for band in range(3):
        src, ref = out[..., band][shared], reference[..., band][shared].astype(np.float32)
        scale = ref.std() / src.std() if src.std() > 1e-6 else 1.0
        out[..., band] = (out[..., band] - src.mean()) * scale + ref.mean()
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def best_available_pixel(layers: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[float], float]:
    """Fill a composite from (image [H, W, 3], clear mask [H, W]) layers in priority order.

    Each pixel takes the first layer in which it is clear. Returns the composite,
    the fraction of pixels taken from each layer, and the fraction left unfilled.
    """
    if not layers:
        raise ValueError("no layers to composite")
    height, width = layers[0][1].shape
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    filled = np.zeros((height, width), dtype=bool)
    contributions = []
    for image, clear in layers:
        if image.shape != (height, width, 3) or clear.shape != (height, width):
            raise ValueError("composite layers must share one grid")
        take = clear & ~filled
        composite[take] = image[take]
        filled |= take
        contributions.append(float(take.mean()))
    return composite, contributions, float((~filled).mean())


def acquire_imagery(scenarios_dir: Path, output_dir: Path, *, px_per_cell: int = DEFAULT_PX_PER_CELL,
                    margin_days: int = DEFAULT_MARGIN_DAYS, window_days: int = DEFAULT_WINDOW_DAYS,
                    ) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("imagery output must be new or empty")
    # Only grid/CRS metadata is used, so any family layout with split folders works.
    scenarios = [json.loads(path.read_text())
                 for split in ("development", "validation", "reserved")
                 for path in sorted((scenarios_dir / split).glob("*.json"))]
    if not scenarios:
        raise ValueError(f"no scenario JSON files under {scenarios_dir}")
    groups = {a: [s for s in scenarios if s["activation_id"] == a]
              for a in sorted({s["activation_id"] for s in scenarios})}
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    for activation, group in groups.items():
        rows, cols, size, x0, y0, crs = family_grid(group)
        zone = int(group[0]["crs"]["utm_zone"])
        grid = target_grid(rows, cols, size, x0, y0, px_per_cell)
        cems = _http_json(CEMS_ACTIVATION_URL.format(code=cems_code(activation)))
        start, end = search_window(cems["activationTime"], margin_days, window_days)
        bbox = _bbox_wgs84(x0, y0, cols * size, rows * size, zone)
        found = _http_json(STAC_SEARCH_URL, {
            "collections": [COLLECTION], "bbox": bbox, "limit": 200,
            "datetime": f"{_iso(start)}/{_iso(end)}",
        })
        items = [it for it in found.get("features", []) if "visual" in it["assets"] and "scl" in it["assets"]]
        scl_grid = target_grid(rows, cols, size, x0, y0, 1)
        ranked = rank_scenes(
            items, lambda it: _read_aligned(it["assets"]["scl"]["href"], crs, scl_grid, "nearest")[0],
            buffer_pixels(scl_grid["pixel_size_m"]))
        layers, used = [], []
        for item, fraction in ranked[:MAX_SCENES_COMPOSITED]:
            visual = np.moveaxis(_read_aligned(item["assets"]["visual"]["href"], crs, grid, "bilinear"), 0, -1)
            scl = _read_aligned(item["assets"]["scl"]["href"], crs, grid, "nearest")[0]
            # Bilinear TCI is zero where the scene has no data.
            clear = ~obscured_mask(scl, buffer_pixels(grid["pixel_size_m"])) & (visual.max(axis=-1) > 0)
            visual = visual.astype(np.uint8)
            if layers:
                visual = match_to_reference(visual, clear, layers[0][0], layers[0][1])
            layers.append((visual, clear))
            used.append((item, fraction))
            if best_available_pixel(layers)[2] == 0.0:
                break
        image, contributions, unfilled = best_available_pixel(layers)
        if unfilled > 0.0:
            raise ValueError(f"{activation}: {unfilled:.2%} of the crop is obscured in every composited scene")
        primary = used[0][0]
        array_path = output_dir / f"{activation.lower()}.npy"
        np.save(array_path, north_up_to_grid(image.astype(np.uint8)))
        try:
            from PIL import Image
            Image.fromarray(image.astype(np.uint8)).save(output_dir / f"{activation.lower()}_preview.png")
        except ImportError:
            pass
        sources[activation] = {
            "provider": "Copernicus Sentinel-2 via Earth Search (Element84, AWS Open Data)",
            "collection": COLLECTION, "asset": "visual (TCI, 8-bit true colour, 10 m native)",
            "primary_item_id": primary["id"], "primary_datetime_utc": primary["properties"]["datetime"],
            "compositing": ("best-available pixel: each pixel from the least-obscured scene where its SCL class is clear; "
                            "filler scenes are per-channel mean/std matched to the primary over shared clear pixels"),
            "scenes_used": [{"id": it["id"], "datetime_utc": it["properties"]["datetime"],
                             "tile_cloud_cover_pct": it["properties"].get("eo:cloud_cover"),
                             "crop_obscured_fraction": round(fr, 6), "pixel_fraction": round(share, 6)}
                            for (it, fr), share in zip(used, contributions)],
            "candidates_screened": [{"id": it["id"], "datetime_utc": it["properties"]["datetime"],
                                     "crop_obscured_fraction": round(fr, 6)} for it, fr in ranked],
            "obscured_scl_classes": list(OBSCURED_SCL_CLASSES), "cloud_buffer_m": CLOUD_BUFFER_M,
            "cems_activation_time_utc": cems["activationTime"],
            "search_window_utc": [_iso(start), _iso(end)], "margin_days": margin_days,
            "crs": crs, "grid": {"rows": rows, "cols": cols, "cell_size_m": size,
                                 "origin_utm_easting_m": x0, "origin_utm_northing_m": y0},
            "px_per_cell": px_per_cell, "pixel_size_m": grid["pixel_size_m"], "transform": grid["transform"],
            "resampling": "bilinear (true colour), nearest (SCL screening)",
            "array": array_path.name, "array_shape": list(image.shape), "array_dtype": "uint8",
            "array_row_order": "row 0 = southern edge (scenario grid convention); preview PNG is north-up",
            "array_sha256": sha256(array_path.read_bytes()).hexdigest(),
            "license": "Copernicus Sentinel data terms (free, full and open)",
            "attribution": f"Contains modified Copernicus Sentinel data {primary['properties']['datetime'][:4]}",
            "retrieved_at_utc": _iso(datetime.now(timezone.utc)),
        }
        manifest = {"schema_version": SCHEMA_VERSION, "seed": None,
                    "units": {"coordinates": "m", "pixel_size": "m", "time": "UTC"},
                    "created_at_utc": _iso(datetime.now(timezone.utc)), "activations": sources}
        (output_dir / "sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"{activation}: {len(used)} scene(s), primary {primary['id']} "
              f"({contributions[0]:.1%} of pixels) -> {array_path.name}", flush=True)
    return sources


def downsample_imagery(source_dir: Path | str, output_dir: Path | str, factor: int = 2) -> dict:
    """Write a copy of an imagery directory with ``factor`` x ``factor`` mean-pooled pixels.

    Keeps the grid, CRS and provenance; updates pixels per cell, pixel size,
    shape and hash, and records the derivation. Used for 4 px per cell (D-64).
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("downsampled imagery output must be new or empty")
    manifest = json.loads((source_dir / "sources.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported imagery manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    for activation, source in manifest["activations"].items():
        if source["px_per_cell"] % factor:
            raise ValueError(f"{activation}: {source['px_per_cell']} px per cell is not divisible by {factor}")
        image = np.load(source_dir / source["array"]).astype(np.float32)
        h, w, _ = image.shape
        small = np.rint(image.reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3))).astype(np.uint8)
        np.save(output_dir / source["array"], small)
        source.update(px_per_cell=source["px_per_cell"] // factor, pixel_size_m=source["pixel_size_m"] * factor,
                      array_shape=list(small.shape),
                      array_sha256=sha256((output_dir / source["array"]).read_bytes()).hexdigest(),
                      derived_from={"imagery_dir": str(source_dir), "pooling": f"{factor}x{factor} mean",
                                    "source_sha256": source["array_sha256"]})
    manifest["created_at_utc"] = _iso(datetime.now(timezone.utc))
    (output_dir / "sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_activation_image(imagery_dir: Path | str, scenario: dict) -> np.ndarray:
    """Return the [rows*px, cols*px, 3] uint8 base image for a scenario, row 0 south.

    Rejects images whose recorded grid, CRS or hash disagree with the scenario.
    """
    imagery_dir = Path(imagery_dir)
    manifest = json.loads((imagery_dir / "sources.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported imagery manifest")
    source = manifest["activations"].get(scenario["activation_id"])
    if source is None:
        raise KeyError(f"no imagery for {scenario['activation_id']}")
    g, expected = scenario["grid"], source["grid"]
    if (int(g["rows"]), int(g["cols"])) != (expected["rows"], expected["cols"]) or any(
        abs(float(g[k]) - float(expected[k])) > 1e-3
        for k in ("cell_size_m", "origin_utm_easting_m", "origin_utm_northing_m")
    ) or scenario["crs"]["code"] != source["crs"]:
        raise ValueError(f"imagery grid disagrees with scenario {scenario.get('scenario_id')}")
    path = imagery_dir / source["array"]
    if sha256(path.read_bytes()).hexdigest() != source["array_sha256"]:
        raise ValueError(f"imagery hash mismatch for {scenario['activation_id']}")
    image = np.load(path)
    px = int(source["px_per_cell"])
    if image.shape != (expected["rows"] * px, expected["cols"] * px, 3) or image.dtype != np.uint8:
        raise ValueError("imagery array shape or dtype disagrees with manifest")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios_dir", type=Path, help="scenario family, or an imagery dir with --downsample")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--downsample", type=int, help="mean-pool an existing imagery dir by this factor instead")
    parser.add_argument("--px-per-cell", type=int, default=DEFAULT_PX_PER_CELL)
    parser.add_argument("--margin-days", type=int, default=DEFAULT_MARGIN_DAYS)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    args = parser.parse_args()
    if args.downsample:
        downsample_imagery(args.scenarios_dir, args.output_dir, args.downsample)
        return
    acquire_imagery(args.scenarios_dir, args.output_dir, px_per_cell=args.px_per_cell,
                    margin_days=args.margin_days, window_days=args.window_days)


if __name__ == "__main__":
    main()
