"""Build the self-contained offline 3D replay viewer used by the geographic workspace."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
import re
import struct
from pathlib import Path
from typing import Sequence
import zlib

from .replay_view import normalize_replay

VIEWER_VERSION = "pandion-viewer-0.8"
_ASSETS = Path(__file__).with_name("viewer_assets")


def _safe_json(value: object) -> str:
    """Serialize data so it cannot terminate the embedded script element."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def _png_rgb(image) -> bytes:
    """Encode an [H, W, 3] uint8 array as a PNG without optional imaging dependencies."""
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _grid_key(rows: int, cols: int, size: float, easting: float, northing: float, crs: str) -> tuple:
    return (int(rows), int(cols), round(float(size), 3), round(float(easting), 2), round(float(northing), 2), str(crs))


def attach_display_imagery(replays: Sequence[dict], imagery_dirs: Sequence[Path]) -> dict:
    """Pair normalized replays with pre-fire Sentinel-2 base images for display only.

    A replay receives an image only when an imagery manifest records exactly its
    grid size, cell size, origin and CRS; the array is hash-checked on load.
    Returns {imagery_id: payload} and sets ``display_imagery_id`` on matched replays.
    Images never reach simulation, observations or controllers.
    """
    from .imagery_acquire import SCHEMA_VERSION, load_activation_image

    sources: dict[tuple, tuple[Path, str, dict]] = {}
    for directory in imagery_dirs:
        directory = Path(directory)
        manifest = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported imagery manifest in {directory}")
        for activation, source in manifest["activations"].items():
            g = source["grid"]
            key = _grid_key(g["rows"], g["cols"], g["cell_size_m"], g["origin_utm_easting_m"], g["origin_utm_northing_m"], source["crs"])
            sources.setdefault(key, (directory, activation, source))
    payloads: dict[str, dict] = {}
    for replay in replays:
        grid, origin = replay["grid"], replay.get("origin_m") or [0, 0]
        match = sources.get(_grid_key(grid["rows"], grid["cols"], grid["cell_size_m"], origin[0], origin[1], replay.get("crs")))
        if match is None:
            continue
        directory, activation, source = match
        image_id = f"{activation}:{source['array_sha256'][:12]}"
        if image_id not in payloads:
            g = source["grid"]
            image = load_activation_image(directory, {
                "activation_id": activation, "scenario_id": replay.get("scenario_id"), "crs": {"code": source["crs"]},
                "grid": {k: g[k] for k in ("rows", "cols", "cell_size_m", "origin_utm_easting_m", "origin_utm_northing_m")}})
            north_up = image[::-1]  # stored row 0 = south; canvas row 0 = north
            payloads[image_id] = {
                "png_data_uri": "data:image/png;base64," + base64.b64encode(_png_rgb(north_up)).decode("ascii"),
                "activation_id": activation, "provider": source.get("provider"), "collection": source.get("collection"),
                "asset": source.get("asset"), "primary_datetime_utc": source.get("primary_datetime_utc"),
                "scenes_used": len(source.get("scenes_used", [])), "px_per_cell": source.get("px_per_cell"),
                "pixel_size_m": source.get("pixel_size_m"), "array_sha256": source.get("array_sha256"),
                "attribution": source.get("attribution"), "license": source.get("license"),
                "note": "Pre-fire base image for display only; not used by the simulator or controllers in this view.",
            }
        replay["display_imagery_id"] = image_id
    return payloads


def build_viewer_html(replays: Sequence[dict], title: str = "Pandion Replay Viewer",
                      imagery_dirs: Sequence[Path] | None = None) -> str:
    """Build a complete offline viewer. Replay inputs are normalized at the boundary."""
    if not replays:
        raise ValueError("at least one replay is required to generate viewer HTML")
    normalized = [normalize_replay(replay) for replay in replays]
    imagery = attach_display_imagery(normalized, imagery_dirs) if imagery_dirs else {}
    template = (_ASSETS / "viewer.html").read_text(encoding="utf-8")
    css = (_ASSETS / "viewer.css").read_text(encoding="utf-8")
    license_text = (_ASSETS / "vendor" / "THREE-LICENSE.txt").read_text(encoding="utf-8")
    js = "/*\n" + license_text + "\n*/\n" + "\n".join(
        (_ASSETS / name).read_text(encoding="utf-8") for name in
        ("vendor/three-r128.min.js", "vendor/OrbitControls-r128.js", "scene3d.js", "fleet.js", "viewer.js")
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    values = {"TITLE": html.escape(title, quote=True), "CSS": css,
              "REPLAYS": _safe_json(normalized), "IMAGERY": _safe_json(imagery), "JS": js,
              "GENERATED_AT": html.escape(generated_at, quote=True)}
    # Replace template markers once; user labels can contain the same strings.
    return re.sub(r"__(TITLE|CSS|REPLAYS|IMAGERY|GENERATED_AT|JS)__", lambda m: values[m[1]], template)


def write_viewer_html(replays: Sequence[dict], output_path: Path, title: str | None = None,
                      imagery_dirs: Sequence[Path] | None = None) -> None:
    """Generate and write the standalone viewer HTML file."""
    if not replays:
        raise ValueError("at least one replay is required to generate viewer HTML")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_title = title or ((replays[0].get("scenario_id") or "Pandion") + " · viewer")
    output_path.write_text(build_viewer_html(replays, title=page_title, imagery_dirs=imagery_dirs), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path, help="Paths to replay JSON files")
    parser.add_argument("--output", type=Path, default=Path("runs/viewer.html"), help="Destination HTML file")
    parser.add_argument("--title", type=str, default=None, help="Custom viewer page title")
    parser.add_argument("--imagery", type=Path, action="append", default=None,
                        help="Sentinel-2 imagery directory (sources.json); repeatable. Replays whose grid, origin "
                             "and CRS match a manifest entry get it as a display-only base layer")
    args = parser.parse_args()
    replays = []
    for replay_path in args.replays:
        if not replay_path.exists():
            parser.error(f"replay file not found: {replay_path}")
        replays.append(json.loads(replay_path.read_text(encoding="utf-8")))
    write_viewer_html(replays, args.output, title=args.title, imagery_dirs=args.imagery)
    print(f"Generated Pandion 3D viewer at: {args.output}")


if __name__ == "__main__":
    main()
