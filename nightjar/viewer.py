"""Build the self-contained Spanish 3D replay viewer used by the geographic workspace."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import re
from pathlib import Path
from typing import Sequence

from .replay_view import normalize_replay

VIEWER_VERSION = "nightjar-viewer-0.6"
_ASSETS = Path(__file__).with_name("viewer_assets")


def _safe_json(value: object) -> str:
    """Serialize data so it cannot terminate the embedded script element."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def build_viewer_html(replays: Sequence[dict], title: str = "Nightjar Replay Viewer") -> str:
    """Build a complete offline viewer. Replay inputs are normalized at the boundary."""
    if not replays:
        raise ValueError("at least one replay is required to generate viewer HTML")
    normalized = [normalize_replay(replay) for replay in replays]
    template = (_ASSETS / "viewer.html").read_text(encoding="utf-8")
    css = (_ASSETS / "viewer.css").read_text(encoding="utf-8")
    license_text = (_ASSETS / "vendor" / "THREE-LICENSE.txt").read_text(encoding="utf-8")
    js = "/*\n" + license_text + "\n*/\n" + "\n".join(
        (_ASSETS / name).read_text(encoding="utf-8") for name in
        ("vendor/three-r128.min.js", "vendor/OrbitControls-r128.js", "scene3d.js", "fleet.js", "viewer.js")
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    values = {"TITLE": html.escape(title, quote=True), "CSS": css,
              "REPLAYS": _safe_json(normalized), "JS": js,
              "GENERATED_AT": html.escape(generated_at, quote=True)}
    # Replace template markers once; user labels can contain the same strings.
    return re.sub(r"__(TITLE|CSS|REPLAYS|GENERATED_AT|JS)__", lambda m: values[m[1]], template)


def write_viewer_html(replays: Sequence[dict], output_path: Path, title: str | None = None) -> None:
    """Generate and write the standalone viewer HTML file."""
    if not replays:
        raise ValueError("at least one replay is required to generate viewer HTML")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_title = title or ((replays[0].get("scenario_id") or "Nightjar") + " · visor")
    output_path.write_text(build_viewer_html(replays, title=page_title), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path, help="Paths to replay JSON files")
    parser.add_argument("--output", type=Path, default=Path("runs/viewer.html"), help="Destination HTML file")
    parser.add_argument("--title", type=str, default=None, help="Custom viewer page title")
    args = parser.parse_args()
    replays = []
    for replay_path in args.replays:
        if not replay_path.exists():
            parser.error(f"replay file not found: {replay_path}")
        replays.append(json.loads(replay_path.read_text(encoding="utf-8")))
    write_viewer_html(replays, args.output, title=args.title)
    print(f"Generated Nightjar 3D viewer at: {args.output}")


if __name__ == "__main__":
    main()
