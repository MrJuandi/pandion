"""Extract the landing-page episode from a built workspace viewer (index.html).

Usage: python3 site/extract_episode.py runs/workspace-vine-d85-20261120/index.html

Writes site/data/episode.js (replays of one scenario under three controllers plus a
per-scenario summary of the whole workspace), site/data/basemap.png (the pre-fire
Sentinel-2 display image already embedded in the viewer) and site/viewer.html (the
offline 3D viewer with every controller recorded for that scenario). Values are copied,
not recomputed, so the pages show exactly what the simulator recorded.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pandion.viewer import build_viewer_html  # noqa: E402

SCENARIO = "geographic-dev-emsr837_aoi07-06-mf00"
POLICIES = {"PPO geográfico": "learned", "Geographic PPO": "learned", "heuristic": "heuristic", "no_intervention": "none"}
OUT = Path(__file__).with_name("data")


def main(viewer: Path) -> None:
    text = viewer.read_text(encoding="utf-8")
    match = re.search(r"const REPLAYS = (.*?); const IMAGERY = (.*?);</script>", text, re.S)
    replays, imagery = json.loads(match.group(1)), json.loads(match.group(2))

    summary: dict[str, dict] = {}
    for r in replays:
        row = summary.setdefault(r["scenario_id"], {"id": r["scenario_id"], "name": r["metadata"]["name"]})
        row[POLICIES.get(r["policy"], r["policy"])] = r["outcomes"]["final_burned_ha"]

    chosen = [r for r in replays if r["scenario_id"] == SCENARIO and r["policy"] in POLICIES]
    base = chosen[0]
    image = imagery[base["display_imagery_id"]]
    episode = {
        "source_viewer_sha256": hashlib.sha256(viewer.read_bytes()).hexdigest(),
        "scenario_id": SCENARIO, "name": base["metadata"]["name"],
        "scenario_hash": base["scenario_hash"], "crs": base["crs"], "origin_utm": base["origin_utm"],
        "grid": base["grid"], "horizon_s": base["horizon_s"],
        "elevation_m": base["elevation_m"], "fuel": base["fuel_grid"],
        "assumptions": base["metadata"]["assumptions"], "wind_to_m_s": base["metadata"]["wind_to_m_s"],
        "training": {k: base["metadata"]["training"][k] for k in ("label", "seed", "updates", "checkpoint_sha256")},
        "terrain": {k: base["environment"]["terrain"]["source"][k] for k in ("provider", "dataset", "native_resolution_m")},
        "imagery": {k: image[k] for k in ("provider", "collection", "primary_datetime_utc", "attribution")},
        "aircraft_initial": base["aircraft_initial_state"],
        "runs": {},
        "summary": list(summary.values()),
    }
    for r in chosen:
        episode["runs"][POLICIES[r["policy"]]] = {
            "arrival": r["arrival_time_s"], "burnout": r["burnout_time_s"],
            "final_ha": r["outcomes"]["final_burned_ha"], "initial_ha": r["outcomes"]["initial_burned_ha"],
            "metrics": r["aircraft_metrics"],
            "track": [[p["aircraft_index"], p["time_s"], *p["position_m"]] for p in r["aircraft_track"]],
            "drops": [[d["aircraft_index"], d["time_s"], d["water_l"], d["newly_blocked_cells"]] for d in r["drops"]],
        }
    OUT.mkdir(exist_ok=True)
    payload = json.dumps(episode, ensure_ascii=False, separators=(",", ":"))
    (OUT / "episode.js").write_text(f"window.PANDION_EPISODE={payload};\n", encoding="utf-8")
    png = image["png_data_uri"].split(",", 1)[1]
    (OUT / "basemap.png").write_bytes(base64.b64decode(png))
    scenario_runs = [r for r in replays if r["scenario_id"] == SCENARIO]
    page = build_viewer_html(scenario_runs, title=f"{base['metadata']['name']} · Pandion viewer",
                             imagery={base["display_imagery_id"]: image})
    OUT.parent.joinpath("viewer.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
