"""Per-step RGB view: pre-fire base image with the current fire and treatment state.

Models the real-time aerial view assumed in D-49. The base is the pre-fire
Sentinel-2 composite from ``imagery_acquire``; burning, burned and treated cells
are drawn over it from the controller observation only (never simulator truth).

The overlay colours are a declared rendering convention, not a sensor model:
burning cells are blended toward flame orange, burned cells toward char, and
treated cells toward a water/retardant tint. Burning is drawn last because it
is the most decision-relevant state.

Arrays follow the scenario grid convention: row 0 is the southern edge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BURNING_RGB = (255, 96, 16)
BURNED_RGB = (38, 34, 30)
TREATED_RGB = (60, 110, 205)
BURNING_ALPHA = 0.85
BURNED_ALPHA = 0.80
TREATED_ALPHA = 0.55


def _upsample(mask: np.ndarray, px_per_cell: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, px_per_cell, axis=0), px_per_cell, axis=1)


def _blend(image: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int], alpha: float) -> None:
    if mask.any():
        colour = np.asarray(rgb, dtype=np.float32)
        image[mask] = (1.0 - alpha) * image[mask] + alpha * colour


def state_masks(observation: dict) -> dict[str, np.ndarray]:
    """Boolean [rows, cols] masks for burning, burned and treated cells.

    Requires the real-time observation (``fire_state`` per cell); delayed
    observations have no burning/burned distinction and are rejected.
    """
    cells = observation["fire"]["cells"]
    rows, cols = len(cells), len(cells[0])
    burning = np.zeros((rows, cols), dtype=bool)
    burned = np.zeros((rows, cols), dtype=bool)
    for r, line in enumerate(cells):
        for c, cell in enumerate(line):
            if not isinstance(cell, dict) or "fire_state" not in cell:
                raise ValueError("composite requires real-time observations with fire_state per cell")
            burning[r, c] = cell["fire_state"] == "burning"
            burned[r, c] = cell["fire_state"] == "burned"
    treated = np.zeros((rows, cols), dtype=bool)
    for r, c in observation.get("known_treated_cells", []):
        treated[r, c] = True
    return {"burning": burning, "burned": burned, "treated": treated}


def compose_view(base: np.ndarray, observation: dict, px_per_cell: int) -> np.ndarray:
    """Return the uint8 [rows*px, cols*px, 3] view for one decision, row 0 south."""
    masks = state_masks(observation)
    rows, cols = masks["burning"].shape
    if base.shape != (rows * px_per_cell, cols * px_per_cell, 3) or base.dtype != np.uint8:
        raise ValueError("base image does not match the observation grid")
    image = base.astype(np.float32)
    _blend(image, _upsample(masks["treated"], px_per_cell), TREATED_RGB, TREATED_ALPHA)
    _blend(image, _upsample(masks["burned"], px_per_cell), BURNED_RGB, BURNED_ALPHA)
    _blend(image, _upsample(masks["burning"], px_per_cell), BURNING_RGB, BURNING_ALPHA)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render north-up PNG views of a replay's decision observations.")
    parser.add_argument("replay", type=Path, help="geographic replay JSON (schema 0.4, realtime mode)")
    parser.add_argument("scenario", type=Path, help="scenario JSON the replay was run on")
    parser.add_argument("imagery_dir", type=Path, help="output directory of imagery_acquire")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--every", type=int, default=5, help="render every Nth decision")
    args = parser.parse_args()
    from PIL import Image

    from .imagery_acquire import load_activation_image

    scenario = json.loads(args.scenario.read_text())
    replay = json.loads(args.replay.read_text())
    base = load_activation_image(args.imagery_dir, scenario)
    px = base.shape[0] // int(scenario["grid"]["rows"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, decision in enumerate(replay["decisions"]):
        if index % args.every:
            continue
        view = compose_view(base, decision["observation"], px)
        name = f"step{index:03d}_t{int(decision['decision_time_s']):05d}s.png"
        Image.fromarray(view[::-1]).save(args.output_dir / name)
        print(name, flush=True)


if __name__ == "__main__":
    main()
