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


# Simulated thermal (infrared) band, values in [0, 1] relative to ambient.
# A declared rendering convention for a drone-type RGB-T sensor (D-58), not a
# radiometric model: burning cells saturate, burned cells cool exponentially
# from front arrival, and radiant heat around burning cells decays with distance.
THERMAL_BURNING = 1.0
THERMAL_BURNED_PEAK = 0.7
THERMAL_COOLING_S = 900.0
THERMAL_HALO_PEAK = 0.5
THERMAL_HALO_LENGTH_M = 75.0
# With the measured brightness (D-94, D-96) burned ground and halo are compressed to this level, so a burning
# cell (76-255 of 255) is clearly apart from ground that is not burning and its own range is wide.
THERMAL_GROUND_MAX = 0.2


def _bilinear_upsample(grid: np.ndarray, px_per_cell: int) -> np.ndarray:
    """Smoothly resample a [rows, cols] cell-centred field to pixel resolution."""
    def axis_weights(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.clip((np.arange(n * px_per_cell) + 0.5) / px_per_cell - 0.5, 0.0, n - 1.0)
        low = np.floor(pos).astype(int)
        high = np.minimum(low + 1, n - 1)
        return low, high, pos - low
    r0, r1, wr = axis_weights(grid.shape[0])
    c0, c1, wc = axis_weights(grid.shape[1])
    rows = grid[r0] * (1.0 - wr)[:, None] + grid[r1] * wr[:, None]
    return rows[:, c0] * (1.0 - wc)[None, :] + rows[:, c1] * wc[None, :]


def thermal_view(observation: dict, px_per_cell: int, cell_size_m: float,
                 masks: dict[str, np.ndarray] | None = None, measured: bool = False,
                 state_key: str = "fire_state") -> np.ndarray:
    """Float32 [rows*px, cols*px] thermal band in [0, 1], row 0 south.

    Burning cells are 1, or with ``measured`` show the sensor's ``thermal_byte`` / 255 and burned ground and halo
    are scaled below 0.2 (D-94, D-96). Burned cells start at 0.7 and cool as
    exp(-(t - arrival) / 900 s). Around burning cells radiant heat adds
    0.5 * exp(-d / 75 m), d being the distance to the nearest burning cell
    centre. Cell values are resampled bilinearly and burning cells are then
    drawn as sharp full-intensity blocks. Uses only the controller observation.
    """
    if masks is None:
        masks = state_masks(observation, state_key)
    rows, cols = masks["burning"].shape
    now = float(observation.get("time_s", 0.0))
    field = np.zeros((rows, cols), dtype=np.float64)
    for r, line in enumerate(observation["fire"]["cells"]):
        for c, cell in enumerate(line):
            if cell[state_key] == "burned" and cell.get("arrival_time_s") is not None:
                age = max(now - float(cell["arrival_time_s"]), 0.0)
                field[r, c] = THERMAL_BURNED_PEAK * np.exp(-age / THERMAL_COOLING_S)
    burning = np.argwhere(masks["burning"])
    if len(burning):
        grid = np.argwhere(np.ones((rows, cols), dtype=bool))
        nearest = np.full(len(grid), np.inf)
        for start in range(0, len(burning), 512):
            block = burning[start:start + 512]
            d = np.hypot(grid[:, None, 0] - block[None, :, 0], grid[:, None, 1] - block[None, :, 1])
            nearest = np.minimum(nearest, d.min(axis=1))
        halo = THERMAL_HALO_PEAK * np.exp(-nearest.reshape(rows, cols) * cell_size_m / THERMAL_HALO_LENGTH_M)
        field = np.maximum(field, halo)
    if measured:
        field *= THERMAL_GROUND_MAX / THERMAL_BURNED_PEAK  # burned ground and halo stay well under any burning cell (D-96)
    image = _bilinear_upsample(field, px_per_cell)
    if not measured:
        image[_upsample(masks["burning"], px_per_cell)] = THERMAL_BURNING
    else:
        heat = np.zeros((rows, cols))
        for r, c in burning:
            heat[r, c] = float(observation["fire"]["cells"][r][c]["thermal_byte"]) / 255.0
        mask = _upsample(masks["burning"], px_per_cell)
        image[mask] = np.repeat(np.repeat(heat, px_per_cell, 0), px_per_cell, 1)[mask]
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _upsample(mask: np.ndarray, px_per_cell: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, px_per_cell, axis=0), px_per_cell, axis=1)


def _blend(image: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int], alpha: float) -> None:
    if mask.any():
        colour = np.asarray(rgb, dtype=np.float32)
        image[mask] = (1.0 - alpha) * image[mask] + alpha * colour


def state_masks(observation: dict, state_key: str = "fire_state") -> dict[str, np.ndarray]:
    """Boolean [rows, cols] masks for burning, burned and treated cells.

    ``state_key`` is ``fire_state`` (D-67) or ``visible_state``, which does not depend on water (D-95).

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
            burning[r, c] = cell[state_key] == "burning"
            burned[r, c] = cell[state_key] == "burned"
    treated = np.zeros((rows, cols), dtype=bool)
    for r, c in observation.get("known_treated_cells", []):
        treated[r, c] = True
    return {"burning": burning, "burned": burned, "treated": treated}


def compose_view(base: np.ndarray, observation: dict, px_per_cell: int,
                 masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Return the uint8 [rows*px, cols*px, 3] view for one decision, row 0 south."""
    if masks is None:
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
    parser.add_argument("--thermal", action="store_true", help="also write the thermal band as *_thermal.png")
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
        if args.thermal:
            heat = thermal_view(decision["observation"], px, float(scenario["grid"]["cell_size_m"]))
            Image.fromarray(np.rint(heat[::-1] * 255).astype(np.uint8)).save(
                args.output_dir / name.replace(".png", "_thermal.png"))
        print(name, flush=True)


if __name__ == "__main__":
    main()
