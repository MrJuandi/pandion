"""Learned map of which cells burn by the end, fitted to outcomes of flights (D-100).

A dilated convolutional network maps the observed grid (reached and burning cells, brightness, time since
arrival, drops, terrain, fuel, wind, moisture, time left) to the probability that each unburned burnable
cell has burned by the end of the horizon. No fire formula, no hold limit, no simulation. Weights are read
from an ``.npz`` in a run directory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .environment import MOISTURE_KEYS
from .policies import _is_burnable_fuel, treated_counts

NUM_CHANNELS = 25


def burn_channels(observation: dict, horizon_s: float) -> np.ndarray:
    """float32 [NUM_CHANNELS, rows, cols] of observed values (same order as the training script)."""
    from .policy_inputs import elevation_gradient, fuel_family
    cells = observation["fire"]["cells"]
    rows, cols, now = len(cells), len(cells[0]), float(observation["time_s"])
    elevation = np.asarray(observation["elevation_m"], dtype=np.float64)
    gx, gy = elevation_gradient(elevation, float(observation["cell_size_m"]))
    fuel = [[int(v) for v in line] for line in observation["fuel_model"]]
    x = np.zeros((NUM_CHANNELS, rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cell = cells[r][c]
            if cell["fire_detected"]:
                x[0, r, c] = 1.0
                x[3, r, c] = min(max(now - (cell["arrival_time_s"] or 0.0), 0.0) / 3600.0, 1.0)
            if cell["visible_state"] == "burning":
                x[1, r, c] = 1.0
                x[2, r, c] = (cell["thermal_byte"] or 0) / 255.0
            x[8 + fuel_family(fuel[r][c]), r, c] = 1.0
            x[24, r, c] = float(_is_burnable_fuel(str(fuel[r][c])))
    for (r, c), n in treated_counts(observation).items():
        x[4, r, c] = min(n, 3) / 3.0
    x[5] = (elevation - elevation.min()) / max(float(elevation.max() - elevation.min()), 1.0)
    x[6], x[7] = np.clip(gx, -1.0, 1.0), np.clip(gy, -1.0, 1.0)
    east, north = observation["midflame_wind_to_m_s"]
    x[15], x[16] = east / 10.0, north / 10.0
    for k, key in enumerate(MOISTURE_KEYS):
        x[17 + k] = float(observation["fuel_moisture"][key])
    x[23] = (horizon_s - now) / horizon_s
    return x


@lru_cache(maxsize=4)
def load_net(path: str) -> tuple[Any, Any, Any]:
    import torch

    class BurnNet(torch.nn.Module):
        def __init__(self, channels: int, width: int = 32) -> None:
            super().__init__()
            self.inp = torch.nn.Conv2d(channels, width, 3, padding=1)
            self.blocks = torch.nn.ModuleList(torch.nn.Sequential(
                torch.nn.Conv2d(width, width, 3, padding=d, dilation=d), torch.nn.SiLU(),
                torch.nn.Conv2d(width, width, 3, padding=d, dilation=d)) for d in (1, 2, 4, 8, 4, 2, 1))
            self.out = torch.nn.Conv2d(width, 1, 1)

        def forward(self, x: Any) -> Any:
            h = torch.nn.functional.silu(self.inp(x))
            for block in self.blocks:
                h = torch.nn.functional.silu(h + block(h))
            return self.out(h)[:, 0]

    with np.load(path) as data:
        net = BurnNet(NUM_CHANNELS)
        outputs = int(data["net/out.weight"].shape[0])
        if outputs != 1:
            net.out = torch.nn.Conv2d(32, outputs, 1)
        net.load_state_dict({k[4:]: torch.from_numpy(data[k]) for k in data.files if k.startswith("net/")})
        minutes = data["minutes"] if "minutes" in data.files else np.array([1e9], np.float32)
        return net.eval(), data["mu"], data["sd"], minutes


def burn_outputs(observation: dict, horizon_s: float, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """([K, rows, cols] probabilities, [K] minutes): reached within each number of minutes (1e9: by the end).

    Reached cells are 1, nonburnable ones 0. A burn-map file (D-100) has one output, by the end; an
    arrival file (D-102) has several, the last of them by the end.
    """
    import torch
    net, mu, sd, minutes = load_net(str(path))
    x = burn_channels(observation, horizon_s)
    with torch.no_grad():
        h = torch.nn.functional.silu(net.inp(torch.from_numpy((x - mu[:, None, None]) / sd[:, None, None])[None].float()))
        for block in net.blocks:
            h = torch.nn.functional.silu(h + block(h))
        logit = net.out(h)[0].numpy()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    out = np.where(x[0][None] > 0, 1.0, np.where(x[24][None] > 0, prob, 0.0))
    return np.maximum.accumulate(out, axis=0).astype(np.float32), np.asarray(minutes, dtype=np.float32)


def burn_map(observation: dict, horizon_s: float, path: str | Path) -> np.ndarray:
    """float32 [rows, cols] probability that each cell burns by the end: 1 on reached cells, 0 on nonburnable ones."""
    return burn_outputs(observation, horizon_s, path)[0][-1]


def reached_before(probs: np.ndarray, minutes: np.ndarray, r: int, c: int, within_min: float) -> float:
    """Probability that cell (r, c) is reached within ``within_min`` minutes, interpolated from the outputs."""
    knots = np.concatenate([[0.0], minutes[:-1]])
    values = np.concatenate([[0.0], probs[:-1, r, c]])
    return float(np.interp(within_min, knots, values))
