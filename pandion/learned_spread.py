"""Learned head-fire rate of spread, fitted to observed arrival times (no fire formula).

A small network predicts the speed at which fire crosses each edge between neighbouring
cells from fuel codes, terrain, wind and moisture. The weights are exported to an ``.npz``
(several fits are averaged in log speed) and are read from a run directory, never from the
package. Only observation fields are used.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np

from .environment import MOISTURE_KEYS

_DIRS = np.array(((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)))
_UNIT = _DIRS / np.linalg.norm(_DIRS, axis=1, keepdims=True)


@lru_cache(maxsize=8)
def load_weights(path: str) -> list[dict[str, np.ndarray]]:
    """Networks stored in ``path``: keys ``{index}/{name}``."""
    with np.load(path) as data:
        count = 1 + max(int(key.split("/")[0]) for key in data.files)
        return [{key.split("/", 1)[1]: data[key] for key in data.files if key.startswith(f"{i}/")} for i in range(count)]


def _forward(net: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    for layer in (0, 1):
        x = x @ net[f"w{layer}"].T + net[f"b{layer}"]
        x = x / (1.0 + np.exp(-x))  # SiLU
    return (x @ net["w2"].T + net["b2"])[:, 0]


def learned_head_speed(observation: dict, path: str | Path) -> np.ndarray:
    """[rows, cols] m/s: fastest learned speed leaving each cell in any direction (0 on fuel 0)."""
    nets = load_weights(str(path))
    fuel = np.asarray([[int(v) for v in row] for row in observation["fuel_model"]])
    elevation = np.asarray(observation["elevation_m"], dtype=np.float64)
    cell = float(observation["cell_size_m"])
    gx, gy = np.gradient(elevation, cell, axis=1), np.gradient(elevation, cell, axis=0)
    rows, cols = fuel.shape
    wx, wy = (float(v) for v in observation["midflame_wind_to_m_s"])
    moisture = np.array([float(observation["fuel_moisture"][k]) for k in MOISTURE_KEYS])
    r, c, d = (v.ravel() for v in np.meshgrid(range(rows), range(cols), range(8), indexing="ij"))
    rr, cc = r + _DIRS[d, 0], c + _DIRS[d, 1]
    ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    r, c, d, rr, cc = r[ok], c[ok], d[ok], rr[ok], cc[ok]
    uy, ux = _UNIT[d, 0], _UNIT[d, 1]
    along, cross = wx * ux + wy * uy, -wx * uy + wy * ux
    sign = np.where(cross < 0, -1.0, 1.0)
    feats = [along / 3, np.abs(cross) / 3, np.hypot(along, cross) / 3]
    for a, b in ((r, c), (rr, cc)):
        feats += [(gx[a, b] * ux + gy[a, b] * uy) * 3, (-gx[a, b] * uy + gy[a, b] * ux) * sign * 3]
    diag = d >= 4
    sides = ((fuel[rr, c] > 0) & (fuel[r, cc] > 0)) | ~diag
    feats += [diag.astype(float), sides.astype(float), (along > 0) * along ** 2 / 9, np.zeros_like(along)]
    base = np.column_stack(feats).astype(np.float32)
    moist = np.broadcast_to(((moisture - 0.3) * 3).astype(np.float32), (len(r), len(moisture)))
    logs = []
    for net in nets:
        idx = net["idx"]
        x = np.concatenate([net["emb"][idx[fuel[r, c]]], net["emb"][idx[fuel[rr, cc]]], base, moist], axis=1)
        logs.append(np.clip(_forward(net, x), -16, 2))
    speed = np.exp(np.mean(logs, axis=0)) * (fuel[rr, cc] > 0)
    out = np.zeros((rows, cols))
    np.maximum.at(out, (r, c), speed)
    return np.where(fuel > 0, out, 0.0)


_SPEED_CACHE: dict[tuple, np.ndarray] = {}
_SPEED_CACHE_LIMIT = 8192


def cached_head_speed(observation: dict, path: str | Path) -> np.ndarray:
    """``learned_head_speed`` memoised across episodes of a process (map, wind, moisture)."""
    digest = blake2b(digest_size=12)
    digest.update(str(observation["fuel_model"]).encode())
    digest.update(np.asarray(observation["elevation_m"], dtype=np.float32).tobytes())
    key = (digest.digest(), tuple(observation["midflame_wind_to_m_s"]),
           tuple(float(observation["fuel_moisture"][k]) for k in MOISTURE_KEYS), str(path))
    if key not in _SPEED_CACHE:
        if len(_SPEED_CACHE) >= _SPEED_CACHE_LIMIT:
            _SPEED_CACHE.clear()
        _SPEED_CACHE[key] = learned_head_speed(observation, path).astype(np.float32)
    return _SPEED_CACHE[key]
