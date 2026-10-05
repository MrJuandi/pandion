"""Image-based policy inputs for geographic real-time observations (D-49).

Splits each controller observation into three tensors:

- ``rgb`` [3, rows*px, cols*px]: pre-fire Sentinel-2 base with the current
  burning/burned/treated state drawn over it (``imagery_composite``). Fire
  state reaches the policy only through this image.
- ``cells`` [C, rows, cols]: per-cell state known without the image:
  normalized elevation, elevation gradient (east, north), one-hot fuel family
  from the prior fuel map, treated cells, aircraft distance, and optionally the
  Rothermel head-fire rate of spread (ROS) under the current wind and moisture.
- ``vector`` [V]: quantities uniform over the grid: wind, station temperature
  and RH (with a validity flag), the six fuel moisture classes, aircraft water
  and phase, refill time remaining and elapsed time.

Everything is computed from the observation dict; nothing reads simulator truth.
Fuel moisture is uniform in the current simulator; if it becomes per-cell it
belongs in ``cells``, not ``vector``.
"""
from __future__ import annotations

from math import atan2, degrees, hypot, log1p
from typing import Any

import numpy as np

from .environment import MOISTURE_KEYS
from .geographic_integrated import DEFAULT_CAPACITY_L
from .imagery_composite import compose_view

# Fuel families from Scott & Burgan (2005) code ranges and Anderson (1982) 1-13.
FUEL_FAMILIES = ("nonburnable", "grass", "grass_shrub", "shrub", "timber_understory", "timber_litter", "slash_blowdown")
ANDERSON_FAMILY = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 3, 7: 3, 8: 5, 9: 5, 10: 4, 11: 6, 12: 6, 13: 6}
ROS_SCALE_M_MIN = 100.0
GRADIENT_SCALE = 1.0  # tan(45°)
WIND_SCALE_M_S = 10.0
TEMPERATURE_SCALE_C = 50.0


def fuel_family(code: int) -> int:
    """Index into FUEL_FAMILIES for a Scott & Burgan or Anderson fuel code."""
    if code in ANDERSON_FAMILY:
        return ANDERSON_FAMILY[code]
    for low, high, family in ((101, 109, 1), (121, 124, 2), (141, 149, 3),
                              (161, 165, 4), (181, 189, 5), (201, 204, 6)):
        if low <= code <= high:
            return family
    return 0  # 0, 91-99 and unknown codes are nonburnable


def elevation_gradient(elevation: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference dz/dx (east) and dz/dy (north, rows increase north)."""
    rows, cols = elevation.shape
    gx = np.zeros_like(elevation)
    gy = np.zeros_like(elevation)
    if cols > 1:
        gx = np.gradient(elevation, cell_size_m, axis=1)
    if rows > 1:
        gy = np.gradient(elevation, cell_size_m, axis=0)
    return gx, gy


class GeographicInputTensorizer:
    """Builds (rgb, cells, vector) policy inputs for one geographic scenario."""

    BASE_CELL_CHANNELS = 3 + len(FUEL_FAMILIES) + 2
    NUM_VECTOR_FEATURES = 2 + 3 + len(MOISTURE_KEYS) + 4

    FIRE_CHANNELS = 3  # burning, burned, time since front arrival (ablation without image)

    def __init__(self, scenario: dict, base_image: np.ndarray | None, *, include_ros: bool = False,
                 explicit_fire: bool = False) -> None:
        """``base_image=None`` with ``explicit_fire=True`` is the no-image ablation:
        fire state enters as per-cell channels and ``extract`` returns (cells, vector)."""
        grid = scenario["grid"]
        self.rows, self.cols = int(grid["rows"]), int(grid["cols"])
        self.cell_size_m = float(grid["cell_size_m"])
        fire = scenario.get("fire", scenario)
        self.horizon_s = float(fire.get("horizon_s", scenario.get("horizon_s", 3600.0)))
        self.capacity_l = float(scenario.get("aircraft", {}).get("tank_capacity_l", DEFAULT_CAPACITY_L))
        if base_image is None and not explicit_fire:
            raise ValueError("without a base image, fire state must enter as explicit channels")
        self.px_per_cell = 0
        if base_image is not None:
            if base_image.ndim != 3 or base_image.shape[2] != 3 or base_image.dtype != np.uint8:
                raise ValueError("base image must be uint8 [H, W, 3]")
            if base_image.shape[0] % self.rows or base_image.shape[1] % self.cols:
                raise ValueError("base image size must be a multiple of the grid")
            self.px_per_cell = base_image.shape[0] // self.rows
            if base_image.shape[:2] != (self.rows * self.px_per_cell, self.cols * self.px_per_cell):
                raise ValueError("base image must use the same pixels per cell on both axes")
        self.base_image = base_image
        self.include_ros = include_ros
        self.explicit_fire = explicit_fire
        self.num_cell_channels = (self.BASE_CELL_CHANNELS + (1 if include_ros else 0)
                                  + (self.FIRE_CHANNELS if explicit_fire else 0))
        self._static: np.ndarray | None = None
        self._ros_cache: dict[tuple, np.ndarray] = {}
        self._diagonal_m = hypot(self.rows * self.cell_size_m, self.cols * self.cell_size_m)
        anchors_y, anchors_x = np.meshgrid(np.arange(self.rows) * self.cell_size_m,
                                           np.arange(self.cols) * self.cell_size_m, indexing="ij")
        self._anchors = (anchors_x, anchors_y)

    def _static_channels(self, observation: dict) -> np.ndarray:
        if self._static is None:
            elevation = np.asarray(observation["elevation_m"], dtype=np.float64)
            fuel = np.asarray([[int(v) for v in row] for row in observation["fuel_model"]])
            if elevation.shape != (self.rows, self.cols) or fuel.shape != (self.rows, self.cols):
                raise ValueError("observation grid disagrees with scenario")
            span = max(float(elevation.max() - elevation.min()), 1.0)
            gx, gy = elevation_gradient(elevation, self.cell_size_m)
            families = np.vectorize(fuel_family)(fuel)
            channels = [(elevation - elevation.min()) / span,
                        np.clip(gx / GRADIENT_SCALE, -1.0, 1.0),
                        np.clip(gy / GRADIENT_SCALE, -1.0, 1.0)]
            channels += [(families == k).astype(np.float64) for k in range(len(FUEL_FAMILIES))]
            self._static = np.stack(channels).astype(np.float32)
        return self._static

    def ros_channel(self, observation: dict) -> np.ndarray:
        """Normalized head-fire ROS per cell: log1p(ros) / log1p(100 m/min)."""
        moisture = observation.get("fuel_moisture")
        if moisture is None:
            raise ValueError("ROS channel needs fuel_moisture in the observation")
        wind = tuple(float(v) for v in observation["midflame_wind_to_m_s"])
        moisture_values = tuple(float(moisture[k]) for k in MOISTURE_KEYS)
        key = (wind, moisture_values)
        if key not in self._ros_cache:
            from pyretechnics import fuel_models, surface_fire

            elevation = np.asarray(observation["elevation_m"], dtype=np.float64)
            gx, gy = elevation_gradient(elevation, self.cell_size_m)
            upwind = degrees(atan2(-wind[0], -wind[1])) % 360 if wind != (0.0, 0.0) else 0.0
            wind_m_min = hypot(*wind) * 60.0
            base: dict[int, Any] = {}
            out = np.zeros((self.rows, self.cols), dtype=np.float32)
            for r, line in enumerate(observation["fuel_model"]):
                for c, value in enumerate(line):
                    code = int(value)
                    if code == 0:
                        continue
                    if code not in base:
                        model = fuel_models.moisturize(fuel_models.get_fuel_model(code), moisture_values)
                        base[code] = (surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
                                      if model.get("burnable", True) else None)
                    if base[code] is None:
                        continue
                    slope = hypot(gx[r, c], gy[r, c])
                    aspect = degrees(atan2(-gx[r, c], -gy[r, c])) % 360 if slope > 0 else 0.0
                    ros = float(surface_fire.calc_surface_fire_behavior_max(
                        base[code], wind_m_min, upwind, slope, aspect)["max_spread_rate"])
                    out[r, c] = min(log1p(max(ros, 0.0)) / log1p(ROS_SCALE_M_MIN), 1.0)
            self._ros_cache[key] = out
        return self._ros_cache[key]

    def fire_channels(self, observation: dict) -> np.ndarray:
        """Explicit [3, H, W] burning, burned and time-since-arrival / horizon."""
        out = np.zeros((self.FIRE_CHANNELS, self.rows, self.cols), dtype=np.float32)
        now = float(observation.get("time_s", 0.0))
        for r, line in enumerate(observation["fire"]["cells"]):
            for c, cell in enumerate(line):
                state = cell["fire_state"]
                out[0, r, c] = state == "burning"
                out[1, r, c] = state == "burned"
                if cell.get("arrival_time_s") is not None:
                    out[2, r, c] = min(max(now - float(cell["arrival_time_s"]), 0.0) / max(self.horizon_s, 1.0), 1.0)
        return out

    def extract(self, observation: dict) -> tuple[np.ndarray, ...]:
        """Return (rgb [3, H*px, W*px], cells [C, H, W], vector [V]) as float32 arrays,
        or (cells, vector) when there is no base image (explicit-fire ablation)."""
        if observation.get("observation_mode") != "realtime":
            raise ValueError("image inputs require real-time observations (D-49)")
        rgb = None
        if self.base_image is not None:
            view = compose_view(self.base_image, observation, self.px_per_cell)
            rgb = np.moveaxis(view, -1, 0).astype(np.float32) / 255.0

        treated = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r, c in observation.get("known_treated_cells", []):
            treated[r, c] = 1.0
        aircraft = observation["aircraft"]
        ax, ay = (float(v) for v in aircraft.get("position_m", [0.0, 0.0]))
        distance = np.hypot(self._anchors[0] - ax, self._anchors[1] - ay) / self._diagonal_m
        dynamic = [treated, np.clip(distance, 0.0, 1.5).astype(np.float32)]
        if self.include_ros:
            dynamic.append(self.ros_channel(observation))
        parts = [self._static_channels(observation), np.stack(dynamic)]
        if self.explicit_fire:
            parts.append(self.fire_channels(observation))
        cells = np.concatenate(parts).astype(np.float32)

        wind = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
        station = (observation.get("environment_weather") or {}).get("station_weather") or {}
        temperature, humidity = station.get("air_temperature_c"), station.get("relative_humidity_pct")
        weather_known = temperature is not None and humidity is not None
        moisture = observation.get("fuel_moisture") or {}
        vector = [float(wind[0]) / WIND_SCALE_M_S, float(wind[1]) / WIND_SCALE_M_S,
                  float(temperature) / TEMPERATURE_SCALE_C if weather_known else 0.0,
                  float(humidity) / 100.0 if weather_known else 0.0,
                  1.0 if weather_known else 0.0]
        vector += [min(float(moisture.get(k, 0.0)), 3.0) for k in MOISTURE_KEYS]
        vector += [float(aircraft.get("water_l", 0.0)) / self.capacity_l,
                   1.0 if aircraft.get("phase", "idle") == "idle" else 0.0,
                   min(float(aircraft.get("refill_remaining_s", 0.0)) / max(self.horizon_s, 1.0), 1.0),
                   min(float(observation.get("time_s", 0.0)) / max(self.horizon_s, 1.0), 1.0)]
        vector_array = np.asarray(vector, dtype=np.float32)
        return (cells, vector_array) if rgb is None else (rgb, cells, vector_array)

    def to_torch(self, observation: dict, device: Any = None) -> tuple[Any, ...]:
        """Batched tensors matching ``extract`` with a leading batch dimension of 1."""
        import torch

        return tuple(torch.from_numpy(a).unsqueeze(0).to(device) for a in self.extract(observation))
