"""Neural policy architecture and observation/action tensorization for Stage 1.

Provides:
- ObservationTensorizer: Converts delivered observation dictionaries into
  aligned spatial raster channels and normalized vector features without latent leakage.
- ActionMapper: Bidirectional mapping between discrete action indices and order
  dictionaries, plus boolean action masking for valid candidates.
- SpatialActorCritic: PyTorch Actor-Critic module combining a spatial CNN and a
  non-spatial MLP with invalid-action masking.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .aircraft import tank_capacity_l
from .policies import candidate_cells, _containment_target, _is_burnable_fuel

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]
    HAS_TORCH = False


class ObservationTensorizer:
    """Extracts aligned spatial raster channels and non-spatial vector features.

    All features are derived strictly from delivered observations, ensuring zero
    leakage of latent simulator state or unobserved fire arrivals.
    """

    NUM_RASTER_CHANNELS = 10
    NUM_VECTOR_FEATURES = 10

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size_m: float,
        horizon_s: float,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.cell_size_m = cell_size_m
        self.horizon_s = horizon_s
        self.max_x_m = max(cols * cell_size_m, 1.0)
        self.max_y_m = max(rows * cell_size_m, 1.0)

    def extract_features(self, observation: dict) -> tuple[list[list[list[float]]], list[float]]:
        """Extract raster [C, H, W] and vector [V] as pure Python lists.

        Raster channels (10):
          0: burnable_fuel (1.0 for burnable fuel, 0.0 for NONE)
          1: fire_detected (1.0 if fire_detected is True, else 0.0)
          2: fire_unburned_observed (1.0 if fire_detected is False, else 0.0)
          3: unobserved (1.0 if fire_detected is None, else 0.0)
          4: known_treated (1.0 if cell was previously dropped/treated, else 0.0)
          5: candidate (1.0 if cell is an observed unburned perimeter candidate, else 0.0)
          6: observation_age (normalized age_s / horizon_s in [0, 1] if covered, else 0.0)
          7: elevation (grid-normalized public elevation_m, flat grids are 0.0)
          8: slope_magnitude (finite-difference slope magnitude, clamped to [0, 1])
          9: spread_risk_heatmap (physical spread advection/burning pressure in [0, 1])

        Vector features (10):
          0: aircraft_norm_x (position_x / max_x)
          1: aircraft_norm_y (position_y / max_y)
          2: water_fraction (water_l / tank capacity)
          3: is_idle (1.0 if aircraft phase is 'idle', else 0.0)
          4: is_busy (1.0 if aircraft phase is not 'idle', else 0.0)
          5: water_source_norm_x (water_source_x / max_x)
          6: water_source_norm_y (water_source_y / max_y)
          7: wind_norm_u (midflame_wind_u / 10.0)
          8: wind_norm_v (midflame_wind_v / 10.0)
          9: time_norm (time_s / horizon_s)
        """
        fuel_model = observation["fuel_model"]
        elevation = observation["elevation_m"]
        fire_cells = observation["fire"]["cells"]
        treated_set = {tuple(p) for p in observation.get("known_treated_cells", [])}
        candidates_set = {tuple(p) for p in candidate_cells(observation)}

        raster = [[[0.0] * self.cols for _ in range(self.rows)] for _ in range(self.NUM_RASTER_CHANNELS)]
        z_min = min(min(row) for row in elevation)
        z_max = max(max(row) for row in elevation)
        z_span = max(z_max - z_min, 1.0)
        cell_size = max(float(observation.get("cell_size_m", self.cell_size_m)), 1e-9)
        east_wind, north_wind = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
        w_mag = math.hypot(east_wind, north_wind)

        burning_set = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if isinstance(fire_cells[r][c], dict) and fire_cells[r][c].get("fire_detected") is True
        }

        for r in range(self.rows):
            for c in range(self.cols):
                # Channel 0: burnable fuel
                if _is_burnable_fuel(fuel_model[r][c]):
                    raster[0][r][c] = 1.0

                cell_obs = fire_cells[r][c]
                if isinstance(cell_obs, dict):
                    det = cell_obs.get("fire_detected")
                    if det is True:
                        raster[1][r][c] = 1.0
                    elif det is False:
                        raster[2][r][c] = 1.0
                    else:
                        raster[3][r][c] = 1.0

                    age = cell_obs.get("age_s")
                    if age is not None and self.horizon_s > 0:
                        raster[6][r][c] = min(max(age / self.horizon_s, 0.0), 1.0)
                else:
                    # Unobserved cell
                    raster[3][r][c] = 1.0

                # Channel 4: known treated
                if (r, c) in treated_set:
                    raster[4][r][c] = 1.0

                # Channel 5: candidate perimeter
                if (r, c) in candidates_set:
                    raster[5][r][c] = 1.0

                # Channel 7: normalized public elevation
                raster[7][r][c] = (float(elevation[r][c]) - z_min) / z_span

                # Channel 8: finite-difference slope magnitude, tan(45°) clamp
                if self.cols == 1:
                    gx = 0.0
                elif c == 0:
                    gx = (float(elevation[r][1]) - float(elevation[r][0])) / cell_size
                elif c == self.cols - 1:
                    gx = (float(elevation[r][c]) - float(elevation[r][c - 1])) / cell_size
                else:
                    gx = (float(elevation[r][c + 1]) - float(elevation[r][c - 1])) / (2.0 * cell_size)

                if self.rows == 1:
                    gy = 0.0
                elif r == 0:
                    gy = (float(elevation[1][c]) - float(elevation[0][c])) / cell_size
                elif r == self.rows - 1:
                    gy = (float(elevation[r][c]) - float(elevation[r - 1][c])) / cell_size
                else:
                    gy = (float(elevation[r + 1][c]) - float(elevation[r - 1][c])) / (2.0 * cell_size)
                raster[8][r][c] = min(max(math.hypot(gx, gy), 0.0), 1.0)

                # Channel 9: spread risk heatmap
                if not _is_burnable_fuel(fuel_model[r][c]) or (r, c) in treated_set:
                    raster[9][r][c] = 0.0
                elif (r, c) in burning_set:
                    raster[9][r][c] = 1.0
                else:
                    risk = 0.0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in burning_set:
                                dist_factor = 1.0 if (dr == 0 or dc == 0) else 0.707
                                dx = -dc * cell_size
                                dy = -dr * cell_size
                                d_len = math.hypot(dx, dy)
                                if w_mag > 1e-6 and d_len > 1e-6:
                                    wind_align = (dx * east_wind + dy * north_wind) / (d_len * w_mag)
                                else:
                                    wind_align = 0.0
                                dz = float(elevation[r][c]) - float(elevation[nr][nc])
                                slope_align = math.tanh(dz / (cell_size * 2.0))
                                wind_mult = math.exp(wind_align * 0.8)
                                slope_mult = math.exp(slope_align * 0.5)
                                risk += dist_factor * wind_mult * slope_mult
                    raster[9][r][c] = round(1.0 - math.exp(-0.5 * risk), 4)

        # Vector features
        aircraft = observation["aircraft"]
        pos = aircraft.get("position_m", [0.0, 0.0])
        water_l = aircraft.get("water_l", 0.0)
        phase = aircraft.get("phase", "idle")
        water_src = observation.get("water_source_m", [0.0, 0.0])
        wind = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
        time_s = observation.get("time_s", 0.0)

        vector = [
            float(pos[0]) / self.max_x_m,
            float(pos[1]) / self.max_y_m,
            float(water_l) / tank_capacity_l(aircraft),
            1.0 if phase == "idle" else 0.0,
            0.0 if phase == "idle" else 1.0,
            float(water_src[0]) / self.max_x_m,
            float(water_src[1]) / self.max_y_m,
            float(wind[0]) / 10.0,
            float(wind[1]) / 10.0,
            float(time_s) / max(float(self.horizon_s), 1.0),
        ]

        return raster, vector

    def to_torch(
        self,
        observation: dict,
        device: Any = None,
    ) -> tuple[Any, Any]:
        """Convert observation to PyTorch tensors (raster: [1, C, H, W], vector: [1, V])."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for to_torch(). Install requirements-rl.txt.")
        raster_list, vector_list = self.extract_features(observation)
        raster_t = torch.tensor(raster_list, dtype=torch.float32, device=device).unsqueeze(0)
        vector_t = torch.tensor(vector_list, dtype=torch.float32, device=device).unsqueeze(0)
        return raster_t, vector_t


TACTICAL_ACTION_NAMES = (
    "HOLD",
    "RETURN_TO_REFILL",
    "ATTACK_HEAD_NEAREST",
    "ATTACK_HEAD_LARGEST",
    "ATTACK_CONTAINMENT_NEAREST",
    "ATTACK_FLANK_LEFT",
    "ATTACK_FLANK_RIGHT",
    "ATTACK_REAR",
    "ATTACK_CLOSEST",
)


def _get_fire_clusters(observation: dict) -> list[list[tuple[int, int]]]:
    """Find 8-connected components of detected burning cells strictly from observation."""
    cells = observation["fire"]["cells"]
    rows, cols = len(cells), len(cells[0])
    burning = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if isinstance(cells[r][c], dict) and cells[r][c].get("fire_detected") is True
    }
    visited = set()
    clusters = []
    for cell in sorted(burning):
        if cell in visited:
            continue
        cluster = []
        queue = [cell]
        visited.add(cell)
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            cr, cc = curr
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in burning and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        clusters.append(sorted(cluster))
    return clusters


class TacticalActionMapper:
    """Action mapper for structured tactical macro-actions in Stage 1 fire suppression.

    Macro-Actions (9):
      0: HOLD
      1: RETURN_TO_REFILL
      2: ATTACK_HEAD_NEAREST (attack downwind head of closest active fire cluster)
      3: ATTACK_HEAD_LARGEST (attack downwind head of largest active fire cluster)
      4: ATTACK_CONTAINMENT_NEAREST (attack containment bottleneck balancing downwind, downstream fuel, and cycle transit)
      5: ATTACK_FLANK_LEFT (attack left flank relative to wind of closest cluster)
      6: ATTACK_FLANK_RIGHT (attack right flank relative to wind of closest cluster)
      7: ATTACK_REAR (attack backing/heel fire of closest cluster)
      8: ATTACK_CLOSEST (attack candidate perimeter cell closest to aircraft)
    """

    def __init__(self, cell_size_m: float) -> None:
        self.cell_size_m = float(cell_size_m)
        self.num_actions = len(TACTICAL_ACTION_NAMES)
        self.action_names = TACTICAL_ACTION_NAMES
        self.action_mode = "tactical"

    def compute_mask(self, observation: dict) -> list[bool]:
        """Return boolean mask list of length 8 where mask[a] is True iff valid."""
        mask = [False] * self.num_actions
        aircraft = observation["aircraft"]
        if aircraft["phase"] != "idle":
            mask[0] = True
            return mask
        if aircraft["water_l"] <= 0:
            mask[1] = True
            return mask

        # Aircraft is idle and has water
        mask[0] = True
        if aircraft["water_l"] < tank_capacity_l(aircraft):
            mask[1] = True

        candidates = candidate_cells(observation)
        if candidates:
            # Tactical drop actions (2..7) are valid when unburned perimeter candidates exist
            for idx in range(2, self.num_actions):
                mask[idx] = True
        return mask

    def to_mask_torch(self, observation: dict, device: Any = None) -> Any:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for to_mask_torch(). Install requirements-rl.txt.")
        mask_list = self.compute_mask(observation)
        return torch.tensor(mask_list, dtype=torch.bool, device=device).unsqueeze(0)

    def to_order(self, action_index: int, observation: dict | None = None) -> dict:
        """Resolve a tactical action index to a concrete simulator order."""
        if action_index == 0:
            return {"type": "HOLD"}
        if action_index == 1:
            return {"type": "RETURN_TO_REFILL"}
        if not (2 <= action_index < self.num_actions):
            raise IndexError(f"action_index {action_index} out of range [0, {self.num_actions - 1}]")

        if observation is None:
            raise ValueError("observation is required to resolve tactical drop actions")

        candidates = candidate_cells(observation)
        if not candidates:
            return {"type": "HOLD"}

        tactic = self.action_names[action_index]
        if tactic == "ATTACK_CONTAINMENT_NEAREST":
            target = _containment_target(observation, candidates)
            return {"type": "DROP", "target_m": [target[1] * self.cell_size_m, target[0] * self.cell_size_m]}

        pos_x, pos_y = observation["aircraft"]["position_m"]
        east, north = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
        w_len = math.hypot(east, north)

        clusters = _get_fire_clusters(observation)
        if not clusters:
            best = min(candidates, key=lambda c: (math.hypot(c[1] * self.cell_size_m - pos_x, c[0] * self.cell_size_m - pos_y), c[0], c[1]))
            return {"type": "DROP", "target_m": [best[1] * self.cell_size_m, best[0] * self.cell_size_m]}

        # Sort clusters by distance from centroid to aircraft
        def cluster_dist(clust: list[tuple[int, int]]) -> float:
            cx = sum(c * self.cell_size_m for r, c in clust) / len(clust)
            cy = sum(r * self.cell_size_m for r, c in clust) / len(clust)
            return math.hypot(cx - pos_x, cy - pos_y)

        clusters_by_dist = sorted(clusters, key=cluster_dist)
        clusters_by_size = sorted(clusters, key=lambda cl: (-len(cl), cluster_dist(cl)))

        nearest_cluster = set(clusters_by_dist[0])
        largest_cluster = set(clusters_by_size[0])

        target_cluster = largest_cluster if tactic == "ATTACK_HEAD_LARGEST" else nearest_cluster

        # Filter candidates adjacent to the target cluster
        clust_cands = [
            c for c in candidates
            if any((c[0] + dr, c[1] + dc) in target_cluster for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        ]
        if not clust_cands:
            clust_cands = candidates

        def score_cand(cand: list[int]) -> tuple[float, float, float, int, int]:
            r, c = cand
            projections = []
            crosses = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in target_cluster:
                    dx = -dc * self.cell_size_m
                    dy = -dr * self.cell_size_m
                    if w_len > 1e-6:
                        proj = (dx * east + dy * north) / w_len
                        cross = (east * dy - north * dx) / w_len
                    else:
                        proj = 0.0
                        cross = 0.0
                    projections.append(proj)
                    crosses.append(cross)
            p = max(projections) if projections else 0.0
            cr = crosses[projections.index(p)] if projections else 0.0
            dist = math.hypot(c * self.cell_size_m - pos_x, r * self.cell_size_m - pos_y)
            return (p, cr, dist, r, c)

        if tactic in ("ATTACK_HEAD_NEAREST", "ATTACK_HEAD_LARGEST"):
            best = max(clust_cands, key=lambda c: (score_cand(c)[0], -score_cand(c)[2], -c[0], -c[1]))
        elif tactic == "ATTACK_FLANK_LEFT":
            best = max(clust_cands, key=lambda c: (score_cand(c)[1], -score_cand(c)[2], -c[0], -c[1]))
        elif tactic == "ATTACK_FLANK_RIGHT":
            best = min(clust_cands, key=lambda c: (score_cand(c)[1], score_cand(c)[2], c[0], c[1]))
        elif tactic == "ATTACK_REAR":
            best = min(clust_cands, key=lambda c: (score_cand(c)[0], score_cand(c)[2], c[0], c[1]))
        else:  # ATTACK_CLOSEST
            best = min(clust_cands, key=lambda c: (score_cand(c)[2], -score_cand(c)[0], c[0], c[1]))

        return {"type": "DROP", "target_m": [best[1] * self.cell_size_m, best[0] * self.cell_size_m]}


class ActionMapper:
    """Bidirectional mapping between discrete action index and order dictionary.

    Supports two modes:
      - "spatial" (default): 2 + rows * cols actions (HOLD, REFILL, and cell drops).
      - "tactical": 9 tactical macro-actions (HOLD, REFILL, HEAD, CONTAINMENT, FLANKS, REAR, CLOSEST).
    """

    def __init__(self, rows: int, cols: int, cell_size_m: float, action_mode: str = "spatial") -> None:
        self.rows = rows
        self.cols = cols
        self.cell_size_m = cell_size_m
        self.action_mode = action_mode
        if action_mode == "tactical":
            self._tactical = TacticalActionMapper(cell_size_m)
            self.num_actions = self._tactical.num_actions
        elif action_mode == "parametric":
            from .parametric_actions import ParametricActionMapper
            self._tactical = ParametricActionMapper(cell_size_m)
            self.num_actions = self._tactical.num_actions
        else:
            self._tactical = None
            self.num_actions = 2 + rows * cols

    def to_order(self, action_index: int, observation: dict | None = None) -> dict:
        """Convert a discrete action index to an order dictionary."""
        if self._tactical is not None:
            return self._tactical.to_order(action_index, observation=observation)
        if action_index == 0:
            return {"type": "HOLD"}
        if action_index == 1:
            return {"type": "RETURN_TO_REFILL"}
        if 2 <= action_index < self.num_actions:
            drop_idx = action_index - 2
            r = drop_idx // self.cols
            c = drop_idx % self.cols
            return {
                "type": "DROP",
                "target_m": [c * self.cell_size_m, r * self.cell_size_m],
            }
        raise IndexError(f"action_index {action_index} out of range [0, {self.num_actions - 1}]")

    def to_action_index(self, order: dict) -> int:
        """Convert an order dictionary to a discrete action index."""
        if self._tactical is not None:
            order_type = order.get("type")
            if order_type == "HOLD":
                return 0
            if order_type == "RETURN_TO_REFILL":
                return 1
            raise ValueError(f"Direct spatial order cannot be mapped back to unique tactical action: {order}")
        order_type = order.get("type")
        if order_type == "HOLD":
            return 0
        if order_type == "RETURN_TO_REFILL":
            return 1
        if order_type == "DROP":
            target = order.get("target_m")
            if target is None or len(target) != 2:
                raise ValueError(f"Invalid target_m in DROP order: {order}")
            c = int(round(target[0] / self.cell_size_m))
            r = int(round(target[1] / self.cell_size_m))
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"DROP coordinates ({target}) outside grid [rows={self.rows}, cols={self.cols}]")
            return 2 + r * self.cols + c
        raise ValueError(f"Unknown order type: {order_type}")

    def compute_mask(self, observation: dict) -> list[bool]:
        """Return boolean mask list of length num_actions where mask[a] is True iff valid."""
        if self._tactical is not None:
            return self._tactical.compute_mask(observation)
        mask = [False] * self.num_actions
        aircraft = observation["aircraft"]
        if aircraft["phase"] != "idle":
            mask[0] = True
            return mask
        if aircraft["water_l"] <= 0:
            mask[1] = True
            return mask

        mask[0] = True
        if aircraft["water_l"] < tank_capacity_l(aircraft):
            mask[1] = True
        for r, c in candidate_cells(observation):
            if 0 <= r < self.rows and 0 <= c < self.cols:
                mask[2 + r * self.cols + c] = True
        return mask

    def to_mask_torch(self, observation: dict, device: Any = None) -> Any:
        """Return boolean mask as PyTorch tensor of shape [1, num_actions]."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for to_mask_torch(). Install requirements-rl.txt.")
        mask_list = self.compute_mask(observation)
        return torch.tensor(mask_list, dtype=torch.bool, device=device).unsqueeze(0)


if HAS_TORCH:
    class SpatialActorCritic(nn.Module):
        """Actor-Critic network combining spatial CNN and non-spatial vector MLP.

        Supports feedforward architecture or recurrent LSTM memory core.
        Outputs unmasked logits for 2 + H*W actions and scalar state value V(s).
        Enforces invalid-action masking during action selection and evaluation.
        """

        def __init__(
            self,
            in_channels: int = ObservationTensorizer.NUM_RASTER_CHANNELS,
            vector_dim: int = ObservationTensorizer.NUM_VECTOR_FEATURES,
            grid_rows: int = 11,
            grid_cols: int = 11,
            hidden_dim: int = 128,
            recurrent: bool = False,
            lstm_hidden_dim: int = 128,
            action_dim: int | None = None,
        ) -> None:
            super().__init__()
            self.grid_rows = grid_rows
            self.grid_cols = grid_cols
            self.action_dim = action_dim if action_dim is not None else (2 + grid_rows * grid_cols)
            self.recurrent = recurrent
            self.hidden_dim = hidden_dim
            self.lstm_hidden_dim = lstm_hidden_dim

            # Spatial CNN backbone
            self.spatial_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.LeakyReLU(0.01),
                nn.Flatten(),
            )
            spatial_out_dim = 16 * grid_rows * grid_cols

            # Vector MLP backbone
            self.vector_mlp = nn.Sequential(
                nn.Linear(vector_dim, 32),
                nn.LeakyReLU(0.01),
                nn.Linear(32, 32),
                nn.LeakyReLU(0.01),
            )

            # Combined fusion representation
            self.fusion = nn.Sequential(
                nn.Linear(spatial_out_dim + 32, hidden_dim),
                nn.LeakyReLU(0.01),
            )

            if self.recurrent:
                self.lstm = nn.LSTMCell(hidden_dim, lstm_hidden_dim)
                head_in_dim = lstm_hidden_dim
            else:
                head_in_dim = hidden_dim

            # Heads
            self.actor_head = nn.Linear(head_in_dim, self.action_dim)
            self.critic_head = nn.Linear(head_in_dim, 1)

        def initial_state(
            self,
            batch_size: int = 1,
            device: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
            """Return zeroed initial (h_0, c_0) state if recurrent, else None."""
            if not self.recurrent:
                return None
            h0 = torch.zeros(batch_size, self.lstm_hidden_dim, device=device)
            c0 = torch.zeros(batch_size, self.lstm_hidden_dim, device=device)
            return (h0, c0)

        def forward(
            self,
            raster: torch.Tensor,
            vector: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """Compute action logits and state-value V(s).

            Args:
                raster: Tensor [B, C, H, W]
                vector: Tensor [B, V]
                state: Optional (h, c) tuple for recurrent forward pass

            Returns:
                If recurrent: (logits, values, (h_next, c_next))
                Else: (logits, values)
            """
            spatial_feat = self.spatial_cnn(raster)
            vector_feat = self.vector_mlp(vector)
            combined = torch.cat([spatial_feat, vector_feat], dim=-1)
            latent = self.fusion(combined)

            if self.recurrent:
                if state is None:
                    state = self.initial_state(batch_size=raster.shape[0], device=raster.device)
                h_next, c_next = self.lstm(latent, state)
                logits = self.actor_head(h_next)
                values = self.critic_head(h_next).squeeze(-1)
                return logits, values, (h_next, c_next)

            logits = self.actor_head(latent)
            values = self.critic_head(latent).squeeze(-1)
            return logits, values

        def get_action(
            self,
            raster: torch.Tensor,
            vector: torch.Tensor,
            action_mask: torch.Tensor,
            deterministic: bool = False,
            state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """Sample or select action with invalid-action masking.

            Args:
                raster: [B, C, H, W]
                vector: [B, V]
                action_mask: BoolTensor [B, action_dim] where True indicates valid action
                deterministic: If True, selects argmax of valid logits
                state: Optional (h, c) recurrent state

            Returns:
                If recurrent: (actions, log_probs, entropies, values, next_state)
                Else: (actions, log_probs, entropies, values)
            """
            if self.recurrent:
                logits, values, next_state = self.forward(raster, vector, state=state)
            else:
                logits, values = self.forward(raster, vector)
                next_state = None

            masked_logits = torch.where(
                action_mask, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            )
            dist = Categorical(logits=masked_logits)
            if deterministic:
                actions = torch.argmax(masked_logits, dim=-1)
            else:
                actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            if self.recurrent:
                return actions, log_probs, entropies, values, next_state
            return actions, log_probs, entropies, values

        def evaluate_actions(
            self,
            raster: torch.Tensor,
            vector: torch.Tensor,
            action_mask: torch.Tensor,
            actions: torch.Tensor,
            initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for policy gradient updates.

            If recurrent=True, unrolls sequentially over the batch dimension [T, ...]
            maintaining recurrent state across timesteps with BPTT.

            Args:
                raster: [B, C, H, W] or [T, C, H, W]
                vector: [B, V] or [T, V]
                action_mask: BoolTensor [B, action_dim] or [T, action_dim]
                actions: LongTensor [B] or [T]
                initial_state: Optional initial (h_0, c_0) for recurrent evaluation

            Returns:
                log_probs: [B] or [T]
                entropies: [B] or [T]
                values: [B] or [T]
            """
            if not self.recurrent:
                logits, values = self.forward(raster, vector)
            else:
                spatial_feat = self.spatial_cnn(raster)
                vector_feat = self.vector_mlp(vector)
                combined = torch.cat([spatial_feat, vector_feat], dim=-1)
                latents = self.fusion(combined)

                T = latents.shape[0]
                if initial_state is None:
                    h, c = self.initial_state(batch_size=1, device=latents.device)
                else:
                    h, c = initial_state

                h_list = []
                for t in range(T):
                    h, c = self.lstm(latents[t : t + 1], (h, c))
                    h_list.append(h)
                all_h = torch.cat(h_list, dim=0)

                logits = self.actor_head(all_h)
                values = self.critic_head(all_h).squeeze(-1)

            masked_logits = torch.where(
                action_mask, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            )
            dist = Categorical(logits=masked_logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            return log_probs, entropies, values

        def select_action(
            self,
            observation: dict,
            tensorizer: ObservationTensorizer,
            mapper: ActionMapper,
            deterministic: bool = False,
            state: tuple[torch.Tensor, torch.Tensor] | None = None,
            device: Any = None,
        ) -> int | tuple[int, tuple[torch.Tensor, torch.Tensor]]:
            """Convenience helper to select a single action index from an observation dict.

            If recurrent=True, returns (action_index, next_state).
            If recurrent=False, returns action_index.
            """
            raster_t, vector_t = tensorizer.to_torch(observation, device=device)
            mask_t = mapper.to_mask_torch(observation, device=device)
            with torch.no_grad():
                if self.recurrent:
                    actions, _, _, _, next_state = self.get_action(
                        raster_t, vector_t, mask_t, deterministic=deterministic, state=state
                    )
                    return int(actions.item()), next_state
                else:
                    actions, _, _, _ = self.get_action(
                        raster_t, vector_t, mask_t, deterministic=deterministic
                    )
                    return int(actions.item())

else:
    class SpatialActorCritic:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for SpatialActorCritic. Install requirements-rl.txt.")
