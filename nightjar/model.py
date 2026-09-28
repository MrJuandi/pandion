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

from .aircraft import CAPACITY_L
from .policies import candidate_cells

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

    NUM_RASTER_CHANNELS = 9
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

        Raster channels (9):
          0: burnable_fuel (1.0 for burnable fuel, 0.0 for NONE)
          1: fire_detected (1.0 if fire_detected is True, else 0.0)
          2: fire_unburned_observed (1.0 if fire_detected is False, else 0.0)
          3: unobserved (1.0 if fire_detected is None, else 0.0)
          4: known_treated (1.0 if cell was previously dropped/treated, else 0.0)
          5: candidate (1.0 if cell is an observed unburned perimeter candidate, else 0.0)
          6: observation_age (normalized age_s / horizon_s in [0, 1] if covered, else 0.0)
          7: elevation (grid-normalized public elevation_m, flat grids are 0.0)
          8: slope_magnitude (finite-difference slope magnitude, clamped to [0, 1])

        Vector features (10):
          0: aircraft_norm_x (position_x / max_x)
          1: aircraft_norm_y (position_y / max_y)
          2: water_fraction (water_l / CAPACITY_L)
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

        for r in range(self.rows):
            for c in range(self.cols):
                # Channel 0: burnable fuel
                if fuel_model[r][c] != "NONE":
                    raster[0][r][c] = 1.0

                cell_obs = fire_cells[r][c]
                det = cell_obs.get("fire_detected")
                if det is True:
                    raster[1][r][c] = 1.0
                elif det is False:
                    raster[2][r][c] = 1.0
                else:
                    raster[3][r][c] = 1.0

                # Channel 4: known treated
                if (r, c) in treated_set:
                    raster[4][r][c] = 1.0

                # Channel 5: candidate perimeter
                if (r, c) in candidates_set:
                    raster[5][r][c] = 1.0

                # Channel 6: observation age
                age = cell_obs.get("age_s")
                if age is not None and self.horizon_s > 0:
                    raster[6][r][c] = min(max(age / self.horizon_s, 0.0), 1.0)

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
            float(water_l) / float(CAPACITY_L),
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


class ActionMapper:
    """Bidirectional mapping between discrete action index and order dictionary.

    Action Index Mapping:
      - 0: {"type": "HOLD"}
      - 1: {"type": "RETURN_TO_REFILL"}
      - 2 + r * cols + c: {"type": "DROP", "target_m": [c * cell_size, r * cell_size]}

    Total discrete actions: 2 + (rows * cols).
    """

    def __init__(self, rows: int, cols: int, cell_size_m: float) -> None:
        self.rows = rows
        self.cols = cols
        self.cell_size_m = cell_size_m
        self.num_actions = 2 + rows * cols

    def to_order(self, action_index: int) -> dict:
        """Convert a discrete action index to an order dictionary."""
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
        mask = [False] * self.num_actions
        # HOLD is always valid
        mask[0] = True

        aircraft = observation["aircraft"]
        if aircraft["phase"] == "idle":
            if aircraft["water_l"] < CAPACITY_L:
                mask[1] = True
            if aircraft["water_l"] > 0:
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
        ) -> None:
            super().__init__()
            self.grid_rows = grid_rows
            self.grid_cols = grid_cols
            self.action_dim = 2 + grid_rows * grid_cols

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

            # Heads
            self.actor_head = nn.Linear(hidden_dim, self.action_dim)
            self.critic_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            raster: torch.Tensor,
            vector: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute action logits and state-value V(s).

            Args:
                raster: Tensor [B, C, H, W]
                vector: Tensor [B, V]

            Returns:
                logits: Tensor [B, action_dim]
                values: Tensor [B]
            """
            spatial_feat = self.spatial_cnn(raster)
            vector_feat = self.vector_mlp(vector)
            combined = torch.cat([spatial_feat, vector_feat], dim=-1)
            latent = self.fusion(combined)
            logits = self.actor_head(latent)
            values = self.critic_head(latent).squeeze(-1)
            return logits, values

        def get_action(
            self,
            raster: torch.Tensor,
            vector: torch.Tensor,
            action_mask: torch.Tensor,
            deterministic: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Sample or select action with invalid-action masking.

            Args:
                raster: [B, C, H, W]
                vector: [B, V]
                action_mask: BoolTensor [B, action_dim] where True indicates valid action
                deterministic: If True, selects argmax of valid logits

            Returns:
                actions: LongTensor [B]
                log_probs: FloatTensor [B]
                entropies: FloatTensor [B]
                values: FloatTensor [B]
            """
            logits, values = self.forward(raster, vector)
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
            return actions, log_probs, entropies, values

        def evaluate_actions(
            self,
            raster: torch.Tensor,
            vector: torch.Tensor,
            action_mask: torch.Tensor,
            actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for policy gradient updates.

            Args:
                raster: [B, C, H, W]
                vector: [B, V]
                action_mask: BoolTensor [B, action_dim]
                actions: LongTensor [B]

            Returns:
                log_probs: [B]
                entropies: [B]
                values: [B]
            """
            logits, values = self.forward(raster, vector)
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
            device: Any = None,
        ) -> int:
            """Convenience helper to select a single action index from an observation dict."""
            raster_t, vector_t = tensorizer.to_torch(observation, device=device)
            mask_t = mapper.to_mask_torch(observation, device=device)
            with torch.no_grad():
                actions, _, _, _ = self.get_action(raster_t, vector_t, mask_t, deterministic=deterministic)
            return int(actions.item())

else:
    class SpatialActorCritic:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for SpatialActorCritic. Install requirements-rl.txt.")
