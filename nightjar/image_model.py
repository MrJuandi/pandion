"""Image-based actor-critic for geographic real-time observations (D-49).

Inputs come from ``policy_inputs.GeographicInputTensorizer``:

1. The RGB view is encoded by small 3x3 convolutions and then a convolution
   with kernel = stride = pixels per cell, giving one feature vector per cell.
2. These features are concatenated with the per-cell channels and processed by
   residual 3x3 blocks. Each block is FiLM-conditioned on the global vector
   (per-channel scale and shift), so spatial features depend on wind, weather
   and aircraft state from the first layer on.
3. Mean- and max-pooled features plus the vector embedding form a latent that
   optionally passes through an LSTM cell (recurrent memory, kept by owner
   decision to infer fire trends).
4. Heads: value from the latent; actions either as ``tactical`` macro-actions
   from the latent or ``spatial`` HOLD/REFILL logits plus one logit per cell
   from a 1x1 convolution over cell features conditioned on the latent. Cell
   logits are flattened row-major (index 2 + row * cols + col), matching
   ``ActionMapper`` in spatial mode.
"""
from __future__ import annotations

from typing import Any

from .model import TACTICAL_ACTION_NAMES
from .parametric_actions import ASPECTS, MAX_SLOTS, NUM_ACTIONS as PARAMETRIC_ACTIONS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without PyTorch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


if HAS_TORCH:

    class FiLMResidualBlock(nn.Module):
        def __init__(self, width: int, cond_dim: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(width, width, 3, padding=1)
            self.conv2 = nn.Conv2d(width, width, 3, padding=1)
            self.film = nn.Linear(cond_dim, 2 * width)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = F.leaky_relu(self.conv1(x), 0.01)
            h = self.conv2(h) * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            return F.leaky_relu(x + h, 0.01)

    class ImageActorCritic(nn.Module):
        """Actor-critic over (rgb, cells, vector) with FiLM conditioning and optional LSTM."""

        def __init__(
            self,
            *,
            cell_channels: int,
            vector_dim: int,
            px_per_cell: int,
            grid_rows: int,
            grid_cols: int,
            action_mode: str = "tactical",
            width: int = 32,
            num_blocks: int = 3,
            hidden_dim: int = 128,
            recurrent: bool = True,
            lstm_hidden_dim: int = 128,
            use_image: bool = True,
            image_channels: int = 3,
            slot_raw_features: bool = False,
        ) -> None:
            super().__init__()
            if action_mode not in ("tactical", "spatial", "parametric"):
                raise ValueError(f"Unknown action_mode: {action_mode}")
            self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
            self.action_mode = action_mode
            self.grid_rows, self.grid_cols = grid_rows, grid_cols
            self.px_per_cell = px_per_cell
            self.use_image = use_image
            self.recurrent = recurrent
            self.lstm_hidden_dim = lstm_hidden_dim
            self.action_dim = {"tactical": len(TACTICAL_ACTION_NAMES), "parametric": PARAMETRIC_ACTIONS}.get(
                action_mode, 2 + grid_rows * grid_cols)
            # Parametric mode: the last MAX_SLOTS cell channels are focus-slot masks.
            self.slot_channels = MAX_SLOTS if action_mode == "parametric" else 0
            # D-65: per-slot averages of the raw input channels (e.g. aircraft distance,
            # ROS, fuel) and slot area, alongside pooled trunk features.
            self.slot_raw_features = slot_raw_features
            self._raw_cells: torch.Tensor | None = None
            self._slot_masks: torch.Tensor | None = None

            if use_image:
                self.rgb_encoder = nn.Sequential(
                    nn.Conv2d(image_channels, 16, 3, padding=1), nn.LeakyReLU(0.01),
                    nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(0.01),
                    nn.Conv2d(16, width, kernel_size=px_per_cell, stride=px_per_cell), nn.LeakyReLU(0.01),
                )
            self.vector_mlp = nn.Sequential(
                nn.Linear(vector_dim, 64), nn.LeakyReLU(0.01),
                nn.Linear(64, 64), nn.LeakyReLU(0.01),
            )
            self.stem = nn.Conv2d((width if use_image else 0) + cell_channels, width, 3, padding=1)
            self.blocks = nn.ModuleList(FiLMResidualBlock(width, 64) for _ in range(num_blocks))
            self.fusion = nn.Sequential(nn.Linear(2 * width + 64, hidden_dim), nn.LeakyReLU(0.01))
            head_dim = lstm_hidden_dim if recurrent else hidden_dim
            if recurrent:
                self.lstm = nn.LSTMCell(hidden_dim, lstm_hidden_dim)
            self.critic_head = nn.Linear(head_dim, 1)
            if action_mode == "tactical":
                self.actor_head = nn.Linear(head_dim, self.action_dim)
            elif action_mode == "parametric":
                self.global_actor_head = nn.Linear(head_dim, 2)
                extra = (cell_channels + 1) if slot_raw_features else 0
                self.slot_actor_head = nn.Sequential(
                    nn.Linear(width + head_dim + 1 + extra, 64), nn.LeakyReLU(0.01), nn.Linear(64, len(ASPECTS)))
            else:
                self.global_actor_head = nn.Linear(head_dim, 2)
                self.cell_condition = nn.Linear(head_dim, width)
                self.cell_actor_head = nn.Conv2d(width, 1, kernel_size=1)

        def initial_state(self, batch_size: int = 1, device: Any = None) -> tuple[torch.Tensor, torch.Tensor] | None:
            if not self.recurrent:
                return None
            zeros = torch.zeros(batch_size, self.lstm_hidden_dim, device=device)
            return zeros, zeros.clone()

        # Frames per checkpointed encoder chunk during training; None disables
        # checkpointing. It saves memory but triples update time on MPS (D-64).
        IMAGE_CHUNK: int | None = None

        def _encode_image(self, rgb: torch.Tensor) -> torch.Tensor:
            """Encode the view, accepting uint8 (0-255) or float (0-1) pixels.

            With gradients enabled, chunks of ``IMAGE_CHUNK`` frames are
            checkpointed: full-resolution activations are recomputed during
            backward instead of stored. Gradients are unchanged; peak memory
            for a long episode drops from all frames to one chunk.
            """
            if rgb.dtype == torch.uint8:
                rgb = rgb.float() / 255.0
            if (self.IMAGE_CHUNK is None or not (torch.is_grad_enabled() and self.training)
                    or rgb.shape[0] <= self.IMAGE_CHUNK):
                return self.rgb_encoder(rgb)
            return torch.cat([checkpoint(self.rgb_encoder, part, use_reentrant=False)
                              for part in rgb.split(self.IMAGE_CHUNK)], dim=0)

        def _split(self, inputs: tuple) -> tuple:
            expected = 3 if self.use_image else 2
            if len(inputs) != expected:
                raise ValueError(f"expected {expected} input tensors, got {len(inputs)}")
            return inputs if self.use_image else (None,) + tuple(inputs)

        def encode(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Inputs (rgb, cells, vector), or (cells, vector) without image.

            Returns (cell features [B, width, H, W], latent [B, hidden_dim])."""
            rgb, cells, vector = self._split(inputs)
            if self.slot_channels:
                self._slot_masks = cells[:, -self.slot_channels:]
                self._raw_cells = cells
            stem_input = cells
            if self.use_image:
                image_features = self._encode_image(rgb)
                if image_features.shape[-2:] != cells.shape[-2:]:
                    raise ValueError("RGB view does not reduce to the cell grid; check pixels per cell")
                stem_input = torch.cat([image_features, cells], dim=1)
            cond = self.vector_mlp(vector)
            x = F.leaky_relu(self.stem(stem_input), 0.01)
            for block in self.blocks:
                x = block(x, cond)
            pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=-1)
            return x, self.fusion(torch.cat([pooled, cond], dim=-1))

        def _heads(self, features: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values = self.critic_head(h).squeeze(-1)
            if self.action_mode == "tactical":
                return self.actor_head(h), values
            if self.action_mode == "parametric":
                # Focus k's logits come from cell features pooled over its mask, plus the latent.
                masks = self._slot_masks
                area = masks.sum(dim=(2, 3))
                pooled = torch.einsum("bkhw,bchw->bkc", masks, features) / area.clamp(min=1.0)[..., None]
                present = (area > 0).float()[..., None]
                parts = [pooled, h[:, None, :].expand(-1, masks.shape[1], -1), present]
                if self.slot_raw_features:
                    raw = torch.einsum("bkhw,bchw->bkc", masks, self._raw_cells) / area.clamp(min=1.0)[..., None]
                    parts += [raw, (area / 100.0)[..., None]]
                slot_in = torch.cat(parts, dim=-1)
                slot_logits = self.slot_actor_head(slot_in).flatten(1)
                return torch.cat([self.global_actor_head(h), slot_logits], dim=-1), values
            conditioned = F.leaky_relu(features + self.cell_condition(h)[:, :, None, None], 0.01)
            cell_logits = self.cell_actor_head(conditioned).flatten(1)
            return torch.cat([self.global_actor_head(h), cell_logits], dim=-1), values

        def forward(self, *inputs: torch.Tensor,
                    state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple:
            features, latent = self.encode(*inputs)
            if not self.recurrent:
                return self._heads(features, latent)
            if state is None:
                state = self.initial_state(batch_size=latent.shape[0], device=latent.device)
            h, c = self.lstm(latent, state)
            logits, values = self._heads(features, h)
            return logits, values, (h, c)

        @staticmethod
        def _masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))

        def get_action(self, *inputs_and_mask: torch.Tensor, deterministic: bool = False,
                       state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple:
            """``get_action(*inputs, action_mask, ...)``; same contract as ``SpatialActorCritic``."""
            *inputs, action_mask = inputs_and_mask
            if self.recurrent:
                logits, values, next_state = self.forward(*inputs, state=state)
            else:
                logits, values = self.forward(*inputs)
            masked = self._masked(logits, action_mask)
            dist = Categorical(logits=masked)
            actions = torch.argmax(masked, dim=-1) if deterministic else dist.sample()
            result = (actions, dist.log_prob(actions), dist.entropy(), values)
            return result + (next_state,) if self.recurrent else result

        def evaluate_actions(self, *inputs_mask_actions: torch.Tensor,
                             initial_state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple:
            """``evaluate_actions(*inputs, action_mask, actions)``; recurrent inputs are one
            episode [T, ...] (BPTT). Returns log-probs, entropies, values."""
            *inputs, action_mask, actions = inputs_mask_actions
            features, latents = self.encode(*inputs)
            if not self.recurrent:
                logits, values = self._heads(features, latents)
            else:
                h, c = initial_state if initial_state is not None else self.initial_state(1, latents.device)
                hs = []
                for t in range(latents.shape[0]):
                    h, c = self.lstm(latents[t:t + 1], (h, c))
                    hs.append(h)
                logits, values = self._heads(features, torch.cat(hs, dim=0))
            dist = Categorical(logits=self._masked(logits, action_mask))
            return dist.log_prob(actions), dist.entropy(), values

else:

    class ImageActorCritic:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("ImageActorCritic requires PyTorch (pip install -r requirements-rl.txt)")
