"""Parameterised tactical actions: attack fire focus k at an aspect (D-61).

The nine fixed macro-actions (``TacticalActionMapper``) bake the target choice
into rules such as "head of the largest cluster", so trained policies collapsed
to one default rule (D-59, D-60). Here the policy chooses *which* focus and
*where* on it:

- index 0: ``HOLD``; index 1: ``RETURN_TO_REFILL``;
- index ``2 + 4 * k + a``: drop on focus slot ``k`` (0..MAX_SLOTS-1) at aspect
  ``a`` in (head, left flank, right flank, rear), relative to the observed wind.

Focus slots are 8-connected clusters of currently burning cells, ordered by
size (largest first, ties by centroid row then column), so the order is the
same for every aircraft. Extra clusters beyond ``MAX_SLOTS`` are not
addressable. A slot is valid only if it has unburned, untreated burnable cells
4-adjacent to its burning cells. Everything is computed from the controller
observation; ``slot_masks`` gives the tensorizer the matching cell masks.
"""

from __future__ import annotations

import math
from typing import Any

from .aircraft import CAPACITY_L
from .policies import candidate_cells

MAX_SLOTS = 4
ASPECTS = ("HEAD", "FLANK_LEFT", "FLANK_RIGHT", "REAR")
NUM_ACTIONS = 2 + MAX_SLOTS * len(ASPECTS)
ACTION_NAMES = ("HOLD", "RETURN_TO_REFILL") + tuple(
    f"ATTACK_F{k}_{aspect}" for k in range(MAX_SLOTS) for aspect in ASPECTS)
_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def fire_slots(observation: dict, max_slots: int = MAX_SLOTS) -> list[list[tuple[int, int]]]:
    """Burning-cell clusters (8-connected), largest first, at most ``max_slots``."""
    cells = observation["fire"]["cells"]
    burning = {(r, c) for r, line in enumerate(cells) for c, cell in enumerate(line)
               if isinstance(cell, dict) and cell.get("fire_state") == "burning"}
    seen: set[tuple[int, int]] = set()
    clusters = []
    for start in sorted(burning):
        if start in seen:
            continue
        stack, cluster = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            cluster.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nxt = (r + dr, c + dc)
                    if nxt in burning and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        clusters.append(sorted(cluster))

    def key(cluster: list[tuple[int, int]]) -> tuple:
        return (-len(cluster), sum(r for r, _ in cluster) / len(cluster), sum(c for _, c in cluster) / len(cluster))

    return sorted(clusters, key=key)[:max_slots]


def slot_masks(observation: dict, max_slots: int = MAX_SLOTS) -> list[list[list[float]]]:
    """[max_slots][rows][cols] 1.0 on the burning cells of each slot (zeros if absent)."""
    cells = observation["fire"]["cells"]
    rows, cols = len(cells), len(cells[0])
    masks = [[[0.0] * cols for _ in range(rows)] for _ in range(max_slots)]
    for k, cluster in enumerate(fire_slots(observation, max_slots)):
        for r, c in cluster:
            masks[k][r][c] = 1.0
    return masks


def _slot_candidates(cluster: list[tuple[int, int]], candidates: list[list[int]]) -> list[list[int]]:
    members = set(cluster)
    return [cand for cand in candidates if any((cand[0] + dr, cand[1] + dc) in members for dr, dc in _FOUR)]


class ParametricActionMapper:
    """Maps the 18 parameterised actions to simulator orders, with masking."""

    action_mode = "parametric"
    action_names = ACTION_NAMES
    num_actions = NUM_ACTIONS

    def __init__(self, cell_size_m: float) -> None:
        self.cell_size_m = float(cell_size_m)

    def compute_mask(self, observation: dict) -> list[bool]:
        mask = [False] * self.num_actions
        aircraft = observation["aircraft"]
        if aircraft["phase"] != "idle":
            mask[0] = True
            return mask
        if aircraft["water_l"] <= 0:
            mask[1] = True
            return mask
        mask[0] = True
        if aircraft["water_l"] < float(aircraft.get("capacity_l", CAPACITY_L)):
            mask[1] = True
        candidates = candidate_cells(observation)
        for k, cluster in enumerate(fire_slots(observation)):
            if _slot_candidates(cluster, candidates):
                for a in range(len(ASPECTS)):
                    mask[2 + k * len(ASPECTS) + a] = True
        return mask

    def to_mask_torch(self, observation: dict, device: Any = None) -> Any:
        import torch

        return torch.tensor(self.compute_mask(observation), dtype=torch.bool, device=device).unsqueeze(0)

    def target_cell(self, action_index: int, observation: dict) -> tuple[int, int]:
        """Row/column the drop action aims at; raises if the slot has no target."""
        slot, aspect = divmod(action_index - 2, len(ASPECTS))
        clusters = fire_slots(observation)
        if not (0 <= slot < len(clusters)):
            raise ValueError(f"focus slot {slot} does not exist")
        cluster = set(clusters[slot])
        targets = _slot_candidates(clusters[slot], candidate_cells(observation))
        if not targets:
            raise ValueError(f"focus slot {slot} has no attackable perimeter cell")
        east, north = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
        w_len = math.hypot(east, north)
        pos_x, pos_y = observation["aircraft"]["position_m"]

        def geometry(cand: list[int]) -> tuple[float, float, float]:
            r, c = cand
            proj, cross = [], []
            for dr, dc in _FOUR:
                if (r + dr, c + dc) in cluster:
                    dx, dy = -dc * self.cell_size_m, -dr * self.cell_size_m  # outward from the fire
                    proj.append((dx * east + dy * north) / w_len if w_len > 1e-6 else 0.0)
                    cross.append((east * dy - north * dx) / w_len if w_len > 1e-6 else 0.0)
            best = max(range(len(proj)), key=proj.__getitem__)
            distance = math.hypot(c * self.cell_size_m - pos_x, r * self.cell_size_m - pos_y)
            return proj[best], cross[best], distance

        scored = [(geometry(cand), cand) for cand in targets]
        name = ASPECTS[aspect]
        if name == "HEAD":
            chosen = max(scored, key=lambda x: (x[0][0], -x[0][2], -x[1][0], -x[1][1]))
        elif name == "REAR":
            chosen = min(scored, key=lambda x: (x[0][0], x[0][2], x[1][0], x[1][1]))
        elif name == "FLANK_LEFT":
            chosen = max(scored, key=lambda x: (x[0][1], -x[0][2], -x[1][0], -x[1][1]))
        else:
            chosen = min(scored, key=lambda x: (x[0][1], x[0][2], x[1][0], x[1][1]))
        return int(chosen[1][0]), int(chosen[1][1])

    def to_order(self, action_index: int, observation: dict | None = None) -> dict:
        if action_index == 0:
            return {"type": "HOLD"}
        if action_index == 1:
            return {"type": "RETURN_TO_REFILL"}
        if not (2 <= action_index < self.num_actions):
            raise IndexError(f"action_index {action_index} out of range [0, {self.num_actions - 1}]")
        if observation is None:
            raise ValueError("observation is required to resolve parameterised drops")
        r, c = self.target_cell(action_index, observation)
        return {"type": "DROP", "target_m": [c * self.cell_size_m, r * self.cell_size_m]}
