"""Parameterised tactical actions: attack fire focus k at an aspect (D-61).

The nine fixed macro-actions (``TacticalActionMapper``) bake the target choice
into rules such as "head of the largest cluster", so trained policies collapsed
to one default rule (D-59, D-60). Here the policy chooses *which* focus and
*where* on it:

- index 0: ``HOLD``; index 1: ``RETURN_TO_REFILL``;
- index ``2 + 4 * k + a``: drop on focus slot ``k`` (0..MAX_SLOTS-1) at aspect
  ``a`` in (head, left flank, right flank, rear), relative to the observed wind.

With drop distances (D-79) each slot and aspect is repeated per distance ``d``
in cells: index ``2 + (4 * k + a) * D + j`` aims ``distances[j]`` cells beyond the
chosen front cell, away from the fire, so a drop can lead a fast front or land
on fuel ahead. The default ``(0,)`` is the D-61 action set.

Focus slots are fire fronts (D-67): the border cells, next to attackable
cells, of 8-connected groups of fire-reached cells, ordered by front length
(ties by centroid row then column). Extra fronts beyond ``MAX_SLOTS`` are not
addressable. Every candidate cell borders some front, so with at most
``MAX_SLOTS`` fronts the heuristic's targets are expressible as slot actions. Everything is computed from the controller
observation; ``slot_masks`` gives the tensorizer the matching cell masks.
"""

from __future__ import annotations

import math
from typing import Any

from .aircraft import tank_capacity_l
from .policies import (_is_burnable_fuel, _observed_burning, _observed_unburned_fuel,
                       candidate_cells, claimed_cells, full_cells)

MAX_SLOTS = 4
ASPECTS = ("HEAD", "FLANK_LEFT", "FLANK_RIGHT", "REAR")
NUM_ACTIONS = 2 + MAX_SLOTS * len(ASPECTS)
ACTION_NAMES = ("HOLD", "RETURN_TO_REFILL") + tuple(
    f"ATTACK_F{k}_{aspect}" for k in range(MAX_SLOTS) for aspect in ASPECTS)


def validate_distances(distances_cells: Any) -> tuple[int, ...]:
    """Distinct non-negative drop distances in cells, in the given order."""
    distances = tuple(int(d) for d in distances_cells)
    if not distances or len(set(distances)) != len(distances) or min(distances) < 0:
        raise ValueError(f"drop distances must be distinct non-negative cell counts, got {distances_cells}")
    return distances


def num_actions(distances_cells: Any = (0,)) -> int:
    return 2 + MAX_SLOTS * len(ASPECTS) * len(validate_distances(distances_cells))


def action_names(distances_cells: Any = (0,)) -> tuple[str, ...]:
    distances = validate_distances(distances_cells)
    if distances == (0,):
        return ACTION_NAMES
    return ("HOLD", "RETURN_TO_REFILL") + tuple(
        f"ATTACK_F{k}_{aspect}_D{d}" for k in range(MAX_SLOTS) for aspect in ASPECTS for d in distances)
_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def fire_slots(observation: dict, max_slots: int = MAX_SLOTS) -> list[list[tuple[int, int]]]:
    """Fire-front clusters, largest first, at most ``max_slots`` (D-67).

    Cells reached by fire (burning or burned) are grouped 8-connectedly; each
    group's slot is its front: the member cells 4-adjacent to an attackable
    candidate cell. Groups without an attackable border are dropped. Slots are
    ordered by front length, ties by centroid row then column.
    """
    cells = observation["fire"]["cells"]
    reached = {(r, c) for r, line in enumerate(cells) for c, cell in enumerate(line)
               if isinstance(cell, dict) and cell.get("fire_detected") is True}
    attackable = {tuple(cand) for cand in candidate_cells(observation)}
    seen: set[tuple[int, int]] = set()
    fronts = []
    for start in sorted(reached):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nxt = (r + dr, c + dc)
                    if nxt in reached and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        front = sorted(cell for cell in group
                       if any((cell[0] + dr, cell[1] + dc) in attackable for dr, dc in _FOUR))
        if front:
            fronts.append(front)

    def key(front: list[tuple[int, int]]) -> tuple:
        return (-len(front), sum(r for r, _ in front) / len(front), sum(c for _, c in front) / len(front))

    return sorted(fronts, key=key)[:max_slots]


def slot_masks(observation: dict, max_slots: int = MAX_SLOTS) -> list[list[list[float]]]:
    """[max_slots][rows][cols] 1.0 on the front cells of each slot (zeros if absent)."""
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
    """Maps parameterised actions (slot x aspect x drop distance) to simulator orders, with masking."""

    action_mode = "parametric"

    def __init__(self, cell_size_m: float, distances_cells: Any = (0,)) -> None:
        self.cell_size_m = float(cell_size_m)
        self.distances = validate_distances(distances_cells)
        self.action_names = action_names(self.distances)
        self.num_actions = num_actions(self.distances)

    def targets(self, observation: dict) -> dict[int, tuple[int, int]]:
        """Target cell of every currently valid drop action, keyed by action index."""
        return {i: (r, c) for i, (r, c, *_) in self.target_details(observation).items()}

    def target_details(self, observation: dict) -> dict[int, tuple[int, int, int, int, int]]:
        """For every currently valid drop action: (row, col, outward_row, outward_col, distance_cells).

        (row, col) is the target cell; outward is the unit step from the fire to the front cell
        (the direction a positive distance moves along).

        Distance 0 is the D-61 front cell for the slot and aspect. A positive
        distance steps that many cells from it away from the fire, along the
        edge joining it to the front; the cell must be in the grid, burnable,
        observed unburned, not already holding the maximum stacked drops (D-76)
        and not claimed by another aircraft.
        """
        clusters = fire_slots(observation)
        candidates = candidate_cells(observation)
        cells = observation["fire"]["cells"]
        fuel = observation["fuel_model"]
        rows, cols = len(cells), len(cells[0]) if cells else 0
        blocked = full_cells(observation) | claimed_cells(observation)
        east, north = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
        w_len = math.hypot(east, north)
        pos_x, pos_y = observation["aircraft"]["position_m"]
        per_slot = len(ASPECTS) * len(self.distances)
        out: dict[int, tuple[int, int]] = {}
        for k, cluster_cells in enumerate(clusters):
            cluster = set(cluster_cells)
            slot_targets = _slot_candidates(cluster_cells, candidates)
            if not slot_targets:
                continue

            def geometry(cand: list[int]) -> tuple[float, float, float, tuple[int, int]]:
                r, c = cand
                proj, cross, outward = [], [], []
                for dr, dc in _FOUR:
                    if (r + dr, c + dc) in cluster:
                        dx, dy = -dc * self.cell_size_m, -dr * self.cell_size_m  # outward from the fire
                        proj.append((dx * east + dy * north) / w_len if w_len > 1e-6 else 0.0)
                        cross.append((east * dy - north * dx) / w_len if w_len > 1e-6 else 0.0)
                        outward.append((-dr, -dc))
                best = max(range(len(proj)), key=proj.__getitem__)
                distance = math.hypot(c * self.cell_size_m - pos_x, r * self.cell_size_m - pos_y)
                return proj[best], cross[best], distance, outward[best]

            scored = [(geometry(cand), cand) for cand in slot_targets]
            for a, name in enumerate(ASPECTS):
                if name == "HEAD":
                    chosen = max(scored, key=lambda x: (x[0][0], -x[0][2], -x[1][0], -x[1][1]))
                elif name == "REAR":
                    chosen = min(scored, key=lambda x: (x[0][0], x[0][2], x[1][0], x[1][1]))
                elif name == "FLANK_LEFT":
                    chosen = max(scored, key=lambda x: (x[0][1], -x[0][2], -x[1][0], -x[1][1]))
                else:
                    chosen = min(scored, key=lambda x: (x[0][1], x[0][2], x[1][0], x[1][1]))
                (r0, c0), (orow, ocol) = chosen[1], chosen[0][3]
                for j, d in enumerate(self.distances):
                    r, c = r0 + d * orow, c0 + d * ocol
                    if d and not (0 <= r < rows and 0 <= c < cols and _is_burnable_fuel(fuel[r][c])
                                  and isinstance(cells[r][c], dict) and cells[r][c].get("fire_detected") is False
                                  and (r, c) not in blocked):
                        continue
                    out[2 + k * per_slot + a * len(self.distances) + j] = (int(r), int(c), int(orow), int(ocol), int(d))
        return out

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
        if aircraft["water_l"] < tank_capacity_l(aircraft):
            mask[1] = True
        for index in self.targets(observation):
            mask[index] = True
        return mask

    def to_mask_torch(self, observation: dict, device: Any = None) -> Any:
        import torch

        return torch.tensor(self.compute_mask(observation), dtype=torch.bool, device=device).unsqueeze(0)

    def target_cell(self, action_index: int, observation: dict) -> tuple[int, int]:
        """Row/column the drop action aims at; raises if the action has no valid target."""
        slot = (action_index - 2) // (len(ASPECTS) * len(self.distances))
        if not (0 <= slot < len(fire_slots(observation))):
            raise ValueError(f"focus slot {slot} does not exist")
        targets = self.targets(observation)
        if action_index not in targets:
            raise ValueError(f"action {action_index} has no valid target cell")
        return targets[action_index]

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


# D-104: any front cell. The fixed four cells per front (head, flanks, rear) left most of a large
# front out of reach (52 % of the fair rule's drops in fast, steep fires), and repeated drops on a
# watered head. Here every candidate cell is an action of its own, scored by the per-target head.
MAX_FRONTIER = 160
MAX_BURNING_FRONTIER = 320


def frontier_cells(observation: dict, ceiling: int | None = None) -> list[tuple[int, int, int, int]]:
    """Candidate cells in row-major order with the outward step from the fire:
    (row, col, outward_row, outward_col). The outward step is from the 4-neighbour reached cell the
    observed wind pushes hardest towards the target (diagonal neighbours when none is 4-adjacent).
    For a burning target the step instead points towards an observed unburned burnable 4-neighbour."""
    cells = observation["fire"]["cells"]
    reached = {(r, c) for r, line in enumerate(cells) for c, cell in enumerate(line)
               if isinstance(cell, dict) and cell.get("fire_detected") is True}
    burning_enabled = observation.get("water_on_burning", False)
    if ceiling is None:
        ceiling = MAX_BURNING_FRONTIER if burning_enabled else MAX_FRONTIER
    east, north = observation.get("midflame_wind_to_m_s", [0.0, 0.0])
    out = []
    for r, c in sorted(tuple(cand) for cand in candidate_cells(observation))[:ceiling]:
        if burning_enabled and _observed_burning(cells[r][c]):
            steps = [(dr, dc) for dr, dc in _FOUR
                     if _observed_unburned_fuel(cells, observation["fuel_model"], r + dr, c + dc)]
        else:
            steps = [(-dr, -dc) for dr, dc in _FOUR if (r + dr, c + dc) in reached] or [
                (-dr, -dc) for dr in (-1, 1) for dc in (-1, 1) if (r + dr, c + dc) in reached]
        steps = steps or [(0, 0)]
        orow, ocol = max(steps, key=lambda s: (s[1] * east + s[0] * north) / math.hypot(*s) if any(s) else 0.0)
        out.append((r, c, orow, ocol))
    return out


class FrontierActionMapper(ParametricActionMapper):
    """HOLD, RETURN_TO_REFILL, then one drop action per candidate cell (D-104): index ``2 + j`` drops
    on the j-th candidate in row-major order; unused indices are masked. Drops land on the cell itself."""

    action_mode = "frontier"

    def __init__(self, cell_size_m: float, distances_cells: Any = (0,), *, ceiling: int = MAX_FRONTIER) -> None:
        if tuple(distances_cells) != (0,):
            raise ValueError("frontier actions drop on the candidate cell itself (distance 0)")
        self.cell_size_m = float(cell_size_m)
        self.distances = (0,)
        self.ceiling = ceiling
        self.num_actions = 2 + ceiling
        self.action_names = ("HOLD", "RETURN_TO_REFILL") + tuple(f"DROP_C{j}" for j in range(ceiling))

    def target_details(self, observation: dict) -> dict:
        return {2 + j: (r, c, orow, ocol, 0)
                for j, (r, c, orow, ocol) in enumerate(frontier_cells(observation, self.ceiling))}

    def target_cell(self, action_index: int, observation: dict) -> tuple[int, int]:
        targets = self.targets(observation)
        if action_index not in targets:
            raise ValueError(f"action {action_index} has no valid target cell")
        return targets[action_index]
