# System architecture and environment contract

**Status:** proposed design, 2026-09-26. This document specifies boundaries for implementation; numeric parameters remain to be calibrated or sourced.

## 1. Separation of concerns

1. **Scenario data:** georeferenced terrain, fuels or land cover, water sources, assets, and time-indexed weather. Every layer has provenance and a native resolution.
2. **True simulator state:** latent fire and fuel state, weather realization, completed interventions, and resource state. This is available to the simulator and evaluator, not automatically to a controller.
3. **Observation model:** publishes only information available by the current decision time, including source age, coverage, missingness, noise, and uncertainty. No future weather realization or true fire perimeter can leak into an observation.
4. **Decision controller:** selects missions or actions from the observation. Baselines and learned policies use the same interface and information budget.
5. **Resource execution:** translates an order into travel, work, delay, refill, and possible failure. The controller cannot alter the true state directly.
6. **Evaluation:** records outcomes from true state separately from reward and from what the controller observed.

These boundaries let the first controller operate one aircraft while preserving the same API for a variable roster of resources.

## 2. Time, space, and fidelity

The simulation step, decision interval, weather update frequency, observation update frequency, and replay frame rate are independent parameters. A five-minute command interval does not imply that satellite imagery updates every five minutes. Inputs at different resolutions must be reprojected to a declared coordinate reference system and resampled with methods appropriate to their data type.

The cell size must be chosen after considering both fire spread and the footprint of an aircraft drop. If a drop is narrower than a cell, the model must use a validated subcell representation or a finer grid in the affected area; it must not claim geometry that the grid cannot represent. Any accelerated or coarse model must preserve the outcome statistics needed for policy evaluation.

## 3. First environment: one aircraft

### Latent state

- Terrain and fuel properties; fuel moisture and their uncertainty-generating parameters.
- Fire activity, consumed fuel, heat or arrival state, and interventions.
- Weather realization and its temporal evolution.
- Aircraft position, heading, flight status, water/retardant load, fuel or endurance if modeled, assigned mission, and refill status.
- Water-source availability and protected assets.

### Controller observation

- Aligned raster channels derived from obtainable products: elevation/slope, land cover or estimated fuel, water, assets/population, most recent fire observations, observation coverage, and layer age/uncertainty.
- Complementary vector or token features: aircraft state, current reported weather, decision time, and mission status.
- Sensor-specific updates delivered at realistic intervals. A static background layer may be old; a fire detection has a timestamp and can be missing. The agent never receives the true fire map simply because the simulator computes it.

Raster channels are numerical geospatial data, not an RGB screenshot. RGB or multispectral imagery can be evaluated as an additional input when its source, date, preprocessing, and task value are clear. A CNN encodes spatial inputs; a separate encoder handles non-spatial features. Any LSTM or other memory module must be tested against a feed-forward baseline.

### Action contract

The first controller chooses a mission from a finite, generated candidate set, for example `HOLD`, `DROP(candidate, pattern)`, and `RETURN_TO_REFILL`. The candidate generator may use only observable information. An action mask marks orders impossible because of water load, distance/endurance, airspace or mission state; an executed order records whether it succeeded and why. Travel, arrival, drop, and refill follow resource dynamics with elapsed time.

Candidate generation and masking are part of the environment specification: if they use hidden true state they can leak information even when the observation tensor does not.

### Transition and termination

Every decision advances the physical simulator and resource executor for the declared interval. Episodes end when the fire is contained under a defined criterion, when a fixed horizon is reached, or when an explicit terminal condition occurs. Containment must be operationally defined and tested against reignition or active residual cells.

## 4. Reward and outcomes

Define one primary outcome before tuning: for example, final affected area over a fixed horizon. Record protected-asset loss, containment time, water use, flight time, and unusable orders as secondary outcomes. Reward may include intermediate damage and resource costs to make learning feasible, but each term must have a stated unit, coefficient, and reason. A reward improvement alone is not proof of fire-management benefit.

Required checks: do-nothing has a plausible trajectory; extinguishing useful fire cannot be worse by construction; repeated waste is penalized; rewards cannot be farmed by hovering or cycling through orders; final metrics are independent of shaping. Reward weights are selected using training/validation scenarios only.

## 5. Policy progression

1. One aircraft: CNN for raster channels, encoder for vector features, actor and critic, PPO as first learned baseline; compare with simple and informed heuristics.
2. Two aircraft: identical environment contract with a variable roster; compare independent per-aircraft decisions, a central assignment controller, and an auction or dispatch heuristic. Measure duplicate work and resource imbalance.
3. Additional resource types: separate capability and execution models, shared mission vocabulary, type-specific valid actions. Consider hierarchical planning or centralized training with decentralized execution only if experiments justify it.

The decision layer should not encode assumptions such as “there are exactly two aircraft” in data layout or model weights. A fixed-size first implementation may pad/mask a declared maximum roster, while a later set/attention encoder can be evaluated for true variable-size generalization.

## 6. Interface boundary

Simulation, controller, benchmark, and replay must run headlessly. A browser interface can read versioned replay artifacts and show observed versus true state clearly for analysis. The interface must never be the source of scientific metrics. Desktop packaging is a later product decision.
