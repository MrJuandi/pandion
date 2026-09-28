# System architecture and environment contract

**Status:** proposed system design with a synthetic integrated episode and decision interface, 2026-09-28. The FM1 one-size Rothermel/Albini subset, aircraft executor, delayed observation schedule, three simple baselines, and a reset/step interface run headlessly. A separate standard-fuel fire path is also available. Real sensor behavior, sourced water effects, and independent incident validation remain unimplemented.

The first surface-fire equation and CL-415 aircraft reference are documented in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md). FM1 grid coupling is implemented but not validated against incidents; suppression response remains open.

### Implemented synthetic aircraft contract (`schema_version: "0.1"`)

`python3 -m nightjar SCENARIO.json ORDERS.json [--output REPLAY.json]` runs one headless episode. The scenario requires `scenario_id`, integer `seed`, `provenance: "synthetic"`, `crs: "LOCAL_METERS"`, `horizon_s`, `decision_interval_s`, one `water_source_m: [x_m, y_m]`, and `aircraft` with `initial_position_m`, `initial_water_l`, `loaded_ground_speed_m_s`, `empty_ground_speed_m_s`, and `refill_time_s`. All durations and speeds are positive; positions are finite; initial water is between 0 and 6,137 L. See the executable example in `examples/`.

`ORDERS.json` is a list indexed by decision interval. Each item is `{"type":"HOLD"}`, `{"type":"DROP","target_m":[x,y]}`, or `{"type":"RETURN_TO_REFILL"}`. Missing entries mean `HOLD`. A busy aircraft rejects new missions; `HOLD` lets an existing mission continue. `DROP` travels to the target and releases the entire current load on arrival. `RETURN_TO_REFILL` travels to the one declared source and fills to capacity after the declared service time. Movement is straight line at the scenario's **synthetic ground-speed proxies**; wind, acceleration, approach, airspace, waterbody suitability, partial drops, fuel, and fire are absent. A drop has no effect on any fire state.

The replay contains decision-time observations (time, aircraft state, water-source coordinates), submitted orders, rejection reasons, timed execution events, final aircraft state, and flight time/water/rejection counts. It includes `replay_schema_version`, scenario ID and SHA-256 of the input JSON, seed, CRS, provenance, and units. There is no latent fire state in this component. The seed is recorded for the future stochastic environment; this executor makes no random draw. The aircraft state is considered directly observable in this synthetic slice; future sensor restrictions must be specified explicitly.

### Implemented unmitigated synthetic fire (`schema_version: "fire-synthetic-0.1"`)

`python3 -m nightjar.fire SCENARIO.json [--output REPLAY.json]` computes the first time fire reaches each cell centre on a local metre grid. Inputs are scenario ID, integer seed, synthetic provenance, local CRS, positive row/column counts, `cell_size_m`, `horizon_s`, nonnegative `base_ros_m_s` and finite `elevation_m` grids, `wind_to_m_s: [east, north]`, nonnegative `wind_coefficient_s_m` and `slope_coefficient`, and at least one burnable ignition cell `[row, col]`. Row numbers increase northward and column numbers eastward. A zero base rate is an impassable cell. The replay records arrival seconds (`null` for unreachable cells), affected area in hectares, and the scenario hash.

For each directed horizontal, vertical, or diagonal edge, let `d` be centre-to-centre distance, `u` its unit direction, and `s` signed elevation rise divided by `d`. The synthetic directional multiplier is `exp(k_w * dot(wind_to_m_s, u) + k_s * s)`. Travel time is `d/(2*R_source*multiplier) + d/(2*R_target*multiplier)`. A diagonal cannot pass between blocked orthogonal neighbours. A shortest-path calculation gives first arrival. The exponent is limited to `[-20, 20]` to reject extreme synthetic inputs. An ignition starts at its cell centre at time zero. A cell counts as affected when its centre arrival is at or before the horizon; each counted cell contributes its full area. There is no combustion duration, fuel consumption, reignition, spotting, crown fire, physical time step, or aircraft interaction. The displayed area is grid dependent and is **not** a validated burned-area estimate.

### FM1 Rothermel/Albini calculation (`schema_version: "rothermel-fm1-0.1"`)

`python3 -m nightjar.fire_rothermel SCENARIO.json [--output REPLAY.json]` uses a local-metre grid and arrival-time solver. The scenario requires ID, integer seed, synthetic provenance, local CRS, positive rows/columns, cell size and horizon, `fuel_model` (`FM1` or `NONE` per cell), `dead_fuel_moisture_fraction` in `[0,1]`, `elevation_m`, `midflame_wind_to_m_s: [east,north]`, and ignition cells. FM1 cells at or above 0.12 dead-fuel moisture do not spread; `NONE` blocks travel. The replay records first arrivals, area counted by reached cell centres, units, provenance, seed, and scenario hash. Example: `examples/fm1_fire.json`.

The local FM1 calculation implements the **one-size dead-fuel** equations in Andrews (2018), table 3, with the published FM1 parameters. Wind and slope factors form a head-spread vector; the section 6 ellipse yields rate in the edge direction. Each edge uses half its length at the source and destination rates. A shortest-path solver produces first arrivals. This edge coupling and the 8-neighbour discretization are Nightjar numerical choices. Elevation gradients use centered differences in the interior and one-sided differences at boundaries. Input wind is already at **midflame height**; no 10 m or 20 ft wind conversion is implied. There is no wind-speed cap, spotting, crown fire, fuel consumption, weather change, fire-atmosphere feedback, or suppression. This is a narrow published equation implementation, not a validated incident model.

`python3 -m nightjar.sensitivity [--output RESULT.json]` holds a 1,000 m centre-to-centre domain, flat uniform FM1, 6% dead moisture, 2 m/s eastward midflame wind, one central ignition, and a 600 s horizon fixed while testing 100, 50, 25, 10, and 5 m cells. It reports affected area and successive percent changes. This tests one homogeneous configuration only; it does not certify a grid size for other fuels or topography.

`python3 -m nightjar.sensitivity --varied` repeats 25, 10, and 5 m comparisons for seven uniform moisture, wind, and slope cases. It reports each scenario input hash and the coarse-to-fine area difference. This probes the FM1 numerical grid choice; it does not provide a time-varying spread model or incident calibration.

### Standard fuel-model calculation (`schema_version: "standard-fuel-0.1"`)

`.venv/bin/python -m nightjar.fire_standard SCENARIO.json [--output REPLAY.json]` requires the optional pinned Pyretechnics dependency. Its synthetic, local-metre scenario has ID, integer seed, positive grid dimensions/cell size/horizon, numeric `fuel_model` cells (0 or supported standard model number), finite `elevation_m`, six fixed moisture fractions under `fuel_moisture_fraction` (`dead_1h`, `dead_10h`, `dead_100h`, `dead_herbaceous`, `live_herbaceous`, `live_woody`), fixed `midflame_wind_to_m_s: [east,north]`, and burnable ignition cells. Moisture fractions are constrained to `[0,3]`. The replay has first arrivals, whole-cell affected area, units, ID, seed, scenario hash, and dependency version. The Pyretechnics local directional rates feed Nightjar's eight-neighbour half-edge shortest-arrival solver. The input grid is a fuel-model map, not a vegetation classification or a geographic loader. The standard-fuel fire path is separate from the aircraft/observation integration. Example: `examples/standard_fuel_fire.json`. The full scope and next evidence steps are in [STANDARD_FUEL_PATH.md](STANDARD_FUEL_PATH.md).

### Synthetic observation timing (`synthetic-fire-observation-0.1`)

`python3 -m nightjar.observation FIRE_REPLAY.json SCHEDULE.json --time-s T [--output VIEW.json]` consumes a precomputed true arrival-time map. Each scheduled report has `source_id`, `observed_at_s`, `available_at_s`, and explicit `coverage_cells`. Capture records whether fire had reached each covered cell at `observed_at_s`. A decision at `T` sees only reports with `available_at_s <= T`; each cell uses the latest delivered capture and carries its source, capture time, and age. Uncovered cells are `null`, not assumed unburned. An observation at a later capture time cannot leak through an earlier delivery. Tests change hidden cells and future arrivals without changing the current view.

This component deliberately uses exact synthetic fire presence in covered cells and a user-provided schedule. Real sensor false detections, omissions, cloud/smoke masks, geolocation uncertainty, and latency distributions are still unimplemented. The integrated episode uses the same capture/delivery functions; the standalone CLI still creates one view.

### Integrated synthetic episode (`integrated-synthetic-0.1`)

`python3 -m nightjar.integrated examples/synthetic_integrated.json --policy heuristic [--output REPLAY.json]` combines the versioned FM1 and aircraft scenarios with equal ID, seed, CRS, and horizon. `observation_schedule` entries use the existing capture/delivery times and either explicit `coverage_cells` or `coverage: "all"`; the latter expands to every grid cell. The example grid has 50 m cells, a 600 s horizon, 60 s decisions, an initial full 6,137 L load, and a full-coverage survey captured every 60 s (30 s delivery delay except the initial survey). The reference aircraft's speeds and refill time remain synthetic. Policy choices are `no_intervention`, `random_valid`, `heuristic`, and `containment_heuristic`; random draws use an explicit policy seed. Each decision observation includes the static water-source position so the containment policy can estimate refill transit from the same information available to every controller.

The fire solver processes ignition and neighbor arrivals in time order. At each decision, the policy receives only delivered fire reports, the static fuel map, public elevation grid, wind, its aircraft state, and cells covered by its own previous drops. A candidate is an observed-unburned FM1 cell orthogonally adjacent to an observed burning cell, excluding known treated cells. Busy aircraft hold; empty idle aircraft return to refill. The random policy samples uniformly from the same candidates; the heuristic prefers the largest source-to-target downwind projection, then shortest aircraft distance and coordinate order. A drop travels and consumes its full load through the existing executor. At arrival it creates a **permanent barrier** in not-yet-burning FM1 cells whose centres fall within `water_effect.radius_m` of the target; cells already reached remain reached. The example radius is 35 m, so a centre-targeted drop affects one 50 m cell. This is an explicitly unsourced software-test hypothesis, not a water deposition or fire-suppression law. Fire arrivals at the same timestamp are processed before a drop; a survey capture at that timestamp precedes the drop. The barrier has no decay, dose dependence, canopy interaction, or extinguishing action on burning cells.

The replay stores observations, candidate sets, orders, accepted flags, drop times and newly blocked cells, arrival times through the horizon, area, water and flight metrics, seed, units, and scenario hash. `python3 -m nightjar.integrated_benchmark examples/synthetic_integrated.json` runs all three baselines on the same inputs with ten fixed random-policy seeds and writes per-run area, difference from no intervention, drops, water, and flight time. This remains a diagnostic comparison.

### Exploratory step interface

`IntegratedTrainingEnv(scenario).reset(seed=None)` returns the first delivered observation. A supplied seed must match the scenario seed. `env.action_mask` lists valid order objects for that observation. `step(action)` accepts `HOLD`, `RETURN_TO_REFILL`, or `DROP` at a current candidate-cell centre, and returns `(next_observation, reward, terminated, info)`; `info["action_mask"]` describes the next decision. Invalid orders raise an error rather than silently becoming a different action. Each step advances one decision interval, and a terminal step reaches the scenario horizon. The returned terminal observation is the last decision observation; `info["time_s"]` gives the actual terminal time. A defensive copy of the full replay is available from `env.replay` only after termination. The implementation replays the stored action prefix deterministically on each transition; this prioritizes a transparent first contract over runtime efficiency.

The step reward is `-(A_t - A_{t-1})` hectares, with `A` the simulator's cumulative cell-count affected area at adjacent decision boundaries. Its sum is minus final area apart from area already reached at reset; it has no water-use or flight-time coefficient. True affected area is used only for reward and evaluation after the transition, never for action generation, and is absent from the returned observation and action mask. The interface is for a frozen synthetic experiment; the 35 m permanent barrier, fixed weather, exact covered-cell detections, and one scenario remain provisional.

`IntegratedTrainingEnv` provides `env.action_mapper()`, `env.observation_tensorizer()`, and `env.step_discrete(action_index)` from `nightjar.model`. The exploratory `nightjar.ppo` path collects rollouts from this interface, tensorizes only delivered observations, applies the environment's boolean action mask to `SpatialActorCritic`, and updates it with a small clipped PPO objective. Evaluation uses deterministic masked actions and reports simulator outcomes separately from reward. The discrete action space maps index 0 to `HOLD`, index 1 to `RETURN_TO_REFILL`, and indices $2 + r \cdot W + c$ to `DROP` at cell $(r, c)$. `ObservationTensorizer` extracts nine spatial raster channels and ten vector features strictly from delivered observations with zero latent leakage, and `SpatialActorCritic` provides a PyTorch Actor-Critic module with invalid-action masking. The two topographic channels are public normalized elevation and finite-difference slope magnitude clamped to [0, 1].

`python3 -m nightjar.scenario_family --output-dir runs/scenario-family` makes eight fixed variants from the committed integrated example. It retains the same 11×11 grid, 50 m cells, 600 s horizon, 60 s decisions, FM1 fuel, and observation schedule. Four base configurations vary ignition, constant wind direction, and initial aircraft position; each is paired across barrier radii of 35 and 55 m. Each scenario and component file carries its schema version, synthetic provenance, seed, local-metre CRS, units, and a declared synthetic UTC time origin; each replay carries those metadata and the scenario hash. The family runner writes a manifest before policy execution, then 96 complete policy replays and a descriptive report. Development and reserved labels do not imply independent incidents.

`python3 -m nightjar.challenge_family --output-dir runs/challenge-family` uses the same integrated solver, action contract, FM1 fuel model, and 35 m provisional water radius. It generates an 11×11 grid with seeded `NONE` cells, a seeded ignition and wind vector, a 1,200 s horizon, and full-grid synthetic surveys captured every 60 s with 30, 90, or 150 s delivery delay after the initial report. Four development indices were chosen after exploratory baseline inspection; four initially reserved indices use a separate seed stream. This command writes all eight scenario files and their hashes, but runs baselines only on development. It emits 48 full development replays (one no-intervention, one heuristic, ten seeded random choices per case). A prior exploratory controller trial opened the four initially reserved outcomes; they cannot be reused as untouched tests. The scenario files and replay carry synthetic provenance, schema, local CRS, units, seed, and a declared synthetic UTC origin.

### Browser replay tool (`nightjar.viewer`)

`python3 -m nightjar.viewer REPLAY1.json [REPLAY2.json ...] [--output runs/viewer.html]` generates a self-contained offline HTML5 interactive visualization. It animates fire spread arrival, delivered observation latency, candidate cell targets, water drops, and aircraft trajectory over the scenario horizon. When passed multiple replays for the same scenario, it supports synchronized side-by-side and tabbed multi-policy comparisons (e.g. comparing PPO against `containment_heuristic`).

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

Raster channels are numerical geospatial data, not an RGB screenshot. `nightjar.model.ObservationTensorizer` formats nine explicit raster channels (burnable fuel, fire detected, unburned observed, unobserved mask, known treated, candidate perimeter, observation age, normalized public elevation, and finite-difference slope magnitude) and ten non-spatial vector features (normalized aircraft position, water fraction, idle/busy state, water source position, reported wind, and decision time). A 3-layer CNN processes spatial inputs, a 2-layer MLP processes non-spatial features, and their fusion feeds an Actor-Critic architecture (`SpatialActorCritic`) with invalid-action masking over $2 + H \times W$ discrete actions. Any LSTM or other memory module must be tested against a feed-forward baseline.

### Action contract

The first controller chooses a mission from a finite, generated candidate set, for example `HOLD`, `DROP(candidate, pattern)`, and `RETURN_TO_REFILL`. The candidate generator may use only observable information. An action mask marks orders impossible because of water load, distance/endurance, airspace or mission state; an executed order records whether it succeeded and why. Travel, arrival, drop, and refill follow resource dynamics with elapsed time.

Candidate generation and masking are part of the environment specification: if they use hidden true state they can leak information even when the observation tensor does not.

### Transition and termination

Every decision advances the physical simulator and resource executor for the declared interval. Episodes end when the fire is contained under a defined criterion, when a fixed horizon is reached, or when an explicit terminal condition occurs. Containment must be operationally defined and tested against reignition or active residual cells.

## 4. Reward and outcomes

Define one primary outcome before tuning: for example, final affected area over a fixed horizon. Record protected-asset loss, containment time, water use, flight time, and unusable orders as secondary outcomes. Reward may include intermediate damage and resource costs to make learning feasible, but each term must have a stated unit, coefficient, and reason. A reward improvement alone is not proof of fire-management benefit.

Required checks: do-nothing has a plausible trajectory; extinguishing useful fire cannot be worse by construction; repeated waste is penalized; rewards cannot be farmed by hovering or cycling through orders; final metrics are independent of shaping. Reward weights are selected using training/validation scenarios only.

## 5. Policy progression

1. One aircraft: baseline references comprise `no_intervention`, valid uniform random choice, the greedy downwind heuristic, and the informed `containment_heuristic` (which incorporates downstream fuel exposure, bottleneck containment, and refill cycle distance). Future learned candidate models (e.g. CNN for raster channels, encoder for vector features, actor and critic, PPO) are evaluated against both heuristic references under the frozen evaluation criteria.
2. Two aircraft: identical environment contract with a variable roster; compare independent per-aircraft decisions, a central assignment controller, and an auction or dispatch heuristic. Measure duplicate work and resource imbalance.

3. Additional resource types: separate capability and execution models, shared mission vocabulary, type-specific valid actions. Consider hierarchical planning or centralized training with decentralized execution only if experiments justify it.

The decision layer should not encode assumptions such as “there are exactly two aircraft” in data layout or model weights. A fixed-size first implementation may pad/mask a declared maximum roster, while a later set/attention encoder can be evaluated for true variable-size generalization.

## 6. Interface boundary

Simulation, controller, benchmark, and replay must run headlessly. A browser interface can read versioned replay artifacts and show observed versus true state clearly for analysis. The interface must never be the source of scientific metrics. Desktop packaging is a later product decision.
