# Pandion

Research project for learning and evaluating decisions in wildfire response. The long-term goal is a decision-making system that assigns missions to heterogeneous resources using the information available at the time of each decision. Development begins with one water-dropping aircraft, then two aircraft, before introducing other resource types.

**Status:** early Stage 0. The repository contains deterministic aircraft and fire components plus a small synthetic integrated episode. Its full-load drop blocks not-yet-burning grid cells under a provisional, unsourced water rule. Synthetic scenario families, a headless decision interface, and an exploratory PPO smoke path are available for evaluation design. This is not a physically validated or operational decision tool.

The [EMSR578 environmental pilot](docs/PILOT_ENVIRONMENT.md) now ingests scenario-bound numeric terrain and audits historical Baldomar station weather. Its exploratory spread comparisons report actual satellite elapsed times, newly burned area and an initial-only baseline. Station weather is not yet connected to time-dependent fire propagation; the physical-validation gate remains open. Acquisition, audit and simulation commands are documented in that pilot record.

## Research question

Can a learned policy choose where and when to deploy firefighting resources more effectively than transparent baselines when all controllers receive the same delayed and uncertain information?

The first experiment asks this question for **one aircraft**. The second introduces **two aircraft** to study assignment, coordination, duplicate work, and competition for water or airspace. Later experiments may add reconnaissance aircraft and ground crews. Each expansion requires a separate evaluation.

## Design principles

- **Physical state and observed state are separate.** The simulator tracks the best justified approximation of the fire and resources. A controller sees only information that its specified sensors and data feeds could have delivered by that time.
- **Spatial data has provenance.** Every terrain, land-cover, weather, population, and fire-observation layer records its source, resolution, timestamp, transformation, and uncertainty. A real elevation layer does not make every other layer real.
- **Actions have operational consequences.** Orders consume time and resources. Aircraft travel, drop, return, and refill; they do not teleport.
- **Results are reproducible.** Scenarios, seeds, simulator version, controller version, training settings, and evaluation splits are recorded. Checkpoints are selected on validation data; the reserved test is not used for tuning.
- **Claims follow evidence.** A policy can outperform a baseline in a simulator without being safe or effective in a real incident. Failures and worst cases are reported alongside averages.

## Planned system

```text
Data sources ──> scenario builder ──> simulator true state
                    │                       │
                    └──> observation model <─┘
                              │
                    controller or baseline
                              │
                    mission and action order
                              │
                      resource dynamics
                              │
                     evaluation and replay
```

The initial policy is expected to encode aligned raster layers with a CNN, encode aircraft and weather features separately, then combine them in an actor–critic network. Recurrence will be tested when delayed observations make memory useful; it is not a prerequisite for the first experiment. The eventual command layer will operate on variable sets of resources and missions. See [system architecture](docs/ARCHITECTURE.md).

## Development stages

| Stage | Question | Completion evidence |
|---|---|---|
| 0 — foundations | Are scenario inputs, fire dynamics, observation timing, aircraft movement, and metrics defined and testable? | Versioned scenario contract; unit and integration checks; simulator calibration report. |
| 1 — one aircraft | Can a policy improve decisions over fair baselines? | Held-out comparison with paired scenarios, multiple training seeds, confidence intervals, and failure cases. |
| 2 — two aircraft | Does the policy coordinate resources better than independent or rule-based assignment? | Assignment and duplication metrics, resource-use metrics, and held-out comparison. |
| 3 — heterogeneous response | What changes when sensing and ground interventions are available? | Resource ablations and evaluation under partial observation and delayed data. |

These are research gates, not claims that any stage is complete. The detailed plan is in [ROADMAP.md](docs/ROADMAP.md).

## Documentation

- [RESEARCH_WORKFLOWS.md](docs/RESEARCH_WORKFLOWS.md) — reproducible scenario seeds, current checkpoint inference, multi-aircraft replay, searchable artifact catalog and incident-group report audits.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — state, observation, actions, reward, policy, and growth to multiple resources.
- [PHYSICAL_MODEL_V0.md](docs/PHYSICAL_MODEL_V0.md) — sourced surface-fire and CL-415 references, implemented FM1 subset, and remaining validation limits.
- [STANDARD_FUEL_PATH.md](docs/STANDARD_FUEL_PATH.md) — standard-fuel scenario, reproduction command, boundaries, and ordered steps toward real maps.
- [DATA_AND_VALIDATION.md](docs/DATA_AND_VALIDATION.md) — candidate data, provenance, calibration, and validation requirements.
- [VALIDATION_EMSR578.md](docs/VALIDATION_EMSR578.md) — audited independent burned-area snapshots and the reasons they cannot yet validate FM1 spread.
- [VALIDATION_ANNABURROO.md](docs/VALIDATION_ANNABURROO.md) — no-fit comparison of FM1 head rate with experimental grass fires.
- [TRAINING_READINESS.md](docs/TRAINING_READINESS.md) — evidence gate for physical claims and the separate scope of exploratory synthetic training.
- [TRAINING_PERFORMANCE.md](docs/TRAINING_PERFORMANCE.md) — measured training costs, acceleration alternatives, and the quality checks needed before adopting them.
- [EVALUATION.md](docs/EVALUATION.md) — baselines, splits, metrics, statistics, and failure analysis.
- [ROADMAP.md](docs/ROADMAP.md) — milestones and acceptance criteria.
- [DECISIONS.md](docs/DECISIONS.md) — settled choices and open decisions.
- [REFERENCES.md](docs/REFERENCES.md) — related research and official data sources.
- [PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) — checks before creating and publishing the GitHub repository.
- [MAINTENANCE.md](docs/MAINTENANCE.md) — documentation ownership and change review rules.
- [site/](site/index.html) — static project page. Serve the repository root with `python3 -m http.server` and open `/site/`. Its replay is copied, not recomputed, from a built workspace by `python3 site/extract_episode.py <workspace>/index.html`, which writes `site/data/`.

## Development and use

The synthetic aircraft episode uses Python 3 and the standard library; no installation step is needed from the repository root:

```sh
mkdir -p runs
python3 -m pandion examples/synthetic_aircraft.json examples/synthetic_orders.json --output runs/aircraft-replay.json
python3 -m pandion.fire examples/synthetic_fire.json --output runs/fire-replay.json
python3 -m pandion.fire_rothermel examples/fm1_fire.json --output runs/fm1-fire-replay.json
python3 -m pandion.observation runs/fm1-fire-replay.json examples/synthetic_observation_schedule.json --time-s 400 --output runs/observation-400s.json
python3 -m pandion.sensitivity --output runs/fm1-resolution.json
python3 -m pandion.sensitivity --varied --output runs/fm1-varied-resolution.json
python3 -m pandion.integrated examples/synthetic_integrated.json --policy heuristic --output runs/integrated-heuristic.json
python3 -m pandion.integrated_benchmark examples/synthetic_integrated.json --output runs/integrated-comparison.json
python3 -m pandion.scenario_family --output-dir runs/scenario-family
python3 -m pandion.challenge_family --output-dir runs/challenge-family
python3 -m pandion.discrimination_family --output-dir runs/discrimination-family
python3 -m pandion.cems_catalog list
python3 -m pandion.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-real-v2
# Offline fixtures must be requested explicitly and are labeled synthetic:
python3 -m pandion.geographic_family --synthetic --output-dir runs/geographic-family-synthetic
python3 -m pandion.geographic_integrated runs/geographic-family-real-v2/development/geographic-dev-emsr578-00.json --policy heuristic --output runs/geographic-family-real-v2/smoke-development-00.json
python3 -m pandion.geographic_benchmark runs/geographic-family-real-v2/development/geographic-dev-emsr578-00.json --output runs/geographic-family-real-v2/benchmark-development-00.json
# Sensitivity of reference-policy results to the CL-415 water-effect parameters (D-74, D-76, D-78); development scenarios only, the output path must not exist:
python3 -m pandion.water_sensitivity runs/geographic-multi-ignition-family-d73-20261006/development --output runs/water-sensitivity.json --scenarios 64 --seed 7 --workers 6
python3 -m pandion.geographic_ppo --smoke --scenarios-dir runs/geographic-family-real-v2
# Offline PPO checks require the synthetic family generated above and explicit mode:
python3 -m pandion.geographic_ppo --smoke --synthetic --scenarios-dir runs/geographic-family-synthetic
python3 -m pandion.evaluate_reserved --split-version v2
python3 -m pandion.viewer runs/integrated-heuristic.json --output runs/heuristic-viewer.html
python3 -m unittest discover -s tests -v
```

An optional synthetic path uses [Pyretechnics](https://github.com/pyregence/pyretechnics) and numbered standard fire-behavior fuel models. Install its pinned dependency in a virtual environment, then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-standard.txt
.venv/bin/python -m pandion.fire_standard examples/standard_fuel_fire.json --output runs/standard-fuel-fire.json
```

The example mixes GR1 (101), SH1 (141), and nonburnable (0) cells. It is a synthetic, no-intervention spread example with hand-assigned fuels; geographic land-cover mapping is implemented separately in `pandion.cems_catalog`. Its replay does not feed the aircraft episode. See [STANDARD_FUEL_PATH.md](docs/STANDARD_FUEL_PATH.md) for the full contract, evidence, and next steps.

The same unit suite runs in GitHub Actions for pushes and pull requests. The `unit` job checks the standard-fuel path; `unit-rl` also installs CPU PyTorch and requires its import before running the suite. Tests that need a downloaded external source skip when the audited file is absent.

`runs/` is ignored by Git. Omit `--output` to print a result to standard output for commands that accept it. The aircraft example exercises movement, a full-water drop, return, and refill. The fire examples produce first-arrival times and grid-cell affected area without intervention. The resolution command compares the FM1 assumptions on one physical domain at several cell sizes. Scenario contracts and limitations are in [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [PHYSICAL_MODEL_V0.md](docs/PHYSICAL_MODEL_V0.md). `pandion.viewer` bundles one or more replay files (add `--imagery DIR`, repeatable, to drape the pre-fire Sentinel-2 base image of matching grids) into the self-contained offline 3D viewer described under [Explore the geographic 3D workspace](#explore-the-geographic-3d-workspace).

The integrated example compares no intervention, a seeded valid random choice, and a deterministic downwind heuristic over the same synthetic scenario. At its 600 s horizon, the example gives 2.00 ha without intervention, 0.75 ha with the heuristic, and 1.30 ha mean over ten random-policy seeds. These figures depend on the assumed permanent one-cell water barrier and do not estimate real suppression effectiveness; see [EVALUATION.md](docs/EVALUATION.md).

### Explore the geographic 3D workspace

Build the offline interface with real heights in the simulation and historical station conditions for Artesa:

```sh
# Requires the four verified local CEMS ZIPs under data/raw/cems.
.venv/bin/python -m pandion.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-source-corrected-20261001
.venv/bin/python -m pip install -r requirements-terrain.txt
.venv/bin/python -m pandion.terrain_acquire --scenarios-dir runs/geographic-family-source-corrected-20261001 --output-dir data/raw/geographic-environment-source-corrected-20261001
.venv/bin/python -m pandion.environment_family --scenarios-dir runs/geographic-family-source-corrected-20261001 --terrain-dir data/raw/geographic-environment-source-corrected-20261001 --weather-audit runs/pilot-environment-20260930/weather.json --pilot-scenario runs/emsr578-pilot-scenario.json --output-dir runs/geographic-environment-family-20261001
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.workspace --scenarios-dir runs/geographic-environment-family-20261001 --output-dir runs/environment-workspace
```

Every output directory must be new or empty. The weather audit and original pilot are prepared with the commands in [PILOT_ENVIRONMENT.md](docs/PILOT_ENVIRONMENT.md). Omit both `--weather-audit` and `--pilot-scenario` to use clearly labeled synthetic constant weather with real heights. Acquisition uses IGN/CNIG terrain in Spain and Copernicus GLO-30 surface heights for France/Italy; it needs network access, `curl`, and optional Rasterio for the latter. The derived family records hashes, CRS, source and resolution and rejects missing heights. Its grid remains 100 m; source resolution is 25/30 m.

Pre-fire Sentinel-2 base images for the planned image-based policy input (D-50) are acquired per activation grid with:

```sh
.venv/bin/python -m pandion.imagery_acquire runs/geographic-environment-family-20261001 data/raw/geographic-imagery-s2
```

It needs network access and Rasterio (`requirements-terrain.txt`); Pillow is optional and only writes north-up previews. See [ENVIRONMENT_INTEGRATION.md](docs/ENVIRONMENT_INTEGRATION.md#pre-fire-sentinel-2-base-imagery-d-50) for scene screening, compositing and limits.

Train the D-49 image policy (Sentinel-2 view plus per-cell layers and a global vector, D-52) on the same development split; `--action-mode spatial` gives one drop action per cell and `--ros` adds the Rothermel rate-of-spread channel:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.geographic_ppo --smoke --seed 42 --scenarios-dir runs/geographic-environment-family-20261001 --input-mode image --imagery-dir data/raw/geographic-imagery-s2
```

On Apple silicon, `--device mps` (or `--device auto`) runs the network on the GPU, about ten times faster for image inputs (D-54). `--ppo-epochs 4 --minibatches 4 --gae-lambda 0.95 --reward-scaling --target-kl 0.02` enables the D-55 PPO settings; the defaults keep the original single-step update. For throughput (D-64) add `--skip-forced --rollout-workers 8` and use 4 px per cell imagery from `python -m pandion.imagery_acquire IMAGERY_DIR OUT_DIR --downsample 2`; one two-aircraft image update then takes about 4 s instead of 34 s. For long runs, keep macOS from throttling background work with `caffeinate -i` and save intermediate weights with `--checkpoint-every N`, for example `caffeinate -i .venv/bin/python -m pandion.geographic_ppo --select ... --device auto --checkpoint runs/x/model.pt --checkpoint-every 10`.

`python3 -m pandion.geographic_integrated SCENARIO.json --policy hold_aware_heuristic` runs the third reference policy (D-83): the downwind heuristic that skips cells whose water already holds the front and prefers fronts a drop can hold. `pandion.geographic_benchmark` and the PPO holdout and evaluation reports include it.

With `--select`, `--selection-activations EMSR583 EMSR602_AOI04` leaves those development incidents out of training and keeps the best weights by their scenarios instead of by the validation split (D-84); they must differ from `--holdout-activation`.

`--vine-states 4 --vine-actions 4` trains on branched advantages (D-85): at four decisions of every training episode, four actions are sampled from the policy and each is followed to the end of the episode with common random numbers, and the policy term uses those branch returns instead of Monte Carlo advantages. It needs `--gamma 1.0` (the default), image or cells inputs and `--rollout-envs 1`; `--rollout-workers N` runs the branches in N processes.

With parametric actions, `--reward-reference heuristic` trains on hectares saved relative to the heuristic in the same scenario (D-81; evaluation and selection still score final burned area), and `--action-features` lets the head see, for each drop action, the cell it would drop on, including how many stacked drops the front reaching it needs (D-82). Both are off by default, and both work with `--drop-distances` and with either collection mode.

`--rollout-envs 8 --rollout-workers 4` offers a second collection mode (D-71): up to eight independent policy decisions run together on the parent GPU, while four persistent CPU processes prepare inputs and step simulators. With `--rollout-envs 1`, the existing CPU model-copy worker mode above is retained. Defaults remain serial (`--rollout-envs 1 --rollout-workers 1`); zero workers uses local simulation. Keep `--episodes-per-update` at the intended training budget, since batching does not multiply it. This mode preserves the configured imagery, actions, recurrent memory and PPO settings and supports `--skip-forced`, augmentation, slot features and periodic best-checkpoint selection. Reports record collection settings; logs separate collection from reward-target/PPO update time. Batched sampling changes the seeded random sequence. The initial implementation measured 35% less time over five updates against `8623ec1`; that historical ratio is not a speed estimate against the newer D-64–D-68 code. See [TRAINING_PERFORMANCE.md](docs/TRAINING_PERFORMANCE.md) for source-specific evidence, limits and an exercised command.

Open `runs/environment-workspace/index.html` in a browser, or serve the folder locally (for example `python3 -m http.server 8000 --directory runs/environment-workspace`). This workspace uses the 3D viewer in `pandion.viewer` (`python3 -m pandion.viewer REPLAY.json --output runs/viewer.html` bundles individual replays). The self-contained Spanish workspace follows the layout of the local FireLab viewer: a full-screen interactive Three.js scene, a header with scenario and policy selectors, a bottom timeline with playback, speed and a strip of the primary episode's reached area A(t) behind the decision and drop ticks, and an analysis sidebar. **Mapa 3D**, **Comparación**, **Condiciones** and **Datos y modelos** are separate views that preserve the selected scenario and elapsed time.

The scene draws the DEM as a lit, shadowed mesh on a base with side walls and a 1 km reference floor that extends beyond the modelled grid, so recorded refill positions outside the crop remain visible. The terrain texture combines fuel, hillshade and hypsometric tint; **Color** switches to altitude or slope with a numeric key. When the bundle includes D-50 imagery (`python3 -m pandion.viewer REPLAY.json --imagery runs/geographic-imagery-s2-20261002`, or `--imagery-dir` in `pandion.workspace`), **Imagen Sentinel-2** is the default colour: the pre-fire true-colour composite of that grid with a light hillshade, used only for display. A replay receives an image only when a manifest records exactly its grid size, cell size, origin and CRS, and the array hash must match; the map shows the Copernicus attribution and **Datos y modelos** the acquisition date, scene count and hash. At the selected time it shows burned cells as a scar tinted over the terrain (warm char for cells reached in the last 10 minutes, ash afterwards, so relief and contours stay visible), the active front as a glowing line along the boundary between front cells and burnable unreached neighbours, delivered detections as dashed cell-union outlines, and drop-treated cells. The active front is derived from the arrival raster: reached cells with at least one burnable 4-neighbour not yet reached. **Estado** switches to simulator truth, the delivered observation only, or historical cartography. Equal-elevation contours, labels, a cell grid and an illustrative smoke layer can be toggled in **Simbología y capas**. The smoke is a visual effect and is not simulated.

The aircraft is a stylised amphibious tanker (CL-415 reference; the recorded tank holds 6,137 L) that follows the recorded horizontal track, interpolated linearly between samples. Its trail (brightest on the most recent leg, earlier legs as a faint trace), nadir marker and label (phase and tank level from the recorded decisions) are drawn with it, and a short water curtain plays after each recorded drop. Aircraft altitude and symbol size are display choices, not replay data. The instrument rail on the right of the map offers **3D Táctica**, **2D Orto** (orthographic, north up), **Persecución** (follows the aircraft) and **Frente** (centres the active front). Drag to orbit, right-drag or Shift-drag to pan, wheel to zoom, double-click to focus a point; with a map focused, arrow keys/WASD pan, +/− zoom, Home resets and Space toggles playback. **Relieve** sets vertical exaggeration (1×, 1.5×, 2×, 3×; default 1.5×); the active factor is always shown in the terrain panel and the physics uses unexaggerated heights. The display surface interpolates cell-corner heights (Catmull-Rom) without adding physical resolution. The HUD shows UTC time, decision step, camera pitch/azimuth/altitude, a compass with the simulation wind and a scale bar valid at the view centre. When WebGL is unavailable, an explicit message appears and a 2D compatibility map remains available. **Exportar PNG** saves the current view.

The sidebar separates the live state at the selected time (reached area, active-front area, water released so far, aircraft phase and tank) from final episode totals. **Dinámica de crecimiento A(t)** plots reached area from each policy's arrival raster with the current time marked; **Perfil topográfico** samples elevation along the wind direction through the front, shading reached segments. The decision inspector shows the latest decision issued at or before the selected instant and its delivered detections; the decision list moves the clock to any decision. The cell tooltip reports coordinates, fuel, elevation and slope; observation mode does not expose simulator arrival times. Initial burned area is a cumulative footprint, not active flames.

**Condiciones** includes the timeline and follows the selected episode time: elevation range, UTC time, wind, temperature, relative humidity, interval rain and estimated dead-fuel moisture, plus weather already delivered to the controller. **Comparación** shows exactly two selected policies from the same scenario side by side, with a shared clock, a synchronized camera, a two-row results table and both curves in A(t); shorter episodes retain their completed state. **Política** selects the primary episode and **Comparar con** a distinct reference. Clicking a panel title swaps the two roles. Scenarios without multiple episodes show an explicit unavailable-comparison message. **Datos y modelos** records source, assumptions, training provenance and the display conventions above. The interface remains an offline replay inspector; it adds no simulation or policy capabilities. [Three.js r128, its OrbitControls and the MIT license](pandion/viewer_assets/vendor/README.md) are embedded in the generated HTML, with no runtime CDN dependency.

The default runs three baselines on the first two development cases, showing the remaining cases as static maps. `--episodes 0` produces cartography only. The 24 cases represent four incidents, not 24 independent fires: Artesa (EMSR578), Ladrillar/Las Hurdes (EMSR590), Landiras (EMSR592, France), and Massarosa (EMSR600, Italy). Validation/reserved cases never run episodes through this command. The 2026-10-01 source audit corrected the previous incident names, acquisition times and Massarosa CRS. Regenerate stale `geographic-family-real-v2` outputs; they are rejected by current provenance checks.

**The new environment family uses sourced elevations in Pyretechnics slope and public PPO terrain channels.** Artesa uses time-dependent Baldomar 2 m station wind as an explicitly uncalibrated, spatially uniform midflame proxy. Temperature/RH drive a Simard equilibrium plus 1/10/100 h moisture-lag estimate; initial/live moisture remains modeled. The dry-interval mode rejects rain instead of inventing wetting. Other incidents retain synthetic weather. Half-hour aggregates reach the controller only at interval end; source delivery latency is unknown and assumed zero. These assumptions are exploratory, not a reconstruction or physical validation. See [ENVIRONMENT_INTEGRATION.md](docs/ENVIRONMENT_INTEGRATION.md).

To train fresh weights on this changed environment and show them alongside baselines:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.geographic_ppo --train --updates 6 --episodes-per-update 1 --scenarios-dir runs/geographic-environment-family-20261001 --seed 42 --checkpoint runs/geographic-environment-ppo-20261001/model.pt --output runs/geographic-environment-ppo-20261001/train.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.workspace --scenarios-dir runs/geographic-environment-family-20261001 --output-dir runs/environment-workspace-with-ppo --checkpoint runs/geographic-environment-ppo-20261001/model.pt --training-report runs/geographic-environment-ppo-20261001/train.json
```

Keep weights with their matching report. Manifest-bound development hashes must match; old flat-terrain checkpoints cannot be represented as trained on the new family. Weights do not embed report identity or optimizer state. The optional `--terrain` + `--terrain-scenario` display overlay remains for legacy flat episodes and full pilot cartography; it is unnecessary for the environment family. The full pilot is historical cartography without an episode. Training is development-only and exploratory; the number of updates does not demonstrate skill or geographic generalization.

### Initial-attack scenarios and additional development incidents

Small ignitions inside the CEMS burned area replace the delineation as the initial state (D-56), and five more Spanish delineation products extend development only (D-57). Validation and reserved incidents are unchanged.

```sh
# Requires the five additional verified packages in data/raw/cems (emsr583_del.zip, emsr625-aoi01_del.zip, emsr625-aoi02_del.zip, emsr602-aoi02_del.zip, emsr602-aoi04_del.zip).
.venv/bin/python -m pandion.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-expanded-20261002 --extra-development all
.venv/bin/python -m pandion.terrain_acquire --scenarios-dir runs/geographic-family-expanded-20261002 --output-dir data/raw/geographic-environment-new-incidents-20261002 --activations EMSR583 EMSR625_AOI01 EMSR625_AOI02 EMSR602_AOI02 EMSR602_AOI04
# Combine with the earlier four terrain files and sources.json entries into data/raw/geographic-environment-expanded-20261002, then:
.venv/bin/python -m pandion.environment_family --scenarios-dir runs/geographic-family-expanded-20261002 --terrain-dir data/raw/geographic-environment-expanded-20261002 --weather-audit runs/pilot-environment-20260930/weather.json --pilot-scenario runs/emsr578-pilot-scenario.json --output-dir runs/geographic-environment-family-expanded-20261002
.venv/bin/python -m pandion.ignition_family --scenarios-dir runs/geographic-environment-family-expanded-20261002 --output-dir runs/geographic-ignition-family-expanded-20261002 --development-ignitions 4
```

D-73 adds fourteen more Spanish delineation products to development (EMSR365, EMSR375, EMSR457, EMSR458, EMSR522, EMSR538, EMSR545, EMSR581, EMSR597, EMSR599 AOI01/AOI02, EMSR620, EMSR623, EMSR632), giving 20 development incidents. Fetch each with `python -m pandion.cems_catalog fetch --activation ID`, then:

```sh
.venv/bin/python -m pandion.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-d71-20261006 --extra-development all
.venv/bin/python -m pandion.terrain_acquire --scenarios-dir runs/geographic-family-d71-20261006 --output-dir data/raw/geographic-environment-d71-new-incidents-20261006 --activations EMSR365 EMSR375 EMSR457 EMSR458 EMSR522 EMSR538 EMSR545 EMSR581 EMSR597 EMSR599_AOI01 EMSR599_AOI02 EMSR620 EMSR623 EMSR632
# Combine with data/raw/geographic-environment-expanded-20261002 into data/raw/geographic-environment-d71-20261006, then:
.venv/bin/python -m pandion.environment_family --scenarios-dir runs/geographic-family-d71-20261006 --terrain-dir data/raw/geographic-environment-d71-20261006 --weather-audit runs/pilot-environment-20260930/weather.json --pilot-scenario runs/emsr578-pilot-scenario.json --output-dir runs/geographic-environment-family-d71-20261006
.venv/bin/python -m pandion.multi_ignition_family --scenarios-dir runs/geographic-environment-family-d71-20261006 --output-dir runs/geographic-multi-ignition-family-d71-20261006
# Imagery for the new incidents only (--activations), merged with the earlier 8 px directory, then downsampled to 4 px:
.venv/bin/python -m pandion.imagery_acquire runs/geographic-multi-ignition-family-d71-20261006 runs/geographic-imagery-s2-d71-new-20261006 --activations EMSR365 ...
```

D-74 changes the physics and the scenario mix: a drop treats one cell that holds only fronts up to 1,000 kW/m, fuel moisture follows the Scott and Burgan D1L1–D4L4 scenarios and multi-focus ignitions are single cells (family 0.2). Regenerate the chain above (same terrain and imagery) into `runs/geographic-family-d72-20261006`, `runs/geographic-environment-family-d72-20261006` and `runs/geographic-multi-ignition-family-d72-20261006`; results from earlier families are not comparable.

D-75 adds 22 Rapid Mapping incidents (2023–2026) whose fuel comes from CORINE 2018. For each, fetch the package and the pinned land-cover extract, then regenerate the chain (terrain and imagery only for the new incidents, `--activations`):

```sh
.venv/bin/python -m pandion.cems_catalog fetch --activation EMSR921
.venv/bin/python -m pandion.cems_catalog fetch-landcover --activation EMSR921
```

`python -m pandion.fold_report RUN_DIR [RUN_DIR ...] --output report.json` collects the `--holdout-activation` selection reports of a cross-validation sweep (one per fold and seed, same configuration) and prints, per held-out incident and pooled over folds, the policy and baseline means and the paired policy - heuristic difference with a 95% bootstrap interval (D-73).

Checks for the workspace and presentation adapters:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

With Node, Playwright and its Chromium browser installed and resolvable, the optional standalone browser regression check uses a temporary deterministic fixture and requires no geographic source data:

```sh
PANDION_PYTHON=.venv/bin/python node tests/viewer_browser.cjs
```

The browser check generates a deterministic five-replay fixture and covers the WebGL scene, live fire and aircraft state, height-datum invariance, camera zoom/reset/keyboard/orthographic/chase/orbit controls, layer and colour changes, the causal decision inspector, observation-only rendering, strict two-policy comparison with a synchronized camera, static cartography, playback, view navigation and deep links, the 390 px layout, and a forced WebGL-disabled fallback. It forces SwiftShader so it runs headless. These checks establish interface behavior, not physical or PPO performance; pixel comparisons are not cross-platform rendering guarantees.

Replay files generated before geographic replay 0.2 may lack arrival rasters and recorded resolution. Regenerate those files; the viewer rejects ambiguous dimensions. `pandion.viewer` also accepts individual synthetic and geographic replay files.

The scenario-family command writes eight fixed synthetic variants, a manifest, 96 full replays, and a per-run comparison report under ignored `runs/scenario-family/`. Four variants are marked development and four reserved. Four base configurations vary ignition, wind, and starting position; each is paired at 35 and 55 m provisional water radii. The reserved results are included in this diagnostic, so they are **not an untouched test** for a future controller selected after reading them. The heuristic returns the same affected area across all four base configurations at a given radius, suggesting these cases are too simple for a meaningful training comparison; see [EVALUATION.md](docs/EVALUATION.md) for the numbers and limits.

The challenge-family command keeps the FM1 fire and 35 m water rule but extends each episode to 20 minutes and adds synthetic nonburnable cells and longer observation delays. It writes eight versioned scenarios and evaluates only the four development cases: 48 full replays and a report. Four separately generated scenarios were opened during a prior exploratory controller test and cannot be treated as untouched test cases. In development, the downwind heuristic beats the random-policy mean in two cases and loses in two. This tests whether action choices matter without implying that the simulator matches a real fire.

A prior exploratory linear-controller trial did not show a convincing gain over the heuristic or seeded random baseline. Its implementation has been removed; the four challenge outcomes it opened remain documented in [EVALUATION.md](docs/EVALUATION.md).

The discrimination-family command generates 24 synthetic scenarios under `nightjar-discrimination-family-0.1` partitioned into disjoint development, validation, and unopened reserved splits (eight scenarios each). It features multi-foci ignitions in separate quadrants, edge water refill locations, and observation delays of 60–120 s. All eight evaluated development cases exceed 4.0 ha unmitigated; that threshold has not been verified for the unopened splits. The command evaluates only the development split (104 full replays) across four policies: `no_intervention` (10.44 ha mean), greedy `heuristic` (8.94 ha mean), `containment_heuristic` (9.34 ha mean), and `random_valid` (9.23 ha mean). Validation and reserved outcomes remain unopened for future model selection and testing; see [EVALUATION.md](docs/EVALUATION.md).

The geographic family command requires the four verified local CEMS products `emsr578_del.zip`, `emsr590_del.zip`, `emsr592_del.zip`, and `emsr600_del.zip` under `data/raw/cems/` (or `--raw-dir`). It never downloads during generation. Fetching existing catalog products is explicit, for example `.venv/bin/python -m pandion.cems_catalog fetch --activation EMSR578`; repeat for EMSR590, EMSR592, and EMSR600 before generating. Fetch verifies each registered SHA-256. Run `.venv/bin/python -m pandion.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-real-v2` only after all four are present; output must be new/empty. Development uses EMSR578, validation EMSR590, and reserved scenarios round-robin EMSR592/EMSR600. Use `--synthetic` only for offline fixtures; the manifest and each scenario identify `synthetic_fixture`. Rows increase north, columns east, and source rasters sample cell centers from a southwest-boundary origin; discrete DROP targets remain cell-anchor proxies, not measured physical drop centers. This base builder leaves topography flat and environmental settings modeled; use the derived environment-family workflow above to attach sourced heights and Artesa station forcing. This is not physical validation. See [DATA_AND_VALIDATION.md](docs/DATA_AND_VALIDATION.md) for the source-coverage limitation and geographic data gate.


The same fixed scenario now has a small headless interface for exploratory controller code:

```python
import json
from pandion.training_env import IntegratedTrainingEnv

with open("examples/synthetic_integrated.json") as source:
    env = IntegratedTrainingEnv(json.load(source))
observation = env.reset(seed=42)
done = False
while not done:
    observation, reward, done, info = env.step({"type": "HOLD"})
```

`env.action_mask` lists valid orders for the current delivered observation; `step` rejects others. The reward is negative newly affected area in hectares. This interface remains an exploratory synthetic contract.

For exploratory PPO research (Stage 1), install `requirements-rl.txt` (`torch>=2.0.0`) and run the development-only smoke training command:

```sh
python3 -m pandion.ppo --output runs/ppo-smoke.json --updates 1 --episodes-per-update 1
# Or with recurrent memory (LSTM), tactical macro-actions, and procedural domain randomization:
python3 -m pandion.ppo --recurrent --domain-randomization --action-mode tactical --updates 1 --episodes-per-update 1
```

The command writes no artifact unless `--output` is supplied, and never opens validation or reserved scenarios. Results are synthetic diagnostics only. To train multiple candidates on development and select one deterministically on validation, save per-case metrics, provenance, and a checkpoint with:

```sh
python3 -m pandion.ppo --select --output runs/ppo-selection.json --checkpoint runs/ppo-selection.pt --candidates 2 --updates 1 --episodes-per-update 1 --action-mode tactical
```

Selection minimizes mean validation affected area, breaking exact ties by seed and candidate index; reserved scenarios are never evaluated. Each validation case records PPO, `heuristic`, and `containment_heuristic` metrics matched by `scenario_id`, with paired PPO-minus-baseline means, counts, wins/losses, and rejected-order criteria for the selected candidate. The completed 16x8 five-seed run selected a validation mean of 8.625 ha, a 0.53125 ha reduction versus the heuristic.

To evaluate a selected checkpoint strictly once against all eight reserved scenarios and baselines under the predeclared protocol:

```sh
python3 -m pandion.evaluate_reserved --checkpoint runs/ppo-matched-5seeds-16x8.pt --output runs/reserved-evaluation/report.json
```

On the held-out reserved split, the selected policy achieved zero rejected orders and improved over valid random choice (`6.719 ha` vs `7.013 ha`, 6 wins), but failed to surpass transparent heuristics (`6.531 ha` greedy, `6.281 ha` containment); see [EVALUATION.md](docs/EVALUATION.md). The `pandion.model` module provides observation tensorization (10 spatial raster channels including public normalized elevation, slope magnitude, and forward physical spread risk heatmap, plus 10 vector features), discrete action mapping (spatial $2 + H \times W$ or 9 tactical macro-actions), and a hybrid CNN+MLP `SpatialActorCritic` with invalid-action masking and optional recurrent memory (`recurrent=True`):

```python
from pandion.model import ActionMapper, ObservationTensorizer, SpatialActorCritic

mapper = env.action_mapper()
tensorizer = env.observation_tensorizer()
model = SpatialActorCritic(grid_rows=11, grid_cols=11, recurrent=True)

# Step using discrete action indices with action masking and recurrent memory
state = None
action_idx, state = model.select_action(observation, tensorizer, mapper, deterministic=True, state=state)
observation, reward, done, info = env.step_discrete(action_idx)
```

For real-world geographic decision research with Copernicus CEMS scenarios:

Geographic PPO requires complete manifest membership with no duplicate scenario IDs across files. Real source metadata must match a single registered catalog product, including its stage, checksum, code, filename, and acquisition time. Smoke reports identify the development training pool of at most two scenarios.

```sh
# Requires the verified real family generated above:
.venv/bin/python -m pip install -r requirements-standard.txt -r requirements-rl.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.geographic_ppo --smoke --seed 42 --scenarios-dir runs/geographic-family-real-v2 --checkpoint runs/geographic-ppo-verification/smoke.pt --output runs/geographic-ppo-verification/smoke.json
```

`--seed` defaults to 42 and is recorded in training reports. `--smoke --checkpoint` saves model weights; `--resume` requires an existing weights file and creates a fresh optimizer. It does not restore optimizer or random-generator state. The [geographic PPO verification](docs/GEOGRAPHIC_PPO_VERIFICATION.md) records the development-only smoke, checkpoint checks, and reproducibility limits.

The observation example reads the FM1 replay and a synthetic schedule of capture and delivery times. It exposes only delivered, covered cells. It is an information-flow test, not a real satellite sensor model.

An independent-data source audit is also available:

```sh
python3 -m pandion.emsr578 fetch
python3 -m pandion.emsr578 report --output runs/emsr578-aoi01.json
```

The audited Copernicus polygons are not a validation of FM1; see [VALIDATION_EMSR578.md](docs/VALIDATION_EMSR578.md). Downloads remain in ignored `data/raw/`.

The independent grass-rate diagnostic requires `curl` and retrieves two small tables from the versioned CSIRO Annaburroo collection:

```sh
python3 -m pandion.annaburroo --output runs/annaburroo-source.json
python3 -m pandion.annaburroo_compare --output runs/annaburroo-fm1-diagnostic.json
```

Its 120 eligible plots yield 0.5264 m/s mean absolute error for unchanged FM1; scope and exclusions are in [VALIDATION_ANNABURROO.md](docs/VALIDATION_ANNABURROO.md). This does not close the training gate.


To inspect completed leave-one-incident-out folds with their own checkpoints,
use `--folds-dir` instead of a single `--checkpoint`. This selects held-out
cases from the development family and shows the fold/checkpoint in the viewer:

```sh
python3 -m pandion.workspace --scenarios-dir runs/geographic-multi-ignition-family-d71-20261006 --folds-dir runs/geographic-holdout-folds-d71-20261006 --output-dir runs/workspace-d71 --episodes 1 --skip-forced
```

`--episodes` is per fold. D-71's archived reports omitted `skip_forced`; the
explicit flag reproduces their recorded launch configuration. Existing output
directories are never overwritten. Water sources can lie outside the tactical
map; aircraft remain at the source when the policy issues HOLD after refilling.
Refill tracks distinguish arrival from service, and unfinished missions at the
episode horizon do not teleport or complete a refill.


## Scope and limitations

Pandion is a research environment. Early scenarios will use a mixture of observed and modeled data, with each layer labeled. No model should be used to direct real firefighting operations without independent physical validation, operational review, and appropriate human oversight.

## License and citation

The project license, data redistribution rules, and citation metadata will be chosen by the project owner before public release. Until then, no license is asserted for this new project or for materials in the earlier FireLab-RL prototype.
