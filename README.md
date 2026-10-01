# Nightjar

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

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — state, observation, actions, reward, policy, and growth to multiple resources.
- [PHYSICAL_MODEL_V0.md](docs/PHYSICAL_MODEL_V0.md) — sourced surface-fire and CL-415 references, implemented FM1 subset, and remaining validation limits.
- [STANDARD_FUEL_PATH.md](docs/STANDARD_FUEL_PATH.md) — standard-fuel scenario, reproduction command, boundaries, and ordered steps toward real maps.
- [DATA_AND_VALIDATION.md](docs/DATA_AND_VALIDATION.md) — candidate data, provenance, calibration, and validation requirements.
- [VALIDATION_EMSR578.md](docs/VALIDATION_EMSR578.md) — audited independent burned-area snapshots and the reasons they cannot yet validate FM1 spread.
- [VALIDATION_ANNABURROO.md](docs/VALIDATION_ANNABURROO.md) — no-fit comparison of FM1 head rate with experimental grass fires.
- [TRAINING_READINESS.md](docs/TRAINING_READINESS.md) — evidence gate for physical claims and the separate scope of exploratory synthetic training.
- [EVALUATION.md](docs/EVALUATION.md) — baselines, splits, metrics, statistics, and failure analysis.
- [ROADMAP.md](docs/ROADMAP.md) — milestones and acceptance criteria.
- [DECISIONS.md](docs/DECISIONS.md) — settled choices and open decisions.
- [REFERENCES.md](docs/REFERENCES.md) — related research and official data sources.
- [PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) — checks before creating and publishing the GitHub repository.
- [MAINTENANCE.md](docs/MAINTENANCE.md) — documentation ownership and change review rules.

## Development and use

The synthetic aircraft episode uses Python 3 and the standard library; no installation step is needed from the repository root:

```sh
mkdir -p runs
python3 -m nightjar examples/synthetic_aircraft.json examples/synthetic_orders.json --output runs/aircraft-replay.json
python3 -m nightjar.fire examples/synthetic_fire.json --output runs/fire-replay.json
python3 -m nightjar.fire_rothermel examples/fm1_fire.json --output runs/fm1-fire-replay.json
python3 -m nightjar.observation runs/fm1-fire-replay.json examples/synthetic_observation_schedule.json --time-s 400 --output runs/observation-400s.json
python3 -m nightjar.sensitivity --output runs/fm1-resolution.json
python3 -m nightjar.sensitivity --varied --output runs/fm1-varied-resolution.json
python3 -m nightjar.integrated examples/synthetic_integrated.json --policy heuristic --output runs/integrated-heuristic.json
python3 -m nightjar.integrated_benchmark examples/synthetic_integrated.json --output runs/integrated-comparison.json
python3 -m nightjar.scenario_family --output-dir runs/scenario-family
python3 -m nightjar.challenge_family --output-dir runs/challenge-family
python3 -m nightjar.discrimination_family --output-dir runs/discrimination-family
python3 -m nightjar.cems_catalog list
python3 -m nightjar.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-real-v2
# Offline fixtures must be requested explicitly and are labeled synthetic:
python3 -m nightjar.geographic_family --synthetic --output-dir runs/geographic-family-synthetic
python3 -m nightjar.geographic_integrated runs/geographic-family-real-v2/development/geographic-dev-emsr578-00.json --policy heuristic --output runs/geographic-family-real-v2/smoke-development-00.json
python3 -m nightjar.geographic_benchmark runs/geographic-family-real-v2/development/geographic-dev-emsr578-00.json --output runs/geographic-family-real-v2/benchmark-development-00.json
python3 -m nightjar.geographic_ppo --smoke --scenarios-dir runs/geographic-family-real-v2
# Offline PPO checks require the synthetic family generated above and explicit mode:
python3 -m nightjar.geographic_ppo --smoke --synthetic --scenarios-dir runs/geographic-family-synthetic
python3 -m nightjar.evaluate_reserved --split-version v2
python3 -m nightjar.viewer runs/integrated-heuristic.json --output runs/viewer.html
python3 -m unittest discover -s tests -v
```

An optional synthetic path uses [Pyretechnics](https://github.com/pyregence/pyretechnics) and numbered standard fire-behavior fuel models. Install its pinned dependency in a virtual environment, then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-standard.txt
.venv/bin/python -m nightjar.fire_standard examples/standard_fuel_fire.json --output runs/standard-fuel-fire.json
```

The example mixes GR1 (101), SH1 (141), and nonburnable (0) cells. It is a synthetic, no-intervention spread example with hand-assigned fuels; geographic land-cover mapping is implemented separately in `nightjar.cems_catalog`. Its replay does not feed the aircraft episode. See [STANDARD_FUEL_PATH.md](docs/STANDARD_FUEL_PATH.md) for the full contract, evidence, and next steps.

The same unit suite runs in GitHub Actions for pushes and pull requests. The `unit` job checks the standard-fuel path; `unit-rl` also installs CPU PyTorch and requires its import before running the suite. Tests that need a downloaded external source skip when the audited file is absent.

`runs/` is ignored by Git. Omit `--output` to print a result to standard output for commands that accept it. The aircraft example exercises movement, a full-water drop, return, and refill. The fire examples produce first-arrival times and grid-cell affected area without intervention. The resolution command compares the FM1 assumptions on one physical domain at several cell sizes. Scenario contracts and limitations are in [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [PHYSICAL_MODEL_V0.md](docs/PHYSICAL_MODEL_V0.md). The local browser replay tool (`nightjar.viewer`) generates a self-contained offline HTML5 viewer to audit fire arrival, delivered observations, flight path, and drop decisions interactively or compare multiple policies side-by-side.

The integrated example compares no intervention, a seeded valid random choice, and a deterministic downwind heuristic over the same synthetic scenario. At its 600 s horizon, the example gives 2.00 ha without intervention, 0.75 ha with the heuristic, and 1.30 ha mean over ten random-policy seeds. These figures depend on the assumed permanent one-cell water barrier and do not estimate real suppression effectiveness; see [EVALUATION.md](docs/EVALUATION.md).

### Explore the research workspace

Build the Spanish offline interface with real heights in the simulation and historical station conditions for Artesa:

```sh
# Requires the four verified local CEMS ZIPs under data/raw/cems.
.venv/bin/python -m nightjar.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-source-corrected-20261001
.venv/bin/python -m pip install -r requirements-terrain.txt
.venv/bin/python -m nightjar.terrain_acquire --scenarios-dir runs/geographic-family-source-corrected-20261001 --output-dir data/raw/geographic-environment-source-corrected-20261001
.venv/bin/python -m nightjar.environment_family --scenarios-dir runs/geographic-family-source-corrected-20261001 --terrain-dir data/raw/geographic-environment-source-corrected-20261001 --weather-audit runs/pilot-environment-20260930/weather.json --pilot-scenario runs/emsr578-pilot-scenario.json --output-dir runs/geographic-environment-family-20261001
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m nightjar.workspace --scenarios-dir runs/geographic-environment-family-20261001 --output-dir runs/environment-workspace
```

Every output directory must be new or empty. The weather audit and original pilot are prepared with the commands in [PILOT_ENVIRONMENT.md](docs/PILOT_ENVIRONMENT.md). Omit both `--weather-audit` and `--pilot-scenario` to use clearly labeled synthetic constant weather with real heights. Acquisition uses IGN/CNIG terrain in Spain and Copernicus GLO-30 surface heights for France/Italy; it needs network access, `curl`, and optional Rasterio for the latter. The derived family records hashes, CRS, source and resolution and rejects missing heights. Its grid remains 100 m; source resolution is 25/30 m.

Open `runs/environment-workspace/index.html` in a browser, or serve the folder locally (for example `python3 -m http.server 8000 --directory runs/environment-workspace`). The self-contained Spanish workspace follows the layout of the local FireLab viewer: a full-screen interactive Three.js scene, a header with scenario, policy, speed and playback controls, and an analysis sidebar. **Mapa 3D**, **Comparación**, **Condiciones** and **Datos y modelos** are separate views that preserve the selected scenario and elapsed time.

The scene draws the DEM as a lit, shadowed mesh on a base with side walls and a 1 km reference floor that extends beyond the modelled grid, so recorded refill positions outside the crop remain visible. The terrain texture combines fuel, hillshade and hypsometric tint; **Color** switches to altitude or slope with a numeric key. At the selected time it shows burned cells as ash, the active front as glowing cells, delivered detections as dashed cell-union outlines, and drop-treated cells. The active front is derived from the arrival raster: reached cells with at least one burnable 4-neighbour not yet reached. **Estado** switches to simulator truth, the delivered observation only, or historical cartography. Equal-elevation contours, labels, a cell grid and an illustrative smoke layer can be toggled in **Simbología y capas**. The smoke is a visual effect and is not simulated.

The aircraft is a stylised amphibious tanker (CL-415 reference; the recorded tank holds 6,137 L) that follows the recorded horizontal track, interpolated linearly between samples. Its trail, nadir marker and label (phase and tank level from the recorded decisions) are drawn with it, and a short water curtain plays after each recorded drop. Aircraft altitude and symbol size are display choices, not replay data. The camera dock offers **3D Táctica**, **2D Orto** (orthographic, north up), **Persecución** (follows the aircraft) and **Frente** (centres the active front). Drag to orbit, right-drag or Shift-drag to pan, wheel to zoom, double-click to focus a point; with a map focused, arrow keys/WASD pan, +/− zoom, Home resets and Space toggles playback. **Relieve** sets vertical exaggeration (1×, 1.5×, 2×, 3×; default 1.5×); the active factor is always shown in the terrain panel and the physics uses unexaggerated heights. The display surface interpolates cell-corner heights (Catmull-Rom) without adding physical resolution. The HUD shows UTC time, decision step, camera pitch/azimuth/altitude, a compass with the simulation wind and a scale bar valid at the view centre. When WebGL is unavailable, an explicit message appears and a 2D compatibility map remains available. **Exportar PNG** saves the current view.

The sidebar separates the live state at the selected time (reached area, active-front area, water released so far, aircraft phase and tank) from final episode totals. **Dinámica de crecimiento A(t)** plots reached area from each policy's arrival raster with the current time marked; **Perfil topográfico** samples elevation along the wind direction through the front, shading reached segments. The decision inspector shows the latest decision issued at or before the selected instant and its delivered detections; the decision list moves the clock to any decision. The cell tooltip reports coordinates, fuel, elevation and slope; observation mode does not expose simulator arrival times. Initial burned area is a cumulative footprint, not active flames.

**Condiciones** includes the timeline and follows the selected episode time: elevation range, UTC time, wind, temperature, relative humidity, interval rain and estimated dead-fuel moisture, plus weather already delivered to the controller. **Comparación** shows exactly two selected policies from the same scenario side by side, with a shared clock, a synchronized camera, a two-row results table and both curves in A(t); shorter episodes retain their completed state. **Política** selects the primary episode and **Comparar con** a distinct reference. Clicking a panel title swaps the two roles. Scenarios without multiple episodes show an explicit unavailable-comparison message. **Datos y modelos** records source, assumptions, training provenance and the display conventions above. The interface remains an offline replay inspector; it adds no simulation or policy capabilities. [Three.js r128, its OrbitControls and the MIT license](nightjar/viewer_assets/vendor/README.md) are embedded in the generated HTML, with no runtime CDN dependency.

The default runs three baselines on the first two development cases, showing the remaining cases as static maps. `--episodes 0` produces cartography only. The 24 cases represent four incidents, not 24 independent fires: Artesa (EMSR578), Ladrillar/Las Hurdes (EMSR590), Landiras (EMSR592, France), and Massarosa (EMSR600, Italy). Validation/reserved cases never run episodes through this command. The 2026-10-01 source audit corrected the previous incident names, acquisition times and Massarosa CRS. Regenerate stale `geographic-family-real-v2` outputs; they are rejected by current provenance checks.

**The new environment family uses sourced elevations in Pyretechnics slope and public PPO terrain channels.** Artesa uses time-dependent Baldomar 2 m station wind as an explicitly uncalibrated, spatially uniform midflame proxy. Temperature/RH drive a Simard equilibrium plus 1/10/100 h moisture-lag estimate; initial/live moisture remains modeled. The dry-interval mode rejects rain instead of inventing wetting. Other incidents retain synthetic weather. Half-hour aggregates reach the controller only at interval end; source delivery latency is unknown and assumed zero. These assumptions are exploratory, not a reconstruction or physical validation. See [ENVIRONMENT_INTEGRATION.md](docs/ENVIRONMENT_INTEGRATION.md).

To train fresh weights on this changed environment and show them alongside baselines:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m nightjar.geographic_ppo --train --updates 6 --episodes-per-update 1 --scenarios-dir runs/geographic-environment-family-20261001 --seed 42 --checkpoint runs/geographic-environment-ppo-20261001/model.pt --output runs/geographic-environment-ppo-20261001/train.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m nightjar.workspace --scenarios-dir runs/geographic-environment-family-20261001 --output-dir runs/environment-workspace-with-ppo --checkpoint runs/geographic-environment-ppo-20261001/model.pt --training-report runs/geographic-environment-ppo-20261001/train.json
```

Keep weights with their matching report. Manifest-bound development hashes must match; old flat-terrain checkpoints cannot be represented as trained on the new family. Weights do not embed report identity or optimizer state. The optional `--terrain` + `--terrain-scenario` display overlay remains for legacy flat episodes and full pilot cartography; it is unnecessary for the environment family. The full pilot is historical cartography without an episode. Training is development-only and exploratory; the number of updates does not demonstrate skill or geographic generalization.

Checks for the workspace and presentation adapters:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

With Node, Playwright and its Chromium browser installed and resolvable, the optional standalone browser regression check uses a temporary deterministic fixture and requires no geographic source data:

```sh
NIGHTJAR_PYTHON=.venv/bin/python node tests/viewer_browser.cjs
```

The browser check generates a deterministic five-replay fixture and covers the WebGL scene, live fire and aircraft state, height-datum invariance, camera zoom/reset/keyboard/orthographic/chase/orbit controls, layer and colour changes, the causal decision inspector, observation-only rendering, strict two-policy comparison with a synchronized camera, static cartography, playback, view navigation and deep links, the 390 px layout, and a forced WebGL-disabled fallback. It forces SwiftShader so it runs headless. These checks establish interface behavior, not physical or PPO performance; pixel comparisons are not cross-platform rendering guarantees.

Replay files generated before geographic replay 0.2 may lack arrival rasters and recorded resolution. Regenerate those files; the viewer rejects ambiguous dimensions. The original `nightjar.viewer` CLI remains available for synthetic and geographic replay files.

The scenario-family command writes eight fixed synthetic variants, a manifest, 96 full replays, and a per-run comparison report under ignored `runs/scenario-family/`. Four variants are marked development and four reserved. Four base configurations vary ignition, wind, and starting position; each is paired at 35 and 55 m provisional water radii. The reserved results are included in this diagnostic, so they are **not an untouched test** for a future controller selected after reading them. The heuristic returns the same affected area across all four base configurations at a given radius, suggesting these cases are too simple for a meaningful training comparison; see [EVALUATION.md](docs/EVALUATION.md) for the numbers and limits.

The challenge-family command keeps the FM1 fire and 35 m water rule but extends each episode to 20 minutes and adds synthetic nonburnable cells and longer observation delays. It writes eight versioned scenarios and evaluates only the four development cases: 48 full replays and a report. Four separately generated scenarios were opened during a prior exploratory controller test and cannot be treated as untouched test cases. In development, the downwind heuristic beats the random-policy mean in two cases and loses in two. This tests whether action choices matter without implying that the simulator matches a real fire.

A prior exploratory linear-controller trial did not show a convincing gain over the heuristic or seeded random baseline. Its implementation has been removed; the four challenge outcomes it opened remain documented in [EVALUATION.md](docs/EVALUATION.md).

The discrimination-family command generates 24 synthetic scenarios under `nightjar-discrimination-family-0.1` partitioned into disjoint development, validation, and unopened reserved splits (eight scenarios each). It features multi-foci ignitions in separate quadrants, edge water refill locations, and observation delays of 60–120 s. All eight evaluated development cases exceed 4.0 ha unmitigated; that threshold has not been verified for the unopened splits. The command evaluates only the development split (104 full replays) across four policies: `no_intervention` (10.44 ha mean), greedy `heuristic` (8.94 ha mean), `containment_heuristic` (9.34 ha mean), and `random_valid` (9.23 ha mean). Validation and reserved outcomes remain unopened for future model selection and testing; see [EVALUATION.md](docs/EVALUATION.md).

The geographic family command requires the four verified local CEMS products `emsr578_del.zip`, `emsr590_del.zip`, `emsr592_del.zip`, and `emsr600_del.zip` under `data/raw/cems/` (or `--raw-dir`). It never downloads during generation. Fetching existing catalog products is explicit, for example `.venv/bin/python -m nightjar.cems_catalog fetch --activation EMSR578`; repeat for EMSR590, EMSR592, and EMSR600 before generating. Fetch verifies each registered SHA-256. Run `.venv/bin/python -m nightjar.geographic_family --raw-dir data/raw/cems --output-dir runs/geographic-family-real-v2` only after all four are present; output must be new/empty. Development uses EMSR578, validation EMSR590, and reserved scenarios round-robin EMSR592/EMSR600. Use `--synthetic` only for offline fixtures; the manifest and each scenario identify `synthetic_fixture`. Rows increase north, columns east, and source rasters sample cell centers from a southwest-boundary origin; discrete DROP targets remain cell-anchor proxies, not measured physical drop centers. This base builder leaves topography flat and environmental settings modeled; use the derived environment-family workflow above to attach sourced heights and Artesa station forcing. This is not physical validation. See [DATA_AND_VALIDATION.md](docs/DATA_AND_VALIDATION.md) for the source-coverage limitation and geographic data gate.


The same fixed scenario now has a small headless interface for exploratory controller code:

```python
import json
from nightjar.training_env import IntegratedTrainingEnv

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
python3 -m nightjar.ppo --output runs/ppo-smoke.json --updates 1 --episodes-per-update 1
# Or with recurrent memory (LSTM), tactical macro-actions, and procedural domain randomization:
python3 -m nightjar.ppo --recurrent --domain-randomization --action-mode tactical --updates 1 --episodes-per-update 1
```

The command writes no artifact unless `--output` is supplied, and never opens validation or reserved scenarios. Results are synthetic diagnostics only. To train multiple candidates on development and select one deterministically on validation, save per-case metrics, provenance, and a checkpoint with:

```sh
python3 -m nightjar.ppo --select --output runs/ppo-selection.json --checkpoint runs/ppo-selection.pt --candidates 2 --updates 1 --episodes-per-update 1 --action-mode tactical
```

Selection minimizes mean validation affected area, breaking exact ties by seed and candidate index; reserved scenarios are never evaluated. Each validation case records PPO, `heuristic`, and `containment_heuristic` metrics matched by `scenario_id`, with paired PPO-minus-baseline means, counts, wins/losses, and rejected-order criteria for the selected candidate. The completed 16x8 five-seed run selected a validation mean of 8.625 ha, a 0.53125 ha reduction versus the heuristic.

To evaluate a selected checkpoint strictly once against all eight reserved scenarios and baselines under the predeclared protocol:

```sh
python3 -m nightjar.evaluate_reserved --checkpoint runs/ppo-matched-5seeds-16x8.pt --output runs/reserved-evaluation/report.json
```

On the held-out reserved split, the selected policy achieved zero rejected orders and improved over valid random choice (`6.719 ha` vs `7.013 ha`, 6 wins), but failed to surpass transparent heuristics (`6.531 ha` greedy, `6.281 ha` containment); see [EVALUATION.md](docs/EVALUATION.md). The `nightjar.model` module provides observation tensorization (10 spatial raster channels including public normalized elevation, slope magnitude, and forward physical spread risk heatmap, plus 10 vector features), discrete action mapping (spatial $2 + H \times W$ or 9 tactical macro-actions), and a hybrid CNN+MLP `SpatialActorCritic` with invalid-action masking and optional recurrent memory (`recurrent=True`):

```python
from nightjar.model import ActionMapper, ObservationTensorizer, SpatialActorCritic

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
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m nightjar.geographic_ppo --smoke --seed 42 --scenarios-dir runs/geographic-family-real-v2 --checkpoint runs/geographic-ppo-verification/smoke.pt --output runs/geographic-ppo-verification/smoke.json
```

`--seed` defaults to 42 and is recorded in training reports. `--smoke --checkpoint` saves model weights; `--resume` requires an existing weights file and creates a fresh optimizer. It does not restore optimizer or random-generator state. The [geographic PPO verification](docs/GEOGRAPHIC_PPO_VERIFICATION.md) records the development-only smoke, checkpoint checks, and reproducibility limits.

The observation example reads the FM1 replay and a synthetic schedule of capture and delivery times. It exposes only delivered, covered cells. It is an information-flow test, not a real satellite sensor model.

An independent-data source audit is also available:

```sh
python3 -m nightjar.emsr578 fetch
python3 -m nightjar.emsr578 report --output runs/emsr578-aoi01.json
```

The audited Copernicus polygons are not a validation of FM1; see [VALIDATION_EMSR578.md](docs/VALIDATION_EMSR578.md). Downloads remain in ignored `data/raw/`.

The independent grass-rate diagnostic requires `curl` and retrieves two small tables from the versioned CSIRO Annaburroo collection:

```sh
python3 -m nightjar.annaburroo --output runs/annaburroo-source.json
python3 -m nightjar.annaburroo_compare --output runs/annaburroo-fm1-diagnostic.json
```

Its 120 eligible plots yield 0.5264 m/s mean absolute error for unchanged FM1; scope and exclusions are in [VALIDATION_ANNABURROO.md](docs/VALIDATION_ANNABURROO.md). This does not close the training gate.

## Scope and limitations

Nightjar is a research environment. Early scenarios will use a mixture of observed and modeled data, with each layer labeled. No model should be used to direct real firefighting operations without independent physical validation, operational review, and appropriate human oversight.

## License and citation

The project license, data redistribution rules, and citation metadata will be chosen by the project owner before public release. Until then, no license is asserted for this new project or for materials in the earlier FireLab-RL prototype.
