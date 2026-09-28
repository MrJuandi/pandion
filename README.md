# Nightjar

Research project for learning and evaluating decisions in wildfire response. The long-term goal is a decision-making system that assigns missions to heterogeneous resources using the information available at the time of each decision. Development begins with one water-dropping aircraft, then two aircraft, before introducing other resource types.

**Status:** early Stage 0. The repository contains deterministic aircraft and fire components plus a small synthetic integrated episode. Its full-load drop blocks not-yet-burning grid cells under a provisional, unsourced water rule. Synthetic scenario families and a headless decision interface are available for evaluation design; no trained controller is part of the current code. This is not a physically validated or operational decision tool.

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
python3 -m unittest discover -s tests -v
```

An optional synthetic path uses [Pyretechnics](https://github.com/pyregence/pyretechnics) and numbered standard fire-behavior fuel models. Install its pinned dependency in a virtual environment, then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-standard.txt
.venv/bin/python -m nightjar.fire_standard examples/standard_fuel_fire.json --output runs/standard-fuel-fire.json
```

The example mixes GR1 (101), SH1 (141), and nonburnable (0) cells. It is a synthetic, no-intervention spread example; no geographic land-cover product has been mapped to these codes yet. Its replay does not feed the aircraft episode. See [STANDARD_FUEL_PATH.md](docs/STANDARD_FUEL_PATH.md) for the full contract, evidence, and next steps.

The same unit suite runs in GitHub Actions for pushes and pull requests. Tests that need a downloaded external source skip when the audited file is absent.

`runs/` is ignored by Git. Omit `--output` to print a result to standard output for commands that accept it. The aircraft example exercises movement, a full-water drop, return, and refill. The fire examples produce first-arrival times and grid-cell affected area without intervention. The resolution command compares the FM1 assumptions on one physical domain at several cell sizes. Scenario contracts and limitations are in [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [PHYSICAL_MODEL_V0.md](docs/PHYSICAL_MODEL_V0.md). A local browser replay and analysis view remains optional; scientific artifacts are usable without it.

The integrated example compares no intervention, a seeded valid random choice, and a deterministic downwind heuristic over the same synthetic scenario. At its 600 s horizon, the example gives 2.00 ha without intervention, 0.75 ha with the heuristic, and 1.30 ha mean over ten random-policy seeds. These figures depend on the assumed permanent one-cell water barrier and do not estimate real suppression effectiveness; see [EVALUATION.md](docs/EVALUATION.md).

The scenario-family command writes eight fixed synthetic variants, a manifest, 96 full replays, and a per-run comparison report under ignored `runs/scenario-family/`. Four variants are marked development and four reserved. Four base configurations vary ignition, wind, and starting position; each is paired at 35 and 55 m provisional water radii. The reserved results are included in this diagnostic, so they are **not an untouched test** for a future controller selected after reading them. The heuristic returns the same affected area across all four base configurations at a given radius, suggesting these cases are too simple for a meaningful training comparison; see [EVALUATION.md](docs/EVALUATION.md) for the numbers and limits.

The challenge-family command keeps the FM1 fire and 35 m water rule but extends each episode to 20 minutes and adds synthetic nonburnable cells and longer observation delays. It writes eight versioned scenarios and evaluates only the four development cases: 48 full replays and a report. Four separately generated scenarios were opened during a prior exploratory controller test and cannot be treated as untouched test cases. In development, the downwind heuristic beats the random-policy mean in two cases and loses in two. This tests whether action choices matter without implying that the simulator matches a real fire.

A prior exploratory linear-controller trial did not show a convincing gain over the heuristic or seeded random baseline. Its implementation has been removed; the four challenge outcomes it opened remain documented in [EVALUATION.md](docs/EVALUATION.md).

The discrimination-family command generates 24 synthetic scenarios under `nightjar-discrimination-family-0.1` partitioned into disjoint development, validation, and unopened reserved splits (eight scenarios each). It features multi-foci ignitions in separate quadrants, edge water refill locations, and observation delays of 60–120 s. All eight evaluated development cases exceed 4.0 ha unmitigated; that threshold has not been verified for the unopened splits. The command evaluates only the development split (104 full replays) across four policies: `no_intervention` (10.44 ha mean), greedy `heuristic` (8.94 ha mean), `containment_heuristic` (9.34 ha mean), and `random_valid` (9.23 ha mean). Validation and reserved outcomes remain unopened for future model selection and testing; see [EVALUATION.md](docs/EVALUATION.md).


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

For neural policy research (Stage 1), install `requirements-rl.txt` (`torch>=2.0.0`). The `nightjar.model` module provides observation tensorization (9 spatial raster channels and 10 vector features, including public normalized elevation and slope magnitude), discrete action mapping ($2 + H \times W$), and a hybrid CNN+MLP `SpatialActorCritic` with invalid-action masking:

```python
from nightjar.model import ActionMapper, ObservationTensorizer, SpatialActorCritic

mapper = env.action_mapper()
tensorizer = env.observation_tensorizer()
model = SpatialActorCritic(grid_rows=11, grid_cols=11)

# Step using discrete action indices with action masking
action_idx = model.select_action(observation, tensorizer, mapper, deterministic=True)
observation, reward, done, info = env.step_discrete(action_idx)
```

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
