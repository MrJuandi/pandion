# Nightjar

Research project for learning and evaluating decisions in wildfire response. The long-term goal is a decision-making system that assigns missions to heterogeneous resources using the information available at the time of each decision. Development begins with one water-dropping aircraft, then two aircraft, before introducing other resource types.

**Status:** early Stage 0. The repository contains deterministic aircraft and fire components plus a small synthetic integrated episode. Its full-load drop blocks not-yet-burning grid cells under a provisional, unsourced water rule. A three-policy comparison exercises information flow and accounting; it is not a physical validation, training result, or operational decision tool.

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
- [DATA_AND_VALIDATION.md](docs/DATA_AND_VALIDATION.md) — candidate data, provenance, calibration, and validation requirements.
- [VALIDATION_EMSR578.md](docs/VALIDATION_EMSR578.md) — audited independent burned-area snapshots and the reasons they cannot yet validate FM1 spread.
- [VALIDATION_ANNABURROO.md](docs/VALIDATION_ANNABURROO.md) — no-fit comparison of FM1 head rate with experimental grass fires.
- [TRAINING_READINESS.md](docs/TRAINING_READINESS.md) — evidence gate and current blockers before the first result-bearing training run.
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
python3 -m unittest discover -s tests -v
```

The same unit suite runs in GitHub Actions for pushes and pull requests. Tests that need a downloaded external source skip when the audited file is absent.

`runs/` is ignored by Git. Omit `--output` to print a result to standard output. The aircraft example exercises movement, a full-water drop, return, and refill. The fire examples produce first-arrival times and grid-cell affected area without intervention. The resolution command compares the FM1 assumptions on one physical domain at several cell sizes. Scenario contracts and limitations are in [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [PHYSICAL_MODEL_V0.md](docs/PHYSICAL_MODEL_V0.md). There is no training command yet. The first interface will be a local browser replay and analysis view if experiments show it is useful; scientific artifacts must remain usable without the interface.

The integrated example compares no intervention, a seeded valid random choice, and a deterministic downwind heuristic over the same synthetic scenario. At its 600 s horizon, the example gives 2.00 ha without intervention, 0.75 ha with the heuristic, and 1.30 ha mean over ten random-policy seeds. These figures depend on the assumed permanent one-cell water barrier and do not estimate real suppression effectiveness; see [EVALUATION.md](docs/EVALUATION.md).

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
