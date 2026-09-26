# Nightjar

Research project for learning and evaluating decisions in wildfire response. The long-term goal is a decision-making system that assigns missions to heterogeneous resources using the information available at the time of each decision. Development begins with one water-dropping aircraft, then two aircraft, before introducing other resource types.

**Status:** design phase. This repository does not yet contain a simulator, trained policy, benchmark result, or operational decision tool. No performance or real-world effectiveness is claimed.

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
- [DATA_AND_VALIDATION.md](docs/DATA_AND_VALIDATION.md) — candidate data, provenance, calibration, and validation requirements.
- [EVALUATION.md](docs/EVALUATION.md) — baselines, splits, metrics, statistics, and failure analysis.
- [ROADMAP.md](docs/ROADMAP.md) — milestones and acceptance criteria.
- [DECISIONS.md](docs/DECISIONS.md) — settled choices and open decisions.
- [REFERENCES.md](docs/REFERENCES.md) — related research and official data sources.
- [PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) — checks before creating and publishing the GitHub repository.
- [MAINTENANCE.md](docs/MAINTENANCE.md) — documentation ownership and change review rules.

## Development and use

There are no installation or training commands yet. They will be added only when the corresponding implementation exists and has been verified. The first interface will be a local browser replay and analysis view if experiments show it is useful; scientific artifacts must remain usable without the interface.

## Scope and limitations

Nightjar is a research environment. Early scenarios will use a mixture of observed and modeled data, with each layer labeled. No model should be used to direct real firefighting operations without independent physical validation, operational review, and appropriate human oversight.

## License and citation

The project license, data redistribution rules, and citation metadata will be chosen by the project owner before public release. Until then, no license is asserted for this new project or for materials in the earlier FireLab-RL prototype.
