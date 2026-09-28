# Research and implementation roadmap

No dates or performance targets are promised. A stage advances when its evidence gate passes.

## Stage 0 — reproducible foundation

- Specify the first candidate spread equation and aircraft reference, including their limits, in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md).
- Implemented first slice: a versioned synthetic one-aircraft movement and water-accounting episode with deterministic replay and invariant tests. Physical calibration and the full Stage 0 gate remain open.
- Implemented second slice: a separate unmitigated synthetic grid-arrival episode with directional wind/slope test factors and deterministic checks. Its test-rate model remains available for interface checks.
- Implemented third slice: the one-size dead-fuel Rothermel/Albini FM1 rate and published directional extension, with a fixed-domain resolution comparison. Broader fuel models, heterogeneous grid studies, independent spread validation, real sensor latency, and calibration remain open.
- Added an optional synthetic standard-fuel path using Pyretechnics, GR1/SH1 codes, and fixed moisture/wind. Its [physical-evidence plan](STANDARD_FUEL_PATH.md) begins with a pilot data inventory and candidate fuel crosswalk, then aligned weather/moisture, numerical comparison, held-out perimeter validation, and integration. None of those steps is complete; exploratory controller-interface work proceeds on the separate fixed FM1 episode.
- Implemented a seven-case synthetic resolution audit and an independent-source audit of three Copernicus EMSR578 AOI01 burned-area products. The audit documents why those observations do not yet validate FM1 or close the Stage 0 gate.
- Implemented a no-fit external rate diagnostic against 120 eligible CSIRO Annaburroo grass plots. Its 0.5264 m/s mean absolute error shows the need for additional physical work; it is not a Spanish incident or grid-perimeter validation.
- Implemented a synthetic observation-timing component with explicit coverage, capture, delivery, age, and leakage tests. The integrated episode uses it for aircraft decisions; sensor error remains open.
- Implemented a small event-driven integrated episode with one aircraft, an explicitly provisional permanent water barrier, and observation-matched no-intervention, random, and heuristic comparisons. A deterministic reset/step wrapper now exposes its observed action choices and incremental area reward for exploratory controller work. This closes an interface smoke test, not physical validation or the Stage 0 gate.
- Implemented a versioned eight-variant synthetic scenario family with development and reserved labels, paired 35/55 m water radii, three observation-matched baselines, full replays, and a descriptive report. The heuristic's identical outcome across all four base configurations at a given radius motivated the harder observation and geometry challenge used for the first learned-policy experiment. The first family's reserved outcomes have been inspected and cannot serve as an untouched future test.
- Implemented a separate 1,200 s synthetic decision challenge with seeded obstacles and delayed surveys. The heuristic beats the random mean in two inspected development cases and loses in two. Four independently generated cases were kept unopened until the first selected learned controller was tested, and are now opened. This is a useful action-choice check within a provisional simulator, not a physical or geographic validation.
- A prior exploratory linear-controller trial opened four reserved challenge cases and found no convincing advantage. The controller and its training-only scenario generator were removed; those outcomes remain documented and cannot be reused as an untouched test.
- Implemented the `nightjar-discrimination-family-0.1` synthetic benchmark with 24 scenarios across disjoint splits (development, validation, unopened reserved). It features two simultaneous ignitions, edge water refill locations, dynamic observation delays, and non-trivial spread (>= 4.0 ha unmitigated, 10.44 ha mean) to force spatial trade-offs and eliminate floor effects.
- Implemented the `containment_heuristic` reference baseline and predeclared quantitative improvement criteria (MRE >= 0.50 ha vs heuristic, non-inferiority vs containment, paired dominance >= 50%, zero rejected orders) prior to future model search.
- Define units, coordinates, time steps, map resolution, scenario format, and provenance manifest.
- Extend the headless integrated experiment beyond its synthetic smoke case with a justified water-effect model, scenario loader for sourced layers, uncertainty, and numerical sensitivity.
- Audit for hidden-state leakage and test physical invariants, time accounting, termination, action feasibility, and deterministic seed replay.
- Calibrate unmitigated spread and aircraft/drop assumptions against identified evidence or mark uncertain parameters and ranges.

**Gate:** another developer can load a versioned scenario, reproduce an episode, distinguish true from observed state, and regenerate physical calibration figures.

The current status of physical and result-bearing training requirements is tracked in [TRAINING_READINESS.md](TRAINING_READINESS.md). Stage 0 has not passed its physical-evidence gate. The existing integrated FM1 episode is frozen as the provisional environment for an exploratory decision-interface experiment; this does not mark Stage 0 complete.

## Stage 1 — one aircraft decision research

- Designed and verified the new untouched test family (`discrimination_family`) and stronger baseline (`containment_heuristic`) with predeclared improvement criteria before the next model-selection run. Keep learned-policy claims exploratory and simulator-specific.
- Extend the implemented synthetic no-intervention, random-valid, and observation-matched heuristic baselines to validated scenarios.
- Freeze primary metric, splits, and evaluation script.
- Implemented an exploratory minimal CNN + non-spatial encoder PPO path: smoke mode trains on development only, while selection mode records deterministic validation metrics, provenance, and a selected checkpoint without evaluating reserved scenarios. It is not physical or geographic evidence. Add recurrence only as a measured experiment.
- Completed the L1/L2 exploratory PPO diagnostic and matched-budget comparison. The old selected checkpoint lost on validation-01 (`+0.5 ha`) and validation-03 (`+1.0 ha`) after divergent initial `DROP` targets, so later actions are not causal evidence. At 16x8, seed `2` was selected (`8.625 ha` mean), reducing the heuristic by `0.53125 ha` (5 wins/2 losses/1 tie), with `0.21875 ha` reduction versus containment and zero rejected orders; all predeclared criteria passed, but validation was reused adaptively after 8x8, so this is not an independent test. The reserved split remains unopened; the independent Stage 1 test and Stage 0 physical-evidence gate remain open.
- Run independent training seeds and paired held-out tests; inspect failure cases.
- Build a local browser replay if it helps audit observations, orders, and outcomes.

**Gate:** comparison against baselines is reproducible, accompanied by raw records and uncertainty intervals; shortcomings are documented.

## Stage 2 — two aircraft coordination

- Generalize resource roster, order schema, water-source contention, and mission assignment.
- Implement an assignment heuristic or auction and an independent-controller baseline.
- Measure duplicate work, response time, utilization, and resource balance.
- Evaluate variable starting positions, different loads/speeds, and loss of one aircraft.

**Gate:** evidence explains whether joint coordination improves outcomes and when it does not.

## Stage 3 — sensing and heterogeneous resources

- Add UAV reconnaissance with explicit coverage, latency, and uncertainty; test its causal value using ablations.
- Add ground crews only after access/travel and construction dynamics are sufficiently specified.
- Compare flat, hierarchical, and other coordination architectures under the same environment and budget.

**Gate:** each added resource has a validated executor and a measured contribution; the controller can handle a variable roster.

## Deferred until evidence warrants them

Desktop application packaging, full 3D rendering, many simultaneous aircraft, detailed ember transport, complex airspace deconfliction, GPU simulator rewrite, and any operational deployment. These remain documented research candidates, not implicit Stage 1 requirements.
