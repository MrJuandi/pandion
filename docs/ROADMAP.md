# Research and implementation roadmap

No dates or performance targets are promised. A stage advances when its evidence gate passes.

## Stage 0 — reproducible foundation

- Specify the first candidate spread equation and aircraft reference, including their limits, in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md).
- Implemented first slice: a versioned synthetic one-aircraft movement and water-accounting episode with deterministic replay and invariant tests. Physical calibration and the full Stage 0 gate remain open.
- Implemented second slice: a separate unmitigated synthetic grid-arrival episode with directional wind/slope test factors and deterministic checks. Its test-rate model remains available for interface checks.
- Implemented third slice: the one-size dead-fuel Rothermel/Albini FM1 rate and published directional extension, with a fixed-domain resolution comparison. Broader fuel models, heterogeneous grid studies, independent spread validation, real sensor latency, and calibration remain open.
- Implemented a seven-case synthetic resolution audit and an independent-source audit of three Copernicus EMSR578 AOI01 burned-area products. The audit documents why those observations do not yet validate FM1 or close the Stage 0 gate.
- Implemented a no-fit external rate diagnostic against 120 eligible CSIRO Annaburroo grass plots. Its 0.5264 m/s mean absolute error shows the need for additional physical work; it is not a Spanish incident or grid-perimeter validation.
- Implemented a no-fit temporal head-distance diagnostic on 576 burn-period records from the same 120 plots. Its 73.81 m mean absolute final-distance error and four moisture-extinction failures reinforce that fixed FM1 is not validated. Digitized perimeters and independent incidents remain open.
- Implemented a synthetic observation-timing component with explicit coverage, capture, delivery, age, and leakage tests. The integrated episode uses it for aircraft decisions; sensor error remains open.
- Implemented a small event-driven integrated episode with one aircraft, an explicitly provisional permanent water barrier, and observation-matched no-intervention, random, and heuristic comparisons. This closes an interface smoke test, not physical validation or the Stage 0 gate.
- Define units, coordinates, time steps, map resolution, scenario format, and provenance manifest.
- Extend the headless integrated experiment beyond its synthetic smoke case with a justified water-effect model, scenario loader for sourced layers, uncertainty, and numerical sensitivity.
- Audit for hidden-state leakage and test physical invariants, time accounting, termination, action feasibility, and deterministic seed replay.
- Calibrate unmitigated spread and aircraft/drop assumptions against identified evidence or mark uncertain parameters and ranges.

**Gate:** another developer can load a versioned scenario, reproduce an episode, distinguish true from observed state, and regenerate physical calibration figures.

The current status of each pre-training requirement is tracked in [TRAINING_READINESS.md](TRAINING_READINESS.md). Stage 0 has not passed its gate.

## Stage 1 — one aircraft decision research

- Extend the implemented synthetic no-intervention, random-valid, and observation-matched heuristic baselines to validated scenarios.
- Freeze primary metric, splits, and evaluation script.
- Train CNN + non-spatial encoder with PPO; add recurrence only as a measured experiment.
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
