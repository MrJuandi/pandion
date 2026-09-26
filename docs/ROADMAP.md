# Research and implementation roadmap

No dates or performance targets are promised. A stage advances when its evidence gate passes.

## Stage 0 — reproducible foundation

- Define units, coordinates, time steps, map resolution, scenario format, and provenance manifest.
- Implement a headless fire environment with one aircraft, scenario loader, observation model, action candidates, aircraft dynamics, and replay log.
- Audit for hidden-state leakage and test physical invariants, time accounting, termination, action feasibility, and deterministic seed replay.
- Calibrate unmitigated spread and aircraft/drop assumptions against identified evidence or mark uncertain parameters and ranges.

**Gate:** another developer can load a versioned scenario, reproduce an episode, distinguish true from observed state, and regenerate physical calibration figures.

## Stage 1 — one aircraft decision research

- Implement no-intervention, random-valid, and observation-matched heuristic baselines.
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
