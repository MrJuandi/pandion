# Research and implementation roadmap

No dates or performance targets are promised. A stage advances when its evidence gate passes.

## Stage 0 — reproducible foundation

- Specify the first candidate spread equation and aircraft reference, including their limits, in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md).
- Implemented first slice: a versioned synthetic one-aircraft movement and water-accounting episode with deterministic replay and invariant tests. Physical calibration and the full Stage 0 gate remain open.
- Implemented second slice: a separate unmitigated synthetic grid-arrival episode with directional wind/slope test factors and deterministic checks. Its test-rate model remains available for interface checks.
- Implemented third slice: the one-size dead-fuel Rothermel/Albini FM1 rate and published directional extension, with a fixed-domain resolution comparison. Broader fuel models, heterogeneous grid studies, independent spread validation, real sensor latency, and calibration remain open.
- Added an optional synthetic standard-fuel path using Pyretechnics, GR1/SH1 codes, and fixed moisture/wind. Its [physical-evidence plan](STANDARD_FUEL_PATH.md) begins with a pilot data inventory and candidate fuel crosswalk, then aligned weather/moisture, numerical comparison, held-out perimeter validation, and integration.
- Implemented the real geographic pilot data pipeline and fuel crosswalk for Copernicus EMSR578 AOI01 (`nightjar.pilot_emsr578`, Decision D-32). Establishes sub-millimeter ETRS89 / UTM Zone 31N geodesy, translates all 13 CORINE Land Cover classes to Scott & Burgan 40 fuel models with documented botanical rationales, extracts waterbody waypoints for CL-415 scooping, rasterizes perimeters at 100 m resolution, and implements spatial overlap verification metrics (IoU, Dice, commission, omission).
- Implemented physical Rothermel surface-fire spread simulation and satellite evaluation on the EMSR578 AOI01 pilot (`nightjar.pilot_simulation`, Decision D-33). Executes 8-neighbor Dijkstra shortest-arrival on 23,712 cells with Scott & Burgan fuels; at equivalent active duration ($T = 3,600\text{ s}$), matches satellite area (2,071.0 ha vs 2,108.0 ha observed, $-1.8\%$ error) with $\text{IoU} = 0.6630$ and $\text{Dice} = 0.7973$. Confirms 24.7 h unmitigated constant-wind overprediction due to real suppression and nighttime humidity recovery. Atmospheric time-series and suppression modeling remain open.
- Implemented universal geodesy (`nightjar.geodesy`), multi-incident CEMS catalog (`nightjar.cems_catalog`), and real-world geographic scenario family (`nightjar.geographic_family`, Decision D-34). Generalizes Krüger transverse Mercator series for UTM Zones 29N, 30N, and 31N, maps 18 CORINE Land Cover classes to Scott & Burgan fuels, ingests major Spanish wildfires (EMSR578, EMSR590, EMSR592, EMSR600), and generates tactical decision scenarios partitioned strictly by incident across development, validation, and held-out reserved splits to prevent data leakage.
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
- Completed the L1/L2 exploratory PPO diagnostic and matched-budget comparison. The old selected checkpoint lost on validation-01 (`+0.5 ha`) and validation-03 (`+1.0 ha`) after divergent initial `DROP` targets, so later actions are not causal evidence. At 16x8, seed `2` was selected (`8.625 ha` mean), reducing the heuristic by `0.53125 ha` (5 wins/2 losses/1 tie), with `0.21875 ha` reduction versus containment and zero rejected orders; all predeclared criteria passed on validation, but validation was reused adaptively after 8x8.
- Completed the strictly one-time blind evaluation on the eight reserved scenarios (`discrimination-reserved-00` to `07`) using the selected 16x8 checkpoint (`c8825858...`). While PPO maintained zero rejected orders across all episodes and improved over valid random choice (`6.719 ha` vs `7.013 ha`, 6 wins), it failed the predeclared criteria against transparent heuristics on held-out scenarios (`6.719 ha` PPO vs `6.531 ha` greedy heuristic and `6.281 ha` containment heuristic, losing 4/8 cases vs greedy and 3/8 vs containment). These eight scenarios are now opened and cannot be reused as an untouched benchmark; shortcomings and trajectory divergences are documented in `docs/EVALUATION.md`.
- Implemented Recurrent PPO with LSTM temporal memory core (`recurrent=True`) and Backpropagation Through Time (BPTT) in `nightjar.model` and `nightjar.ppo` to resolve delayed observation amnesia, combined with procedural domain randomization (`generate_domain_randomized_scenario`) in `nightjar.discrimination_family` to prevent geometric overfitting. Pure reinforcement learning without behavioral cloning or imitation.
- Implemented Tactical Action Abstraction (`TacticalActionMapper`, `action_mode="tactical"`) providing structured macro-actions (`HOLD`, `RETURN_TO_REFILL`, `ATTACK_HEAD_NEAREST`, `ATTACK_HEAD_LARGEST`, `ATTACK_CONTAINMENT_NEAREST`, `ATTACK_FLANK_LEFT`, `ATTACK_FLANK_RIGHT`, `ATTACK_REAR`, `ATTACK_CLOSEST`). Eliminates cross-map transit oscillation and spatial pixel-hunting. Combined with forward physical Spread Risk Heatmap (Channel 10) in `ObservationTensorizer`, reduced 32-scenario validation mean burned area to `7.4219 ha`, surpassing both `containment_heuristic` (`7.6484 ha`, `0.2266 ha` reduction, 53.1% wins) and greedy `heuristic` (`7.7266 ha`, `0.3047 ha` reduction, 53.1% wins) with 100% zero rejected orders.
- Completed the strictly one-time blind evaluation on 32 fresh, untouched held-out scenarios (`discrimination-reserved-v2-00` to `31`, schema `nightjar-reserved-evaluation-0.2`) with the recurrent tactical PPO policy (`runs/ppo-tactical-v2.pt`). PPO achieved 7.7734 ha mean affected area vs 8.0234 ha greedy heuristic (59.4% wins, 18.8% losses, 95% bootstrap CI [-0.0156, +0.5234] ha) and 8.0312 ha containment heuristic (50.0% wins, 18.8% losses, 95% bootstrap CI [0.0000, +0.5469] ha), with 100% zero rejected orders. Direct diagnostic on the 8 opened v1 cases confirmed resolution of the prior reserved-06 amnesia failure (PPO 5.75 ha vs old PPO 8.25 ha, saving 2.50 ha). Shortcomings and per-scenario records are documented in `docs/EVALUATION.md`.
- Built interactive browser replay viewer (`nightjar.viewer`) to audit observations, orders, and spatial fire perimeters.

**Gate:** comparison against baselines is reproducible, accompanied by raw records and uncertainty intervals; shortcomings are documented. **[CLOSED for provisional synthetic simulator, 2026-09-29]** Transition to Stage 2 (two aircraft coordination) is open.

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
