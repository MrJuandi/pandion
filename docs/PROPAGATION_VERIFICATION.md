# Numerical propagation verification

`nightjar.propagation_verification` is an isolated, deterministic numerical benchmark for the existing standard-fuel Dijkstra propagation path. It does not alter simulator, controller, or replay interfaces, and it does not use RxCADRE observations. The suite compares calm homogeneous point arrivals with the analytic Euclidean point-source time, checks arrivals at fixed physical coordinates over four grid sizes, and runs the same calm inputs through Pyretechnics' Eulerian level-set engine.

## Reproduce

From the repository root, use a Python environment with the dependencies from `requirements-standard.txt` installed and run:

```sh
python -m nightjar.propagation_verification --output runs/propagation-verification.json
```

The command writes a JSON report with its UTC creation time, fixed synthetic time origin, `LOCAL_METERS` CRS, units, seed policy and sensitivity probe, package versions, configuration and source-code SHA-256 values, per-case results, and explicit pass/fail classifications. `runs/` is ignored by Git. The process exits successfully when the benchmark completes even when checks fail; inspect `overall_pass`, `checks`, and `resolution_pair_acceptance` in the report. Use `--skip-levelset` only to run the Dijkstra and analytic portion.

Focused invariants and classification tests:

```sh
python -m unittest discover -s tests -p 'test_propagation_verification.py'
```

## Case definition and limits

Inputs use Scott & Burgan fuel model 101, the fixed moisture fractions recorded in the report, zero midflame wind, flat elevation, zero canopy values, and disabled spotting. The point ignition is centered at local coordinate (0, 0). Targets are (40, 10), (40, 20), (40, 40), and (80, 0) m, evaluated on 10, 5, 2, and 1 m grids. All target coordinates align to all four cell sizes. The report records target distance divided by cell size (`d / R`) for each case. The analytic reference is distance divided by the common still-air rate; it treats the ignition as a mathematical point and does not subtract a burned-cell radius.

The level-set path receives the same fuel and moisture values, no 10 m wind, and the same single-cell ignition. It uses Pyretechnics' canonical minutes, metres, and 10 m wind in km/h (zero here); arrival times are converted back to seconds. The run checks that its observed positive spread-rate values agree with the Dijkstra still-air rate within 1%. The report computes adjacent-grid arrival changes for both engines and includes a resolution-pair gate requiring every target to meet the analytical limit at both resolutions and the maximum pairwise change to meet the refinement limit. This is a numerical cross-check between two discretizations, not independent physical validation. The level-set mesh is confined to the same finite domain; no rotated grids, wind-driven or sloped fronts, nonuniform fuels, or empirical RxCADRE metrics are included. Empirical nMAE <30% and absolute bias <20% are recorded as future acceptance criteria and are not implemented here.

The 5% analytical arrival threshold intentionally remains strict. On the recorded run, the Dijkstra off-axis (40, 20) m target is about 7.97% late at all resolutions, and the 4:1 off-axis target also exceeds 5%. These proportional errors persist as cell size changes because the eight-neighbor graph uses the same directional path proportions for coordinates aligned to every grid. The unchanged arrivals produce a 0% adjacent-grid metric change, which passes the stated refinement comparison but does not establish convergence to the Euclidean solution. It exposes a spatial-direction bias that ordinary refinement on these aligned cases cannot remove.

The Eulerian level-set maximum analytical errors in the recorded run were approximately 14.19%, 7.89%, 3.31%, and 1.78% at 10, 5, 2, and 1 m respectively. The coarser level-set cases therefore also fail the 5% check, while the 2 m and 1 m cases pass. Its largest adjacent-grid change is 6.38% from 10 to 5 m, so that refinement pair fails. The 5-to-2 m pair has a 4.58% maximum change but fails the two-resolution analytical gate because the 5 m level-set results exceed 5%. The 2-to-1 m pair passes both gates: maximum analytical error is below 5% at both resolutions and its maximum arrival change is 1.53%. This identifies 2 m to 1 m as the first tested level-set pair satisfying both numerical criteria, not as an empirical validation resolution. Its finite-difference front and point-cell initialization effects remain part of those measurements.

## Recorded run

Evidence was generated on 2026-09-30 using Pyretechnics 2026.8.10, NumPy 1.26.4, SciPy 1.12.0, and Python 3.12.4 on macOS x86_64. The successful command completed all 16 grid/target cases in each engine. Results: Dijkstra analytical check failed (maximum relative error about 7.97%); level-set analytical check failed (maximum about 14.19%); Dijkstra adjacent-grid refinement passed (maximum metric change 0%); level-set global adjacent-grid refinement failed (maximum 6.38%). The level-set 2-to-1 m pair passed both the analytical and refinement gates (maximum change 1.53%). The JSON reports each engine's measured ROS; its relative difference from the Dijkstra ROS is approximately 0.0000084%, well inside the 1% alignment check. A seed-sensitivity probe comparing seed 0 with seed 1 produced 0 s arrival change; the seed is metadata only because no random process is used. The failed global checks are reported without excluding cases or changing tolerances. This output is a synthetic numerical verification result, not a claim of empirical RxCADRE validation.

## Separate local-rate evidence

The [Cheney 1998 grass comparison](VALIDATION_CHENEY_1998.md) evaluates local potential head speeds through constant-rate trajectories with no grid. Its small CSIRO residuals cannot cancel numerical arrival errors. The extended tests below measure wind, slope and layered-fuel behavior separately. Production propagation is unchanged.

## Extended frozen experiment (2026-09-30)

`nightjar.propagation_extended_verification` adds an isolated benchmark without editing either production solver. Its [scientific protocol](PROPAGATION_EXTENDED_PROTOCOL.md) fixes all inputs, targets, analytic references, initialization, three grids and acceptance criteria before new errors. The executed freeze was created at **2026-09-30T17:00:03.223522+00:00**, SHA-256 `6f3c92e9946c99f478e03d55026003079079cd4867f010bf58b82a261c83019f`. Runtime: Python 3.12.4, Pyretechnics 2026.8.10, NumPy 1.26.4, SciPy 1.12.0, macOS x86_64. The freeze hashes readable package sources and loaded extension binaries, repository code and the protocol. Earlier experiment outputs and configurations are retained.

Commands verified from the worktree root:

```sh
python -m nightjar.propagation_extended_verification freeze --protocol runs/propagation-extended-protocol.json
python -m nightjar.propagation_extended_verification run --protocol runs/propagation-extended-protocol.json --output runs/propagation-extended-verification.json
python -m nightjar.propagation_extended_verification run --protocol runs/propagation-extended-protocol.json --output runs/propagation-extended-verification-repeat.json
python -m unittest discover -s tests -p 'test_propagation_extended_verification.py' -v
```

All outputs use exclusive creation. On rerun, retain the existing protocol and choose a new report filename; recreating a protocol also needs a new filename. Code/runtime/configuration changes invalidate the old freeze. A successful process means execution completed: inspect `overall_numerical_pass`, `checks` and `resolution_pair_acceptance`. A failed scientific gate does not change the process exit status. `physical_validation_pass` is always null.

Maximum absolute relative arrival errors across every retained target, in percent:

| Case | Dijkstra 4 / 2 / 1 m | Level set 4 / 2 / 1 m | Level-set 2→1 m pair |
|---|---:|---:|---|
| Calm control | 7.9669 / 7.9669 / 7.9669 | 6.5958 / 3.5188 / 2.0213 | Pass |
| East wind | 15.1938 / 15.1938 / 15.1938 | 6.2533 / 3.2685 / 2.0209 | Pass |
| Northeast wind | 16.0785 / 16.0785 / 16.0785 | 4.5077 / 2.0159 / 1.4274 | Pass |
| East-rising plane | 10.5573 / 10.5573 / 10.5573 | 6.3308 / 3.3107 / 2.0152 | Pass |
| North wind on east-rising plane | 10.5573 / 10.5573 / 10.5573 | 4.8883 / 2.2027 / 1.5319 | Pass |
| Layered 101→104 | 3.4694 / 1.7347 / 0.8674 | 12.5732 / 6.6402 / 3.6059 | Fail: 2 m analytical error |
| Layered 104→101 | 5.3126 / 2.6563 / 1.3282 | 12.5732 / 6.6402 / 3.6059 | Fail: 2 m analytical error |

All **156** primary engine/target/grid records reached their targets; execution errors: **0**. The report retains 104 adjacent-grid comparisons, 28 resolution-pair decisions and 20 domain probes. Overall numerical acceptance is **false** for each candidate when all cases/resolutions are required. Dijkstra fails all five point cases at every resolution. Its layered 2→1 m pairs pass; the 4→2 m pair passes only for 101→104. For level set, the 4→2 m pair passes only northeast wind and combined wind/slope. Its five homogeneous 2→1 m pairs pass; neither layered pair passes.

The repeat report matches every field except its UTC creation time. The focused command passes **9 tests**; `python -m unittest discover -s tests -q` completes **233 tests: 230 passed, 3 optional external-data skips**. `git diff --check` passes. Full-suite tests rewrite two pre-existing ignored synthetic evaluation reports; both were restored byte-for-byte from the pre-run backup. All prior code, tests, raw sources and generated results remain unchanged. Documentation changes and the new isolated runner/tests constitute this continuation.

The largest adjacent-grid change for level set is **5.9331%** at 4→2 m, and **3.0343%** at 2→1 m. Refinement alone would pass the latter even though its 2 m layered arrival errors fail: retain the conjunctive gate. The domain probe (120→160 m half-extent, 1 m cells, east wind and both layer orders) changes Dijkstra arrivals by **0%** and level-set arrivals by at most **0.1450%**, below the fixed 1% gate. This limited probe does not establish boundary independence for every case.

The constant-slope Dijkstra axial arrival is early by **10.5573%**, equal to `1−1/sqrt(1+0.5²)`: it combines horizontal edge distances with rates along the sloping surface. Wind also exposes directional routing error. Both are numerical geometry/coupling effects in the existing path; no wind, moisture, fuel or timing parameter is fitted to remove them. Errors in the layered level-set case include the sampled interface and line initialization. Its largest error is at x=20 m, before the interface, so it cannot be attributed solely to fuel-transition treatment. At 1 m both layered cases meet the analytical gate, but finer paired verification is still needed; no post-result 0.5 m case or corrected initial offset is introduced.

**Conclusion: numerical improvement in the tested homogeneous wind/slope cases, failure of the full frozen suite, and insufficient evidence for production replacement or physical acceptance.** The references independently solve spatial ellipse and layered travel geometry but share the same local Rothermel constitutive law. They do not establish empirical rate accuracy. These tests cover two wind directions, one uniform slope and two calm one-dimensional layer orders; arbitrary rotated grids, curved terrain, two-dimensional heterogeneity/refraction, nonburnable barriers, temporal forcing and spotting remain unverified.
