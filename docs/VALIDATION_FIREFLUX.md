# FireFlux I independent-campaign upper-bound diagnostic

**Status (2026-09-30): executed, unfavorable conditional result.** FireFlux I (Texas, 2006) is separate from the Australian 1986 campaign used to develop the Mell/Cheney relation. The unchanged candidate fails a restricted comparison using published ambient wind and estimated moisture. This is a retrospective, one-burn local-law diagnostic with unresolved source and forcing uncertainty; it does not establish physical acceptance, validate a spatial engine, or close Stage 0.

## Selection, sources and observation access

The selected campaign has tall grass, measured wind at a stated height, dated ignition, a photographed first head, and later tower passage observations. Primary evidence is [Clements et al. (2007), BAMS 88, 1369–1382](https://research.fs.usda.gov/download/treesearch/14829.pdf) and [Kochanski et al. (2013), GMD 6, 1109–1126](https://gmd.copernicus.org/articles/6/1109/2013/). Both final PDFs were acquired without authentication and checked by byte hash. This is not an acquisition of the original instrument archive.

The law is [Mell et al. (2007), Eq. 2](https://doi.org/10.1071/WF06002), derived from Cheney's earlier Australian grass experiments. The 2006 Texas burn does not overlap those development experiments. Campaign independence does **not** imply blind observations: the preliminary 0.7–1.3 m/s overview, published 1.61 m/s tower rate, and original photographic/tower times were examined during eligibility review, before computing candidate predictions. All are retained in the observation-access history; no model was fitted on FireFlux.

The protocol was saved before prediction/error calculation at **2026-09-30T13:42:33.422789Z** in ignored `runs/fireflux-protocol-freeze.json`, SHA-256 `b38ad3f9598480d4dadb16f5da55f5595c297f13a3fb2aac0b1db5565aa37b48`. It fixed the candidate, forcing, primary rate, secondary trajectory, sensitivities and existing project thresholds. Its candidate-code hash is checked on protocol-bound reruns. The implementation and source-audit refinements followed this freeze; no scientific input or primary endpoint was changed. Figure 4's apparent timing discrepancy and the filtered export were inspected after freezing and are reported as audits, not replacements for the primary observable.

| Candidate / exclusion | Reason retained or excluded |
|---|---|
| FireFlux I | Selected for a limited published local-rate test; missing effective head width prevents an actual finite-width prediction. |
| Annaburroo C064/F19 | Development-campaign overlap; previous diagnostic retained, excluded from independent acceptance. |
| RxCADRE | Original-source HTTP 403 and authenticated mirrors remain unresolved; no raw ZIP acquired. |
| FireFlux 2005 pilot | Less complete instrumentation; not reconstructed or scored here. |
| FireFlux II | A separate burn with different ignition and forcing; not reconstructed or scored here. |

## Frozen comparison and available uncertainty

For nonnegative wind and positive width the finite-width factor is at most one:

`R(U2, M, W) = (0.165 + 0.534 U2) exp((-0.859 - 2.036 U2)/W) exp(-0.108 M)`

`R_upper(U2, M) = (0.165 + 0.534 U2) exp(-0.108 M)`

Here wind is m/s at 2 m, width is metres, and **M is percent**; the API receives moisture as a fraction. The primary forcing is published ambient `U2 = 3 m/s` and estimated dead moisture `M = 9%` at burn time (Clements, p. 1375). The initial wind profile in Kochanski, p. 1113, separately agrees with 3 m/s near 2 m. The law's cured-grass assumption approximates the reported 3–5% live component. No fitted curing, wind or fuel multiplier is introduced.

No observed effective head width is supplied. Taking its infinite-width limit gives a mathematical **upper bound**, not a forecast or a measured width. If even that bound is too low, every finite width under the same forcing is too low. If a bound is compatible, that cannot accept a finite-width model. The primary observed rate is the literal **1.61 m/s** in Kochanski §5.1, p. 1114; nMAE for this single interval is absolute error divided by this rate. The retained project gates are nMAE <30% and absolute normalized bias <20%, not literature standards. Scores are not pooled with CSIRO or counted as multiple independent burns.

The published 2 m tower wind has a known height, but its sheltering/averaging convention is not established to match Annaburroo's upwind 2 m measurement. Fire-affected horizontal winds exceeded 10 m/s. The constant ambient representation therefore cannot test the candidate under the true evolving front wind. Moisture 9% is estimated from drying, not a contemporaneous dead-only measurement with confidence bounds. The 18.5% morning mixed live/dead sample is retained as a deliberately ineligible moisture substitution; it is not a bound on dead moisture at burn time.

Rate rounding is assessed using 1.605–1.615 m/s, **conditional on rounding to the nearest 0.01**. A separate sensitivity lets each 1 Hz passage endpoint vary by ±1 s. These quantify decimal/sample effects only; instrument response, distance error, wind representativeness and clock/endpoint conventions remain unquantified. Neither is a confidence interval. The preliminary [Clements AMS overview](https://ams.confex.com/ams/7firenortheast/techprogram/paper_127373.htm) supplies 3–4 m/s for a source sensitivity, not a measured interval uncertainty distribution.

## Executed result and preserved discrepancies

| Frozen or secondary forcing | Infinite-width upper rate (m/s) | Finite 385 m surrogate (m/s) | Minimum relative error versus published 1.61 m/s |
|---|---:|---:|---:|
| **Primary: 3 m/s, 9%** | **0.668501** | 0.656513 | **58.48%** |
| Secondary: 4 m/s, 9% | 0.870527 | 0.850407 | 45.93% |
| Ineligible moisture substitution: 3 m/s, 18.5% | 0.239616 | 0.235319 | 85.12% |
| Ineligible moisture substitution: 4 m/s, 18.5% | 0.312030 | 0.304818 | 80.62% |

The 385 m quantity is the published ignition-line length, **not observed effective head width**; its finite-width rates are secondary mathematical scenarios, excluded from physical acceptance. All residuals and source sensitivities are saved. For the primary upper bound, signed normalized bias is −58.48%, and every finite-width bias is at least as negative. Both project rate gates fail. Decimal-rounding sensitivity gives minimum errors **58.35–58.61%**, preserving that conditional failure. Matching 1.61 m/s at 9% moisture would require at least **7.6603 m/s** at 2 m in the infinite-width limit; finite width requires more. This inverse calculation diagnoses the input mismatch; it is not an adjusted forecast.

Kochanski §2 states a 300 m downwind tower separation. `300 / 1.61 = 186.34 s` is a **rate-implied** transit, not a separately measured arrival interval. The bound implies a conditional minimum transit of **448.76 s**. Figure 4 visually appears to place the temperature peaks near 219 s and 460 s: roughly 240 s across the stated distance implies about 1.25 m/s. These approximate visual readings conflict with the literal 1.61 m/s statement and require raw timing/geometry/endpoint reconciliation; they are not a new scored series or a more precise estimate. The published 1.61 m/s remains the frozen primary, openly conditional on that source discrepancy.

The separate original photo trajectory starts at `(t, x) = (152 s, −45 m)` relative to ignition and the main tower (Clements, p. 1375). Advancing only this first head gives maximum predicted `x(190) = −19.60 m`, against the reported tower front passage at 190 s (p. 1377); earliest conditional arrival is 219.31 s. The 195 s 4.5 m plume peak (p. 1376) and Kochanski's 219 s MT peak stay separate, with 24/29 s source differences recorded. No later head resets the trajectory. This photo interval overlaps the last 11 s of eastern walking ignition, which ends at 163 s; it is retained as secondary evidence and excluded from free-spread acceptance. The primary tower interval is after deliberate ignition. No plume peak is silently converted to ground-fire arrival.

The preliminary 0.7–1.3 m/s overview also remains in the access history; its unspecified windows cannot be substituted for the declared primary rate. A single burn and heterogeneous endpoints do not support a confidence interval over campaigns. No IoU, closed perimeter, predicted flank/back spread, suppression or Spanish forest result is produced.

## Acquired filtered observations: separate source audit

[Costes et al., Zenodo record 6372268](https://doi.org/10.5281/zenodo.6372268), published 2022-03-21, CC-BY-4.0, actually supplies a downloadable NPZ with **filtered observations and simulation ensembles**. `MainTower_Anel_25m.npz` was downloaded; its publisher MD5 `e95a3d64ea2759e85ebdcd5eaae78e5a` matched. [Costes et al. (2022), §5.1](https://doi.org/10.1029/2021MS002955) explains the Hodrick–Prescott filtering and five-second temporal standard deviations. These deviations describe fluctuation, not measurement-error confidence bounds.

`nightjar.fireflux_source_audit` reads only `TimeMean`, `ObsTInterp`, `ObsHWSInterp`, `ObsWInterp` and their `Std` arrays, with pickle disabled. It checks shapes, source hashes and finite values. There are **219** one-second points, from −97.49035 to +120.50965 s on a processed passage-centered axis; observation shapes are 219×6 for temperature and 219×4 for each wind quantity. Simulation `ROS`, `ArrivalTime`, ensemble predictions and object arrays are excluded. No statistic is fed back into the frozen candidate comparison.

This export has main-tower records only, no uncentered clock and no explicit column-to-height labels. Failed-sensor masks are not decoded: finite interpolated values do not establish valid measurements at every timestamp. It therefore does not reconcile the two-tower or absolute timing discrepancies or establish a time-resolved front wind. Downloading it is not a claim to possess raw 20 Hz signals, measured front widths or complete perimeters. The audit preserves unknown mappings instead of guessing them.

## Reproduction and artifacts

Use an environment with NumPy (`requirements-standard.txt`) and `curl`. All raw sources and generated reports remain ignored. Acquire the two pinned primary PDFs and execute the comparison:

```sh
python -m nightjar.fireflux_validation --fetch-sources --output runs/fireflux-upper-bound.json
python -m nightjar.fireflux_validation --output runs/fireflux-upper-bound-offline.json
python -m nightjar.fireflux_source_audit --fetch-sources --output runs/fireflux-source-audit.json
python -m nightjar.fireflux_source_audit --output runs/fireflux-source-audit-offline.json
python -m unittest tests.test_fireflux_validation tests.test_fireflux_source_audit -v
python -m unittest discover -s tests -q
```

`--sources` selects a source directory. Fetch refuses to overwrite changed sources. Its inventory records verification time and which files were downloaded during that call; cached verification is not relabeled a new acquisition. Default execution never downloads; hash changes require review. Optional `--protocol runs/fireflux-protocol-freeze.json` verifies the local historical freeze, including the unchanged candidate-code hash, and records its access history. That ignored artifact is absent in a fresh checkout; the ordinary commands reproduce the numeric comparison without pretending to recreate its historical freeze. JSON success reports experiment completion, not acceptance; `physical_validation_pass` is always null.

| Acquired file | SHA-256 |
|---|---|
| `clements2007.pdf` | `861a38a211887d7d06115506b557be39975f0bc6f0ce83584a4ddbde184a3053` |
| `kochanski2013.pdf` | `7317593e559aea86030c6b2568656d69086db211c88cea3e08f12fa84bb604ae` |
| `MainTower_Anel_25m.npz` | `2c576a05c2c15d1b3fb322455b411038466ce04540ce1b6989798903b2443321` |
| `costes2022.pdf` | `080dc7859c7341829c01671811e24cfb31f1e96efff0b7a930c2ec860370706f` |

The rate report stores UTC creation/experiment origin, local transect coordinates, units, deterministic seed 0, source/configuration/code/observation hashes and every discrepancy. Unlike Annaburroo's unresolved local clocks, FireFlux explicitly states CST: 12:43:30 CST on 2006-02-23 converts to 18:43:30 UTC. This offset conversion does not solve inconsistent event clocks. The separate filtered audit correctly leaves its absolute UTC origin and coordinate/height mapping unresolved.

All commands above completed with Python 3.12.4 and NumPy 1.26.4. The 15 focused tests passed, including actual pinned PDFs/filtered observations, finite-width upper bounds, protocol mismatch rejection, no head resets, accurate cached verification metadata and exclusion of simulated/pickled arrays. The full suite ran 205 tests: 202 passed and three skipped for absent external files. Fresh-directory acquisition verified downloads independently of cached files; both offline scientific reports matched their acquired-source runs after removing creation timestamps. These are implementation/reproduction checks, not a physical pass.

## Consequence for acceptance

The prior CSIRO rerun reproduced all rates, observations and scores exactly, including FM1's 49.19%/57.29% failures; the focused 18 tests passed and exact legacy FM1 parity was rechecked in 75 configurations. Their favorable grass-reference residuals remain in-campaign evidence only. FireFlux adds independent-campaign **negative transfer evidence for the frozen ambient-static grass representation**, not independent validation of a complete propagator or intrinsic rejection of the equation for all wind conventions.

Before acceptance, resolve published rate versus tower timing/geometry, obtain window-matched ambient/front wind with a defensible convention and uncertainty, dead-only moisture and effective width, and obtain further independent burns. The [separate numerical extension](PROPAGATION_VERIFICATION.md#extended-frozen-experiment-2026-09-30) passes five homogeneous level-set 2→1 m pairs including wind/slope but fails both layered pairs and the full suite. Broader rotations/terrain and two-dimensional heterogeneity remain open; changing Dijkstra cannot fix this no-grid rate/input disagreement. Production propagation is unchanged and Stage 0 remains open.

## Follow-up: complete grass-system reconstruction (2026-09-30)

The [Cheney 1998 reconstruction](VALIDATION_CHENEY_1998.md) leaves this upper-bound failure and source audit intact. It requires open 10 m wind and curing: FireFlux is **not evaluable** in its main comparison with currently reconciled inputs. An explicitly assumed Australian U 2/U 10=0.81 conversion and fixed moisture/curing/pasture grid give 24 separate sensitivities; all fail the literal 1.61 m/s contrast. The fully cured undisturbed U 2=3 m/s,M=9% sensitivity is 0.674245 m/s (58.12% low), close to the unchanged Mell bound. These assumed-input failures do not resolve wind conventions, filtered columns, clocks, masks or observation disagreements, and do not reject the whole model family under correctly reconstructed inputs.
