# Executed Cheney–Gould–Catchpole (1998) reconstruction

**2026-09-30: restricted improvement over FM1; insufficient evidence for physical acceptance.** The isolated `nightjar.cheney1998` candidate implements three pasture conditions, both wind regimes, moisture branches and the original curing function. It is reconstructed from full published coauthor restatements and primary cross-checks. Direct verification of the original 1998 PDF remains blocked by access; the abstract was not used to supply equations. See the [scientific protocol and source/version reconciliation](CHENEY_1998_PROTOCOL.md).

The protocol was written before new empirical prediction/error calculations, at **2026-09-30T15:30:07.912327+00:00**, after explicitly disclosed prior observation access. Protocol SHA-256: `58d9bfe512c748593689244a277cb15dd4e4fcda47bdbf85c5b048099c372dda`; definition SHA-256: `6668576525687f859b18ecccdac218270e6ab4df52ab8ba2d144e6479a9cef2c`. The definition binds sources, model, runner, prior model code, Python/Pyretechnics versions and the protocol document. Both original artifacts remain outside Git. A subsequent **locator-only amendment** removed an incorrect equation-number label from the Wotton 2009 citation. It changed no equation, input, conversion, metric, gate or result. The final protocol was saved at **2026-09-30T15:42:58.306261+00:00**, SHA-256 `5e2274f25814e4505ba5b5cc968e3382aac2b3fed83bf9145b793276a54047a3`, definition SHA-256 `8b92de10868f5a67512c2231dbab8be6e39daf995747c4edd35b03d06b2a2fd2`. This administrative amendment followed observed new errors; the original scientific choices had already been frozen. Final and offline numeric results are exactly identical to the first run. It does not establish historical blindness. No inputs were adjusted after seeing these new errors.

## CSIRO timed-head comparison

The supplied `10mWind.ms` column is an open 10 m **profile extrapolation**, as defined by the original data authors, not a direct 10 m observation. No new conversion from 2 m or midflame wind is fitted. C064 uses cut/grazed and F19 undisturbed; curing is 100% and 95% respectively. C064's separate period-table variant uses 95% and retains E3 versus plot E2. Both treatment codes are cut grass. Dead moisture stays measured at 6.2%/5.8%. The earlier diagnostic's times, first-head anchor, digitized max-x endpoint, equal-interval metrics and exclusion of post-86 s F19 intervals from primary scores are unchanged.

| Configuration | C064 R (m/s) | C064 nMAE / signed bias | C064 head MAE (m) | F19 R (m/s) | F19 nMAE / signed bias | F19 head MAE (m) |
|---|---:|---:|---:|---:|---:|---:|
| FM1, original inputs | 0.597731 | 49.19% / −49.19% | 28.72 | 2.083990 | 57.29% / +57.29% | 22.77 |
| FM1, Mell inputs | 0.809376 | 31.20% / −31.20% | 18.24 | 2.083990 | 57.29% / +57.29% | 22.77 |
| Anderson 1 / 3, original inputs | 0.597762 | 49.19% / −49.19% | 28.71 | 1.589746 | 19.988% / +19.988% | 7.94 |
| Measured dead, original inputs | 0.366307 | 68.86% / −68.86% | 40.17 | 4.437714 | 234.94% / +234.94% | 93.38 |
| Measured dead, Mell inputs | 0.481357 | 59.09% / −59.09% | 34.48 | 4.437714 | 234.94% / +234.94% | 93.38 |
| Cheney/Mell finite-width reference | 1.106194 | 5.98% / −5.98% | 3.55 | 1.379877 | 4.15% / +4.15% | 1.65 |
| **1998 reconstruction, plot inputs** | **1.231152** | **4.65% / +4.65%** | **2.64** | **1.381167** | **4.25% / +4.25%** | **1.69** |
| 1998 reconstruction, C064 period fuel variant | 1.127299 | 4.18% / −4.18% | 2.50 | — | Same inputs as plot table | — |

The prior six models' complete scores reproduce the archived CSIRO report **exactly**, including all original source series and failures. None was removed, refitted or replaced. The candidate is slightly worse than the previous in-campaign grass reference in F19; it does not dominate every earlier formulation. The predeclared comparison baseline was original-input FM1.

| Candidate/source representation | Rate nMAE | Signed bias | Head MAE (m) | Restricted rate gates |
|---|---:|---:|---:|---|
| C064 plot / digitized max x | 4.65% | +4.65% | 2.64 | Pass |
| C064 plot / prepared head | 8.61% | +8.35% | 4.94 | Pass |
| C064 plot / original cumulative travel | 3.81% | +3.81% | 2.96 | Pass |
| C064 period variant / digitized max x | 4.18% | −4.18% | 2.50 | Pass |
| C064 period variant / prepared head | 8.61% | −0.79% | 2.51 | Pass |
| C064 period variant / original cumulative travel | 4.95% | −4.95% | 2.47 | Pass |
| F19 plot / digitized max x | 4.25% | +4.25% | 1.69 | Pass |
| F19 plot / prepared head | 3.59% | +3.59% | 1.44 | Pass |
| F19 plot / original cumulative travel | 6.24% | +6.24% | 2.44 | Pass |

Both burns and all three observation definitions pass nMAE<30%, |bias|<20%, and improve on FM1, meeting the **restricted improvement** criterion. Every interval residual, arrival proxy and later F19 record remains in the JSON. These series describe different observables, not repeated independent burns or uncertainty intervals. There are only three primary digitized intervals across two development-campaign burns. The observations describe finite developing heads whereas this law estimates potential wide-head spread; small residuals do not verify acceleration or width evolution. `physical_validation_pass` remains null.

## FireFlux eligibility and sensitivity

The main comparison is **not evaluable** for this candidate: reconciled open 10 m wind and curing are absent. No wind-height/column/clock or failure-mask interpretation is inferred from the filtered main-tower NPZ. The prior Mell diagnostic remains a conditional rejection at its upper bound 0.668501 m/s versus literal published 1.61 m/s (58.48% minimum error), with all source discrepancies and four prior sensitivity pairs retained unchanged.

The separate arithmetic sensitivity uses the frozen Australian ratio U2/U10=0.81. It is not a validated conversion for the Texas burn. M=18.5% remains an incompatible mixed live/dead morning sample; C=95/100% and pasture conditions remain assumed. The table displays all 24 scenarios; each cell is **rate m/s (signed bias versus 1.61 m/s)**. Because every prediction is below 1.61, nMAE equals the absolute bias for this single-rate contrast.

| Assumed U2 (m/s) | Assumed M (%) | Assumed C (%) | Undisturbed | Cut/grazed | Eaten-out |
|---:|---:|---:|---:|---:|---:|
| 3 | 9 | 95 | 0.617369 (−61.65%) | 0.517660 (−67.85%) | 0.258542 (−83.94%) |
| 3 | 9 | 100 | 0.674245 (−58.12%) | 0.565349 (−64.89%) | 0.282360 (−82.46%) |
| 3 | 18.5 | 95 | 0.204307 (−87.31%) | 0.171310 (−89.36%) | 0.085560 (−94.69%) |
| 3 | 18.5 | 100 | 0.223129 (−86.14%) | 0.187092 (−88.38%) | 0.093442 (−94.20%) |
| 4 | 9 | 95 | 0.827052 (−48.63%) | 0.696566 (−56.74%) | 0.347870 (−78.39%) |
| 4 | 9 | 100 | 0.903245 (−43.90%) | 0.760737 (−52.75%) | 0.379918 (−76.40%) |
| 4 | 18.5 | 95 | 0.273698 (−83.00%) | 0.230516 (−85.68%) | 0.115121 (−92.85%) |
| 4 | 18.5 | 100 | 0.298912 (−81.43%) | 0.251752 (−84.36%) | 0.125727 (−92.19%) |

All fail the research rate gates under those assumed inputs. Even the fully cured undisturbed U2=3, M=9 scenario is 58.12% low, close to the previous 58.48% bound failure. These are conditional mathematical failures; they do not establish a general rejection of the full family under correctly reconstructed FireFlux inputs. No inverse wind/moisture/curing matching or selection by error is performed. RxCADRE and CEMS remain listed as ineligible with their source/input/domain blockers.

## Reproduce and verify

From the worktree root with Python and `requirements-standard.txt`, using the existing hash-pinned CSIRO/FireFlux sources:

```sh
python -m nightjar.cheney1998_validation freeze --fetch-literature --protocol runs/cheney1998-protocol-final.json
python -m nightjar.cheney1998_validation run --protocol runs/cheney1998-protocol-final.json --output runs/cheney1998-comparison-final.json
python -m nightjar.cheney1998_validation run --protocol runs/cheney1998-protocol-final.json --output runs/cheney1998-comparison-final-offline.json
python -m unittest discover -s tests -p 'test_cheney1998*.py' -v
python -m unittest discover -s tests -q
```

The initial fetch/freeze and comparison completed successfully. The final-version freeze, comparison and offline repeat above also completed. `--fetch-literature` checked the already acquired pinned PDF on the freeze invocation; acquisition had independently succeeded before freezing. Existing artifacts are never overwritten: on a rerun, reuse the unchanged protocol and choose a new report filename. An existing protocol makes the first command refuse overwrite. Custom roots, first-time CSIRO/FireFlux acquisition prerequisites and exact source evidence are in the [protocol](CHENEY_1998_PROTOCOL.md) and earlier validation reports. Downloaded literature under `data/raw/cheney1998/` and JSON results under `runs/` are ignored by Git.

The 19 focused tests pass. Five equation fixtures use independent 50-digit mpmath 1.3.0 calculations, including U=5, M=12, high-wind/higher-moisture, curing endpoints and negative rounded moisture attenuation. Inputs at branch boundaries, unsupported domains, explicit 10 m column selection, source-variant retention, zero-speed arrivals, exclusive artifact creation, freeze mismatch and all sensitivity/baseline retention are tested with deterministic synthetic fixtures. No stochastic confidence or independent physical evidence follows from these software tests. The initially attempted package-style `python -m unittest tests.test_cheney1998 tests.test_cheney1998_validation -v` failed because this environment resolves another `tests` package; the documented discovery command above is verified.

Execution runtime: Python **3.12.4**, Pyretechnics **2026.8.10**; the grass candidate uses only the standard library. Full suite: **224 tests run, 221 passed, 3 skipped** in 81.287 s; an NNPACK unsupported-hardware warning did not fail tests. The final 19 focused tests passed again after the citation-only amendment. `git diff --check` passed; the staged diff is empty. New files and all changes relative to the pre-task manifest were reviewed. Previous implementation/test files and production propagation are byte-identical to their pre-task state.

## Remaining evidence gates

Obtain and reconcile the original 1998 article against the restated coefficients and equality conventions. Recover correctly interpreted open wind, curing, contemporaneous dead moisture and front timing/geometry for FireFlux or another independent grass campaign. Match potential steady head rate to its observable or independently implement and verify finite-head/acceleration treatment before making whole-front claims. The [follow-up source audit](GRASS_INDEPENDENT_EVIDENCE.md) confirms institutional/purchase access for the original paper and identifies promising later experiments without acquiring eligible matched per-burn records. No original equations or results are amended. The separate [extended numerical experiment](PROPAGATION_VERIFICATION.md#extended-frozen-experiment-2026-09-30) passes five homogeneous level-set 2→1 m pairs but fails both layered pairs and the complete suite. Broader rotation/heterogeneity and empirical verification remain open. The production propagator is unchanged and Stage 0 physical validation remains open.

An offline preservation check ran `python -m nightjar.cheney1998_validation run --protocol runs/cheney1998-protocol-final.json --output runs/cheney1998-comparison-preservation-check.json`. Every numerical result/configuration matches the earlier report; only new execution timestamps differ, including the embedded prior CSIRO calculation's `configuration_set_at_utc`. The original frozen protocol remains byte-identical (SHA-256 `5e2274f25814e4505ba5b5cc968e3382aac2b3fed83bf9145b793276a54047a3`), as do all previous source files and result artifacts. This rerun provides reproducibility evidence, not a new empirical test.

## Artifact preservation audit

All previous CSIRO, FireFlux and propagation source/results artifacts remain byte-identical by the saved pre-task SHA-256 inventory. Previous CSIRO scores reproduce exactly; the FireFlux scientific payload reproduces exactly apart from current run timestamp/runtime metadata. The new candidate never overwrites protocols, reports or literature and leaves generated files ignored.

The full suite exposed an existing side effect in `tests/test_evaluate_reserved.py`: its mocked evaluation writes the default `runs/reserved-evaluation/report.json` and `runs/reserved-v2-evaluation/report.json` instead of temporary outputs. These **two unrelated synthetic test reports were overwritten**. Their prior hashes were saved, but prior bytes were not backed up; no matching copy was found, so exact restoration cannot be claimed. All other 608 files in the 610-file prior-artifact inventory are unchanged. No existing test or production file was modified to broaden this task. This preservation limitation is disclosed explicitly; future full-suite runs should isolate or back up those default outputs.

Documents updated in this task: README, CHENEY_1998_PROTOCOL, VALIDATION_CHENEY_1998, DATA_AND_VALIDATION, EVALUATION, REFERENCES, DECISIONS, VALIDATION_CSIRO_HEAD, VALIDATION_FIREFLUX, PROPAGATION_VERIFICATION, ROADMAP and TRAINING_READINESS. Architecture/replay/controller contracts need no change because the candidate is an offline scalar-rate experiment. No commits, pushes or PR changes were made.
