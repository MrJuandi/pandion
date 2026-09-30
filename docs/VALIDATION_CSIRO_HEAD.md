# CSIRO C064/F19 timed head-advance diagnostic

**Status (2026-09-30): executed on experimental observations, without fitting.** This is a restricted retrospective comparison of local spread laws with measured head advance. It provides negative evidence for transferring FM1 to these burns and a candidate grass-specific reference. It does not validate complete perimeters, a spatial propagator, Spanish forest fires or suppression. Stage 0 remains open.

## Observations and reconciliation

Sources are the original [Annaburroo collection](https://data.csiro.au/collection/csiro:58746), version 1, archive file `64873323`; the experimental CSVs in [Firemodels](https://github.com/firemodels/exp/tree/e5de6811036d252b4f9085866f08da66efdda7d3/CSIRO_Grassland_Fires), revision `e5de6811036d252b4f9085866f08da66efdda7d3`; and [Mell et al. (2007)](https://doi.org/10.1071/WF06002), Table 1, pp. 16–18 and Figure 16. Raw files and generated reports remain outside Git. The runner verifies all six CSV byte hashes before running; a source change requires review rather than automatic acceptance.

Figure 16 was inspected visually. Its experimental symbols represent the **leading edge** of the fire, not a closed burned polygon. The local x axis follows the initial direction of spread, in metres; no geographic CRS or north bearing is inferred. The primary observable is the largest x coordinate among each observed leading-edge curve's samples. It is a sampled projected head location, with unquantified digitization/measurement error. No IoU is calculated by closing these open curves.

| Burn | Elapsed time since ignition (s) | Digitized max x (m) | Prepared scalar head (m) | Original cumulative distance (m) |
|---|---:|---:|---:|---:|
| C064 | 27 | 15.035 | 15 | 15.5 |
| C064 | 53 | 46.329 | 42 | 46.7 |
| C064 | 85 | — | — | 82.9 |
| C064 | 100 | 100.350 | 100 | 101.3 |
| F19 | 56 | 79.180 | 80 | 71 |
| F19 | 86 | 118.927 | 120 | 110 |
| F19 | 121 | — | — | 191.5 |
| F19 | 138 | 201.893 | 200 | — |

These are **three separate series**, retained and scored separately. Prepared scalar heads are not exact maxima of the supplied curves. Original `CumDist.m` is cumulative head travel; `FireDir.deg` records head-travel bearing, not ambient wind orientation. For F19, `IncDist.m` also differs from the subtraction of successive cumulative distances: 45.5 versus 39 m at 86 s, and 84.5 versus 81.5 m at 121 s. The original audit records the supplied values; rates for that secondary series are explicitly calculated from cumulative-distance differences rather than silently substituting reported ROS. The series cannot be combined into one supposedly more precise observation.

Mell et al. report a wind shift **after 86 s in F19** and deliberate ignition lasting until 56 s. Initialization therefore uses its observed 56 s head. The 56→86 s interval is primary; the later 86→138 s curve/scalar interval and 86→121 s original interval remain visible as secondary results under unsupported constant forcing. C064 initializes from 27 s; primary intervals are 27→53 and 53→100 s. Boundary-touching endpoints remain included and identified as potentially censored by firebreaks; predictions are never clipped to the observed plot boundary to conceal overrun.

Original ignition clocks are 1986-08-18 12:56:00 (C064) and 1986-08-20 12:56:00 (F19). Their timezone is unresolved. The report preserves local clocks, elapsed seconds and a null experimental UTC origin rather than inventing UTC conversion. Report creation/configuration timestamps are UTC. Schema, coordinate convention, units and deterministic seed `0` are persisted. This offline diagnostic is not a production geographic scenario.

## Inputs and predictions

No ROS, RSS or later observed head position is used to set rates. A pure forecast receives the first head, a fixed rate, and later time: `x(t) = x_first + R * (t - t_first)`. It never resets at later observed heads. This isolates the **local spread law** without a grid discretization error or invented ignition-line geometry. The resulting conditional arrival residual is the time at which that same constant-rate trajectory reaches each observed head coordinate; it is not an independent sensor-arrival measurement.

| Input | C064 original tables | C064 Mell comparison | F19 original / Mell |
|---|---:|---:|---:|
| Dry load (kg/m²) | 0.283 | 0.283 | 0.313 |
| Bed depth (m) | 0.21 | 0.21 | 0.51 |
| SAV (m⁻¹) | 9770 | 9770 | 12240 |
| Dead moisture (fraction) | 0.062 | 0.063 | 0.058 |
| Midflame wind (m/s) | 2.4 | 2.8 | 4.4 |

Original SAV is `SAVR.cm * 100`. Original midflame wind is profile-derived from measured 2 m wind, not a direct spatial wind field. The alternative moisture/wind values are retained as published source variants, not confidence bounds. C064 plot versus period tables also disagree on treatment (`E2`/`E3`) and curing (100%/95%); no favorable treatment or fitted curing correction is selected.

Six unfitted configurations are retained:

- Existing FM1 with original moisture/midflame wind, and FM1 with the separate Mell inputs. Legacy FM1 calculations are preserved exactly.
- Pyretechnics 2026.8.10 Anderson model 1 for C064 and 3 for F19, following the [pinned FDS chapter's mapping](https://github.com/firemodels/fds/blob/bb1fb3a6f90a9a7de16ecf436c0092bfbfe852f2/Manuals/FDS_Validation_Guide/Experiment_Chapter.tex). Measured midflame wind enters directly; measured 2 m wind is never relabeled midflame. Flat terrain and the engine's default wind limit are used. Both models have load only in the dead 1 h class, so other moisture classes do not affect these rates.
- Rothermel/Albini with measured load/depth/SAV, using original and Mell moisture/wind separately. The published comparison's extinction moisture is 22% and heat is 18608 kJ/kg. A single dead class approximates all measured load as dead; particle density 32 lb/ft³ and mineral fractions 0.0555/0.01 remain standard assumptions. This is a transparent approximation, not a complete calibrated field-fuel description or exact BehavePlus reproduction. Its Mell-input predictions, 0.481 and 4.438 m/s, are close to that paper's approximately 0.47 and 4.4 m/s.
- `cheney_mell_reference`: the published finite-head-width grass relation in Mell et al., Eq. 2, `R = (0.165 + 0.534 U2) exp((-0.859 - 2.036 U2)/W) exp(-0.108 M)`, where `U2` is 2 m wind in m/s, `W` is metres and `M` is **percent** dead moisture. Original inputs are U2=4.67/4.83 m/s and effective width=50/175 m. It assumes cured grass and applies no invented 95% curing correction. This relation derives from the same experimental campaign; it is an in-campaign reference, **excluded from acceptance**. It is not an implementation of the complete operational Cheney (1998) system.

Whole-burn average wind and effective width are **retrospective forcing**. They are not inputs causally available at initialization to a controller or operational forecast. Configurations and rates are mechanically set before the runner loads outcome series, but all observations had already been inspected during study design. No blind split, independent parameter selection, or historical pre-observation freeze is claimed.

## Executed results

Primary rate nMAE and normalized signed bias divide equal-interval mean error by the mean observed interval rate, separately for each burn. The same proposed project thresholds are retained: nMAE <30% and absolute bias <20%. They are research thresholds, not published standards or a complete physical-acceptance criterion. Initial observations are excluded from scoring.

| Configuration | C064 rate (m/s) | C064 rate nMAE / bias | C064 head-position MAE (m) | F19 rate (m/s) | F19 rate nMAE / bias | F19 head-position MAE (m) |
|---|---:|---:|---:|---:|---:|---:|
| FM1, original inputs | 0.598 | 49.19% / −49.19% | 28.72 | 2.084 | 57.29% / +57.29% | 22.77 |
| FM1, Mell inputs | 0.809 | 31.20% / −31.20% | 18.24 | 2.084 | 57.29% / +57.29% | 22.77 |
| Anderson 1 / 3, original inputs | 0.598 | 49.19% / −49.19% | 28.71 | 1.590 | 19.988% / +19.988% | 7.94 |
| Measured dead, original inputs | 0.366 | 68.86% / −68.86% | 40.17 | 4.438 | 234.94% / +234.94% | 93.38 |
| Measured dead, Mell inputs | 0.481 | 59.09% / −59.09% | 34.48 | 4.438 | 234.94% / +234.94% | 93.38 |
| Grass relation, in-campaign reference | 1.106 | 5.98% / −5.98% | 3.55 | 1.380 | 4.15% / +4.15% | 1.65 |

Digitized mean observed interval rates are C064 **1.176493 m/s** and F19 **1.324921 m/s**. Every residual and secondary series is retained in the JSON report, including after-shift F19 results. No burn-pooled score hides C064 failure.

FM1 fails both rate gates in both source-input variants. The tall-grass model is borderline for F19: digitized curves and prepared scalar heads give biases 19.988% and 19.231%, while original cumulative travel gives **22.288%** and fails. These definitions are not interchangeable; the gate is not robust to the available source representation. C064 fails for Anderson 1 under all three series. Neither standard-fuel mapping passes both burns. Measured-fuel substitution does not fix the law's disagreement.

The grass relation's small residuals identify a scientifically motivated **candidate for development**, not independent validation: its originating campaign overlaps this evaluation, and the retrospective width is observed rather than predicted. A different campaign with timed fronts, known forcing and quantified uncertainty is required before accepting it. No uncertainty interval is invented from two burns or from the difference between source series.

## Reproduction and checks

Use Python with `requirements-standard.txt` installed and `curl` available. Explicit acquisition of the two original tables and four pinned experimental CSVs, followed by the comparison:

```sh
python -m nightjar.csiro_head_validation --fetch-sources --output runs/csiro-head-diagnostic.json
```

Offline rerun with already acquired files:

```sh
python -m nightjar.csiro_head_validation --output runs/csiro-head-diagnostic.json
python -m unittest tests.test_csiro_head_validation tests.test_rothermel -v
python -m unittest discover -s tests -q
```

Input roots default to `data/raw/annaburroo` and `data/raw/csiro-firemodels`; `--annaburroo` and `--firemodels` select existing roots. Default execution never downloads. Reports under `runs/` and raw sources under `data/raw/` are Git-ignored. Six source hashes, configuration and code hashes, engine version, units, clocks, observations, source differences and every score are persisted; `physical_validation_pass` is explicitly null. CLI success means the experiment completed, not that rate gates passed.

The fetch-and-run command and offline command completed with Python 3.12.4 and Pyretechnics 2026.8.10. The 18 focused tests passed, including actual pinned sources, ragged front parsing, no resets, input-column whitelisting, post-shift exclusion with residual retention, unit conversions and FM1 behavior. A separate comparison against the base revision confirmed bit-for-bit FM1 component and directional parity in 75 moisture/wind/slope configurations. Full-suite evidence is recorded in EVALUATION.md.

## Decision and remaining work

The follow-up [FireFlux I comparison](VALIDATION_FIREFLUX.md) now executes an unfitted upper-bound contrast on an independent Texas campaign. The unchanged relation under published ambient wind and estimated moisture fails by at least 58.48% versus the declared published rate; actual wind convention and source timing/geometry remain unresolved. The favorable C064/F19 reference is therefore still not promoted to production. Continue with source/input reconciliation and additional independently sourced burns. The [separate spatial extension](PROPAGATION_VERIFICATION.md#extended-frozen-experiment-2026-09-30) now passes five homogeneous level-set 2→1 m pairs including wind/slope but fails both layered pairs and the full suite; broader rotation/terrain and two-dimensional heterogeneity remain open. RxCADRE acquisition/eligibility remains an independent route, subject to its existing authentication blocker. Synthetic controller work can continue under its stated provisional simulator assumptions; physical and operational claims remain gated. This experiment isolates a rate-law mismatch and source sensitivity that changing Dijkstra alone cannot fix.

Documentation updated: README, DATA_AND_VALIDATION, EVALUATION, REFERENCES, ROADMAP, DECISIONS, TRAINING_READINESS, STANDARD_FUEL_PATH, VALIDATION_RXCADRE and this report. Existing EMSR578/numerical-document corrections remain in the same uncommitted change. Controller, resource execution and production spread interfaces are unchanged; the shared FM1 equations were refactored with exact legacy parity to expose an SI offline fuel calculation.

## Follow-up: full grass-system reconstruction (2026-09-30)

The [separate frozen Cheney 1998 experiment](VALIDATION_CHENEY_1998.md) preserves this diagnostic and all six formulations. Using source-provided 10 m open-wind extrapolations, measured moisture and original curing, it gives 4.65%/4.25% digitized-head nMAE and passes the restricted gates across all three source definitions. C064's 95% curing/treatment variant is retained separately. The primary observed finite-head windows are unchanged. This improvement over FM1 remains retrospective and in the model-development campaign; it does not validate asymptotic head-width evolution or independently accept a physical model. Original 1998 PDF verification remains open; full restatement/version evidence is explicit.
