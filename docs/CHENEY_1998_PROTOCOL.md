# Cheney–Gould–Catchpole (1998) grass candidate

## Frozen retrospective protocol (2026-09-30)

This is an isolated **local potential head-rate** experiment. Observed C064/F19 and FireFlux outcomes were already known before this protocol; no blind validation or fitting is claimed. Production propagation and controllers are unchanged. Numerical arrival error is not included in these local-rate scores. The eight-neighbor analytical failure and calm-only level-set evidence in [PROPAGATION_VERIFICATION](PROPAGATION_VERIFICATION.md) remain intact.

### Equation and source contract

Target: Cheney, Gould and Catchpole (1998), [DOI 10.1071/WF9980001](https://doi.org/10.1071/WF9980001). Its publisher supplies only an abstract without subscribed access; the author-uploaded original PDF could not be acquired (ResearchGate download unavailable/HTTP 403). **The implementation is a reconstruction from full published restatements, not a claim to have checked the original PDF.**

The full corrected coauthor restatement [Cruz et al. (2015), section 2.1.2, pp.126–128, Eqs.6–11](https://www.researchgate.net/publication/281801393_Empirical-based_models_for_predicting_head-fire_rate_of_spread_in_Australian_fuel_types), DOI 10.1080/00049158.2015.1055063 (corrigendum 10.1080/00049158.2015.1080789), supplies the wind coefficients and original curing Eq.10a. It includes original model author Gould; it is an author restatement in a review, not the original 1998 primary article. With U = open wind at 10 m in km/h, M = dead moisture percent of oven-dry mass, C = visually assessed percent dead/fully cured grass, the base speeds in km/h are:

| Pasture condition | U ≤ 5 km/h | U > 5 km/h |
|---|---|---|
| Undisturbed (uncut, very lightly grazed) | 0.054 + 0.269 U | 1.4 + 0.838 (U − 5)^0.844 |
| Cut/grazed (cut or partly trampled) | 0.054 + 0.209 U | 1.1 + 0.715 (U − 5)^0.844 |
| Eaten-out (very heavily grazed) | Unsupported; reject | 0.55 + 0.357 (U − 5)^0.844 |

R = base × φM × φC / 3.6 gives m/s. φM = exp(−0.108 M) for M < 12; otherwise 0.684 − 0.0342 M when U < 10, and 0.547 − 0.0228 M when U ≥ 10. Original φC = 1.120 / [1 + 59.2 exp(−0.124(C − 50))]. Optional moisture estimation is separate: M = 9.58 − 0.205 T + 0.138 RH, with T in °C and RH in percent; measured M is used in CSIRO.

Primary-source cross-checks: [Cruz, Alexander and Kilinc (2022), Appendix A.3, pp.13–14](https://doi.org/10.3390/fire5020055) restates all wind/moisture branches in its wildfire experiment, but A4 prints **0.705** rather than 0.715 and assigns U=5 to the high branch; A7 explicitly uses the **2015** curing function. Neither alteration is silently imported. [Wotton, Alexander and Taylor (2009), GLC-X-10, section 3.2.2, p.9, original Australian curing curve](https://publications.gc.ca/collections/collection_2010/nrcan/Fo123-2-10-2009-eng.pdf) independently prints the original Australian curing formula while developing a different Canadian function. [Cruz et al. (2015), original curing experiments](https://doi.org/10.1071/WF14146), pp.838–848, show limitations of the old curing assumptions; the later function is outside this candidate's scope. Source access date: 2026-09-30. The acquired 2022 PDF is hash-pinned outside Git; readable web restatements are cited by version and locator, without claiming a downloaded-byte hash. Direct original-paper reconciliation remains open.

### Inputs, domain and boundaries

Required inputs are explicit open 10 m wind (km/h), dead moisture (percent dry mass), curing (percent), and one of the three pasture conditions. All must be finite. M is restricted to 2–24%, C to 50–100% including the tabulated endpoints. The experiment imposes U=0–80 km/h as an explicit guard, not a newly established calibration limit. Reject unsupported inputs rather than extrapolating curing below 50%. Eaten-out requires U>5; do not invent its low-wind branch or replace 0.357 with half of 0.715. At U=5 use the 2015 restatement's low branch; at M=12 use the linear branch; at U=10 use the high-wind moisture branch. Rounded linear moisture factors can become negative near extinction; floor them at zero and expose the raw factor. This is an explicit physical nonnegative-speed safeguard. Do not normalize φC(100) to exactly one or smooth coefficient discontinuities. These decisions precede new observed errors.

The law represents potential quasi-steady wide-head spread in Australian continuous grasslands. It does not predict acceleration, head-width evolution, slope, flanks, backing fire, spotting, suppression, or forest transfer. A finite developing head is not an observation of asymptotic potential spread. Fuel load/height are not adjustable coefficients. Low-wind directional consistency and sparse fuel continuity remain limitations. A zero-speed result is retained; arrival time is undefined, never divided by zero.

### Eligibility and comparisons fixed before calculation

[Gould, Cruz and Sullivan (2024), Table 1](https://doi.org/10.1071/WF23100) defines `10mWind.ms` as **profile-based extrapolation** of measured 2 m mean wind, not a direct 10 m measurement. Use that supplied column ×3.6; do not recompute a favorable profile. Whole-burn means are retrospective inputs. `DFMC.%` is measured dead moisture; `Cure.%` is visual curing. E2 is cut with clippings retained; E3 is cut with clippings removed; both map to cut/grazed. T1 maps to undisturbed. Retain the C064 treatment discrepancy even though the model category agrees.

| Case | Fixed input representation | Eligibility |
|---|---|---|
| C064 plot table | U10=5.93 m/s, M=6.2%, C=100%, E2 → cut/grazed | Conditional retrospective diagnostic; finite developing head and development-campaign overlap exclude physical acceptance |
| C064 period-table variant | Same supplied U10 and M; C=95%, E3 → cut/grazed | Separate source sensitivity; do not average or select by error |
| F19 plot and period tables | U10=5.87 m/s, M=5.8%, C=95%, T1 → undisturbed | Conditional diagnostic; after 86 s is secondary; same campaign exclusion |
| FireFlux I | Ambient U2=3 m/s and estimated M=9%; no reconciled open U10 or curing | **Not evaluable in main comparison**; filtered NPZ columns/clock/failure masks remain unresolved |
| RxCADRE | Existing source authentication/input eligibility blocker | Not evaluable; preserve blocker |
| EMSR578 and other CEMS inputs | Inferred fuel, unmatched wind/moisture, interventions, mixed forest fuels | Not evaluable for this grass law |

CSIRO metrics and windows reuse the previous experiment unchanged: digitized leading-edge max-x primary; prepared heads and original cumulative travel separately; C064 27→53→100 s and F19 56→86 s primary; all later F19 records retained as secondary. First head initializes the single constant-rate forecast; no later resets or clipping. Equal-interval rate nMAE and signed bias are normalized by mean observed rate, per burn and source. Research rate gates remain nMAE<30% and |bias|<20%, independent of physical acceptance. Candidate qualifies as a restricted improvement only if **both burns and all three source definitions** pass and reduce nMAE versus the original-input FM1 baseline. Otherwise report failure of that fixed diagnostic or insufficient evidence where inputs/observable do not match. No pooled score or tuning. Preserve all six earlier formulations and their acceptance exclusions.

FireFlux sensitivity alone assumes U2/U10=0.81, the profile conversion reported in [Cruz et al. (2018), Appendix, DOI 10.1071/WF18082](https://doi.org/10.1071/WF18082), also described in the [2017 CSIRO fuel-load experiments](https://doi.org/10.13140/RG.2.2.28866.68804), section 3.1.4. This Australian ratio is **not validated for FireFlux**. Fixed grid: U2=3 or 4 m/s, M=9 or 18.5%, C=95 or 100%, each of the three pasture types. 18.5% is mixed live/dead morning moisture, not contemporaneous dead M. All 24 scenarios are separate arithmetic sensitivities, never alternate eligible forecasts or confidence limits. Contrast with literal 1.61 m/s and retain all prior 4 Mell sensitivity pairs. Do not infer wind/curing from observed rate. No filtered instrument series enters the predictor.

The freeze command writes an exclusive-create protocol with this document hash, model/runner/code and source hashes, exact configuration, UTC timestamp, units, seed=0 and observation-access disclosure. Run requires that unchanged protocol; a mismatch fails. Freeze does not calculate observed prediction errors. Reports and sources stay in ignored `runs/` and `data/raw/`, with new names to preserve prior artifacts. See [the execution record](VALIDATION_CHENEY_1998.md) for commands and results.

### Frozen command contract

Run from the repository root with `requirements-standard.txt` installed (Pyretechnics **2026.8.10**). Use the existing hash-pinned CSIRO and FireFlux sources from their respective documented acquisition commands. Only the new literature PDF is acquired by `--fetch-literature`. No download occurs by default. Custom roots use `--annaburroo`, `--firemodels`, `--fireflux-sources`, and `--literature`; the same roots must be used for freeze and run.

```sh
python -m nightjar.cheney1998_validation freeze --fetch-literature --protocol runs/cheney1998-protocol.json
python -m nightjar.cheney1998_validation run --protocol runs/cheney1998-protocol.json --output runs/cheney1998-comparison.json
python -m unittest discover -s tests -p 'test_cheney1998*.py' -v
```

Protocol and report are created exclusively; existing files are never overwritten. To reproduce with an existing unchanged protocol, use a new output filename. The run rechecks protocol version, timestamp, full definition, runtime, code, this document and source hashes before calculating rates or errors. Freeze checks input/source eligibility without calculating rates or loading outcome series. CLI success means experiment execution, even if rate gates fail. There is no empirical physical-pass exit code.
