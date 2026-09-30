# RxCADRE surface-spread experiment

**Status (2026-09-30):** numerical experiment executed; RxCADRE empirical experiment blocked on source-file acquisition. Separate [C064/F19 timed-head](VALIDATION_CSIRO_HEAD.md) and [independent-campaign FireFlux I upper-bound](VALIDATION_FIREFLUX.md) diagnostics have been executed using accessible sources. FireFlux adds conditional negative evidence and a verified filtered main-tower export, not the eligible multi-burn RxCADRE study. No RxCADRE burn has been reconstructed or physically validated. The implemented metric scorer is tested on explicitly synthetic fixtures. This distinction is an acceptance requirement, not an empirical result.

## Immediate finding

The [numerical benchmark](PROPAGATION_VERIFICATION.md) rejects the existing eight-neighbor Dijkstra path at the proposed 5% analytical arrival tolerance. Its maximum point-arrival error is 7.97% even at 1 m resolution. Pyretechnics' Eulerian level-set path is a candidate for the offline surface-spread experiment at fine resolution. Agreement between engines is numerical verification of a common local rate law; it cannot establish physical validity. Production propagation has not been replaced on the strength of this calm, flat benchmark.

## Data and eligibility

Use [Jimenez and Butler's in-situ archive](https://doi.org/10.2737/RDS-2016-0038) for synchronized fire arrival, camera records, sensor positions and local wind, [Seto and Clements' background weather](https://doi.org/10.2737/RDS-2015-0027), and [Ottmar and Restaino's fuel observations](https://doi.org/10.2737/RDS-2014-0028). Original archives returned HTTP 403 from this runtime on 2026-09-30. DataLumos mirrors were then verified through the site's own search: [fire behavior E249600V1](https://doi.org/10.3886/E249600V1), [background weather E249651V1](https://www.datalumos.org/datalumos/project/249651/version/V1/view), [fuels E249691V1](https://www.datalumos.org/datalumos/project/249691/version/V1/view), and [boom-lift thermal observations E249544V1](https://www.datalumos.org/datalumos/project/249544/version/V1/view). The fire-behavior mirror lists the original ZIP filenames, but clicking the supplement download redirects to ICPSR authentication. No raw ZIP was acquired; an account/institutional login is the concrete blocker. Keep originals and generated reports outside Git, and verify archive checksums against the publisher before extraction.

The accessible [Butler et al. paper](https://doi.org/10.1071/WF14154), Tables 1 and 4 and pp. 77–83, reports S5 and L2G, which the authors selected for comparatively uniform fronts. S5 was line-ignited; large burns used successive strip ignitions. Most S5 sensors encountered flank fire. Ignition-to-sensor averages, sensor-to-sensor rates and ocular local rates are different measurements. Published fuel totals do not determine all Rothermel parameters; an ambient wind summary does not determine a time-resolved midflame wind field. The paper therefore supplies a development diagnostic, not three independently eligible burns or sufficient geometry to simulate their fronts. Do not invent missing headings, ignition timestamps, fuel parameters, wind heights or uncertainty intervals from these summaries. Do not treat L2G's three HIPs as three independent burns.

Before inspecting prediction errors, choose one development burn and at least two additional evaluation burns by coverage of:

- a surveyed, timestamped ignition line or first observed front, plus all later deliberate ignitions in the prediction window;
- fuel composition, loading, bed depth, moisture and defensible standard-model mapping or custom fuel parameters, with uncertainty;
- pre-front ambient wind at a known height, measurement convention and conversion to midflame wind; terrain and coordinate transformations;
- independently measured arrival/front positions and uncertainties in that same window.

Record all candidate burns and every exclusion. Choose short intervals in which the forcing representation is supported. Surface fire only; no water, crown fire or spotting. Freeze the development-selected fuel/forcing representation, engine settings, resolution and source/code hashes before opening evaluation observations. Selection is currently **pending**, not frozen. Published summaries already inspected here cannot be called blind observations. If the raw archive lacks two eligible evaluation burns, revise the study openly or obtain another experimental campaign rather than relaxing eligibility after seeing errors.

## Metrics and contract

`nightjar.rxcadre_metrics` scores matched external predictions and observations. It does not download or reconstruct burns, run a propagation engine, fit parameters, or enforce the historical chronology of a freeze. A configuration hash documents identity; a separately recorded freeze and observation-access history establish independence.

The proposed project thresholds are rate nMAE <0.30 and absolute normalized bias <0.20 **for each evaluation burn**. Both divide by that burn's mean observed rate; samples have equal weight within a burn. These are research decision thresholds, not literature standards. Arrival MAE (seconds), projected front-position MAE (metres), uncertainty-interval distance and every residual are reported; arrival/position acceptance limits still require instrument and window uncertainty. Physical acceptance also requires the numerical gates and credible input uncertainty analysis.

Observation JSON schema `rxcadre-observations-0.1` requires `crs`, `time_origin_utc`, integer `seed`, `units` exactly `{"rate":"m/s","arrival":"s","position":"m"}`, and `records`. Each record requires `burn_id`, `sample_id`, `split` (`development` or `evaluation`), `source`, `window_start_s`, `window_end_s`, `direction_deg` (clockwise from north), `rate_definition`, `rate_m_s`, `rate_interval_m_s` (documented lower/upper bounds, not guessed confidence limits), `arrival_s`, and `front_position_m` (distance along the same declared transect). Times are elapsed seconds from the UTC origin. Observed rates must be positive; missing measurements must be excluded with scientific justification before freezing the scored set.

Prediction JSON schema `rxcadre-predictions-0.1` repeats matching metadata and records, with the same keys, windows, direction and rate definition plus predicted rate, arrival and position. It also requires `frozen_configuration_sha256` and `observations_sha256`, the SHA-256 of observation JSON serialized with `json.dumps(observations, sort_keys=True, separators=(",", ":"))`. The latter binds the comparison contract, not proof that the simulator did not read future observations. Engine execution must receive only initial state and permitted forcing. Unreached sensors cannot be silently removed or assigned zero-error arrivals: a prediction with missing/nonfinite results is rejected and must be reported as a model failure separately. Extra/omitted/duplicate samples and mismatched metadata are rejected.

After acquiring and reconstructing eligible data, the scoring command is:

```sh
python -m nightjar.rxcadre_metrics --observations data/raw/rxcadre/observations.json --predictions runs/rxcadre/predictions.json --output runs/rxcadre/metrics.json
```

Those input files do not exist yet; this is the implemented scoring interface, not an executed empirical experiment. The CLI is exercised end-to-end on temporary synthetic contracts by:

```sh
python -m unittest discover -s tests -p 'test_rxcadre_metrics.py' -v
```

Every report includes file/configuration hashes, schema, UTC origin/creation time, CRS, units, seed, split coverage and per-burn results. Completing the command is not a validation pass: the report explicitly leaves physical validity unestablished. No Spanish forest, suppression response or operational transfer claim follows from this experiment.

The rate endpoint covers positive observed spread segments. Stationary/unburned locations need a separate false-spread endpoint; this scorer does not assess them. An empirical result must not silently discard such controls or treat this rate endpoint as full-perimeter validation.

## Implementation checks (2026-09-30)

### Accessible alternative source (2026-09-30)

Nine files were acquired without account authentication from the Firemodels experimental database and FDS validation repository: three experimental CSVs plus one FDS input per case C064/F19, and the experiment chapter. The source revisions are `firemodels/exp@e5de6811036d252b4f9085866f08da66efdda7d3` and `firemodels/fds@bb1fb3a6f90a9a7de16ecf436c0092bfbfe852f2`. All nine stored byte hashes were verified against the acquisition manifest. Raw sources and that manifest are Git-ignored; they are not reconstructed scenarios or empirical results. The [experimental source](https://github.com/firemodels/exp/tree/e5de6811036d252b4f9085866f08da66efdda7d3/CSIRO_Grassland_Fires) provides front-position curves and three head-position/time samples per burn. The [experiment chapter](https://github.com/firemodels/fds/blob/bb1fb3a6f90a9a7de16ecf436c0092bfbfe852f2/Manuals/FDS_Validation_Guide/Experiment_Chapter.tex) separates measured wind/fuel properties from assumptions. FDS input files are model configurations, not observed fire outcomes.

These two burns belong to the already inspected CSIRO Annaburroo campaign, not a new independent campaign or untouched test split. C064's 27/53/100 s and F19's 56/86/138 s snapshots could support an unfitted, restricted grass-front diagnostic once their coordinates, time origin, ignition and measurement uncertainty are resolved. Their prepared head positions differ from `Burn_Period_Data.csv` (for example C064 at 53 s: 42 m versus original cumulative distance 46.7 m). FDS documentation uses C064 moisture 6.3% versus 6.2% in the original plot table, and F19 input wind 4.9 m/s versus 4.8 m/s in its measured-property table. Do not silently blend these definitions or choose the favorable value. Resolve the original/publication/prepared-series relation and wind-height conversion before simulation. Two burns do not satisfy the proposed one-development-plus-two-evaluation RxCADRE design. This source removed the acquisition blocker for a separate limited diagnostic, now [executed and documented](VALIDATION_CSIRO_HEAD.md). Leading-edge curves are open, local x follows initial spread, and elapsed seconds preserve unresolved absolute timezone. Original/published forcing variants and all three observation series are kept separately. FM1 fails both primary cases; the grass relation is an in-campaign reference, not independent acceptance. Full-front reconstruction and quantified input/measurement uncertainty remain open.

With Python 3.12.4 and the pinned standard-fuel dependency, `python -m nightjar.propagation_verification --output runs/propagation-verification.json` completed and recorded the numerical failures and passing fine level-set pair. `python -m unittest discover -s tests -v` ran 179 tests: 175 passed, four skipped because optional external Annaburroo/CEMS data were absent in the isolated checkout. The new seven tests passed, including both engines, an independent octile-distance formula, per-burn failure preservation, exact sample/window matching, nonfinite rejection and an end-to-end scoring CLI on synthetic fixtures. `git diff --check` passed. A code-test pass is distinct from the reported numerical/physical gates.

Documentation updated in this change: README, DATA_AND_VALIDATION, EVALUATION, REFERENCES, ROADMAP, DECISIONS, TRAINING_READINESS, STANDARD_FUEL_PATH and VALIDATION_EMSR578; PROPAGATION_VERIFICATION and this document were added. EMSR578's unsupported drought-percentile, active-duration equivalence and causal suppression/diurnal claims were corrected. Architecture and production interfaces did not change: both new modules are offline experiments and no controller, aircraft or production fire code was modified.
