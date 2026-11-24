# Reproducible research workspace

This workflow adds deterministic scenario derivation, current-policy inference,
multi-aircraft replay, artifact indexing and incident-group evaluation. It does
not change the provisional spread or suppression assumptions.

## Scenario identity and experimental seeds

Environment-family 0.2 keeps creation time in the family manifest, outside the
scenario payload. Terrain acquisition dates, weather capture/delivery times,
source hashes and scientific inputs remain part of the scenario. Generation
clocks named `created_at_utc` and `generated_at_utc` are removed from newly
derived scenario payloads, including nested terrain metadata.

Ignition and multi-ignition family 0.2 accept `--seed` (default 0). Their draws
use the parent content digest, experimental seed and variant index. With the
same inputs, settings and seed, regeneration at another wall-clock time gives
the same environmental scenarios, foci, wind and scenario hashes. Change the
experimental seed explicitly to request another draw.

```sh
python -m pandion.multi_ignition_family --scenarios-dir runs/environment-family \
  --output-dir runs/multi-family-seed42 --seed 42
python -m pandion.ignition_family --scenarios-dir runs/environment-family \
  --output-dir runs/ignition-family-seed42 --seed 42
```

Existing artifacts are not rewritten. Their complete-payload `scenario_hash`
and manifest checks remain unchanged. The additional `content_sha256` ignores
generation clocks, while preserving scientific timestamps. A child retains
its original parent's full hash for lineage: different legacy parent artifacts
can therefore share a content digest and a draw while retaining distinct full
child hashes. The 0.2 generator is a new experimental configuration, not an
exact reproduction of the 0.1 draw algorithm.

## Current checkpoint inference

Workspace and evaluation use the same model factory and rollout executor.
Supported configurations are legacy tactical, image/cell inputs, tactical/
spatial/parametric actions, recurrence, ROS/thermal channels, raw slot features,
forced-step skipping and multiple aircraft. Inference collects no PPO training
records and performs no weight updates.

```sh
python -m pandion.workspace --scenarios-dir runs/family --output-dir runs/workspace-new \
  --checkpoint runs/model.pt --training-report runs/selection.json \
  --imagery-dir runs/images --episodes 1
python -m pandion.geographic_ppo --eval --scenarios-dir runs/family \
  --checkpoint runs/model.pt --training-report runs/selection.json \
  --split development --output runs/evaluation-new.json
```

The workspace executes development episodes only. Validation and reserved
entries remain static maps. Baselines use the same fleet override as the model;
replay identity retains the actual executed scenario's hash. Each aircraft has
its own track, tank, callsign, decision history and water releases in both the
3D and compatibility maps. The fleet selector chooses the chase-camera target.
New replay audit fields record initial capacity/water per aircraft and the
aircraft index and litres of every executed drop. Display interpolation remains
an inspection aid rather than a measured continuous trajectory.

New smoke/train/selection reports written with a checkpoint contain its hash,
inference settings, source-file fingerprint, package versions and dataset
identity. `<checkpoint>.artifact.json` also binds the exact report bytes. Raw
tensor checkpoint format is preserved; loading uses `weights_only=True`.
Evaluation outputs bind the checkpoint without replacing its training sidecar.
Intermediate `.partial` and `.best` files are not automatically bound to the
final report unless their bytes match a registered checkpoint hash.

Legacy weights remain usable with a compatible, dataset-bound report. Their
association is explicitly labelled caller-supplied. Missing raw slot features
can be reconstructed from the parametric head shape; missing `skip_forced`
cannot be inferred from weights. It defaults to false with an explicit note;
use workspace `--skip-forced` or `--no-skip-forced` when its historical setting
is known. Bound reports reject conflicting overrides. Legacy imagery remains
checked by the existing array/grid loader; new reports additionally bind the
imagery manifest content. A relocated directory must retain those contents.

## Artifact catalog

```sh
python -m pandion.experiment_catalog runs/experiment-a runs/experiment-b \
  --output runs/catalog.json --html runs/catalog.html
```

The searchable offline catalog indexes reports, checkpoint bytes, family
manifests, replays and workspaces. It records configuration, results, source
version, hashes, associations, missing checkpoints and read errors. It does not
load models or run episodes. Scenario directories (`development`, `validation`,
`reserved`) and raw inputs are omitted from recursive scans; family membership
comes from manifests. Existing legacy weights are listed as unbound rather than
paired by filename. Each newly built workspace also emits `experiments.json`
and `experiments.html`, with hashes of its replay artifacts in `manifest.json`.

## Comparable incident-group reports

`--holdout-activation EMSR602_AOI02` now excludes **all AOIs of EMSR602** from
training and scores that group after selection on validation. This changes the
meaning of future folds. Historical AOI-specific folds retain their original
meaning and cannot be pooled as separate independent incidents from one group.

```sh
python -m pandion.fold_report runs/folds --output runs/fold-summary.json
python -m pandion.fold_report runs/folds --strict --output runs/fold-summary-strict.json
```

The report rejects changed configurations/source/runtime versions, different
dataset universes or selection sets, repeated group/seed folds, mismatched
scenario hashes, inconsistent baselines, invalid areas and shared incident,
content, source-product or parent lineage across partitions. Complete metadata
allows dataset/imagery relocation without relying on directory names. Strict
mode requires complete provenance; default mode retains explicitly limited
legacy summaries. This is a metadata audit, not independent reconstruction of
every historical experiment.

Scenario bootstrap intervals are conditional on the observed groups and the
mean over available training seeds. A separate pooled interval resamples
incident groups and their scenarios. Per-seed means and sample standard
deviations are descriptive; one available seed yields no estimate of seed
variation. Neither interval closes the physical-validation gate.

## Verification recorded on 2026-10-06

- Python suite: 335 tests, OK, four optional external-data skips (Python 3.12,
  existing local PyTorch/Pyretechnics environment).
- Browser regression: WebGL and compatibility map, two independent aircraft
  tracks/tanks, exact water totals, chase selection, comparison, mobile layout
  and zero runtime errors passed with Chromium 151.
- A saved D-70 recurrent image/thermal parametric two-aircraft checkpoint loaded
  through the workspace and the report-driven evaluation CLI. Both scored the
  same first development scenario at 398 ha, with identical fleet/forced-step
  configuration. This is execution parity, not new policy-quality evidence.
- Catalog inspection of the existing D-70/D-71 fold directories indexed 36
  artifacts without loading models or altering their files. Legacy associations
  remain unbound. A development-only preview and catalog are generated under
  ignored `runs/research-workspace-review/`; no training campaign or reserved
  evaluation was run.


## Fold workspace correction (2026-10-06)

The earlier preview used D-70 `EMSR583.pt`, not the six D-71 checkpoints.
`workspace --folds-dir` loads each completed selection report's same-stem `.pt`
and selects that report's held-out development cases. The training family and
held-out scenario hashes are checked after the fleet override; validation and
reserved scenarios are not run. It rejects missing weights, unmatched held-out
cases and incompatible family content. `--episodes` applies per fold. Checkpoint
name, fold and weights hash appear in the viewer and each PPO replay's metadata.
Archived reports remain marked legacy; their original association is not
retroactively certified. For D-71 use `--skip-forced`, as recorded in `run.sh`.

The D-71 registered catalog entries are carried forward from the training
commit `79d5a7b` (unpublished working commit; `da057c3` has the same catalogue) so its twenty development areas
pass source/partition checks.
This adds no new acquisitions and preserves the original package hashes.
Refill arrivals and service intervals are separate track events. Missions still
in progress at the horizon retain their actual position and water; completion
is not moved earlier to the episode boundary. The policy still chooses HOLD
when its trained action distribution prefers waiting.


Verification after correction: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m
unittest discover -s tests -v` passed 339 tests with four optional-data skips.
`PANDION_PYTHON=... node tests/viewer_browser.cjs` passed with Chromium 151;
the generated D-71 viewer also showed the correct checkpoint, two aircraft and
no page errors. All six first-case PPO areas matched the archived fold rows
(201, 74, 61, 68, 27, 215 ha); all 24 replay tracks ended within the horizon.
The first D-71 EMSR578 case executes 17 drops, compared with four in the older
D-70 preview. These are matched-input checks, not a claim of winning every case.

After integrating the current training-batching main, the complete suite passed
353 tests with four optional-data skips. The six first-case D-71 areas were
checked again and still matched; all 24 tracks remained within the horizon.
