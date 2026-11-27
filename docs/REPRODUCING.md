# Reproducing the headline result

This page reproduces the README results table: PPO with branched advantages (D-85) and with Monte Carlo/GAE advantages (D-84), three seeds each, two aircraft, scored against the downwind heuristic and the hold-aware rule on the 288 held-out EMSR837 scenarios.

## What is reproducible, and how far

| Step | Status |
|---|---|
| Scoring the published checkpoints | **Exact.** On 2026-11-27, with this repository's code, the D-85 checkpoints gave the same final burned area as the recorded evaluation in all 1,224 policy rows (3 seeds × 408 scenarios), and the references matched row by row. |
| Three-seed summary (README table) | Exact. `experiments/headline-d85/evaluate.sh` scored all six checkpoints (D-85 and D-84); every mean, interval, per-seed value and paired difference equals the recorded `three-seed.json`. |
| Training new weights | Recipe only. The checkpoints were trained from commit `18362dc` plus the D-85 branching code (later committed), on Apple MPS. The exact scoring match shows that the simulator and the inference path have not changed for this configuration since then. Training on a GPU is not bit-for-bit deterministic, however, and the training code has changed since (D-86 to D-107). A new run therefore reproduces the recipe, not the exact weights, and that has not been checked by retraining. |
| Building the scenario family and imagery from the raw sources | Documented step by step in [USAGE.md](USAGE.md) (CEMS catalogue, geographic family, environment family, multi-ignition family, Sentinel-2 imagery and its 2 × 2 downsample). Not re-run end to end for this page; the inputs are identified by hash instead. |

## Environment

Python 3.12 with the pinned extras:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[standard,rl]"
```

`pyproject.toml` pins `numpy==1.26.4`, `pyretechnics==2026.8.10` and `torch==2.14.0`, the versions used for the check above. The `requirements-*.txt` files carry the same pins for CI.

## Inputs

Put the following under one folder (default `runs/`), at these relative paths. [`experiments/headline-d85/SHA256SUMS`](../experiments/headline-d85/SHA256SUMS) lists the SHA-256 of every file, and `evaluate.sh` checks them before it runs:

- `geographic-multi-ignition-family-d73-20261006/`: the scenario family (manifest hash in `SHA256SUMS`). The loader verifies every scenario against the manifest's scenario hashes.
- `geographic-imagery-s2-d73-px4-20261006/`: pre-fire Sentinel-2 imagery at 4 px per cell.
- `geographic-vine-d85-20261117/seed{42,43,44}.pt.best`: the selected D-85 weights, and `seed42.json`, the training report that gives the inference options.
- `geographic-selection-incidents-d84-20261117/seed{42,43,44}.pt.best`: the selected D-84 weights.

These files are not in Git (`runs/` is ignored). The project owner will decide how to publish them, for example as a release asset with the data licences reviewed (see [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md)). The scenario family contains modified Copernicus EMS products and the imagery contains modified Copernicus Sentinel data.

## Commands

Score the six checkpoints and rebuild the table (CPU only):

```sh
RUNS=runs WORKERS=6 experiments/headline-d85/evaluate.sh
```

It writes `runs/headline-d85-eval/evaluation.json` (per-scenario rows, with checkpoint and report hashes) and `summary.json`. Compare `summary.json` with [`experiments/headline-d85/expected-summary.json`](../experiments/headline-d85/expected-summary.json). Held out, `branched` minus `heuristic` should be −3.09 [−4.52, −1.88] and minus `hold_aware_heuristic` −0.03 [−0.44, +0.35].

Train one seed of the recipe (2 to 3 hours per seed on an M-series GPU):

```sh
experiments/headline-d85/train.sh 42          # D-85, branched advantages
VINE=0 experiments/headline-d85/train.sh 42   # D-84, Monte Carlo/GAE advantages
```

`pandion.headline_eval evaluate --help` and `summarize --help` describe the general commands. They replace the ad hoc `eval_ckpt.py` and `three_seed.py` archived with the runs (D-110).
