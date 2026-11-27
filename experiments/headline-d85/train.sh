#!/bin/sh
# Train one seed of the headline policy (D-85: PPO with branched advantages).
# Monte Carlo/GAE counterpart (D-84): run with VINE=0.
#
# usage: experiments/headline-d85/train.sh SEED
# environment (defaults in brackets):
#   PYTHON   [.venv/bin/python]
#   RUNS     [runs]           holds the scenario family and imagery listed in SHA256SUMS
#   OUT      [runs/headline-d85-train]
#   DEVICE   [auto]           the recorded runs used mps
#   WORKERS  [9]              rollout workers (seed 42 used 9, seeds 43 and 44 used 5)
#   VINE     [1]              1 = branched advantages (D-85), 0 = Monte Carlo/GAE (D-84)
#
# The recorded checkpoints were trained from commit 18362dc plus the D-85 branching code
# (archived with the runs). Training on a GPU is not bit-for-bit deterministic, so a new run
# reproduces the recipe, not the exact weights; the published checkpoints are what evaluate.sh scores.
set -eu
SEED=${1:?usage: train.sh SEED}
PYTHON=${PYTHON:-.venv/bin/python}
RUNS=${RUNS:-runs}
OUT=${OUT:-runs/headline-d85-train}
DEVICE=${DEVICE:-auto}
WORKERS=${WORKERS:-9}
VINE=${VINE:-1}
mkdir -p "$OUT"
if [ "$VINE" = 1 ]; then VINE_ARGS="--vine-states 4 --vine-actions 4"; else VINE_ARGS=""; fi
# shellcheck disable=SC2086
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -u -m pandion.geographic_ppo --select --candidates 1 \
  --scenarios-dir "$RUNS/geographic-multi-ignition-family-d73-20261006" \
  --imagery-dir "$RUNS/geographic-imagery-s2-d73-px4-20261006" \
  --device "$DEVICE" --ppo-epochs 4 --minibatches 4 --gae-lambda 0.95 --reward-scaling --target-kl 0.02 \
  --entropy-coef 0.01 --entropy-coef-final 0.001 --input-mode image --thermal --ros \
  --action-mode parametric --fleet-size 2 --skip-forced --drop-bonus 0 --augment --slot-features \
  --drop-distances 0 1 2 --gamma 1.0 --reward-reference heuristic --action-features \
  --holdout-activation EMSR837_AOI06 --selection-activations EMSR719 EMSR892 EMSR632 EMSR457 \
  --updates 250 --episodes-per-update 32 --checkpoint-every 25 --eval-every 25 \
  --rollout-workers "$WORKERS" $VINE_ARGS --seed "$SEED" \
  --checkpoint "$OUT/seed$SEED.pt" --output "$OUT/seed$SEED.json" 2>&1 | tee "$OUT/seed$SEED.log"
