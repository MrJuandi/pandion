#!/bin/sh
# Score the published headline checkpoints and rebuild the README table.
#
# usage: experiments/headline-d85/evaluate.sh
# environment (defaults in brackets):
#   PYTHON   [.venv/bin/python]
#   RUNS     [runs]           holds every file listed in SHA256SUMS, at those relative paths
#   OUT      [runs/headline-d85-eval]
#   WORKERS  [6]
#
# Checks the inputs against SHA256SUMS, evaluates the three D-85 seeds (branched advantages) and the
# three D-84 seeds (Monte Carlo/GAE advantages) on the 288 held-out EMSR837 scenarios and a seeded
# sample of 120 development scenarios, then writes the three-seed summary. Compare it with
# experiments/headline-d85/expected-summary.json.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-.venv/bin/python}
RUNS=${RUNS:-runs}
OUT=${OUT:-runs/headline-d85-eval}
WORKERS=${WORKERS:-6}
(cd "$RUNS" && shasum -a 256 -c --quiet "$HERE/SHA256SUMS")
mkdir -p "$OUT"
D85="$RUNS/geographic-vine-d85-20261117"
D84="$RUNS/geographic-selection-incidents-d84-20261117"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m pandion.headline_eval evaluate \
  --report "$D85/seed42.json" \
  --scenarios-dir "$RUNS/geographic-multi-ignition-family-d73-20261006" \
  --imagery-dir "$RUNS/geographic-imagery-s2-d73-px4-20261006" \
  --checkpoint d85-seed42="$D85/seed42.pt.best" --checkpoint d85-seed43="$D85/seed43.pt.best" \
  --checkpoint d85-seed44="$D85/seed44.pt.best" --checkpoint d84-seed42="$D84/seed42.pt.best" \
  --checkpoint d84-seed43="$D84/seed43.pt.best" --checkpoint d84-seed44="$D84/seed44.pt.best" \
  --workers "$WORKERS" --output "$OUT/evaluation.json"
"$PYTHON" -m pandion.headline_eval summarize --evaluation "$OUT/evaluation.json" \
  --policy branched=d85-seed42,d85-seed43,d85-seed44 --policy monte_carlo=d84-seed42,d84-seed43,d84-seed44 \
  --pair branched monte_carlo --output "$OUT/summary.json"
