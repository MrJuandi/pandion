# Geographic PPO technical verification

**Date:** 2026-09-30. The geographic training interface has passed a development-only CPU smoke and weights-file verification. This closes the technical execution check, while the physical-validation and geographic policy-performance gates remain open.

## Runtime and inputs

- Python 3.12.8, PyTorch 2.14.0, NumPy 1.26.4, Pyretechnics 2026.8.10; macOS ARM64, CPU, one PyTorch thread.
- Geographic PPO source SHA-256: `bce1c0542a5f9d1fea1d71789b26434a5a33608f7e800854a02296300dd87953`.
- Training seed 42; recurrent model; one update and one episode per smoke invocation.
- Verified real development training pool: `geographic-dev-emsr578-00` and `geographic-dev-emsr578-01`, with immutable scenario seeds 100000 and 100001. Their hashes are `ed1c1b200a9eb3bb5cddc0b205ba6b2c9ad3cd52279bc84c63ab424a92024cb1` and `f850c4d8d50c97e2e7b93029911e2d28598937c7f29a1150f8431b4cbfdddb57`.
- Both scenarios use a 40×40 grid, 100 m cells, EPSG:25831, and a 3,600 s horizon. Geographic inputs retain the documented inferred fuel crosswalk, synthetic environment/survey settings, and flat terrain.
- Deterministic smoke evaluation and checkpoint reload use development-00. No real validation or reserved episodes, baseline comparison, or model selection were run.

## Commands and checks

Prepare the existing four-product real family as documented in [USAGE.md](USAGE.md). Generated families, reports, weights, and logs stay under ignored `runs/` and `data/` paths.

```sh
.venv/bin/python -m pip install -r requirements-standard.txt -r requirements-rl.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.geographic_ppo --smoke --seed 42 --scenarios-dir runs/geographic-family-real-v2 --checkpoint runs/geographic-ppo-verification/smoke.pt --output runs/geographic-ppo-verification/smoke.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.geographic_ppo --smoke --seed 42 --scenarios-dir runs/geographic-family-real-v2 --checkpoint runs/geographic-ppo-verification/repeat.pt --output runs/geographic-ppo-verification/repeat.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pandion.geographic_ppo --smoke --seed 42 --scenarios-dir runs/geographic-family-real-v2 --resume runs/geographic-ppo-verification/smoke.pt --checkpoint runs/geographic-ppo-verification/resumed.pt --output runs/geographic-ppo-verification/resumed.json
```

The full local suite passed **172 tests without failures or skips**, including the previously optional PyTorch paths and locally cached external-source checks. The new checkpoint/seed tests use fixed seeds and small generated fixtures. Selection-seed wiring uses mocked evaluations; it does not open real validation outcomes. CI adds a `unit-rl` job that requires CPU PyTorch and runs the same suite; external-source checks still skip in a checkout without those audited files.

| Invocation | PPO loss | Rejected orders in deterministic evaluation |
|---|---:|---:|
| Fresh smoke | 5505.31201171875 | 0 |
| Repeated fresh smoke, same seed | 5505.31201171875 | 0 |
| Smoke restarted from saved weights | 5495.3994140625 | 0 |

All 20 saved model tensors were finite. The two fresh runs produced equal reports and exactly equal tensors. A zero-update load through `train_geographic_ppo` preserved every tensor and reproduced the original deterministic rollout metrics. Restarting from those weights performed another finite update and changed the saved weights. A missing or directory-valued resume path raises `FileNotFoundError`; it cannot silently start a fresh model.

The following executable check verifies tensor identity and the deterministic reload using the artifacts above:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
import json
import math
from pathlib import Path
import torch
from pandion.geographic_ppo import load_geographic_split, train_geographic_ppo, rollout_geographic

root = Path("runs/geographic-ppo-verification")
reports = {name: json.loads((root / f"{name}.json").read_text())
           for name in ("smoke", "repeat", "resumed")}
states = {name: torch.load(root / f"{name}.pt", weights_only=True, map_location="cpu")
          for name in reports}
assert reports["smoke"] == reports["repeat"]
assert states["smoke"].keys() == states["repeat"].keys()
assert all(torch.equal(value, states["repeat"][key])
           for key, value in states["smoke"].items())
assert all(torch.isfinite(value).all() for state in states.values() for value in state.values())
assert any(not torch.equal(value, states["resumed"][key])
           for key, value in states["smoke"].items())
for report in reports.values():
    assert math.isfinite(report["loss"])
    assert report["seed"] == 42
    assert report["evaluation_metrics"]["rejected_orders"] == 0
    assert set(report["dataset"]["splits"]) == {"development"}
scenarios = load_geographic_split("development", Path("runs/geographic-family-real-v2"))
model, losses = train_geographic_ppo(scenarios[:2], updates=0, recurrent=True,
                                    resume_checkpoint=root / "smoke.pt")
assert losses == []
assert all(torch.equal(value, model.state_dict()[key])
           for key, value in states["smoke"].items())
_, metrics = rollout_geographic(model, scenarios[0], deterministic=True)
assert metrics == reports["smoke"]["evaluation_metrics"]
print("Checkpoint weights, repeatability, and deterministic reload: OK")
PY
```

## Interpretation and limits

`--seed` seeds Python scenario selection and PyTorch initialization/action sampling, and is stored separately from scenario seeds in smoke/train/selection reports. Default candidate seeds remain 42, 43, and so on. Exact repeatability was checked within the same CPU runtime; it is not promised across devices, architectures, or dependency versions. The dependency range `torch>=2.6` (the first release where `torch.load` defaults to `weights_only=True`; every loader in the package also passes it explicitly) is not a claim that every allowed version was tested.

Checkpoints retain the existing raw `state_dict` format. `--resume` restores model weights into a fresh optimizer; optimizer state, random-generator state, model configuration, and dataset identity are not restored from that file. Preserve the matching report and input manifest. It is a restart from weights, not exact continuation of an interrupted optimizer trajectory.

Finite loss and weights establish execution and serialization checks only. The magnitude or change of the loss does not establish stable optimization, policy improvement, generalization, or physical validity. The reward/physical-model audit and a frozen geographic evaluation protocol remain separate work, as recorded in [TRAINING_READINESS.md](TRAINING_READINESS.md).

The flat-terrain smoke evidence above remains specific to its original hashes. The 2026-10-01 environment family changes heights/forcing and requires fresh weights; see [ENVIRONMENT_INTEGRATION.md](ENVIRONMENT_INTEGRATION.md). The broader catalog identity audit also corrected EMSR590/592/600 metadata, so old full-family outputs must be regenerated.
