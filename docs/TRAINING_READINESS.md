# Gate before the first policy-training run

**Status:** not ready, 2026-09-28. Passing unit and synthetic integration tests is not enough to start a result-bearing controller experiment.

The first training run may begin only after the following evidence is saved with versioned inputs and reproducible commands. An exploratory algorithm smoke test, if ever used, must be labeled separately and must not enter policy comparisons.

| Gate | Required evidence | Current state |
|---|---|---|
| Scenario and provenance | Versioned map/time contract, fuel, terrain, moisture, weather, ignition, resource data, source/version/date/license and split assignment | Synthetic examples only. EMSR578 AOI01 polygons audited, but matching inputs absent. |
| Spread equation and numerical behavior | Source-matched local equation checks; varied fuel/wind/slope/grid comparisons; selected resolution justified for the endpoint | FM1 one-size equations and one published figure check done. Seven homogeneous sensitivity cases show 25 m versus 5 m area differences from −3.94% to +19.05%. Heterogeneous and alternative-scheme checks absent. |
| Independent physical validation | No-intervention spread compared with observed perimeters at matching times, with uncertainty and incidents kept separate from calibration | Annaburroo fixed-FM1 plot-rate diagnostic has 0.5264 m/s MAE across 120 plots; this does not validate Spanish spread. EMSR578 cannot validate FM1 without fuel/weather reconstruction and accounting for suppression. |
| Aircraft and water action | Sourced travel/pickup/drop constraints, deposition footprint, water-effect and persistence model, volume/time invariants, uncertainty ranges | Synthetic movement and full-volume accounting integrated with a permanent 35 m grid barrier. No sourced water response, persistence, or uncertainty range. |
| Controller information | Versioned observation schedule, latency, missingness/error, action candidates generated from observation, leakage tests | Synthetic capture/delivery/coverage and observation-only candidate/action tests done. Real sensor error absent. |
| Evaluation protocol | Primary endpoint/horizon, fair no-intervention/random/heuristic baselines, geographic train/validation/test split, seed and checkpoint protocol fixed before test | Three synthetic baselines implemented with per-run records. Geographic split, calibration, and frozen result-bearing protocol absent. |
| Training interface | Headless step/reset contract, resource limits, action mask, reward with units and exploit checks, replay and checkpoint metadata | Not implemented. |

The next evidence step is a pilot with **matched time-stamped fuel, terrain, weather, and fire perimeters**, plus at least one separate incident reserved for testing. Candidate EMSR578 AOI01 is described in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md). If credible source matching or incident independence cannot be achieved, document that limitation and keep training as exploratory rather than claiming a validated environment.

The independent grass-rate diagnostic and its limits are in [VALIDATION_ANNABURROO.md](VALIDATION_ANNABURROO.md). Its error is a reason to retain the physical-validation gate rather than to tune FM1 on this same collection and call it validated.

No pull request is to be described as “training ready” until every row has linked evidence. Repository tests and documentation checks must pass, and the pull request must follow `.github/pull_request_template.md`. Scientific readiness is distinct from code review readiness.
