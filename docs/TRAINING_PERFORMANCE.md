# Training performance investigation

Date: 2026-10-05. Initial source inspected: `8623ec1`. Scope: geographic PPO, with the older synthetic PPO path inspected for comparison. The initial investigation is followed here by the implemented D-71 acceleration and its measurements. Neither phase establishes long-run policy quality.

## Integration with current main

This change is integrated with `143a87e`, which already includes D-64–D-68: a CPU model-copy rollout pool, optional forced-step skipping, no image checkpointing by default, augmentation, slot raw features, best-checkpoint selection and corrected front observations. It was then rebased onto `99160c1`, which adds D-69 (causal `burning` state, stepped episodes scored at the horizon) and D-70 (screened CL-415 refill points); both change environment behaviour, not the collectors, and the batched collector uses the same `GeographicTrainingEnv`. D-71 adds central-device batched inference as an additional mode. `--rollout-envs 1 --rollout-workers N` retains the existing collector; `--rollout-envs N` for N > 1 uses the new collector with CPU episode owners. The serial defaults are now `--rollout-envs 1 --rollout-workers 1`; zero workers is also accepted for local simulation. Batched collection and deterministic periodic selection support the existing options without changing their configuration.

**The timing tables below predate this integration.** They describe the implementation developed from `8623ec1`, with checkpoint chunks of 8 and the earlier observation/action semantics. The 35.3% result is not a comparison against current main, and D-72's prototypes do not override the upstream checkpoint default. Preserve their source/configuration hashes when using that evidence. Re-running the disposable historical scripts on the updated tree creates a new experiment rather than reproducing the old timings. The downsample metadata issue reported below likewise describes the earlier local derivative; upstream D-64 now provides its own downsampling path.

Post-integration verification ran `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v`: **320 tests in 40.313 s, OK, four skipped** for absent external/local data. Coverage includes augmentation, slot features, forced-step skipping, all-forced episodes and reuse of the collector for periodic best-checkpoint selection. The two-update MPS CLI below was exercised again with eight environments/four workers, original 320×320 RGB/thermal inputs and checkpoint chunks of 8. Its update logs were 11.7 s and 9.6 s; final loss was 0.149616, all 44 saved tensors were finite and development evaluation rejected zero orders. These are integration checks, not a controlled speed or policy-quality comparison against `143a87e`. After the rebase onto `99160c1` and the mixed-shape grouping, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests` ran 331 tests, OK, four skipped; the MPS CLI was not re-run. Artifacts are ignored under `runs/pr-integration-20261005/cli/`.

## Findings and recommendation

The recent image/thermal, recurrent, parametric, two-aircraft runs already use Apple MPS. Their recorded update durations explain the reported hours of training. Merely enabling the GPU is not a new acceleration for those runs.

The initial implementation recommended **eight environments with four persistent CPU workers**, while keeping its existing 320×320 image/thermal input, recurrent model, two-aircraft actions and PPO budget. Five development updates took **54.29 s versus 83.92 s before those changes**, a 35.3% reduction (1.55× throughput) on this M3 Pro. This is a short historical measurement, not a forecast for a full campaign or a comparison against newer main.

The original investigation also measured approximately **1.8× throughput** from reducing image sampling from 8 to 4 pixels per simulation cell. The physical grid stays 40×40 at 100 m; the image changes from 320×320 to 160×160. That separate alternative requires fresh compatible weights and a policy-quality comparison. This change does not alter the chosen imagery; upstream D-64 exposes downsampling separately.

Success should mean **less wall-clock time to a predeclared validation quality**, not only fewer seconds per update or lower PPO loss.

## Implemented acceleration (D-71)

`--rollout-envs` batches policy decisions from independent episodes on the parent accelerator. `--rollout-workers` optionally assigns simulation and input construction to persistent spawned CPU processes, with one PyTorch CPU thread per worker. Typed arrays and scalar outcomes travel through pipes; full observations and replays stay with their episode owner. There is one GPU model, with frozen weights throughout collection, independent recurrent states, and complete episodes returned in scenario order. The episode budget is unchanged. Active episodes whose input or mask shapes differ (for example a tactical window clipped by a small incident grid) are evaluated in separate forward passes instead of failing. Whether a step is forced is decided by the episode owner, so the backend's `skip_forced` setting is the only one that applies.

Input construction now keeps RGB/thermal bytes instead of normalizing and quantizing them again, and reuses the same state masks for RGB and thermal composition. Values and targets cross the device boundary in bulk. PPO inputs, masks and fixed targets are packed once per update and reused across epochs. The existing loss weighting, optimizer steps and KL stopping statistic are preserved; notably, the pre-existing KL statistic averages episode means equally. Packing retains an additional rollout-sized tensor buffer during optimization, so peak memory still needs measurement on larger budgets.

Serial collection remains the default (`--rollout-envs 1 --rollout-workers 1`) and receives the preprocessing/transfer/packing improvements automatically. Logs now separate rollout time from reward-target/PPO update time, and reports record the requested collection configuration. In central-device batched mode, worker counts are capped by active environments and episodes per update.

### End-to-end comparison

Each variant trained a fresh model for five consecutive updates, seed 42, eight complete episodes/update, using the same manifest-verified development family and original 320×320 imagery as the initial investigation. Settings were recurrent parametric policy, fleet size 2, thermal enabled, four PPO epochs, four minibatches, GAE 0.95, reward scaling, target KL 0.02, and fixed entropy coefficient 0.05. The baseline loads saved source snapshots of `geographic_ppo.py` and `policy_inputs.py` from `8623ec1`, invoking image composition through its unchanged default interface. A copy of the original `imagery_composite.py` is also retained for reference; that file's only implementation change is optional mask reuse. Other model/simulator code is unchanged. All variants ran sequentially on MPS with the runtime described below.

Total time includes model construction and worker startup/teardown, and excludes checkpoint writing. Warm means average the timed rollout plus PPO phases of updates 2–5; accelerator synchronization brackets those phases.

| Variant | Environments/workers | Five-update total (s) | Warm mean/update (s) | Decisions | Time reduction vs original |
|---|---:|---:|---:|---:|---:|
| Original source | 1 / 0 | 83.92 | 16.40 | 1,994 | — |
| Optimized serial | 1 / 0 | 74.76 | 14.58 | 1,994 | 10.9% |
| Batched, local simulation | 8 / 0 | 73.64 | 13.30 | 1,965 | 12.2% |
| Batched, two CPU workers | 8 / 2 | 65.22 | 12.10 | 1,965 | 22.3% |
| Batched, four CPU workers | 8 / 4 | 54.29 | 10.20 | 1,965 | **35.3%** |

The original and optimized serial variants produced identical losses and **bit-identical final model tensors**. All three batched variants also produced identical losses and bit-identical final tensors to each other. This supports equivalence of preprocessing/packing and worker execution in this experiment. Serial versus batched sampling consumes random numbers in a different order and yields different actions/episode lengths (1,994 versus 1,965 decisions). The serial/batched time ratio therefore includes that small workload difference; local/two/four-worker comparisons use the same trajectory.

These are five-update, single-seed samples. Kernel warm-up, sequential run order, thermal state and CPU contention can affect the result; whole-machine telemetry and peak memory were not captured. The local batched variant's first update was particularly slow. No validation or reserved split was used in these probes. Preserving sensors and PPO semantics does not establish equal learned quality across seeds, and the measured ratio should not be extrapolated as a guaranteed reduction from hours.

### Using the implementation

Add `--rollout-envs 8 --rollout-workers 4` to the existing training command, keeping `--episodes-per-update 8` and the intended PPO settings. Four workers were best in this measurement; compare two workers when other CPU jobs are active. Zero workers batches inference with local simulation. Programmatic use with spawned workers must run under a `if __name__ == "__main__":` guard.

The following two-update development command was exercised with the existing local datasets (dataset paths can be changed to their actual locations). It saves a fresh checkpoint/report and evaluates one development scenario after training:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pandion.geographic_ppo \
  --train --updates 2 --episodes-per-update 8 \
  --rollout-envs 8 --rollout-workers 4 \
  --scenarios-dir runs/geographic-multi-ignition-family-20261003 \
  --input-mode image --imagery-dir runs/geographic-imagery-s2-20261002 \
  --thermal --action-mode parametric --fleet-size 2 --device mps --image-checkpoint-chunk 8 \
  --ppo-epochs 4 --minibatches 4 --gae-lambda .95 --reward-scaling \
  --target-kl .02 --entropy-coef .05 --entropy-coef-final .005 \
  --checkpoint runs/pr-integration-20261005/cli/model.pt \
  --output runs/pr-integration-20261005/cli/train.json
```

The ignored `runs/training-acceleration-20261005/` directory holds the disposable `compare.py`, original-source snapshots, per-variant JSON/checkpoints, summary and CLI artifacts. `compare.py --variant baseline|serial|batch|workers2|workers4 --updates 5 --output <fresh-path>` reproduces the A/B probe using those local development artifacts. These machine-local scripts/data are not shipped as project dependencies.

### Verification

Regression coverage checks deterministic serial/batched inputs, masks, actions, rewards, metrics and GAE targets; recurrent isolation across different episode lengths and fresh groups; stochastic reproducibility within batched mode; worker reuse, failure propagation and cleanup; CLI configuration; fixed episode budgets; one-time tensor packing; exact image-byte equivalence; and the existing PPO gradient-accumulation reference.

Before integration, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v` ran 302 tests in 53.37 s: **OK, four skipped** because external Annaburroo/CEMS source files and a local pilot artifact were absent. No failed tests were waived. The real two-update MPS CLI was also checked for a finite saved checkpoint/loss and recorded collection settings. The current integration's suite and CLI results are recorded above. `git diff --check` passed.

## Additional learner experiments (2026-10-05)

These were process-local prototypes, **not new production settings**. The experiment did not change `image_model.py` or its then-current defaults. Eight development episodes were collected once with the original 320×320 RGB/thermal inputs, recurrent parametric policy, fleet size 2 and seed 42. Their lengths are 64, 57, 56, 59, 62, 38, 38 and 39 decisions (413 total). Every learner variant reuses these exact observations, masks, sampled actions, old log probabilities, reward-scaled returns and GAE targets.

Each repetition reloads the same initial weights, creates fresh Adam, resets PyTorch shuffle seed 101 and runs four PPO epochs/four minibatches with target KL 0.02 and entropy 0.05. There are 16 optimizer steps; only the explicitly labeled two-epoch ablation has eight. Each variant runs in a separate process, with one warm-up repetition followed by three timed repetitions. Times bracket the complete PPO call with MPS synchronization and exclude collection, model creation and checkpoint writing. Variants run sequentially, with an unchanged baseline repeated at the end. All use one CPU thread and an allocator limit of 65% of MPS recommended memory (about 9.30 GB on this machine).

| Learner variant | Median warm PPO time (s) | Time change vs initial control | Sampled maximum driver allocation (GB) |
|---|---:|---:|---:|
| Current implementation, checkpoint chunks of 8 | 7.919 | — | 2.40 |
| Full-sequence `nn.LSTM`, mapped current cell weights | 7.880 | −0.5% | 2.40 |
| Checkpoint chunks of 4 | 8.114 | +2.5% | 1.33 |
| Checkpoint chunks of 16 | 8.255 | +4.3% | 3.47 |
| Selective recomputation: second image convolution only, chunks of 8 | 7.375 | **−6.9%** | 3.47 |
| No image activation checkpointing | 7.589 | −4.2% | 6.69 |
| Two episodes per backward, padded recurrent batch using `LSTMCell` | 8.028 | +1.4% | 2.40 |
| Two episodes per backward, padded recurrent batch using `nn.LSTM` | 7.851 | −0.9% | 2.40 |
| Two PPO epochs (changed optimization budget) | 4.174 | **−47.3%** | 2.40 |
| Unchanged control repeated at end | 8.364 | +5.6% | 2.40 |

The repeated baseline's 5.6% drift is a material limit: sub-percent changes are not established speedups, and the no-checkpoint result does not justify its much larger memory footprint. Three repeats on one fixed batch are not statistical independent-policy comparisons. Chunk 4 is a memory-saving option, not a measured speed improvement. The selective variant is the most promising equivalent-work candidate and receives the fresh on-policy follow-up below. Two epochs demonstrably halve optimizer steps and nearly halve learner time, but this does not establish faster learning or acceptable policy quality.

Driver allocation and active tensor allocation were sampled every 50 ms, together with process RSS, system available memory and system swap-out deltas. The table reports maxima across all four repetitions including warm-up; it is **sampled driver allocation, not an exact activation peak or process RSS**. Sampling can miss short peaks; allocator caches contribute to driver allocation. System swap-out counters are global and cannot be attributed solely to the probe. Available system memory and sample traces are retained locally. Monitoring overhead is included consistently in every timed repetition.

### Equivalence checks and profile

Before timing, each candidate evaluates and backpropagates through the same first two real episodes. Sequence batching preserves separate zero initial recurrent states and excludes padded positions from all losses/targets. The grouped updater preserves the original four optimizer groups, real-decision loss weighting, clipping and equal-episode KL statistic. The `nn.LSTM` prototype uses current `LSTMCell` weights with matching gates/biases, one layer, no dropout and no projection.

All equivalent-work candidates passed output comparisons (`rtol=2e-4`, `atol=2e-5`) and relative gradient-L2 error below `2e-4`; the largest observed relative gradient error was `1.48e-7`. Selective checkpointing produced exactly equal checked outputs/gradients and bit-identical final tensors after the fixed-batch PPO update. Other equivalent-work variants can differ in rounding and final Adam tensors (maximum observed absolute parameter difference about `3.2e-4`); near-equal gradients do not prove an identical long-run trajectory. The two-epoch run's forward/backward oracle checks the same model computation only, not equivalence of its shorter optimization schedule. Every timed model had finite weights.

A separate intrusive, synchronized phase probe took 9.113 s, versus the uninstrumented control median of 7.919 s. Its sums were: encoding 1.296 s (image forward 0.947 s), recurrent forward 0.296 s, heads/distribution 0.141 s, backward 6.757 s (including image recomputation 0.885 s), and optimizer steps 0.214 s. Nested values overlap and must not be added; residual time includes packing, losses, clipping and scalar reads. This identifies backward as the larger remaining phase and explains why removing the recurrent Python loop alone has little end-to-end leverage. These intrusive timings are not throughput forecasts.

### Fresh on-policy follow-up

The control and selective prototypes each trained fresh models for five updates on seeds 42 and 43, with eight episodes/update, eight active environments and four CPU workers. The current D-71 settings and original sensors/model remained fixed; entropy was constant at 0.05. Every variant used the same scenario-selection and policy-sampling seeds. Runs were sequential, ordered control→selective for seed 42 and selective→control for seed 43. Timers include model construction and worker startup/teardown, and exclude checkpoint writing and the subsequent deterministic evaluation on the same first eight development cases. Allocator limits and 50 ms memory sampling match the fixed-rollout tests.

| Seed | Control total (s) | Selective total (s) | Total time reduction | Control/selective warm mean (s/update) | Control/selective sampled max driver (GB) |
|---|---:|---:|---:|---:|---:|
| 42 | 54.13 | 51.34 | 5.2% | 10.20 / 9.56 | 3.40 / 3.40 |
| 43 | 54.76 | 53.98 | 1.4% | 9.12 / 9.38 | 2.34 / 4.48 |

Warm means use updates 2–5. In seed 43 the selective variant is **2.8% slower after warm-up**, despite a slightly lower total time; its warm PPO phase alone is 6.584 versus 6.468 s. Pooling the two short runs gives a 3.3% total time reduction, not a reliable long-run speed estimate. Machine drift, first-update compilation/initialization and dynamic sequence lengths remain important. The phase profile, fixed-batch repeats and fresh-trajectory tests must not be substituted for one another.

For both seeds, all final parameter tensors, reported losses, per-update collected episode metrics, and the eight deterministic development evaluations were exactly equal between control and selective variants. The compared runs processed the same 1,965 decisions for seed 42 and 1,818 for seed 43. All weights were finite and development evaluations had zero rejected orders. No held-out policy-quality claim follows from these short execution checks.

The selective prototype retains the first full-resolution convolution's activations across the episode, so its memory requirement grows with episode length. The larger sampled allocation on seed 43 was a reason not to promote a small, variable throughput result automatically. This change adopts none of these prototypes and preserves upstream D-64's current checkpoint configuration. Sustained warm comparisons and a memory budget for longer episodes would be required before adopting selective recomputation; two epochs remain a separate learning-quality experiment.

### Local reproduction

The ignored `runs/training-learner-probes-20261005/` directory contains the fixed rollout, process-local prototypes, per-variant measurements/checkpoints, phase profile, source/runtime/dataset hashes and follow-up artifacts. A private probe-only copy of `psutil 7.2.2` lives under its `dependencies/`; project/runtime requirements were not changed. The scripts use the existing local manifest-verified development datasets and do not acquire data or open validation/reserved scenarios.

The exercised commands are:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runs/training-learner-probes-20261005/probe.py --prepare
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runs/training-learner-probes-20261005/probe.py --variant baseline
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runs/training-learner-probes-20261005/sweep.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runs/training-learner-probes-20261005/followup.py
```

These are local diagnostic commands, not shipped CLI features or portable fixtures. Preserve existing outputs or use a fresh directory before rerunning. No additional regression run is claimed for these prototypes: production source was unchanged during this experimental follow-up; D-71's earlier full-suite result remains recorded above.

## Evidence from existing runs

Existing local artifacts were read; no existing model, report, or dataset was changed. The inspected reports identify MPS, image inputs, thermal enabled, recurrent memory, parametric actions, fleet size 2, 8 episodes/update, 4 PPO epochs, 4 minibatches, GAE 0.95, reward scaling, target KL 0.02, and entropy 0.05 → 0.005.

| Local run | Seed | Completed updates in log | Mean seconds/update | Sum of logged update durations |
|---|---:|---:|---:|---:|
| `geographic-multi-parametric-2ac-strict-20261004` | 42 | 200 | 37.12 | 2 h 03 m 45 s |
| same | 43 | 200 | 37.68 | 2 h 05 m 37 s |
| `geographic-multi-parametric-2ac-20261004` | 42 | 200 | 35.54 | 1 h 58 m 28 s |
| same | 43 | 200 | 35.25 | 1 h 57 m 31 s |
| same | 44 | 200 | 45.79 | 2 h 32 m 39 s |
| same | 45 | 200 | 47.59 | 2 h 38 m 39 s |

These sums exclude startup, validation, and checkpoint writing outside the timed update. Summing jobs does not establish their elapsed campaign time if they ran concurrently. Incomplete strict-run logs contain 75/76 updates for seeds 44/45; the investigation did not establish why they ended.

The older thermal, single-aircraft tactical run has about 23.1–23.3 seconds/update over 300 updates, also about two hours per seed. Different policy trajectories, fleets, scenario draws, source revisions, resource contention, and sustained machine conditions can all change timing. The old logs cannot establish which explains the difference from the short probe below.

## Initial development-only measurements

Runtime: Apple M3 Pro, 11 logical CPU cores, 18 GiB unified memory; Python 3.12.8, PyTorch 2.14.0, NumPy 1.26.4, Pyretechnics 2026.8.10. MPS available; CUDA unavailable. One PyTorch CPU thread and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. No other Pandion training process was found during the initial process check; whole-machine utilization and thermals were not recorded.

The probe loads only the manifest-verified development split of `geographic-multi-ignition-family-20261003`, applies the existing two-aircraft override, and samples eight scenarios with seed 42. Each repetition starts fresh weights and a fresh Adam optimizer. It performs rollout collection, reward scaling/GAE, and one PPO update with the recent run's initial settings. Device synchronization brackets the timed phases. All timing experiments ran sequentially.

Two repetitions per configuration are recorded; the second repetition reduces startup effects. This is a **small first-update throughput probe**, not a sustained 200-update training run, statistical speed guarantee, or learning comparison. Different input architectures yield different trajectories even with the same seed. None of the new experiments evaluates validation or reserved scenarios.

| Configuration | First total (s) | Second rollout (s) | Second PPO update (s) | Second total including preparation (s) | Decisions | Baseline/second total |
|---|---:|---:|---:|---:|---:|---:|
| Existing 8 px image/thermal, 4 epochs | 16.83 | 5.28 | 7.86 | 13.23 | 418 | 1.00× |
| 4 px image/thermal, 4 epochs | 7.78 | 3.81 | 3.38 | 7.29 | 418 | 1.82× |
| Existing 8 px image/thermal, 2 epochs | 9.99 | 5.51 | 4.03 | 9.64 | 418 | 1.37× |
| Explicit cell inputs, no RGB/thermal encoder | 5.35 | 2.96 | 1.88 | 4.94 | 412 | 2.68× |
| Existing inputs, `masked_fill` instead of scalar-tensor `where` | 13.29 | 5.33 | 7.69 | 13.12 | 418 | 1.01× |
| Existing inputs, image checkpoint chunk 32 instead of 8 | 14.02 | 5.53 | 7.88 | 13.50 | 418 | 0.98× |

Preparation accounts for approximately 0.10 s per sample. Baseline second-repetition time is approximately 40% rollout collection and 59% PPO update. Four-pixel images reduce pixel count by 4×, but not total runtime by 4×: the simulator, cell-resolution trunk, recurrent computation and other overhead remain.

The cell-input ablation changes what the policy perceives and removes the RGB/thermal perception problem. D-58 explicitly preserves image-mediated fire perception; this ablation is a diagnostic alternative, not a transparent replacement for that requirement. Its decision count also differs. Two PPO epochs reduce optimization work and may require more updates to achieve the same quality. Neither result proves faster learning.

The mask replacement returned the same rollout metrics and reported loss as baseline in these samples; no meaningful speedup was established. The larger checkpoint chunk returned matching rollout metrics and a slightly different floating-point loss. MPS driver allocation after the update rose from about 2.47 GB to 5.69 GB, without a speed benefit. Those are end-of-update driver allocations, **not peak memory measurements**. Do not remove checkpointing or enlarge chunks by default on this evidence.

Four-pixel images came from the existing local `geographic-imagery-s2-20261002-px4` derivative and passed the current loader's hash/grid checks. Its manifest declares 2×2 mean pooling and 25 m pixels, but retains a 12.5 m affine transform from the parent. Fix/regenerate this derivative's metadata before using it as an authoritative training dataset; passing the current loader is not a complete metadata audit. No derivative metadata was edited in this investigation.

### Profile interpretation

A separate `cProfile` run took 15.26 s, including profiling overhead. Selected cumulative host times were: input tensorization 2.42 s, environment discrete steps 1.81 s, image composition 0.93 s, thermal rendering 0.44 s, defensive environment copies 0.47 s, and candidate-cell computation 0.54 s. These functions overlap/nest; their times must not be summed.

Tensor `.item()` calls accumulated 5.92 s, and scalar `torch.tensor` construction accumulated 2.02 s. On an asynchronous accelerator these include waiting for previously submitted work. They do **not** prove that Python scalar extraction or masking itself consumes that much arithmetic. The negative `masked_fill` experiment illustrates why a profile hotspot needs an end-to-end comparison. PyTorch also advises reducing avoidable host/device synchronization in its [performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

### Local evidence and reproduction

The ignored `runs/performance-investigation-20261005/` directory contains `probe.py`, the JSON samples (`baseline`, `px4`, `epochs2`, `cells`, `fast-mask`, `chunk32`), and `profile.prof`/`profile.profile.txt`. These local diagnostic artifacts are deliberately not versioned. The script accepts `--root` for an existing run directory; it does not acquire data or persist trained weights. The measured source files were unchanged.

The exercised invocation shape is:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runs/performance-investigation-20261005/probe.py \
  --root /path/to/existing/runs --repeats 2 \
  --output runs/performance-investigation-20261005/baseline.json
```

Use the Python environment with the versions listed above. `/path/to/existing/runs` is a placeholder for the local artifacts, not a downloadable fixture. Add `--imagery geographic-imagery-s2-20261002-px4`, `--epochs 2`, `--mode cells`, `--fast-mask`, or `--chunk 32` for the respective probes. `--profile --repeats 1` produces the host profile. Repeating these experiments overwrites the chosen diagnostic output, so use a fresh output filename to preserve prior measurements.

## Alternatives, ordered by applicability

This catalogue describes the initial source (`8623ec1`); references to current behavior in its rows mean that inspected version. D-71 now implements batching, parallel CPU episode owners, byte-preserving RGB/thermal preparation, mask reuse within that preparation, bulk value/target transfers and one-time PPO packing as described above. Upstream D-64–D-68 also now implement CPU model-copy collection, downsampling, forced-step skipping, augmentation, slot features, periodic best-checkpoint selection and clean incident holdout. Read those entries as historical opportunities; other entries remain alternatives, not implemented claims.

### Existing configuration and experiment design

| Alternative | Applicability and expected tradeoff | Priority |
|---|---|---|
| Explicit `--device mps`/`auto` | CPU remains the CLI default. Essential for runs accidentally on CPU, but already enabled in the investigated runs. | Check first; no new gain here |
| One CPU thread per worker; compare 1/2/4 threads | Avoid oversubscription when adding processes. The probe uses one thread; more threads is not automatically better for small networks. | Low-cost measurement |
| Keep machine awake, powered and uncongested | `caffeinate -i` prevents idle system sleep; it is not proof against thermal throttling. Compare one training job versus concurrent jobs using total completed work/hour. | Operational |
| 4 pixels/cell instead of 8 | Measured ~1.8× for the short probe. Same physical grid and action semantics, changed image information and RGB reduction-kernel shape. Existing 8 px weights do not load unchanged. | Best measured candidate, quality gate |
| Two PPO epochs instead of four | Measured ~1.37× per update, with fewer gradient steps. Keep KL diagnostics and compare time-to-quality. | Controlled learning experiment |
| Fewer episodes/update or fewer updates | Less work produces shorter runs, but changes variance or total training budget. It is not an equivalent-quality acceleration. | Pilot budgets only |
| Stage experiments and prune weak configurations | Short development pilots first; promote promising settings to longer runs and multiple seeds. Do not prune on PPO loss alone or drop difficult scenarios to improve apparent speed. | High campaign leverage |
| Periodic validation and best-so-far checkpoints | At `8623ec1`, `--select` evaluated each candidate only after its entire training budget. Upstream D-66 now provides periodic selection and best-checkpoint restoration; the new collector supports that path. | Implemented upstream; plateau stopping remains a separate experiment |
| Limit expensive evaluation during pilots | `--eval-scenarios` exists for selection; a representative fixed subset can screen settings. It is not a substitute for full final validation, and choosing the subset after seeing results biases selection. | Conditional |
| Cell-input ablation or feedforward model | `--input-mode cells` and `--feedforward` exist. Cells measured faster; feedforward was not benchmarked. Both change the research/model contract; keep LSTM where memory is required. | Diagnostic, not default |

Successive halving/Hyperband can allocate trial budgets automatically; their pruning policy still needs a meaningful intermediate metric and enough time for slow-starting RL policies. See [Optuna's pruning guide](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).

### Implementation changes with an equivalence target

| Alternative | Specific Pandion opportunity and constraints | Priority |
|---|---|---|
| Batched inference over 4–8 active environments | `train_geographic_ppo` currently completes episodes sequentially; `get_action` sees batch size 1. Batch RGB/cells/vectors/masks on one accelerator, keep one recurrent state per environment, reset on episode boundaries. Initially preserve eight episodes/update. | High |
| Parallel CPU simulation/tensorization | Use persistent spawned workers, initially 2–4, with central batched MPS inference. Transport arrays via shared memory rather than full nested dictionaries. Avoid separate GPU model copies in every worker. | High after batching |
| Keep collection on-policy | Freeze policy weights until that update's complete rollout batch has been collected. Parallel environment execution does not require collecting with stale weights while the learner changes them. | Required design property |
| Store observations in typed arrays | Current observations contain thousands of dictionaries; reset/step performs defensive serialization copies, and simulator observations rebuild fuel/elevation data. Share immutable static arrays and copy dynamic state at ownership boundaries. Preserve public observation isolation. | Medium |
| Compute per-decision masks/features once | Burning masks and focus clusters are repeatedly computed for composition, thermal input, slot channels, valid actions and target resolution. Cache only within the same observation/ego state; fleet reservations affect validity. | Medium |
| Eliminate uint8 → float → uint8 image round trips | `compose_view` already returns uint8; `extract` normalizes it and `to_torch` converts back before the network normalizes again. Add a byte-preserving internal tensor path and retain the public extraction contract. | Medium, small bounded change |
| Faster thermal distance computation | The current halo computes distances from every grid cell to every burning cell. An exact Euclidean distance transform is an alternative, with explicit handling of no burning cells and byte-output comparisons. Thermal rendering is only ~0.44 s in this profile, so it cannot deliver a large total speedup alone. | Medium/low here |
| Pack episode training tensors once per update | `_stack_inputs`, masks, actions, old log probabilities, returns and advantages are rebuilt on each epoch. Cache inputs/targets, keeping network outputs fresh after optimizer steps. Account for the extra retained memory. | Medium |
| Batch training across episodes by length | Current minibatches are groups of episodes with per-episode backward accumulation, not one batched network call. Bucket/pad compatible sequences, mask padding in loss, and preserve normalization by real decisions and optimizer-step boundaries. | Medium/high |
| Fused recurrent sequence evaluation | `evaluate_actions` loops over `LSTMCell` in Python. Test `nn.LSTM` with mapped weights and full sequence semantics; compare outputs, gradients, checkpoint conversion and backend performance. | Medium |
| Defer unnecessary scalar reads/transfers | Retain values on-device until episode/batch end, transfer returns/advantages in bulk and aggregate metrics before reading. The CPU simulator still needs each action; batched actions reduce the unavoidable boundary count. | Medium |
| Lighter evaluation and training replay bookkeeping | Deterministic evaluation calls the same rollout that stores training records, log probabilities, values and GAE. Add a metrics-only path; training already disables observation snapshots but still materializes/copies a final replay. Preserve full replay as the audit/debug mode. | Medium |
| Cache static physical quantities across episodes | Current behavior/rate caches are per episode. Repeated terrain/fuel/weather combinations could reuse immutable gradients/behavior tables keyed by complete inputs. Never reuse arrivals, water effects or policy-dependent trajectories. Never expose future forcing through controller features. | Profile-dependent |
| Numba/Cython/C++ for verified CPU hotspots | Move numeric arrays and pure kernels first. Nested dictionaries, generators and Python extension calls prevent a one-decorator conversion of the current environment. | Later |

[Gymnasium AsyncVectorEnv](https://gymnasium.farama.org/api/vector/async_vector_env/) provides subprocess execution and shared observation buffers, but `GeographicTrainingEnv` is not currently a Gymnasium environment with declared spaces and the standard termination/truncation API. Either add an adapter or build a small dedicated collector. Simulated aircraft should remain inside one environment with the existing shared reward and decision ordering; parallelize independent scenarios.

For thermal input, SciPy documents an [exact Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html). Numba's [compilation contract](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html) explains why native numeric kernels are a more realistic target than the entire Python generator.

### Accelerator, memory and model changes

| Alternative | Assessment for this project |
|---|---|
| AMP FP16/BF16 | Test supported operators on the exact backend/runtime; likely most attractive on CUDA. Keep sensitive probability ratios, returns and value losses in FP32 as needed. The current `-1e9` action-mask sentinel needs dtype-aware handling for FP16. Verify finite losses/gradients and learning, not just successful execution. |
| `channels_last`, CUDA cuDNN tuning, TF32 | Benchmark the convolution-heavy image trunk. Layout conversion can erase gains; cuDNN/TF32 settings are CUDA-specific. TF32 changes numerical precision. |
| `torch.compile` | Try a stable-shape encoder/loss region first. Whole-episode variable lengths, Python recurrence and mutable `_slot_masks` complicate capture and recompilation. MPS has a Metal code generator in this installed runtime; do not repeat old claims that compilation is categorically unavailable. Actual model coverage and amortized speed are unmeasured. |
| CUDA graphs/static buffers | Potentially useful after batching and shape stabilization, especially for many small launches. Requires a CUDA run and memory-address/shape discipline; not a direct MPS switch. |
| MPS fast math / preferred Metal kernels | Experimental backend options, not established Pandion wins. Fast math changes numerical behavior; prefer-Metal mainly targets relevant matrix operations. Measure before adopting. |
| Tune activation checkpointing | A genuine compute/memory tradeoff. Chunk 32 did not help in this probe and retained more driver memory. Test selective checkpointing and memory budgets; do not disable it blindly. |
| Smaller CNN / earlier downsampling / fewer residual blocks | Can reduce image convolution cost substantially; changes capacity and image detail. Expose/version model configuration, retrain, compare quality. A strided stem at image resolution can help while preserving the cell-resolution output. |
| Truncated BPTT | Bounds recurrent activation memory and may enable larger batches, but changes the gradient horizon. Correct hidden-state carry, burn-in and return bootstrapping are needed; simply detaching every few steps is not equivalent. |
| Frozen/static encoder caching | The current image contains dynamic fire/treatment/thermal overlays, and encoder weights are trained. Caching its features across decisions or optimizer steps is invalid. A separate frozen static-image branch is possible, but is an architecture/learning change. |
| MLX or JAX migration | Possible redesign paths, not automatic accelerations. MLX targets Apple unified memory; JAX favors functional array computation. Neither removes the existing Python environment overhead without further work. Porting model, optimizer, masking, recurrence, serialization and reproducibility has significant cost. |
| Quantization/pruning/LoRA | Quantization mainly targets inference; structured pruning needs kernels and retraining to realize speed. LoRA is not an obvious fit for a small CNN trained from scratch. Defer unless later profiling changes the model regime. |

Official references: [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [compiler overview](https://docs.pytorch.org/docs/stable/torch.compiler.html), [MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html), [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint), and [MLX unified memory](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst). Backend support is version-dependent; the installed runtime was not upgraded during this investigation.

### Hardware, frameworks and different learning strategies

| Alternative | Assessment for this project |
|---|---|
| One Linux CUDA GPU | Run the same benchmark on a rented or existing GPU before purchasing hardware or launching a campaign. A 24 GB device is a reasonable benchmark candidate given the small model and image activations; verify actual peak demand. CPU core allocation also matters for future environment workers. No CUDA speedup was measured here. |
| Multiple GPUs/machines for independent seeds | Usually simpler than distributed training for one small model. One seed per device preserves separate experiments and shortens campaign elapsed time; it does not reduce work per seed. |
| DDP/multiple learners for a single run | Useful only when the learner has enough work and environments supply data fast enough. Communication, small batches and changed effective optimization can erase the gain. FSDP/ZeRO address model-state capacity, which is not the identified issue. |
| Ray RLlib | Offers separate environment-runner and learner scaling; substantial adapter/model/protocol migration. Consider if remote workers, fault tolerance and distributed sweeps become requirements. |
| Sample Factory / PufferLib / EnvPool | Potential collectors/backends for larger-scale throughput. Published environment FPS is not a speed estimate for Pandion. Preserve custom recurrent masking and event-driven semantics; EnvPool's native environment path entails C++ integration. |
| Stable-Baselines3 migration | Can improve maintainability, but not a drop-in speed fix. SB3-Contrib's documented `MaskablePPO` does not support recurrent policies; replacing the current masked LSTM policy requires custom integration. |
| Behavior cloning / DAgger then PPO | Existing observation-only heuristics can provide development demonstrations, potentially avoiding early random exploration. Check whether their chosen targets are representable by the current parametric mapper. A teacher may be suboptimal; compare eventual quality, not only initial convergence. |
| Curriculum / transfer / auxiliary perception losses | Train easier development cases first or reuse compatible weights; fire-perception auxiliary targets must remain consistent with the declared sensor input contract. Changed curricula/rewards require new learning comparisons. |
| Off-policy algorithms or learned world models | Replay reuse can improve sample efficiency, but masked recurrent discrete actions, critic stability and surrogate-model error create a major research change. SAC is not a drop-in discrete-action replacement. A2C, DQN variants, model-based learning and search each need a separate design comparison. |
| Heuristic controller without training | Useful as a fast baseline or interim controller. It changes the objective of learning a policy; use only if that meets the actual project need. |
| Coarser simulator grid, shorter horizon, fewer aircraft or less frequent decisions | Reduces work by changing the decision problem/physics. Keep as explicitly labeled curriculum/sensitivity studies, not claimed equivalent training acceleration. |

Sources: [RLlib scaling](https://docs.ray.io/en/latest/rllib/scaling-guide.html), [Sample Factory policy lag](https://www.samplefactory.dev/07-advanced-topics/policy-lag/), [PufferLib](https://puffer.ai/docs.html), [EnvPool](https://envpool.readthedocs.io/en/stable/), [MaskablePPO limitations](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html), and [DAgger](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html). Asynchronous learning can introduce stale-policy data, with extra sensitivity for recurrence; this is distinct from parallel environments under frozen weights.

For hardware sizing, NVIDIA documents [24 GB on L4](https://www.nvidia.com/es-es/data-center/l4/). Commercial rental is an option, for example [Runpod GPU pricing](https://www.runpod.io/pricing), but availability, CPU allocation, storage charges and end-to-end measured runtime determine value. No purchase, upload or cloud training was performed.

## What is already optimized, and what would be misleading

- D-54 already replaced prefix resimulation with a persistent episode generator, switched defensive observation copying to pickle, added MPS/CUDA selection and accumulated gradients one episode at a time. Do not count these as proposed new speedups.
- Rollouts already run under `torch.no_grad()`, base imagery is cached by activation, static tensor channels and ROS have caches, and stored image inputs are uint8. A generic suggestion to add these would miss the current code.
- `--minibatches 4` does not create four parallel workers; `--episodes-per-update 8` also does not imply eight simultaneous environments.
- Generic `DataLoader(num_workers=...)` advice does not parallelize this online environment loop. Faster SSDs/network will not accelerate the steady-state convolution/rollout workload identified here.
- A profiler's accelerator wait time is not independently removable Python overhead. The tested mask substitution saved no meaningful total time.
- A 4× pixel reduction, 4 workers or 4 GPUs does not imply a 4× total speedup. Using the measured 40/60 rollout/learner split, making only the rollout phase 4× faster would yield about 1.43× overall; making only the learner 4× faster gives about 1.82×. These are illustrative Amdahl calculations, not forecasts.
- Changing the imagery pixel count alters the RGB encoder's reduction-kernel shape. Loading an 8 px checkpoint into a 4 px model is not an exact continuation.

## Recovery and experiment waste

The checkpoint remains weights-only after integration: `--resume` reconstructs Adam and reward-scaling state, restarts the entropy schedule and does not restore random-generator states or update position. At the initial source (`8623ec1`), the `--select` branch also did not pass `args.resume` into `train_geographic_ppo`. These are source-level findings, not new implemented behavior.

For reliable long or interruptible jobs, add a versioned resumable checkpoint containing model/optimizer, reward scaler, RNG states, update number, schedule position, model/input configuration and dataset identity. Save atomically, distinguish candidate files and keep weights export for existing consumers. Also correct the local-variable reuse where `resume_checkpoint` replaces `checkpoint_path` inside `train_geographic_ppo`; otherwise periodic partial saves can follow the resume source instead of the intended destination. This improves recovery and avoids repeating work, rather than accelerating one uninterrupted update.

## Proposed next experiments and acceptance criteria

These were the initial investigation's proposals. D-71 completes the collection and preprocessing parts of items 3–4; the broader validation, memory and sustained timing gates remain open.

1. **Establish a sustained baseline.** Record at least 20 consecutive warm updates under the actual long-run command: rollout/preprocessing/inference/backward/optimizer seconds, decisions/s, effective epochs before KL stopping, episode lengths, memory pressure, device and source/config hashes. Measure timing separately from heavy profiling. Record scenario draws and compare more than one seed.
2. **Validate 4 px images.** Correct the derivative provenance, train fresh matched 4 px and 8 px models on development data with equal wall-time and equal-sample comparisons. Preserve physical grid, fleet, actions, recurrence and sensor assumptions. Predeclare the acceptable quality difference and evaluate on a fixed validation protocol. The short speed ratio is not a substitute for this gate.
3. **Improve collection without changing the learning task.** Batch policy inference first, then compare 1/2/4 persistent workers with fixed episodes/update. Confirm each sample's policy version, recurrent reset, action mask and episode boundary. Compare simulator traces and rewards under fixed action sequences before throughput measurements.
4. **Reduce repeated preprocessing and tensor assembly.** Benchmark byte-preserving image construction, per-observation derived-feature reuse, packed rollout buffers and a metrics-only evaluation path. Check input bytes/masks, rewards and gradients where equivalence is intended.
5. **Only then broaden spending or algorithm changes.** Benchmark one CUDA machine; promote candidates using wall-clock quality and total campaign cost. Try AMP/compilation and smaller models as separate experiments. Keep held-out reserved scenarios unopened for tuning and performance probing.

For the initial investigation, no full regression suite was run because production training code and data contracts were unchanged. That phase used the exercised timing probes, matching selected rollout metrics/losses for equivalent candidates, manifest-bound development loading, source inspection and documentation link checks. D-71's separate implementation verification is described above. Neither phase establishes unchanged learned quality, peak memory, long-run speed ratios, or physical validity.
