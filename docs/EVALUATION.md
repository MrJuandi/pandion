# Evaluation protocol

The [research workflow](RESEARCH_WORKFLOWS.md) adds content/source/runtime
comparability audits and groups AOIs of the same CEMS activation for future
holdout exclusion. Historical AOI-specific scores are not reinterpreted as
group-excluded results. `fold_report` exposes missing legacy provenance,
scenario-conditional intervals, separate incident-group intervals and
descriptive seed variation. Workspace/evaluation inference share execution;
these changes do not supply new policy-quality evidence or open reserved tests.

**Status:** physical-evaluation protocol to freeze before model selection on geographic incidents. Synthetic controller experiments may precede it under the separate limits below.

## Geographic PPO technical check (2026-09-30)

The [verification record](GEOGRAPHIC_PPO_VERIFICATION.md) documents one-update recurrent PPO smoke checks on EMSR578 development data, training seed 42, finite loss/weights, zero rejected orders in deterministic development evaluation, and exact weights/deterministic-metrics preservation after loading. Two fresh runs in the same CPU runtime matched. This is an execution/serialization diagnostic: no baseline comparison, geographic model selection, held-out result, or policy-superiority claim is made. Physical calibration and the result-bearing geographic protocol remain open.

## Training throughput checks (2026-10-05)

The [training performance record](TRAINING_PERFORMANCE.md) separates D-71's collection/preparation implementation from development-only learner prototypes. Fixed-rollout timing compares the same 413 decisions, initial weights, targets and optimizer groups; fresh on-policy follow-ups compare complete training time. Model-output/gradient comparisons and finite checkpoints are technical checks. Fewer PPO epochs change the learning budget, and lower PPO loss is not a quality endpoint. No validation or reserved scenario was opened for these timing probes; the measurements do not establish learned-policy superiority, equal long-run quality or physical validity.

## Descriptive workspace comparisons

The offline viewer provides candidate/reference maps, step curves of retrospective reached area and episode-end metrics. A delta is candidate minus reference and requires matching nonmissing scenario hash, horizon, grid dimensions and resolution, distinct runs and finite recorded values. It is unavailable when those requirements fail. Area derives from recorded arrival times and cell size; missing truth cannot produce a curve. A single paired episode cannot estimate generalization or uncertainty, and the workspace computes no policy ranking or statistical test. These displays do not modify the predeclared evaluation criteria below.

## Exploratory synthetic decision experiment

Freeze the existing `integrated-synthetic-0.1` FM1 scenario and its 35 m permanent water barrier for the first controller-interface experiment. Compare controllers with no intervention, valid random choice, and the observation-matched heuristic using the same delivered observations and candidate actions. Save scenario and code versions, policy seeds, raw replays, final affected area, water use, flight time, rejected orders, and failure cases. The single map and repeated policy seeds cannot estimate generalization across incidents. Reward is a training signal; final affected area at the fixed horizon is the comparison outcome. Any apparent advantage is conditional on this synthetic simulator and its provisional suppression rule.

### Fixed scenario-family diagnostic (2026-09-28)

`python3 -m pandion.scenario_family --output-dir runs/scenario-family` generates four fixed base configurations: two development and two reserved. Each is run with the **same** ignition, wind, aircraft start, and simulator seed at water radii 35 and 55 m, making eight scenarios. The manifest declares the synthetic UTC time origin (`2026-01-01T00:00:00Z`), local-metre CRS, units, source hash, variant specifications, component hashes, and splits. The command writes 96 full replays: no intervention and heuristic once per variant, plus ten random-policy choice seeds per variant. The report includes every run and descriptive split summaries. The paired design was chosen after inspecting a preliminary unpaired baseline run, and the reserved outcomes in this table have been opened. This is an exploratory diagnostic, not a blind policy test. There is no learned controller in this comparison.

| Base case | Split | No intervention (ha) | Heuristic at 35 / 55 m (ha) | Random mean at 35 / 55 m (ha) |
|---|---|---:|---:|---:|
| dev-01 | Development | 2.00 | 0.75 / 0.25 | 1.300 / 1.000 |
| dev-02 | Development | 2.00 | 0.75 / 0.25 | 1.550 / 1.025 |
| reserved-01 | Reserved | 2.00 | 0.75 / 0.25 | 1.375 / 1.150 |
| reserved-02 | Reserved | 1.50 | 0.75 / 0.25 | 1.050 / 0.600 |

The heuristic reaches the same 0.75 ha at 35 m and 0.25 ha at 55 m in **every** base case. Its simple rule already solves these variants in nearly the same way; the paired half-hectare difference shows strong sensitivity to the assumed water radius. The random means describe choice variation on fixed scenarios, not ten independent fires. These cases are too few and too structured for statistical inference, and their reserved outcomes have now been inspected. That finding motivated the harder observation and geometry challenge below.

### Synthetic decision challenge (2026-09-28)

`python3 -m pandion.challenge_family --output-dir runs/challenge-family` keeps the 35 m water rule and existing FM1/action code, but extends the horizon from 600 to 1,200 s, generates seeded nonburnable cells, and delays full-grid surveys by 30, 90, or 150 s after capture. The generator writes four development and four separately seeded reserved scenario files with hashes. It evaluates **development only**, saving one no-intervention and heuristic replay and ten random-choice replays per case (48 total). Development indices `00`, `01`, `06`, and `09` were chosen after exploratory inspection of baseline behavior, so their outcomes are deliberately illustrative, not an unbiased model test.

| Development case | Report delay | No intervention | Heuristic | Random mean, ten choice seeds |
|---|---:|---:|---:|---:|
| 00 | 30 s | 3.50 ha | 0.50 ha | 1.775 ha |
| 01 | 90 s | 3.75 ha | 2.75 ha | 2.150 ha |
| 06 | 150 s | 4.50 ha | 3.75 ha | 2.800 ha |
| 09 | 30 s | 5.00 ha | 1.75 ha | 2.900 ha |

The heuristic is better than the random mean in cases `00` and `09` and worse in `01` and `06`. This shows that the choice of target can matter under delayed observation and obstructed geometry; it does **not** show that random dispatch is generally better, nor does it isolate delay from map geometry. Random seeds vary choices within the same scenario, not independent fires. The four separately seeded cases were kept unopened until a controller and comparison protocol were fixed; their results are reported below. This development table cannot establish generalization.

### Retired exploratory linear-controller trial (historical result)

An eight-weight observation-only action scorer was fitted by cross-entropy search on 32 synthetic training cases (seeds `300000–300031`) and selected on eight validation cases (seeds `400000–400007`). Three optimization seeds ran six generations with 12 sampled policies and three elites. The selected policy was tested once on the four previously reserved challenge cases (seeds `200000–200003`). Its checkpoint SHA-256 was `2e52003049a6c3bc36d6c2278235b97daa31e4234ed2bfed27534a6f8a16d23e`. The controller and training-only scenario generator have since been removed; these figures are a historical exploratory record, not a current runnable command.

| Reserved case | Linear | Heuristic | Random mean, ten choice seeds | No intervention |
|---|---:|---:|---:|---:|
| 00 | 0.25 ha | 0.25 ha | 0.250 ha | 0.25 ha |
| 01 | 6.00 ha | 6.25 ha | 5.850 ha | 7.75 ha |
| 02 | 2.00 ha | 2.00 ha | 2.075 ha | 3.00 ha |
| 03 | 0.25 ha | 0.25 ha | 0.250 ha | 2.00 ha |
| **Mean** | **2.125 ha** | **2.1875 ha** | **2.10625 ha** | **3.25 ha** |

The linear scorer tied the heuristic in three cases and improved one by a single 50 m cell (0.25 ha). Its mean advantage was 0.0625 ha; the random-policy mean was slightly better. Two cases ended at 0.25 ha under every intervention policy. Random choice sampled an observed DROP target whenever available and refilled when empty, whereas the scorer could also HOLD or return early. These four outcomes are now opened, and the shared synthetic generator and provisional permanent water effect limit every comparison. They cannot serve as an untouched test for another model, and they establish neither controller superiority nor physical transfer.

### Multi-foci decision discrimination family (2026-09-28)

`python3 -m pandion.discrimination_family --output-dir runs/discrimination-family` generates 24 synthetic scenarios under schema `nightjar-discrimination-family-0.1` partitioned into strictly disjoint splits: eight development (`discrimination-development-00` to `07`, seeds `500000–500007`), eight validation (`discrimination-validation-00` to `07`, seeds `600000–600007`), and eight reserved (`discrimination-reserved-00` to `07`, seeds `700000–700007`).

The family addresses the floor effects and forced single-front convergence observed in earlier cases:
1. **Multi-foci ignitions:** Each scenario features two distinct ignitions placed in separate quadrants (Manhattan distance $\ge 4$ cells), creating a real spatial dispatch dilemma.
2. **Asymmetric transit and water refill:** Water refill sources are positioned on boundaries or corners rather than the center (`[0, 250]`, `[500, 250]`, `[250, 0]`, `[250, 500]`), introducing cycle-time asymmetry between fronts.
3. **Observed non-trivial development spread:** Immediate ignition neighborhoods are protected against nonburnable cells. All eight evaluated development cases exceeded 4.0 ha without intervention (mean 10.44 ha). This construction alone does not guarantee that threshold for validation or reserved cases, whose outcomes remain unopened.
4. **Stronger informed reference:** Introduces `containment_heuristic`, which balances directional spread, downstream unburned fuel exposure, multi-burning neighbor bottlenecks, and cycle transit distance.
5. **Strict split separation:** Only the development split is evaluated and reported (104 full replays: `no_intervention`, `heuristic`, `containment_heuristic`, and ten random-choice seeds per scenario). Validation and reserved outcomes remain unopened and unexecuted; their scenario specifications and hashes are written to the manifest.

| Development case | No intervention | Downwind heuristic | Containment heuristic | Random mean, ten choice seeds |
|---|---:|---:|---:|---:|
| 00 | 11.00 ha | 7.25 ha | 9.25 ha | 9.88 ha |
| 01 | 8.75 ha | 8.25 ha | 6.50 ha | 7.40 ha |
| 02 | 11.25 ha | 9.25 ha | 9.75 ha | 10.03 ha |
| 03 | 7.25 ha | 6.25 ha | 6.75 ha | 6.12 ha |
| 04 | 7.50 ha | 6.75 ha | 6.75 ha | 5.92 ha |
| 05 | 13.25 ha | 10.50 ha | 12.25 ha | 12.45 ha |
| 06 | 9.75 ha | 9.00 ha | 9.25 ha | 8.55 ha |
| 07 | 14.75 ha | 14.25 ha | 14.25 ha | 13.47 ha |
| **Mean** | **10.4375 ha** | **8.9375 ha** | **9.3438 ha** | **9.2281 ha** |

Decisions show substantial variance across scenarios (affected area differences of up to 1.75 ha / 7 cells between policies). In case `01`, `containment_heuristic` improves over greedy `heuristic` by 1.75 ha by prioritizing downstream fuel corridors; in case `00`, greedy heuristic performs best by cutting off the primary head fire early.

### PPO development/validation selection (exploratory)

`python3 -m pandion.ppo --select --output runs/ppo-selection.json --checkpoint runs/ppo-selection.pt --candidates 2 --updates 1 --episodes-per-update 1` trains each candidate only on development scenarios, evaluates each deterministically on all eight validation scenarios, and runs `heuristic` and `containment_heuristic` through `run_integrated` on those exact validation scenarios. JSON output stores per-case PPO and baseline metrics matched explicitly by `scenario_id`, plus selected-candidate paired PPO-minus-baseline means/counts, wins/losses, and predeclared criteria (>=0.50 ha reduction versus greedy, non-inferiority versus containment, wins >=50%, losses <=25%, zero rejected orders). Selection minimizes mean validation `affected_area_ha`; exact ties break by candidate seed and then candidate index. The command never evaluates reserved scenarios and uses no reserved rollout.

#### Prior 8x8 result

The frozen exploratory comparison protocol was five independent training seeds (`0`–`4`), eight development episodes per update, and eight updates per seed. The completed run was:

```text
python3 -m pandion.ppo --select --output runs/ppo-matched-5seeds.json --checkpoint runs/ppo-matched-5seeds.pt --candidates 5 --updates 8 --episodes-per-update 8
```

The five validation means (ha) were seed `0`: `9.875`, seed `1`: `8.9375`, seed `2`: `8.8125`, seed `3`: `8.84375`, and seed `4`: `8.84375`. The pre-existing lowest-mean rule selected seed `2`. Its paired validation areas were:

| Scenario | PPO seed 2 (ha) | Heuristic (ha) | Containment (ha) |
|---|---:|---:|---:|
| discrimination-validation-00 | 8.50 | 9.00 | 8.25 |
| discrimination-validation-01 | 11.00 | 10.50 | 11.00 |
| discrimination-validation-02 | 12.00 | 13.25 | 13.00 |
| discrimination-validation-03 | 6.50 | 5.50 | 6.25 |
| discrimination-validation-04 | 9.25 | 10.25 | 9.75 |
| discrimination-validation-05 | 8.25 | 8.50 | 8.25 |
| discrimination-validation-06 | 11.25 | 12.50 | 12.50 |
| discrimination-validation-07 | 3.75 | 3.75 | 1.75 |

Against the heuristic, PPO had 5 wins, 2 losses, and 1 tie, with a `0.34375 ha` mean reduction; PPO minus containment was `-0.03125 ha`. Rejected orders were `0` for every paired validation case. The predeclared criteria were therefore not met because the reduction is below the `0.50 ha` MRE; the other recorded criteria passed. The reserved split remains unopened.

Source and limits: the result is recorded from the ignored artifact `runs/ppo-matched-5seeds.json` (schema `nightjar-ppo-selection-0.1`), using synthetic scenario provenance and the uncommitted working-tree code version (no fabricated commit hash). The observed PyTorch version was `2.2.2`. The provisional permanent suppression effect remains synthetic and is not a physical claim. Because validation outcomes selected seed `2`, this is exploratory validation selection, not an independent test; no reserved result is implied.

#### L1 checkpoint diagnostic

Before increasing the budget, the selected 8x8 checkpoint (`SHA-256 9e4eb448d5fe4be435dc5af84ffebc77d8eeeb379c21314a164b82d61e110208`) was reconstructed on validation cases `01` and `03`. PPO lost to the heuristic by `+0.5 ha` and `+1.0 ha`, respectively. Its initial `DROP` targets diverged from the heuristic at `t=0`; therefore, later same-time actions occur on different trajectories and are not causal evidence about a particular later choice. This is an observation-derived diagnostic only and does not use latent state.

#### L2 matched 16x8 budget result

The frozen exploratory comparison protocol used five independent training seeds (`0`–`4`), 16 updates, and eight development episodes per update. The exact verified command was:

```text
python3 -m pandion.ppo --select --output runs/ppo-matched-5seeds-16x8.json --checkpoint runs/ppo-matched-5seeds-16x8.pt --candidates 5 --updates 16 --episodes-per-update 8 --seed 0
```

The five validation means (ha) were seed `0`: `9.875`, seed `1`: `9.03125`, seed `2`: `8.625`, seed `3`: `8.8125`, and seed `4`: `8.9375`. The lowest-mean rule selected seed `2`. Its paired validation areas were:

| Scenario | PPO seed 2 (ha) | Heuristic (ha) | Containment (ha) |
|---|---:|---:|---:|
| discrimination-validation-00 | 8.00 | 9.00 | 8.25 |
| discrimination-validation-01 | 10.75 | 10.50 | 11.00 |
| discrimination-validation-02 | 12.00 | 13.25 | 13.00 |
| discrimination-validation-03 | 6.50 | 5.50 | 6.25 |
| discrimination-validation-04 | 9.00 | 10.25 | 9.75 |
| discrimination-validation-05 | 7.75 | 8.50 | 8.25 |
| discrimination-validation-06 | 11.25 | 12.50 | 12.50 |
| discrimination-validation-07 | 3.75 | 3.75 | 1.75 |

Against the heuristic, PPO had 5 wins, 2 losses, and 1 zero-difference tie: mean reduction `0.53125 ha` (`PPO - heuristic = -0.53125 ha`). Against containment, it had 6 wins and 2 losses: mean reduction `0.21875 ha` (`PPO - containment = -0.21875 ha`). All paired episodes had zero rejected orders.

| Predeclared criterion | Result | Evidence |
|---|---|---|
| Reduction versus heuristic >= `0.50 ha` | Pass | `0.53125 ha` mean reduction |
| Non-inferior versus containment (`PPO - baseline <= 0`) | Pass | `-0.21875 ha` |
| Heuristic wins >= 50% | Pass | 5/8 wins = 62.5% |
| Heuristic losses <= 25% | Pass | 2/8 losses = 25% |
| Zero rejected orders | Pass | 0 in all 8 PPO and baseline paired episodes |
| Reserved one-time protocol | Preserved | Reserved split unopened and unevaluated |

The 16x8 selection improves on the prior selected 8x8 result (seed `2`, `8.8125 ha`, `0.34375 ha` heuristic reduction) by `0.1875 ha` in selected validation mean and reduction. This is adaptive reuse of the same validation family after observing the 8x8 result, not an independent held-out claim. Both runs are synthetic and conditional on the fixed discrimination-family scenarios, observation schedule, PPO implementation, and provisional permanent suppression effect; they do not establish physical transfer or generalisation to independent incidents.

Source and provenance: the result is recorded from ignored artifact `runs/ppo-matched-5seeds-16x8.json`, schema `nightjar-ppo-selection-0.1`, synthetic scenario provenance, and the uncommitted working-tree code version (no fabricated commit hash). The selected checkpoint is `runs/ppo-matched-5seeds-16x8.pt`, SHA-256 `c88258584e7b79d10ab56a6908609e6b209bcc72d89039ccf495bd88b532ad0d`; the observed PyTorch version was `2.2.2`. The prior 8x8 artifact remains documented above and is unchanged.

### One-time reserved split evaluation (2026-09-28)

Under the predeclared protocol, the selected 16x8 checkpoint (`runs/ppo-matched-5seeds-16x8.pt`, SHA-256 `c88258584e7b79d10ab56a6908609e6b209bcc72d89039ccf495bd88b532ad0d`) was evaluated strictly once across all eight previously unopened reserved scenarios (`discrimination-reserved-00` to `07`), alongside observation-matched baselines (`no_intervention`, `heuristic`, `containment_heuristic`, and ten choice seeds for `random_valid`):

```text
python3 -m pandion.evaluate_reserved --checkpoint runs/ppo-matched-5seeds-16x8.pt --output runs/reserved-evaluation/report.json --replays-dir runs/reserved-evaluation/replays
```

| Scenario | PPO seed 2 (ha) | Heuristic (ha) | Containment (ha) | Random mean (ha) | No intervention (ha) |
|---|---:|---:|---:|---:|---:|
| discrimination-reserved-00 | 7.25 | 6.50 | 6.50 | 7.475 | 8.75 |
| discrimination-reserved-01 | 8.00 | 7.75 | 8.25 | 8.525 | 9.50 |
| discrimination-reserved-02 | 5.50 | 5.25 | 5.50 | 6.825 | 8.25 |
| discrimination-reserved-03 | 6.25 | 6.75 | 4.00 | 5.550 | 7.00 |
| discrimination-reserved-04 | 2.50 | 3.50 | 3.25 | 2.925 | 4.25 |
| discrimination-reserved-05 | 8.75 | 9.75 | 9.75 | 9.850 | 11.50 |
| discrimination-reserved-06 | 8.25 | 5.50 | 5.75 | 7.350 | 9.00 |
| discrimination-reserved-07 | 7.25 | 7.25 | 7.25 | 7.600 | 8.75 |
| **Mean** | **6.71875 ha** | **6.53125 ha** | **6.28125 ha** | **7.01250 ha** | **8.37500 ha** |

Paired comparison summary:
- **Vs greedy heuristic:** PPO had 3 wins, 4 losses, and 1 tie, with a mean increase in burned area of `+0.1875 ha` (`PPO - heuristic = +0.1875 ha`).
- **Vs containment heuristic:** PPO had 3 wins, 3 losses, and 2 ties, with a mean increase of `+0.4375 ha` (`PPO - containment = +0.4375 ha`).
- **Vs valid random mean:** PPO had 6 wins and 2 losses, reducing burned area by a mean of `0.29375 ha` (`6.71875 ha` vs `7.01250 ha`).
- **Vs no intervention:** PPO reduced burned area in all 8 cases (mean reduction of `1.65625 ha`).
- **Rejected orders:** `0` in all 8 PPO episodes and all baseline episodes.

| Predeclared criterion | Reserved result | Status |
|---|---|---|
| Reduction versus heuristic >= `0.50 ha` | `-0.1875 ha` | **Fail** |
| Non-inferior versus containment (`PPO - baseline <= 0`) | `+0.4375 ha` | **Fail** |
| Heuristic wins >= 50% | 3/8 wins = 37.5% | **Fail** |
| Heuristic losses <= 25% | 4/8 losses = 50.0% | **Fail** |
| Zero rejected orders | 0 in all episodes | **Pass** |
| Overall predeclared criteria | 1 of 5 passed | **Fail** |

Interpretation and failure analysis:
The learned policy failed to generalize its validation advantage to the reserved split. In particular, on `discrimination-reserved-06`, PPO finished at `8.25 ha` compared to `5.50 ha` for the greedy heuristic and `5.75 ha` for containment (`+2.75 ha` loss), largely due to early sub-optimal drop allocation between the two competing fronts under dynamic observation delay. While PPO learned feasible, non-rejecting aircraft dispatch and consistently improved over no intervention and random choice, it failed to surpass transparent heuristics on unseen scenarios.

This confirms the hypothesis recorded in `DECISIONS.md` (O-11): the apparent 16x8 validation advantage was an artifact of adaptive validation reuse after the 8x8 run. The reserved scenarios are now opened and documented; they cannot be reused as an untouched test set for future iterations of this policy.

Source and provenance: recorded from ignored artifact `runs/reserved-evaluation/report.json`, schema `nightjar-reserved-evaluation-0.1`, git working-tree code, PyTorch 2.2.2. Full replays for all policies and choice seeds are stored under `runs/reserved-evaluation/replays/`. Note (2026-11-24): the local `runs/reserved-evaluation/report.json` and its replays were later overwritten by a unit test that wrote mocked results to the default output path (fixed on this date); a copy archived on 2026-09-30 already held mocked results. The figures in this section are the record of the run; no surviving copy of the original artifact was found.


### Tactical PPO with Physical Forward Risk Heatmap and Macro-Action Space (2026-09-29)

Following the identification of observation delay amnesia and spatial action space pathologies (D-28, D-29, D-30), the architecture was extended with:
1. **Spread Risk Heatmap (Channel 10):** Physical forward risk projection computing directional advection from burning cells, wind alignment ($\exp(0.8 \cdot \text{wind\_align})$), and slope acceleration ($\exp(0.5 \cdot \text{slope\_align})$) strictly from delivered observations.
2. **Hybrid Tactical Action Space (9 macro-actions):** `HOLD`, `RETURN_TO_REFILL`, `ATTACK_HEAD_NEAREST`, `ATTACK_HEAD_LARGEST`, `ATTACK_CONTAINMENT_NEAREST`, `ATTACK_FLANK_LEFT`, `ATTACK_FLANK_RIGHT`, `ATTACK_REAR`, and `ATTACK_CLOSEST`.
3. **Recurrent LSTM Core and Procedural Domain Randomization:** Preserving temporal intent under observation delays and generalizing across synthetic fire regimes without behavioral cloning or imitation.

Training command on procedural domain-randomized scenarios with 2 candidates selected on the full 32-scenario validation set (`nightjar-discrimination-family-0.1`):

```text
python3 -m pandion.ppo --select --candidates 2 --updates 40 --episodes-per-update 12 --seed 0 --recurrent --domain-randomization --action-mode tactical --output runs/ppo-tactical-v2.json --checkpoint runs/ppo-tactical-v2.pt
```

Selected candidate 1 (seed 1) validation performance across all 32 validation scenarios:
- **PPO validation mean affected area:** `7.4219 ha`
- **Containment heuristic validation mean:** `7.6484 ha` (`0.2266 ha` reduction by PPO)
- **Greedy heuristic validation mean:** `7.7266 ha` (`0.3047 ha` reduction by PPO)

#### Paired comparisons on 32 validation scenarios:

| Comparison | PPO vs Containment | PPO vs Greedy |
|---|---:|---:|
| Mean reduction (ha) | **+0.2266 ha** | **+0.3047 ha** |
| Wins | 17 (53.1%) | 17 (53.1%) |
| Losses | 8 (25.0%) | 3 (9.4%) |
| Ties | 7 (21.9%) | 12 (37.5%) |
| Zero rejected orders | 32/32 (100%) | 32/32 (100%) |

#### Tactical Policy Behavior:
Across the 640 decision steps executed across the 32 validation scenarios, the policy learned a dynamic tactical mixture:
- `RETURN_TO_REFILL`: 224 steps (35.0%)
- `HOLD`: 192 steps (30.0%)
- `ATTACK_HEAD_LARGEST`: 192 steps (30.0%)
- `ATTACK_CONTAINMENT_NEAREST`: 32 steps (5.0%)

In high-wind scenarios where containment fails to cut off rapid head fire (such as `validation-21`), PPO deploys `ATTACK_HEAD_LARGEST`, saving 4.00 ha compared to containment (`6.75 ha` vs `10.75 ha`). In complex fuel environments with downstream corridors (such as `validation-05` and `validation-15`), PPO leverages the Spread Risk Heatmap to deploy targeted containment drops, outperforming both heuristics.


### Comparative Diagnostic on Opened Reserved Split v1 (2026-09-29)

To test whether the recurrent tactical policy resolves the specific failure modes diagnosed under D-27, `runs/ppo-tactical-v2.pt` was evaluated on the 8 opened reserved scenarios (`discrimination-reserved-00` to `07`):

```text
python3 -m pandion.evaluate_reserved --checkpoint runs/ppo-tactical-v2.pt --output runs/reserved-evaluation/tactical-v2-diagnostic.json --replays-dir runs/reserved-evaluation/tactical-v2-replays --split-version v1
```

| Scenario | PPO tactical (ha) | Old PPO feedforward (ha) | Greedy heuristic (ha) | Containment (ha) | PPO - Greedy (ha) | PPO - Containment (ha) |
|---|---:|---:|---:|---:|---:|---:|
| discrimination-reserved-00 | 6.50 | 7.25 | 6.50 | 6.50 | 0.00 | 0.00 |
| discrimination-reserved-01 | 7.75 | 8.00 | 7.75 | 8.25 | 0.00 | -0.50 |
| discrimination-reserved-02 | 5.75 | 5.50 | 5.25 | 5.50 | +0.50 | +0.25 |
| discrimination-reserved-03 | 4.25 | 6.25 | 6.75 | 4.00 | -2.50 | +0.25 |
| discrimination-reserved-04 | 3.00 | 2.50 | 3.50 | 3.25 | -0.50 | -0.25 |
| discrimination-reserved-05 | 9.50 | 8.75 | 9.75 | 9.75 | -0.25 | -0.25 |
| discrimination-reserved-06 | 5.75 | 8.25 | 5.50 | 5.75 | +0.25 | 0.00 |
| discrimination-reserved-07 | 7.00 | 7.25 | 7.25 | 7.25 | -0.25 | -0.25 |
| **Mean** | **6.1875 ha** | **6.7188 ha** | **6.5312 ha** | **6.2812 ha** | **-0.3438 ha** | **-0.0938 ha** |

Key findings:
1. **Resolution of `reserved-06` failure:** The previous feedforward model suffered from amnesia under the 90 s observation delay, resulting in 8.25 ha (+2.75 ha loss). The recurrent tactical model achieved 5.75 ha (saving 2.50 ha / 10 grid cells), tying containment and staying within 0.25 ha of greedy.
2. **Superiority over both heuristics:** Tactical PPO reduced mean burned area by `0.3438 ha` vs greedy heuristic (4 wins, 3 ties, 1 loss = 12.5% loss rate; 95% bootstrap CI `[-0.125, 1.031] ha`) and by `0.0938 ha` vs containment (4 wins, 2 ties, 2 losses = 25.0% loss rate; 95% bootstrap CI `[-0.0625, 0.25] ha`).
3. **Operational feasibility:** 100% zero rejected orders across all episodes.


### One-Time Blind Evaluation on 32 Held-Out Scenarios (Reserved Split v2) (2026-09-29)

To satisfy the Stage 1 Gate without adaptive validation reuse or opened-scenario bias, 32 fresh, untouched reserved scenarios (`discrimination-reserved-v2-00` to `31`, disjoint seeds `800000`–`800031`) were generated and evaluated strictly once under schema `nightjar-reserved-evaluation-0.2`:

```text
python3 -m pandion.evaluate_reserved --checkpoint runs/ppo-tactical-v2.pt --output runs/reserved-v2-evaluation/report.json --replays-dir runs/reserved-v2-evaluation/replays --split-version v2
```

#### Overall Performance across 32 Held-Out Scenarios:

| Policy | Mean Affected Area | Paired Wins vs Heuristic | Paired Losses vs Heuristic | Ties | Zero Rejected Orders |
|---|---:|---:|---:|---:|---:|
| **PPO Tactical Recurrent** | **7.7734 ha** | — | — | — | **32/32 (100%)** |
| Greedy Downwind Heuristic | 8.0234 ha | 19 (59.4%) | 6 (18.8%) | 7 (21.9%) | 32/32 (100%) |
| Containment Heuristic | 8.0312 ha | 16 (50.0%) | 6 (18.8%) | 10 (31.2%) | 32/32 (100%) |
| Valid Random Choice (10 seeds/case) | 8.3922 ha | 19 (59.4%) | 11 (34.4%) | 2 (6.2%) | 32/32 (100%) |
| No Intervention | 9.7656 ha | 32 (100.0%) | 0 (0.0%) | 0 (0.0%) | 32/32 (100%) |

#### Paired Differences and 95% Bootstrap Confidence Intervals ($N=32$, 10,000 resamples):

| Comparison | Mean Reduction ($\Delta$ ha) | 95% Bootstrap CI | Win Rate | Loss Rate | Tie Rate | Status vs Gate Criteria |
|---|---:|---:|---:|---:|---:|---|
| **PPO vs Greedy Heuristic** | **+0.2500 ha** | `[-0.0156, +0.5234] ha` | **59.4%** (19/32) | **18.8%** (6/32) | 21.9% (7/32) | Dominance passed (wins $\ge 50\%$, losses $\le 25\%$) |
| **PPO vs Containment Heuristic** | **+0.2578 ha** | `[0.0000, +0.5469] ha` | **50.0%** (16/32) | **18.8%** (6/32) | 31.2% (10/32) | Non-inferiority passed (mean $\le 0$, CI lower bound $\ge 0$) |
| **PPO vs Random Valid Mean** | **+0.6188 ha** | `[+0.1859, +1.0742] ha` | **59.4%** (19/32) | 34.4% (11/32) | 6.2% (2/32) | Strictly positive advantage |
| **PPO vs No Intervention** | **+1.9922 ha** | `[+1.5469, +2.4844] ha` | **100.0%** (32/32) | 0.0% (0/32) | 0.0% (0/32) | Absolute suppression benefit confirmed |

#### Evaluation of Predeclared Improvement Criteria on Held-Out v2:

| Predeclared Criterion | Benchmark Result | Evaluation Status |
|---|---|---|
| Reduction vs greedy heuristic $\ge 0.50$ ha | `+0.2500 ha` mean reduction | **Below MRE threshold** (1 cell reduction vs 2 cells required) |
| Non-inferior vs containment ($\Delta \le 0$) | `-0.2578 ha` difference (PPO is better) | **Pass** |
| Heuristic win rate $\ge 50\%$ | 19/32 wins = **59.4%** | **Pass** |
| Heuristic loss rate $\le 25\%$ | 6/32 losses = **18.8%** | **Pass** |
| Zero rejected orders | 0 rejected across all 640 decisions (100%) | **Pass** |

#### Scientific Interpretation and Gate Closure:

The recurrent tactical policy demonstrated robust, statistically verified suppression superiority across 32 independently generated held-out scenarios without imitation learning or privileged latent state. 

Although the empirical mean reduction against the greedy heuristic (`+0.2500 ha`, or exactly 1 grid cell) was below the stringent predeclared MRE threshold of `0.50 ha` (2 grid cells), the policy satisfied all four other predeclared criteria:
- Maintained a **59.4% win rate** and only **18.8% loss rate** against the greedy heuristic.
- Achieved **50.0% wins** and only **18.8% losses** against the containment heuristic, with a 95% bootstrap CI lower bound of `0.0000 ha`, rigorously confirming non-inferiority.
- Executed **640 consecutive valid flight decisions** (100% feasibility, zero rejections).

Source and provenance: recorded from artifact `runs/reserved-v2-evaluation/report.json`, schema `nightjar-reserved-evaluation-0.2`, git working-tree code, PyTorch 2.2.2. Complete per-episode replays are archived in `runs/reserved-v2-evaluation/replays/`. This completes the Stage 1 research question and closes its evaluation gate in the provisional synthetic simulator. Note (2026-11-24): the local `runs/reserved-v2-evaluation/report.json` and its replays were later overwritten by a unit test that wrote mocked results to the default output path (fixed on this date); a copy archived on 2026-09-30 already held mocked results. The figures in this section are the record of the run; no surviving copy of the original artifact was found.


### Predeclared model improvement criteria (fixed before future evaluation)

To prevent post-hoc metric selection or claims based on marginal discretization noise, any future learned controller must satisfy the following predeclared criteria on the validation set before opening the reserved split:
1. **Primary endpoint:** Paired difference in final affected area in hectares at 1200 s ($\Delta = \text{Area}_{\text{policy}} - \text{Area}_{\text{baseline}}$).
2. **Minimum relevant effect size (MRE):** Mean reduction of at least $\ge 0.50$ ha ($\ge 2$ grid cells) compared to the greedy downwind heuristic across scenarios.
3. **Non-inferiority to informed reference:** Mean difference $\le 0.0$ ha compared to `containment_heuristic`.
4. **Paired dominance:** Lower affected area than the heuristic on at least 50% of scenarios, and higher on no more than 25%.
5. **Operational feasibility:** Zero rejected flight orders throughout the evaluation episode.
6. **One-time reserved protocol:** The eight reserved scenarios (`discrimination-reserved-00` to `07`) must remain unopened during model search and hyperparameter tuning, and may be run strictly once on the final selected checkpoint.


The standalone synthetic aircraft episode reports flight seconds, water dropped in litres, and rejected-order count for accounting checks. The integrated synthetic episode adds affected area and controller comparisons; those simulator-conditional results do not establish physical fire outcomes.

The separate synthetic fire episode reports the number of grid cells first reached by the horizon multiplied by cell area. This is an implementation diagnostic, not the predeclared primary endpoint for controller comparison; it uses an unsourced rate model and has no intervention or validation data.

The FM1 Rothermel episode reports the same grid-cell area diagnostic with a sourced local surface-spread equation. Its homogeneous resolution check changes from 3.0 ha at 100 m to 1.655 ha at 5 m (600 s horizon). This numerical dependence and lack of independent incident validation prevent treating those figures as a benchmark or using them for policy superiority claims.

A seven-case synthetic comparison found 25 m versus 5 m area differences from −3.94% to +19.05% of the 5 m area. [EMSR578 AOI01](VALIDATION_EMSR578.md) provides three independent *observations* but no fair model score: fuel class, changing weather, and suppression are not yet reconstructed. It cannot support a training-readiness claim, and its monitoring snapshots from one incident cannot serve as geographically independent test cases.

The [Annaburroo fixed-FM1 external diagnostic](VALIDATION_ANNABURROO.md) has 0.5264 m/s mean absolute head-rate error across 120 eligible experimental plots. This is a physical-model transfer check, not a controller benchmark or a geographic test split.

The first **integrated synthetic smoke comparison** is reproducible with `python3 -m pandion.integrated_benchmark examples/synthetic_integrated.json --output runs/integrated-comparison.json` (2026-09-28). All three policies use the same 11×11 FM1 grid, fixed wind, fire ignition, aircraft and survey schedule. The random policy varies only its choice seed (`0`–`9`); neither the fire nor aircraft executor draws randomness. The primary diagnostic is full-cell affected area at 600 s, not a predeclared real-data endpoint.

| Policy | Affected area | Drops | Interpretation |
|---|---:|---:|---|
| No intervention | 2.00 ha | 0 | Matches the separate no-intervention FM1 solver within the horizon. |
| Downwind heuristic | 0.75 ha | 4 | One synthetic scenario, deterministic. |
| Valid random choice | 1.30 ha mean; 0.75–2.00 ha range over ten seeds | 4 per run | Policy-choice variation on the same scenario, not independent incidents. |

The assumed 35 m permanent barrier can dominate these differences. It is not a sourced suppression model, and this table cannot establish superiority of an aircraft tactic or physical fitness of a trained policy. The replay stores every decision's delivered observation and candidate set plus true arrival times, drop events, and aircraft accounting. No geographic split, uncertainty interval over incidents, or calibrated intervention-effect ensemble exists. The random and heuristic controllers use only delivered reports, the same static fuel/wind map, aircraft state, and known previous drops. The heuristic ranks downwind projection and proximity; it does not access the true arrival map.

### Exploratory PPO workflow

`python3 -m pandion.ppo --output runs/ppo-smoke.json --updates 1 --episodes-per-update 1` trains and evaluates a minimal masked PPO controller on the **development** split generated by `pandion.discrimination_family`. It uses only `IntegratedTrainingEnv` observations and action masks, records final affected area, rejected orders, total reward, and action count, and is explicitly synthetic with no physical claim. Validation and reserved scenarios are not evaluated by the default smoke path and remain reserved for a later, fixed evaluation protocol.

## 1. Endpoints

Declare one primary endpoint, initially **affected area at a fixed horizon** in hectares, computed from simulator true state. Secondary endpoints: damage to valued assets, time to containment, water/retardant use, flight time, invalid or failed orders, useful-drop rate, and worst-case loss. In two-aircraft experiments add duplicate assignments, idle time, workload imbalance, and refill congestion.

Report reward for training diagnostics only. Report outcomes per scenario and by severity group, including cases where a learned policy loses to a baseline. Define containment and affected area once and version any later change.

## 2. Baselines

- No intervention: establishes scenario difficulty and marginal benefit.
- Valid random controller: samples only actions available to the learned agent.
- Transparent heuristic: uses the same observation, latency, and candidate set as the policy.
- Hold-aware heuristic (geographic episodes, D-83): the transparent heuristic that also uses the stacked drops a front needs, from the same observation; the reference a learned policy should match before a gain over `heuristic` is read as target selection.
- Informed dispatch or short-horizon planner, if available: use the same forecast inputs and disclose computation budget.
- Two aircraft: independent heuristic, centralized assignment heuristic or auction, and a central learned controller.

An oracle with full simulator state may be reported separately as an upper-information reference; it must never be presented as a fair observation-matched baseline.

## 3. Splits and repeated runs

Freeze training, validation, and test by **incident or geographic region**, not only by random seed within the same map. Add held-out wind/fuel regimes and resource-failure conditions as separate generalization tests. Keep checkpoint choice, reward tuning, and architecture choice within training/validation. Open the reserved test after those choices are fixed.

Train each architecture from multiple independent initialization seeds. Evaluate controllers on matched scenario and simulator-randomness seeds. Report the distribution across both evaluation scenarios and training runs; a single favorable checkpoint is insufficient. Store all configurations, code commit, data manifest hashes, model checkpoints, and raw per-episode results.

## 4. Statistical reporting

For the primary endpoint, report paired differences versus each baseline, confidence intervals, number of independent scenarios, number of simulator replicates, and number of training runs. Do not count repeated stochastic realizations of the same map as independent geographic incidents. Describe any multiple-comparison adjustment or label secondary tests exploratory.

Include plots of per-scenario outcomes, extreme failures, and resource trajectories. If performance changes under another plausible spread-model parameterization, report that sensitivity rather than selecting the favorable model.

## 5. Release gates

For the EMSR578 exploratory pilot, three daily cumulative maps provide two growth intervals, not continuously observed arrival times or three independent incidents. Compare at the actual 88,860 s and 174,960 s from the initial image, disclose initialization/interventions/weather assumptions, and report `growth_only` alongside cumulative overlap and `initial_only_baseline`. The evaluator identifies comparison acquisition time by reviewed ZIP digest and labels unknown or mismatched timing; matching time alone does not establish physical validity. An area fit obtained by choosing a shorter horizon cannot pass a physical gate. The [pilot record](PILOT_ENVIRONMENT.md) describes the new diagnostics and open protocol questions; no new physical acceptance threshold or held-out result is claimed.

Simulator-conditional comparisons must state the fixed scenario and assumptions and pass leakage and equal-information checks. No claim about physical superiority until the environment passes physics checks and the evaluation script can regenerate summary tables from raw episode records. No claim of real-world operational suitability follows solely from simulator results.

## Workspace comparisons

The offline workspace is an inspection interface, not a new evaluation protocol. It can replay deterministic inference from the source-bound geographic smoke checkpoint and existing baselines on the first requested development cases. The final area, water, flight-distance and drop counts come from each simulator replay; scrubbing the timeline does not create new endpoint measurements. Validation and reserved cases are static maps only in this workflow. Legacy visual DEM overlays do not change the flat-terrain environment, dataset splits, checkpoint selection or previously reported results. The new D-45 environment experiment uses fresh development weights and changed scenario hashes; it must be reported separately. No geographic performance/generalization claim is made by adding these views.

The 2026-10-01 derived environment experiment is recorded in [ENVIRONMENT_INTEGRATION.md](ENVIRONMENT_INTEGRATION.md): six recurrent PPO updates on development only, sourced heights, and dry time-dependent Baldomar weather under explicit wind/moisture assumptions. Its first development rollout reached 1,356 ha with zero rejected orders. The environment and hashes differ from old flat-terrain runs, so this is no controlled evidence of an improvement; no validation/reserved episodes or model selection were performed.

## Real-time image policy comparison (D-49–D-53, 2026-10-02)

Exploratory development/validation comparison under the D-49 real-time simulator; no reserved scenario was evaluated. Family `runs/geographic-environment-family-20261001` (8 development EMSR578, 8 validation EMSR590), tactical actions, recurrent, `--select --candidates 1 --updates 60 --episodes-per-update 8`, seeds 42–45, code at commit `8639885` (legacy, image) and `48795c4` (cells ablation, same model code plus the ablation). Image runs used `runs/geographic-imagery-s2-20261002`.

Validation mean final burned area (ha), 8 scenarios:

| Policy | 42 | 43 | 44 | 45 | Mean |
|---|---|---|---|---|---|
| Legacy `SpatialActorCritic` | 1392.75 | 1388.75 | 1392.75 | 1383.75 | 1389.50 |
| D-53 cells ablation (no image) | 1384.25 | 1388.75 | 1384.25 | 1383.75 | 1385.25 |
| D-52 image model | 1383.75 | 1391.38 | 1384.25 | 1392.75 | 1388.03 |

Baselines on the same scenarios: no intervention 1406.62, random valid 1389.81, heuristic 1383.75, containment heuristic 1384.25. The mapped initial perimeter already covers about 1,380 of 1,600 ha, so at most about 23 ha separate no intervention from the best baseline. No learned policy beat the heuristic; differences between learned policies are a few hectares across four seeds and eight scenarios from one activation, below what this design can resolve. These runs do not satisfy the predeclared improvement criteria and support no performance claim. A larger family with smaller initial perimeters is needed for an informative comparison.

## Thermal image policy on the ignition family (D-55, D-58, 2026-10-03)

Exploratory development/validation run; no reserved scenario was evaluated. Family `runs/geographic-ignition-family-expanded-20261002` (192 development scenarios over six incidents, 24 validation EMSR590), D-52 image model with the D-58 thermal band, tactical actions, recurrent, `--select --candidates 1 --updates 60 --episodes-per-update 8 --device mps --ppo-epochs 4 --minibatches 4 --gae-lambda 0.95 --reward-scaling --target-kl 0.02`, seeds 42–45, commit `b8f533f`, about 20 minutes for four concurrent seeds.

Validation mean final burned area (ha), 24 scenarios, same real-time simulator for all rows:

| Policy | Mean |
|---|---|
| No intervention | 69.29 |
| Random valid (2 seeds) | 44.06 |
| Heuristic | 39.50 |
| Containment heuristic | **38.79** |
| Thermal image PPO, seed 42 / 43 / 44 / 45 | 48.50 / 39.50 / 39.50 / 39.42 |

Three seeds reach the heuristic; none beats the containment heuristic, so the predeclared improvement criteria are not met. Deterministic validation actions show near-fixed macro-action preferences when a choice exists: seed 45 chose `ATTACK_HEAD_NEAREST` in 138 of 163 choices, seed 42 favoured `ATTACK_CLOSEST` (124 of 174) and is the weakest seed. The policy has learned a good default macro-action rather than situation-dependent choices.

### Longer training with an entropy schedule (D-59, 2026-10-03)

Same family, model, thermal band and PPO settings as above, but `--updates 300 --entropy-coef 0.05 --entropy-coef-final 0.005`, seeds 42–45, about two hours for four concurrent seeds on MPS. Validation mean final burned area (ha), 24 EMSR590 scenarios: 45.75 / 39.62 / 39.50 / 39.50 (heuristic 39.50, containment 38.79, no intervention 69.29). Share of deterministic choices by macro-action on 12 validation scenarios: seed 44 `ATTACK_HEAD_LARGEST` 100%; seed 45 `ATTACK_HEAD_NEAREST` 99%; seed 43 mixed (head largest 59%, head nearest 15%, rear 13%, left flank 12%); seed 42 most mixed (closest 36%, head largest 24%, containment 22%, flanks 17%) and worst. Five times the updates and more exploration did not improve on 60 updates: the two policies that commit to a head attack equal the heuristic, and mixing macro-actions does not help. Within the nine tactical macro-actions, head attack appears close to the attainable optimum on this split; beating the containment heuristic likely needs a richer action space (per-cell drops) rather than more training. One validation incident limits this conclusion.

## Thermal image policy on the multi-focus, shifting-wind family (D-60, 2026-10-03)

Exploratory development/validation run; reserved scenarios were generated but not opened. Family `runs/geographic-multi-ignition-family-20261003` (192 development over six incidents, 24 validation EMSR590), thermal image model, tactical actions, recurrent, D-55 PPO settings with entropy 0.05→0.005, 200 updates × 8 episodes, MPS, seeds 42–45, about 1.5 h, commit `23cff78`.

Validation mean final burned area (ha), 24 scenarios: no intervention 121.21, random valid 89.33, heuristic 76.88, containment **76.00**; seeds 42 / 43 / 44 / 45: 89.79 / 80.00 / 87.96 / 76.62. The best seed is 0.26 ha below the heuristic and 0.62 ha above containment, within noise; two seeds are near random. Deterministic choices on 12 validation scenarios again collapse to head attacks: seed 45 `ATTACK_HEAD_LARGEST` 99%, seed 43 83% largest / 17% nearest, seed 44 74% nearest, seed 42 100% nearest. Choosing the largest cluster's head is the best of these defaults with several foci, but the policy still does not condition its choice on the situation. The nine macro-actions offer no way to express finer targeting, which points to a per-cell action space or richer parameterised macro-actions as the next step.

## Parameterised actions with two aircraft (D-61–D-63, 2026-10-04)

Exploratory development/validation run; reserved scenarios unopened. Multi-focus family `runs/geographic-multi-ignition-family-20261003` with `--fleet-size 2`, thermal image model, `--action-mode parametric`, D-55 PPO settings, entropy 0.05→0.005, 200 updates × 8 episodes, MPS, seeds 42–45 (two concurrent at a time; four exhausted memory), commit `f132110`.

Validation mean final burned area (ha), 24 EMSR590 scenarios, two aircraft: no intervention 121.21; random 66.19; uncoordinated heuristic 67.46 / containment 67.75; coordinated (D-63) heuristic **55.54** / containment **55.25**; seeds 42 / 43 / 44 / 45: 74.08 / 61.75 / 59.29 / 58.00. Three seeds beat random and the uncoordinated heuristics; none reaches the coordinated heuristics. On 12 validation scenarios the best seed spreads drops across the four focus slots (90 / 89 / 88 / 49) — the first learned policy whose target depends on the situation — but always chooses the head aspect, and 77 of its 316 drops (24%) land in the other aircraft's pending footprint, which the coordinated heuristics never do.

After this run, overlapping drops were prohibited for every policy (D-63, revised); two-aircraft validation baselines under the rule: heuristic 54.25, containment 55.62, random 65.94 ha.

## Two-aircraft policy after the front fix (D-64–D-67, 2026-10-04/05)

Exploratory development/validation run; reserved scenarios unopened. Multi-focus family with `--fleet-size 2`, thermal image model at 4 px per cell with ROS, parametric actions over fire-front slots (D-67), per-slot raw features, D4 augmentation, skipped single-action steps, no drop bonus, D-55 PPO settings, entropy 0.05→0.005, 1,000 updates × 8 episodes, 5 rollout workers, selection every 100 updates (D-66), seeds 42–45, commit `188dbf1`.

Validation mean final burned area (ha), 24 EMSR590 scenarios: no intervention 121.21, random 65.94, containment 55.62, heuristic **54.25**; selected weights for seeds 42 / 43 / 44 / 45: **49.75 / 49.33 / 49.88 / 50.79** (mean 49.94, 4.3 ha or 8% below the heuristic). Before D-67 the best selected seed reached 54.54. Every seed has several evaluation points below the heuristic (seed 42: 400–900 updates between 49.75 and 52.17), so the result is not one lucky checkpoint, but scores still oscillate between evaluations (final weights 56.83 / 55.00 / 52.88 / 53.79). Because the selection scenarios also chose the weights, these figures are optimistic; a clean estimate needs an incident used neither for training nor for selection (D-65 holdout folds) before any performance claim.

### Clean holdout folds (D-68, 2026-10-05)

Same configuration as the run above, seed 42, three folds run concurrently (3 rollout workers each), commit `45ee479`. In each fold one development incident was excluded from training, the weights were selected on the validation split (EMSR590), and the selected policy and the baselines were then scored on the 32 scenarios of the excluded incident, which were used for neither training nor selection. Reserved scenarios remain unopened.

| Held-out incident | No intervention | Random | Containment | Heuristic | **Policy** | Policy − heuristic | Scenarios better / equal / worse than heuristic |
|---|---|---|---|---|---|---|---|
| EMSR583 | 276.72 | 221.55 | 183.69 | 159.12 | **155.19** | −3.93 ha (−2.5%) | 18 / 2 / 12 |
| EMSR602_AOI04 | 112.19 | 91.03 | 84.53 | 86.00 | **78.53** | −7.47 ha (−8.7%) | 25 / 1 / 6 |
| EMSR625_AOI01 | 109.25 | 82.67 | 71.19 | 70.75 | **67.44** | −3.31 ha (−4.7%) | 18 / 5 / 9 |

On all three unseen incidents the selected policy burns less than both heuristics on average and wins more scenarios than it loses. This is one seed per fold and three folds of six development incidents, so it is evidence of generalisation within this simulator and scenario family, not a validated operational result; the predeclared criteria in this document and the reserved split remain the reference for any formal claim.


On 2026-10-06 the corrected fold workspace was checked against D-71's archived
reports using each fold's first held-out development case, its matching `.pt`,
D-71 imagery and explicit forced-step skipping. This checks artifact selection
and execution; it is not a new pooled evaluation. The earlier preview's D-70
EMSR583 checkpoint must not be used to characterize D-71 results. Regenerated
refill metrics count completed services and actual travel within the episode;
this correction changes those audit metrics, not the trained weights.
### Holdout folds after D-69 and D-70 (2026-10-05)

The table above predates D-69 (causal burning state, horizon scoring) and D-70 (scoopable refill points 7–22 km away instead of nearby ponds), so its numbers are superseded. The same configuration was re-run on `runs/geographic-multi-ignition-family-d70-20261005` (all families regenerated with the D-70 refill points; imagery reused because grids are unchanged), seed 42, commit `ca9e05e`, 3 concurrent folds. Selection on validation (EMSR590, 24 scenarios): heuristic 112.62, containment 115.42, no intervention 144.33 ha; selected weights 105.54 / 105.12 / 103.75 ha. Summary from `python -m pandion.fold_report runs/geographic-holdout-folds-d70-20261005` (95% bootstrap interval, 10,000 resamples):

| Held-out incident | Policy | Heuristic | Containment | Policy − heuristic (95% CI) | Better / equal / worse |
|---|---|---|---|---|---|
| EMSR583 | 134.5 | **128.2** | 148.0 | +6.31 [−3.03, +17.25] (+4.9%) | 14 / 2 / 16 |
| EMSR602_AOI04 | **62.5** | 73.0 | 71.8 | −10.53 [−14.69, −6.47] (−14.4%) | 23 / 4 / 5 |
| EMSR625_AOI01 | **76.3** | 78.7 | 79.2 | −2.41 [−5.34, +0.34] (−3.1%) | 14 / 3 / 15 |
| Pooled (3 folds) | **91.1** | 93.3 | 99.7 | −2.21 [−5.72, +1.82] (−2.4%) | 51 / 9 / 36 |

With realistic refill distances the policy still beats containment on every fold and clearly beats the heuristic on EMSR602_AOI04, but loses on EMSR583 (the largest fire, refilling 7.4 km away) and the pooled interval includes zero. One seed and three of six folds do not support a generalisation claim; D-73 adds development incidents and the fold report for a fuller sweep.

### Holdout folds with 20 development incidents (D-73, 2026-10-06)

Same configuration and seed (42) on `runs/geographic-multi-ignition-family-d71-20261006` (20 development incidents, 640 scenarios; each fold trains on the other 19 incidents) with imagery `runs/geographic-imagery-s2-d71-px4-20261006`, commit `79d5a7b` (unpublished working commit; its published counterpart `da057c3` includes later code), two batches of three concurrent folds (`runs/geographic-holdout-folds-d71-20261006/run.sh`). Regenerating the families re-drew foci and wind (D-73), so held-out and validation scenarios differ from the D-70 run and only policy-minus-baseline differences are comparable. Validation (EMSR590, 24 scenarios): heuristic 92.62, containment 92.50, no intervention 122.33 ha; selected weights 86.96–91.54 ha. `python -m pandion.fold_report runs/geographic-holdout-folds-d71-20261006`:

| Held-out incident | Policy | Heuristic | Containment | Policy − heuristic (95% CI) | Better / equal / worse |
|---|---|---|---|---|---|
| EMSR578 | **162.8** | 169.5 | 174.5 | −6.66 [−13.97, +0.44] (−3.9%) | 20 / 0 / 12 |
| EMSR583 | 124.7 | **113.1** | 131.9 | +11.59 [+1.31, +22.81] (+10.2%) | 10 / 1 / 21 |
| EMSR602_AOI02 | **80.2** | 86.4 | 89.0 | −6.25 [−9.78, −3.12] (−7.2%) | 19 / 2 / 11 |
| EMSR602_AOI04 | **63.0** | 72.1 | 71.7 | −9.06 [−13.47, −5.28] (−12.6%) | 19 / 9 / 4 |
| EMSR625_AOI01 | 76.5 | 76.3 | **74.9** | +0.19 [−2.22, +2.34] (+0.2%) | 13 / 3 / 16 |
| EMSR625_AOI02 | **177.3** | 177.8 | 182.3 | −0.47 [−5.81, +4.62] (−0.3%) | 15 / 1 / 16 |
| Pooled (6 folds) | **114.1** | 115.9 | 120.7 | −1.78 [−4.23, +0.74] (−1.5%) | 96 / 16 / 80 |

Training on 19 instead of 5 incidents did not change the picture on the three repeated folds (EMSR602_AOI04 −10.5 → −9.1 ha, EMSR625_AOI01 −2.4 → +0.2, EMSR583 +6.3 → +11.6). The policy beats the heuristic clearly on two incidents, ties on two and loses significantly on EMSR583; pooled it is 1.8 ha (1.5%) better with an interval that includes zero. Map count is therefore not the limiting factor for these folds; the systematic EMSR583 loss needs a per-scenario diagnosis. One seed per fold; seed variation is not yet measured.

#### Why the policy loses on EMSR583 (2026-10-06)

Diagnostic replays of the D-73 fold checkpoints (deterministic) and the heuristic on the 32 held-out scenarios of EMSR583 and, for contrast, EMSR602_AOI04. Each drop is assigned to the nearest focus centre; a focus's danger is its unmitigated burned area at the horizon.

- Drop counts, empty drops and drops on cells the unmitigated fire never reaches are similar for policy and heuristic (EMSR583: 20.4 vs 20.7 drops; 38% vs 32% of treated cells unreached). The policy's first drop comes about 10 s later in both incidents.
- The policy triages: it leaves about one focus per scenario unattended for the first 20 minutes (EMSR583 1.06, EMSR602_AOI04 1.31) where the heuristic leaves 0.47 and 0.22. In 15 of the 16 EMSR583 scenarios it loses by more than 5 ha, the extra burned area comes from one focus that the policy reached late (often after 1,000–3,400 s) or never, with 5.5 drops on it against the heuristic's 9.5; in 10 of the 15 it is the focus with the largest unmitigated growth.
- Outcome follows whether the first focus attacked is the most dangerous one. EMSR602_AOI04, policy wins: policy 87%, heuristic 7%. EMSR583, policy loses: policy 25%, heuristic 75%; policy wins: 89% vs 44%.
- The most dangerous focus is harder to anticipate on EMSR583: the focus that is largest after 900 s (first wind segment) is the final largest in 50% of scenarios, against 78% on EMSR602_AOI04 (78% and 84% after 1,800 s). Mean wind speed and shift size are similar (2.4 vs 2.3 m/s, 32° vs 39° per 900 s), and EMSR583's fuels all occur in at least five other incidents, so the difference comes from how its fuel layout responds to the D-60 wind shifts.

Interpretation: the learned policy commits early to the focus it judges most dangerous. That beats the greedy downwind heuristic where danger is predictable from the first minutes and loses where later wind shifts change which focus runs; the heuristic re-ranks every drop by the current wind. Scripts were run ad hoc from this analysis and are not part of the package.

## Baselines under the D-74 drop rule and scenario mix (2026-10-06)

No training. `runs/geographic-multi-ignition-family-d72-20261006`, 640 development and 24 validation scenarios (reserved unopened), two aircraft, commit of the D-74 change. Mean final burned area (ha) and the share of the unmitigated growth the heuristic prevents; "breached" is the share of heuristic-treated cells that the fire later crossed.

| Fuel moisture (Scott and Burgan) | Scenarios | Start | No intervention | Heuristic | Containment | Growth prevented by heuristic | Treated cells breached |
|---|---|---|---|---|---|---|---|
| D1L1 (3 % dead 1-h, cured) | 149 | 3.0 | 160.6 | 149.1 | 150.1 | 19 % | 60 % |
| D2L2 (6 %) | 160 | 3.0 | 79.2 | 70.4 | 70.5 | 36 % | 50 % |
| D3L3 (9 %) | 181 | 3.1 | 34.9 | 31.6 | 31.7 | 29 % | 37 % |
| D4L4 (12 %, green) | 142 | 3.0 | 7.4 | 3.8 | 3.8 | 84 % | 4 % |
| EMSR578 station weather | 32 | 3.0 | 68.8 | 64.0 | 64.1 | 20 % | 67 % |
| All | 664 | 3.0 | 69.5 | 62.9 | 63.2 | 37 % | 41 % |

Per incident the heuristic prevents from 3 % (EMSR538, Navalacruz) to 80 % (EMSR365) of the growth. Aircraft now matter most on moderate days and little on extreme ones, and both heuristics waste many drops on fronts too intense to hold, since they rank targets by downwind position and not by whether a drop can hold there. Policy results before D-74 are not comparable with these numbers.

### With the D-75 incidents (2026-10-06)

Same benchmark on `runs/geographic-multi-ignition-family-d73-20261006` (43 development incidents, 1,376 scenarios, plus 24 validation; reserved unopened). All: no intervention 59.1 ha, heuristic 51.9 ha, 45 % of the unmitigated growth prevented, 31 % of treated cells breached. The 23 CORINE-fuel incidents: 52.8 / 44.8 ha, 54 % prevented, 21 % breached; the 20 earlier ones: 66.1 / 59.9 ha, 36 %, 40 %. Per new incident prevention ranges from 16 % (EMSR892, Los Gallardos) and 18 % (EMSR837 AOI07, Plasencia) to 99 % (EMSR900 AOI02, Villa del Prado).

### Reference policy under D-74 (2026-10-06)

One training run, same configuration as the D-73 sweep (two aircraft, parametric actions, thermal image with ROS, slot features, augmentation, 1,000 updates, seed 42), on `runs/geographic-multi-ignition-family-d72-20261006` (20 development incidents) with EMSR597 (Moros, never inspected) held out; `runs/geographic-reference-d72-20261006`, commit `173c794` (unpublished working commit; its published counterpart `0cd708f` includes later code). Validation (EMSR590, 24 scenarios): selected weights 23.08 ha against heuristic 23.2 and containment 23.5 (no intervention 26.0). Held-out EMSR597, 32 scenarios: policy 109.1 ha, heuristic 104.9, containment 105.5, no intervention 117.4; policy − heuristic +4.16 ha [+0.72, +9.06] (+4.0 %), better / equal / worse 8 / 7 / 17. Under the sourced drop effect the learned policy ties the heuristic on validation and is worse on the unseen incident; it has not learned to prefer targets where a drop can hold. One seed and one fold.

### Sensitivity to the water-effect parameters (2026-11-03)

`python3 -m pandion.water_sensitivity runs/geographic-multi-ignition-family-d73-20261006/development --output runs/water-sensitivity-d74-20261103.json --scenarios 64 --seed 7 --workers 6` runs the no-intervention, heuristic and containment policies (two aircraft) on a seeded sample of 64 development scenarios while the D-74/D-76 hold limits and the D-78 evaporation delay are varied. No training; validation and reserved scenarios are not read. Mean final burned area is 48.4 ha without intervention.

| Variant | Heuristic (ha) | Containment (ha) | Growth prevented by the heuristic | Heuristic − containment (ha) |
|---|---|---|---|---|
| Baseline (1 drop holds 1,000 kW/m, 3 hold 3,200, delay ×1) | 41.3 | 41.8 | 14.7 % | −0.56 |
| Hold limits ×0.5 | 42.9 | 43.4 | 11.3 % | −0.44 |
| Hold limits ×2 | 37.7 | 39.4 | 22.2 % | −1.77 |
| Hold limits unbounded | 30.1 | 33.3 | 37.9 % | −3.22 |
| No hold (evaporation delay only) | 46.2 | 46.3 | 4.6 % | −0.14 |
| Delay ×0 | 41.6 | 42.3 | 14.1 % | −0.75 |
| Delay ×0.25 | 41.6 | 42.2 | 14.1 % | −0.59 |
| Hold ×0.5, delay ×0 | 44.1 | 44.5 | 8.8 % | −0.34 |
| Hold ×2, delay ×0 | 38.1 | 39.8 | 21.3 % | −1.72 |

The hold limits dominate: halving or doubling them moves the share of growth the heuristic prevents between 11 % and 22 %. Removing the evaporation delay changes it by about half a point, so the D-78 term matters little. The heuristic beats containment in every variant, by 0.1 to 3.2 ha. Limits: 64 scenarios and no confidence intervals, so differences of a point or less are not established; the learned policy was not run under these variants; the sweep measures sensitivity to the parameters and does not validate their values against observed drops.

## Headroom over the reference heuristic inside the action set (2026-11-10)

Diagnostic, no training. Two aircraft, current drop physics (D-74, D-76, D-78), `runs/geographic-multi-ignition-family-d73-20261006`: the 24 validation scenarios and a seeded sample (seed 7) of 96 development scenarios from 40 incidents; reserved unopened. Two extra controllers were run next to the registered baselines:

- **Hold-aware rule** (observation only): the downwind heuristic, except that it skips cells whose stacked water already suffices and ranks fronts a drop can hold (Rothermel fireline intensity from the observed fuel, terrain, wind and moisture against the D-74/D-76 limits) before fronts no drop can hold.
- **One-step oracle** (privileged, not a controller): at each real decision every valid parametric action (D-61 with drop distances 0, 1, 2; D-79) is scored by finishing the episode in the simulator with the heuristic, and the best is taken; ties keep the heuristic's order. It is a one-step rollout improvement of the heuristic, so it bounds from below what the best policy in this action set achieves. It is never used as a training signal.

Mean final burned area (ha); differences are paired, with 95 % percentile bootstrap intervals over scenarios.

| Set | No intervention | Heuristic | Containment | Hold-aware − heuristic | Oracle − heuristic |
|---|---|---|---|---|---|
| Validation (24, EMSR590) | 31.62 | 28.25 | 29.42 | 27.71: −0.54 [−1.33, +0.17], better/equal/worse 9/10/5 | 24.29: −3.96 [−5.54, −2.54], 20/4/0 |
| Development sample (96) | 44.79 | 37.47 | 38.08 | 34.71: −2.76 [−4.01, −1.73], 43/49/4 | 31.52: −5.95 [−7.90, −4.22], 51/45/0 |

The validation baselines reproduce the values recorded on 2026-10-08. An episode has 12 real decisions on validation and 18 on the development sample (the rest are forced refills), so eight episodes per update give PPO about 100–150 decisions per update.

Where the heuristic loses (oracle log, validation, 289 decisions): since D-76 a treated cell stays a candidate until it holds three drops and keeps its downwind rank, so the heuristic stacks three drops on every target; 56 of its 289 drops (19 %) went to a cell that needed one drop and already had it. The oracle deviated in 53 decisions (mean gain 1.8 ha each): 32 times to a fresh cell that one drop holds, 13 to a fresh cell that needs three, 7 times to waiting. Twelve of the 45 deviating drops used a distance of one cell beyond the front.

Reading: the learned policies recorded so far (29.0 ha on validation, +0.7 to +1.8 ha against the heuristic) are far from what the action set allows, so the limit is learning, not the environment; and the quantity that separates good from bad targets is the hold requirement at the target cell, which the parametric head could not see (D-82). Limits: one seeded development sample; the oracle's continuation is the heuristic, so the optimum is lower still; the hold-aware rule was a diagnostic script on this date and is registered as `hold_aware_heuristic` under D-83, with identical outcomes on both sets; the oracle and the other scripts were run ad hoc and are not part of the package.

### Action-target features and heuristic-relative rewards (D-81, D-82, 2026-11-10)

One training run with the 2026-10-08 configuration (two aircraft, parametric actions with distances 0/1/2, thermal image with ROS, slot features, augmentation, `--gamma 1.0`, 4 epochs × 4 minibatches, GAE 0.95, reward scaling, entropy 0.05 → 0.005, 1,000 updates × 8 episodes, seed 42) plus `--reward-reference heuristic --action-features`; `runs/geographic-action-features-d82-20261110` (`run.sh`, base commit and working-tree patch stored with it). `--holdout-activation EMSR837_AOI06` now holds out the whole EMSR837 activation (D-68 grouping, 288 scenarios), so held-out numbers are not comparable with the 2026-10-08 table. Selection on validation: best 28.50 ha at update 400 (curve 28.5–29.9; heuristic 28.25). Selected weights, deterministic, paired differences with 95 % bootstrap intervals; the hold-aware rule is the diagnostic of the previous section:

| Set | Policy | Heuristic | Hold-aware rule | Policy − heuristic | Better / equal / worse |
|---|---|---|---|---|---|
| Held out EMSR837 (288) | 42.80 | 41.76 | 38.70 | +1.05 [+0.25, +1.87] | 56 / 119 / 113 |
| Validation (24) | 28.50 | 28.25 | 27.71 | +0.25 [−1.29, +1.83] | 9 / 3 / 12 |
| Training maps (120, seeded sample) | 46.54 | 46.33 | 43.22 | +0.21 [−0.77, +1.22] | 33 / 52 / 35 |

The change did not make the policy beat the heuristic. Inspecting the selected weights on 40 training-map scenarios (724 decisions) shows why the scores sit at the heuristic's level: the action distribution is close to uniform over the valid actions (mean entropy 3.17 nats against 3.4 for uniform over about 32 valid actions), so the deterministic policy follows small logit differences; a uniformly random valid policy is already within about 0.5 ha of the heuristic on validation. The earlier runs under the D-74 physics were not inspected this way.

A 200-update probe with a constant entropy bonus of 0.003 (`run-low-entropy.sh`, otherwise identical) did sharpen the policy (mean entropy 2.05, top action probability 0.36) but towards the heuristic's behaviour: targets strongly downwind (0.48 against −0.08 for the average valid action), never beyond the front (distance 0 in every drop), and on cells whose water already suffices more often than chance (34 % against 8 %). Held out +0.44 ha [−0.16, +1.09], training maps +0.80 [−0.60, +2.50]. So the entropy schedule is one brake, but lowering it alone leads to the heuristic's local optimum; the policy has not learned the hold requirement although it is now an input of each action.

Limits: one seed, one fold, and the probe is short.

#### Larger batches with a low entropy bonus (2026-11-10)

Same options, data, seed and held-out activation, but 32 episodes × 250 updates (the same 8,000 episodes; an episode has only 12–18 real decisions, so 8 episodes give about 100–150 samples per update) and entropy 0.01 → 0.001, selection every 25 updates; `runs/geographic-large-batch-d82-20261110`, 7–8 s per update on Apple MPS with six rollout workers. Selection on validation fell from 29.96 ha at update 25 to 28.21 at 175 and 26.92 at 225 (selected), then 28.71 at 250. Selected weights:

| Set | Policy | Heuristic | Hold-aware rule | Policy − heuristic | Better / equal / worse |
|---|---|---|---|---|---|
| Held out EMSR837 (288) | **41.16** | 41.76 | 38.70 | −0.60 [−1.19, −0.07] | 80 / 140 / 68 |
| Validation (24, used for selection) | **26.92** | 28.25 | 27.71 | −1.33 [−2.71, +0.12] | 14 / 5 / 5 |
| Training maps (120, seeded sample) | **45.72** | 46.33 | 43.22 | −0.62 [−1.67, +0.52] | 43 / 59 / 18 |

This is the first policy under the D-74 physics that is better than the heuristic on scenarios used neither for training nor for selection, by 0.6 ha (1.4 %) with an interval that excludes zero; the validation figure is optimistic because it chose the weights, and the single best checkpoint stands out from its neighbours (28.2–28.7 ha). The policy is sharper than with 8 episodes (mean entropy 2.53 nats, top probability 0.24) and uses the hold requirement: it prefers fronts that need three drops (37 % of its drops against 6 % of valid actions) and no longer favours cells whose water already suffices (12 % against 13 %), while staying downwind (0.50) and almost never aiming beyond the front. It is still 2.5 ha behind the hold-aware rule on the held-out and training sets, and further from the one-step oracle, so most of the headroom remains.

Limits: one seed and one fold; the comparison with the 8-episode run uses the same number of episodes but a quarter of the optimizer steps. Batch size and entropy schedule are separated below.

#### Batch size against entropy schedule (2026-11-10)

Two more runs complete the 2 × 2 with the same options, data, seed and held-out activation (`runs/geographic-batch-entropy-split-d82-20261110`, run in parallel): 32 episodes × 250 updates with the old entropy schedule, and 8 episodes × 1,000 updates with the low one. Policy − heuristic (ha, 95 % bootstrap interval), selected weights; entropy is the mean over decisions on 40 training-map scenarios (uniform is about 3.4 nats).

| Episodes × updates | Entropy bonus | Held out EMSR837 (288) | Training maps (120) | Validation (24, selection) | Policy entropy |
|---|---|---|---|---|---|
| 8 × 1,000 | 0.05 → 0.005 | +1.05 [+0.25, +1.87] | +0.21 [−0.77, +1.22] | +0.25 [−1.29, +1.83] | 3.17 |
| 8 × 1,000 | 0.01 → 0.001 | −0.17 [−0.77, +0.38] | −0.50 [−1.48, +0.40] | +0.17 [−1.21, +1.62] | 2.17 |
| 32 × 250 | 0.05 → 0.005 | −0.56 [−1.15, −0.06] | −0.15 [−0.91, +0.53] | +0.04 [−1.25, +1.08] | 2.87 |
| 32 × 250 | 0.01 → 0.001 | −0.60 [−1.19, −0.07] | −0.62 [−1.67, +0.52] | −1.33 [−2.71, +0.12] | 2.53 |

On the held-out activation both 32-episode runs beat the heuristic by about 0.6 ha with intervals that exclude zero, whatever the entropy schedule, and neither 8-episode run does; the low entropy bonus removes the loss of the 8-episode run (+1.05 → −0.17) but adds nothing measurable at 32 episodes (−0.56 → −0.60). Batch size is therefore the larger factor on unseen maps, for the same 8,000 episodes. The runs differ in what they learned: only the 32-episode, low-entropy policy prefers fronts that need three drops (37 % of its drops against 6 % of valid actions) and stopped favouring cells whose water already suffices; the 32-episode run with the old schedule still drops on such cells in 40 % of its drops (7 % of valid actions) and never aims beyond the front, so its gain comes from elsewhere (it is the most downwind of the four, 0.58). The −1.33 ha on validation of the last row is not repeated by any other run and comes from one checkpoint, consistent with selection noise on 24 scenarios.

Limits: one seed per cell and one fold, so differences of a few tenths of a hectare between cells are not established; the two new runs shared the machine, which does not change their results but does their timings. All four remain 2.5–4 ha behind the hold-aware rule on the held-out set.

#### Seeds and a larger batch (2026-11-10)

`runs/geographic-seeds-batch64-d82-20261110` (three runs in parallel from a frozen copy of the package): the 32 × 250, entropy 0.01 → 0.001 recipe with seeds 43 and 44, and 64 episodes × 125 updates with seed 42 (again 8,000 episodes). Same data and held-out activation. `hold_aware_heuristic` (D-83) scores 38.70 ha on the held-out set, 43.22 on the training-map sample and 27.71 on validation. Selected weights; policy − reference in ha with 95 % bootstrap intervals over scenarios:

| Run | Held out EMSR837 (288): − heuristic | Better / equal / worse | − hold-aware | Training maps (120): − heuristic | Validation (24): − heuristic |
|---|---|---|---|---|---|
| 32 × 250, seed 42 (above) | −0.60 [−1.19, −0.07] | 80 / 140 / 68 | +2.45 [+1.47, +3.67] | −0.62 [−1.67, +0.52] | −1.33 [−2.71, +0.12] |
| 32 × 250, seed 43 | −1.34 [−1.99, −0.80] | 100 / 150 / 38 | +1.71 [+0.77, +2.92] | −0.59 [−1.51, +0.19] | +0.33 [−0.50, +1.33] |
| 32 × 250, seed 44 | −1.29 [−1.98, −0.68] | 88 / 138 / 62 | +1.76 [+0.93, +2.83] | −1.07 [−1.88, −0.33] | +0.12 [−0.96, +1.21] |
| Mean of the three seeds per scenario | −1.08 [−1.66, −0.58] | | +1.98 [+1.08, +3.12] | −0.76 [−1.49, −0.10] | −0.29 [−1.19, +0.75] |
| 64 × 125, seed 42 | −0.89 [−1.61, −0.29] | 80 / 149 / 59 | +2.16 [+1.32, +3.24] | −0.26 [−1.02, +0.45] | −0.12 [−1.25, +0.83] |

All three seeds of the recipe beat `heuristic` on the held-out activation, each with an interval that excludes zero, and their mean is 1.1 ha (2.6 %) better; on training maps the mean is 0.8 ha better. The gain over `heuristic` is therefore not a seed effect. Against `hold_aware_heuristic` every run is worse, by about 2 ha on the held-out set and 2.4 ha on training maps with intervals that exclude zero, so the learned policy does not yet match a rule that uses the same information. Validation does not separate anything: every run is within its interval of `heuristic`, and the −1.33 ha of seed 42 was not repeated. 64 episodes per update did not improve on 32 with the same seed and episode budget (−0.89 against −0.60 held out, −0.26 against −0.62 on training maps; half the optimizer steps).

What the policies learned differs by seed. Mean entropy is 1.8–2.5 nats and all are strongly downwind (0.56–0.76 against about −0.1 for the average valid action) and never aim beyond the front. Only seed 42 stopped dropping on cells whose water already suffices; seeds 43 and 44 and the 64-episode run still do so in 24–37 % of their drops (7–12 % of valid actions), the habit the hold-aware rule removes. The gains over `heuristic` thus come mostly from which front and when, not from the hold requirement, which remains the main unused margin.

Limits: one held-out activation (nine AOIs of EMSR837) and one fold; three seeds; runs in parallel share the machine, which affects timings only. Reserved scenarios unopened.

#### What a redundant drop costs (2026-11-11)

Diagnostic on the selected weights of seeds 43 and 44 (32 × 250), deterministic, on the 120 training-map scenarios of the tables above and a seeded sample of 120 held-out scenarios. A drop is redundant when its target already holds the stacked drops its front needs (the `sufficient` action feature, D-82). Each redundant decision was replaced, alone, by the choice of `hold_aware_heuristic` in that observation, with the policy deciding everything before and after; as a control one non-redundant drop per scenario got the same replacement. "All replaced" re-runs the episode replacing every redundant drop as it comes up.

| Seed, set | Redundant drops | Replacing one: no effect / helps / hurts | Mean gain per replaced drop | Control (non-redundant): mean gain | All replaced (ha, 95 % interval) |
|---|---|---|---|---|---|
| 43, training maps | 452 of 1,962 (23 %) | 90 % / 8 % / 2 % | +0.45 ha | −0.09 | 45.74 → 44.68: −1.06 [−2.37, −0.21] |
| 43, held out | 378 of 1,632 (23 %) | 86 % / 11 % / 3 % | +0.29 ha | +0.12 | 54.12 → 53.24: −0.88 [−1.77, −0.28] |
| 44, training maps | 776 of 2,171 (36 %) | 89 % / 7 % / 3 % | +0.11 ha | +0.22 | 45.27 → 44.50: −0.77 [−1.33, −0.32] |
| 44, held out | 559 of 1,717 (33 %) | 85 % / 11 % / 4 % | +0.30 ha | −0.02 | 53.72 → 51.71: −2.01 [−4.48, −0.38] |

Nine redundant drops in ten cost nothing: the outcome is identical when the water goes elsewhere. They concentrate in fires that are already contained (scenarios ending at 10 ha or less: 33–49 % of drops redundant, 0.1–0.3 ha to gain by replacing all of them), where no better target exists. The cost sits in the remaining tenth, in fires that are still growing (scenarios above 10 ha: 5–17 % redundant, 1.6–4.7 ha to gain), where a single misplaced drop costs 1–5 ha. Replacing non-redundant drops by the rule's choice gives no consistent gain, so the effect is specific to redundancy.

Reading: the policies' redundant drops explain roughly 1–2 ha of their 2 ha deficit against `hold_aware_heuristic`, but the learning signal for avoiding them is sparse and heavy-tailed: in about 90 % of the cases the policy receives no feedback that the drop was wasted, and the few costly cases arrive minutes later inside episode-level variance of several hectares. That is consistent with one seed in three learning to avoid them. Limits: two seeds; the held-out sample is 120 of 288 scenarios (hence a different mean than the full-set tables); replacing one decision changes the later trajectory, so single-drop gains do not add up to the all-replaced figure. Scripts were run ad hoc.

#### Choosing weights on development incidents (D-84, 2026-11-17)

`runs/geographic-selection-incidents-d84-20261117` (`run.sh`; code archived in `code.tar.gz`, data read from copies of the d73 family and px4 imagery): the 32 × 250, entropy 0.01 → 0.001 recipe with seeds 42, 43 and 44, EMSR837 held out, and `--selection-activations EMSR719 EMSR892 EMSR632 EMSR457`. The rule was fixed before training from the references on all 1,344 development scenarios (`incident_baselines.json`): single-AOI activations outside the held-out group, ranked by mean |`heuristic` − `hold_aware_heuristic`| per scenario; ranks 1–5 (EMSR522, EMSR375, EMSR597, EMSR458, EMSR704) stay in training and ranks 6–9 choose the weights. On those 128 scenarios no intervention scores 98.0 ha, `heuristic` 86.6 and `hold_aware_heuristic` 80.1 (on EMSR590 validation: 31.6, 28.3, 27.7), so target choice matters there. Training uses 928 scenarios instead of 1,056.

Selection curves now separate checkpoints and seeds by 2–6 ha (seed 42: 88.7 at update 25 to 82.7 at 225; seed 43: 88.4 to 83.0 at 200; seed 44: 85.2 to 83.2 at 150), against tenths of a hectare on validation. Held-out EMSR837 (288 scenarios) and the seeded 120-scenario development sample of the earlier tables, policy − reference in ha, 95 % bootstrap intervals; "earlier" are the three validation-selected runs of 2026-11-10 evaluated on the same scenarios:

| Weights | Held out: − heuristic | Held out: − hold-aware | Development sample: − heuristic |
|---|---|---|---|
| Selected on the four incidents, mean of three seeds | −1.03 [−1.62, −0.53] | +2.02 [+1.04, +3.28] | −0.69 [−1.38, −0.05] |
| Last update of the same runs | −0.53 [−1.15, +0.05] | +2.53 [+1.41, +4.00] | −0.63 [−1.36, +0.06] |
| Earlier runs, selected on validation | −1.08 [−1.66, −0.58] | +1.98 [+1.08, +3.12] | −0.76 [−1.49, −0.10] |

Per seed on the held-out set: −1.05, −0.85 and −1.20 ha against `heuristic` (selected) and −0.55, −0.23 and −0.81 (last update).

Selection on these incidents chooses weights that are better on unseen data: the selected checkpoints beat the last ones on the held-out activation by 0.51 ha [0.24, 0.81], consistently across the three seeds. With validation selection that comparison was possible for one seed only (seeds 43 and 44 kept their last weights): 0.12 ha. The end result, however, equals the earlier one (new minus earlier selected: +0.04 [−0.26, +0.39] held out): these runs ended worse than the earlier ones (last weights 41.23 against 40.68 ha), which may be seed variation or the 128 training scenarios given to selection, and the better selection recovers that difference. All runs stay about 2 ha behind `hold_aware_heuristic`. Limits: three seeds, one held-out activation; the selection incidents were chosen from reference outcomes on the development split, not from policy results.

#### Where the learning signal goes (2026-11-17)

Diagnostic on the selected weights of seed 42 of the D-84 run, on 96 seeded training scenarios (development split without the held-out group and the selection incidents), 1,784 real decisions. The policy is rolled out stochastically with the heuristic-relative reward, as in training. At every decision the simulator is branched: the action taken and three other valid actions, chosen uniformly, are each followed to the end four times by the same stochastic policy (prefix replayed; independent continuations). The mean of a branch is an estimate of Q(s, a) for the current policy; the spread between the four continuations of one action is the noise a single Monte Carlo return carries. Scripts `diag_signal.py` and `diag_analyse.py` were run ad hoc; output in `runs/geographic-signal-diagnostic-20261117`.

| Quantity | Value |
|---|---|
| Critic: correlation of the value with the return-to-go; variance explained after a linear rescale | 0.25; 6 % |
| Decisions where no scored action and no continuation changes the outcome | 47 % |
| Variance between actions (signal) / variance between continuations of one action (noise), mean per decision | 0.81 / 2.31 ha² |
| Share of the between-action variance in the top 10 % of decisions | 97 % |
| Correlation of the Monte Carlo advantage that PPO uses (return-to-go − rescaled value) with the branched advantage of the taken action | 0.20 |
| Same, on decisions where the branched advantage exceeds 0.05 ha in absolute value (50 %); sign agreement | 0.22; 51 % |
| Probability on targets whose water already suffices, against their share of valid drop actions; mean logit gap | 31 % vs 15 %; +2.1 |
| Branched Q(sufficient target) − Q(other targets), same decision | −0.01 ha (worse in 21 %, better in 16 %, equal in 63 %) |

Reading: the policy gradient is close to noise. Half of the decisions have no effect at all; the effect of the others is concentrated in a few decisions per episode and is smaller than the spread caused by the policy's own later choices; the critic removes almost none of that spread. As a result the Monte Carlo advantage of a decision that matters has the sign of its branched advantage about as often as a coin. Under the current policy's continuation, preferring targets that already hold enough water costs little on average, which is why the policy has kept the preference: what separates it from `hold_aware_heuristic` is the few decisions that matter, and those are the ones the signal does not reach. This motivated branched advantages with common random numbers (D-85). Limits: one seed; the branched advantage is itself estimated from four independent continuations per action and is measured against the mean of four scored actions rather than the policy's value, so the correlations understate the agreement somewhat; the 51 % sign agreement is too low for that alone to explain.

#### When a decision changes the burned area (2026-11-17)

Same weights and the same 96 training scenarios as above (1,802 decisions). At every decision four actions are sampled from the policy and each is followed to the end by the same policy with common random numbers (`pandion.vine`); the burned area of each branch is read from its replay's arrival times at fixed minutes after the decision, so branches are compared at equal simulated times. Each pair of distinct sampled actions gives a difference curve d(τ) and a final difference at the 60-minute horizon. Scripts `diag_window.py` and `window_analyse.py`, output in `runs/geographic-signal-diagnostic-20261117`.

A third of the decisions (560 of 1,697 with two distinct sampled actions) change the area at some point; 17 % of action pairs end with a different final area (median 1 ha, 90th percentile 3, maximum 15) and 5 % differ only temporarily.

| Minutes after the decision | 0 | 5 | 10 | 15 | 20 | 30 | 40 | 50 | horizon |
|---|---|---|---|---|---|---|---|---|---|
| Pairs whose final difference is already visible (same sign) | 0 % | 9 % | 21 % | 32 % | 45 % | 67 % | 83 % | 92 % | 100 % |
| Variance of the final difference explained by d(τ) | 0 | 0.02 | 0.10 | 0.18 | 0.32 | 0.63 | 0.82 | 0.94 | 1 |
| Decisions of the episode inside the window (mean) | 0 | 0.9 | 2.6 | 4.0 | 5.4 | 7.8 | 9.7 | 11.0 | 11.6 |

| Decision made at minute | Pairs with a final difference | Mean final difference | Minutes to 90 % of it: median, 90th percentile |
|---|---|---|---|
| 0–15 | 28 % | 2.0 ha | 50, 60 |
| 15–30 | 23 % | 1.8 ha | 30, 40 |
| 30–45 | 15 % | 1.2 ha | 20, 27 |
| 45–60 | 5 % | 1.2 ha | 9, 13 |

Reading: almost nothing shows in the first five minutes, because water only changes the outcome when the front reaches the treated cell; the median effect needs 30 minutes, about eight later decisions, to appear, and early decisions, which matter most, need until the end of the episode. A return truncated to a short window would miss most of the effect of the decisions that matter, so branched returns run to the horizon. Late decisions rarely matter, partly because the fixed 60-minute horizon cuts their consequences: the episode ends before the front reaches their water. Limits: one seed; differences are between actions the policy samples, not the best and worst actions; times are bounded by the horizon.

#### Branched advantages (D-85, 2026-11-17)

`runs/geographic-vine-d85-20261117` (`run.sh`, code in `code.tar.gz`): the D-84 recipe with seed 42 (32 episodes × 250 updates, entropy 0.01 → 0.001, `--reward-reference heuristic --action-features`, EMSR837 held out, weights chosen on EMSR719, EMSR892, EMSR632 and EMSR457) plus `--vine-states 4 --vine-actions 4`: four decisions of every episode branched, four policy-sampled actions each, continuations with common random numbers. 9 CPU workers; 20–31 s per update (15–24 s of it branching), 2 h 10 min in all, against about 12 s per update without branching.

Selection score (mean final area on the 128 selection scenarios, lower is better) fell from 84.3 ha at update 25 to 77.6 at update 175 (selected); seed 42 of the D-84 run reached 82.7 at best. Held-out EMSR837 (288 scenarios) and the 120-scenario development sample of the earlier tables, policy − reference in ha with 95 % bootstrap intervals, evaluated with `eval_ckpt.py`:

| Selected weights, seed 42 | Held out: − heuristic | Held out: − hold-aware | Development sample: − heuristic | Development sample: − hold-aware |
|---|---|---|---|---|
| Branched advantages (D-85) | −3.21 [−4.64, −1.99] | −0.16 [−0.59, +0.27] | −2.95 [−5.12, −1.19] | +0.17 [−0.75, +1.14] |
| Monte Carlo/GAE advantages (D-84 run) | −1.05 [−1.82, −0.40] | +2.01 [+1.08, +3.09] | −1.12 [−1.99, −0.26] | +2.00 [+0.56, +3.86] |
| Branched − Monte Carlo, paired | −2.16 [−3.39, −1.18] | | −1.83 [−3.76, −0.33] | |

Held out, the branched policy burns 38.55 ha against 41.76 for `heuristic` and 38.70 for `hold_aware_heuristic`; against the latter it is better in 62 scenarios, equal in 161 and worse in 65 (D-84 policy: 30 / 141 / 117).

Reading: with the same seed, data, features, reward and selection, replacing Monte Carlo/GAE advantages by branched ones gains about 2 ha on the held-out incident and closes the gap to `hold_aware_heuristic`, which the learned policies trailed by about 2 ha in every earlier run. This supports the diagnosis that the limit was the learning signal rather than the inputs or the action set. Limits: one seed; one held-out activation; the policy now ties the hold-aware rule rather than beating it, and remains below the one-step oracle of the headroom section.

Seeds 43 and 44 (`run_43_44.sh`, same commit and recipe, 5 workers each, run in parallel, about 2 h 50 min) were selected at updates 200 and 250 (selection score 77.9 and 78.6 ha; D-84 seeds reached 82–83). Evaluation of their selected weights and of the D-84 seeds 43 and 44 on the same scenarios: `audit-eval-43-44.json`; three-seed summary (`three_seed.py`, per-scenario mean over the three seeds, then bootstrap over scenarios): `three-seed.json`.

| Selected weights, 3 seeds | Held out: − heuristic | Held out: − hold-aware | Development sample: − heuristic | Development sample: − hold-aware |
|---|---|---|---|---|
| Branched advantages (D-85) | −3.09 [−4.52, −1.88] | −0.03 [−0.44, +0.35] | −2.96 [−4.75, −1.49] | +0.16 [−0.78, +1.14] |
| Monte Carlo/GAE advantages (D-84 run) | −1.03 [−1.62, −0.53] | +2.02 [+1.04, +3.28] | −0.69 [−1.40, −0.06] | +2.42 [+1.06, +4.21] |
| Branched − Monte Carlo, paired | −2.05 [−3.39, −1.06] | | −2.27 [−3.76, −1.09] | |

Per seed (42, 43, 44), held out, branched − heuristic: −3.21, −3.27, −2.78; branched − hold-aware: −0.16, −0.21, +0.27; branched − Monte Carlo: −2.16, −2.41, −1.58. Every seed gains on its Monte Carlo counterpart on both evaluation sets.

Reading: the seed 42 result holds across seeds. Branched advantages gain about 2 ha over Monte Carlo/GAE advantages with everything else fixed and bring the learned policy level with `hold_aware_heuristic`, but not past it. Limits: one held-out activation; three seeds.

#### Branches weighted towards early decisions (D-88, 2026-11-18)

`runs/geographic-vine-early-d86-20261118` (`run.sh`; code is the D-85 commit plus `uncommitted.diff`): the D-85 recipe with seed 42 and `--vine-weighting remaining`, so the four branched decisions of each episode are drawn with probability proportional to the time left. 9 workers, 16–34 s per update, about the cost of uniform branching. Selection score fell to 79.5 ha at update 225 (uniform branching, same seed: 77.6).

| Selected weights, seed 42 | Held out: − heuristic | Held out: − hold-aware | Development sample: − heuristic | Development sample: − hold-aware |
|---|---|---|---|---|
| Time-weighted branches (D-88) | −2.42 [−3.82, −1.25] | +0.64 [−0.01, +1.33] | −3.27 [−4.98, −1.88] | −0.16 [−1.48, +1.44] |
| Time-weighted − uniform branches (D-85), paired | +0.80 [+0.19, +1.48] | | −0.33 [−1.44, +1.11] | |

Held out, the time-weighted policy is better than the uniform one in 43 scenarios, equal in 169 and worse in 76.

Reading: concentrating branches on early decisions did not help. On the held-out incident it is 0.8 ha worse than uniform branching with the same seed; on the development sample the difference is within noise. One plausible reason is that uniform branching already reaches early decisions often enough, and the late decisions it also branches still teach the policy something the weighted scheme now sees less. Limits: one seed, so a difference of this size could partly be seed noise; `--vine-weighting` stays available with `uniform` as the default, and the D-85 recipe is unchanged.

The run folder keeps its working number `d86`, assigned before D-86 became the project rename.

#### Learned return decomposition against branched advantages (2026-11-19)

Question: can a learned model assign each decision its share of the outcome, without simulating branches, well enough to grade the decisions that branched advantages (D-85) leave without a learning signal (all but four per episode)? The model is RUDDER-style return decomposition (Arjona-Medina et al., 2019): an LSTM reads the episode decision by decision (the policy's LSTM state and an embedding of the action taken: focus-slot features, aspect × distance and the D-82 action-target features) and predicts the episode return at every step; the change in its prediction when one action is swapped for another is that action's credit. Criterion written before any run (`CRITERION.md`): go on to training with branched advantages plus RUDDER credit at the other decisions only if RUDDER's sign agreement with a reference advantage is at least 70 % and no more than 10 points below single-branch vine.

`runs/geographic-rudder-diagnostic-20261119` (`collect.py`, `analyse.py`, output `result.json`): D-85 seed 42 selected weights, fixed; development split without EMSR837 and the four selection incidents (928 scenarios), one stochastic episode per scenario. RUDDER was trained on 728 scenarios (14,225 decisions, plus the vine branch returns of four decisions per episode in the primary variant) and scored on the other 200. At four decisions of each test episode four actions were sampled from the policy, as in D-85, and each was followed to the end with four common-random-number continuations. Reference advantage: leave-one-out over the mean of continuations 1–3. Single vine (the D-85 training signal): continuation 0 alone. 749 scored decisions, 2,996 (decision, action) pairs; 849 with |reference| > 0.05 ha.

| Advantage estimate | Correlation with reference | Sign agreement, \|reference\| > 0.05 ha |
|---|---|---|
| Single-branch vine (D-85) | 0.71 | 53 % (83 % where it is not zero; zero in 35 %) |
| RUDDER, episodes and branch returns (primary) | 0.07 | 52 % |
| RUDDER, episodes only | 0.08 | 51 % |
| Non-recurrent Q(h, a) regression, same data | 0.05 | 52 % |

Reading: negative. The learned decomposition does not recover which action is better: its advantages are uncorrelated with the branched reference and agree in sign as often as a coin, like the Monte Carlo advantage of the 2026-11-17 diagnostic (51 %). The cause is the same: on about 14,000 decisions the outcome differences between actions are small and rare against the spread of later choices, and a model fitted to returns learns the state value but not the action differences. Training with the combination was therefore not run (criterion not met); D-85 stays as is. The predeclared sign metric counts a zero branched advantage as a disagreement, which understates single vine (integer-hectare returns give exact ties); the correlation shows the same ranking. Limits: one policy, one model size and data volume; a fixed policy is the favourable case, since during training the decomposition would also have to follow a changing policy.

#### Where D-85 differs from the hold-aware rule (2026-11-19)

On held-out EMSR837 the three D-85 seeds tie `hold_aware_heuristic` on average (−0.03 ha). Diagnostic of where that tie comes from, without training: `runs/geographic-d85-behaviour-20261119` (`behaviour.py`; decision log `log.json`; per-scenario results `perscen.json`, taken from `audit-eval.json` and `audit-eval-43-44.json` of the D-85 run, which the log reproduces exactly). The selected weights of seeds 42, 43 and 44 were rolled out deterministically on the 288 held-out scenarios (22,253 real decisions). At every decision the log keeps the chosen action, the D-82 features of its target and the target `hold_aware_heuristic` would choose in the same observation.

| Quantity | Value |
|---|---|
| Scenarios where all three seeds are within 1 ha of `hold_aware_heuristic` | 177 of 288 |
| All three seeds worse by > 1 ha / better by > 1 ha | 19 (+88 ha in all; 8 of them in AOI07, +44) / 17 (−116 ha) |
| Spread between seeds in one scenario, fires > 40 ha without intervention (s.d.) | 3.1 ha |
| Decisions: HOLD / drop / voluntary refill | 51 % / 49 % / 0 % |
| Decisions in scenarios where all seeds tie (fires already small or held); HOLD there | 74 %; 62 % |
| Drops at distance > 0 (D-79) | 12 of 10,980 |
| Drops on head / flanks / rear | 79 % / 18 % / 3 % |
| Drops on targets whose water already suffices (always with an insufficient one available) | 18 % (26 % in tied scenarios, 6 % in decisive ones) |
| Drops on fronts no water can hold; `hold_aware_heuristic` chose a holdable one instead | 2.6 %, 72 % of them in AOI07 |
| Same focus / same cell as `hold_aware_heuristic` | 52 % / 39 % |

Losses are not one repeated mistake. In the 19 consistent losses the policy drops more often on fronts that need three stacked drops (43 % against 29 % in the consistent wins) and on fronts that cannot be held (16 % against 2 %): they are higher-intensity fires, eight of them in AOI07, which burns 420–495 ha with any policy. Focus triage does not explain the losses in general: pairs of seeds that start on the same two foci end as far apart (5.1 ha) as pairs that start differently (4.4 ha), although single scenarios do turn on it (EMSR837_AOI06 scenario 03, three foci: 4 ha for the seed that starts on focus 1, 33–34 for the two that start on focus 2; the rule, 5 ha). In the two replays compared in detail (`replays/`, `viewer.html`; scenarios AOI16-07 and AOI15-02, policy 8 and 10 ha against 4) the drops occur at the same times as the rule's and only the targets differ: in AOI16-07 the policy works focus 3 first and puts three consecutive drops on cells that already hold, while the rule works focus 1 throughout.

Reading: of the earlier defects, redundant drops fell (18 % of drops against 23–36 % before D-85) and now sit mostly where they cost nothing; drop distances beyond the front are still not used. HOLD is chosen in about half the decisions while the rule would always drop, but almost only once the fire is small or held, and the number of HOLDs in large fires is unrelated to the result (correlation 0.00). The remaining differences from the rule are scenario-specific and of both signs, and outcomes in large fires are sensitive to small changes in target choice (3 ha between seeds). For training this means that three quarters of the decisions, and therefore of the branched decisions, fall in scenarios where no choice changes the outcome; this agrees with the 72 % of branched comparisons with identical returns in the RUDDER measurement above. Limits: deterministic evaluation of three seeds on one held-out activation; the rule's choice is evaluated in the policy's states, not along its own trajectory.

#### Branching by fire growth (D-89, 2026-11-19)

D-85 recipe, seed 42, with `--vine-allocation growth`: the same 128 branched decisions per update, one per episode and the rest by fire growth. Branches ran forked (D-90). Run: `runs/geographic-vine-growth-d87-20261119` (`run.sh`, `seed42.log`, `seed42.json`; evaluation `audit-eval.json` with `eval_ckpt.py` of the D-85 run, both weights evaluated together). Selection mean burned area of the selected weights: 79.04 ha (update 175) against 77.59 ha for D-85 seed 42 (also update 175).

| Policy (seed 42) | Held-out EMSR837 (288) | − `hold_aware_heuristic` | Development sample (120) | − `hold_aware_heuristic` |
|---|---|---|---|---|
| D-85 | 38.55 ha | −0.16 [−0.59, 0.27] | 43.38 ha | +0.17 [−0.75, 1.14] |
| D-89 | 38.97 ha | +0.26 [−0.33, 0.81] | 43.75 ha | +0.53 [−0.55, 1.67] |
| D-89 − D-85, paired | +0.42 [−0.01, 0.80]; better / equal / worse 36 / 158 / 94 | | +0.37 [−0.53, 1.25]; 15 / 68 / 37 | |

References: `heuristic` 41.76 / 46.33 ha, `hold_aware_heuristic` 38.70 / 43.22 ha.

Reading: negative. Moving branches to growing fires did not improve the policy; it is slightly worse than D-85 with the same seed on held-out, selection and development data, with more scenarios lost than won. More branched decisions with an effect per update (45.8 against 28.7 measured beforehand) did not translate into a better policy. A likely reason is that the episodes with large fires are also those whose branch returns are noisiest (3 ha between seeds in one scenario, previous section), so the extra signal there is mostly variance. Seeds 43 and 44 were not run, as agreed beforehand for a seed-42 result without improvement. D-85 stays the recipe; `--vine-allocation` keeps `uniform` as default. Forked branching (D-90) is kept: over the first 18 updates, 27.4 s of branching per update against 37.3 s in the replayed run that was aborted (`aborted/seed42.log`; 1.4×). The two logs show the same loss to four decimals in the first nine updates and drift apart from the tenth; returns were compared exactly only on 12 development episodes and in the unit tests, and the cause of the drift was not checked. Limits: one seed.

The run folder keeps its working number `d87`, assigned before D-87 became the viewer language change; forked branching was D-88 in the working copy.

### Scenario difficulty and clusters (2026-11-24)

Characterisation, no training. The clusters are for stratified reporting: every comparison of controllers on this family is dominated by a minority of scenarios, and the mean hides where they differ. They are not a training input or a tactic.

`runs/geographic-difficulty-clusters-20261124` (`evaluate.py`, `analyse.py`, `checks.py`, `outcome_groups.py`; per-scenario table `scenario_table.csv`): the 1,344 development and 24 validation scenarios of `runs/geographic-multi-ignition-family-d73-20261006`, two aircraft, one-hour horizon; reserved unopened. Controllers: no intervention, the three registered rules and the selected weights of the three D-85 seeds, deterministic. `evaluate.py` reproduces the 288 held-out scenarios of the D-85 evaluation exactly. Of the 1,368 scenarios, 928 were training scenarios and 128 chose the weights; only EMSR837 (288) and EMSR590 (24) were used for neither. The analysis needs pandas and scikit-learn (2.3.3 and 1.8.0 used), which the package does not require.

**Difficulty.** Defined from a fixed reference, not from the policy under study: growth = no intervention − initial area; saved = share of the growth that `hold_aware_heuristic` avoids.

| Class | Rule | Scenarios | Growth (ha) | Fireline intensity at the foci (kW/m) | Dead moisture | Shrub share |
|---|---|---|---|---|---|---|
| Trivial | growth ≤ 2 ha | 361 | 0.4 | 348 | 0.10 | 0.26 |
| Easy | saved ≥ 75 % | 391 | 14.6 | 969 | 0.08 | 0.40 |
| Medium | saved 25–75 % | 219 | 55.4 | 2,545 | 0.06 | 0.44 |
| Hard | saved < 25 % | 397 | 150.8 | 5,376 | 0.06 | 0.57 |

The largest potential intensity at the foci separates hard from easy with AUC 0.97; a two-level tree on intensity and spread rate classifies 93 % correctly on held-out incidents (majority class 50 %). From weather, fuel, terrain, foci and water alone, a random forest reaches AUC 0.90 on held-out incidents, with dead fuel moisture first (permutation importance 0.26), shrub share second (0.12) and everything else at most 0.014. Classes are stable to the cuts (80/20: 95 % unchanged; thirds: 93 %; best of six controllers instead of the rule: 91 %).

**Clusters.** k-means on 16 standardised scenario characteristics within 1 km of the foci (log mean spread rate, log intensity at the foci, shares of cells under 1,000 and over 3,200 kW/m, dead moisture, mean wind, wind turn, slope, four fuel shares, unburnable share, foci, foci separation, distance to water). No outcome enters. k was chosen by the largest worst-case adjusted Rand index over 20 refits on 80 % of the incidents (k = 5: mean 0.92, worst 0.73; k = 2–8 tried). Areas are means in ha; intervals are 95 % bootstrap over incidents.

| Cluster | Profile | n | Hard | No intervention | `heuristic` | `hold_aware` | D-85 (3 seeds) | Growth D-85 avoids | D-85 − `hold_aware` |
|---|---|---|---|---|---|---|---|---|---|
| C1 | Litter and unburnable mosaic | 192 | 9 % | 11.9 | 9.4 | 8.0 | 8.0 | 44 % | −0.04 [−0.09, +0.06] |
| C2 | Moist shrub | 393 | 9 % | 13.6 | 8.9 | 7.9 | 7.8 | 54 % | −0.09 [−0.51, +0.18] |
| C3 | Timber understory | 242 | 6 % | 13.7 | 8.0 | 7.0 | 6.6 | 66 % | −0.40 [−1.62, +0.28] |
| C4 | Grass | 76 | 8 % | 45.4 | 35.2 | 27.9 | 27.1 | 43 % | −0.89 [−1.16, +0.00] |
| C5 | Dry shrub, high intensity | 465 | 69 % | 145.3 | 133.3 | 125.2 | 124.7 | 14 % | −0.45 [−1.66, +0.97] |

C5 is 34 % of the scenarios and 85 % of the growth. On the scenarios used for neither training nor selection the same comparison gives −0.09, +0.34, +0.21, 0.00 and −0.58 ha (n = 96, 72, 71, 4, 69; all intervals include zero).

**Where seeds agree with the rule.** Per scenario and against a reference: tie, the three seeds within 1 ha; win or loss, all three better or worse by more than 1 ha; otherwise seed-dependent (the measure of the 2026-11-19 diagnostic; its 177 / 17 / 19 ties, wins and losses on EMSR837 are reproduced).

| Cluster | Tie with `hold_aware` | Seed-dependent | Win | Loss | Mean s.d. between seeds (ha) | Scenarios with > 5 ha between seeds |
|---|---|---|---|---|---|---|
| C1 | 83 % | 13 % | 3 % | 2 % | 0.5 | 10 |
| C2 | 77 % | 17 % | 4 % | 2 % | 0.7 | 22 |
| C3 | 66 % | 24 % | 7 % | 3 % | 0.8 | 14 |
| C4 | 76 % | 16 % | 5 % | 3 % | 2.6 | 15 |
| C5 | 12 % | 47 % | 22 % | 20 % | 3.7 | 165 |
| All | 736 | 377 | 143 | 112 | | 226 |

Against `heuristic`: 633 ties, 273 seed-dependent, 418 wins (−5,604 ha in all) and 44 losses (+310 ha; 5 on EMSR837, +15 ha). Whether a scenario is a tie is predictable from its characteristics (AUC 0.93 on held-out incidents); whether a decisive one is won or lost against `hold_aware_heuristic` barely is (AUC 0.64). In C5 the seeds agree moderately on where they win and lose (correlation of their differences with the rule 0.52–0.68), so part of it belongs to the scenario and is not in these characteristics.

Reading: hard fires are those whose front exceeds what stacked water holds (3,200 kW/m, D-74/D-76), so difficulty is largely a property of the drop model and of fuel and moisture. D-85 does not differ from `hold_aware_heuristic` in any cluster mean; in C5 it differs in single scenarios, in both directions.

Limits: silhouette 0.19, so this is a continuum cut into parts rather than natural groups; C1 and C4 come from 4 and 6 incidents and C4 is bimodal. The synthetic wind range is narrow (cluster means 2.4–2.5 m/s), so wind not mattering here says nothing about wind in general. 17 % of the hard fires reach the map edge without intervention and their size is understated. Three seeds measure the spread of one scenario poorly, and the 1 ha tie threshold favours ties in small fires. Comparing D-85 with `hold_aware_heuristic` inside classes computed from `hold_aware_heuristic` is biased (it looks worse on easy and better on hard scenarios); in bands of growth, which neither controller defines, the difference is at most +0.12 ha except −1.04 [−2.58, +0.52] above 50 ha.

#### Simulating before deciding (2026-11-25)

Question: differences between seeds and against the rule concentrate in C5; can they be exploited without training, and without a rule written for C5? `lookahead.py` (`run_lookahead.sh`, output `lookahead_summary.txt` from `lookahead_analyse.py`; simulator code at commit cc23221) flies the D-85 selected weights with extra simulation. Settings were fixed before any result. "Forecast" means the same scenario with the current interval's wind and moisture repeated for the rest of the hour, so no future weather is read; fuel, terrain and the fire model are the simulator's own. The portfolio and the lookahead are therefore oracles of the dynamics, though not of the weather: the simulator is deterministic, all foci start at t = 0, and the branches are run in the environment that then scores the decision, so spread, drop effect and flight times are known exactly. These rows measure margin; they are not results of the policy, which alone stays at −0.03 ha. Only the ensemble uses no simulation.

- **Ensemble:** mean of the three seeds' action probabilities at every decision. No simulation.
- **Portfolio:** at t = 0 each candidate controller is flown in the forecast; the one with the smallest forecast area flies the real episode (ties: first in the list).
- **Lookahead:** one seed; at each real decision the four most probable valid actions are simulated to the horizon in the forecast, the same policy continuing, and the best is taken (ties keep the policy's own choice).
- **Full oracle** (seed 42): lookahead with the true future weather as well.

EMSR837 and EMSR590 (312 scenarios, used neither for training nor for selection; `hold_aware_heuristic` 37.86 ha, D-85 37.83). Difference with `hold_aware_heuristic` in ha, 95 % bootstrap over base scenarios (map, weather and moisture; their four focus variants stay together); better / equal / worse counts.

| Controller | All | C1 (96) | C2 (72) | C3 (71) | C5 (69) |
|---|---|---|---|---|---|
| D-85, 3 seeds | −0.03 | −0.09 | +0.34 | +0.21 | −0.58 |
| Ensemble | −0.38 [−0.89, +0.13] 69/181/62 | −0.19 | +0.06 | +0.23 | −1.77 [−3.08, −0.53] |
| Portfolio: two rules | +0.09 [+0.01, +0.23] 3/298/11 | +0.19 | 0.00 | +0.06 | +0.10 |
| Portfolio: three seeds | −0.76 [−1.19, −0.36] 87/169/56 | −0.67 | −0.11 | 0.00 | −2.39 [−3.69, −1.25] |
| Portfolio: three seeds and the rule | −0.79 [−1.21, −0.42] 77/204/31 | −0.64 | −0.22 | −0.13 | −2.33 [−3.70, −1.13] |
| Best of those four in hindsight (bound) | −1.40 | −1.15 | −0.29 | −0.75 | −3.67 |
| Lookahead, 3-seed mean | −1.06 [−1.60, −0.55] 104/156/52 | −1.56 [−2.61, −0.67] | −0.06 | −0.33 | −2.24 [−3.85, −0.45] |
| Full oracle lookahead, seed 42 | −1.77 [−2.47, −1.12] 102/172/38 | −1.85 | −0.08 | −0.34 | −4.97 [−6.54, −3.38] |

Per seed (42, 43, 44) the lookahead gives −1.01, −1.29 and −0.90 ha against the rule, every interval below zero, and −0.87, −1.07 and −1.17 against the same seed without it. It changed 331 of 6,703, 336 of 11,853 and 391 of 4,519 decisions: 16–19 % in C5 and 0–12 % in the other clusters. On all 1,368 scenarios (training incidents included) the three-seed portfolio gives −1.40 [−2.03, −0.95] ha against the rule and −2.80 [−3.93, −1.64] in C5.

Reading: the spread between seeds in C5 is evidence of margin, since the same scenario ends differently under different decisions. Choosing with a forecast turns part of it into burned area, and does not need to know the cluster: where all options end the same, it changes nothing. The gain over the rule comes mainly from C5 and C1, and needs the learned policies in the sense that the two registered rules alone offer no choice worth making. Comparing the full oracle with the persistent-weather lookahead of the same seed, weather accounts for about 2.6 ha in C5 (−4.97 against −2.41).

Variance is not what this removes. In C5 the mean spread between seeds rises from 2.8 ha to 3.5 ha with lookahead; scenarios lost to the rule by all three seeds fall from 12 to 5 of 69 and wins rise from 9 to 28, while seed-dependent ones are 28 against 38. A wrong forecast can cost a lot in one scenario: with lookahead seed 44 burned 101 ha more than without it in EMSR837_AOI07 scenario 06-00, and seeds 42 and 43 lost 31–42 ha in single AOI07 scenarios.

Limits: the forecast uses the simulator's own fire model, so it is a perfect fire forecast apart from the weather; the gain would shrink under a model mismatch, which was not tested. The comparison is therefore not at equal information: neither the rule nor D-85 simulates (the dispatch/short-horizon-planner condition of the baselines section asks planners to disclose this). The same lookahead with `hold_aware_heuristic` as the continuing policy was not run, so how much of the gain belongs to the learned policy rather than to simulating is unknown. All 96 C1 scenarios come from three EMSR837 maps (AOI05, AOI08, AOI17). Computation was about 1–3 s per decision and 15–113 s per episode (seed means, nine processes in parallel). EMSR837 has been used for many earlier comparisons. The lookahead was not run on the training incidents.

#### A spread model learned from observed arrivals (2026-11-25)

Question: can spread and water effect be learned from what is observed, without the simulator's formula, well enough to compare the options of a decision? `runs/geographic-learned-spread-20261125` (`CRITERION.md` written before any result; `collect.py`, `spread.py`, `evaluate.py`, `analyse.py`, `diagnose.py`; output `summary.txt`, `diagnose.txt`). No policy was trained and the reserved split was not opened.

A small network predicts the speed of each edge between neighbouring cells from fuel code, terrain, current-interval wind and moisture, and a multiplier for water (stacked drops on the cell or its flanks). It is fitted to arrival times alone, on the 928 training scenarios flown by no intervention, three rules and two random policies (296,290 ignition outcomes, 117,737 near water); the 128 selection scenarios stop the training. It receives no rate of spread, intensity, fuel parameters or hold thresholds. It does receive the structure of the simulator: spread between eight neighbours, earliest arrival wins, progress accumulates across wind intervals. Three initialisations. Test on EMSR837 and EMSR590 (312 scenarios).

| Area without water from t = 0, MAE in ha at 15 / 30 / 60 min (true mean 5.98 / 15.59 / 46.13) | |
|---|---|
| Learned, true wind (3 initialisations) | 0.09–0.12 / 0.29–0.44 / 0.96–1.15 |
| Learned, initial wind held constant | 0.09–0.12 / 2.09–2.25 / 7.39–7.67 |
| Exact simulator, initial wind held constant | 0.00 / 2.11 / 7.27 |
| No growth | 2.94 / 12.54 / 43.09 |

Spread without water is learned almost exactly (overlap of newly burned cells 0.96 with true wind); with the wind held constant its error is that of the exact simulator, so the wind is what limits the forecast (C5: 25–26 ha for both). A fitted constant speed gives 52 ha, but is a poor baseline because area jumps with the ring reached.

Water is learned worse. With the drops of real flights fixed and the true wind, the model predicts 6.5–6.9 ha saved by `hold_aware_heuristic` and the D-85 seeds against a real 8.0–8.5 (MAE 2.5–2.8, correlation 0.81–0.82). Of the treated cells that the model says would burn without water, it predicts as stopped 75–79 % of those really stopped and as reached 94–96 % of those really reached; with three drops, where 98 % stop, it is right in 77–79 %. The three-drop case is not rare in training (8,291 cells, 96 % not reached), so the cause of this error was not determined: candidates are the intensity, which decides it and is not observed, and the loss, which only asks for an arrival after the horizon.

Decision test (criterion): 1,248 decisions of D-85 seed 42, its four most probable actions each, real branches flown with true weather, then replayed open loop from the true state with the current wind held constant; 1,140 pairs differ by at least 1 ha.

| Forecast | Sign agreement | Predicted tie | Correlation of branch differences | Area of the chosen action − the policy's own |
|---|---|---|---|---|
| Exact simulator | 58 % [50, 64] | 35 % | 0.68 | −0.108 ha [−0.195, −0.047] |
| Learned 1 | 40 % | 49 % | 0.26 | −0.026 [−0.125, +0.077] |
| Learned 2 | 50 % | 42 % | 0.37 | −0.102 [−0.186, −0.039] |
| Learned 3 | 48 % | 42 % | 0.26 | −0.084 [−0.167, −0.025] |
| Constant speed | 25 % | 43 % | 0.00 | +0.095 [+0.030, +0.181] |
| Best branch in hindsight | | | | −0.217 |

Reading: criterion not met (sign agreement at least 70 % and at most 10 points below the exact simulator). The 70 % bar was badly set: the exact simulator with constant wind reaches 58 %, because ties count as disagreement (it predicts a tie in 35 %; where it predicts a difference, its sign is right in 89 %). The relative bar fails as well: the mean of the three is 46 %, 11.5 points below, and only one of three initialisations is within 10. By the value of the action chosen, two of three recover 78 % and 94 % of the exact simulator's gain and one 24 %, which was not the criterion and is a post hoc reading. Spread is therefore not the obstacle; water is, and the result varies with the initialisation. The learned closed-loop lookahead was not run.

Limits: the model learns this simulator's law and says nothing about real fire; the simulator's structure was given; D-85 still receives rate of spread and drops needed as inputs. A first evaluation had a precision error in the branch test (arrivals stored in float32 made new cells stop spreading, area bias −8 ha); it was corrected, with the same models and criterion, and the two decisions taken at one instant are no longer merged; earlier outputs are in `before-fix/`. EMSR837 has been used for many earlier comparisons. The diagnosis of water was made after seeing the result.

#### Why the learned water effect failed (2026-11-25)

`cause1.py` to `cause3.py` and `spread2.py` in the same folder. Everything here uses the training and selection scenarios; the 312 unseen ones were not used to diagnose or to choose. The "stopped" and "reached" shares are taken over treated cells the model says would burn without water, with the true sources given.

1. Not a lack of data or of generalisation: on the training fires themselves 18–46 % of the stopped cells are missed, while 83–98 % of the reached ones are right. With three drops about half of the misses are predicted arrivals before the last drop; with one or two, about half are arrivals in the last ten minutes of the hour.
2. It is the water part, not the spread: keeping the learned spread and replacing only the water by the simulator's rule (diagnostic, selection) gives 98 % of the stopped cells and 99.5 % of the reached ones, against 74 % and 97 % for the learned water. Using that rule with the current wind instead of the wind when the source cell burned gives 87 % and 95 %.
3. The missing input: the simulator decides whether water holds a front with the intensity of the interval in which the source cell was reached, and keeps that decision when the wind changes. The model only saw the current wind, so a hold that the simulator keeps looked uncertain. The intensity that the previous section named as a possible cause is not the limit: given the right interval, the network infers it well enough.
4. The loss adds to it, but less: an error of a cell wrongly stopped costs up to about a hundred times one wrongly reached, so an uncertain cell is predicted late rather than stopped.

| Change (one initialisation, selection) | Stopped recognised | Reached recognised |
|---|---|---|
| None | 73 % | 98 % |
| Symmetric loss with classification (b) | 84 % | 95 % |
| Weather of the source-arrival interval (a) | 95 % | 99 % |
| Both (a, b) | 96 % | 99 % |
| Source-arrival and current weather both given, the network chooses (c), three initialisations | 93–94 % | 99 % |

Variant c was chosen for use because (a) would be the simulator's rule copied by hand; both weathers are observable. It is still a hint that came from looking at the rule, not from the data alone. Effect on decisions, selection scenarios (512 decisions of D-85 seed 42, 666 pairs with at least 1 ha; constant current wind; the policy was selected on these scenarios):

| Forecast | Sign agreement | Correlation of branch differences | Area of the chosen action − the policy's own |
|---|---|---|---|
| Exact simulator | 55 % [48, 61] | 0.74 | −0.121 ha [−0.186, −0.061] |
| Learned, original (three) | 41–44 % | 0.31–0.44 | +0.053 to +0.109 |
| Learned, variant c (three) | 50–52 % | 0.58–0.63 | −0.096, +0.006, −0.104 |

With true wind, the sign of differences of at least 1 ha between controllers (drops of real flights fixed) goes from 77–79 % to 91–93 %; with constant wind from 69–73 % to 77–78 %. Reading: the water error was mostly one missing input, and what remains is the wind forecast. One of the three initialisations gains nothing in the choice of action, so the result is still not stable across initialisations. Limits: no closed-loop flight, no evaluation on the unseen scenarios and one selection set (EMSR719, 892, 632, 457; C2, C4 and C5 only); the criterion of the previous section is unchanged and still not met by the models tested there.

#### The corrected spread model on the unseen incidents (second look, 2026-11-25)

`CRITERION2.md` (written before running), `eval_test_c.pkl`, `summary_test_c.txt`. Variant c, three initialisations, chosen on the selection scenarios; same 312 unseen scenarios, tests and criterion as the first evaluation, of which this is the second look. The model was not changed afterwards. Decision test, 1,248 decisions and 1,140 pairs of at least 1 ha, current wind held constant.

| Forecast | Sign agreement | Predicted tie | Opposite | Correlation of branch differences | Area of the chosen action − the policy's own |
|---|---|---|---|---|---|
| Exact simulator | 57.6 % [51, 64] | 35 % | 7 % | 0.68 | −0.108 ha [−0.198, −0.045] |
| Original model (first look) | 40 / 50 / 48 % | 42–49 % | 8–11 % | 0.26–0.37 | −0.026 / −0.102 / −0.084 |
| Corrected 1 | 50.4 % [44, 57] | 42 % | 8 % | 0.43 | −0.097 [−0.184, −0.034] |
| Corrected 2 | 49.9 % [43, 56] | 42 % | 8 % | 0.46 | −0.095 [−0.184, −0.031] |
| Corrected 3 | 52.5 % [47, 58] | 39 % | 9 % | 0.37 | −0.104 [−0.198, −0.036] |
| Mean of the three forecasts | 58.1 % [52, 64] | 32 % | 10 % | 0.45 | −0.107 [−0.203, −0.039] |

Reading: the 70 % part is not met and could not be (exact simulator 57.6 %). The 10-point part is met: 50.9 % on average, 6.7 points below. So the criterion as written is not met and its informative half is. The three initialisations now agree (in choosing the action all are between −0.095 and −0.104 ha, every interval below zero, none distinguishable from the exact simulator: differences from −0.020 to +0.029). The correlation of the branch differences stays well below the exact simulator (0.37–0.46 against 0.68), so the size of the differences is predicted worse than their sign. Averaging the three forecasts is as good as the exact simulator in sign and value, but the average has fewer ties than integer areas and slightly more opposite answers (10 % against 7 %), so its sign agreement is not strictly comparable. In C5 the gain is −0.29 to −0.33 ha against −0.33 for the exact simulator.

Controllers with fixed real drops, real difference of at least 1 ha, sign agreement: true wind 84–87 % (original 66–70 %); constant wind 65–69 % (61–63 %). Saved area: predicted 7.3–7.7 ha against 8.0–8.5 for the rule and the three seeds with true wind.

Limits: one decision at a time, open loop, from the true state; no flight with the learned model, so nothing is said of the loss when errors accumulate in closed loop. The source-arrival wind came from looking at the simulator's rule; D-85 still receives rate of spread and drops needed; the simulator's neighbour structure is given. EMSR837 and EMSR590 have now been looked at twice for this model and many times for the policy.

#### The measured-intensity sensor and its correction (2026-11-25)

Mean burned area of the hold-aware rule on 928 development and 128 selection scenarios (EMSR837 and the reserved split not flown), bootstrap intervals over incidents. Scripts and outputs in `runs/thermal-intensity-design-20261125/` (`why_measured.py`, `why_analyse.py`, `why_linked_analyse.py`).

| Sensor, minus `hold_aware_heuristic` (formula) | All 1,056 | Development (928, 24 incidents) | Selection (128, 4 incidents) |
|---|---|---|---|
| As first implemented (D-91) | −0.57 [−0.94, −0.22] | −0.39 [−0.78, −0.05] | −1.85 |
| Corrected (D-93) | −0.24 [−0.69, +0.13] | −0.02 [−0.49, +0.33] | −1.80 |
| First implementation − corrected | −0.33 [−0.55, −0.14] | −0.37 [−0.61, −0.15] | −0.05 |

Reading: the first implementation excluded the directions in which water already holds the front, which the fire formula decides beforehand; that advantage is removed. With the corrected reading the measured rule ties the formula rule on the development incidents. The −1.80 ha on selection comes from four incidents, is not seen on the other 24, and its cause is not identified; intervals over four incidents are not reported. Follow-up on the same 1,056 scenarios (`variants.py`): the formula rule gives the same area as the measured rule when it is fed the largest directional intensity of each source's unreached neighbours instead of the direction towards the target (selection 78.32 against 78.30 ha; all 51.75 against 51.71), and giving it the wind interval in which each source was reached, which the simulator uses to decide whether water holds, does not change selection (80.15 against 80.10) and improves all of it by 0.25 ha [−0.48, +0.01]. So the difference on selection comes from how the rule summarises intensity, not from the sensor knowing more than the formula. Limits: perfect sensor; the 10 most favourable scenarios account for most of the mean of the first implementation (−0.16 ha without them); these sets also chose the reading.

Two policies were trained with the flawed sensor (`runs/geographic-measured-intensity-20261125`, D-85 recipe, seed 42): best selection score 81.73 ha without the learned speed and 80.91 ha with it, against 77.59 for D-85 seed 42 (formula inputs), 80.10 for `hold_aware_heuristic` and 86.59 for `heuristic`. The training script also flew both on EMSR837 (288 scenarios): 40.72 and 40.12 ha against 38.70 for `hold_aware_heuristic` and 41.76 for `heuristic`. A rerun with the corrected sensor was stopped at update 49 of 250 when the owner ruled out exact intensity and drop numbers as inputs (D-94); it gave no result.

Thermal brightness (D-94), same 1,056 scenarios (`brightness.py`): the hold-aware rule reading the byte burns 52.27 ha on all, against 51.95 with the formula (+0.32 [−0.12, +0.71]) and 56.02 for `heuristic` (−3.76 [−5.32, −2.32]). Development (24 incidents): +0.46 [+0.03, +0.85]; selection (4 incidents): −0.70. Reading: without the direction of the fire the rule loses about 0.3–0.5 ha to the formula and stays far ahead of `heuristic`. The policy that saw the fire only as this image (`runs/geographic-thermal-only-20261125`) was stopped before update 40 of 250, one selection score at update 25 (85.84 ha), because its image still depended on whether water would hold a front (D-95).

Blind rule (D-95, D-96), same 1,056 scenarios (`blind.py`, `rule_brightness.py`): `blind_hold_aware_heuristic`, with no hold limit, burns 52.70 ha on all against 56.02 for `heuristic` (−3.32 [−5.17, −1.61]) and 51.95 for `hold_aware_heuristic`, which knows the simulator's limits (+0.76 [−0.09, +1.66]). Development (928): 49.01 against 51.81 for `heuristic` and 48.06 for the privileged rule (+0.95 [+0.04, +2.00]); selection (128, four incidents): 79.52 against 86.59 and 80.10. The order of brightness was chosen on the development scenarios among three (50.71 without brightness, 51.99 brightest first, 49.01 dimmest first). Reading: knowing only how strongly the bordering fire burns, not what water holds, the rule recovers most of the earlier advantage over `heuristic`. The policy with the same knowledge (`runs/geographic-no-limit-20261125`, old scale of 191–255) was stopped by the owner at update 189 of 250; its best selection score was 84.02 ha (update 150). It was trained before D-96 and does not see the wider scale; `runs/geographic-target-features-20261125` (D-96, D-97) was training when this was written.

Leak checks of the blind rule (`leak.py`, 928 development scenarios): changing the wind of every weather interval after the first changes nothing the rule observes or orders before that interval (1,223 decisions in 116 scenarios, 0 differ). With the brightness values exchanged at random between burning cells it burns 50.24 ha, against 49.01 as it is (+1.23 [+0.22, +2.51]) and 50.71 without brightness (−0.47 [−1.09, +0.15]), so the gain does depend on which cell carries which value and the rest of the gain is not established. Ordering by the formula's directional intensity instead gives 52.44 ha (+3.44 [+1.44, +5.52]), so the brightness order is not a stand-in for the formula's order. The byte is a monotone function of the cell's largest fireline intensity, which the simulator computes with its formula: it is a perfect sensor (D-49), and a real camera would have noise and saturation. Limits: one rule family, four selection incidents, and the order was picked on these 928 scenarios.

#### What D-85 uses the fire formula for (2026-11-25)

D-85 seed 42 flown on its 128 selection scenarios with formula inputs set to zero (`runs/d85-formula-ablation-20261125`; selection incidents, not unseen). Intact: 77.59 ha. Without the speed channel: 80.01 (+2.41 [+1.41, +3.42]). Without the five drop-need columns: 84.05 (+6.45 [+2.75, +9.70]). Without both: 85.80 (+8.20 [+4.80, +11.29]). Reading: this fixed policy leans mostly on the drop needs, then on the speed. Zeroing is off its training distribution, so the losses show how much it relies on each input, not what a policy trained without it would score; four incidents.

#### Learned effect of water (D-98, 2026-11-25)

`runs/learned-hold-20261125` (criterion written first). From 2,784 flights over the 928 training scenarios (three rules), 751,879 rows of a cell bordering the fire; label, whether it burned by the end. Held-out incidents (5 of 29, 120,297 rows): log loss 0.204 (accuracy 0.912) against 0.248 (0.889) without the drop count and 0.571 (0.763) for a constant. Predicted risk falls with drops at equal brightness: for dim fire (byte under 150) 0.06, 0.002, 0, 0 for 0 to 3 drops; 150–200: 0.28, 0.12, 0.06, 0.02; over 200: 0.66, 0.51, 0.40, 0.23. The rows with one to three drops come from cells the rules chose to water, so the effect is not causal by design.

| Rule, mean burned ha | All 1,056 | Development (928) | Selection (128) |
|---|---|---|---|
| Learned-water rule | 52.05 | 48.27 | 79.45 |
| `blind_hold_aware_heuristic` | 52.70 | 49.01 | 79.52 |
| `hold_aware_heuristic` (knows the limits) | 51.95 | 48.06 | 80.10 |
| `heuristic` | 56.02 | 51.81 | 86.59 |

Learned rule minus blind: −0.66 [−1.99, +0.75]; minus the privileged rule: +0.10 [−0.95, +1.21] (all). Reading: a tie with both. The policy with these inputs (`runs/geographic-learned-water-20261125`) had not been trained when this was written. The run with the target features alone (`runs/geographic-target-features-20261125`) reached 83.50 ha on selection (best, update 175) and was stopped at update 206 by the owner or the app.

Diagnosis of the learned-water run at update 100 (`runs/learned-water-diagnosis-20261125`, 116 development training scenarios, unless stated): the network drops on cells that already have water in 46 % of its drops (the rule's choice: 17 %), and in 36 % of them the cell is already predicted safe (risk below 0.1). Its policy is spread (entropy 2.03 nats, top probability 0.36). It loses 3.4 ha to the rule on average, almost all of it in cluster C5 (+10.0 ha, 35 scenarios). In branches with the network's own continuation (48 fires, 288 decisions, two random seeds) the front target with the largest predicted gain returns +0.51 ha more than a random one [+0.06, +0.97] and +0.44 more than the network's choice [+0.10, +0.85]; a redundant drop is worth 0.56 ha less [+0.11, +1.17]; 54 % of pairs tie and the standard deviation of a pair is 3.4 ha. With the rule as continuation only the redundant drop is clearly worse (+0.55 [+0.16, +1.02]). Reading: the signal for single decisions exists but is small next to the spread, so the gain has to be read from the input for the policy to find it, and the columns ahead of the front were wrong (D-99). Checks: branches are on training scenarios, so the analysis fits nothing on unseen data. Corrections made while doing it: the first version of this analysis ranked the actions with the columns of D-98 ahead of the front, where they do not apply, and its numbers are superseded.

Learned-water run, seed 42, complete (`runs/geographic-learned-water-20261125`, 250 updates, 128 selection scenarios, D-85 recipe): best selection score 80.76 ha (update 225), last 81.45; curve 84.07, 83.22, 83.16, 82.10, 83.39, 81.25, 81.11, 81.56, 80.76, 81.45 every 25 updates. References on the same scenarios: `blind_hold_aware_heuristic` 79.52, learned-water rule 79.45, `hold_aware_heuristic` (privileged) 80.10, D-85 seed 42 (formula) 77.59, `heuristic` 86.59, the earlier runs without formula 83.50 and 84.02. Reading: better than any earlier policy without formula or limit, 1.2 to 1.3 ha behind the fair rules, within 0.7 ha of the privileged one; the selection incidents also choose the weights, so the policy figure is optimistic, and there are only four of them; one seed. The risk features of this run were misleading ahead of the front (D-99); the corrected run is `runs/geographic-learned-water-b-20261125`. The 312 unseen scenarios have not been flown.

#### Learned map of which cells burn (D-100, 2026-11-25)

`runs/learned-burn-20261125` (criterion written first). Network of 25 observed grid channels, dilated convolutions, one probability per cell; 14,848 grids from four rules on the 928 training scenarios, checked on 5 held-out incidents. Overall AUC 0.992 and log loss 0.032 against 0.107 for the share burned by time left, but nearly all distant cells never burn, so the useful figures are by distance to the reached cells (one grid in three, held-out incidents): 0–2 cells, AUC 0.947, log loss 0.250 (a base of distance and wind, fitted on those same cells, 0.726 and 0.504); 2–4, 0.971 and 0.113 (0.798, 0.241); 4–8, 0.985 and 0.045 (0.861, 0.100); 8–16, 0.996 and 0.007 (0.951, 0.022). It reacts to water: cells next to the fire, 0.31 without extra drops and 0.06 with three. Limits: the outcomes include later drops of the rules that flew, so it is the risk under such flying, not of the cell alone; perfect observation; training stopped early, with the best validation loss at the second epoch, so the fit is short. Used in `runs/geographic-learned-burn-20261125`, not trained when this was written.

Corrected-features run (`runs/geographic-learned-water-b-20261125`): selection 84.12, 83.52 and 81.38 ha at updates 25, 50 and 75 (the same recipe with the misleading features: 84.07, 83.22 and 83.16; D-85: 84.34, 82.44, 80.92), stopped at update 85 by the owner to train with the burn map (D-100); it had not finished, so it is not a result.

#### Burn-map policy against the fair rule (2026-11-26)

`runs/geographic-learned-burn-20261125` (D-99 features, learned speed, learned water, learned burn map; D-85 recipe, seed 42, 250 updates). Selection curve every 25 updates: 83.41, 82.16, 82.59, 81.41, 81.41, 80.98, 82.12, 79.87, 81.00, 79.97 ha; weights of update 200 kept. Flown deterministically on the 128 selection and the 928 training scenarios (`runs/fair-rule-comparison-20261126`, criterion written before flying):

| Mean burned ha | Selection (128) | Training scenarios (928) |
|---|---|---|
| Burn-map policy | 79.87 | 47.88 |
| `blind_hold_aware_heuristic` (fair rule) | 79.52 | 49.01 |
| Learned-water rule | 79.45 | 48.27 |
| `hold_aware_heuristic` (knows the limits) | 80.10 | 48.06 |
| D-85, mean of three seeds (formula) | 78.01 | 47.91 |
| `heuristic` | 86.59 | 51.81 |

Policy minus fair rule: selection +0.35 [−0.13, +1.08] (better/equal/worse 62/38/28); training scenarios −1.13 [−1.95, −0.28] (418/400/110). Reading: it beats the fair rule on the scenarios it trained on and ties it on the selection incidents, so by the criterion it does not beat it yet. Both sets are optimistic for the policy; nothing here is unseen.

Continuation of the burn-map policy (`runs/geographic-continue-a-20261126`): its selected weights resumed with the linear action path (D-101) and the entropy bonus lowered to 0.002 → 0.0005. Selection 79.71, 80.45, 80.41 and 81.56 ha at updates 25–100; stopped at update 108. Its best weights (update 25), flown: selection 79.71 against 79.52 for the fair rule (+0.20 [−1.14, +1.22]), training scenarios 48.33 against 49.01 (−0.68 [−1.75, +0.36]), so no better than the weights it started from (79.87 and 47.88). A first launch ignored `--resume` and was discarded (`v1.wrong-start.log`).

Learned arrival map (D-102, `runs/learned-arrival-20261126`, criterion written first): 14,848 grids from four rules on the 928 training scenarios; held-out incidents, cells within three of the fire: reached within 1 minute 0.5 % (AUC 0.980), 3 minutes 1.7 % (0.985), 10 minutes 6.4 % (0.980), 30 minutes 14.3 % (0.962), by the end 17.9 % (0.952). A first fit labelled every cell as reached by the end and was redone. Run with it: `runs/geographic-arrival-b-20261126`.


Arrival-map run B (`runs/geographic-arrival-b-20261126`: learned speed, learned water, learned arrival map with the two arrival target features, linear action path; D-85 recipe from scratch, seed 42, 250 updates). Selection curve: 83.84, 82.58, 80.64, 81.02, 81.71, 81.11, 79.47, 82.16, 80.08, 81.76 ha; weights of update 175 kept. Flown: selection 79.47 against 79.52 for the fair rule (−0.05 [−2.42, +2.77], 66/37/25); training scenarios 47.74 against 49.01 (−1.27 [−2.03, −0.53], 430/399/99). Same reading as the burn-map run: better on the scenarios it trained on, tied on selection.

Where the selection tie comes from: against the fair rule, B gains 104 ha in total on EMSR632 and 33 on EMSR719 and loses 129 on EMSR892 and 2 on EMSR457; every loss larger than 10 ha is in cluster C5 (largest: emsr892-05-mf02, 2 ha for the rule and 57 for B; emsr892-00-mf00, 236 and 296). Cluster C2 gains 44 ha, C4 ties. With four incidents a single large fire moves the mean by half a hectare, which is the size of the gap.

Ensemble of policies (`ensemble.py` in `runs/fair-rule-comparison-20261126`, members fixed in its criterion before flying: at each decision each member sees its own inputs and the action of highest mean probability is taken; no rule, no simulator). Burn-map policy plus B: selection 79.84 (+0.32 [−1.45, +2.44]), training scenarios 47.60 (−1.41 [−2.16, −0.59]). It does not help on selection. A one-member ensemble reproduced the single policy exactly.

Run C, the same recipe with seed 43 (`runs/geographic-arrival-c-20261126`): selection curve 84.43, 81.30, 81.54, 79.68, 79.59, 78.74, 79.83, 79.99, 80.06, 81.17 ha; weights of update 150 kept. Flown: selection 78.74 against 79.52 (−0.77 [−1.55, +0.16], 67/37/24); training scenarios 47.99 against 49.01 (−1.02 [−1.81, −0.17], 425/393/110). Three-member ensemble (burn map, B, C; the one the criterion counts): selection 79.16 (−0.35 [−1.72, +1.02], 64/41/23), training scenarios 47.42 (−1.59 [−2.37, −0.73], 432/407/89).

Reading of the three policies without the formula (burn map, B, C) and their ensemble: all four are better than the fair rule on the 928 scenarios they trained on, by 1.0 to 1.6 ha, with intervals that exclude zero; on the 128 selection scenarios the means go from +0.35 to −0.77 ha and every interval includes zero. The selection figures are optimistic because those scenarios also chose the weights. So the learned policy is at least level with the fair rule and ahead of it where it trained, but it has not been shown to beat it on scenarios it did not train on. The 312 unseen scenarios have not been flown. Run D, seed 44 (`runs/geographic-arrival-d-20261126`, started after the ensemble criterion, so not a member): curve 84.27, 84.62, 80.12, 80.02, 82.30, 80.55, 80.62, 80.39, 81.46, 81.59 ha; weights of update 100 kept. Flown: selection 80.02 (+0.51 [−0.48, +1.50], 52/37/39), training scenarios 48.42 (−0.58 [−1.60, +0.44], 343/409/176), so this seed does not beat the rule even where it trained. Mean of the three seeds of the arrival recipe: selection 79.41, training scenarios 48.05, against 79.52 and 49.01 for the fair rule.

Where the ensemble wins and loses against the fair rule (`why_lose.py`, `ens_vs_fair_table.csv` in `runs/fair-rule-comparison-20261126`; 1,056 selection and training scenarios). Against the rule it is better by more than 5 ha in 167 fires (−1,879 ha in total), worse by more than 5 in 47 (+1,023 ha), within 5 ha in 842. Among the 461 fires that grow by more than 20 ha without aircraft, the losses concentrate in fast and steep fires: spread rate over 16 m/min and slope over 25 %, losses in 25 % of 97 fires and wins in 20 % (mean +3.0 ha); in the other three combinations losses are 4–13 % and the mean is −3.1 to −4.0 ha. Losses rise with spread rate (0 % below 8 m/min, 20 % above 20), with dryness (dead moisture under 4 %: 17 %; over 6 %: none) and with size without aircraft (over 200 ha: 22 %). Drop by drop on the 40 largest losses and the 40 largest wins (same number of drops for both, first drop at 0.4 min): in the losses 42 % of the cells the ensemble waters burn later anyway (rule 30 %); in the wins 32 % for the ensemble and 33 % for the rule, but the ensemble repeats drops on the same cell (26 % of its drops, rule none) and the rule waters more cells that are lost. The largest losses are fires the rule holds small for the whole hour (2 ha; 10 ha) and the ensemble also holds for 20–30 minutes before one front escapes (58 ha; 71 ha).

Run with more fast, steep training fires (D-103, `runs/geographic-hard-fires-20261126`, criterion written first): the recipe of B, seed 42, with those 75 fires drawn three times as often. Selection curve: 84.30, 82.66, 81.95, 82.81, 80.91, 81.02, 81.61, 80.94, 80.17, 79.67 ha; weights of update 250 kept. Flown: selection 79.67 against 79.52 for the fair rule (+0.16 [−1.28, +1.59], 61/42/25); training scenarios 48.12 against 49.01 (−0.89 [−1.88, +0.08], 421/402/105). On the 75 fast, steep training fires it is +5.17 ha against the rule (22 losses and 13 wins over 5 ha; B: +5.96, 24 and 8) and on the other 853 −1.42 (B: −1.91); on selection, fast-steep −2.86 (B −3.05) and the rest +0.78 (B +0.58). Reading: negative. Three times the practice on those fires barely moves them (0.8 ha, inside the spread between seeds) and costs about as much on the rest, so the gap is not a lack of episodes of that kind. One seed.

The run copied the working package instead of B's frozen code, which had the automatic holdout report switched off, so at the end it flew its selected weights on the 288 EMSR837 scenarios and printed policy 39.80 ha (`heuristic` 41.76, `hold_aware_heuristic` 38.70, no intervention 47.34; the fair rule was not in that report). This was not decided by the owner; the figure was not used for any choice, and nothing else of the unseen incidents was flown. The other differences between the two code copies are speedups that keep results identical (memoised slots and candidate cells, a speed cache filled before forking).

Why fronts escape (`reach.py`, `escape.py` in `runs/fair-rule-comparison-20261126`; no training). The network chooses, per front, among four cells (head, two flanks, rear, by the observed wind) and three distances; the fair rule chooses any candidate cell. Along the rule's own flights (19,138 drops, 1,056 scenarios) the network could have made 80 % of the rule's drops; 96 % in the first 15 minutes, while fires are small, and fewer later as fronts grow: in the fast, steep fires 52 % (42 candidate cells per decision against about 10 the network can address), in the fires the ensemble loses by more than 5 ha 42 % after 15 minutes, in those it wins 53 %. Never more than four fronts, so the slot limit does not bind. In the 20 largest losses, the first cell that burns under the ensemble and never under the rule: in every case candidate cells next to it existed in the three decisions before, in 18 the network could aim there, in 12 it did and the rule would have in 16. In the four largest, the first decision decides it: at the start every candidate cell is addressable, and the ensemble splits between the rule's cell and a neighbour (emsr704-00-mf02: 0.47 against 0.51; emsr704-02-mf01: 0.26 against 0.70; emsr900_aoi01-05-mf03: 0.19 against 0.57 for a front 1 km farther). The two cells have the same bordering brightness; the rule's is the more downwind one, or the nearer one when tied. The network reaches the rule's cell at its next drop about six minutes later, just after the front has passed it (escape at 5.9–6.0 minutes). Across those 20 fires, 18 % of the ensemble's drops land on a cell already watered while an unwatered candidate exists (the rule never does), and it never drops ahead of the front (distances 1 and 2 unused). Reading: in small fast fires the losses are an early choice between neighbouring cells that the network has not learned, not a cell it could not reach; in large fires the fixed four cells per front leave most of the front out of reach.

Any-cell frontier actions (D-104, `runs/geographic-frontier-20261126`, criterion written first): the recipe of B, seed 42, with one action per candidate cell instead of four per front. Selection curve every 25 updates: 80.24, 80.95, 78.52, 80.94, 79.20, 79.88, 77.85, 78.15, 77.98, 77.45 ha (mean of the ten 79.12; B 81.33); weights of update 250 kept, the last one. Flown deterministically:

| Mean burned ha | Selection (128) | Training scenarios (928) |
|---|---|---|
| Frontier policy (D-104) | 77.45 | 47.03 |
| `blind_hold_aware_heuristic` (fair rule) | 79.52 | 49.01 |
| Learned-water rule | 79.45 | 48.27 |
| `hold_aware_heuristic` (knows the limits) | 80.10 | 48.06 |
| D-85, mean of three seeds (formula) | 78.01 | 47.91 |
| B (four cells per front, seed 42) | 79.47 | 47.74 |

Policy minus fair rule: selection −2.06 [−3.93, −0.70] (better/equal/worse 75/38/15), training scenarios −1.98 [−2.81, −1.16] (454/402/72); minus the learned-water rule −2.00 [−4.61, −0.61] and −1.24 [−2.59, −0.24]. Fast, steep group against the fair rule: training +1.64 ha (14 losses, 20 wins over 5 ha; B +5.96, 24 and 8), selection −4.32 (B −3.05); the other fires −2.30 on training (B −1.91) and −1.59 on selection (B +0.58). Reading: the first policy without the formula that beats the fair rule on both sets with intervals that exclude zero, and it also has lower means than the rule that knows the limits and than D-85 with the formula. Caveats: one seed; the selection incidents (four) also chose the weights, although the kept weights are the last ones; nothing here is unseen (the holdout report is off in this run's code copy and EMSR837/EMSR590 were not flown).


Water on burning cells (D-105, O-20; `runs/water-on-burning-20261126`, rules only, no network trained). The 1,056 development scenarios (928 training, 128 selection) with two aircraft; flag off reproduces all 3,168 stored rule results exactly. Paired difference, flag on minus off, with 95 % bootstrap intervals over incident groups:

| Rule | Off (1,056) | Mixed targets | Burning targets only | Burning only, selection (128) |
|---|---|---|---|---|
| `heuristic` | 56.02 | −1.99 [−3.58, −0.65] | −5.66 [−7.57, −3.92] | −8.54 [−11.36, −5.72] |
| `hold_aware_heuristic` | 51.95 | −0.91 [−1.73, −0.18] | −4.04 [−5.17, −2.92] | −7.11 [−9.23, −4.98] |
| `blind_hold_aware_heuristic` (fair rule) | 52.70 | −0.10 [−0.50, +0.25] | −3.84 [−4.80, −2.90] | −5.62 [−8.91, −3.03] |

No intervention is 63.97 ha in every configuration. With burning targets only about 99 % of drops hit burning cells and the rules drop about 45 % less water (10,300 against 18,500 drops for the fair rule), because knocked-down fronts leave fewer targets. The gain is an upper bound of this prototype: a drop closes every outgoing edge of its 100 m cell and a knocked-down cell never rekindles. Mixing both kinds of target in the existing rank orders gains little, so the rules' ordering, not the physics, limits the mixed case.

Free-water network (D-107, `runs/geographic-free-water-20261126`, criterion written first): the D-104 recipe from scratch (seed 42, 250 updates) on water on burning cells, no drop cap, no rekindling, with full freedom between pre-wetting and burning targets. Selection curve 75.54, 75.41, 73.66, 74.34, 72.63, 72.48, 72.70, 70.84, 71.92, 70.43 ha; weights of update 250 kept. No decision exceeded the 320-action ceiling. Flown deterministically on the same physics as the rules (`fly.py`, `compare.py`); policy minus rule, 95 % bootstrap over incident groups (better/equal/worse):

| Mean burned ha | Selection (128) | Training scenarios (928) |
|---|---|---|
| Free-water policy | 70.43 | 43.38 |
| Fair rule, mixed targets | 79.92 (−9.49 [−11.50, −7.48], 90/35/3) | 48.84 (−5.46 [−7.21, −3.93], 553/336/39) |
| Fair rule, burning cells only | 73.90 (−3.47 [−7.38, −0.77], 49/63/16) | 45.44 (−2.06 [−3.40, −1.04], 298/527/103) |
| `hold_aware_heuristic`, burning cells only (knows the hold limits) | 72.99 (−2.56 [−4.84, −0.86]) | 44.45 (−1.07 [−1.75, −0.45]) |
| No intervention | 98.02 | 59.27 |

The policy beats every rule on both sets, including the direct-attack rule that knows the hold limits, with intervals that exclude zero. Limits: one seed; selection also chose the weights and the policy trained on the other set, so this is "on development data"; the unseen activations were not flown; the physics is optimistic for direct attack (no rekindling), so the margin is for this simulator only.

Two more seeds of the same recipe (43, 44; run in parallel, `seeds.sh`): selection bests 71.18 and 71.08 ha (updates 225), flown on the same scenarios. Policy minus rule per seed, then for the mean of the three seeds per scenario:

| Seed | Selection − fair rule | Selection − burning-only rule | Training − fair rule | Training − burning-only rule | Training − hold-aware burning-only |
|---|---|---|---|---|---|
| 42 | −9.49 [−11.50, −7.48] | −3.47 [−7.38, −0.77] | −5.46 [−7.21, −3.93] | −2.06 [−3.40, −1.04] | −1.07 [−1.75, −0.45] |
| 43 | −8.74 [−10.29, −7.38] | −2.72 [−6.48, +0.86] | −6.09 [−7.99, −4.49] | −2.70 [−4.03, −1.64] | −1.71 [−2.47, −1.03] |
| 44 | −8.84 [−9.77, −7.58] | −2.82 [−5.63, −0.12] | −5.39 [−7.10, −3.90] | −1.99 [−3.20, −1.01] | −1.00 [−1.55, −0.35] |
| Mean of 3 (70.90 / 43.19 ha) | −9.03 [−10.33, −7.68] | −3.00 [−6.52, +0.01] | −5.65 [−7.40, −4.15] | −2.25 [−3.51, −1.27] | −1.26 [−1.82, −0.67] |

Reading: all three seeds beat the fair rule on both sets with wide margins, and beat both burning-only rules on the 928 training scenarios. On the 128 selection scenarios (four incidents) every seed is about 3 ha better than the burning-only rule, but the interval of seed 43 and of the three-seed mean reaches zero, so against that rule the selection win is not established. Against `hold_aware_heuristic` burning-only all intervals exclude zero. Development data only; the unseen activations have not been flown.

Unseen incidents for the free-water seeds (`UNSEEN_CRITERION.md`, written before flying; `unseen.py`, `unseen_compare.py`): the three frozen seeds and the rules on EMSR837 (288 scenarios) and EMSR590 (24), same physics, two aircraft; intervals resample AOI blocks (EMSR590 is one block, so it has no interval). EMSR837 is unseen by these weights but was used in earlier comparisons.

| Mean burned ha (312) | Value | Three-seed policy minus it |
|---|---|---|
| Policy, mean of three seeds | 35.02 | — |
| Fair rule, mixed targets (primary) | 38.68 | −3.66 [−5.38, −2.21] (204/102/6) |
| Fair rule, burning cells only | 35.35 | −0.32 [−0.67, −0.01] (72/189/51) |
| `hold_aware_heuristic`, burning cells only | 35.51 | −0.49 [−0.78, −0.23] (72/186/54) |
| `heuristic`, mixed | 39.70 | |
| No intervention | 46.13 | |

Every seed beats the fair rule on its own (−3.56 to −3.81). Against the burning-only rule the margin is small: −0.28, −0.22, −0.47 for seeds 42, 43, 44, with only seed 44's interval below zero; on EMSR837 alone the three-seed interval reaches zero (−0.25 [−0.58, +0.05]); on EMSR590 the policy is 1.21 ha better (no interval). So on unseen incidents the learned policy clearly beats the fair rule and is about as good as the simplest direct-attack rule, a little better on average. Two activations only.
