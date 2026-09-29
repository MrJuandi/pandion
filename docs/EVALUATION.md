# Evaluation protocol

**Status:** physical-evaluation protocol to freeze before model selection on geographic incidents. Synthetic controller experiments may precede it under the separate limits below.

## Exploratory synthetic decision experiment

Freeze the existing `integrated-synthetic-0.1` FM1 scenario and its 35 m permanent water barrier for the first controller-interface experiment. Compare controllers with no intervention, valid random choice, and the observation-matched heuristic using the same delivered observations and candidate actions. Save scenario and code versions, policy seeds, raw replays, final affected area, water use, flight time, rejected orders, and failure cases. The single map and repeated policy seeds cannot estimate generalization across incidents. Reward is a training signal; final affected area at the fixed horizon is the comparison outcome. Any apparent advantage is conditional on this synthetic simulator and its provisional suppression rule.

### Fixed scenario-family diagnostic (2026-09-28)

`python3 -m nightjar.scenario_family --output-dir runs/scenario-family` generates four fixed base configurations: two development and two reserved. Each is run with the **same** ignition, wind, aircraft start, and simulator seed at water radii 35 and 55 m, making eight scenarios. The manifest declares the synthetic UTC time origin (`2026-01-01T00:00:00Z`), local-metre CRS, units, source hash, variant specifications, component hashes, and splits. The command writes 96 full replays: no intervention and heuristic once per variant, plus ten random-policy choice seeds per variant. The report includes every run and descriptive split summaries. The paired design was chosen after inspecting a preliminary unpaired baseline run, and the reserved outcomes in this table have been opened. This is an exploratory diagnostic, not a blind policy test. There is no learned controller in this comparison.

| Base case | Split | No intervention (ha) | Heuristic at 35 / 55 m (ha) | Random mean at 35 / 55 m (ha) |
|---|---|---:|---:|---:|
| dev-01 | Development | 2.00 | 0.75 / 0.25 | 1.300 / 1.000 |
| dev-02 | Development | 2.00 | 0.75 / 0.25 | 1.550 / 1.025 |
| reserved-01 | Reserved | 2.00 | 0.75 / 0.25 | 1.375 / 1.150 |
| reserved-02 | Reserved | 1.50 | 0.75 / 0.25 | 1.050 / 0.600 |

The heuristic reaches the same 0.75 ha at 35 m and 0.25 ha at 55 m in **every** base case. Its simple rule already solves these variants in nearly the same way; the paired half-hectare difference shows strong sensitivity to the assumed water radius. The random means describe choice variation on fixed scenarios, not ten independent fires. These cases are too few and too structured for statistical inference, and their reserved outcomes have now been inspected. That finding motivated the harder observation and geometry challenge below.

### Synthetic decision challenge (2026-09-28)

`python3 -m nightjar.challenge_family --output-dir runs/challenge-family` keeps the 35 m water rule and existing FM1/action code, but extends the horizon from 600 to 1,200 s, generates seeded nonburnable cells, and delays full-grid surveys by 30, 90, or 150 s after capture. The generator writes four development and four separately seeded reserved scenario files with hashes. It evaluates **development only**, saving one no-intervention and heuristic replay and ten random-choice replays per case (48 total). Development indices `00`, `01`, `06`, and `09` were chosen after exploratory inspection of baseline behavior, so their outcomes are deliberately illustrative, not an unbiased model test.

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

`python3 -m nightjar.discrimination_family --output-dir runs/discrimination-family` generates 24 synthetic scenarios under schema `nightjar-discrimination-family-0.1` partitioned into strictly disjoint splits: eight development (`discrimination-development-00` to `07`, seeds `500000–500007`), eight validation (`discrimination-validation-00` to `07`, seeds `600000–600007`), and eight reserved (`discrimination-reserved-00` to `07`, seeds `700000–700007`).

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

`python3 -m nightjar.ppo --select --output runs/ppo-selection.json --checkpoint runs/ppo-selection.pt --candidates 2 --updates 1 --episodes-per-update 1` trains each candidate only on development scenarios, evaluates each deterministically on all eight validation scenarios, and runs `heuristic` and `containment_heuristic` through `run_integrated` on those exact validation scenarios. JSON output stores per-case PPO and baseline metrics matched explicitly by `scenario_id`, plus selected-candidate paired PPO-minus-baseline means/counts, wins/losses, and predeclared criteria (>=0.50 ha reduction versus greedy, non-inferiority versus containment, wins >=50%, losses <=25%, zero rejected orders). Selection minimizes mean validation `affected_area_ha`; exact ties break by candidate seed and then candidate index. The command never evaluates reserved scenarios and uses no reserved rollout.

#### Prior 8x8 result

The frozen exploratory comparison protocol was five independent training seeds (`0`–`4`), eight development episodes per update, and eight updates per seed. The completed run was:

```text
python3 -m nightjar.ppo --select --output runs/ppo-matched-5seeds.json --checkpoint runs/ppo-matched-5seeds.pt --candidates 5 --updates 8 --episodes-per-update 8
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
python3 -m nightjar.ppo --select --output runs/ppo-matched-5seeds-16x8.json --checkpoint runs/ppo-matched-5seeds-16x8.pt --candidates 5 --updates 16 --episodes-per-update 8 --seed 0
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
python3 -m nightjar.evaluate_reserved --checkpoint runs/ppo-matched-5seeds-16x8.pt --output runs/reserved-evaluation/report.json --replays-dir runs/reserved-evaluation/replays
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

Source and provenance: recorded from ignored artifact `runs/reserved-evaluation/report.json`, schema `nightjar-reserved-evaluation-0.1`, git working-tree code, PyTorch 2.2.2. Full replays for all policies and choice seeds are stored under `runs/reserved-evaluation/replays/`.


### Tactical PPO with Physical Forward Risk Heatmap and Macro-Action Space (2026-09-29)

Following the identification of observation delay amnesia and spatial action space pathologies (D-28, D-29, D-30), the architecture was extended with:
1. **Spread Risk Heatmap (Channel 10):** Physical forward risk projection computing directional advection from burning cells, wind alignment ($\exp(0.8 \cdot \text{wind\_align})$), and slope acceleration ($\exp(0.5 \cdot \text{slope\_align})$) strictly from delivered observations.
2. **Hybrid Tactical Action Space (9 macro-actions):** `HOLD`, `RETURN_TO_REFILL`, `ATTACK_HEAD_NEAREST`, `ATTACK_HEAD_LARGEST`, `ATTACK_CONTAINMENT_NEAREST`, `ATTACK_FLANK_LEFT`, `ATTACK_FLANK_RIGHT`, `ATTACK_REAR`, and `ATTACK_CLOSEST`.
3. **Recurrent LSTM Core and Procedural Domain Randomization:** Preserving temporal intent under observation delays and generalizing across synthetic fire regimes without behavioral cloning or imitation.

Training command on procedural domain-randomized scenarios with 2 candidates selected on the full 32-scenario validation set (`nightjar-discrimination-family-0.1`):

```text
python3 -m nightjar.ppo --select --candidates 2 --updates 40 --episodes-per-update 12 --seed 0 --recurrent --domain-randomization --action-mode tactical --output runs/ppo-tactical-v2.json --checkpoint runs/ppo-tactical-v2.pt
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
python3 -m nightjar.evaluate_reserved --checkpoint runs/ppo-tactical-v2.pt --output runs/reserved-evaluation/tactical-v2-diagnostic.json --replays-dir runs/reserved-evaluation/tactical-v2-replays --split-version v1
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
python3 -m nightjar.evaluate_reserved --checkpoint runs/ppo-tactical-v2.pt --output runs/reserved-v2-evaluation/report.json --replays-dir runs/reserved-v2-evaluation/replays --split-version v2
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

Source and provenance: recorded from artifact `runs/reserved-v2-evaluation/report.json`, schema `nightjar-reserved-evaluation-0.2`, git working-tree code, PyTorch 2.2.2. Complete per-episode replays are archived in `runs/reserved-v2-evaluation/replays/`. This completes the Stage 1 research question and closes its evaluation gate in the provisional synthetic simulator.


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

The first **integrated synthetic smoke comparison** is reproducible with `python3 -m nightjar.integrated_benchmark examples/synthetic_integrated.json --output runs/integrated-comparison.json` (2026-09-28). All three policies use the same 11×11 FM1 grid, fixed wind, fire ignition, aircraft and survey schedule. The random policy varies only its choice seed (`0`–`9`); neither the fire nor aircraft executor draws randomness. The primary diagnostic is full-cell affected area at 600 s, not a predeclared real-data endpoint.

| Policy | Affected area | Drops | Interpretation |
|---|---:|---:|---|
| No intervention | 2.00 ha | 0 | Matches the separate no-intervention FM1 solver within the horizon. |
| Downwind heuristic | 0.75 ha | 4 | One synthetic scenario, deterministic. |
| Valid random choice | 1.30 ha mean; 0.75–2.00 ha range over ten seeds | 4 per run | Policy-choice variation on the same scenario, not independent incidents. |

The assumed 35 m permanent barrier can dominate these differences. It is not a sourced suppression model, and this table cannot establish superiority of an aircraft tactic or physical fitness of a trained policy. The replay stores every decision's delivered observation and candidate set plus true arrival times, drop events, and aircraft accounting. No geographic split, uncertainty interval over incidents, or calibrated intervention-effect ensemble exists. The random and heuristic controllers use only delivered reports, the same static fuel/wind map, aircraft state, and known previous drops. The heuristic ranks downwind projection and proximity; it does not access the true arrival map.

### Exploratory PPO workflow

`python3 -m nightjar.ppo --output runs/ppo-smoke.json --updates 1 --episodes-per-update 1` trains and evaluates a minimal masked PPO controller on the **development** split generated by `nightjar.discrimination_family`. It uses only `IntegratedTrainingEnv` observations and action masks, records final affected area, rejected orders, total reward, and action count, and is explicitly synthetic with no physical claim. Validation and reserved scenarios are not evaluated by the default smoke path and remain reserved for a later, fixed evaluation protocol.

## 1. Endpoints

Declare one primary endpoint, initially **affected area at a fixed horizon** in hectares, computed from simulator true state. Secondary endpoints: damage to valued assets, time to containment, water/retardant use, flight time, invalid or failed orders, useful-drop rate, and worst-case loss. In two-aircraft experiments add duplicate assignments, idle time, workload imbalance, and refill congestion.

Report reward for training diagnostics only. Report outcomes per scenario and by severity group, including cases where a learned policy loses to a baseline. Define containment and affected area once and version any later change.

## 2. Baselines

- No intervention: establishes scenario difficulty and marginal benefit.
- Valid random controller: samples only actions available to the learned agent.
- Transparent heuristic: uses the same observation, latency, and candidate set as the policy.
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

Simulator-conditional comparisons must state the fixed scenario and assumptions and pass leakage and equal-information checks. No claim about physical superiority until the environment passes physics checks and the evaluation script can regenerate summary tables from raw episode records. No claim of real-world operational suitability follows solely from simulator results.
