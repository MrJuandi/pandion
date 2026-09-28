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
