# Evaluation protocol

**Status:** protocol to freeze before the first model-selection experiment.

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

No statement of superiority until the environment passes physics and leakage checks, the baseline comparison uses equal information, and the evaluation script can regenerate summary tables from raw episode records. No claim of real-world operational suitability follows solely from simulator results.
