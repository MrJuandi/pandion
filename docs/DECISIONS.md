# Design decisions and open questions

## Agreed direction

| ID | Decision | Rationale |
|---|---|---|
| D-01 | Build a new project separately from the earlier FireLab-RL prototype. | Preserve lessons while avoiding unreviewed legacy code, results, and assets. |
| D-02 | Start with one aircraft, then two, then consider more resource types. | Each increment permits a clear experimental question. |
| D-03 | Target a decision system for multiple heterogeneous resources. | The first aircraft is a research stage, not the final architecture limit. |
| D-04 | Use obtainable spatial products and explicit complementary features as controller inputs. | Align information with possible real deployment. |
| D-05 | Separate latent simulator state from controller observations. | Prevent privileged information and evaluate the value of sensing. |
| D-06 | Begin with CNN + non-spatial encoder and actor–critic; test LSTM later. | Preserve a measurable feed-forward baseline. |
| D-07 | Keep execution dynamics separate from mission selection. | Allow one decision API to scale to additional resources. |
| D-08 | Use a headless scientific core; decide interface packaging after replay needs are known. | Avoid coupling experiments to a UI. |
| D-09 | Name the new project Nightjar. | Use a distinctive, memorable name that can cover the planned progression from one aircraft to multiple resources. |
| D-10 | Use the Rothermel surface-fire equation as the first candidate spread calculation and a CL-415 as the first aircraft reference. | Both have primary technical sources; their use and limitations are specified in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md). This does not settle fuel models, numerical propagation, ground speed, or suppression effectiveness. |
| D-11 | Start implementation with a synthetic, deterministic aircraft kinematics and water-accounting episode. | It tests time, mission state, and volume conservation without implying that the unsourced ground-speed and refill proxies are CL-415 performance data. |
| D-12 | Use a separate synthetic grid-arrival fire episode to test spatial propagation before integrating the aircraft. | Its explicit base rates and exponential directional factors verify interface behavior but are not Rothermel or a physically calibrated fire model. |
| D-13 | Implement the published one-size dead-fuel Rothermel/Albini equations first with Anderson FM1, then test grid sensitivity. | FM1 has a single dead fuel size and published parameters, allowing a narrow equation check before heterogeneous fuels; the grid coupling and transfer to real vegetation remain unvalidated. |
| D-14 | Audit Copernicus EMSR578 AOI01 as a candidate independent observation series without using it to calibrate or claim validation. | It supplies versioned, timed burned-area polygons, but its vegetation, interventions, and weather history are not represented by current FM1 simulations. |
| D-15 | Introduce explicit observation capture, delivery, coverage, and age before a controller interface. | The synthetic schedule exposes information leaks early while making no claim about real sensor accuracy. |
| D-16 | Preserve an unfitted Anderson FM1 comparison to CSIRO Annaburroo grass-plot rates as an external diagnostic. | It tests model transfer without selecting parameters on evaluation data. The measured 0.5264 m/s mean absolute error and differences in fuel/ignition context preclude a validation claim. |
| D-17 | Integrate the existing fire, aircraft, and observation components first with a permanent, centre-based 35 m water barrier in a synthetic example. | This explicit and reversible hypothesis checks event ordering, information flow, action execution, and comparative metrics. It is not a sourced deposition or suppression law. |
| D-18 | Add a separate Pyretechnics-based standard-fuel surface-fire path while retaining the FM1 prototype. | It supports published fuel-model codes and live/dead moisture without treating Annaburroo plots as a universal FM1 calibration target. The grid-arrival coupling and real vegetation mapping remain unvalidated. |
| D-19 | Use the existing integrated FM1 synthetic episode as a fixed provisional decision-research environment. Permit explicitly exploratory controller experiments before geographic physical validation. | The episode already exercises delayed observations, action execution, and matched baselines. More spread-model development would delay testing the decision interface. Results from this environment describe only its declared assumptions; physical validation remains required for geographic or operational claims. |
| D-20 | Freeze a small, versioned family of synthetic variants before testing learned controllers. Keep development and reserved scenarios explicit, and vary ignition, wind, aircraft start, and the provisional water radius. | The single original map can reward a policy that memorizes its geometry or exploits one water-effect assumption. A fixed family can expose this sensitivity while preserving the same observation and action contract. The reserved cases are a diagnostic challenge set, not independent geographic incidents or a physically validated test. |
| D-21 | Add a separate synthetic decision challenge with a longer horizon, masked fuel cells, and delayed reports while retaining FM1 spread, aircraft execution, and the 35 m water rule. Generate reserved cases with independent seeds and do not run their outcomes during development. | The first eight-variant diagnostic leaves the heuristic at the same area for each water radius. Development cases can expose action-choice failures; unopened reserved outcomes provide a cleaner future check within the same provisional simulator. |
| D-22 | Retire the exploratory linear scorer and its training-only scenario generator after the first reserved comparison. | Its four-case result did not establish improvement over observation-matched heuristic or seeded random choice. Preserve the outcome record and treat those cases as opened; focus next on discriminating scenarios and stronger baselines. |
| D-23 | Introduce the multi-foci, asymmetric-transit discrimination family (`nightjar-discrimination-family-0.1`) with non-trivial spread (>= 4.0 ha) and strictly disjoint development, validation, and unopened reserved splits. | Eliminates floor effects and forced single-front convergence, forcing spatial dispatch trade-offs between competing fronts and variable transit costs. Reserved scenarios remain untouched and unopened. |
| D-24 | Add an informed containment reference (`containment_heuristic`) and predeclare quantitative improvement criteria before evaluating future candidate models. | The greedy heuristic ignores downstream fuel exposure, multi-burning neighbor bottlenecks, and transit latency. Predeclaring a minimum relevant effect (>= 0.50 ha vs heuristic, non-inferiority vs containment) and paired dominance prevents post-hoc selection bias. |

## Decisions requiring evidence or owner input before implementation/publication

| ID | Question | How to resolve |
|---|---|---|
| O-01 | Which geographic pilot area and incidents? | EMSR578 AOI01 is an audited candidate, not a settled pilot. Choose calibration and separate test incidents based on source coverage, fuel/weather reconstruction, independent validation availability, and redistribution terms. |
| O-02 | How should the CL-415 reference figures become travel, pickup, and drop dynamics? | Specify airspeed-to-groundspeed assumptions, waterbody eligibility, timing, footprint, and effectiveness; validate against independent evidence. |
| O-03 | Which propagation scheme and geographic fuel crosswalk should use the standard catalogue? | Compare numerical schemes and convergence across varied fuels, wind, and slopes; establish a sourced vegetation-to-fuel mapping and validate against independent incidents before policy training. |
| O-04 | What is the first controller's exact decision interval and action granularity? | Match aircraft dynamics, drop scale, and realistically available observations. |
| O-05 | Which data can be stored publicly? | Review each provider's terms and use manifests/scripts when redistribution is unsuitable. |
| O-06 | Which license and citation details? | Owner selects the license and citation details before GitHub publication. |
| O-07 | What constitutes a meaningful improvement? | Predeclare primary outcome, minimum relevant effect, baseline, and statistical protocol. (Resolved for Stage 1 synthetic simulator under D-24). |
| O-08 | Is a browser replay enough? | Prototype after Stage 0; choose desktop packaging only if concrete workflows require it. |
| O-09 | Which synthetic cases distinguish learned decisions reliably? | Resolved under D-23: `nightjar-discrimination-family-0.1` implements multi-foci spread, edge water sources, non-trivial spread (>= 4.0 ha unmitigated), and 24 scenarios across disjoint splits. |
| O-10 | What would establish a meaningful learned-policy gain in this simulator? | Resolved under D-24: primary endpoint final affected area in ha, minimum effect >= 0.50 ha vs greedy heuristic, non-inferiority vs containment heuristic, paired dominance in >= 50% cases, and zero rejected orders. |

Open questions are tracked here so they are resolved deliberately rather than embedded as undocumented defaults.

