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

## Decisions requiring evidence or owner input before implementation/publication

| ID | Question | How to resolve |
|---|---|---|
| O-01 | Which geographic pilot area and incidents? | Choose based on data coverage, independent validation availability, and redistribution terms. |
| O-02 | Which aircraft class and published performance envelope? | Select a real class with sourced speed, load, flight and refill assumptions; define abstraction limits. |
| O-03 | Which fire-spread model and spatial/time resolution? | Compare credible models and convergence/sensitivity before policy training. |
| O-04 | What is the first controller's exact decision interval and action granularity? | Match aircraft dynamics, drop scale, and realistically available observations. |
| O-05 | Which data can be stored publicly? | Review each provider's terms and use manifests/scripts when redistribution is unsuitable. |
| O-06 | Which license and citation details? | Owner selects the license and citation details before GitHub publication. |
| O-07 | What constitutes a meaningful improvement? | Predeclare primary outcome, minimum relevant effect, baseline, and statistical protocol. |
| O-08 | Is a browser replay enough? | Prototype after Stage 0; choose desktop packaging only if concrete workflows require it. |

Open questions are tracked here so they are resolved deliberately rather than embedded as undocumented defaults.
