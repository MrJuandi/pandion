# Research replay workspace

The first interface is an offline analysis workspace over recorded episodes, implemented in `nightjar.viewer` and `nightjar/viewer_assets/`. Simulation, controller execution and evaluation stay headless; the CLI can record an explicit synthetic scenario with a supplied PPO checkpoint before opening its results. It does not close a physical-validation gate or establish policy performance.

## Reference review

Reviewed primary documentation on 2026-09-30 and 2026-10-01. The latter review revisited workstation organization and native browser controls for version `nightjar-viewer-0.4`. This is a focused interface review, not a systematic scientific comparison. The products were not installed or benchmarked. Web documentation can change; version identifiers below refer to the reviewed material rather than an independently verified latest release.

| Reference and evidence | Useful pattern | Application in Nightjar |
|---|---|---|
| [FARSITE within FlamMap 6, WFDSS documentation](https://wfdss.firenet.gov/help/Content/20-Incidents/22-Analysis/FARSITE.htm), page revised 2024-10-22. Hourly weather and burn periods produce spatial vectors, perimeters and rasters. [FlamMap 6.2 rate-of-spread help](https://owfflammaphelp62.firenet.gov/FileTypes/PU_OUT_Spread_Rate.htm) distinguishes multiple output calculations and units. | Identify exactly what each map quantity represents. | Explicit layers and units; label first arrival as reached area, never an active flame front. |
| [IFTDSS Map Studio Swipe](https://iftdss.firenet.gov/firenetHelp/help/pageHelp/content/10-mapstudio/functions/swipe.htm), page revised 2026-08-03. Compare visible layers using a horizontal or vertical divider. | Compare spatial outputs on consistent visual terms with visible layer controls. | Synchronized maps and layer toggles. A swipe overlay remains a possible later addition when map identity and alignment are verified. |
| [WIFIRE Firemap project poster](https://wifire.ucsd.edu/sites/default/files/pdf/Firemap-36x60_AGU2016.pdf), AGU 2016; [current SCIL FAQ](https://scil.ucsd.edu/firemap-faq). The poster describes repeatable model runs, ensembles and time-series animation; the FAQ lists sources and differing perimeter/detection update frequencies. | Combine temporal playback with source awareness. | Decision navigation, observation timestamps and age; source records remain inspectable. Ensembles, live feeds and uncertainty surfaces are outside this slice. |
| [Prometheus technical report](https://publications.gc.ca/collections/collection_2010/nrcan/Fo133-1-417-eng.pdf), Canadian Forest Service, 2010, Information Report NOR-X-417. Project scenarios organize inputs; map, statistics and component views divide analysis tasks. | Keep scenario identity, spatial analysis and statistics connected but distinct. | Separate Replay, Compare, Models and Records views connect spatial inspection, outcomes and model provenance. Scenario editing is deferred. |
| [Cell2FireW / Fire2a documentation](https://fire2a.github.io/docs/Cell2FireW/), unversioned website. Cell-based output examples distinguish fire scars, growth trees and risk metrics, with a QGIS toolbox for spatial workflows. | Inspect cell values and distinguish descriptive output from derived risk. | Row/column inspection and recorded fuel codes. No probability or risk field is fabricated from a single deterministic replay. |
| [Fire2a QGIS simulator](https://fire2a.github.io/docs/qgis-toolbox/algo_simulator.html) and [toolbox](https://fire2a.github.io/docs/qgis-toolbox/index.html), reviewed 2026-10-01, unversioned website. Inputs group landscape, ignitions and weather; optional configuration is separate. | Group parameters by purpose and defer advanced configuration. | Physical inputs are visible together; source hashes, full configuration and additional details live in disclosures. |
| [IFTDSS Map Studio](https://iftdss.firenet.gov/firenetHelp/help/pageHelp/content/10-mapstudio/mapstudioabout.htm) and [landscape comparison](https://iftdss.firenet.gov/firenetHelp/help/pageHelp/content/30-tasks/devtrtalts/trtmtaltscomparelndscp.htm), reviewed 2026-10-01. Mapping and model tasks are separate; comparison explicitly selects a pair. | Keep map controls adjacent to the map and comparisons in a dedicated workflow. | Persistent Replay telemetry/layers, separate Models, and a Compare map stage with an adjacent analytics rail. |
| [NIST FDS and Smokeview](https://pages.nist.gov/fds-smv/), unversioned project overview. The simulation code and output visualization have separate roles. | Visualization consumes recorded output without owning the physical model. | The portable browser workspace reads replays; the Python engines remain independent. Smoke/heat physics and 3D rendering are not imported. |

The earlier FireLab-RL prototype was inspected as a local layout reference, including its HTML/CSS and archived 1600 px screenshot on 2026-10-01. The reviewed working copy had uncommitted changes relative to `e268a6a`; that commit does not identify the reviewed UI exactly. Useful ideas were its dark compact toolbar, large map, adjacent telemetry inspector and bottom playback. No legacy code, assets, checkpoints or scientific results were transferred. Decorative 3D terrain, static meteorological captions and unrecorded physical quantities are unsuitable for this replay contract.

## Design rules

The workspace has four task views, with Replay as the default:

- **Replay:** scenario and controller selectors, one prominent map and bottom timeline, a persistent telemetry/conditions rail, layer controls and cell inspection. Narrow screens move the rail below the map.
- **Compare:** an explicit candidate/reference pair, two synchronized maps, retrospective area curves, and final-outcome differences.
- **Models:** supplied PPO architecture and dimensions alongside physical inputs and assumptions. Provenance, complete configuration and source fingerprints are expandable.
- **Records:** selected replay metadata, final outcomes, recorded decisions, targets and acceptance, with original JSON export.

A dark slate workstation, system sans-serif typography, thin separators and tabular numbers support inspection. Map colors remain semantic: terracotta for reached cells, teal for delivered detections, blue for intervention and aircraft markers, ochre for candidates. Shapes and labels accompany colors. Native selects and buttons have explicit appearance, height and focus styling for the in-app browser. Long scenario IDs cannot force adjacent text into a narrow column. Metadata and decision logs have their own view to keep the main map legible.

The Nightjar identity uses a single filled SVG bird silhouette beside the wordmark. It replaces the thin multi-path outline, which lost definition at toolbar size. The vector remains inline in the portable HTML, with an accessible name and no external asset dependency.

Models and Records own vertical scrolling inside the desktop workspace, keeping the top navigation available. Their focusable panels also accept keyboard scrolling. At 1100 px width and below, the document scrolls as a whole; the workspace must not clip the end of either page. Tables may scroll horizontally without trapping vertical page movement.

Comparison panels are both attached before their canvases are measured; otherwise adding the second panel can shrink the first canvas width without updating its height. Each canvas fits the available panel after padding while preserving equal horizontal/vertical cell size. Scale notes do not overlay comparison maps. Short desktop windows reduce map height to keep playback available; very short windows and narrow screens use page scrolling.

A retrospective truth layer reveals recorded first-arrival times. Disabling it removes truth-derived current area and arrival information from the Replay inspector. The delivered observation remains the latest eligible decision observation, with capture time and age distinct from the playback cursor. Playback never selects a future decision before the first recorded sample. Aircraft markers are recorded decision samples, without path interpolation or connecting flight lines.

Local `position_m` fields use cell-anchor proxies. A half-cell visual offset places glyphs within the corresponding cells; it changes no simulator coordinates and is not a measured drop center. Rows increase north, columns east. Rectangular grids preserve cell aspect ratio. Scale requires recorded resolution; missing CRS, time origin, provenance or scenario hash remains unavailable.

## Comparison semantics

Compare selects exactly two runs. The timeline spans the larger horizon and each replay clamps to its own horizon. Area curves count reached cells from recorded arrival rasters, multiply by the recorded cell area, and use a step representation. Missing arrival or resolution data produces no curve. These are retrospective reached-area curves, not active fronts, risk probabilities or controller observations.

Final outcomes are separate from values at the cursor. Deltas are candidate minus reference, and are available only for distinct runs with matching nonmissing scenario hash, horizon and grid geometry and finite recorded metrics. A positive area delta is more affected area; water use and flight time are resource quantities rather than policy rankings. Missing values remain unavailable. Other pairs may be inspected spatially with an explicit comparison limitation. The UI computes no averages, confidence intervals or statistical superiority claims; it does not replace the evaluation protocol.

## Model and execution boundary

`--demo` records the three existing deterministic baselines on `discrimination-development-00`, seed 500000. With `--checkpoint`, the CLI loads actual CPU tensors using PyTorch's weights-only loader, checks the declared configuration against tensor dimensions and the current action mapping, and records one deterministic PPO episode on the same case. No training, validation or reserved evaluation is performed. `--scenario` instead runs the same comparison on an explicitly supplied `integrated-synthetic-0.1` scenario; the caller is responsible for choosing an appropriate development case.

`nightjar-viewer-model-0.1` descriptors carry actual architecture/configuration, checkpoint filename/SHA-256/byte count, parameter/tensor counts, available training metadata, creation time and current source-file fingerprints. Raw state dictionaries require explicit JSON `--model-config` with `grid_rows`, `grid_cols`, `recurrent` and `action_mode`; ambiguous grid or action semantics are never inferred. Configured dimensions must match the tensors, and nonfinite or incompatible weights are rejected. Source fingerprints identify the currently inspected implementation, not an undocumented training-code revision. Loading a checkpoint alone establishes neither its training provenance nor performance.

The controller receives only the existing delivered-observation tensorizer and masked action mapper. The browser receives descriptors and recorded replays, never checkpoint tensors or an execution API. A loaded descriptor is associated with a replay only through its recorded controller/checkpoint identity. The physical descriptor binds its actual scenario hash, grid, wind, moisture, fuels, aircraft proxies, observation schedule and provisional water-effect rule. This supports inspection of the FM1 synthetic engine; it does not introduce validated physics or a geographic physical descriptor.

## Supported evidence and limits

Synthetic integrated replays contain first arrivals, timed treatments, delivered `fire.cells`, local aircraft samples and final metrics. Current `nightjar-geographic-replay-0.1` files omit arrival rasters and several provenance fields. They display available observations/effects with explicit missing truth; the UI does not infer geographic extent, imagery, incident time or CRS from a scenario name.

JSON imports remain local to the browser and append replay/model records. Invalid batches are rejected atomically. Exporting JSON retains recorded data. PNG export is a visualization, not a source artifact. File fields render as text, never executable markup. The HTML bundles styles, scripts, replays and model descriptors without remote fonts, tiles or services. Baseline-only generation requires the standard library; checkpoint inspection/execution requires the optional RL dependencies.

Interactive scenario editing, browser execution, training management, validated geographic overlays, uncertainty ensembles and desktop packaging remain future decisions. Geographic truth maps need a versioned arrival/provenance contract in the producer.

## Verification record

Checked on 2026-10-01 with Python 3.12.8, Node.js, PyTorch 2.14.0 and Pyretechnics 2026.8.10. Commands and results are recorded below. Development checks use fixed inputs and seeds; they establish interface behavior, not physical validity, statistical policy quality or exhaustive browser/accessibility conformance. Missing external data tests remain explicitly skipped.

| Command/check | Result |
|---|---|
| `python3 -m nightjar.viewer --demo --output runs/viewer.html` | Three baselines plus a scenario-bound physical descriptor; development case 00 only. |
| `python3 -m nightjar.viewer --scenario examples/synthetic_integrated.json --output runs/scenario-viewer.html` | Three baselines and actual physical inputs from the supplied example. |
| `python3 -m nightjar.viewer runs/integrated-heuristic.json --output runs/heuristic-viewer.html` | Single-replay offline workspace; no policy execution. |
| `python3 -m nightjar.viewer --demo --checkpoint runs/ppo-tactical-v2.pt --output runs/viewer.html` in the RL environment with an existing checkpoint | Actual recurrent tactical PPO rollout plus three baselines, two model descriptors, shared scenario hash and matching controller/checkpoint identity. Checkpoint SHA-256 `8f64b9f2bc4d4061be0f2bee0c5273a768919d3f1cac71ab237239b070ea3c12`; 399,514 parameters, 20 tensors, selected seed 1. This is development inspection, not a new held-out result. |
| `python3 -m unittest discover -s tests -v` after final v0.4 integration, in the scientific environment | 198 tests: 194 passed, four skipped because the external Annaburroo table, CEMS ZIP and local pilot artifacts were absent. No failures. |
| `python3 -m unittest discover -s tests -p 'test_viewer*.py' -v` after v0.4 integration | 18 passed: finite/script-safe replay and model JSON, development-only CLI execution, actual weight/config checks, deterministic rollout and weight preservation, physical input/source bindings, time/geometry, paired/unpaired metrics, arrival steps, recorded drop counts, malformed import rejection and square-cell frame fitting for square/rectangular grids. |
| `node --check nightjar/viewer_assets/workspace.js` | Passed. |
| Earlier v0.3 Chromium automation | Four views, actual model identities, synchronized pair maps/curves, same-run and unrelated-pair delta rejection, missing geographic truth, layer switches, playback, decision/cell inspection, atomic replay/model imports, inert markup and JSON/model/pair-PNG exports. This did not establish native in-app-browser layout correctness. |
| v0.4 native in-app-browser visual and DOM checks | Replay, Compare, Models and Records; compact controls, persistent telemetry, model architecture and expandable provenance; 1440 × 900, 1440 × 700, 1024 × 768 and 390 × 844 layouts. Both comparison canvases retain square cells after panel insertion; an imported 2 × 4 geographic fixture retains its 2:1 ratio. Scenario/controller filtering, playback, truth-off values and same-run delta rejection passed. Desktop Models/Records scroll with wheel and keyboard to expanded provenance and the last decision; 390 px layouts scroll the document to the end. |
| `git diff --check` | Passed; generated artifacts and browser state remain under ignored `runs/`. |

Browser automation used a local HTTP server; its tool blocked direct `file:` navigation, so direct-file opening was not verified in this revision. The native in-app-browser export click produced no console error but its download capture did not return a file; saved exports in that browser remain unverified. The export notice consequently describes a download request, not confirmed disk persistence. Earlier Chromium export checks are retained separately. These checks are development QA, not exhaustive browser compatibility, formal accessibility conformance or behavior on the absent external data. Deterministic cases use fixed seeds in the same CPU runtime; no cross-platform numerical identity is promised.

### Scrolling regression procedure

With the generated checkpoint demo served over local HTTP:

1. At 1440 × 900, open Models, focus the page and press End. Expand controller provenance and scroll down over either column. The last source-fingerprint disclosure must be reachable, with navigation still visible. Repeating with wheel movement must move the panel, not leave content clipped by the workspace. The checked expanded panel reached offset 832 px of 832 px.
2. Open Records and scroll down over the decision table. The final 19:00 decision must be fully visible; Home or upward wheel movement must return to the beginning. The checked page reached approximately 679 px of 679 px.
3. Repeat at 390 × 844. The document must scroll vertically, without a constrained desktop panel. The checked Records page reached 1040 px of 1040 px and displayed the last decision. Tables retain horizontal scrolling for columns outside the narrow viewport.

Offsets are evidence from this demo and browser, not fixed layout requirements. The invariant is access to the final content using ordinary scrolling.
