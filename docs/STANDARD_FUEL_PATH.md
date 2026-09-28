# Standard fuel-model fire path

**Status (2026-09-28):** executable synthetic example, not an incident reconstruction or an aircraft episode. This path is an alternative to the existing Anderson FM1 prototype. It does not supersede the evidence requirements in [TRAINING_READINESS.md](TRAINING_READINESS.md).

## What is implemented

`nightjar.fire_standard` reads a `standard-fuel-0.1` JSON scenario and uses [Pyretechnics 2026.8.10](https://github.com/pyregence/pyretechnics) for local Rothermel surface-fire behavior. It accepts numbered standard fire-behavior fuel models from the [Scott and Burgan catalogue](https://research.fs.usda.gov/treesearch/9521), including dynamic models with live herbaceous fuel. The example assigns GR1 (101) and SH1 (141) by hand; `0` is Nightjar's nonburnable mask. Land-cover labels are **not** fuel-model numbers, and no automatic land-cover conversion is implemented.

For each cell, the input gives a fuel code and elevation. Six moisture fractions are spatially constant in this version: dead 1 h, dead 10 h, dead 100 h, dead herbaceous, live herbaceous, and live woody. The midflame wind vector is also constant, with east/north **wind-to** components in m/s. The code derives slope and downslope aspect from the elevation grid, converts wind to the units and upwind azimuth expected by Pyretechnics, and calculates directional rates. Nightjar then joins neighboring cell centers with half-edge travel times and an eight-neighbor shortest-arrival solver. A diagonal cannot cross between two blocked orthogonal cells. Every cell whose center is reached by the horizon contributes its full area. The replay records the scenario hash, dependency version, arrival times, and affected area.

The current scenario requires `provenance: "synthetic"` and `crs: "LOCAL_METERS"`. Rows increase northward and columns eastward. It has no clock date, geographic coordinates, temporal weather, fuel consumption, crown fire, spotting, suppression, uncertainty, or observation model. The FM1 aircraft episode still uses its own fire path; it does not consume this replay. The numerical grid coupling is Nightjar's choice, not Pyretechnics' level-set propagation algorithm. Neither the synthetic example nor the standard fuel catalogue validates fire behavior for a particular landscape.

## Reproduce the example

From the repository root, with Python 3.12:

```sh
mkdir -p runs
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-standard.txt
.venv/bin/python -m nightjar.fire_standard examples/standard_fuel_fire.json --output runs/standard-fuel-fire.json
.venv/bin/python -m unittest discover -s tests -q
```

With the pinned dependency and committed example, the output reaches **0.375 ha at 3,600 s** (six complete 25 m cells). This value is a regression reference for the synthetic input, not an observed-fire estimate. The 42 unit tests passed on 2026-09-28 with Pyretechnics 2026.8.10. The replay records the SHA-256 of the committed source scenario and the dependency version as `engine_version`.

## Next steps and evidence required

1. **Choose a pilot area and independent incidents.** Inventory available terrain, vegetation/fuel, weather, ignition, intervention history, and time-stamped perimeters. Record source version, date, CRS, resolution, license, and uncertainty for each layer. If the inputs cannot be matched at the relevant times and places, choose a different pilot rather than treating a perimeter alone as validation.
2. **Build a transparent fuel crosswalk.** Map the source vegetation categories to candidate Scott and Burgan fuel codes using documented evidence about fuel structure and loading. Preserve ambiguous or unknown cells as uncertain or masked. Review the crosswalk independently; do not infer a precise model from land cover alone. Produce an aligned fuel raster and a manifest of every transformation.
3. **Reconstruct environmental inputs.** Align elevation, moisture, wind, and ignition with the chosen perimeter times. State how observed or modeled wind is converted to midflame wind and how moisture is estimated for each class. Add explicit time and geographic coordinates to a new scenario contract; keep the current synthetic contract unchanged.
4. **Check numerical behavior.** Compare local rates with published cases or an independent implementation. Test heterogeneous fuels, wind reversal, slopes, nonburnable barriers, and multiple cell sizes. Compare Nightjar's shortest-arrival coupling with an alternative propagation scheme, such as Pyretechnics' level-set method, before choosing a grid resolution. Report area and perimeter sensitivity, not just a passing test suite.
5. **Evaluate unmitigated spread on held-out incidents.** Select calibration and test incidents before tuning. Compare predicted and observed perimeters at matching times; report area error, perimeter overlap/distance, uncertainty, and missing or active suppression. If suppression or ignition history is unknown, state what the comparison can and cannot establish.
6. **Integrate only after the spread path is interpretable.** Connect the selected fire engine to observations and the aircraft event loop, with explicit timing and no hidden-state leakage. Separately source or bracket water footprint, persistence, and effect; rerun no-intervention, random-valid, and heuristic decisions under the same information. Training remains gated by [TRAINING_READINESS.md](TRAINING_READINESS.md).

These are ordered tasks, not a claim that a real fuel map or validated simulator already exists. The immediate next deliverable is the **pilot data inventory and candidate fuel crosswalk**, with unknown classes and source limitations visible.
