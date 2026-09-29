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

1. **Choose a pilot area and independent incidents.** [Partially completed under D-32]: Selected Copernicus EMSR578 AOI01 (Artesa de Segre wildfire, June 2022) as the calibration pilot. Sourced versioned vectors, established geodetic projection (ETRS89 / UTM Zone 31N, EPSG:25831), and audited burned-area snapshots. Independent held-out incidents remain required.
2. **Build a transparent fuel crosswalk.** [Completed for pilot incident under D-32]: `nightjar.pilot_emsr578` implements an explicit mapping from all 13 CORINE Land Cover classes in AOI01 to Scott & Burgan 40 fuel models with documented botanical rationales, confidence ratings, and alternatives. Produces an aligned metric fuel raster and scenario manifest. See [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md).
3. **Reconstruct environmental inputs.** Align elevation (DEM), moisture, wind, and ignition with the chosen perimeter times. [Initial baseline under D-33]: Used 97th percentile heatwave drought moisture and synoptic NNE wind vector ($1.97\text{ m/s}$ midflame). Time-resolved hourly weather and high-resolution DEM remain open.
4. **Check numerical behavior.** [Initial baseline under D-33]: `nightjar.pilot_simulation` implements 8-neighbor shortest arrival Dijkstra over 23,712 cells (100 m resolution) with Scott & Burgan fuels, running in under 80 ms. Alternative level-set methods and grid resolution sensitivity remain open.
5. **Evaluate unmitigated spread on held-out incidents.** [Initial baseline under D-33]: Compared simulated progression against Copernicus MONIT01. At active burning horizon ($T = 3,600\text{ s}$), reaches 2,071.0 ha (vs 2,108.0 ha observed, $-1.8\%$ error) with IoU 0.6630 and Dice 0.7973. Confirms unmitigated 24.7 h constant-wind overprediction due to real suppression and nighttime cooling. Testing on a second separate incident remains required.
6. **Integrate only after the spread path is interpretable.** Connect the selected fire engine to observations and the aircraft event loop, with explicit timing and no hidden-state leakage. Separately source or bracket water footprint, persistence, and effect; rerun no-intervention, random-valid, and heuristic decisions under the same information. Training remains gated by [TRAINING_READINESS.md](TRAINING_READINESS.md).

These are ordered tasks for the **physical-validation track**, not a claim that a real fuel map or validated simulator already exists. The **pilot data inventory, fuel crosswalk, and initial spread simulation** are now implemented in `nightjar.pilot_emsr578` (D-32) and `nightjar.pilot_simulation` (D-33). Next is environmental reconstruction (DEM, hourly weather) and held-out incident evaluation. Exploratory decision-interface work proceeds separately on the fixed synthetic FM1 episode described in [TRAINING_READINESS.md](TRAINING_READINESS.md).


