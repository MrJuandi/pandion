# Data, provenance, physical calibration, and observation validity

**Status:** design checklist. Candidate sources below have different scales and licenses; inclusion requires a source-specific review.

## 1. Dataset contract

Each input layer and derived product must record: provider, product/version, acquisition or validity time, download time, spatial and temporal resolution, CRS, geographic coverage, units, processing steps, missing-data mask, uncertainty representation, license/redistribution terms, checksum, and whether it is observed, inferred, or synthetic. Scenario manifests must preserve these fields and the exact train/validation/test assignment.

No data layer is promoted from “inferred” to “observed” because it is displayed on a real map. Fuel models derived from land cover require a documented crosswalk and uncertainty; land cover is not itself a measured fuel load.

## 2. Candidate sources for Spanish scenarios

| Need | Candidate | Appropriate use and limitation |
|---|---|---|
| Elevation | [IGN/PNOA terrain products](https://pnoa.ign.es/pnoa-lidar/productos-a-descarga) | Static terrain input at declared resolution. Verify coverage, vertical datum, and resampling. |
| Land cover / forest type | [Copernicus Forest Type](https://land.copernicus.eu/en/products/high-resolution-layer-forests-and-tree-cover/forest-type-2021-raster-10-m-100-m-europe-3-yearly) and [CLC+ Backbone](https://library.land.copernicus.eu/products/CLCplus_Backbone_2021_PUM_v1.html) | Spatial prior for fuel classification, with product date and uncertainty. Not a direct fire-behavior fuel measurement. |
| Background imagery | [Copernicus Sentinel-2](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html) | Multispectral context or derived vegetation products. Revisits occur over days, so imagery cannot be treated as a live five-minute fire map. Clouds and smoke can obscure observation. |
| Active-fire detection | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Delayed, sparse hotspots. Global near-real-time detections may be available within three hours of acquisition; they are not a continuously observed perimeter. |
| Weather observations | [AEMET OpenData](https://www.aemet.es/en/datos_abiertos/AEMET_OpenData) | Historical or current station/forecast products subject to availability, interpolation and valid timestamps. |
| Historical meteorological context | [ERA5 hourly data](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-data-single-levels-1940-present) | Reanalysis useful for historical experiments; its grid is much coarser than tactical fire cells and it is not an operational minute-by-minute measurement. |

Protected assets, roads, water access, airspace, aircraft performance and operational constraints need their own sourced datasets or clearly marked initial approximations. Licensing and redistribution must be reviewed before including files in a public repository.

## 3. Physical model validation

The sourced Rothermel equation and CL-415 reference figures selected for initial study are recorded in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md). Published component values do not validate the assembled simulator.

The executable `examples/synthetic_aircraft.json` has `provenance: "synthetic"` and local metre coordinates. Its ground speeds and refill duration are test inputs without empirical provenance. The v0.1 replay records a seed but uses no randomness. Passing its deterministic checks establishes accounting and time behavior only, not aircraft or fire realism.

`examples/synthetic_fire.json` likewise declares synthetic provenance and local coordinates. Its per-cell base spread rates and wind/slope coefficients are unsourced test values; no fuel model, moisture measurement, or observational fire perimeter is present. The replay stores a SHA-256 of the input scenario JSON for traceability. Directional-response tests are structural checks, not physical calibration. Actual spread-model validation, sensitivity to grid resolution, and held-out incident comparison remain open.

The FM1 example uses a **published stylized fuel model**, but its assignment to every cell, constant 6% dead moisture, flat terrain, 2 m/s midflame wind, ignition, and horizon are synthetic. FM1 load, depth, SAV, extinction moisture, and heat content come from Anderson (1982) and Andrews (2018), as recorded in [PHYSICAL_MODEL_V0.md](PHYSICAL_MODEL_V0.md). The implementation reproduces the approximate 4 mi/h reference in Andrews figure 16. A fixed-domain 100–5 m grid comparison is recorded there; it is a numerical sensitivity check, not calibration to an observed incident. Geographic fuel mapping, observed-weather conversion to midflame wind, independent perimeter validation, and convergence under heterogeneous conditions remain open.

An audited candidate observed series is documented in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md): three Copernicus EMSR578 AOI01 cumulative burned-area polygons with source ZIP checksums and image acquisition times. They are kept in ignored `data/raw/` and are not a physical-validation result. A seven-case synthetic grid sensitivity report now covers wind direction, moisture, and slope but not heterogeneous fuel or changing weather. The 25 m versus 5 m affected-area differences ranged from −3.94% to +19.05% of the 5 m area on 2026-09-27.

The [Annaburroo diagnostic](VALIDATION_ANNABURROO.md) compares the unchanged FM1 head-rate equation with 120 measured grass-plot averages after one declared missing-data exclusion. Its 0.5264 m/s mean absolute error is a transfer diagnostic, not independent incident validation. The source version, license, table hashes, selection, command, and per-plot output are recorded there.

A second no-fit check uses the same source's burn-period `TimeSinceIgn`, `CumDist.m`, and `ROS.ms` fields. Across 120 eligible fires and 576 timed observations, its mean absolute **final** head-distance error is 73.81 m and its mean per-fire period-rate MAE is 0.610 m/s. It reuses the prior diagnostic's plots and holds derived plot-average midflame wind constant, so it is neither an independent test set nor a 2-D perimeter validation. Four fires spread with reported 12.1% moisture despite FM1's 12% extinction threshold. See [VALIDATION_ANNABURROO.md](VALIDATION_ANNABURROO.md) for exclusions and limitations.

A further same-data sensitivity substitutes measured load, grass height, and species SAV into a one-class dead-fuel Rothermel calculation and varies a hypothetical ignition-growth timescale. At immediate steady spread, the measured-bed case reduces final-distance MAE from 70.85 to 56.74 m in 73 fully cured plots, but raises it from 78.41 to 144.97 m in 47 partly cured plots. The aggregate measured-bed MAE is 91.30 m. None is independent validation or a selected production parameter. [Full variant results and assumptions](VALIDATION_ANNABURROO.md).

The `CURED_GRASS` [example](../examples/cured_grass_fire.json) copies measured A023 plot load, height, dead moisture, derived midflame wind, and species SAV into a synthetic grid scenario. It explicitly labels full curing and retains an assumed 12% extinction moisture. A single measured plot does not supply spatial fuel variation, fire perimeters, or independent validation. The example tests configuration and solver plumbing only.

`examples/synthetic_observation_schedule.json` tests information timing and coverage on a synthetic arrival map. Its capture and delivery times are illustrative, not measurements of a real sensor. Tests confirm no change to the controller view from unobserved cells or undelivered reports. Real acquisition cadence, missed detections, and uncertainty distributions still need sourced definitions before training.

`examples/synthetic_integrated.json` combines the FM1 grid, one aircraft, and a repeated full-grid survey. Every spatial input, aircraft speed, refill time, survey time, and `water_effect.radius_m: 35` is synthetic. The 35 m permanent barrier blocks future arrivals at cell centres but leaves already reached cells burning; it has no empirical water footprint, dose, or persistence basis. Its only legitimate use is an integration and evaluation smoke test. A sensitivity range for water effect, independent spread data, measured aircraft cycles, and observed drop response are still required before interpreting policy outcomes physically. The integrated replay records what the controller received, its orders, and the hidden outcome separately.

Before policy training, test and document:

- Spread direction and rate response to wind, slope, fuel, and moisture, with units and plausible parameter ranges from identified fire-behavior literature or data.
- Sensitivity to spatial and temporal resolution, stochastic parameters, and boundary conditions.
- No-intervention fire evolution compared with independent observations at matching scale and timestamps. Report perimeter/area errors and uncertainty; do not calibrate and evaluate on the same incidents.
- Drop and refill mechanics: footprint, transit and service time, extinguishing or retardant effect, decay, and limits of evidence. If available evidence is insufficient, label the effect as a hypothesis and run sensitivity analyses.
- Resource movement and access: speed distributions, turn and approach constraints, refill location/cycle, and any intentionally omitted aircraft safety constraints.

An independently validated spread model is a prerequisite for strong claims about learned suppression. A policy should also be stress-tested across plausible parameter ensembles, since a single calibrated parameter set can reward simulator-specific behavior.

## 4. Observation validity and leakage audit

For every channel, specify when and by whom it becomes known. Test that changing latent fire or fuel **outside observed coverage** does not immediately change the controller input. Test that future weather, final burned area, and intervention outcomes are unavailable before their timestamps. The candidate-action generator, heuristic baselines, renderer exports, and normalization statistics are part of the leakage audit.

Observation latency is a parameterized distribution or schedule, not an annotation shown only in the UI. Missing data and sensor failures must be represented in the observation and included in evaluation.
