# Geographic heights and environmental forcing

**Implemented, 2026-10-01:** the derived `nightjar-geographic-environment-family-0.1` contains 24 scenario-0.2 cases with sourced heights. Geographic replay 0.3 records the physical forcing and each decision's causally delivered weather separately. The Spanish offline viewer exposes 3D terrain, altitude/slope colors, cell height/slope tooltips, UTC time, wind, station temperature/RH/rain and estimated dead-fuel moisture. This is an exploratory environment, not a validated incident reconstruction or operational tool.

## Corrected source identity

The prior registered ZIP bytes were real, but catalog names/times were wrong. A hash check alone did not establish geographic identity. The official AOI01 DEL maps and source GeoJSON/XML identify:

| Product | Actual selected location | Post-event acquisition UTC | CRS |
|---|---|---|---|
| EMSR578 | Artesa de Segre, Spain | 2022-06-17 10:06 | EPSG:25831 |
| EMSR590 | Ladrillar / Las Hurdes, Spain | 2022-07-14 10:48 | EPSG:25830 |
| EMSR592 | Landiras, France | 2022-07-17 10:26 | EPSG:25830 |
| EMSR600 | Massarosa, Italy | 2022-07-21 10:23 | EPSG:25832 |

EMSR592 AOI01 is not Sierra de Mijas; EMSR600 is not Vall d'Ebo; EMSR590 is not Casas de Miravete. The catalog now checks broad WGS84 location envelopes after digest verification. Massarosa is regenerated in zone 32, so all derived positions/rasters/hashes change. Its old zone-30 coordinates were outside the useful projection region. EMSR592 reported burned area is the sum of nine source feature areas (6,726.49 ha), not the first polygon (41.68 ha). Original products and ZIP digests are preserved. Stale families fail current catalog provenance checks and must be regenerated. Official map links and primary references are in [REFERENCES.md](REFERENCES.md).

## Heights and weather contract

`terrain_acquire` reads corrected verified splits, requests 40×40 samples at the tactical 100 m grid, and saves ignored ESRI ASCII crops plus source metadata. IGN/CNIG WCS supplies Spanish terrain from `Elevacion25830_25`; the request does not explicitly specify interpolation. Its acquisition epoch and source footprint are unknown. France/Italy use public Copernicus GLO-30 COGs reprojected with Rasterio 1.4.4 nearest-neighbor sampling. GLO-30 is a DSM including vegetation/buildings, not bare-earth terrain; its EGM2008 heights are a slope proxy. Main acquisition is 2011–2015, with editing/infilling and unknown crop-specific release/epoch. No missing values are synthesized. Crop hashes identify the sampled inputs; the remote full COG bytes/version are not pinned by those hashes. Raw data remain ignored by Git.

`environment_family` verifies every original split, cell size/origin/shape, source hash and CRS. Origins must align within 0.1 m. ASCII north-first rows are reversed once into the south-to-north simulator convention. Real heights enter `fire.elevation_m`, Pyretechnics slope/aspect and the policy's public elevation/slope channels. The source crop, original scenario hash and new canonical scenario hash are retained. Source resolution 25/30 m does not change the simulation's 100 m grid. Environmental timestamps participate in artifact hashes, so a newly generated family can have new hashes even with the same physical arrays.

Only development EMSR578 cases use the previously audited Meteocat/XEMA Baldomar `X6` series. The original pilot SHA256 must match the weather audit. Temperature, RH, precipitation, scalar 2 m wind speed and mean direction must have consistent complete validated half-hour intervals and expected units. The engine holds each interval's aggregate constant as retrospective forcing; it does not claim an observed instantaneous trajectory. For t0 10:06 UTC, intervals end at simulation t=1440, 3240 and 5040 s; the 3600 s episode uses only part of the third interval.

The wind-to vector is `[-speed*sin(from_degrees), -speed*cos(from_degrees)] × station_wind_factor`, with default factor 1.0. This combines separately measured scalar speed/direction as a proxy, not an exact vector mean or a calibrated midflame wind field. No wind-height/canopy adjustment has been established. Other incidents remain explicitly synthetic constant weather. Artesa's station wind replaces the original variants' synthetic wind; these variants do not become independent weather observations.

In `emc_lag` mode, station air temperature/RH determine Simard equilibrium moisture (temperature converted to Fahrenheit, RH percent). Dead 1 h, 10 h and 100 h moisture follows a simple first-order exponential relaxation over elapsed intervals, held piecewise for fire rates; dead herbaceous follows the 1 h lag. Initial values and live moisture stay modeled. Radiation, canopy/fuel microclimate, sorption hysteresis and rain wetting are absent. The default rejects intervals with precipitation rather than silently inventing wetting; the audited Artesa episode is dry. `--moisture-mode fixed` keeps original fuel moisture and records rain without a calibrated wetting response.

## Physics and observation separation

Directional Pyretechnics behavior is cached by forcing interval. For each grid edge, the harmonic mean of source/target rates gives its speed. Remaining travel distance is integrated across interval boundaries; using a single start-time speed for an edge crossing a weather change is avoided. Constant-weather cases keep the legacy propagation path. The existing wind limit and minimum burnable rate remain model assumptions; heterogeneous convergence and alternative numerical schemes remain untested.

The full retrospective forcing belongs to simulator state and replay audit only. Controllers/baselines receive public terrain and the latest interval whose `available_at_s` has passed. Aggregated values are delivered at interval end, with extra publication latency unknown and assumed zero. Before the first completed interval, the observation records `weather_status=not_yet_delivered` and a zero wind placeholder, not measured calm. The PPO vector has no dedicated weather-validity channel; dimensions remain unchanged. Future station aggregates, latent fire arrivals and the full forcing array never enter policy observations. An interval clipped by the horizon remains undelivered until its original end. Source station-to-landscape calibration and realistic reporting delays remain open.

The reward, tactical actions and provisional permanent five-cell water barriers are unchanged. `GeographicTrainingEnv` continues to replay action prefixes deterministically. The new fresh checkpoint uses public terrain and delivered wind through the existing channels; temperature/RH influence fire spread through estimated moisture rather than new direct network channels.

## Reproduction and local evidence

The exact acquisition/family/training/workspace commands are in [README.md](../README.md). They were run successfully with the four verified local ZIPs and source-bound pilot/weather artifacts. Output paths are ignored and must be new/empty when rebuilding.

| Tactical crop | Height range (m) | Source | Raw crop SHA256 |
|---|---:|---|---|
| EMSR578 | 290–1025 | IGN/CNIG 25 m terrain | `32616c5a5add1bedbdf8d3eae1d4c5db1da2cda42e90473f9a13f0326edcb1eb` |
| EMSR590 | 587–1408 | IGN/CNIG 25 m terrain | `17c43516bdb44536142a435f768f75e74114650b8ed3cdc238ca29158311961d` |
| EMSR592 | 58.696–95.783 | Copernicus GLO-30 DSM | `89414204bb0d9990782c182ec67a8bfd3052a4f1b53be5a6ca760cebdfdfc6ae` |
| EMSR600 | 1.979–455.177 | Copernicus GLO-30 DSM | `f9635dd6c3e41815cdd0d82dd526607340ea0fcd01ce92e4513bc5d7020224d7` |

The acquired environment family manifest SHA256 is `3c03697a4d25aab328f27ce6e465e9b788d574c61cb78d36736281a14545cdc7`; weather audit SHA256 is `a87db3582da6215cea8f31db3a5f0958e0b9e83fa3dd953ad83d2b00c1a07837`. Development/validation/reserved membership remains 8/8/8. Reserved/validation data were acquired and inspected as static maps, with no episodes or policy selection on either split.

A fresh recurrent geographic PPO run used seed 42, six updates, one episode per update and the eight development cases. Its first development deterministic rollout produced 1,356 ha final reached area (initial footprint 1,146 ha), 14 drops/refills, 85,918 L, 123.521 km and zero rejected orders in 29 steps. These are technical endpoint records from a changed environment and six training episodes, not evidence of policy improvement or physical/generalization performance. Initial cumulative burned cells remain the engine's ignition proxy. Older flat-terrain metrics are not comparable as a controlled policy improvement experiment.

Weights SHA256: `6d4120424e5b0081fc7c493cd6ab91fd3a8903a4ec04cb22c67883774811943d`. Training report SHA256: `548ca91037b22defc592efb16b6bdc973a7618f0876bea52d893726cf8587299`. The report binds dataset hashes; the bare weights/report pair remains caller-associated and has no optimizer or RNG state.

Checks run on 2026-10-01:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
NIGHTJAR_PYTHON=.venv/bin/python node tests/viewer_browser.cjs
```

Python: 216 tests passed, no skips. Browser: scenario/policy boundaries, rectangular 2D/3D, altitude/slope colors, delivered-weather timing, distinct current/completed intervals, synthetic-weather blanks, comparisons, static controls, orbit/playback and 390 px overflow checks passed. The browser command requires resolvable Playwright/Chromium; local verification used bundled Node 24.19.0 and Playwright Chromium 151.0.7922.34, and the same test passed with system Chrome. Actual workspace inspection at 1440 and 390 px reported zero page errors and zero horizontal overflow. Fixtures are deterministic and do not establish statistical policy performance. Missing/nonfinite/misaligned heights, variable gaps/units/quality, duplicate weather rows and premature delivery are rejected by focused tests.
