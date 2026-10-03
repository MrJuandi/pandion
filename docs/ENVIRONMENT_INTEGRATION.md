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

## Pre-fire Sentinel-2 base imagery (D-50)

`imagery_acquire` gives each activation grid one true-colour base image for the planned image-based policy input (D-49). It reads only grid and CRS metadata from scenario files under `development/`, `validation/` and `reserved/`, and requires every variant of an activation to share one grid. The CEMS activation time comes from the public CEMS activation API. Candidate scenes are Sentinel-2 L2A items from the Earth Search STAC API (Element84, AWS Open Data) acquired between 75 and 14 days before activation. The 14-day margin is a conservative stand-in for the unknown ignition time; scenes after it are never searched.

Every candidate is screened with its own L2A scene classification (SCL) inside the crop; tile-wide `eo:cloud_cover` is only metadata. Obscured classes are no data, saturated/defective, cloud shadow, cloud medium/high probability, thin cirrus and snow/ice, buffered by 100 m because SCL misses diffuse cloud and shadow edges. Unclassified pixels are kept because they persist on bright bare ground in every scene. Scenes are ranked by obscured crop fraction and then recency. The image is a best-available-pixel composite: each pixel comes from the highest-ranked scene in which it is clear, using up to ten scenes; acquisition fails rather than leaving an obscured pixel. Each filler scene is matched to the primary scene by per-channel mean and standard deviation over pixels clear in both, which reduces visible seams but does not remove phenological differences between dates. The manifest lists every screened scene and the pixel fraction each contributing scene supplies, so a multi-date composite is never presented as one acquisition.

The `visual` (TCI) asset, 8-bit true colour at 10 m native sampling, is reprojected with Rasterio bilinear resampling to the scenario CRS at 8 pixels per 100 m cell (12.5 m/pixel); SCL uses nearest-neighbour. This is close to native resolution and adds no detail. Arrays are saved as `<activation>.npy` (uint8, `[rows·8, cols·8, 3]`, row 0 at the southern edge like the simulator grid) with a north-up PNG preview. `load_activation_image` returns the array only if grid, CRS, shape and SHA-256 match the manifest. Images and manifests stay out of Git; the attribution "Contains modified Copernicus Sentinel data 2022" applies to any redistributed derivative.

Acquisition on 2026-10-02 for `runs/geographic-environment-family-expanded-20261002` (output `runs/geographic-imagery-s2-20261002`, 9 grids) used one clear scene for eight grids, dated 2022-05-28 to 2022-08-03 and 14–24 days before each activation, and a three-scene spring composite for EMSR578 (2022-04-30, 47 days before activation, for 91.9% of pixels; 2022-05-30 and 2022-05-15 filling 5.0% and 3.1% of cloud-buffered pixels). As a registration check, the correlation between burnable cells in the fuel map and cell greenness (green minus red) is highest for the stored orientation where the fuel map has nonburnable contrast: EMSR600 +0.50 (flipped +0.02, mirrored +0.21), EMSR583 +0.23 (−0.11, −0.01), EMSR602_AOI04 +0.24 (−0.03, +0.15). Grids with almost no nonburnable cells give no usable signal.

Limits: TCI is a display product and not radiometrically calibrated reflectance. Pre-fire vegetation can differ from the state at ignition. Composite seams can join different dates. The image does not show the CEMS-mapped initial perimeter, which is drawn over it from the scenario state in the planned per-step composite. No overlay, policy input or simulator behaviour uses the imagery yet.

## Multi-focus, shifting-wind family (D-60)

`python3 -m nightjar.multi_ignition_family --scenarios-dir PARENT --output-dir OUT` derives, for each parent scenario, scenarios with 2–4 separated foci inside the CEMS burned area and synthetic wind that shifts every 900 s. The wind is stored as `weather_mode: "synthetic_intervals"` forcing, validated to be continuous, to cover the horizon and to be known at each interval's start. Parents with measured station intervals keep them. The viewer labels the mode "Viento sintético variable".

## Physics and observation separation

Directional Pyretechnics behavior is cached by forcing interval. For each grid edge, the harmonic mean of source/target rates gives its speed. Remaining travel distance is integrated across interval boundaries; using a single start-time speed for an edge crossing a weather change is avoided. Constant-weather cases keep the legacy propagation path. The existing wind limit and minimum burnable rate remain model assumptions; heterogeneous convergence and alternative numerical schemes remain untested.

The full retrospective forcing belongs to simulator state and replay audit only. Controllers/baselines receive public terrain and the latest interval whose `available_at_s` has passed. Aggregated values are delivered at interval end, with extra publication latency unknown and assumed zero. Before the first completed interval, the observation records `weather_status=not_yet_delivered` and a zero wind placeholder, not measured calm. The PPO vector has no dedicated weather-validity channel; dimensions remain unchanged. Future station aggregates, latent fire arrivals and the full forcing array never enter policy observations. An interval clipped by the horizon remains undelivered until its original end. Source station-to-landscape calibration and realistic reporting delays remain open. Since D-49, geographic episodes default to `observation_mode="realtime"`: the controller sees the interval that contains the decision time and no future interval; the completed-interval rule above applies to `"scheduled"` mode.

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
