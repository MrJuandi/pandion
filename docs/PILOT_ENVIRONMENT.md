# EMSR578 environmental inventory and diagnostic protocol

**Status, 2026-09-30:** numeric terrain ingestion and historical station-coverage audit implemented; incident reconstruction and physical validation remain open. This pilot is development evidence. The geographic PPO family still uses its existing flat terrain and synthetic environment.

## What three daily maps can establish

Copernicus EMSR578 AOI01 provides cumulative burned-area snapshots at 2022-06-17 10:06 UTC, 2022-06-18 10:47 UTC, and 2022-06-19 10:42 UTC. Their packages and image timestamps were audited in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md). There are two observed growth intervals in one incident, not three independent incidents or a continuous fire-front record.

Start from the initial map and compare at **88,860 s** and **174,960 s** from that same start, without resetting the model to a later observed map. Report cumulative area/overlap, newly burned area outside the initial mask, and the initial-only baseline. A separate one-step restart from MONIT01 could be useful diagnostically but must be labeled as such. Do not infer an hourly arrival trajectory by interpolating the three polygons, or choose a shorter simulation horizon to match the final area.

These endpoints can expose daily growth discrepancies. They cannot establish within-day speed, pauses, which parts of the initial burn were actively flaming, or the causal effect of suppression. The current simulator seeds all initially burned cells; active-front initialization remains an assumption. No physical acceptance threshold is declared here. Define thresholds, spatial/timing uncertainty and eligible intervention conditions before a stronger separate-incident evaluation; EMSR590 can serve that evaluation role, while EMSR592/EMSR600 remain reserved.

## Source inventory

| Layer | Source and audit outcome | Remaining limits |
|---|---|---|
| Burned area | Three checksum-verified CEMS AOI01 packages, timestamps from map PDFs | Daily cumulative footprints, not continuously observed fronts; uncertainty and interventions unresolved |
| Numeric terrain | IGN/CNIG PNOA-LiDAR WCS `Elevacion25830_25`; native 25 m grid, service reprojection to EPSG:25831 and nearest-neighbor sampling to the 100 m pilot grid | Orthometric elevation; exact tile acquisition epoch/vertical realization unverified. Returned ASCII has no NODATA header, so the source footprint remains unknown |
| Station weather | Meteocat/XEMA Baldomar `X6`, 41.92173 N, 1.02901 E, altitude 366 m; five variables have 98/98 valid half-hour intervals overlapping the observation window | Point measurements; representativeness across the landscape and conversion to midflame wind/fuel moisture remain open |
| Additional fire activity | [NASA FIRMS archive](https://firms.modaps.eosdis.nasa.gov/download/) lists VIIRS S-NPP/NOAA-20 and MODIS coverage for June 2022 | Not acquired in this audit; archive requests require Earthdata/email authentication. Thermal detections are intermittent samples with clouds/geometry/size limits, not exact burned-area perimeters |
| Intervention chronology | [Interior/Bombers incident-information page](https://interior.gencat.cat/ca/incendis-forestals/coneixem/) includes Artesa/Baldomar material for June 2022 | Quantified, time-stamped intervention locations/effects have not been obtained; narrative material alone does not supply them |
| Aircraft deposition | [US Forest Service drop-test method](https://www.fs.usda.gov/t-d/pubs/pdfpubs/pdf04572813/pdf04572813dpi300.pdf) measures ground patterns with collecting cups | A method/source lead, not a measured CL-415 water footprint or a water-effect coefficient for this scenario |

The [IGN WCS description](https://servicios.idee.es/wcs-inspire/mdt?service=WCS&request=DescribeCoverage&version=1.0.0&coverage=Elevacion25830_25) advertises EPSG:25831 responses, ArcGrid, and nearest-neighbor as default interpolation. Capabilities advertise CC BY 4.0 attribution to scne.es. This numeric service was chosen instead of treating the [ICGC display WMS](https://www.icgc.cat/es/Geoinformacion-y-mapas/Servicios-en-linea-Geoservicios/WMS-y-WCS-Elevaciones/WMS-Elevaciones-territorial) as elevations: a rendered map is not a numeric height raster. The ASCII format has no CRS, so the requested CRS is an explicit caller assertion, not a CRS discovered from file bytes.

[Meteocat open data](https://www.meteo.cat/wpweb/serveis/dades-obertes/) supplies station/variable metadata and measurements. [XEMA documentation](https://apidocs.meteocat.gencat.cat/documentacio/dades-de-la-xema/) states all times are UTC and readings label the **following** interval: 10:00 labels 10:00–10:30. Only quality code `V` is counted as usable. Half-hour scalar speed and mean direction at 2 m are retained as separate measurements; their combination is not asserted to be an exact interval vector mean. Air humidity is not vegetation moisture. Raw exports stay ignored; redistribution terms for derived weather products require review before publishing those products.

## Implemented interfaces

`nightjar.pilot_terrain` ingests local ESRI ASCII, checks shape/origin/cell size against the scenario, reverses north-first file rows to Nightjar's south-to-north rows, and emits `nightjar-pilot-terrain-0.1`. Hashes bind raw bytes and canonical scenario content; source/CRS/unit/UTC timestamp and deterministic null seed are recorded. NODATA stays null with a separate validity mask; a zero-metre elevation is valid. Without a NODATA header, that mask means numeric validity, not a verified source footprint. The origin comes from the existing scenario metadata rounded to 0.1 m; original fuel-raster sampling before rounding can differ by up to 0.05 m per axis. Service resampling and that convention remain numerical limits.

`pilot_simulation run --terrain` rejects artifacts for another scenario, wrong grid orientation/alignment, and missing/nonfinite heights. It records terrain artifact/raw hashes and source/coverage metadata. Omission of `--terrain` retains the explicitly synthetic 350 m flat grid. Wind and moisture are still constant exploratory parameters in either case; the weather audit is **not** engine forcing.

`nightjar.pilot_weather` reads local station/variable/measurement JSON exports and emits `nightjar-pilot-weather-audit-0.1`. It audits the 98 forward intervals overlapping t0–t2, preserves quality flags and explicit gaps without filling, checks station identity, units, cadence and duplicate readings, and binds all three raw hashes plus the scenario hash. It has no controller or engine dependency and does not supply future observations to a policy.

`pilot_simulation evaluate` emits `nightjar-pilot-comparison-0.1` with CRS, source/scenario hashes, UTC creation time, deterministic null seed, `growth_only`, `initial_only_baseline` and `timing`. Comparison-image time is identified by the actual ZIP digest in the reviewed EMSR578 catalog, not its filename. Unknown products are explicitly time-unknown; legacy simulations without a scenario hash cannot establish scenario binding. Spatial metrics are exploratory even when elapsed time matches.

## Reproduce acquisition and audit

Use the documented Python 3.12 environment with `requirements-standard.txt`; `curl` is needed only for explicit acquisition. Existing pilot source-audit and manifest commands are in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md). All commands below were exercised on the existing pilot scenario.

```sh
mkdir -p data/raw/pilot-environment runs/pilot-environment-source-audit
curl -fsSL --max-time 30 'https://analisi.transparenciacatalunya.cat/resource/yqwd-vj5e.json?$limit=1000' -o data/raw/pilot-environment/stations.json
curl -fsSL --max-time 30 'https://analisi.transparenciacatalunya.cat/resource/4fb2-n3yi.json?$limit=1000' -o data/raw/pilot-environment/variables.json
curl -fsSL --max-time 30 'https://analisi.transparenciacatalunya.cat/resource/nzvn-apee.json?$where=codi_estacio%3D%27X6%27%20AND%20data_lectura%20between%20%272022-06-16T00:00:00%27%20and%20%272022-06-20T00:00:00%27&$limit=5000&$order=data_lectura,codi_variable' -o data/raw/pilot-environment/baldomar-measurements.json
curl -fsSL --max-time 50 'https://servicios.idee.es/wcs-inspire/mdt?service=WCS&version=1.0.0&request=GetCoverage&coverage=Elevacion25830_25&crs=EPSG:25831&response_crs=EPSG:25831&bbox=326636.4,4636071.6,342236.4,4651271.6&width=156&height=152&format=ArcGrid' -o data/raw/pilot-environment/terrain.asc
.venv/bin/python -m nightjar.pilot_terrain \
  --scenario runs/emsr578-pilot-scenario.json \
  --ascii-grid data/raw/pilot-environment/terrain.asc --crs EPSG:25831 \
  --source-url 'https://servicios.idee.es/wcs-inspire/mdt?service=WCS&version=1.0.0&request=GetCoverage&coverage=Elevacion25830_25&crs=EPSG:25831&response_crs=EPSG:25831&bbox=326636.4,4636071.6,342236.4,4651271.6&width=156&height=152&format=ArcGrid' \
  --provider IGN/CNIG --dataset Elevacion25830_25 --native-resolution-m 25 \
  --vertical-reference orthometric --resampling nearest-neighbor \
  --output runs/pilot-environment-source-audit/terrain.json
.venv/bin/python -m nightjar.pilot_weather \
  --scenario runs/emsr578-pilot-scenario.json \
  --stations data/raw/pilot-environment/stations.json \
  --variables data/raw/pilot-environment/variables.json \
  --measurements data/raw/pilot-environment/baldomar-measurements.json \
  --output runs/pilot-environment-source-audit/weather.json
.venv/bin/python -m nightjar.pilot_simulation run \
  --scenario runs/emsr578-pilot-scenario.json \
  --terrain runs/pilot-environment-source-audit/terrain.json --horizon-s 88860 \
  --output runs/pilot-environment-source-audit/simulation-t1.json
.venv/bin/python -m nightjar.pilot_simulation evaluate \
  --simulation runs/pilot-environment-source-audit/simulation-t1.json \
  --comparison-zip data/raw/emsr578/monit01.zip \
  --scenario runs/emsr578-pilot-scenario.json \
  --output runs/pilot-environment-source-audit/comparison-t1.json
.venv/bin/python -m nightjar.pilot_simulation run \
  --scenario runs/emsr578-pilot-scenario.json \
  --terrain runs/pilot-environment-source-audit/terrain.json --horizon-s 174960 \
  --output runs/pilot-environment-source-audit/simulation-t2.json
.venv/bin/python -m nightjar.pilot_simulation evaluate \
  --simulation runs/pilot-environment-source-audit/simulation-t2.json \
  --comparison-zip data/raw/emsr578/monit02.zip \
  --scenario runs/emsr578-pilot-scenario.json \
  --output runs/pilot-environment-source-audit/comparison-t2.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

These live exports can change. The measurement request returned 2,509 records (13 variables, 193 labels including the final inclusive boundary); the required five variables each have 98 valid overlapping intervals. Hashes describe this audit, not a promise of immutable remote exports. The terrain request is specific to the existing 152×156, 100 m pilot; a different scenario/grid needs a different request and must pass alignment checks.

| Raw export | SHA-256 recorded on 2026-09-30 |
|---|---|
| `terrain.asc` | `f0149bf383bb62e7f8ea9affaa14cddc8dc0062cdc3c4c53024b2bab899d6aee` |
| `stations.json` | `5a8294b9e61120ecab22ea3648afafd75746b6cd46ab73665918cbee86ef9462` |
| `variables.json` | `dc304d79dc5d0b9afeccd2f13849f41e828b6f8fa83eb4b16674a1c7822647e1` |
| `baldomar-measurements.json` | `f3c5d179900a01265493a384655108630eb44fccc135e42e43c8db5ff1dc8f0f` |

## Diagnostic evidence and next gate

The numeric terrain covers 23,712 cells with finite heights of 266–1,386 m; the unadvertised source footprint and acquisition epoch remain unknown. Over the fire observation window, Baldomar valid temperatures range 18–39.8 °C, relative humidity 10–56%, half-hour rain is zero, and 2 m scalar wind speeds range 0.1–4.3 m/s. These are station summaries, not inferred fire-cell inputs.

Runs below use the acquired terrain but retain the old constant wind/moisture and all-burned-cell initialization. Both are continuous from t0 and labeled **exploratory**:

| Target | Elapsed seconds | Simulated / observed cumulative ha | Cumulative IoU | New-growth IoU | Initial-only IoU |
|---|---:|---:|---:|---:|---:|
| MONIT01 | 88,860 | 15,900 / 2,108 | 0.132579 | 0.042292 | 0.710763 |
| MONIT02 | 174,960 | 17,612 / 2,692 | 0.152850 | 0.074044 | 0.556628 |

The large discrepancy is evidence that this constant-environment configuration does not reproduce these endpoints. It does not measure suppression effectiveness or distinguish causes. The new terrain input does not solve the missing temporal/environmental/intervention assumptions.

Next: specify/test time-dependent propagation driven by sourced weather, justified station-to-landscape/midflame conversion and fuel-moisture estimates with uncertainty; seek more active-front/intervention evidence; compare heterogeneous resolutions; then fix an independent-incident protocol. For water, separately obtain aircraft-comparable deposition evidence and pure-water fire-response/drying evidence; published gel/retardant persistence must not be copied to plain water. Until those data exist, explicitly bracket simulator assumptions instead of claiming measured effectiveness.

Local verification used Python 3.12.8, NumPy 1.26.4, Pyretechnics 2026.8.10 and CPU PyTorch 2.14.0: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v` passed **184 tests, no failures or skips** in 34.240 s. Deterministic fixtures test row orientation, NODATA versus valid zero heights, alignment/provenance failures, weather gaps/quality/duplicates/units, uphill direction, terrain CLI integration and growth/timing diagnostics. The numerical and scientific gates above remain open.
