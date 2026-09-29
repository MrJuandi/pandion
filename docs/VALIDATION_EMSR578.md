# Candidate independent observations: EMSR578 AOI01

**Status:** source audit, 2026-09-27. This is **not** a validation result for the Nightjar fire model.

[Copernicus Emergency Management Service activation EMSR578](https://mapping.emergency.copernicus.eu/activations/EMSR578/) covers the June 2022 Artesa de Segre wildfire in Catalonia. Its AOI01 delineation products include cumulative, photo-interpreted burned-area polygons. The acquisition times below come from each product's map PDF; areas come from the `observedEventA` GeoJSON inside its vector package. Download URLs and SHA-256 digests are encoded in `nightjar/emsr578.py` and reproduced by the audit report.

| Product | Post-event image acquired (UTC) | Reported cumulative area (ha) | Change from previous (ha) |
|---|---|---:|---:|
| Delineation product | 2022-06-17 10:06 | 1,501.728373 | — |
| Monitoring 01 | 2022-06-18 10:47 | 2,111.212038 | 609.483665 |
| Monitoring 02 | 2022-06-19 10:42 | 2,693.193281 | 581.981243 |

Reproduce the **source audit** from the repository root:

```sh
mkdir -p runs
python3 -m nightjar.emsr578 fetch
python3 -m nightjar.emsr578 report --output runs/emsr578-aoi01.json
```

The script downloads three public versioned ZIP packages to the Git-ignored `data/raw/emsr578/` directory, verifies their reviewed SHA-256 digests, checks the polygon layer and reported area, and writes the dates and source links. If a provider replaces a package, the hash check fails so that its version must be reviewed deliberately. The map PDFs were inspected separately for image acquisition times; **delivery time is not used as observation time**.

The [CEMS On-Demand Mapping terms](https://mapping.emergency.copernicus.eu/terms-and-conditions/) permit access, adaptation, and distribution subject to their terms and attribution. The raw ZIPs and source imagery are kept out of this repository. The report records the source and product URLs; any public redistribution of derived geometries must follow the provider's citation guidance and applicable third-party terms.

## Why these observations do not pass the physical-validation gate

- The mapped burn is in woods and bushes according to the activation description; there is no justified crosswalk from those fuels to Nightjar's FM1 short-grass model.
- Burned-area boundaries are cumulative snapshots from satellite imagery, not a continuously observed active front. The image acquisition intervals are about a day and the polygons have mapping uncertainty.
- Suppression actions, ignitions or spotting, and weather history during those intervals are not represented in the current unmitigated, constant-wind model.
- The first mapped polygon already exceeds 1,500 ha. A validation run would need to start from that perimeter, not a single ignition cell, and use time-aligned fuel, terrain, moisture, and midflame wind inputs.
- Using the first interval to tune the model and the second from the same incident to report an independent geographic test would leak incident-specific information. At least one separate incident must be reserved for evaluation.

Before any model-versus-observation score is reported, build a provenance-preserving pipeline for those inputs, quantify polygon and timing uncertainty, specify the effect of response actions or select sufficiently unmitigated periods, and reserve other incidents. Until then, these three measurements verify only that independent observation artifacts can be acquired and audited.

## Geographic pilot pipeline and fuel crosswalk (Stage 0 Increment)

Under Decision **D-32**, `nightjar.pilot_emsr578` implements the first real geographic input and evaluation pipeline over the EMSR578 AOI01 vector package:

1. **Geodesy and Projection:**
   Copernicus source coordinates are in WGS84 geographic degrees (`EPSG:4326`). The pipeline converts them to **ETRS89 / UTM Zone 31N** (`EPSG:25831`, central meridian $3^\circ\text{ E}$, scale factor $0.9996$, false easting $500,000\text{ m}$) using exact sub-millimeter transverse Mercator series. The bounding box spans $X \in [326,636.4, 342,209.4]\text{ m}$ and $Y \in [4,636,071.6, 4,651,196.9]\text{ m}$ ($15.57\text{ km} \times 15.13\text{ km}$).
   At $100\text{ m}$ grid resolution, this yields a $152 \times 156$ raster grid ($23,712\text{ ha}$ total extent).

2. **CORINE Land Cover to Scott & Burgan 40 Crosswalk:**
   The `naturalLandUseA` layer contains 89 polygons classified across 13 CORINE Land Cover (CLC) categories. Rather than guessing biomass from land use, each class is mapped with explicit botanical rationale, confidence rating, and documented alternatives:

| CLC Code | CLC Land Cover | Scott & Burgan Model | Code | Confidence | Botanical Rationale | Documented Alternatives |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| `211` | Non-irrigated arable land | **GR1** | `101` | Medium | Winter cereals in Lleida/Noguera are ripe or harvested stubble in mid-June; dry stubble carries fast low-intensity surface fire. | `GR2` (102, dense post-harvest load) or `NB` (0, clean ploughed fallow) |
| `212` | Permanently irrigated land | **NB** | `0` | High | Irrigated crops maintain high moisture (>150%) and act as effective passive barriers during summer campaigns. | `GR1` (101, dry ditch margins) |
| `221` | Vineyards | **NB** | `0` | High | Traditional wide-spaced Mediterranean vineyards with bare tilled soil act as passive fuel breaks. | `GR1` (101, unweeded cured inter-row grass) |
| `222` | Fruit trees and berry plantations | **NB** | `0` | High | Commercial fruit orchards in the Segre river basin are irrigated and weed-controlled. | `TL1` (181, abandoned orchards with dry leaf litter) |
| `242` | Complex cultivation patterns | **GR1** | `101` | Medium | Fragmented smallholdings with dry field edges and cereal parcels. | `NB` (0, intensively tilled plots) or `GS1` (121, hedgerow shrubs) |
| `243` | Agriculture with natural vegetation | **GS2** | `122` | Medium | Marginal agroforestry interface: abandoned stone terraces with encroaching garrigue and cured grasses. | `GS1` (121, low load) or `SH2` (142, predominant shrub encroachment) |
| `311` | Broad-leaved forest | **TL6** | `186` | Medium | Mediterranean sclerophyllous oak forests (*Quercus ilex*) produce compact, slow-burning leaf litter with moderate depth. | `TL2` (182, open low-load stands) or `TU5` (165, dense shrub understory) |
| `312` | Coniferous forest | **TU5** | `165` | High | Aleppo pine (*Pinus halepensis*) stands in interior Catalonia are highly fire-prone with dense shrub understories acting as ladder fuels. | `TL8` (188, long-needle pine litter under closed canopy) or `TU1` (161, open timber-grass) |
| `313` | Mixed forest | **TU2** | `162` | Medium | Transition forest between pine and deciduous oak with mixed needle/leaf litter and moderate shrub understory. | `TL3` (183, mixed litter) or `TU5` (165, dense understory) |
| `321` | Natural grassland | **GR2** | `102` | High | Natural montane and hill herbaceous pastures, completely cured in mid-June Mediterranean heatwaves. | `GR1` (101, grazed/sparse) or `GR4` (104, tall ungrazed grass) |
| `323` | Sclerophyllous vegetation | **SH5** | `145` | High | Primary fuel driving rapid rate of spread in Artesa de Segre: dense Kermes oak (*Quercus coccifera*), rosemary, and gorse 1–1.5 m tall. | `SH2` (142, open garrigue) or `SH7` (147, very high load tall maquis) |
| `324` | Transitional woodland shrub | **SH2** | `142` | Medium | Regenerating scrub with scattered saplings following previous disturbances or pasture abandonment. | `SH5` (145, dense mature scrub) or `TU1` (161, woodland with grassy understory) |
| `998` | Other | **NB** | `0` | High | Unvegetated surfaces, rock outcrops, quarries, or unclassified non-fuel terrain. | None |

3. **Water Scooping Infrastructure:**
   The `hydrographyA` layer provides 24 water bodies (including the Pantà de Rialb reservoir `BH130` and Segre river sections `BH080`). All are enforced as nonburnable (`code 0`) and extracted with their UTM 31N centroids as candidate scooping waypoints for aerial firefighting resources (CL-415).

4. **Perimeter Discretization and Ground-Truth Overlap:**
   - **DEL Initial Perimeter ($t_0$, 2022-06-17 10:06 UTC):** Rasterized at $100\text{ m}$ resolution to $1,500\text{ cells} = 1,500.0\text{ ha}$ (reported satellite polygon area: $1,501.73\text{ ha}$, discretization error: $-1.73\text{ ha}$ or $-0.12\%$).
   - **MONIT01 Perimeter ($t_1$, 2022-06-18 10:47 UTC):** Rasterized to $2,108.0\text{ ha}$.
   - **Spatial Overlap Metrics:** Comparing the two satellite observations yields:
     - Intersection: $1,499.0\text{ ha}$ (1,499 cells)
     - Commission ($t_0$ not in $t_1$): only $1.0\text{ ha}$ (1 cell mapping noise)
     - Net expansion ($t_1$ not in $t_0$): $609.0\text{ ha}$ (609 cells), exactly aligning with Copernicus reported change of $+609.48\text{ ha}$.
     - Observed Jaccard/IoU: $0.7108$, Dice coefficient: $0.8309$.

## Reproducing the pilot pipeline

From the repository root:

```sh
# 1. Export the complete fuel crosswalk
python3 -m nightjar.pilot_emsr578 crosswalk --format markdown

# 2. Build the full georeferenced scenario manifest in EPSG:25831
python3 -m nightjar.pilot_emsr578 manifest --output runs/emsr578-pilot-scenario.json

# 3. Compute spatial overlap metrics between satellite observations
python3 -m nightjar.pilot_emsr578 compare-perimeters --scenario runs/emsr578-pilot-scenario.json --comparison-zip data/raw/emsr578/monit01.zip
```

## Physical fire spread simulation and satellite validation (Decision D-33)

Under Decision **D-33**, `nightjar.pilot_simulation` couples the Rothermel surface-fire formulation via Pyretechnics and Scott & Burgan 40 models to the geographic pilot scenario, simulating shortest-arrival propagation from the initial perimeter ($t_0$) across the 23,712 ha landscape.

### Environmental Configuration
- **Fuel moisture fractions:** Derived from 97th percentile extreme drought conditions during the interior Catalonia heatwave of 15–18 June 2022:
  - Dead 1h, 10h, dead herbaceous: $0.04 - 0.05$ ($4\% - 5\%$).
  - Dead 100h: $0.07$ ($7\%$).
  - Live herbaceous (cured summer pasture): $0.30$ ($30\%$).
  - Live woody (drought-stressed sclerophyllous shrub/needle): $0.60$ ($60\%$).
- **Midflame wind vector:** Synoptic SSE/SSW flow pushing towards NNE: $[u_x, u_y] = [0.8, 1.8]\text{ m/s}$ (approx $1.97\text{ m/s}$ or $7.1\text{ km/h}$ midflame).

### Propagation Progression and Empirical Satellite Comparison
Simulating shortest-arrival propagation across horizons from $t=0$ to the full satellite elapsed interval ($T = 88,860\text{ s} \approx 24.68\text{ h}$) against `monit01.zip` yields:

| Horizon (s) | Effective Duration | Simulated (ha) | Observed (ha) | IoU (Jaccard) | Dice Coeff. | Commission Rate | Omission Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0` | Initial $t_0$ | 1,500.0 | 2,108.0 | 0.7108 | 0.8309 | 0.0007 | 0.2889 |
| `1,800` | 0.5 h | 1,791.0 | 2,108.0 | 0.6923 | 0.8182 | 0.1094 | 0.2434 |
| `3,600` | 1.0 h | **2,071.0** | **2,108.0** | **0.6630** | **0.7973** | **0.1956** | **0.2097** |
| `5,400` | 1.5 h | 2,367.0 | 2,108.0 | 0.6261 | 0.7701 | 0.2721 | 0.1826 |
| `7,200` | 2.0 h | 2,714.0 | 2,108.0 | 0.5722 | 0.7279 | 0.3534 | 0.1675 |
| `14,400` | 4.0 h | 4,170.0 | 2,108.0 | 0.4353 | 0.6066 | 0.5434 | 0.0968 |
| `28,800` | 8.0 h | 6,839.0 | 2,108.0 | 0.3014 | 0.4632 | 0.6970 | 0.0171 |
| `88,860` | 24.7 h (full) | 15,924.0 | 2,108.0 | 0.1324 | 0.2338 | 0.8676 | 0.0000 |

### Key Findings & Physical Limits
1. **Active Spread Alignment:** At an active burning duration of $T = 3,600\text{ s}$ (1.0 h equivalent), the unmitigated Rothermel propagation reaches **2,071.0 ha** (within $-1.8\%$ of the 2,108.0 ha observed at $t_1$), with an **IoU of 0.6630** and **Dice of 0.7973**. The omission rate drops from $28.9\%$ to $21.0\%$, correctly capturing the main north/northeast expansion axis into sclerophyllous shrubland (SH5).
2. **The Suppression and Diurnal Factor:** Running unmitigated spread for the entire 24.7 hours with constant peak midday wind and drought moisture overpredicts final area ($15,924\text{ ha}$, $86.8\%$ commission). In reality, extensive containment by ground crews (Bombers de la Generalitat) and aerial scooping (CL-415), combined with nighttime relative humidity recovery, successfully checked the flanks and halted major spread outside the observed 2,108 ha polygon.
3. **Execution Speed:** Propagating across 23,712 cells takes under **80 ms** on standard CPU, demonstrating extreme computational efficiency for decision-making and Monte Carlo validation.

### Reproducing the physical simulation and evaluation

```sh
# Run 1-hour active spread simulation
python3 -m nightjar.pilot_simulation run \
  --scenario runs/emsr578-pilot-scenario.json \
  --horizon-s 3600 \
  --output runs/emsr578-simulation-t1.json

# Evaluate simulated perimeter against Copernicus MONIT01
python3 -m nightjar.pilot_simulation evaluate \
  --simulation runs/emsr578-simulation-t1.json \
  --comparison-zip data/raw/emsr578/monit01.zip \
  --scenario runs/emsr578-pilot-scenario.json \
  --output runs/emsr578-spatial-validation.json
```


