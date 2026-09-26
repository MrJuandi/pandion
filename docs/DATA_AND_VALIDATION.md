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
