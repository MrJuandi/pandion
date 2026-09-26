# Related work and data sources

This is a starting reading list, not a systematic review. Claims and implementation details should be checked against full papers, software versions, and dataset documentation before publication.

## Environments and decision methods

- [SimFire](https://github.com/mitrefireline/simfire) and [SimHarness](https://github.com/mitrefireline/simharness): open wildfire simulation and RL harness. Relevant for environment boundaries, mitigation actions, and baseline design.
- [Cell2Fire](https://github.com/cell2fire/Cell2Fire): cell-based propagation with fuel, weather, moisture, topography, and spatial/statistical outputs. Candidate external comparator for unmitigated spread.
- [Oulu Wildfire Drones](https://github.com/Arritmic/oulu-wildfire-drones): drone-response simulation with PPO, heuristic, auction, refill constraints, and operational metrics. Relevant to the two-aircraft stage.
- [JaxWildfire paper](https://arxiv.org/abs/2512.06102): vectorized wildfire simulation for RL. Relevant if measured simulation throughput becomes a bottleneck; the linked [repository](https://github.com/ori-goals/JaxWildfire) did not provide implementation code at the time of review.
- [Vo et al. (2025), task-allocation comparison](https://consensus.app/papers/a-comparative-analysis-of-task-allocation-strategies-for-vo-casado/fc5abcb8184c5d789dafec0d79f9f192/): PPO versus heuristic and auction in resource-constrained drone response.
- [Tapley et al. (2023), SimFire/SimHarness](https://consensus.app/papers/reinforcement-learning-for-wildfire-mitigation-in-tapley-dotter/b5817762f0ac5665b44ba705545b5555/): simulation-based wildfire mitigation research.

## Candidate geospatial and meteorological products

- [IGN/PNOA terrain downloads](https://pnoa.ign.es/pnoa-lidar/productos-a-descarga).
- [Copernicus Forest Type](https://land.copernicus.eu/en/products/high-resolution-layer-forests-and-tree-cover/forest-type-2021-raster-10-m-100-m-europe-3-yearly) and [CLC+ Backbone](https://library.land.copernicus.eu/products/CLCplus_Backbone_2021_PUM_v1.html).
- [Copernicus Sentinel-2 documentation](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html).
- [NASA FIRMS active-fire observations](https://firms.modaps.eosdis.nasa.gov/).
- [AEMET OpenData](https://www.aemet.es/en/datos_abiertos/AEMET_OpenData).
- [ECMWF ERA5 hourly reanalysis](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-data-single-levels-1940-present).

Dataset inclusion requires temporal alignment, provenance, and licensing checks described in [DATA_AND_VALIDATION.md](DATA_AND_VALIDATION.md).
