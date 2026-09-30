# Related work and data sources

This is a starting reading list, not a systematic review. Claims and implementation details should be checked against full papers, software versions, and dataset documentation before publication.

## Environments and decision methods

- [Rothermel (1972), USDA Forest Service](https://research.fs.usda.gov/treesearch/32533): original surface-fire spread model; source for the initial candidate equation.
- [Andrews (2018), USDA Forest Service](https://research.fs.usda.gov/treesearch/55928): consolidated Rothermel equations, assumptions, and related spread-direction treatments.
- [Anderson (1982), USDA Forest Service](https://doi.org/10.2737/INT-GTR-122): original 13 fire-behavior fuel models; FM1 short-grass parameters used in the first Rothermel subset.
- [Scott and Burgan (2005), USDA Forest Service](https://research.fs.usda.gov/treesearch/9521): 40 standard fire-behavior fuel models, including GR1 and SH1, with dynamic herbaceous fuel treatment.
- [Pyretechnics](https://github.com/pyregence/pyretechnics), pinned at 2026.8.10 for the optional standard-fuel path: implements the fuel catalogue and Rothermel surface-fire calculations; Nightjar supplies its own grid-arrival coupling. EPL-2.0 applies to that dependency.
- [McFayden et al., *Reference Guide to the Drop Effectiveness of Skimmer and Rotary Wing Airtankers* (Canadian Forest Service, 2023)](https://publications.gc.ca/collections/collection_2023/rncan-nrcan/Fo123-2-35-2023-eng.pdf): CL-415 capacity, typical indicated airspeeds, drop conditions, footprint measurements, and simplified cycle timing; transfer to Nightjar conditions requires validation.
- [Avincis CL-415 specification](https://www.avincis.com/wp-content/uploads/2024/11/A4-CL-415-Specifications-Flyer-Low-Res.pdf): operator specification of 6,137 L water-tank capacity.
- The integrated example's 35 m permanent water barrier is an unsourced software-test assumption. None of the aircraft references above supports that footprint or effect.
- [Copernicus EMSR578](https://mapping.emergency.copernicus.eu/activations/EMSR578/), [EMSR590](https://mapping.emergency.copernicus.eu/activations/EMSR590/), [EMSR592](https://mapping.emergency.copernicus.eu/activations/EMSR592/), and [EMSR600](https://mapping.emergency.copernicus.eu/activations/EMSR600/): registered CEMS incident products used for development, validation, and reserved static maps. The [CEMS data terms](https://mapping.emergency.copernicus.eu/terms-and-conditions/) govern use and attribution; product-level SHA-256 values and acquisition times are recorded in `nightjar/cems_catalog.py` and copied into generated scenario provenance. The EMSR578 source audit and limitations are in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md). Supplied bases must match their activation keys and source classification; synthetic fixtures carry no CEMS ZIP checksum. The identity regression evidence uses generated fixtures, without new external products.

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
## External grass-fire diagnostic

- [Cruz et al. (2024), The 1986 Annaburroo experimental grassland fires: data](https://connectsci.au/wf/article/33/5/WF23100/22126/The-1986-Annaburroo-experimental-grassland-fires): plot variables, experimental context, and definition of the width-adjusted steady-state spread rate.
- [CSIRO Data Access Portal, Annaburroo collection csiro:58746](https://data.csiro.au/collection/csiro:58746): collection version 1 and downloadable data. See [the source audit and diagnostic](VALIDATION_ANNABURROO.md).
