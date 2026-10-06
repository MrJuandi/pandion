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
- [Copernicus EMSR578](https://mapping.emergency.copernicus.eu/activations/EMSR578/), [EMSR590](https://mapping.emergency.copernicus.eu/activations/EMSR590/), [EMSR592](https://mapping.emergency.copernicus.eu/activations/EMSR592/), and [EMSR600](https://mapping.emergency.copernicus.eu/activations/EMSR600/): registered CEMS incident products used for development, validation, and reserved static maps. The [CEMS data terms](https://mapping.emergency.copernicus.eu/terms-and-conditions/) govern use and attribution; product-level SHA-256 values and acquisition times are recorded in `nightjar/cems_catalog.py` and copied into generated scenario provenance. The EMSR578 source audit and limitations are in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md). Supplied bases must match their activation keys and source classification; synthetic fixtures carry no CEMS ZIP checksum. The identity regression evidence uses generated fixtures, without new external products. PPO verifies the stage, ZIP digest, product code, filename, and acquisition time against the same catalog entry; these consistency tests do not establish physical validity.

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

## EMSR578 environmental inventory

- [IGN/CNIG terrain WCS catalog](https://datos.gob.es/es/catalogo/e0dat0002-wcs-de-modelos-digitales-del-terreno-de-espana1) and [25 m coverage description](https://servicios.idee.es/wcs-inspire/mdt?service=WCS&request=DescribeCoverage&version=1.0.0&coverage=Elevacion25830_25): numeric elevations and advertised CRS/format/interpolation; source acquisition epoch remains unverified. The [IGN 2017 report](https://www.ign.es/web/resources/actividades/memoria/MemoriaIGN2017.pdf) describes PNOA-LiDAR-derived orthometric heights.
- [Meteocat open station data](https://www.meteo.cat/wpweb/serveis/dades-obertes/) and [XEMA timestamp semantics](https://apidocs.meteocat.gencat.cat/documentacio/dades-de-la-xema/): UTC and labels at the start of the following interval; station variables/units and validity. Accessed 2026-09-30.
- [NASA FIRMS archive](https://firms.modaps.eosdis.nasa.gov/download/) and [NASA detection/area limitations](https://forum.earthdata.nasa.gov/viewtopic.php?t=5200): candidate additional fire-activity observations, not acquired in the current audit; thermal detections do not define exact burned-area perimeters.
- [Interior/Bombers incident information](https://interior.gencat.cat/ca/incendis-forestals/coneixem/): source lead for Artesa/Baldomar chronology; no quantified intervention map was acquired.
- [US Forest Service, How to conduct drop tests of aerial retardant delivery systems (2004)](https://www.fs.usda.gov/t-d/pubs/pdfpubs/pdf04572813/pdf04572813dpi300.pdf): procedures for deposition measurements including water. Deposition measurements alone do not validate fire suppression or drying time. No water-effect coefficient is adopted in this change.

Exact local export hashes, sampling conventions and diagnostic limits are in [PILOT_ENVIRONMENT.md](PILOT_ENVIRONMENT.md).
## External grass-fire diagnostic

- [Gould et al. (2024), The 1986 Annaburroo experimental grassland fires: data](https://connectsci.au/wf/article/33/5/WF23100/22126/The-1986-Annaburroo-experimental-grassland-fires): plot variables, experimental context, and definition of the width-adjusted steady-state spread rate.
- [Cheney, Gould and Catchpole (1993), The Influence of Fuel, Weather and Fire Shape Variables on Fire-Spread in Grasslands](https://doi.org/10.1071/WF9930031): ignition and perimeter reconstruction methods (pp. 34–35), wind exclusions for obstructed sensors (p. 35), and documented suppression within the experimental area (p. 43). The source PDF inspected is included in CSIRO collection version 1; see the [B111 assessment](ANNABURROO_B111_ASSESSMENT.md) for its digest and remaining uncertainties.
- [CSIRO Data Access Portal, Annaburroo collection csiro:58746](https://data.csiro.au/collection/csiro:58746): collection version 1 and downloadable data. See [the source audit and diagnostic](VALIDATION_ANNABURROO.md).

- [PyTorch installation guide](https://pytorch.org/get-started/locally/): installation and import checks for macOS and CPU Linux. The geographic PPO verification used PyTorch 2.14.0, Python 3.12.8, and Pyretechnics 2026.8.10 on CPU (2026-09-30); the tested version does not imply compatibility across every `torch>=2.0.0` release. CI resolves the CPU requirement from `https://download.pytorch.org/whl/cpu`.

### Environmental integration and catalog correction, consulted 2026-10-01

- Copernicus official AOI01 DEL maps: [EMSR590 Ladrillar](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR590/EMSR590_AOI01_DEL_PRODUCT_r1_RTP01_v1.pdf), [EMSR592 Landiras](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR592/EMSR592_AOI01_DEL_PRODUCT_r1_RTP01_v1.pdf), [EMSR600 Massarosa](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR600/EMSR600_AOI01_DEL_PRODUCT_r1_RTP01_v1.pdf). These establish product location and post-event image UTC times; source ZIP AOI coordinates independently match the registered geographic envelopes.
- [Copernicus DEM public AWS distribution](https://copernicus-dem-30m.s3.amazonaws.com/readme.html) and [product handbook issue 5.0](https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf): GLO-30 DSM, approximately 30 m native sampling, EGM2008 heights, includes surface objects. It is not bare-earth terrain or a measured 2022 event surface.
- [Meteocat XEMA interval semantics](https://apidocs.meteocat.gencat.cat/documentacio/dades-de-la-xema/): UTC labels mark the following interval; only validated `V` measurements drive the new forcing. Source publication latency after interval end is unknown.
- [USFS PSW-GTR-82](https://www.fs.usda.gov/psw/publications/documents/psw_gtr082/psw_gtr082.pdf): Simard equilibrium dead-fuel moisture equations use temperature Fahrenheit and RH percent. Applying station air values with first-order 1/10/100 h relaxation is the project's uncalibrated approximation, not measured fuel moisture or a full NFDRS reconstruction.
- [Andrews (2012), wind adjustment factor](https://research.fs.usda.gov/treesearch/39729): midflame wind depends on shelter/fuel structure. The station at 2 m is used with an explicit factor 1.0 proxy; no validated 20 ft/10 m wind adjustment is asserted for it.

### Real-time observation and base imagery, consulted 2026-10-02

- Anderson, H. E. (1969). *Heat transfer and fire spread*. USDA Forest Service Research Paper INT-69: flaming residence time t_r ≈ 384/σ min (σ surface-area-to-volume ratio in ft⁻¹), used with front crossing time to separate burning from burned cells (D-49).
- [Earth Search STAC API (Element84)](https://element84.com/earth-search/) and [Sentinel-2 L2A COGs on the AWS Registry of Open Data](https://registry.opendata.aws/sentinel-2-l2a-cogs/): `sentinel-2-l2a` items with `visual` (TCI) and `scl` assets used by `imagery_acquire` (D-50).
- [Copernicus SentiWiki, Sentinel-2 processing](https://sentiwiki.copernicus.eu/web/s2-processing): L2A scene classification (SCL) classes used to screen clouds, shadows, cirrus and invalid pixels.
- [CEMS activation API](https://mapping.emergency.copernicus.eu/activations/api/activations/EMSR578/): `activationTime` per activation, the reference for the pre-fire search window.

## Research interface references

The 2026-10-01 layout review used the [Fire2a QGIS simulator parameter groups](https://fire2a.github.io/docs/qgis-toolbox/algo_simulator.html) and [IFTDSS Map Studio task separation](https://iftdss.firenet.gov/firenetHelp/help/pageHelp/content/10-mapstudio/mapstudioabout.htm). These sources inform UI workflow choices; they do not validate Nightjar physics or policy outcomes.


### Additional development incidents and scooping sources, consulted 2026-10-05

- [CEMS activation list API](https://mapping.emergency.copernicus.eu/activations/api/activations/): wildfire activations in Spain, screened for delineation vector packages under `static/activations/{code}/{code}_AOI{nn}_DEL_PRODUCT_r1_RTP01_v{n}_vector.zip` (D-71). Registered packages are SHA-256 pinned in `nightjar.cems_catalog`.
- [CEMS Rapid Mapping public activations API](https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR921): products from 2023 on; their DEL packages (e.g. EMSR921 AOI01) contain areaOfInterestA, observedEventA and imageFootprintA but no naturalLandUseA or hydrography, so they cannot be built with the CEMS land-use fuel path.
- [OpenStreetMap Overpass API](https://overpass-api.de/): `natural=water` and `landuse=reservoir` within 30 km of each tactical window, screened with the D-70 CL-415 thresholds; data © OpenStreetMap contributors, ODbL.

### CL-415 drop effect and moisture scenarios, consulted 2026-10-06

- McFayden, C. B. et al. (2023). [*Reference Guide to the Drop Effectiveness of Skimmer and Rotary Wing Airtankers*](https://publications.gc.ca/collections/collection_2023/rncan-nrcan/Fo123-2-35-2023-eng.pdf). Natural Resources Canada. Table 7-1 (CL-415 blended speed 158 KIAS, 3.0 min increment per cycle), Table 8-2 and A7 (effective drop dimensions by fire intensity), Table 8-3 and A8.5 (drops needed to hold 100 m of line in open grass O-1a: one up to 1,000 kW/m, three up to 3,200 kW/m, breaching above). Used for D-72.
- Scott, J. H., Burgan, R. E. (2005). [*Standard Fire Behavior Fuel Models: A Comprehensive Set for Use with Rothermel's Surface Fire Spread Model*](https://research.fs.usda.gov/download/treesearch/9521.pdf). USDA Forest Service RMRS-GTR-153, tables 3 and 4: dead (D1–D4) and live (L1–L4) fuel moisture scenarios, the D-72 ensemble.
- [Feux de forêt, Canadair CL-415](https://feuxdeforet.fr/en/moyens-de-lutte/avions/canadair-cl-415/): 290 km/h cruise, about 12 s and 410 m to scoop, rotation under 8 min in French doctrine; consistent with the simulated cycle.
