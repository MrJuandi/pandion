# Annaburroo grass-fire external diagnostic

**Status, 2026-09-27:** a no-fit diagnostic of the fixed Anderson FM1 head-spread calculation, **not** validation of Spanish incident spread, grid geometry, or suppression response.

## Source and reproduction

The [CSIRO Annaburroo collection](https://data.csiro.au/collection/csiro:58746), collection version 1, file ID `64873323`, contains the [1986 experimental grassland fire data described by Cruz et al. (2024)](https://connectsci.au/wf/article/33/5/WF23100/22126/The-1986-Annaburroo-experimental-grassland-fires). The collection's downloadable archive is 593,231,549 bytes and marked CC BY 4.0. `nightjar.annaburroo` retrieves only the two CSV members by HTTP byte range; it requires `curl`. Raw tables and generated reports remain in ignored `data/raw/` and `runs/`.

```sh
python3 -m nightjar.annaburroo --output runs/annaburroo-source.json
python3 -m nightjar.annaburroo_compare --output runs/annaburroo-fm1-diagnostic.json
```

The audited `Plot_Average_Data.csv` is 8,669 bytes, SHA-256 `4425d11fe970e263d400b7a9d089b8eea70bda4de4a1e29cbeedea28bc727934`; `Burn_Period_Data.csv` is 84,062 bytes, SHA-256 `bdd5bc1354b7b9e72e17616aa733b5bc869fa3b86686ea102dcb73846761ca0d`. The retrieval checks archive size and response byte ranges, and the comparison refuses a changed plot-table checksum. Archive membership, file identity, and license should be rechecked if CSIRO revises the collection.

## Declared comparison

Use every line-ignition plot average with numeric `MidFlwind.ms`, numeric `DFMC.%`, and positive `RSS.ms`. `RSS.ms` is the source's width-adjusted steady-state rate, derived from observed spread and effective head-fire width. Set FM1 moisture to reported dead-fuel moisture divided by 100 and wind to reported midflame wind; set slope to zero. Predict the one-dimensional head rate `R0 × (1 + φw)` with the unchanged Anderson FM1 fuel constants. No parameter is fitted and the script does not select plots by outcome. The output JSON stores each plot, prediction, observed value, and error.

| Statistic | Result |
|---|---:|
| Plot rows / included / excluded | 121 / 120 / 1 |
| Exclusion | F011: wind and moisture marked `NA` |
| Mean observed RSS | 1.2833 m/s |
| Mean predicted FM1 rate | 1.1746 m/s |
| Mean absolute error | 0.5264 m/s |
| Median absolute error | 0.4214 m/s |
| Signed bias (prediction minus observation) | −0.1087 m/s |
| ER fuel type MAE, n=70 | 0.5068 m/s |
| TH fuel type MAE, n=50 | 0.5538 m/s |

These errors are substantial relative to the mean observed rate. They are diagnostic of transferring *fixed FM1* to these Australian grass plots, which vary in grass species, load, height, curing, treatment, and ignition geometry. Their plot averages are from one experiment and are not independent geographic wildfires. The comparison assumes level ground because plot slopes are not supplied in the selected table; it does not estimate measurement uncertainty. Thus it cannot establish fitness for a Spanish forest fire, an aircraft-drop effect, or a policy-training environment. A later calibrated fuel model must use separate training and test incidents and preserve these unfitted results as a reference.
