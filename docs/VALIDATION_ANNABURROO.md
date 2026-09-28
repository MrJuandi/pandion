# Annaburroo grass-fire external diagnostic

**Status, 2026-09-27:** a no-fit diagnostic of the fixed Anderson FM1 head-spread calculation, **not** validation of Spanish incident spread, grid geometry, or suppression response.

## Source and reproduction

The [CSIRO Annaburroo collection](https://data.csiro.au/collection/csiro:58746), collection version 1, file ID `64873323`, contains the [1986 experimental grassland fire data described by Gould et al. (2024)](https://connectsci.au/wf/article/33/5/WF23100/22126/The-1986-Annaburroo-experimental-grassland-fires). The collection's downloadable archive is 593,231,549 bytes and marked CC BY 4.0. `nightjar.annaburroo` retrieves only the two CSV members by HTTP byte range; it requires `curl`. Raw tables and generated reports remain in ignored `data/raw/` and `runs/`.

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

## Temporal head-distance diagnostic, 2026-09-28

The [source article's variable definitions](https://connectsci.au/wf/article/33/5/WF23100/22126/The-1986-Annaburroo-experimental-grassland-fires) identify `TimeSinceIgn` as the seconds since ignition, `CumDist.m` as observed head-fire travel from ignition at each mapped isochrone, and `ROS.ms` as period forward spread rate. The burn-period table has 699 rows. This diagnostic joins those rows to the plot-average table by `FireID`, then uses every record with numeric plot-average `MidFlwind.ms` and `DFMC.%`, positive elapsed time, and nonnegative head distance and period rate. It uses the unchanged FM1 flat-ground head rate times elapsed time, with no fitted speed, initial offset, or acceleration, and compares the same head rate with each period's `ROS.ms`. FM1 rate is set to zero at or above its 12% extinction moisture. The plot-average midflame wind is a **derived** quantity based on measured 2 m wind and a wind profile; it is held constant across each fire here. The real burn-period wind, grass structure, curing, direction changes, and line-ignition acceleration are not reconstructed.

```sh
python3 -m nightjar.annaburroo --output runs/annaburroo-source.json
python3 -m nightjar.annaburroo_temporal --output runs/annaburroo-temporal-fm1.json
```

The script checks both table hashes listed above and writes each fire's timed observed and predicted distance to ignored `runs/`. Its 120 eligible fire plots contribute 576 period observations. Another 123 period rows have no matching plot-average record and are excluded because the chosen midflame-wind input is unavailable: 96 have zero ignition-line length (point ignitions), while 27 report a nonzero line length. There is no selection by prediction error or outcome. The 576 observations are repeated measurements within 120 fires and are **not** 576 independent cases.

| Statistic | Result |
|---|---:|
| Mean observed final head distance | 124.27 m |
| Mean predicted final head distance | 162.52 m |
| Final-distance mean absolute error across 120 fires | 73.81 m |
| Final-distance median absolute error | 51.59 m |
| Signed final-distance bias (prediction minus observation) | +38.25 m |
| Period-point mean absolute error (descriptive only) | 51.81 m |
| Mean of per-fire period-rate MAE | 0.610 m/s |
| ER final-distance MAE, 70 fires | 72.14 m |
| TH final-distance MAE, 50 fires | 76.15 m |

Four plots (`B032`, `B033`, `B041`, `B044`) report 12.1% dead-fuel moisture, just above the fixed FM1 extinction threshold, yet their observed heads traveled 93–116.7 m. They remain in the error calculation rather than being removed as difficult cases. Period rate and cumulative distance are related measurements of the same fronts, not two independent validations. The same 120 fires were used in the plot-average rate diagnostic above; this is a second endpoint on the **same data**, not an independent held-out validation. No uncertainty bounds are supplied for distances or wind conversion. The archived fire-isocrone maps are JPEG images with hand-drawn contours; an A023 map was inspected, but no pixel-to-ground transformation or repeatable contour extraction is established. This check therefore scores the tabulated **one-dimensional head position**, not full perimeters, area, or the 8-neighbour grid solver. Fixed FM1 fails this transfer check and cannot support the first result-bearing training run.

## Diagnosis and next model comparison

Applying one predicted rate from ignition is a likely contributor to the distance error: across the same 120 fires, mean observed `ROS.ms` is 0.542 m/s in each fire's first recorded period and 1.096 m/s in its last, while the fixed FM1 prediction averages 1.175 m/s in both. The first and last periods have different times and are descriptive, not a fitted acceleration curve or proof of a single cause. The final-distance bias is positive (+38.25 m) even though the earlier width-adjusted steady-state rate comparison has negative bias (−0.109 m/s), so correcting the steady-state rate alone need not correct growth from ignition. In the plot-average diagnostic, the four fires at 12.1% moisture contribute a zero FM1 rate despite observed spread; this is a separate fuel-model mismatch. Wind and curing also vary between plots.

The next comparison should keep this unchanged FM1 result as a baseline, then implement a source-matched grassland rate equation with its required **10 m wind**, moisture, curing, and pasture/treatment inputs. [Cheney, Gould, and Catchpole (1998)](https://doi.org/10.1071/WF9980001) provide such a model and describe its low-wind and higher-wind branches. A separate ignition-growth treatment must be tested against the timed observations, with parameters and selection fixed before evaluation. The Annaburroo fires contributed to development of the published grassland model, so a better result on these same fires would be a model comparison, **not independent validation**. A separate grassland dataset and later fuel-appropriate Spanish cases with time-matched perimeters are needed before changing the default simulator or using outcomes to judge aircraft decisions.
