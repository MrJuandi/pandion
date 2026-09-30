# Wind, slope and layered-fuel numerical verification protocol

Scientific choices fixed on 2026-09-30 before executing the new arrival-error calculations. This is a synthetic numerical experiment, not empirical validation. Existing calm-case results and Cheney grass-rate comparisons have already been observed. Keep the earlier benchmark, equations, protocols and reports unchanged. No production propagator is replaced.

## Inputs and discretizations

Use Pyretechnics **2026.8.10**, NumPy 1.26.4 and the existing Scott & Burgan model 101 moisture fractions: dead 1 h 0.04, dead 10 h 0.05, dead 100 h 0.07, live herbaceous 0.30, live woody 0.60. Dynamic curing follows the package's constitutive implementation. Canopy cover/height/base/density are zero, foliar moisture is 0.60, spread adjustments are one, wind limiting is enabled, ellipse length/width uses `behave`, and spotting is disabled. Seed 0 is metadata: no stochastic model is invoked. CRS is `LOCAL_METERS`, x east and y north. The synthetic time origin is 2000-01-01T00:00:00Z; arrivals are elapsed seconds.

The native level-set engine receives **open 10 m wind in km/h**, as documented by its installed source. Its wind conversion to 20 ft and then midflame is reproduced from the same input API for the Dijkstra comparison and analytical constitutive reference. This is a numerical alignment within that package, not a field-supported conversion for FireFlux or a Cheney-model conversion. Persist both open and midflame speeds. Wind direction is meteorological **from**, clockwise from north. Aspect is downslope, clockwise from north; slope is rise/run. Surface distance and horizontal map distance are distinct.

| Point case | U10 (km/h) | Upwind (degrees) | Slope | Aspect (degrees) |
|---|---:|---:|---:|---:|
| Calm control | 0 | 0 | 0 | 0 |
| East wind | 12 | 270 | 0 | 0 |
| Northeast wind | 12 | 225 | 0 | 0 |
| East-rising plane | 0 | 0 | 0.5 | 270 |
| North wind on east-rising plane | 12 | 180 | 0.5 | 270 |

Use grids of **4, 2 and 1 m**, a domain extending 120 m each side of the origin, and one ignited cell at (0,0). All cases retain targets (40,0), (40,20), (40,40) and (80,0) m. Point-cell initialization error is included against the mathematical point reference; do not subtract an ignition radius or calibrate a delay. Retain unreachable targets and execution failures. Native integration uses `max_cells_per_timestep=0.4`, buffer width 3, no scheduled spot ignitions. Horizon is **1.25 times** the maximum input-derived reference arrival; it must not depend on a measured numerical arrival.

## Independent spatial reference

Obtain local homogeneous head speed h, direction d, eccentricity e and length/width L directly from the package's **input-only** fuel and surface-behavior API. No arrival or output spread-rate raster is used to infer these quantities. The local Rothermel law is common to both solvers; this does not independently verify that law against real fires.

The constant-medium wavefront on the slope plane is an ellipse with semiaxes a t and b t and centre displaced c t along d, where a=h/(1+e), b=a/L and c=a e. Project a horizontal target into the terrain plane: q=(x,y,gx x+gy y), gx=−slope sin(aspect), gy=−slope cos(aspect). Set p=q·d and v²=q·q−p². The positive solution of

`A t² + B t − C = 0`, with `A=1−e²`, `B=2pc/a²`, `C=p²/a²+v²/b²`,

is its analytical arrival time. Use the stable positive root; the circle limit is Euclidean **surface** distance/h. This reference derives from the ellipse geometry, separately from the finite-difference update and eight-neighbour routing. Check circle, heading, backing and tilted-plane examples independently in software tests.

Run Dijkstra through the existing `run_fire_standard` with the same midflame wind and planar elevation. Its horizontal edge distance and slope-plane directional rate conventions are part of the measured solver behavior; do not repair production in this experiment.

## Heterogeneity reference

Two additional flat, calm cases use a front ignited along the entire x=0 grid column. Sample a continuous nominal interface at **x=40 m**: left model 101/right model 104, and the reverse order. Use the same moisture, domains and three cell sizes. All targets are y=0, x=20,60,80 m. For independent input-only still-air speeds Rleft and Rright,

`t(x)=min(x,40)/Rleft + max(x−40,0)/Rright`.

This is a one-dimensional layered-medium reference: lateral travel cannot improve a path through layers that are uniform in y. Cell-centre sampling places the discrete interface at 40−cell/2 m; that interface approximation and finite ignited-column extent are included in numerical error against the nominal continuous problem. Neither receives an a posteriori correction. These cases test a fuel discontinuity; they do not verify arbitrary two-dimensional refraction, moisture heterogeneity, temporal weather or nonburnable barriers.

## Acceptance, preservation and reproducibility

Keep the previous strict analytical-arrival error **<5%** and adjacent-grid arrival-change **<5%**, normalized by the fine-grid analytical arrival. A grid pair passes only when every target in both resolutions passes the analytical gate and their refinement change passes. Report every case and engine separately; no pooled mean hides a failure. The overall numerical gate includes all resolutions and may fail even if a fine-grid pair passes.

As a separate domain probe, repeat 1 m east-wind and both layered cases with extent 160 m. Target arrival changes must be **<1%** of the same analytical reference. Retain any domain dependence rather than selecting the favorable domain. This probe does not modify main-case results or thresholds.

Freeze the executable configuration, this document, code hashes, package versions, units, CRS, seed and UTC timestamp in an exclusively created JSON before running arrivals. Verify those bindings before every run. Exclusively create output reports under ignored `runs/`; never overwrite earlier reports or downloaded data. Existing artifact bytes are backed up before this continuation. Unit tests do not establish physical acceptance; `physical_validation_pass` remains null regardless of numerical results.

Commands from the worktree root, once the implementation is present:

```sh
python -m nightjar.propagation_extended_verification freeze --protocol runs/propagation-extended-protocol.json
python -m nightjar.propagation_extended_verification run --protocol runs/propagation-extended-protocol.json --output runs/propagation-extended-verification.json
python -m unittest discover -s tests -p 'test_propagation_extended_verification.py' -v
```

The original report commands and frozen grass-law protocol are unaffected. Execution evidence and limitations belong in [PROPAGATION_VERIFICATION](PROPAGATION_VERIFICATION.md). Source conventions are verified against installed Pyretechnics 2026.8.10 `burn_cells.py`, `surface_fire.py` and `eulerian_level_set.py`; the report records hashes for these reference files as well as repository code.
