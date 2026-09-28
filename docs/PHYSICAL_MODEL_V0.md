# Initial physical-model specification

**Status:** selected starting references, 2026-09-28. A one-size dead-fuel Rothermel/Albini FM1 calculation and a synthetic integrated episode exist. The episode's permanent water barrier is an unsourced test rule, not a validated water-effect model. Published component equations and aircraft figures do not validate Nightjar's incident predictions.

## Surface-fire spread

Use the Rothermel surface-fire model as the first **candidate** rate-of-spread calculation. In its standard form, the forward spread rate is

\[
R = \frac{I_R\,\xi\,(1 + \phi_w + \phi_s)}{\rho_b\,\epsilon\,Q_{ig}},
\]

where `I_R` is reaction intensity, `xi` is the propagating flux ratio, `phi_w` and `phi_s` are wind and slope factors, `rho_b` is oven-dry bulk density, `epsilon` is effective heating number, and `Q_ig` is heat of preignition. This equation gives a local, quasi-steady **surface-fire** spread rate. It does not by itself define a two-dimensional perimeter, ignition timing in neighboring cells, crown fire, spotting, or suppression effects. The implemented FM1 subset takes its factor equations and source imperial units from the cited technical report; `phi_w` and `phi_s` are not arbitrary tuning multipliers. [Rothermel (1972)](https://research.fs.usda.gov/treesearch/32533); [Andrews (2018)](https://research.fs.usda.gov/treesearch/55928).

The current FM1 grid stores fuel class, dead-fuel moisture, elevation, wind, and ignition at one fixed time. It computes directional edge travel times and first arrival with the approximation described in [ARCHITECTURE.md](ARCHITECTURE.md). Before controller comparisons, extend the fuel catalogue and weather treatment, justify the numerical scheme and grid size, and perform independent no-intervention validation. Rothermel's surface model alone does not establish validity for Spanish vegetation or extreme fires.

The legacy `nightjar.fire` module exercises the grid-arrival interface with **synthetic base spread rates and exponential wind/slope factors** described in [ARCHITECTURE.md](ARCHITECTURE.md). Those factors are implementation test assumptions, not coefficients from Rothermel or measured fire behavior.

`nightjar.rothermel` implements Andrews (2018), table 3, for one size class of dead fuel, with Albini's adjustments included in that table. Its only fuel is **Anderson FM1, short grass**: 0.74 short tons/acre of one-hour dead fuel, 1 ft bed depth, 3,500 ft⁻¹ surface-area-to-volume ratio, 12% dead-fuel moisture of extinction, and 8,000 Btu/lb heat content. The table's common particle-density and mineral values (32 lb/ft³, total 0.0555, effective 0.01) are used. [Anderson (1982)](https://doi.org/10.2737/INT-GTR-122); [Andrews (2018), section 3.1.1, table 3, and section 6](https://research.fs.usda.gov/treesearch/55928).

The input wind is **midflame wind**, not a station or forecast wind without transformation. The code calculates the no-wind/no-slope rate in source imperial units and converts the output to m/s. It combines wind and upslope factors by the vector treatment and uses the elliptical point-ignition directional rate from Andrews section 6. The grid's half-edge travel-time and shortest-path treatment is an additional numerical approximation. The figure-16 FM1 reference at 6% moisture and 4 mi/h midflame wind is about 66 ft/min; the implementation gives 66.76 ft/min. This checks one published example approximately, not independent physical validation. No wind-speed cap is applied; this is explicit because the source discusses the cap's effect and warns about underprediction.

For the fixed homogeneous 600 s sensitivity case, affected area was 3.0, 2.0, 1.8125, 1.63, and 1.655 ha at 100, 50, 25, 10, and 5 m respectively (run on 2026-09-27 with `python3 -m nightjar.sensitivity`). The 100 m result differs sharply from finer grids; 10 to 5 m changes by +1.53%. This is evidence for one synthetic case only. The 8-neighbour lattice and full-cell area count still need comparison with a finer or alternative propagation scheme and varied wind, moisture, and terrain before fixing resolution.

The seven-case follow-up (`python3 -m nightjar.sensitivity --varied`, 2026-09-27) changed the 25 m to 5 m affected-area estimate by between −3.94% and +19.05% of the 5 m estimate. The largest percentage difference was the calm case, where only a few cells burned by 600 s. Wind reversal produced mirrored areas; lower moisture increased spread. These cases expose grid dependence and do not establish a universal acceptable resolution. The source audit in [VALIDATION_EMSR578.md](VALIDATION_EMSR578.md) identifies observed perimeters but cannot validate this FM1 model with the currently available inputs.

The [Annaburroo external diagnostics](VALIDATION_ANNABURROO.md) now test the unchanged FM1 head rate and a separate rate-times-time head position against Australian grass experiments. The temporal check finds 73.81 m mean absolute final-distance error across 120 fires and four observed fires above FM1's extinction moisture. These are limitations of this fuel-model transfer; the temporal test does not exercise the grid or constitute a held-out Spanish incident validation.

A generalized one-class Rothermel calculation has also been checked against the FM1 constants and used in an [Annaburroo sensitivity comparison](VALIDATION_ANNABURROO.md). Substituting measured grass load, height and SAV improves the immediate-speed error in the 73 fully cured plots but worsens it sharply in 47 partly cured plots when all fuel is treated as dead. Illustrative ignition-growth curves reduce error on the same fires, but their timescales are not validated and are not used by the grid simulator.

An explicit `CURED_GRASS` fuel configuration now exposes measured load, grass height as bed depth, and measured species SAV to the standalone and integrated grid solvers. The [example scenario](../examples/cured_grass_fire.json) uses A023 values (0.403 kg/m², 0.23 m, 97.7 cm⁻¹, 7.4% dead moisture, 4.6 m/s derived midflame wind) on synthetic geometry. Its 12% extinction moisture and remaining particle properties are FM1 assumptions, not Annaburroo measurements. The grid still assumes immediate local quasi-steady spread; the exploratory ignition-growth curves are not installed. This configuration is restricted to declared 100% cured fuel and is not a validated Spanish or Australian incident replay.

## First aircraft: CL-415

Model one water-scooping CL-415. The figures below are from a Canadian government airtanker guide describing typical operating conditions; they are **indicated airspeeds**, not ground speeds or universal operating limits. The guide warns that its elicited operating figures may differ from manufacturer specifications. [McFayden et al., *Reference Guide to the Drop Effectiveness of Skimmer and Rotary Wing Airtankers* (2023), section 6.1](https://publications.gc.ca/collections/collection_2023/rncan-nrcan/Fo123-2-35-2023-eng.pdf).

| Quantity | Reference figure | Initial treatment |
|---|---:|---|
| Water tank capacity | 6,137 L | Upper bound; partial loads and drops remain possible. |
| Cruise, full / empty | 150 / 165 KIAS | Reference only; derive or declare a separate ground-speed approximation before simulating travel. |
| Typical drop speed | 105–115 KIAS | Reference for drop phase, not instant travel speed. |
| Typical drop height | 100–150 ft AGL | Reference for a later footprint model. |
| Water pickup | Scooping aircraft | Requires a suitable waterbody and a modeled approach, pickup, and departure. |

An operator specification independently lists the same 6,137 L tank capacity. [Avincis CL-415 specification](https://www.avincis.com/wp-content/uploads/2024/11/A4-CL-415-Specifications-Flyer-Low-Res.pdf).

The Canadian guide also uses a **158 KIAS blended speed plus a 3.0-minute increment per cycle** to estimate drop frequency as a function of fire-to-water distance (section 7.2). This is a useful cycle-time comparison target, not a direct movement rule: KIAS must not be inserted into `distance / speed` for ground travel without an explicit airspeed-to-groundspeed assumption. The guide reports that its simple model follows the observed trend but underestimates some observed CL-415 drop frequencies.

## Model boundaries before the first simulation

1. **Travel:** select a documented conversion or an explicitly synthetic ground-speed proxy, and account for load state and elapsed time. Do not present the proxy as a CL-415 performance specification.
2. **Pickup:** record waterbody eligibility and all approach/service/departure time. The previously suggested fixed five-minute refill is withdrawn; it was not sourced and would misrepresent scooping.
3. **Drop:** conserve water volume. Define footprint and deposited depth separately from the response of burning fuel. The Canadian guide contains measured footprint summaries, but transfer to another canopy, weather, or fuel requires validation.
4. **Fire response:** do not equate litres dropped with hectares extinguished. A water-effect model needs a stated mechanism, persistence, and sensitivity range before policy comparisons.
5. **Evaluation:** keep scenario, simulator, observation, action, and replay interfaces separate; compare unmitigated spread with independent observations before claims about suppression.

The standalone aircraft slice is a **synthetic kinematics and accounting episode**: one CL-415-shaped resource travels between a target and a declared source, consumes time, transfers at most 6,137 L, and produces a deterministic replay. Its `50/55 m/s` ground speeds and `180 s` refill service time in that example are deliberately synthetic test inputs, not converted CL-415 airspeeds or measured pickup timings. Waterbody eligibility and approach remain unmodeled. The integrated example reuses this executor with a synthetic 60 s refill and the 35 m permanent cell-barrier rule described in [ARCHITECTURE.md](ARCHITECTURE.md). That rule is deliberately provisional; a sourced drop footprint, dose, persistence, and effect on burning fuel remain to be specified and validated.
