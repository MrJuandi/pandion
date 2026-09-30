# Independent grass-fire evidence audit

Access audit: **2026-09-30**. The current Cheney candidate and its original retrospective error calculations remain unchanged. This continuation seeks evidence outside the 1986 development campaign; it does not label newly encountered papers or data as blind observations. Published summaries and some outcomes were read during this audit. No empirical inputs were tuned, invented or inferred from rate errors.

## Original-paper access and equation reconciliation

Crossref's record for [10.1071/WF9980001](https://api.crossref.org/works/10.1071/WF9980001) identifies the publisher's original PDF at [ConnectSci](https://connectsci.au/wf/article-pdf/8/1/1/155519/wf9980001.pdf). Direct retrieval returned **HTTP 403**, including the DOI and legacy CSIRO PDF routes. The publisher's indexed landing page is still an abstract. OpenAlex's record supplies no open full-text location; this is a search result, not proof that no lawful copy exists. The ResearchGate author page did not provide an acquired original PDF. No access restriction was circumvented.

Therefore the 0.715/0.705 cut-pasture discrepancy, equality conventions and direct original-paper domain reconciliation remain open. The candidate is still explicitly a reconstruction. This audit does not change [CHENEY_1998_PROTOCOL](CHENEY_1998_PROTOCOL.md), its code hashes or existing results. A future original-paper correction must be a separately versioned model comparison retaining the current candidate.

The 2018 primary experiment [Cruz et al., *Got to burn to learn*, DOI 10.1071/WF18082](https://www.researchgate.net/publication/327982732_Got_to_burn_to_learn_the_effect_of_fuel_load_on_grassland_fire_behaviour_and_its_management_implications), Methods and Operational model, independently restates the undisturbed branches with U10 ≤5 and U10 >5. It uses profile-supported U2/U10=0.81 and weather-estimated dead moisture, matching its account of the 1998 parameterisation. That supports provenance, not transferring the profile to Texas or replacing measured CSIRO inputs after seeing errors. The cut/eaten-out coefficient dispute is not resolved by this natural-pasture restatement.

The normal browser subsequently passed the publisher's automatic security check and reached the abstract/paywall. It explicitly states that the content is PDF-only and unavailable to the current visitor, offering institutional sign-in or **USD 40** pay-per-view. No account, purchase or CAPTCHA action was performed. This confirms a full-text access impediment beyond the scripted HTTP 403; a supplied lawful copy or institutional access could resolve it.

## Campaign eligibility

| Source | Separation from 1998 development | Available evidence | Current disposition |
|---|---|---|---|
| Annaburroo 1986 / C064 / F19 | No: development campaign | Existing complete source tables and timed fronts | Keep retrospective diagnostic only; more burns from this archive would not make an independent test |
| Cruz et al. 2017/2018 six-site experiments, 58 burns | Later experimental program; promising independent campaign for the unmodified 1998 law | Full author restatements give methods, aggregate statistics, wind-profile conversion and fully cured plots; individual matched rows were not acquired | Not executable as a new comparison. Request per-burn forcing, rates and front-width/steady-state evidence. Published aggregate errors are external evidence, not results of the current reconstruction |
| Cruz, Alexander, Kilinc 2022 wildfire compilation, 58 retained runs | Mixed: includes 24 observations from the 1998 paper before filtering | Acquired full PDF, methods and aggregate results; per-run dataset offered under request | Reject treating all 58 as independent. Need run identifiers/dates and provenance to exclude original-paper runs, then matched input rows and uncertainty |
| Braidwood S03 / E06, later point ignitions | Later campaign | Acquired 2023 study's preprint describes isochrones and wind at **2 m**, 35 m upwind; public code repository inspected | No reconciled U10/dead moisture/curing record acquired. Point growth and variable wind also do not directly observe potential wide-head spread |
| Athanasiou 2021 Mediterranean grass observations, N=20 | Geographically separate | Acquired full conference paper: midflame-wind model and descriptive summaries | No per-observation U10, moisture and curing contract acquired. Do not invert a generic wind adjustment or use summary means as individual burns |
| FireFlux I / RxCADRE | Independent US campaigns | Existing access and source audits | Preserve wind/curing/source interpretation and authentication blockers; no new physical inputs from filtered instrument columns |

The [2022 paper](https://doi.org/10.3390/fire5020055), section 2 and Data Availability Statement, makes the mixed provenance and request requirement explicit. Its 58 retained runs cannot be identified solely by figures. Digitising a rate-versus-wind scatter would leave curing, pasture, weather-estimated moisture, event identity and source overlap unresolved. No such pseudo-dataset is created.

The [2018 six-site primary experiment](https://doi.org/10.1071/WF18082) reports original-law MAE **19.3 m/min** and signed mean error **+1.28 m/min** across its experiments; it also reports overprediction at high fuel loads. These are **authors' reported aggregate results**, not reproduction by Nightjar, not per-burn acceptance, and not evidence that the current candidate inherits those exact errors. Do not adopt that paper's fitted fuel-load correction in this task.

For Braidwood, [Dabrowski et al., arXiv:2212.00970v2](https://arxiv.org/html/2212.00970), section 4.2.1, explicitly states the 2 m wind height. Section 4.2.2 uses an enlarged initial ellipse for S03 to account for acceleration. The [authors' public repository](https://github.com/jjdabr/BPINN-Wildfire) publishes implementation code, not a complete matched empirical input package. Source paper [Sullivan et al. 2018](https://doi.org/10.14195/978-989-26-16-506_71) was located but its repository endpoint did not resolve during this audit. Do not describe it as acquired or independently reanalysed.

For Mediterranean observations, [Athanasiou 2021 full paper](https://eclass.duth.gr/modules/document/file.php/425447/%CE%A5%CE%BB%CE%B9%CE%BA%CF%8C%20%CE%B5%CF%81%CE%B3%CE%B1%CF%83%CF%84%CE%B7%CF%81%CE%AF%CE%BF%CF%85/MiltiadisAthanasiou_Grassfires_ROS.pdf) distinguishes midflame wind from 10 m meteorological wind. That distinction is an exclusion reason, not authorization to silently convert it.

## Concrete data request specification

The most direct next empirical route is the six-site program's **individual burn records**, followed by a provenance-filtered post-1998 wildfire subset. A request is not sent automatically. A reviewable request would ask for:

- Stable burn/run identifiers, date, site, source campaign and whether records entered the 1998 development or wildfire evaluation.
- Open 10 m wind speed, direction, averaging window, measurement height/exposure, station geometry, and measured profile or documented site conversion where needed.
- Contemporaneous dead-fuel moisture samples (% oven-dry mass), sample definition/timing, and temperature/relative humidity for a separately predeclared moisture-estimation comparison.
- Curing fraction/method, pasture treatment, fuel continuity, load and height; topography and interventions.
- Measured head rate with exact interval/definition, front width and acceleration/steady-state information; arrival/front coordinates if available, clock origin, CRS and measurement uncertainty.
- Licence and version of the supplied files, instrument failure masks and units.

Before opening a new rate table, freeze burn selection by provenance and input completeness, model version, conversions, observation definition, metrics and failure retention. Reserve whole independent burns; do not split intervals of one burn across development and evaluation. An author's already published aggregate error prevents claims of completely unknown campaign outcomes, even if individual records have not been opened.

## Artifacts and conclusion

Downloaded PDFs and source-access metadata stay under ignored `data/raw/cheney1998-followup/`. Acquired preprint SHA-256: `e6edb1ab39b501bd7f59206e6d1b883cbc8ab91a95902a7c44802c345fd510db`; Athanasiou PDF SHA-256: `74e72313e7446878c6f67e04470da421ffcb93af7f4c235b9bef54a4196b4926`. Publisher Crossref metadata and the failed original-PDF request are saved separately; no hash for a nonexistent original PDF is claimed. Readable author-uploaded HTML is cited by DOI/version/locator, not passed off as acquired PDF bytes.

**No newly eligible independent empirical burn has been established in this audit.** This is a concrete source/input impediment, not a model success or a family-wide failure. Synthetic wind/slope/layer tests in [PROPAGATION_EXTENDED_PROTOCOL](PROPAGATION_EXTENDED_PROTOCOL.md) proceed independently of this access limitation and cannot close the empirical gate.
