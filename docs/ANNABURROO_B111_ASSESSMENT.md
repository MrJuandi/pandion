# Annaburroo B111 source assessment

**Status, 2026-09-30:** inspected development candidate for a future spatial propagation diagnostic. No digitized perimeter dataset, spatial comparison, intervention-free window, or validation result has been established.

## Selection and source identity

B111 was selected for source assessment because Gould et al. (2024) illustrate its fuel grid and progression map, and its archive folder contains five timed photographs, a progression map, a fuel map, and corner-wind records. Selection preceded any spatial model prediction; the case is now development material and must not later be presented as an untouched test. It represents treated grass, not a Spanish forest fire.

Source: [CSIRO collection csiro:58746](https://data.csiro.au/collection/csiro:58746), version 1, file ID `64873323`, archive size 593,231,549 bytes, [archive endpoint](https://data.csiro.au/dap/ws/v2/collections/58746/data/64873323). The collection archive is marked CC BY 4.0; individual publication rights remain distinct. Selected members were retrieved by HTTPS byte ranges, with archive-size, range-offset, ZIP-header and uncompressed-size checks. Raw files and the generated source audit remain in ignored `data/raw/` and `runs/`; the existing table downloader was not extended to download images or publications.

Members below are under `Annaburroo experimental fire data/B111/` unless another archive path is given. These are exact source names and SHA-256 hashes, not derived scenario identities.

| Member | SHA-256 |
|---|---|
| `B111_132624.jpg` | `16e112c274867e54d8e152f30acbd26fd0d1129b5195fb8c65a31b647548192f` |
| `B111_132705.jpg` | `1d1caf5019a1f8ea18e1928c1f7614835ec5527bf5a1d3d1aa9dd7df121c014f` |
| `B111_132751.jpg` | `e9db6c62fadbea28ec82adbf0635995869f6bd6ae90fd01cfe43398332f6aadd` |
| `B111_132814.jpg` | `4724f7a643d1a5128808d55a8a9efce7c77a4d5684470a6eae2e1c3c94869680` |
| `B111_132844.jpg` | `97b27fc43220384cb6e7917bfbb888e3b2b96dec1949c6822a3c054d8f008f3f` |
| `B111_2mwind.csv` | `d7fb7293a5fe4bb2b759ed9d46bb6de03d7c6f017dd4cffdc97748b0c9457088` |
| `B111_Fire Isochrones.jpg` | `6a56da4b8e427d68dc39bff4f2529b61ce5150dd977fd0592f80eb91d474d9e2` |
| `B111_Fuel Map.jpg` | `a309086cb3d9b1a47a3e5e0fd460674e2f3314e45a77dbdb4b54bb2f0b6cece6` |
| `ReadMe - datafile descriptions.txt` (archive root) | `29cbe4fda32d66aaee2de3f67b7343a19f27a99ea50e9c796d423f9191ee55b8` |
| `Publications/Cheney et al 1993.pdf` | `b1908e5f2e91fff0663449a8fe8b7b2ecd570aa5c401e4c98dec9b7b89f5975d` |
| `Publications/Cheney et al 1989 technical report.pdf` | `fcd49c675ee18a4a6a7f877e6a482bf1c88a369d012fd4da84ddc558df9c9129` |

The two source-table hashes remain recorded in [VALIDATION_ANNABURROO.md](VALIDATION_ANNABURROO.md). The local source-audit metadata uses schema `nightjar-annaburroo-case-source-audit-0.1`, UTC retrieval timestamp, `seed: null` (no stochastic experiment), explicit source units, and `crs: null` with a statement that plot coordinates/georeferencing are not yet reconstructed. Source clock times have no verified timezone in this assessment; they must not be relabelled UTC.

## What is measured and what is unresolved

The burn-period table dates B111 to 1986-08-06, ignition clock time `13:26:10`, ignition length 50 m, fuel type TH (*Themeda australis*), treatment T3 (cut grass with cut material removed), fuel height 0.15 m, curing 85%, and dead-fuel moisture 7%. Its burn-period fuel load is 0.276 kg/m²; the plot-average table gives 0.286 kg/m². Preserve these different source aggregation values; no replacement or reconciliation has been made.

| Burn-period endpoint | Seconds since source ignition | Source cumulative head travel (m) | Photo/map clock agreement |
|---|---:|---:|---|
| 13:26:41 | 31 | 4 | **Disagreement:** first photo filename and map label are 13:26:24, 14 s after ignition. |
| 13:27:05 | 55 | 13 | Matching label and filename. |
| 13:27:51 | 101 | 32 | Matching label and filename. |
| 13:28:14 | 124 | 51 | Matching label and filename. |
| 13:28:44 | 154 | 85 | Matching label and filename. |

No source timestamp has been silently corrected. The two middle comparisons after a possible 13:27:05 start would span 46 s and 69 s. They are candidates for reconstruction, not approved intervention-free windows. The first interval cannot be treated as a matched photograph/table observation until the conflict is resolved. The progression map is a scanned drawing with cardinal-corner labels and a 109 m width annotation, while the fuel map labels the corresponding width 100 m. This disagreement must be reconciled before assigning metric coordinates; neither value is silently preferred. Neither an EPSG code nor a second-axis scale is inferred from those annotations. Handwritten marks and stroke thickness require interpretation and a recorded digitization uncertainty.

The fuel map legend distinguishes E (Eriachne) and T (Themeda), and sample labels include both, whereas the burn tables assign B111 the aggregate fuel type TH. The mixed sample labels and aggregation require review before treating the plot as spatially uniform Themeda. No species crosswalk or fuel-parameter fit is made in this assessment.

The corner-wind CSV has 32 records labelled `fire`, from 13:26:10 to 13:28:45 at 5 s intervals. These are speeds at 2 m, not a complete wind-vector or midflame forcing series. `FireDir.deg` in the burn table is the observed head-fire travel bearing; using it as wind input would leak an outcome into the prediction. A future reconstruction must obtain an independently measured wind direction, state any height conversion, and audit sensor obstruction rather than assume all four raw instruments are usable.

## Human intervention evidence

Gould et al. (2024) describe free-burning experimental fires, bare-earth breaks between plots, and a pre-burned perimeter buffer. Cheney, Gould and Catchpole (1993), p. 43, explicitly acknowledge staff performing suppression within the experimental area. Their p. 35 also records that nearby firefighting vehicles could obstruct wind measurements. The 1989 technical report foreword, printed p. 2, confirms personnel/equipment for suppression and construction of breaks. The foreword and the relevant 1993 methods/acknowledgment pages were visually inspected; the scanned 1989 report was not fully searched or reviewed.

These statements establish preparation, containment, and suppression activity at experimental-area level. They do **not** give a B111-specific intervention timeline. All five photographs were visually inspected, but have not been exhaustively classified for interventions. A lack of visible water application in a few photographs does not establish that none occurred between photographs or outside their view.

## Requirements for the next experiment

Before a quantitative spatial comparison, reconstruct the chosen fronts in plot coordinates with documented scale, times and uncertainty; resolve or exclude conflicting observations; review all relevant photographs and additional records for ignition completion, suppression and boundary effects; and specify independent weather/fuel inputs. If case-specific suppression remains unknown, retain that limitation explicitly and restrict any result to an exploratory diagnostic, or choose another better documented case. Reaching a bare-earth break cannot be scored as natural spread through unbroken grass.

A later experiment may start from one observed front and compare only later fronts; it must exclude the initial burned area from growth metrics, check grid-resolution sensitivity, and freeze the inputs before inspecting model errors. Any fuel adjustment belongs to development; tests must use separate complete burns with uncertainty and acceptance criteria declared beforehand. Passing such tests would support the represented grass-fire conditions only, not forest, suppression, or aircraft-drop physics.
