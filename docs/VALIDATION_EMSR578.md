# Candidate independent observations: EMSR578 AOI01

**Status:** source audit, 2026-09-27. This is **not** a validation result for the Nightjar fire model.

[Copernicus Emergency Management Service activation EMSR578](https://mapping.emergency.copernicus.eu/activations/EMSR578/) covers the June 2022 Artesa de Segre wildfire in Catalonia. Its AOI01 delineation products include cumulative, photo-interpreted burned-area polygons. The acquisition times below come from each product's map PDF; areas come from the `observedEventA` GeoJSON inside its vector package. Download URLs and SHA-256 digests are encoded in `nightjar/emsr578.py` and reproduced by the audit report.

| Product | Post-event image acquired (UTC) | Reported cumulative area (ha) | Change from previous (ha) |
|---|---|---:|---:|
| Delineation product | 2022-06-17 10:06 | 1,501.728373 | — |
| Monitoring 01 | 2022-06-18 10:47 | 2,111.212038 | 609.483665 |
| Monitoring 02 | 2022-06-19 10:42 | 2,693.193281 | 581.981243 |

Reproduce the **source audit** from the repository root:

```sh
mkdir -p runs
python3 -m nightjar.emsr578 fetch
python3 -m nightjar.emsr578 report --output runs/emsr578-aoi01.json
```

The script downloads three public versioned ZIP packages to the Git-ignored `data/raw/emsr578/` directory, verifies their reviewed SHA-256 digests, checks the polygon layer and reported area, and writes the dates and source links. If a provider replaces a package, the hash check fails so that its version must be reviewed deliberately. The map PDFs were inspected separately for image acquisition times; **delivery time is not used as observation time**.

The [CEMS On-Demand Mapping terms](https://mapping.emergency.copernicus.eu/terms-and-conditions/) permit access, adaptation, and distribution subject to their terms and attribution. The raw ZIPs and source imagery are kept out of this repository. The report records the source and product URLs; any public redistribution of derived geometries must follow the provider's citation guidance and applicable third-party terms.

## Why these observations do not pass the physical-validation gate

- The mapped burn is in woods and bushes according to the activation description; there is no justified crosswalk from those fuels to Nightjar's FM1 short-grass model.
- Burned-area boundaries are cumulative snapshots from satellite imagery, not a continuously observed active front. The image acquisition intervals are about a day and the polygons have mapping uncertainty.
- Suppression actions, ignitions or spotting, and weather history during those intervals are not represented in the current unmitigated, constant-wind model.
- The first mapped polygon already exceeds 1,500 ha. A validation run would need to start from that perimeter, not a single ignition cell, and use time-aligned fuel, terrain, moisture, and midflame wind inputs.
- Using the first interval to tune the model and the second from the same incident to report an independent geographic test would leak incident-specific information. At least one separate incident must be reserved for evaluation.

Before any model-versus-observation score is reported, build a provenance-preserving pipeline for those inputs, quantify polygon and timing uncertainty, specify the effect of response actions or select sufficiently unmitigated periods, and reserve other incidents. Until then, these three measurements verify only that independent observation artifacts can be acquired and audited.
