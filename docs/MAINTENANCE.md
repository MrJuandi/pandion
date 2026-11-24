# Documentation maintenance

Documentation is part of each change. A change is complete only when the public description of the affected behavior is accurate and the evidence for any new claim is recorded.

## Change map

| Change | Update at the same time |
|---|---|
| Project summary, headline results, or quick start | `README.md` |
| User-visible behavior, setup, or supported commands | `docs/USAGE.md` |
| State, observations, actions, reward, interfaces, or resource dynamics | `docs/ARCHITECTURE.md` |
| Data sources, transformations, assumptions, or validation evidence | `docs/DATA_AND_VALIDATION.md` and `docs/REFERENCES.md` |
| Metrics, baselines, splits, or reported results | `docs/EVALUATION.md` |
| Scope, sequencing, or milestone status | `docs/ROADMAP.md` |
| A settled design choice or a newly discovered open question | `docs/DECISIONS.md` |

Update the relevant document in the same change as the code or experiment. Remove claims that become outdated. Record source, date, version, and limitations for new data or results. If a change genuinely needs no documentation update, explain why in the pull request.

Before merging, check that README and USAGE commands run, links resolve, diagrams match the implementation, and results can be reproduced from the stated artifacts. Review the staged files for local configuration, generated data, credentials, and unintended metadata.
