# Checklist before publishing the GitHub repository

The owner will decide when to publish the repository after reviewing it. Git history exists locally and an `origin` remote is configured; the repository's public visibility has not been verified here.

## Ownership and rights

- [x] Confirm project name: Nightjar.
- [ ] Confirm owner identity.
- [ ] Choose a software license; include the actual `LICENSE` file.
- [ ] Confirm rights for any code or assets reused from the old prototype.
- [ ] Review each dataset's license and decide whether to publish data, a small permitted sample, or download scripts only.
- [ ] Add citation metadata with the real human authors and correct affiliations if publication is planned.

## Technical quality

- [x] README commands match working code as of 2026-09-27; commands were executed locally. New commands require a fresh check.
- [x] CI runs the standard-library unit suite on push and pull request. External-data tests skip when their raw source is absent.
- [ ] Configuration, seeds, source versions, and expected outputs are documented.
- [ ] No large generated checkpoints, private data, credentials, caches, or local paths are committed.
- [ ] Claims in README match reproducible experiment artifacts.
- [ ] Public issue and contribution templates are added only if maintainers intend to use them.

## Presentation

- [ ] Documentation distinguishes observed, inferred, and synthetic data.
- [ ] Screenshots show actual application behavior and identify the version/scenario.
- [ ] Authorship and project metadata reflect the actual human contributors.
- [ ] Local working instructions and tool configuration are absent from the staged files.
- [ ] Review the final repository diff and metadata before publishing.
