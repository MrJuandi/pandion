# Checklist before creating the public GitHub repository

The owner will create the repository after reviewing the local project. No remote repository or Git history is created by this documentation step.

## Ownership and rights

- [x] Confirm project name: Nightjar.
- [ ] Confirm owner identity.
- [ ] Choose a software license; include the actual `LICENSE` file.
- [ ] Confirm rights for any code or assets reused from the old prototype.
- [ ] Review each dataset's license and decide whether to publish data, a small permitted sample, or download scripts only.
- [ ] Add citation metadata with the real human authors and correct affiliations if publication is planned.

## Technical quality

- [ ] README commands match working code; no placeholders presented as implemented features.
- [ ] CI runs meaningful checks once code exists.
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
