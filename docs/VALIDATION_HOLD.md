# Physical validation on hold

Status recorded **2026-09-30**: physical validation is **on hold by project-owner direction**, not complete or accepted. Preserve the executed experiments and their failures. New physical experiments, source-acquisition campaigns and production propagation replacement are paused. Publishing the preserved research tools, tests and evidence as a draft PR is separate from physical acceptance; it does not resume those experiments. Other research can continue under its explicitly provisional simulator assumptions.

## Preserved evidence and local changes

The work remains on `feature/rxcadre-validation`, based on commit `b3e61e76e67fdd49610fa5f74be5eef6bab25c7d` (merged PR #21). The preserved implementation, tests and public evidence are packaged as a **draft research PR against `main`**, not a production-model substitution or physical-validation acceptance. At the initial local hold checkpoint, before publication preparation, there were **35 pending public files: 11 modified tracked files and 24 untracked files**, with an empty index. The pre-publication byte backup retains that state; raw data and generated reports are excluded from the PR.

| Work retained | Evidence and limits |
|---|---|
| RxCADRE scorer and source/input protocol | Synthetic contract tests exist; original archive/mirror authentication and eligible empirical burns remain unresolved. [Protocol](VALIDATION_RXCADRE.md) |
| Calm analytical propagation benchmark | Eight-neighbour maximum arrival error 7.97%; original level-set 2→1 m pair passes the numerical gates. [Record](PROPAGATION_VERIFICATION.md) |
| CSIRO C064/F19 diagnostic and generic one-class dead-fuel calculations | Separate rate/position/source variants and all failures retained. FM1 behavior is preserved by passing the legacy imperial constants through the generalized calculation; no production propagation algorithm is replaced. [Record](VALIDATION_CSIRO_HEAD.md) |
| FireFlux upper bound and source audit | Conditional 0.6685 m/s bound versus published 1.61 m/s; filtered observations do not resolve wind-height/clock/failure-mask interpretation. [Record](VALIDATION_FIREFLUX.md) |
| Cheney 1998 candidate | Isolated reconstruction; retrospective C064/F19 errors 4.65%/4.25%, no independent acceptance. Original full-text coefficient reconciliation and FireFlux U10/curing are missing. [Protocol](CHENEY_1998_PROTOCOL.md), [record](VALIDATION_CHENEY_1998.md) |
| Extended spatial benchmark | Five homogeneous level-set 2→1 m pairs pass; both layered-fuel pairs and the complete frozen suite fail. Shared constitutive rates do not establish empirical accuracy. [Protocol](PROPAGATION_EXTENDED_PROTOCOL.md), [record](PROPAGATION_VERIFICATION.md#extended-frozen-experiment-2026-09-30) |
| Later independent grass-source audit | No new eligible matched per-burn records acquired; concrete six-site 2018 / provenance-filtered 2022 routes recorded. [Audit](GRASS_INDEPENDENT_EVIDENCE.md) |

Code, tests and affected documentation remain in place. Raw literature/data and result JSON files remain outside Git under ignored `data/` and `runs/`. A complete byte snapshot, file-hash manifest, tracked diff, untracked-file inventory, environment metadata and read-only PR snapshots are stored outside the repository; their location is supplied separately in the local handoff. Keep the worktree and that backup. Publication uses the existing branch and does not discard source files or raw/result artifacts; do not clean, reset, stash or archive this checkout as part of pausing it. The primary checkout is not modified.

Last executed complete software check: `python -m unittest discover -s tests -q`, **233 tests: 230 passed, 3 optional external-data skips**. The extended benchmark's nine focused tests passed and its repeat output matched except creation time. A separate Cheney preservation rerun reproduced every result/configuration except fresh run timestamps. Two older synthetic evaluation reports rewritten by full-suite tests were restored to their pre-run bytes. The initial hold change was documentation-only. Publication preparation also checks the complete software suite, restores any pre-existing generated reports rewritten by tests, and verifies hash preservation, protocol bindings, local links and staged diff hygiene. Software checks do not resume or complete physical validation.

## Open PR snapshot and future integration

The read-only GitHub inspection before publication on 2026-09-30 found these two other open PRs. Both target the same `main` commit above, are individually mergeable and have successful `unit` and `unit-rl` checks. Their CI results apply to their published heads, not to this validation branch. Neither other PR is approved, merged, closed, edited or moved out of draft by publication of this branch.

| PR | Current state / head | Scope and hold disposition |
|---|---|---|
| [#22 — Add traced terrain and weather audits for EMSR578](https://github.com/MrJuandi/nightjar/pull/22) | Open, not draft; `b8aecb00184f0121f45ba1205d3fd58d8f9f1f80` | Terrain ingestion, historical station audit and corrected exploratory satellite interpretation. Keep its current state; separate technical review/integration can proceed without declaring physical validation. |
| [#23 — Assess Annaburroo B111 source timing and intervention limits](https://github.com/MrJuandi/nightjar/pull/23) | Open draft; `4462fb104170e1c21796b9dadf914089b74497de` | Documentation-only source assessment; no accepted observation window or simulation. Keep draft while unresolved timing/scale/intervention limits remain explicit. |

PRs #19–#21 are already merged; their data/interface checks do not close physical validation. PR-state/check/body/file-list inspection and relevant integration diff review are not a new exhaustive runtime/source audit of #22 or #23, nor a merge approval.

**Decision/question identifier collisions resolved before publication:** concurrent branches had assigned different meanings to the same identifiers. This branch updates its decision-table references while preserving every scientific protocol file and recorded hash.

| Before publication (local IDs) | This draft PR | Concurrent identifiers reserved |
|---|---|---|
| D-41–D-46 | D-42–D-47 | #22: D-41 environmental reconstruction |
| O-13–O-18 | O-16–O-21 | #22: O-13 active-front/intervention evidence, O-14 time-varying forcing/moisture; #23: O-15 B111 observation windows |

README, DATA_AND_VALIDATION, DECISIONS, REFERENCES and other public documentation still overlap between branches. GitHub's individual mergeability is not evidence of combined documentary consistency. Before any later integration, recheck the current remote base and both complete diffs, preserve each branch's distinct evidence, verify every cross-reference, and combine the corrected EMSR578 interpretations. No rebase or merge of the other branches is performed for this publication.

## Work that may continue

- Replay inspection, observation/action interfaces, deterministic execution and metadata/CI improvements using existing provisional scenarios.
- Two-aircraft resource bookkeeping, mission assignment and contention experiments under explicitly synthetic aircraft/suppression assumptions. Stage 2 is roadmap work, not implemented by this hold.
- Review of the independent terrain/source-audit PRs, keeping their technical evidence separate from physical acceptance.

Keep reserved splits/checkpoints under their existing protocols; none are opened or retrained here. Controller/synthetic results cannot establish operational suppression effectiveness, independent physical training readiness or justified geographic policy superiority. Aircraft deposition/water-effect and real sensing evidence remain unresolved too. Existing candidate engines stay offline and are not substituted into production or controller training.

## Resume conditions and procedure

Resume only on an explicit decision to return to physical validation. At that point:

1. Recheck branch, local diff/index, current remote `main`/PR heads and the outside-Git backup; retain existing raw sources, protocols and reports. Reconcile concurrent documentation before integration.
2. Select one concrete evidence track: lawful original 1998 full text; matched independent 2018/2022 burn records with provenance/U10/moisture/curing/head-observable metadata; or FireFlux/RxCADRE source interpretation/access. Contact or purchase requires separate authorization; no message or purchase has occurred.
3. If numerical work resumes, separately freeze a finer layered-fuel pair or an isolated surface-distance correction experiment. Preserve today's failures and model/input choices; no retrospective offset fitting or production replacement.
4. Freeze every new model/configuration/conversion/metric/acceptance choice before its new error calculations, disclose prior observation access, use new artifact filenames, run the relevant checks and update the linked evidence records.

The unchanged Cheney frozen JSON has SHA-256 `5e2274f25814e4505ba5b5cc968e3382aac2b3fed83bf9145b793276a54047a3`; the extended spatial freeze has SHA-256 `6f3c92e9946c99f478e03d55026003079079cd4867f010bf58b82a261c83019f`. Their source/runtime bindings still apply. A pause, green CI or passing unit tests cannot turn either experiment into a physical-validation pass.
