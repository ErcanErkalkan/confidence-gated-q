# Changelog

## Unreleased

- Prepared the `v1.4.0` strong ASOC revision.
- Added a controlled application-navigation support-shift environment.
- Added a fuzzy support- and uncertainty-aware adaptive gate baseline.
- Added unsupported-state, branch-use, abstention, exact-memory,
  collision/risk, and per-decision timing metrics.
- Completed 730 new full method--environment--seed runs across application,
  adaptive-gate, and cost/support families; all family audits passed.
- Added generated ASOC tables, vector figures, claims provenance, and a
  strong-revision audit.
- Retargeted the journal-specific manuscript assets and submission package to
  *Applied Soft Computing* and removed superseded journal names and duplicates.
- Removed ten superseded exploratory and sensitivity result families that are
  not used by the manuscript, figures, tables, or current reproduction flow.
- Expanded the public reproduction and audit workflow to eight ASOC evidence
  families.

## 1.3.0 - 2026-06-07

- Reframed the work as a reproducible boundary analysis rather than a broad
  algorithmic superiority claim.
- Added seven-task development-only DQN tuning and independent validation.
- Added a 30-seed, seven-task confirmatory compact expansion.
- Added independently seeded support-abstention replication on three
  held-out-goal FourRooms sizes and MiniGrid Empty-5x5.
- Added six fully observable MiniGrid diagnostics.
- Added explicit vanilla and Double-DQN behavior with target-selection tests.
- Added Taxi version compatibility, environment and observation provenance,
  success-rate AUC, robust outlier diagnostics, and cross-environment ranks.
- Expanded the test suite from 13 to 22 tests.
- Added lossless compressed raw-CSV support for public hosting.
- Regenerated the manuscript, tables, vector figures, and submission package.
- Archived the previous reproducibility artifact at
  https://doi.org/10.5281/zenodo.20581705.

## 1.2.0 - 2026-06-06

- Published the reproducibility artifact on Zenodo:
  https://doi.org/10.5281/zenodo.20578928.
- Published the source repository:
  https://github.com/ErcanErkalkan/confidence-gated-q.
- Added complete author affiliation, email, and ORCID metadata.
- Added post hoc support-abstention, count-scale, and DQN sensitivity results.
- Released the final exact-step, RNG-isolated result sets and audits.

## 1.1.0 - 2026-06-06

- Fixed nominal evaluation-checkpoint overshoot.
- Isolated evaluation from training RNG and exact-state support.
- Separated training runtime from evaluation overhead.
- Reran all confirmatory and external experiments under the corrected protocol.
- Added paired median, win-rate, Wilcoxon-Holm, and sign-Holm analyses.
- Reframed claims around heavy-tailed risk and held-out-support failure.
- Added support-abstention and compact hyperparameter sensitivity studies.

## 1.0.0 - 2026-06-05

- Added exact environment-step budgeting and resumable run shards.
- Added structured FourRooms held-out-goal environments.
- Added TD-residual reliability gating.
- Completed five-seed development, 30-seed confirmation, and ten-seed
  MiniGrid external validation.
- Added paired bootstrap intervals, Cohen's dz, rank-biserial effect size, and
  Holm-adjusted planned contrasts.
- Added result audits and a reproducible code-and-data release.
- Finalized the claim as a bounded positive result with explicit failure cases.
