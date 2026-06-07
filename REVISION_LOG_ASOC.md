# ASOC Major Revision Log

## Baseline

- Started: `2026-06-07T13:29:05+03:00`
- Target journal: *Applied Soft Computing*
- Baseline commit: `2cd433eb1437f1fce6686ea7a7e234737086d01d`
- Baseline worktree: clean
- Baseline tests: 13 passed
- Baseline quick reproduction: `PASS`
- Baseline artifact audit: `PASS`
- Baseline submission audit: `PASS`

## Completed Changes

### Code

- Added Gymnasium Taxi-v3 to Taxi-v4 compatibility with requested and resolved
  IDs recorded separately.
- Added explicit vanilla-DQN and Double-DQN target behavior.
- Added unit tests proving online action selection and target evaluation are
  separated for Double DQN.
- Added environment and observation-encoding tests for FrozenLake 4x4/8x8,
  CliffWalking, Taxi, and five MiniGrid variants.
- Added config-loading tests for all ASOC revision families.
- Added raw-row experiment, config, environment, observation, timing, commit,
  package, Python, PyTorch, NumPy, Gymnasium, and MiniGrid provenance.
- Added success-rate AUC, reproducible bootstrap intervals, median
  differences, win/loss/tie counts, Wilcoxon sensitivity, heavy-tail
  diagnostics, catastrophic outlier counts, and cross-environment ranks.
- Added lossless `raw.csv.gz` support for the frozen ASOC evidence families.
- Made result audits work both with resumable shards and clean-clone raw data.

### Configurations

- `configs/dqn_tuning_development.json`
- `configs/dqn_strong_validation.json`
- `configs/confirmatory_extended_compact.json`
- `configs/support_abstention_replication.json`
- `configs/minigrid_extended_diagnostic.json`
- `configs/application_navigation_case_study.json`
- `configs/adaptive_gate_compact_validation.json`
- `configs/cost_support_metrics.json`

### Results

- All eight ASOC revision families were fully executed.
- No smoke result was substituted for a full result.
- The eight manuscript result families were preserved without changing
  numerical outcomes; superseded exploratory families were removed later.
- All eight current result-family audits pass.
- Public raw data are losslessly compressed where appropriate.

### Manuscript

Revised:

- title and abstract;
- Introduction and contribution list;
- ASOC relevance;
- Related Work;
- estimator, gate, abstention, encoding, evaluation, and provenance methods;
- development/validation/confirmation/replication design;
- RQ1-RQ7 Results;
- Discussion;
- standalone Limitations;
- reproducibility and declarations;
- conservative Conclusion.

The manuscript now states that exact-state gating is support dependent, is not
a generalization mechanism, and is not a generally superior RL algorithm.

### Figures and Tables

Generated under `paper/figures/` and `paper/generated/`:

- compact and MiniGrid learning-curve figures;
- support-boundary and neural-extrapolation figure;
- DQN development/validation figure;
- extended benchmark rank figure;
- execution-status, compact-result, replication, MiniGrid, and DQN-validation
  tables;
- machine-readable `verified_claims.csv`.

### Submission Files

Updated:

- `paper/manuscript.tex` and `paper/manuscript.pdf`
- `paper/title_page.md`
- `paper/highlights.txt`
- `paper/cover_letter.md`
- `paper/declarations.md`
- `paper/CONFIRMATORY_PROTOCOL.md`
- `paper/RESULTS_SUMMARY.md`
- `paper/LITERATURE_REVIEW.md`
- `submission_clean_asoc/`
- `submission_clean_asoc.zip`
- `submission_clean_asoc.zip.sha256`
- `submission_audit.json`
- `submission_audit.md`

## Experiment Status

| Config | Output | Seeds | Status | Raw rows | Audit |
|---|---|---|---|---:|---|
| `dqn_tuning_development.json` | `results/dqn_tuning_development` | 0-4 | completed | 92,690 | PASS |
| `dqn_strong_validation.json` | `results/dqn_strong_validation` | 600-629; 700-709 | completed | 190,902 | PASS |
| `confirmatory_extended_compact.json` | `results/confirmatory_extended_compact` | 500-529 | completed | 663,483 | PASS |
| `support_abstention_replication.json` | `results/support_abstention_replication` | 300-329; 400-429 | completed | 359,339 | PASS |
| `minigrid_extended_diagnostic.json` | `results/minigrid_extended_diagnostic` | 500-509 | completed | 215,786 | PASS |
| `application_navigation_case_study.json` | `results/application_navigation_case_study` | 600-629 | completed | 81,922 | PASS |
| `adaptive_gate_compact_validation.json` | `results/adaptive_gate_compact_validation` | 700-729 | completed | 234,834 | PASS |
| `cost_support_metrics.json` | `results/cost_support_metrics` | 800-809 | completed | 37,081 | PASS |

Total revision runs: 4,145 method-environment-seed runs.

## Statistical Status

| Claim | Source | Metric | Effect and interval | Adjusted p | Status |
|---|---|---|---|---|---|
| Count gate improves validated DQN on FrozenLake 4x4 | `confirmatory_extended_compact/planned_contrasts.csv` | return AUC | +0.316 [0.268, 0.365] | t-Holm 7.78e-12 | confirmatory task expansion |
| Count gate improves validated DQN on FrozenLake 8x8 | same | return AUC | +0.027 [0.014, 0.041] | t-Holm 0.0126 | confirmatory task expansion |
| Count gate improves validated DQN on CliffWalking | same | return AUC | +143.328 [113.611, 173.089] | t-Holm 6.53e-9 | confirmatory task expansion |
| Count gate is worse than tabular on held-out FourRooms 7/9/11 | same | return AUC | -1.149 / -1.231 / -1.584 | all mean Holm significant | confirmatory task expansion |
| Abstention repairs FourRooms 7 | `support_abstention_replication/planned_contrasts.csv` | return AUC | +1.164 [1.117, 1.211] | t-Holm 6.06e-28 | replication after discovery |
| Abstention repairs FourRooms 9 | same | return AUC | +1.190 [1.142, 1.238] | t-Holm 7.98e-28 | replication after discovery |
| Abstention repairs FourRooms 11 | same | return AUC | +1.625 [1.560, 1.689] | t-Holm 4.54e-28 | replication after discovery |
| Abstention is neutral vs count on replicated MiniGrid Empty-5x5 | same | return AUC | +0.019 [-0.034, 0.070] | t-Holm 1.0 | replication after discovery |
| Count improves DQN on MiniGrid Empty-8x8 | `minigrid_extended_diagnostic/planned_contrasts.csv` | return AUC | +0.596 [0.442, 0.750] | t-Holm 0.000761 | post hoc diagnostic |
| Validated DQN is not universally better than original DQN | `dqn_strong_validation/planned_contrasts.csv` | return AUC | task dependent | no Holm-significant task | independent validation |
| Fuzzy improves DQN in application navigation | `application_navigation_case_study/planned_contrasts.csv` | return AUC | +0.575 [0.152, 0.987] | t-Holm 0.0256 | confirmatory application case |
| Fuzzy does not improve count gating in application navigation | same | return AUC | +0.211 [-0.171, 0.626] | t-Holm 0.318 | confirmatory application case |
| Fuzzy improves DQN on FrozenLake 4x4 | `adaptive_gate_compact_validation/planned_contrasts.csv` | return AUC | +0.291 [0.244, 0.338] | t-Holm 1.12e-11 | adaptive-gate validation |
| Fuzzy fails at complete FourRooms support shift | same | return AUC | -1.195 vs tabular | all 30 pairs lose | adaptive-gate validation |

## Reviewer-Risk Checklist

1. **Is DQN sufficiently validated?** Yes for the bounded claim. Seven
   development candidates were compared using only development seeds, and the
   selected candidate was tested on independent seeds. No universal DQN
   improvement is claimed.
2. **Are claims conservative?** Yes. The paper explicitly rejects broad RL
   superiority and generalization claims.
3. **Is support abstention still only post hoc?** The discovery remains post
   hoc, but its FourRooms repair is independently replicated and labeled
   confirmation after diagnostic discovery.
4. **Are broader tasks included?** Yes: seven compact tasks, six MiniGrid
   diagnostics, and a controlled application-navigation case.
5. **Are negative results reported?** Yes, including FourRooms support failure,
   tabular superiority, MiniGrid failures, and DQN tuning sensitivity.
6. **Is ASOC relevance explained?** Yes, through fuzzy arbitration,
   uncertainty-aware support signals, navigation decision support, and
   reproducible failure-boundary analysis.
7. **Are all files reproducible?** Yes. Configs, raw data, summaries,
   provenance, tests, generation commands, and audits are present.
8. **Are remaining research-package pieces missing?** The `v1.4.0` source and
   packages are prepared. A new version-specific Zenodo DOI remains an external
   account action; the persistent concept DOI is unchanged.

## Validation

- Tests: 28 passed.
- Result audits: 8 passed.
- Artifact audit: PASS.
- ASOC strong-revision audit: PASS.
- LaTeX: complete `pdflatex/bibtex/pdflatex/pdflatex` build passed.
- LaTeX log: no overfull boxes, undefined citations, or undefined references.
- PDF: 26 pages.
- Submission audit: PASS.
- Submission ZIP SHA-256 is recorded alongside the generated ZIP.

## Failed Commands and Resolutions

- The first eight-family quick reproduction found stale `config_sha256`
  fields in the five legacy metadata files. Their embedded configs already
  matched the frozen JSON files exactly; only those five hash fields were
  corrected, with no result-row or numerical change.
- Earlier legacy CSVs were migrated without numerical changes during the
  revision. Those superseded result families are no longer part of the current
  manuscript artifact.
- The first package audit found a prohibited promotional phrase in a
  "claims not made" list. It was replaced with neutral wording and the clean
  package audit passed.

## Final Recommendation

`READY_AFTER_EXTERNAL_ASOC_UPLOAD_ITEMS`
## Soft-computing motivation update

- Strengthened the Introduction with an explicit soft-computing motivation.
- Added a reliability-under-partial-support framing for count gating, DQN extrapolation, and support abstention.
- Added a foundational soft-computing citation.
