# MLWA Major Revision Log

## Baseline

- Started: `2026-06-07T13:29:05+03:00`
- Working branch: `mlwa-major-revision`
- Baseline commit: `2cd433eb1437f1fce6686ea7a7e234737086d01d`
- Baseline worktree: clean
- `python -m pytest`: **PASS**, 13 tests
- `python scripts/reproduce_all.py --quick`: **PASS**
- `python scripts/audit_artifact.py --root . --output artifact_audit.json`:
  **PASS**
- `python scripts/audit_submission.py`: **PASS**

The quick reproduction reaggregated and audited existing committed raw results.
It did not run new experiment families.

## Revision Scope

The revision reframes the work as a reproducible boundary analysis of
exact-state support, tabular-neural gating, and neural extrapolation failure.
It does not claim general reinforcement-learning superiority.

## Modified Files

Pending. This section will be finalized after implementation.

## Experiments Added

- `configs/dqn_tuning_development.json`
- `configs/dqn_strong_validation.json`
- `configs/confirmatory_extended_compact.json`
- `configs/support_abstention_replication.json`
- `configs/minigrid_extended_diagnostic.json`

## Experiments Rerun

- `dqn_tuning_development`: **completed**, 245/245 run shards, development
  seeds `0-4`, 92,690 raw rows, audit `PASS`.
- The frozen selection rule chose `vanilla_dqn_buffer100k` with mean
  environment rank `2.785714`. No validation or confirmatory seed informed
  this choice.
- Remaining full experiment families are in progress.

## Results Regenerated

At baseline, all pre-revision result families were reaggregated and audited.
No new numerical result has yet been inserted into the manuscript.

## Manuscript Sections Revised

Pending.

## Limitations and Failed Commands

- `Taxi-v3` is not currently registered by the installed Gymnasium build.
  The revision will either register the available implementation cleanly or
  record Taxi as unavailable; it will not silently substitute another task.
- Full expanded experiment families may exceed the practical compute budget of
  this revision session. Any family not fully rerun will remain explicitly
  marked `not yet rerun`, with smoke output stored separately.

## Current Recommendation

`NOT_READY_MAJOR_GAPS_REMAIN`
