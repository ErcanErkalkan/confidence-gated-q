# Provenance and Public-Release Boundary

## Independent Implementation

All executable files under `src/`, `scripts/`, and `tests/` were created for
this research artifact using public Gymnasium, MiniGrid, NumPy, pandas, SciPy,
Matplotlib, and PyTorch APIs. The artifact contains only the files needed to
reproduce the reported study; local workspace material, caches, and version-control
metadata are excluded from the clean release package.

## Persistent Identifiers

- Repository: https://github.com/ErcanErkalkan/confidence-gated-q
- Persistent concept DOI: https://doi.org/10.5281/zenodo.20578927
- Version-specific DOI: not minted in this environment; replace the concept DOI in final proofs if a journal-specific Zenodo release is created.

## ASOC Result Lineage

| Result directory | Config | Seeds | Status | Generation commit |
|---|---|---|---|---|
| `results/dqn_tuning_development` | `configs/dqn_tuning_development.json` | 0-4 | rerun; PASS | `050aa5196aa8789a1a06cd3bd9bef41fcc20b784` |
| `results/dqn_strong_validation` | `configs/dqn_strong_validation.json` | 600-629; 700-709 | rerun; PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/confirmatory_extended_compact` | `configs/confirmatory_extended_compact.json` | 500-529 | rerun; PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/support_abstention_replication` | `configs/support_abstention_replication.json` | 300-329; 400-429 | rerun; PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/minigrid_extended_diagnostic` | `configs/minigrid_extended_diagnostic.json` | 500-509 | rerun; PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/application_navigation_case_study` | `configs/application_navigation_case_study.json` | 600-629 | new; PASS | `b292c8d` |
| `results/adaptive_gate_compact_validation` | `configs/adaptive_gate_compact_validation.json` | 700-729 | new; PASS | `b292c8d` |
| `results/cost_support_metrics` | `configs/cost_support_metrics.json` | 800-809 | new; PASS | `b292c8d` |

Each `metadata.json` records the complete config, config SHA-256, requested and
resolved environment IDs, observation representation, platform, package
versions, and generation commit. Each raw row repeats the key software and
experiment provenance.

## Result Preservation

Per-run shards are omitted after aggregation because the committed
`raw.csv.gz` files preserve every public row needed for reaggregation and
auditing. Aggregation and audit scripts read compressed files directly.

Only the eight current ASOC evidence directories are retained in the current
artifact. Superseded exploratory and sensitivity families remain recoverable
from the previous archived release and Git history.
