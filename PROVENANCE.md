# Provenance and Public-Release Boundary

## Independent Implementation

All executable files under `src/`, `scripts/`, and `tests/` were created for
this research artifact using public Gymnasium, MiniGrid, gym-pybullet-drones,
PyBullet, NumPy, pandas, SciPy, Matplotlib, and PyTorch APIs. The artifact
contains only the files needed to
reproduce the reported study; local workspace material, caches, and version-control
metadata are excluded from the clean release package.

## Persistent Identifiers

- Repository: https://github.com/ErcanErkalkan/confidence-gated-q
- Persistent concept DOI: https://doi.org/10.5281/zenodo.20578927
- Latest archived release DOI before v1.6.0:
  https://doi.org/10.5281/zenodo.20661403 (`v1.5.0`).
- The `v1.6.0` DOI must be added after the new Zenodo version is minted.

## ASOC Result Lineage

| Result directory | Config | Seeds | Status | Generation commit |
|---|---|---|---|---|
| `results/dqn_tuning_development` | `configs/dqn_tuning_development.json` | 0-4 | PASS | `050aa5196aa8789a1a06cd3bd9bef41fcc20b784` |
| `results/dqn_strong_validation` | `configs/dqn_strong_validation.json` | 600-629; 700-709 | PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/confirmatory_extended_compact` | `configs/confirmatory_extended_compact.json` | 500-529 | PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/support_abstention_replication` | `configs/support_abstention_replication.json` | 300-329; 400-429 | PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/minigrid_extended_diagnostic` | `configs/minigrid_extended_diagnostic.json` | 500-509 | PASS | `a867c4bd9a5206204e82b235e92cf9a3500d38ba` |
| `results/application_navigation_case_study` | `configs/application_navigation_case_study.json` | 600-629 | PASS | `b292c8d` |
| `results/adaptive_gate_compact_validation` | `configs/adaptive_gate_compact_validation.json` | 700-729 | PASS | `b292c8d` |
| `results/cost_support_metrics` | `configs/cost_support_metrics.json` | 800-809 | PASS | `b292c8d` |
| `results/strong_baselines` | `configs/strong_baselines/*.yaml` | 600-629 | PASS | `a654625acfbaab6f0875c6312e90326207cf3149` |
| `results/approx_support` | `configs/approx_support/*.yaml` | 600-629 | PASS | `a654625acfbaab6f0875c6312e90326207cf3149` |
| `results/fuzzy_ablation` | `configs/fuzzy_ablation/fuzzy_ablation_30seed.yaml` | 600-629 | PASS | `a654625acfbaab6f0875c6312e90326207cf3149` |
| `results/application_risk_variants` | `configs/application_risk_variants_30seed.yaml` | 600-629 | PASS | `a654625acfbaab6f0875c6312e90326207cf3149` |
| `results/uav_pybullet_validation` | `configs/uav_pybullet_30seed.yaml` | 900-929 | PASS | `a654625acfbaab6f0875c6312e90326207cf3149` |
| `results/fuzzy_reliability_confirmatory` | `configs/fuzzy_reliability_confirmatory_30seed.yaml` | 1300-1329 | PASS | `ce81c103cc524029aa4bbeda841e350b8f9266ec` |
| `results/fuzzy_reliability_shift_confirmatory` | `configs/fuzzy_reliability_shift_confirmatory_30seed.yaml` | 1400-1429 | PASS | `ce81c103cc524029aa4bbeda841e350b8f9266ec` |

Each `metadata.json` records the complete config, config SHA-256, requested and
resolved environment IDs, observation representation, platform, package
versions, and generation commit. Each raw row repeats the key software and
experiment provenance.

Starting with release `v1.5.0`, newly executed result families also record a
deterministic `source_snapshot_sha256` over the exact config, `src/hybrid_q`
implementation, package metadata, and dependency files. The corresponding
`execution_input_manifest` lists every hashed file. Public experiment runs are
rejected when any of those execution inputs differ from the recorded git
commit. Result audits recompute the snapshot and report `STRICT_PASS` only when
the current artifact matches the executed inputs.

The two `v1.6.0` result families share execution commit `ce81c10` and package
version `1.6.0`. The large reliability-shift raw table is losslessly split by
environment and agent under `raw_parts/`; its manifest records each part hash.

## Result Preservation

Per-run shards are omitted after aggregation because the committed
`raw.csv.gz` files preserve every public row needed for reaggregation and
auditing. Aggregation and audit scripts read compressed files directly.

Main confirmatory, auxiliary smoke, and protocol-only configurations are
explicitly labeled. Protocol-only files contain no generated performance
claims.
