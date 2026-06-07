# Support-Aware Exact-State Memory for Deep Q-Learning

This repository contains the public code, configurations, seed-level results,
tests, and audits for a reproducible boundary analysis of exact-state memory
combined with DQN.

The method is **not** a generally superior reinforcement-learning algorithm.
Count gating is useful only when exact states recur. It is not a
generalization mechanism and can fail under support shift.

Journal submission files are intentionally excluded from the public repository.

## Persistent Identifiers

- Source: https://github.com/ErcanErkalkan/confidence-gated-q
- Current archived version DOI:
  https://doi.org/10.5281/zenodo.20581705
- Persistent artifact concept DOI:
  https://doi.org/10.5281/zenodo.20578927
- ORCID: https://orcid.org/0000-0001-9259-7112

## Main Evidence

- Count gating improves the validated DQN on FrozenLake 4x4, FrozenLake 8x8,
  and CliffWalking after Holm correction.
- It does not generally outperform tabular Q-learning or fixed mixing.
- On three held-out-goal FourRooms sizes, count gating delegates unsupported
  states to harmful neural extrapolation.
- New independent seeds confirm that support abstention repairs that specific
  failure, with all 30 pairs favoring abstention in each FourRooms task.
- Six MiniGrid diagnostics remain heterogeneous; tabular Q-learning has the
  best descriptive cross-task rank.

## Analysis Families

| Family | Config | Seeds | Interpretation |
|---|---|---:|---|
| DQN selection | `configs/dqn_tuning_development.json` | 0-4 | development only |
| DQN validation | `configs/dqn_strong_validation.json` | 600-629; 700-709 | independent validation |
| compact expansion | `configs/confirmatory_extended_compact.json` | 500-529 | confirmatory task expansion |
| abstention replication | `configs/support_abstention_replication.json` | 300-329; 400-429 | confirmation after diagnostic discovery |
| MiniGrid expansion | `configs/minigrid_extended_diagnostic.json` | 500-509 | post hoc diagnostic |

Superseded exploratory and sensitivity families are preserved only in the
archived `v1.3.0` release and are not part of the current manuscript artifact.

## Install

Python `>=3.10` is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Verification

```powershell
python scripts/reproduce_all.py --quick
```

This runs all tests, regenerates summaries from committed `raw.csv` or
`raw.csv.gz` files, audits every result family, and executes the artifact
audit.

## Full Reproduction

```powershell
python scripts/reproduce_all.py --full
```

Individual revision families can be rerun with:

```powershell
python scripts/run_benchmark.py --config configs/dqn_tuning_development.json
python scripts/run_benchmark.py --config configs/dqn_strong_validation.json
python scripts/run_benchmark.py --config configs/confirmatory_extended_compact.json
python scripts/run_benchmark.py --config configs/support_abstention_replication.json
python scripts/run_benchmark.py --config configs/minigrid_extended_diagnostic.json
```

Completed run shards are resumable and ignored by Git. Committed compressed
CSV files are lossless and read directly by pandas.

## Reproducibility Controls

- Development, validation, confirmation, replication, and diagnostic seeds are
  separated.
- Evaluation occurs at exact environment-step checkpoints.
- Evaluation uses an isolated environment and restored agent RNG.
- Evaluation does not update replay, estimators, or exact-state support.
- Observation representation and resolved environment IDs are recorded.
- Every raw row includes code, package, Python, PyTorch, NumPy, Gymnasium, and
  MiniGrid provenance.
- Paired effects, bootstrap intervals, Holm correction, Wilcoxon sensitivity,
  win/loss/tie counts, and heavy-tail diagnostics are generated.

## Repository Contents

- `src/hybrid_q/`: environments, encoding, agents, experiments, statistics.
- `configs/`: frozen JSON protocols.
- `results/`: raw data, summaries, comparisons, diagnostics, metadata, audits.
- `scripts/`: execution, aggregation, reproduction, and audit commands.
- `tests/`: unit and integration tests.

This repository does not contain or import Berkeley CS188 assignment code.

## Citation

> Erkalkan, E. (2026). *Support-Aware Exact-State Memory for Deep Q-Learning:
> Reproducibility Artifact* (Version 1.3.0). Zenodo.
> https://doi.org/10.5281/zenodo.20581705

## Author

- Ercan Erkalkan
- Vocational School of Technical Sciences, Department of Electronics and
  Automation, Artificial Intelligence Operator Program, Marmara University
- Mehmet Genç Campus, 34865 Kartal, Istanbul, Turkey
- ercan.erkalkan@marmara.edu.tr
- https://orcid.org/0000-0001-9259-7112

## License

MIT
