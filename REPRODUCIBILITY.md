# Reproducibility Guide

## Environment

The completed ASOC revision used:

- Python 3.14.3
- PyTorch 2.11.0 CPU
- NumPy 2.4.3
- pandas 3.0.1
- SciPy 1.17.1
- Gymnasium 1.3.0
- MiniGrid 3.1.0

Install with `requirements.txt` or `environment.yml`. Exact tested versions are
listed in `requirements-tested.txt` and every result directory's
`metadata.json`.

## Quick Reanalysis

```powershell
python scripts/reproduce_all.py --quick
```

Expected outputs for every result family:

- `raw.csv` or lossless `raw.csv.gz`
- `seed_metrics.csv`
- `summary.csv`
- `pairwise.csv`
- `planned_contrasts.csv`
- `heavy_tail_diagnostics.csv`
- `cross_environment.csv`
- `metadata.json`
- `audit.json`
- per-environment learning-curve PNG files

## Revision Experiment Commands

```powershell
python scripts/run_benchmark.py --config configs/dqn_tuning_development.json
python scripts/run_benchmark.py --config configs/dqn_strong_validation.json
python scripts/run_benchmark.py --config configs/confirmatory_extended_compact.json
python scripts/run_benchmark.py --config configs/support_abstention_replication.json
python scripts/run_benchmark.py --config configs/minigrid_extended_diagnostic.json
python scripts/run_application_case_study.py --config configs/application_navigation_case_study.json
python scripts/run_adaptive_gate_validation.py --config configs/adaptive_gate_compact_validation.json
python scripts/run_benchmark.py --config configs/cost_support_metrics.json
```

Aggregate and audit one family:

```powershell
python scripts/aggregate_results.py `
  --input results/confirmatory_extended_compact/raw.csv.gz `
  --output results/confirmatory_extended_compact

python scripts/audit_results.py `
  --config configs/confirmatory_extended_compact.json `
  --result-dir results/confirmatory_extended_compact `
  --output results/confirmatory_extended_compact/audit.json
```

## Analysis Classes

- `development_model_selection`: DQN selection using seeds `0-4`.
- `independent_validation_after_development_selection`: untouched validation
  seeds.
- `confirmatory_task_expansion`: planned compact-task comparisons.
- `confirmatory_replication_after_diagnostic_discovery`: new support-abstention
  seeds after the mechanism was discovered.
- `post_hoc_extended_minigrid_diagnostic`: broader diagnostic evidence.
- `confirmatory_application_case_study`: controlled deployment-goal shift.
- `confirmatory_adaptive_gate_validation`: fuzzy-gate compact validation.
- `descriptive_cost_and_support_analysis`: support, memory, and timing metrics.

These labels are stored in config and result metadata and are not inferred from
outcomes.

## Compute

Measured cumulative training time across independent runs:

| Family | Run-hours | Median run | Maximum run |
|---|---:|---:|---:|
| DQN tuning | 0.53 h | 7.57 s | 20.15 s |
| DQN validation | 1.47 h | 10.52 s | 45.41 s |
| compact expansion | 4.95 h | 9.82 s | 60.67 s |
| abstention replication | 4.21 h | 13.91 s | 80.89 s |
| MiniGrid expansion | 6.75 h | 46.68 s | 165.42 s |
| application navigation | 0.47 h | 11.47 s | 14.26 s |
| adaptive gate validation | 0.95 h | 9.29 s | 13.30 s |
| cost/support analysis | 0.32 h | 14.61 s | 22.38 s |

Eight independent runs were normally scheduled in parallel. Wall time depends
on CPU contention and storage.

## Determinism

Environment resets, action selection, replay sampling, and PyTorch
initialization are seed controlled. PyTorch deterministic algorithms are
enabled. Exact bitwise identity across operating systems, processors, or
library builds is not guaranteed; statistical reproduction should use the
recorded versions and seed sets.

Evaluation is read-only. It runs in a separate environment, restores the agent
RNG, does not insert unseen keys, and records evaluation time separately from
training time.

## Manuscript Assets

From the complete local submission workspace:

```powershell
python scripts/generate_asoc_assets.py
python scripts/generate_asoc_strong_revision_tables.py
python scripts/generate_asoc_strong_revision_figures.py
```

This writes vector figures and LaTeX tables under `paper/figures/` and
`paper/generated/`, including `verified_claims.csv`.

Compile:

```powershell
Set-Location paper
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Audits

```powershell
python scripts/audit_artifact.py --root . --output artifact_audit.json
python scripts/audit_asoc_strong_revision.py
```

The public artifact excludes journal-only paper and portal files. The journal submission package is verified separately before upload.
