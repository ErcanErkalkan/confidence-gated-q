# Reproducibility Guide

## Environment

The reported study used:

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

The physics-based UAV validation used a separate Python 3.12.13 environment
with Gymnasium 1.2.3, gym-pybullet-drones 2.1.0 at commit
`9bc12bc583fa3b28807b2f90a8cadf09fb06e1ff`, PyBullet 3.2.7, PyTorch 2.12.0,
and NumPy 2.4.6. Exact versions are in `requirements-tested-uav.txt`.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-uav.txt
```

## Quick Reanalysis

```powershell
python -m pip install -e .
python scripts/reproduce_all.py --quick
pytest
```

The quick path is intentionally bounded to unit tests, one two-seed smoke
experiment, aggregation sanity, and audits. It prints `QUICK_REPRO_PASS`.

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

## Experiment Commands

```powershell
python scripts/run_benchmark.py --config configs/dqn_tuning_development.json
python scripts/run_benchmark.py --config configs/dqn_strong_validation.json
python scripts/run_benchmark.py --config configs/confirmatory_extended_compact.json
python scripts/run_benchmark.py --config configs/support_abstention_replication.json
python scripts/run_benchmark.py --config configs/minigrid_extended_diagnostic.json
python scripts/run_application_case_study.py --config configs/application_navigation_case_study.json
python scripts/run_adaptive_gate_validation.py --config configs/adaptive_gate_compact_validation.json
python scripts/run_benchmark.py --config configs/cost_support_metrics.json
python scripts/run_strong_baselines.py
python scripts/run_approx_support_experiments.py
python scripts/run_fuzzy_ablation.py
python scripts/run_application_risk_variants.py
python scripts/run_uav_validation.py
python scripts/aggregate_uav_validation.py
python scripts/run_uav_validation.py --config configs/uav_sensorized_motor_30seed.yaml
python scripts/aggregate_uav_sensorized_validation.py
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
- `main_confirmatory_strong_neural_baseline`: Double and Dueling Double DQN.
- `main_confirmatory_approximate_support`: kNN and feature-distance support.
- `main_confirmatory_fuzzy_ablation`: fuzzy component and crisp-gate ablation.
- `main_confirmatory_application_fallback_ablation`: hold/no-hold/planner
  fallback risk analysis.
- `main_external_physics_based_uav_validation`: legacy label for the
  state-accessible held-out-waypoint simulator benchmark with seeds `900-929`.
- `main_sensorized_motor_control_sil_validation`: delayed/lossy VIO, high-rate
  IMU, ten-ray ranging, pinhole target visibility, and roll/pitch/collective
  commands converted to motor RPM with seeds `1000-1029`.
- `independent_confirmatory_fuzzy_reliability_stationary`: stationary
  relative-reliability validation with seeds `1300-1329`.
- `independent_confirmatory_reliability_shift`: recurring-state stale-support
  validation with seeds `1400-1429`.
- `auxiliary_smoke_check`: installation and pipeline check only.
- `preregistered_extension_protocol_not_executed`: no result claim permitted.

These labels are stored in config and result metadata and are not inferred from
outcomes.

## Strict Source Provenance

New `v1.5.0` and later experiment families record:

- the clean execution commit;
- package and dependency versions;
- the exact config hash;
- an execution-input manifest with per-file SHA-256 values;
- a combined `source_snapshot_sha256`.

The runner refuses public experiment execution when tracked or untracked
changes affect the exact config, `src/hybrid_q`, package metadata, or dependency
files. `scripts/audit_results.py` independently recomputes the snapshot and
requires the raw rows, metadata, and current source tree to agree.

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
| physics-based UAV validation | 4.78 h | 103.48 s | 201.52 s |
| sensorized UAV SIL validation | 2.04 h | 59.73 s | 111.65 s |
| stationary fuzzy reliability | 2.05 h | 12.89 s | 20.77 s |
| recurring-state reliability shift | 2.17 h | 22.42 s | 25.95 s |

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

Raw results are normally stored as `raw.csv.gz`. Families that exceed GitHub's
single-file limit use lossless `raw_parts/*.csv.gz` files grouped by
environment and agent. `scripts/pack_raw_parts.py` creates the parts and
`scripts/audit_results.py` verifies them as one logical raw table.

## Manuscript Assets

From the complete local submission workspace:

```powershell
python scripts/generate_asoc_assets.py
python scripts/generate_submission_tables.py
python scripts/generate_submission_figures.py
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
python scripts/audit_submission_readiness.py
```

The public artifact excludes journal-only paper and portal files. The journal submission package is verified separately before upload.
