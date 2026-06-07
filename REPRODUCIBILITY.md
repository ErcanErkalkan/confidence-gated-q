# Reproducibility Guide

## Tested environment

See `requirements-tested.txt` and each result set's `metadata.json`.
Experiments were run on CPU with one PyTorch compute thread and one inter-op
thread per independent run. Eight independent runs were scheduled in parallel.
Deterministic PyTorch algorithms were enabled.

## Result sets

- `results/confirmatory`: primary evidence, seeds 100-129
- `results/external_minigrid`: bounded MiniGrid diagnostic, seeds 200-209
- `results/support_abstention_confirmatory`: post hoc mechanism test
- `results/minigrid_supplemental`: missing MiniGrid baselines and abstention
- `results/tau_sensitivity_*`: development-seed count-scale sweep
- `results/dqn_sensitivity_*`: development-seed DQN sweep
- `results/dqn_validation_*`: validation of the development-selected DQN

Each directory contains:

- `raw.csv`: combined immutable run output
- `seed_metrics.csv`: one row per method/environment/seed
- `summary.csv`: descriptive estimates and intervals
- `pairwise.csv`: exploratory all-pairs comparisons
- `planned_contrasts.csv`: predefined comparisons
- `metadata.json`: configuration and platform metadata
- `audit.json`: integrity and coverage audit
- learning-curve figures

The local `runs/` subdirectories are resumable shards and are excluded from the
release ZIP because `raw.csv` contains the same records.

## Determinism

Environment resets, action selection, replay sampling, and PyTorch
initialization are seed-controlled. Exact bitwise identity across operating
systems or PyTorch builds is not guaranteed; statistical reproduction should
use the frozen package versions and seed sets.

Evaluation is read-only. It runs in a separate environment, does not insert
unseen table/count keys, restores the agent RNG state, and records evaluation
time separately from training time.

## Reanalysis without retraining

```powershell
python scripts/reproduce_all.py --quick
```

## Full reproduction

```powershell
python scripts/reproduce_all.py --full
```

The quick path takes about 3-5 minutes on the development machine. A full CPU
rerun typically takes 2-4 hours and resumes completed method/seed shards.
