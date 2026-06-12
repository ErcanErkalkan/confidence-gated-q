# Support-Boundary Diagnostic for Tabular-Neural Reinforcement Learning

This repository contains the public code, configurations, seed-level results,
tests, and audits for a soft-computing boundary analysis of exact-state memory,
deep Q-networks (DQN), relative estimator reliability, fuzzy adaptive
arbitration, and support abstention.

The method is **not** a generally superior reinforcement-learning algorithm.
Count gating is useful only when exact states recur. It is not a
generalization mechanism and can fail under support shift. The artifact now
includes both a controlled grid diagnostic and a physics-based Crazyflie UAV
validation. The latter is a sim-to-real pre-deployment test, not flight-hardware
validation or a safety guarantee.

Journal submission files are intentionally excluded from the public repository.

## Persistent Identifiers

- Source: https://github.com/ErcanErkalkan/confidence-gated-q
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
- A 30-seed application-navigation case measures collision, support, branch
  usage, exact-memory size, and decision time under deployment goal shift.
- The fuzzy adaptive gate improves over the validated DQN in selected tasks
  but does not generally beat tabular learning or support abstention.
- On three stationary compact tasks, a new relative-reliability fuzzy gate
  still does not significantly beat count gating.
- Under two independently seeded recurring-state reliability shifts, the
  relative-reliability fuzzy gate beats count gating in all `30/30` pairs,
  with AUC gains of `0.0872` and `0.0585`.
- A same-input crisp gate remains slightly stronger than fuzzy defuzzification.
  The supported mechanism is relative reliability, not uniquely necessary
  fuzzy inference.
- In 150 matched physics-based UAV runs, the obstacle-aware PID controller
  reaches `0.922` waypoint success and `0.006` collision rate. The best learned
  support variant reaches `0.162` success, exposing a large deployment gap
  rather than supporting an autonomy claim.

## Analysis Families

| Family | Config | Seeds | Interpretation |
|---|---|---:|---|
| DQN selection | `configs/dqn_tuning_development.json` | 0-4 | development only |
| DQN validation | `configs/dqn_strong_validation.json` | 600-629; 700-709 | independent validation |
| compact expansion | `configs/confirmatory_extended_compact.json` | 500-529 | confirmatory task expansion |
| abstention replication | `configs/support_abstention_replication.json` | 300-329; 400-429 | confirmation after diagnostic discovery |
| MiniGrid expansion | `configs/minigrid_extended_diagnostic.json` | 500-509 | post hoc diagnostic |
| application navigation | `configs/application_navigation_case_study.json` | 600-629 | confirmatory application case |
| fuzzy gate validation | `configs/adaptive_gate_compact_validation.json` | 700-729 | confirmatory adaptive-gate validation |
| cost/support analysis | `configs/cost_support_metrics.json` | 800-809 | descriptive cost and support analysis |
| strong neural baselines | `configs/strong_baselines/*.yaml` | 600-629 | main confirmatory |
| approximate support | `configs/approx_support/*.yaml` | 600-629 | main confirmatory |
| fuzzy component ablation | `configs/fuzzy_ablation/fuzzy_ablation_30seed.yaml` | 600-629 | main confirmatory |
| application fallback ablation | `configs/application_risk_variants_30seed.yaml` | 600-629 | main confirmatory |
| physics-based Crazyflie validation | `configs/uav_pybullet_30seed.yaml` | 900-929 | main external validation |
| stationary fuzzy reliability | `configs/fuzzy_reliability_confirmatory_30seed.yaml` | 1300-1329 | independent confirmatory |
| recurring-state reliability shift | `configs/fuzzy_reliability_shift_confirmatory_30seed.yaml` | 1400-1429 | independent confirmatory |

## Install

Python `>=3.10` is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The physics-based UAV family uses Python 3.12 because PyBullet is not packaged
for every newer interpreter. With `uv`:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-uav.txt
```

## Quick Verification

```powershell
python -m pip install -e .
python scripts/reproduce_all.py --quick
pytest
```

This runs unit tests, a two-seed smoke experiment, aggregation sanity checks,
and artifact audits. Success ends with `QUICK_REPRO_PASS`.

## Full Reproduction

```powershell
python scripts/reproduce_all.py --full
```

Individual experiment families can be rerun with:

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
python scripts/aggregate_strong_baselines.py
python scripts/run_approx_support_experiments.py
python scripts/aggregate_approx_support.py
python scripts/run_fuzzy_ablation.py
python scripts/aggregate_fuzzy_ablation.py
python scripts/run_application_risk_variants.py
python scripts/aggregate_application_risk.py
python scripts/run_uav_validation.py
python scripts/aggregate_uav_validation.py
python scripts/run_benchmark.py --config configs/fuzzy_reliability_confirmatory_30seed.yaml
python scripts/run_benchmark.py --config configs/fuzzy_reliability_shift_confirmatory_30seed.yaml
python scripts/aggregate_fuzzy_reliability.py
python scripts/generate_submission_tables.py
python scripts/generate_submission_figures.py
python scripts/generate_submission_assets.py
python scripts/audit_submission_readiness.py
```

## Evidence Classes

- **Main confirmatory experiments:** executed 30-seed baseline, approximate
  support, fuzzy ablation, application fallback, and physics-based UAV
  families, plus stationary and recurring-state relative-reliability tests.
- **Auxiliary smoke checks:** the two-seed quick path; never used as performance
  evidence.
- **Pre-registered extension protocols:** A2C/PPO and convolutional MiniGrid;
  not reported as completed results.

Completed run shards are resumable and ignored by Git. Committed compressed
CSV files are lossless and read directly by pandas. Very large raw families
are stored as semantic `raw_parts/*.csv.gz` files grouped by environment and
agent; `scripts/audit_results.py` validates both single-file and multipart raw
formats.

## Reproducibility Controls

- Development, validation, confirmation, replication, and diagnostic seeds are
  separated.
- Evaluation occurs at exact environment-step checkpoints.
- Evaluation uses an isolated environment and restored agent RNG.
- Evaluation does not update replay, estimators, or exact-state support.
- Observation representation and resolved environment IDs are recorded.
- Every raw row includes code, package, Python, PyTorch, NumPy, Gymnasium,
  MiniGrid, and optional PyBullet/UAV-backend provenance.
- Paired effects, bootstrap intervals, Holm correction, Wilcoxon sensitivity,
  win/loss/tie counts, and heavy-tail diagnostics are generated.
- Evaluation logs unsupported-state, branch-use, abstention, fuzzy-alpha,
  exact-memory, collision/risk, and decision-time metrics.

## Repository Contents

- `src/hybrid_q/`: environments, encoding, agents, experiments, statistics.
- `configs/`: frozen JSON protocols.
- `results/`: raw data, summaries, comparisons, diagnostics, metadata, audits.
- `scripts/`: execution, aggregation, reproduction, and audit commands.
- `tests/`: unit and integration tests.


## Citation

The artifact is prepared as release `v1.6.0`. Use the persistent concept DOI https://doi.org/10.5281/zenodo.20578927 and the source repository https://github.com/ErcanErkalkan/confidence-gated-q when citing the reproducibility package.

## Author

- Ercan Erkalkan
- Vocational School of Technical Sciences, Department of Electronics and
  Automation, Artificial Intelligence Operator Program, Marmara University
- Mehmet Genc Campus, 34865 Kartal, Istanbul, Turkey
- ercan.erkalkan@marmara.edu.tr
- https://orcid.org/0000-0001-9259-7112

## License

MIT
