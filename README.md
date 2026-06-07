# Confidence-Gated Tabular-Neural Q-Learning

This repository contains the public code, configurations, tests, seed-level
results, and integrity audits for a controlled study of exact-state memory
combined with deep Q-learning.

The repository intentionally excludes journal submission files.

## Main Finding

A visitation-count gate can protect action selection from the selected DQN
configuration when compact states recur. It does not consistently outperform
tabular Q-learning and is not a reliable generalization mechanism.

On held-out-goal FourRooms tasks, low exact-state support causes standard gates
to delegate to harmful neural extrapolation. A post hoc support-abstention
baseline repairs that specific failure by reverting unsupported states to
random tie-breaking, but it does not become a broadly superior method.

## Repository Contents

- `src/hybrid_q/`: tabular, DQN, fixed-mixture, count-gated,
  TD-reliability-gated, and support-abstention agents.
- `configs/`: confirmatory, diagnostic, and sensitivity protocols.
- `results/`: current raw outputs, seed metrics, summaries, comparisons,
  metadata, learning curves, and audits.
- `scripts/`: experiment, aggregation, reproduction, and audit commands.
- `tests/`: unit and integration tests.

This repository does not contain or import Berkeley CS188 assignment code.

## Install

Python `>=3.10` is required. The recorded environment used Python `3.14.3`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda users can instead run:

```powershell
conda env create -f environment.yml
conda activate confidence-gated-q
```

## Verify Existing Results

```powershell
python scripts/reproduce_all.py --quick
```

This runs the tests, regenerates summaries from the committed raw data, checks
every current result set, and executes the public artifact audit.

## Full Reproduction

```powershell
python scripts/reproduce_all.py --full
```

The full command reruns all experiments before executing the same checks.
Completed method/seed shards are resumable and excluded from Git because each
result directory already contains the combined `raw.csv`.

## Evaluation Controls

- Development and confirmatory seeds are separated.
- Evaluation occurs at exact environment-step checkpoints.
- Evaluation uses an isolated environment and random-number stream.
- Evaluation does not mutate replay, estimators, or exact-state support.
- Training and evaluation time are recorded separately.
- All configured seeds are retained.

## Verification Status

- Unit and integration tests: 13 passed.
- Current result-set audits: 10 passed.
- Public artifact audit: PASS.

See `REPRODUCIBILITY.md` for result-set details and deterministic-execution
limits.

## License

The independent research implementation is released under the MIT License.
