# Reproducibility

## Reproduction model

The public artifact contains two runtime groups because the compact/UAV diagnostics and the modern continuous-control benchmark use different recorded dependency stacks. Reproduction should preserve that separation rather than forcing all experiments into one environment.

## Environment A — compact, support/reliability, and UAV diagnostics

Install the core project:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The recorded PyBullet UAV stack is pinned separately in `requirements-tested-uav.txt`; Python 3.12 is recommended for that stack.

Quick verification:

```bash
python scripts/reproduce_all.py --quick
python -m pytest
```

The quick path is a smoke/integrity check and is not a substitute for rerunning all registered training families. Its artifact audit is registry-driven: it checks E01-E28 source coverage, all 16 claim-evidence entries, public artifact paths, existing result audits, continuous-control S1/S2 counts and labels, protocol/environment hashes, and the public/private boundary.

Full Environment A reproduction (E01-E28) is:

```bash
python scripts/reproduce_all.py --full
```

Existing-output audit without retraining is:

```bash
python scripts/reproduce_all.py --audit-only
```

Non-training release preflight is:

```bash
python scripts/reproduce_all.py --preflight
```

The preflight compiles `src/`, `scripts/`, and `tests/` using a temporary bytecode cache, runs the full pytest suite, exercises protocol/hash regression tests, and runs the artifact audit. It does not rerun training or regenerate registered scientific results.

Public deterministic derived assets that are not emitted directly by experiment aggregators can be regenerated with:

```bash
python scripts/generate_tables.py
python scripts/generate_figures.py
```

Both commands are constrained to the public `tables/` and `figures/` trees. They do not create or copy manuscript-facing files.

## Environment B — continuous-control supplemental benchmark

Use a separate environment for this benchmark. Install from `requirements-continuous-control.txt` for the exact direct/runtime pins. For the closest reconstruction of the executed environment, install from `requirements-tested-continuous-control.txt`; this file is the full 32-package `pip freeze` snapshot captured by the registered runs and is protected by `requirements-tested-continuous-control.sha256`.

The executed runtime records:

- Python 3.13.x;
- Gymnasium 1.3.0;
- Stable-Baselines3 2.9.0;
- sb3-contrib 2.9.0;
- MuJoCo 3.10.0;
- NumPy 2.5.1;
- pandas 3.0.5;
- psutil 7.2.2;
- PyYAML 6.0.3;
- SciPy 1.18.0;
- Torch 2.13.0;
- CPU execution with one environment per process;
- three independent worker processes, two Torch intra-op threads per worker, and one inter-op thread per worker.

The frozen scientific specification is `configs/continuous_control/CONTINUOUS_CONTROL_PROTOCOL.yaml`, with hashes in `CONTINUOUS_CONTROL_PROTOCOL_SHA256.txt`.

Registered example:

```bash
python scripts/run_continuous_control.py \
  --lock-yaml configs/continuous_control/CONTINUOUS_CONTROL_PROTOCOL.yaml \
  --lock-manifest configs/continuous_control/CONTINUOUS_CONTROL_PROTOCOL_SHA256.txt \
  --mode supplemental \
  --algorithm SAC \
  --environment HalfCheetah-v5 \
  --seed 22000 \
  --output results/continuous_control
```

The registered grid consists of SAC, CrossQ, and TQC; HalfCheetah-v5 and Walker2d-v5; five training seeds per algorithm–environment pair; 100,000 nominal training interactions per agent; and deterministic evaluation under nominal, observation-delay, actuation-authority, and exploratory combined shifts.

Aggregation:

```bash
python scripts/aggregate_continuous_control.py
```

The full frozen supplemental grid can also be executed through the unified driver in this separate environment:

```bash
python scripts/reproduce_all.py --continuous-control
```

The aggregate audit expects 30 complete trained runs, 3,600 final episode rows, 600 checkpoint episode rows, eight S1 controller contrasts, and twelve S2 support contrasts.

### Continuous-control release packaging

The clean public release retains the frozen protocol and release-facing analysis outputs (`condition_summary.csv`, seed/checkpoint summaries, return AUC, S1/S2 contrasts, and `audit.json`). Per-agent training work directories and serialized model checkpoints are execution caches rather than claim-bearing release artifacts; they are retained in the private historical archive and are intentionally excluded from GitHub/Zenodo. They can be regenerated from the frozen protocol with `python scripts/reproduce_all.py --continuous-control`. This packaging change does not alter any reported numerical result.

## Public artifact boundary

This repository reproduces the computational evidence only. Manuscript compilation, supplementary compilation, bibliography processing, reviewer-response generation, and journal-submission packaging are outside this public artifact and are maintained in the separate private project area.

Scripts whose sole purpose is to generate private manuscript-facing assets must not be treated as required public reproduction steps.

## Determinism and statistical reproduction

Random seeds control environment resets, action selection, replay sampling, model initialization, and registered evaluation resets where applicable. Exact bitwise identity across operating systems, processors, BLAS libraries, CUDA/CPU implementations, or dependency builds is not guaranteed. Statistical reproduction should use the recorded package versions, protocol files, and seed sets.

## Support diagnostics

The artifact distinguishes exact-key support, approximate/tolerance support, kernel affinity, replay-neighborhood support, branch-use rates, and fallback rates. These measures have different estimands and must not be interchanged.

Continuous-control support is computed from final nominal replay observations with deterministic subsampling, per-dimension standardization, k=5 cKDTree neighborhoods, and a radius defined by the 95th percentile of fifth non-self-neighbor distances. Reward, failure labels, and shifted observations do not enter the radius calibration.

## Audit order for a frozen public snapshot

A public frozen snapshot should pass, in order:

1. `python scripts/reproduce_all.py --preflight` for compile, pytest, protocol/environment-lock, registry, result-audit, and public-boundary checks;
2. optional `python scripts/reproduce_all.py --quick` smoke reproduction if the core Environment A stack is available;
3. final repository SHA-256 manifest generation from the frozen public tree;
4. `python scripts/reproduce_all.py --preflight --require-manifest` for the frozen-release preflight.

Before step 3, `audit_artifact.py` operates in pre-publication mode and does not require `MANIFEST.sha256`. After the final manifest is generated, `--require-manifest` changes the audit to frozen-release mode and verifies SHA-256 values **and exact public-file coverage**: omitted, duplicate, unexpected, missing, or mismatched entries fail the audit. No manifest from an earlier tree should be reused after file moves, renames, metadata rewrites, or regenerated computational artifacts.
