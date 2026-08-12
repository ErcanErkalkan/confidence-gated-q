# Support-Boundary and Relative-Reliability Diagnostics for Hybrid Memory–Neural Reinforcement Learning

This repository is the independent public reproducibility artifact for an audited study of support boundaries, estimator-relative reliability, fuzzy and crisp arbitration, fallback behavior, sensorized software-in-the-loop control, and continuous-control transfer diagnostics. The Python package retains the internal project name `confidence-gated-q` for import and package compatibility.

The artifact is organized around one scientific question: under which support and estimator-reliability conditions can hybrid memory–neural control make auditable branch-selection decisions, and where do exact, approximate, fuzzy, crisp, fallback, or replay-support mechanisms fail under shift?

The artifact does **not** establish a universally superior reinforcement-learning controller, calibrated correctness confidence, hardware readiness, flight safety, or operational deployment readiness.

## Public/private boundary

This directory is the **public reproducibility artifact only**. Manuscript, supplementary manuscript files, bibliography sources, reviewer responses, editorial correspondence, cover letters, title pages, highlights, and other journal-submission material are maintained separately under the private project area and are intentionally excluded from this repository and from the public Zenodo artifact.

## Scientific evidence families

The repository contains the following connected evidence families:

- compact tabular/DQN recurrence diagnostics and matched stronger neural comparators;
- held-out exact-support shifts and support-abstention replication;
- approximate support through tolerance-kNN and feature-distance/kernel affinity;
- application-navigation fallback and risk-adjusted comparisons;
- fuzzy arbitration, relative-reliability diagnostics, and same-input crisp falsification;
- independent reliability-shift generators and support-estimator selection;
- state-accessible and sensorized Crazyflie/PyBullet software-in-the-loop diagnostics;
- factorial, temporal/interface, safety-trace, and feasibility follow-ups for the sensorized boundary;
- a supplemental/non-confirmatory SAC–CrossQ–TQC benchmark on HalfCheetah-v5 and Walker2d-v5.

Null and negative results are retained as evidence. Continuous-control results are supplemental/non-confirmatory and are not used for universal controller ranking.

## Canonical public-artifact layout

```text
support-boundary-relative-reliability-artifact/
├── .gitignore
├── README.md
├── PROVENANCE.md
├── REPRODUCIBILITY.md
├── CITATION.cff
├── .zenodo.json
├── pyproject.toml
├── requirements.txt
├── requirements-tested.txt
├── requirements-uav.txt
├── requirements-tested-uav.txt
├── requirements-continuous-control.txt
├── requirements-tested-continuous-control.txt
├── requirements-tested-continuous-control.sha256
├── configs/
│   ├── evidence_registry.json
│   ├── claim_evidence_index.yaml
│   ├── diagnostic_extensions/
│   └── continuous_control/
├── src/
├── scripts/
├── tests/
├── results/
│   ├── diagnostic_extensions/
│   └── continuous_control/
├── tables/
├── figures/
└── audits/
```

The manuscript tree is not part of this public layout.

## Evidence and claim control

`configs/evidence_registry.json` defines the active evidence families and evidence classes.

`configs/claim_evidence_index.yaml` maps the principal scientific claim families to stable computational evidence and inference classes. Public claim-evidence control must remain independent of private manuscript file paths.

Protocol locks, seed registries, raw/aggregated outputs, and SHA-256 manifests provide execution traceability. Evaluation is read-only where declared by the protocol.

## Key evidence boundaries

- Exact count support is informative only when recurring exact states provide relevant memory evidence.
- Approximate support softens the exact-key boundary but remains representation-dependent.
- Relative reliability is useful under the targeted stale-memory mechanism but does not generalize as a uniformly superior gate across independent shift generators.
- Same-input crisp comparisons prevent a claim that fuzzy defuzzification is necessary for the observed relative-reliability mechanism.
- Sensorized SIL exposes exact-support collapse and partial kNN coverage without learned waypoint success.
- Continuous-control replay support responds consistently to observation-delay mismatch but not to downstream actuation-authority change as a generic degradation signal.

## Installation

### Core and discrete diagnostics

Python 3.10 or newer is supported by the package metadata.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### PyBullet UAV diagnostics

The recorded UAV environment uses the pinned dependencies in `requirements-tested-uav.txt`. A Python 3.12 environment is recommended for compatibility with the recorded PyBullet stack.

### Continuous-control supplemental benchmark

Use a separate environment for this benchmark. `requirements-continuous-control.txt` contains exact pins for the direct runtime packages used by the runner. `requirements-tested-continuous-control.txt` is the full 32-package `pip freeze` snapshot recorded by the executed runs, with its SHA-256 stored in `requirements-tested-continuous-control.sha256`. The executed environment records Python 3.13.x, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, MuJoCo 3.10.0, NumPy 2.5.1, pandas 3.0.5, psutil 7.2.2, PyYAML 6.0.3, SciPy 1.18.0, and Torch 2.13.0.

## Quick verification

```bash
python -m pip install -e .
python scripts/reproduce_all.py --quick
python -m pytest
```

The quick path runs tests, a small smoke experiment, public-only deterministic asset generation, and a registry-driven integrity audit over E01-E28, the 16-claim claim-evidence index, continuous-control S1/S2 outputs, protocol/environment hashes, result audits, and the public/private boundary. It does not recreate every computationally expensive evidence family.

To rerun the registered Environment A evidence families E01-E28, use:

```bash
python scripts/reproduce_all.py --full
```

The supplemental continuous-control grid is intentionally kept in its separate Environment B and can be rerun with:

```bash
python scripts/reproduce_all.py --continuous-control
```

To audit the existing public tree without rerunning experiments:

```bash
python scripts/reproduce_all.py --audit-only
```

Release preflight without training or result regeneration is:

```bash
python scripts/reproduce_all.py --preflight
```

This compiles `src/`, `scripts/`, and `tests/` into a temporary cache, runs the full pytest suite, verifies protocol/environment SHA-256 locks, and executes the registry-driven public-artifact audit without mutating the repository tree.

After the final `MANIFEST.sha256` is generated from the frozen public tree, require complete manifest coverage with either:

```bash
python scripts/reproduce_all.py --audit-only --require-manifest
python scripts/reproduce_all.py --preflight --require-manifest
```

The frozen-release audit rejects missing, duplicate, unexpected, hash-mismatched, **and omitted public-file** manifest entries.

## Public derived assets

Public-only deterministic assets can be regenerated with:

```bash
python scripts/generate_tables.py
python scripts/generate_figures.py
```

These scripts write only to `tables/` and `figures/`. Manuscript-facing LaTeX tables, graphical abstracts, and private paper copies are generated outside this public repository. Most experiment-specific tables and figures continue to be emitted by their corresponding `aggregate_*` scripts.

## Continuous-control execution

Example registered run:

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

Aggregate the complete registered grid with:

```bash
python scripts/aggregate_continuous_control.py
```

## Artifact identity

- Artifact version: `1.0.0`
- ORCID: `0000-0001-9259-7112`
- Repository URL: https://github.com/ErcanErkalkan/support-boundary-relative-reliability-artifact
- GitHub release: https://github.com/ErcanErkalkan/support-boundary-relative-reliability-artifact/releases/tag/v1.0.0
- Zenodo DOI: https://doi.org/10.5281/zenodo.21897588

No DOI or repository URL from an earlier public artifact series is reused by this artifact.

## License

MIT License. See `LICENSE`.
