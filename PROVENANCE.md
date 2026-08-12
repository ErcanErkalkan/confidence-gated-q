# Provenance

## Scope

This document records the scientific provenance model for the `support-boundary-relative-reliability-artifact` repository. The Python package retains the internal project name `confidence-gated-q` for import and package compatibility. Provenance is organized by evidence family, protocol state, seed registry, source snapshot, and generated artifact rather than by editorial history.

## Canonical evidence controls

The active scientific scope is defined by:

- `configs/evidence_registry.json` — evidence-family and evidence-class registry;
- `configs/claim_evidence_index.yaml` — principal claim-to-evidence map;
- protocol files and protocol SHA manifests under `configs/`;
- registered seed sets in the corresponding configurations and audit records;
- raw and aggregated results under `results/`;
- public derived tables and figures under `tables/` and `figures/`;
- repository-wide integrity manifests generated from the frozen snapshot.

Historical editorial correspondence and packaging records are intentionally excluded from the scientific release tree.

## Evidence classes

The project distinguishes development, confirmatory/independent, descriptive, diagnostic, and supplemental/non-confirmatory evidence. Evidence classes are assigned before interpretation and are not upgraded because of favorable outcomes.

Development selections remain development evidence. Exploratory or descriptive results remain labeled accordingly. The continuous-control SAC–CrossQ–TQC block remains supplemental/non-confirmatory because the resource-constrained design followed a prior compute-feasibility observation.

## Seed isolation

Seed ranges are separated across development, independent evaluation, targeted reliability shifts, sensorized experiments, and continuous-control training/evaluation. Final evaluation seeds are not used for post hoc configuration selection in the declared independent families.

Evaluation is read-only where specified: evaluation trajectories do not expand exact-memory support, update support statistics, or alter frozen reliability estimates.

## Source and runtime traceability

Experiment metadata record source/config hashes, runtime versions, seed identity, and relevant protocol hashes. Artifact audits recompute declared hashes and verify schema/row-count invariants where corresponding audit code is available.

The continuous-control benchmark uses a separate pinned runtime and protocol under `configs/continuous_control/`. The release-facing protocol is a neutral scientific mirror of the immutable execution design. The historical lock SHA is retained inside the canonical protocol provenance metadata, while editorial-origin names remain outside the public project tree.

## Claim boundaries

The provenance system enforces the following interpretation limits:

- support and relative-reliability diagnostics are not calibrated correctness probabilities;
- fuzzy inference is not treated as necessary unless a same-input fuzzy-over-crisp contrast supports that conclusion;
- support coverage is not a generic performance or safety signal;
- software-in-the-loop results are not hardware-in-the-loop or physical-flight evidence;
- null and negative results remain part of the evidence record;
- supplemental continuous-control results do not establish universal controller superiority.

## Released-snapshot and post-release-main rule

The published `v1.0.0` GitHub tag and Zenodo DOI `10.5281/zenodo.21897588` identify the immutable release snapshot at Git commit `29aa392df3670f5ac4062ec831b6b15ca34b88c8`.

Documentation and identifier corrections made later on `main` do not move or rewrite the `v1.0.0` tag and do not alter the archived Zenodo `v1.0.0` files. Any future version derived from a changed public file tree must regenerate `MANIFEST.sha256`, pass the frozen-release preflight, and receive its own release record before being cited as a new version.

## Continuous-control execution-cache boundary

The public v1.0.0 release preserves the frozen continuous-control protocol and release-facing aggregate/seed-level analysis. Serialized trained-model checkpoints and per-agent execution work directories are retained outside the public repository in the private historical archive. They are reproducible execution caches, not additional claim-bearing evidence, and can be regenerated from the frozen public protocol and runner. Their exclusion does not change S1/S2 values, seed-level summaries, or the continuous-control audit.

