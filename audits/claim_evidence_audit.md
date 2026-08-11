# A-Z Claim-Evidence Audit — 2026-08-10

## Result

**PASS after manuscript-independent traceability migration.**

The sixteen top-level scientific claim families are preserved without changing claim wording, numerical results, seeds, protocol scope, or inference class. The release-facing claim index is now resolved entirely through public artifact paths, registered evidence-family identifiers, and explicitly labeled auxiliary or supplemental diagnostics.

## Canonical traceability model

`configs/evidence_registry.json` remains the authority for the registered E01–E28 evidence-family scope. `configs/claim_evidence_index.yaml` links each claim to:

- registered `evidence_families` where an E-family exists;
- `primary_artifacts` containing the direct computational evidence;
- `supporting_artifacts` used for visualization, provenance, implementation, or bounded secondary support;
- an explicit `boundary` that limits interpretation.

The claim index no longer depends on document-layout labels or editorial packaging. Auxiliary programs do not create new E-family identifiers and do not upgrade evidence class.

## Claim-family status

| ID | Claim family | Evidence class | Status |
|---|---|---|---|
| C01_exact_recurrence_benefit | Count gating improves over the development-selected DQN on selected recurrent compact tasks, but not universally. | confirmatory | PASS |
| C02_heldout_support_failure | Held-out exact-state support shift can make count gating harmful by delegating to unsupported neural extrapolation. | confirmatory boundary + independent replication | PASS |
| C03_support_abstention | Support abstention repairs the replicated FourRooms failure but is not a universal improvement. | independent replication | PASS |
| C04_approximate_support | kNN and feature-distance support improve over exact count gating in the application shift, while remaining below strong non-neural comparators. | confirmatory | PASS |
| C05_dqn_robustness | Double DQN and Dueling Double DQN do not materially alter the application conclusion. | independent evaluation | PASS |
| C06_relative_reliability_targeted | Relative reliability is useful in the targeted stale-memory shift. | mechanism-targeted confirmatory | PASS |
| C07_relative_reliability_replication | The targeted relative-reliability gain does not generalize across three independently locked shift generators. | independent replication | PASS |
| C08_fuzzy_not_necessary | The same-input crisp mapping prevents a claim that fuzzy defuzzification is necessary for the observed relative-reliability effect. | matched-input falsification | PASS |
| C09_support_estimator_transfer | The selected raw-normalized support estimator improves over exact counting across the three independent support-final generators, without establishing a universally superior representation. | independent evaluation after development selection | PASS |
| C10_calibration_boundary | Support and relative-reliability scores have ranking evidence but are not calibrated correctness probabilities. | calibration diagnostic | PASS |
| C11_complexity_boundary | Support estimators have distinct theoretical and measured computational costs; indexed kNN is not uniformly faster at higher dimension. | theoretical + descriptive microbenchmark | PASS |
| C12_sensorized_sil_boundary | Sensorized SIL testing exposes exact-support collapse, partial kNN coverage without waypoint success, and zero success for the tested learned controllers. | locked simulator evidence | PASS |
| C13_sensorized_causal_limit | The sensorized learned-control failure cannot be assigned to one causal source with the present design. | negative boundary / causal limitation | PASS |
| C14_continuous_control_transfer | Replay-neighborhood support transfers to SAC, CrossQ, and TQC observations in two MuJoCo domains. | supplemental non-confirmatory | PASS |
| C15_shift_specific_support | Observation delay consistently lowers replay-support coverage, whereas reduced actuation authority does not produce a consistent support decrease. | supplemental non-confirmatory | PASS |
| C16_hardware_boundary | The highest physical evidence level is sensorized SIL; hardware readiness, HIL validity, flight safety, and operational readiness are not established. | scope boundary | PASS |

## Registered and auxiliary evidence

The E01–E28 registry remains unchanged in scope. Claims C14 and C15 use the separate continuous-control program as explicitly supplemental, non-confirmatory evidence and therefore do not receive invented E-family identifiers. C11 uses the support-scaling microbenchmark as an auxiliary theoretical/descriptive program while retaining its registered support-estimator family context.

## Provenance-path cleanup

Release-facing derived tables used by the claim index were normalized so their provenance columns refer to the canonical public diagnostic paths and neutral scientific report-family names. Only provenance/path labels were changed. Numeric tokens were compared before and after migration and were unchanged in all migrated tables.

## Release-facing scientific boundaries

- No universal reinforcement-learning controller superiority claim.
- No claim that fuzzy inference is uniquely necessary.
- No calibrated correctness-probability claim for support or relative-reliability scores.
- No hardware-in-the-loop, flight, physical-safety, or operational-readiness claim.
- Continuous-control results remain supplemental and non-confirmatory.
- Null and negative results remain first-class evidence.
- Development selection and independent evaluation remain separate.

## Audit rule

A release-facing claim must fail traceability if its primary evidence cannot be resolved inside the public artifact, if it relies on private or editorial-only material, if an auxiliary result is promoted beyond its declared evidence class, or if its interpretation exceeds the registered boundary.
