# Strict-review fix log

This package implements the manuscript and artifact fixes requested after the strict review.

## Applied directly

- Retitled and reframed the manuscript as a reproducible support-boundary diagnostic, not a generally superior hybrid RL algorithm.
- Shortened and narrowed the contribution list from eight broad claims to five auditable claims.
- Separated the eight main evidence families from auxiliary smoke validations; the main benchmark-status table now contains only the eight audited result families.
- Removed the auxiliary smoke-validation table from the main results to avoid presenting smoke checks as performance evidence.
- Added stronger caveats around the validated DQN comparator and explicitly limited all hybrid-versus-neural claims to that audited comparator family.
- Added final-checkpoint risk-zone rates to the application-navigation table and strengthened the success/collision/risk trade-off discussion.
- Strengthened the hold-action caveat: collision reduction is a diagnostic result, not a field-safety proof.
- Added a clearer explanation that exact-key support shift is a deliberately strict assumption and that approximate support must be evaluated as a different assumption.
- Added reviewer-response full-run configuration templates for approximate support and stronger neural baselines.
- Removed local-workspace references from release-facing documentation and the LICENSE file.
- Added DOI wording that distinguishes the current concept DOI from a future version-specific Zenodo DOI.
- Clean packaging should exclude `.git`, `.pytest_cache`, `__pycache__`, LaTeX build files, and nested ZIPs.

## Not fabricated

The deposited results still do not contain full 30-seed dueling-DQN/A2C/PPO or full 30-seed approximate-support performance runs. The package now includes executable full-run protocols for these reviewer-requested additions, but the manuscript does not claim results that were not executed.
