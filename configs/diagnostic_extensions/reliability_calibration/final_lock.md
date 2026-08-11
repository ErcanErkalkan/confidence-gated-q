# Independent focal relative-reliability calibration lock

This protocol closes the empirical portion of the independent focal calibration diagnostic without
turning the residual score into a correctness probability or safety guarantee.
It is locked before seeds 12090--12099 are executed.

The frozen fuzzy and same-input crisp mappings are evaluated on two analytic
one-step reliability shifts. The environment exposes both the optimal action
and the complete one-step action-value target. Action correctness and value
error are separate binary targets; their rows and metrics are never pooled.

The primary scope is all post-shift contexts because the declared context set
contains both changed and unchanged regions. No context may be removed after
inspection to manufacture class balance. AUROC is unavailable whenever a
target/scope/agent/severity cell contains only one class. Brier score and
calibration bins refer only to the explicitly named binary target and are not
interpreted as general confidence calibration.

All ten seeds, both severities, and both mappings must complete. Null, negative,
one-class, delayed, and contradictory results are retained. No performance or
fuzzy-superiority hypothesis is tested in this family.

Once `RELIABILITY_CALIBRATION_FINAL_LOCK.sha256` is created, this file and its
YAML companion are immutable. A later change requires a dated amendment that
states whether any result was inspected.
