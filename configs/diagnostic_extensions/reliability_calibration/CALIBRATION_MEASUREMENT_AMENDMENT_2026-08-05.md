# Reliability-calibration instrumentation amendment — 2026-08-05

Status: locked before the instrumentation rerun.

This amendment does not change a hypothesis, target, metric, environment,
severity, seed, budget, checkpoint, evaluation-episode count, agent mapping, or
analysis scope in `RELIABILITY_CALIBRATION_FINAL_LOCK.yaml`.

The first execution completed all 40 agent-by-seed-by-severity runs, but the
fail-closed calibration aggregator rejected it. All 52,480 evaluation rows had
analytic environment labels, while all 52,480 rows lacked
`relative_reliability_score`, branch greedy actions, branch correctness, and
branch Q-error diagnostics. The cause was an instrumentation omission in the
`fuzzy_risk_aware` inference path: it returned the already-computed mixed values
after recording general decision diagnostics but before recording branch
diagnostics.

The correction adds only the same `_record_branch_diagnostics(...)` call already
used by the other hybrid inference paths. It does not alter returned Q values,
action selection, agent updates, random-number draws, or environment behavior. A
regression test checks both the unchanged returned values and the newly recorded
diagnostics.

The initial output at
`results/diagnostic_extensions/reliability_calibration_independent/execution/` is
retained as an invalid instrumentation attempt and must not be used as scientific
evidence. The exact locked seed set is rerun solely to repair measurement under
`execution_instrumentation_rerun/`; this is not a second independent replication.
The rerun must match the initial execution on non-diagnostic episode outcomes.
Any mismatch fails the audit. No seed may be excluded or substituted.
