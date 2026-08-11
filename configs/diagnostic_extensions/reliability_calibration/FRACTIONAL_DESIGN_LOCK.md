# Reliability-calibration development design lock

Locked before execution on 2026-08-03 (Europe/Istanbul).

This is a development-only diagnostic. It uses seeds `10000-10004`, a strict
subset of the `reliability_calibration_development` reservation
`10000-10099`. Reserved final seeds `12000-12099` are prohibited.

The requested Cartesian grid has 48 cells:

- beta: `0.02, 0.05, 0.10, 0.20`;
- lambda (the residual-estimate shrinkage prior strength): `1, 5, 10, 20`;
- denominator epsilon: `1e-8, 1e-6, 1e-4`.

The executed half fraction contains a cell exactly when
`(beta_index + lambda_index + epsilon_index) mod 2 = 0`, using zero-based
indices in the orders above. This yields 24 cells: eight for every epsilon and
six for every beta and lambda. It supports development main-effect sensitivity
while leaving higher-order interactions partially aliased. No interaction claim
will be made.

The two environments are analytic one-step contextual bandits with post-shift
boundaries 0.30 and 0.20. Each exposes the optimal action and full two-action
one-step Q-star vector. Action correctness and Q-vector RMSE are separate
targets. The primary diagnostic is post-shift action-correctness AUROC; the
value-error target, calibration, selective risk, and detection delay are
secondary development diagnostics.
