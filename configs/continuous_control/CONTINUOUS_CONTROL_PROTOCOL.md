# Continuous-Control Supplemental Benchmark Protocol

**Evidence class:** supplemental / non-confirmatory  
**Protocol state:** frozen before resource-constrained performance execution  
**Scientific scope:** support-boundary transfer to standard continuous-control replay observations

## Research question

Does the support-boundary diagnostic remain informative in standard continuous-control domains where exact state recurrence is structurally rare, and how do support/failure signals behave for modern off-policy actor-critic controllers under nominal, observation-delay, and actuation-authority shifts?

## Design

The benchmark uses SAC, CrossQ, and TQC on HalfCheetah-v5 and Walker2d-v5. Five training seeds (22000--22004) are assigned to every algorithm--environment pair. Each trained agent receives exactly 100,000 nominal interactions. Deterministic checkpoint evaluation occurs at 50,000 and 100,000 interactions, and final evaluation uses 30 common reset seeds per condition.

The evaluation conditions are NOMINAL, one-step observation delay (OBS_DELAY_1), 0.75 actuation authority (ACT_GAIN_075), and the exploratory combined delay-plus-gain condition. Performance-based early stopping, seed replacement, and post-freeze hyperparameter tuning are prohibited.

## Support diagnostic

Support is computed from final nominal replay observations. Up to 50,000 observations are sampled deterministically, standardized per dimension with a 1e-6 standard-deviation floor, and indexed with a k=5 SciPy cKDTree. The approximate-support radius is the 95th percentile of fifth non-self-neighbor distances in a deterministic nominal calibration subset. Reward, failure labels, and shifted observations do not enter the support calibration. Exact float-vector recurrence is descriptive only.

## Critic diagnostics

SAC and CrossQ use the absolute gap between the two critics at the deterministic policy action. TQC reports the gap between per-critic mean quantile values and the quantile interquartile range. These diagnostics are not posterior uncertainty estimates or safety confidence measures.

## Inference

The trained seed is the inference unit (n=5 per algorithm--environment pair). S1 contains eight CrossQ/TQC-versus-SAC final-return contrasts across two environments and the two primary shifts. S2 contains twelve shift-minus-nominal support-coverage contrasts across three algorithms, two environments, and the same two primary shifts. Paired mean/median differences, paired wins/losses, 10,000-resample paired bootstrap intervals, paired t tests, Wilcoxon signed-rank tests, and Holm-adjusted p-values are reported. The p-value families are sensitivity summaries only; controller-superiority claims are not permitted.

## Claim boundary

Permitted interpretation is limited to continuous-control transfer/contextualization of the support-boundary diagnostic. Universal reinforcement-learning superiority, calibrated uncertainty, HIL validity, physical-flight safety, and operational deployment readiness are outside the evidence supplied by this benchmark.

## Provenance separation

A prior full-budget feasibility design is retained as historical provenance and is not pooled with this resource-constrained evidence block. The immutable historical YAML hash is `a90160b429ec4456549458c5152269572b5cfa7544700dde96d4c38353c1dafe`. The release-facing protocol preserves the scientific design while removing editorial-origin naming from the public project structure.
