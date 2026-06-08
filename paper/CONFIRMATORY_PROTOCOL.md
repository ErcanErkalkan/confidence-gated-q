# Analysis Protocol and Claim Classes

## Superseded Exploratory Families

Earlier exploratory and sensitivity result families were removed after the
ASOC evidence set was frozen. They are not used by the manuscript, figures,
tables, verified claims, or current reproduction workflow. The archived
the previous archived release preserves the historical snapshot.

## ASOC Development Selection

`configs/dqn_tuning_development.json` uses only seeds `0-4` across seven
compact environments. The frozen selection rule is:

1. lowest mean environment rank in return AUC;
2. lowest worst-environment rank;
3. fewer mean gradient updates;
4. lexicographic agent name.

The selected candidate is `vanilla_dqn_buffer100k`. No validation,
confirmatory, replication, or MiniGrid diagnostic seed informed this choice.

## Independent DQN Validation

`configs/dqn_strong_validation.json` uses compact seeds `600-629` and MiniGrid
seeds `700-709`. It compares the selected vanilla DQN with the original
Double-DQN comparator. This family is labeled
`independent_validation_after_development_selection`.

## Confirmatory Task Expansion

`configs/confirmatory_extended_compact.json` uses seeds `500-529` on seven
compact environments. Planned contrasts are:

- count gate versus validated DQN;
- count gate versus tabular Q-learning;
- count gate versus fixed mixing;
- support abstention versus count gating.

This family is labeled `confirmatory_task_expansion`.

## Confirmatory Replication After Diagnostic Discovery

`configs/support_abstention_replication.json` uses new compact seeds `300-329`
and MiniGrid seeds `400-429`. The support-abstention hypothesis was discovered
after the original support diagnostic, so this family is not described as
part of the original frozen protocol. It is labeled
`confirmatory_replication_after_diagnostic_discovery`.

## Extended MiniGrid Diagnostic

`configs/minigrid_extended_diagnostic.json` uses seeds `500-509` on six fully
observable MiniGrid layouts. It is labeled
`post_hoc_extended_minigrid_diagnostic`.

## Application Navigation Case

`configs/application_navigation_case_study.json` uses seeds `600-629` for one
controlled warehouse-style navigation environment. Training uses recurring
goals and deployment evaluation mixes recurring and held-out goals. Planned
contrasts compare the fuzzy adaptive gate with count gating, support
abstention, validated DQN, and tabular learning. This family is labeled
`confirmatory_application_case_study`.

## Adaptive-Gate Compact Validation

`configs/adaptive_gate_compact_validation.json` uses seeds `700-729` on
FrozenLake 4x4, CliffWalking, and held-out-goal FourRooms 9. It evaluates the
same fuzzy planned contrasts without using these seeds for method selection.
This family is labeled `confirmatory_adaptive_gate_validation`.

## Cost and Support Analysis

`configs/cost_support_metrics.json` uses seeds `800-809` on application
navigation and FourRooms 9. It reports support, branch use, abstention,
exact-memory lower bounds, and action-decision time. This family is labeled
`descriptive_cost_and_support_analysis`; timing claims are descriptive.

## Statistical Rules

- Primary outcome: evaluation-return AUC.
- Pairing unit: common environment and seed.
- Estimates: mean and median paired differences.
- Uncertainty: percentile bootstrap 95% interval with fixed analysis RNG.
- Primary test: paired t test with Holm correction over each configured
  planned-contrast family.
- Sensitivity test: paired Wilcoxon signed-rank test with Holm correction.
- Additional diagnostics: win/loss/tie counts, Cohen's dz, rank-biserial
  effect size, skew, excess kurtosis, and robust catastrophic outliers.
- No failed or unfavorable seed is removed.
- Evaluation occurs at exact environment-step checkpoints and does not modify
  training state or support.
- Application and adaptive-gate claims use only generated summary and planned
  contrast files; manually invented numerical claims are prohibited.
