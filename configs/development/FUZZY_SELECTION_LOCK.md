# Fuzzy Reliability Selection Lock

This file freezes the fuzzy reliability controller before independent
confirmatory seeds 1300-1329 are executed.

## Development Data

- Seeds: 0-4 only.
- Environments: FrozenLake 4x4 slippery, CliffWalking, and FourRooms 9 with
  held-out goals.
- Primary selection metric: lowest average environment rank for return AUC.
- Tie-breaker: largest pooled paired return-AUC difference from count gating.
- Candidate support scales: 5, 20, and 80.
- Candidate singleton vectors:
  - balanced: `[0.0, 0.35, 0.65, 0.75, 0.95]`
  - memory-safe: `[0.0, 0.25, 0.70, 0.75, 0.95]`
  - decisive: `[0.0, 0.20, 0.80, 0.70, 0.98]`

## Locked Controller

- Support scale: `tau=20`.
- Singleton vector: balanced.
- Inputs: exact-state support and relative estimator reliability.
- Relative reliability:
  `neural_td_error / (tabular_td_error + neural_td_error)`.
- Zero-support behavior: memory weight is exactly zero.
- Confirmatory parameters may not be changed after source execution commit.

The development configs preserve the full candidate set and selection rule.
Development results are not treated as confirmatory evidence.
