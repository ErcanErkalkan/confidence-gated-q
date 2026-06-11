# Literature Positioning and Novelty Boundary

## Episodic and Memory-Augmented RL

Model-Free Episodic Control, Neural Episodic Control, Episodic Memory DQN,
Episodic Value Adjustment, and Generalizable Episodic Memory use richer memory
representations than the exact-state scalar mixtures studied here. The paper
does not claim the first tabular-neural hybrid or a replacement for these
methods.

## DQN Reliability

The study follows work on DQN instability and statistical uncertainty by:

- separating development from independent validation;
- retaining every configured seed;
- reporting paired seed-level effects;
- distinguishing mean-based and rank-based sensitivity conclusions;
- adding heavy-tail and catastrophic-outlier diagnostics.

## Fuzzy and Uncertainty-Aware Arbitration

The fuzzy adaptive gate is positioned as an interpretable soft-computing
baseline. It is related in motivation to fuzzy deep-RL systems for
knowledge-guided learning, constrained industrial control, and
uncertainty-aware resource management, but it is narrower: fixed membership
rules map exact support and a TD-residual proxy to a memory weight.

The proxy is not a calibrated posterior. The paper cites uncertainty-aware
collision avoidance to motivate cautious behavior in unfamiliar states while
keeping the present claim limited to measured arbitration behavior.

## Support Shift

The held-out-goal design is a controlled support-shift test. Offline-RL
methods such as Batch-Constrained Q-learning and Conservative Q-learning also
warn against unsupported values, but the online support-abstention rule here
is not claimed to be equivalent to either method.

## Abstention

Support abstention is a neutral rejection rule at exactly zero support. It is
not calibrated uncertainty, selective prediction, or a universal safety
mechanism. Its contribution is mechanistic: it tests whether refusing
unsupported neural preferences removes the observed FourRooms failure.
In the application case, rejection maps to a declared hold action; in the
generic compact tasks, it retains the original neutral tie behavior.

## Application-Oriented Diagnostic

The warehouse-style navigation task is application-inspired, not a real-world
deployment. It provides controlled recurring and shifted goals, collisions,
risk zones, and a safe hold action so that support, outcome, and cost metrics
can be evaluated together.

## Reproducibility Contribution

The defensible methodological contributions are:

- exact-step evaluation;
- read-only evaluation with restored agent RNG;
- explicit observation representations;
- development, validation, confirmation, replication, and diagnostic labels;
- raw-row software and environment provenance;
- unsupported-state, branch-use, abstention, memory, and decision-time metrics;
- paired bootstrap, Holm correction, Wilcoxon sensitivity, and heavy-tail
  diagnostics;
- executable result, artifact, and submission audits.

## Claims Not Made

- general RL superiority;
- benchmark-leading performance;
- calibrated uncertainty;
- robust out-of-distribution generalization;
- continuous-control or real-world validation;
- a complete safety guarantee or verified fallback controller;
- broad superiority over tabular Q-learning.
