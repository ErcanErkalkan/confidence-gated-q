from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_breakpoints(breakpoints: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(item) for item in breakpoints)
    if len(values) != 3:
        raise ValueError("membership breakpoints must contain exactly three values")
    if not all(np.isfinite(values)) or not (
        0.0 <= values[0] < values[1] < values[2] <= 1.0
    ):
        raise ValueError(
            "membership breakpoints must be finite, strictly increasing, and in [0, 1]"
        )
    return values


def three_memberships(
    value: float,
    breakpoints: Sequence[float],
    shape: str,
) -> tuple[float, float, float]:
    """Return low/medium/high memberships for one normalized scalar.

    Triangular memberships use the three declared peaks. Shoulder memberships
    use the low and high breakpoints as logistic midpoints (slope 12), with the
    middle breakpoint declaring and validating the ordering of the overlap.
    Inputs outside [0, 1] are clipped so perturbation diagnostics are defined at
    the domain boundary.
    """

    low_point, middle_point, high_point = _validated_breakpoints(breakpoints)
    item = float(np.clip(value, 0.0, 1.0))
    if shape == "triangular":
        low = (
            1.0
            if item <= low_point
            else float(
                np.clip((middle_point - item) / (middle_point - low_point), 0.0, 1.0)
            )
        )
        medium = float(
            (item - low_point) / (middle_point - low_point)
            if low_point < item <= middle_point
            else (high_point - item) / (high_point - middle_point)
            if middle_point < item < high_point
            else 0.0
        )
        high = (
            1.0
            if item >= high_point
            else float(
                np.clip((item - middle_point) / (high_point - middle_point), 0.0, 1.0)
            )
        )
        return low, float(np.clip(medium, 0.0, 1.0)), high
    if shape == "shoulder":
        low = float(1.0 / (1.0 + np.exp(12.0 * (item - low_point))))
        high = float(1.0 / (1.0 + np.exp(-12.0 * (item - high_point))))
        medium = float(max(0.0, 1.0 - max(low, high)))
        return low, medium, high
    raise ValueError("membership shape must be triangular or shoulder")


def reliability_memberships(
    value: float,
    breakpoints: Sequence[float],
    shape: str,
) -> tuple[float, float]:
    """Collapse declared low/medium/high sets to low/high reliability weights."""

    low, medium, high = three_memberships(value, breakpoints, shape)
    low_weight = low + 0.5 * medium
    high_weight = high + 0.5 * medium
    total = low_weight + high_weight
    if total <= 1e-12:
        return 0.5, 0.5
    return float(low_weight / total), float(high_weight / total)


def crisp_reliability_gate(
    support: float,
    reliability: float,
    *,
    support_threshold: float,
    reliability_threshold: float,
    gate_min: float,
    gate_max: float,
) -> float:
    """Map the two normalized inputs with a same-input hard threshold gate."""

    values = (
        support,
        reliability,
        support_threshold,
        reliability_threshold,
        gate_min,
        gate_max,
    )
    if not all(np.isfinite(float(item)) for item in values):
        raise ValueError("crisp gate inputs and parameters must be finite")
    if not (0.0 <= support_threshold <= 1.0 and 0.0 <= reliability_threshold <= 1.0):
        raise ValueError("crisp thresholds must be in [0, 1]")
    if gate_min > gate_max:
        raise ValueError("gate_min must not exceed gate_max")
    return float(
        gate_max
        if support >= support_threshold and reliability >= reliability_threshold
        else gate_min
    )


def fuzzy_reliability_gate(
    support: float,
    reliability: float,
    *,
    membership_shape: str,
    reliability_membership_shape: str | None = None,
    support_breakpoints: Sequence[float],
    reliability_breakpoints: Sequence[float],
    consequents: Sequence[float],
    gate_min: float,
    gate_max: float,
) -> float:
    """Evaluate the declared five-rule fuzzy mapping on two normalized inputs."""

    consequent_values = tuple(float(item) for item in consequents)
    if len(consequent_values) != 5 or not all(np.isfinite(consequent_values)):
        raise ValueError("fuzzy consequents must contain five finite values")
    if not (np.isfinite(support) and np.isfinite(reliability)):
        raise ValueError("fuzzy mapping inputs must be finite")
    if not (np.isfinite(gate_min) and np.isfinite(gate_max)) or gate_min > gate_max:
        raise ValueError("fuzzy gate bounds must be finite and ordered")
    low_support, medium_support, high_support = three_memberships(
        support, support_breakpoints, membership_shape
    )
    low_reliability, high_reliability = reliability_memberships(
        reliability,
        reliability_breakpoints,
        reliability_membership_shape or membership_shape,
    )
    rules = (
        (low_support, consequent_values[0]),
        (medium_support * low_reliability, consequent_values[1]),
        (medium_support * high_reliability, consequent_values[2]),
        (high_support * low_reliability, consequent_values[3]),
        (high_support * high_reliability, consequent_values[4]),
    )
    total = sum(weight for weight, _ in rules)
    value = (
        sum(weight * consequence for weight, consequence in rules) / total
        if total > 1e-12
        else gate_min
    )
    return float(np.clip(value, gate_min, gate_max))
