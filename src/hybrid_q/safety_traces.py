"""Locked episode-level safety metrics for bounded UAV/SIL decision traces."""

from __future__ import annotations

from typing import Any

import numpy as np


class SafetyTraceMetricError(ValueError):
    pass


def _vector(values: Any, name: str, *, finite: bool = True) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise SafetyTraceMetricError(f"{name} must be a non-empty vector")
    if finite:
        numeric = array.astype(float)
        if not np.isfinite(numeric).all():
            raise SafetyTraceMetricError(f"{name} must be finite")
        return numeric
    return array


def trajectory_deviation(
    positions: Any, reference_positions: Any
) -> np.ndarray:
    """Return pointwise Euclidean trajectory deviation in metres."""

    positions_array = np.asarray(positions, dtype=float)
    reference_array = np.asarray(reference_positions, dtype=float)
    if (
        positions_array.ndim != 2
        or positions_array.shape[1] != 3
        or reference_array.shape != positions_array.shape
        or positions_array.shape[0] == 0
    ):
        raise SafetyTraceMetricError("positions and references must be matched n x 3 arrays")
    if not np.isfinite(positions_array).all() or not np.isfinite(reference_array).all():
        raise SafetyTraceMetricError("positions and references must be finite")
    return np.linalg.norm(positions_array - reference_array, axis=1)


def recovery_from_deviation(
    deviations: Any,
    timestamps: Any,
    *,
    onset_index: int,
    recovery_band_m: float,
    dwell_steps: int,
) -> dict[str, float | bool]:
    """Apply the locked dwell rule and right-censor unrecovered episodes."""

    deviation = _vector(deviations, "deviations")
    time = _vector(timestamps, "timestamps")
    if deviation.size != time.size:
        raise SafetyTraceMetricError("deviations and timestamps must have equal length")
    if np.any(np.diff(time) <= 0):
        raise SafetyTraceMetricError("timestamps must be strictly increasing")
    if onset_index < 0 or onset_index >= deviation.size:
        raise SafetyTraceMetricError("onset_index is outside the trace")
    if not np.isfinite(recovery_band_m) or recovery_band_m < 0:
        raise SafetyTraceMetricError("recovery_band_m must be finite and non-negative")
    if dwell_steps < 1:
        raise SafetyTraceMetricError("dwell_steps must be positive")

    inside = deviation <= float(recovery_band_m)
    last_start = deviation.size - dwell_steps
    for start in range(onset_index, last_start + 1):
        if bool(np.all(inside[start : start + dwell_steps])):
            return {
                "recovered": True,
                "recovery_time_seconds": float(time[start] - time[onset_index]),
                "censor_time_seconds": float("nan"),
            }
    return {
        "recovered": False,
        "recovery_time_seconds": float("nan"),
        "censor_time_seconds": float(time[-1] - time[onset_index]),
    }


def duration_from_samples(active: Any, timestep_seconds: float) -> float:
    """Return active-sample duration under the locked fixed-step convention."""

    flags = _vector(active, "active", finite=False)
    if not np.isfinite(timestep_seconds) or timestep_seconds <= 0:
        raise SafetyTraceMetricError("timestep_seconds must be finite and positive")
    if not np.isin(flags, [0, 1, False, True]).all():
        raise SafetyTraceMetricError("active must contain only binary values")
    return float(np.count_nonzero(flags.astype(bool)) * timestep_seconds)


def episode_safety_metrics(
    *,
    positions: Any,
    reference_positions: Any,
    timestamps: Any,
    perturbation_active: Any,
    saturation_active: Any,
    constraint_active: Any,
    near_miss_active: Any,
    timestep_seconds: float,
    recovery_band_m: float,
    dwell_steps: int,
) -> dict[str, float | int | bool]:
    """Calculate the complete locked Step 17 episode metric set."""

    deviation = trajectory_deviation(positions, reference_positions)
    time = _vector(timestamps, "timestamps")
    perturbation = _vector(
        perturbation_active, "perturbation_active", finite=False
    )
    vectors = [saturation_active, constraint_active, near_miss_active]
    if any(np.asarray(values).size != deviation.size for values in vectors):
        raise SafetyTraceMetricError("all episode vectors must have equal length")
    if time.size != deviation.size or perturbation.size != deviation.size:
        raise SafetyTraceMetricError("all episode vectors must have equal length")
    if not np.isin(perturbation, [0, 1, False, True]).all():
        raise SafetyTraceMetricError("perturbation_active must be binary")
    active_indices = np.flatnonzero(perturbation.astype(bool))
    if active_indices.size == 0:
        raise SafetyTraceMetricError("episode has no post-onset samples")
    onset_index = int(active_indices[0])
    if not perturbation[onset_index:].astype(bool).all():
        raise SafetyTraceMetricError("perturbation cannot deactivate after onset")
    post_deviation = deviation[onset_index:]
    recovery = recovery_from_deviation(
        deviation,
        time,
        onset_index=onset_index,
        recovery_band_m=recovery_band_m,
        dwell_steps=dwell_steps,
    )
    return {
        "sample_count": int(deviation.size),
        "post_onset_sample_count": int(post_deviation.size),
        "perturbation_onset_timestamp": float(time[onset_index]),
        "trajectory_deviation_rmse_m": float(
            np.sqrt(np.mean(np.square(post_deviation)))
        ),
        "maximum_trajectory_deviation_m": float(np.max(post_deviation)),
        **recovery,
        "saturation_duration_seconds": duration_from_samples(
            saturation_active, timestep_seconds
        ),
        "constraint_violation_duration_seconds": duration_from_samples(
            constraint_active, timestep_seconds
        ),
        "near_miss_duration_seconds": duration_from_samples(
            near_miss_active, timestep_seconds
        ),
        "saturation_sample_count": int(np.count_nonzero(saturation_active)),
        "constraint_violation_sample_count": int(
            np.count_nonzero(constraint_active)
        ),
        "near_miss_sample_count": int(np.count_nonzero(near_miss_active)),
    }
