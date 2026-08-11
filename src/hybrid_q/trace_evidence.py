"""Compact, deterministic decision-trace evidence for sensorized SIL runs."""

from __future__ import annotations

import hashlib
from typing import Any, Hashable, Mapping

import numpy as np


SENSOR_DIMENSION = 22
LEGACY_TRACE_SCHEMA_VERSION = "sensorized_sil_trace_v1"
TRACE_SCHEMA_VERSION = "sensorized_sil_trace_v2"
SAFETY_TRACE_SCHEMA_VERSION = "sensorized_sil_trace_v3"
SUPPORTED_TRACE_SCHEMA_VERSIONS = {
    LEGACY_TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    SAFETY_TRACE_SCHEMA_VERSION,
}
SENSOR_COLUMNS = [f"raw_sensor_{index:02d}" for index in range(SENSOR_DIMENSION)]
LEGACY_TRACE_FIELDNAMES = [
    "trace_schema_version",
    "experiment_name",
    "environment",
    "agent",
    "seed",
    "phase",
    "checkpoint",
    "episode",
    "step",
    "latent_position_x",
    "latent_position_y",
    "latent_position_z",
    "latent_velocity_x",
    "latent_velocity_y",
    "latent_velocity_z",
    "target_x",
    "target_y",
    "target_z",
    *SENSOR_COLUMNS,
    "encoded_vector_hash",
    "exact_key_hash",
    "nearest_neighbor_key",
    "nearest_neighbor_key_hash",
    "nearest_neighbor_distance",
    "nearest_neighbor_greedy_action",
    "selected_action",
    "tabular_greedy_action",
    "neural_greedy_action",
    "support_score",
    "optimal_action",
    "reference_control_action",
    "reference_control_label",
    "tabular_action_correct",
    "tabular_reference_action_agreement",
    "localization_dropout",
    "range_dropout",
    "camera_dropout",
    "sensor_dropout",
    "observation_perturbed",
    "observation_timestamp",
    "command_timestamp",
    "effective_latency",
    "target_visibility",
]
TRACE_OUTCOME_FIELDNAMES = [
    "post_action_trajectory_error",
    "post_action_constraint_active",
    "post_action_risk_zone",
    "post_action_collision",
    "post_action_motor_saturation",
    "post_action_saturation_active",
    "post_action_target_visibility",
    "post_action_recovery_event",
    "post_action_terminated",
    "post_action_truncated",
    "post_action_success",
    "post_action_failure_stage",
]
TRACE_FIELDNAMES = [*LEGACY_TRACE_FIELDNAMES, *TRACE_OUTCOME_FIELDNAMES]
SAFETY_TRACE_OUTCOME_FIELDNAMES = [
    "perturbation_onset_step",
    "post_action_perturbation_active",
    "nominal_reference_path_id",
    "control_timestep_seconds",
    "post_action_timestamp",
    "post_action_latent_position_x",
    "post_action_latent_position_y",
    "post_action_latent_position_z",
    "post_action_reference_x",
    "post_action_reference_y",
    "post_action_reference_z",
    "post_action_trajectory_deviation",
    "post_action_minimum_obstacle_clearance",
    "post_action_near_miss",
]
SAFETY_TRACE_FIELDNAMES = [*TRACE_FIELDNAMES, *SAFETY_TRACE_OUTCOME_FIELDNAMES]


def trace_fieldnames(schema_version: str) -> list[str]:
    """Return the exact bounded schema requested by a trace config."""

    if schema_version == LEGACY_TRACE_SCHEMA_VERSION:
        return list(LEGACY_TRACE_FIELDNAMES)
    if schema_version == TRACE_SCHEMA_VERSION:
        return list(TRACE_FIELDNAMES)
    if schema_version == SAFETY_TRACE_SCHEMA_VERSION:
        return list(SAFETY_TRACE_FIELDNAMES)
    raise ValueError(f"Unsupported sensor trace schema: {schema_version}")


def _update_hash(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"none;")
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array:")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        digest.update(b"bytes:")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence:")
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _update_hash(digest, item)
    elif isinstance(value, (str, int, float, bool, np.generic)):
        digest.update(type(value).__name__.encode("ascii", errors="replace"))
        digest.update(b":")
        digest.update(str(value).encode("utf-8"))
        digest.update(b";")
    else:
        representation = repr(value).encode("utf-8")
        if len(representation) > 4096:
            raise ValueError("Unsupported unbounded value in trace hash")
        digest.update(type(value).__qualname__.encode("utf-8"))
        digest.update(b":")
        digest.update(representation)


def stable_trace_hash(value: Any) -> str:
    """Return a type-aware SHA-256 without serializing raw values to a row."""

    digest = hashlib.sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def _finite_or_blank(value: Any) -> float | str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if np.isfinite(number) else ""


def _action_or_blank(value: Any) -> int | str:
    number = _finite_or_blank(value)
    return int(number) if number != "" else ""


def _vector(info: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(info.get(key, []), dtype=np.float32).reshape(-1)
    if value.size != size or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain {size} finite values")
    return value


def nearest_support_evidence(agent: Any) -> tuple[Any, float | str, int | str]:
    """Read the last support query without mutating the estimator or Q table."""

    estimator = getattr(agent, "support_estimator", None)
    if estimator is None:
        return None, "", ""
    diagnostics = estimator.diagnostics()
    nearest_key = diagnostics.get("nearest_key")
    nearest_distance = _finite_or_blank(diagnostics.get("nearest_distance"))
    table = getattr(agent, "table", None)
    nearest_action: int | str = ""
    if nearest_key is not None and table is not None:
        values = table.get(nearest_key)
        if values is not None:
            array = np.asarray(values, dtype=float)
            if array.size and np.isfinite(array).all():
                nearest_action = int(np.argmax(array))
    return nearest_key, nearest_distance, nearest_action


def build_sensor_trace_row(
    *,
    context: Mapping[str, Any],
    observation: np.ndarray,
    encoded_vector: np.ndarray,
    exact_key: Hashable,
    selected_action: int,
    decision: Mapping[str, Any],
    info: Mapping[str, Any],
    agent: Any,
    outcome_info: Mapping[str, Any] | None = None,
    schema_version: str = TRACE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build one bounded row at decision time from pre-action state metadata."""

    if schema_version not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
        raise ValueError("Unknown requested sensor trace schema version")
    if info.get("trace_schema_version") not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
        raise ValueError("Sensor trace info is missing or has an unknown version")
    sensor = _vector(info, "raw_sensor_observation", SENSOR_DIMENSION)
    observed = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observed.size != SENSOR_DIMENSION or not np.array_equal(sensor, observed):
        raise ValueError("raw sensor trace does not match the agent observation")
    if np.any(sensor < -1.000001) or np.any(sensor > 1.000001):
        raise ValueError("raw sensor trace is outside the declared [-1, 1] range")
    position = _vector(info, "latent_position", 3)
    velocity = _vector(info, "latent_velocity", 3)
    target = _vector(info, "target_state", 3)
    flags = info.get("perturbation_flags", {})
    if not isinstance(flags, Mapping):
        raise ValueError("perturbation_flags must be a mapping")

    nearest_key, nearest_distance, nearest_action = nearest_support_evidence(agent)
    nearest_hash = stable_trace_hash(nearest_key) if nearest_key is not None else ""
    tabular_action = _action_or_blank(decision.get("tabular_greedy_action"))
    optimal_action = _action_or_blank(info.get("optimal_action"))
    reference_action = _action_or_blank(info.get("reference_control_action"))
    tabular_correct: float | str = ""
    if tabular_action != "" and optimal_action != "":
        tabular_correct = float(tabular_action == optimal_action)
    reference_agreement: float | str = ""
    if tabular_action != "" and reference_action != "":
        reference_agreement = float(tabular_action == reference_action)

    row = {
        "trace_schema_version": schema_version,
        **{name: context[name] for name in (
            "experiment_name",
            "environment",
            "agent",
            "seed",
            "phase",
            "checkpoint",
            "episode",
            "step",
        )},
        "latent_position_x": float(position[0]),
        "latent_position_y": float(position[1]),
        "latent_position_z": float(position[2]),
        "latent_velocity_x": float(velocity[0]),
        "latent_velocity_y": float(velocity[1]),
        "latent_velocity_z": float(velocity[2]),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "encoded_vector_hash": stable_trace_hash(
            np.asarray(encoded_vector, dtype=np.float32)
        ),
        "exact_key_hash": stable_trace_hash(exact_key),
        "nearest_neighbor_key": nearest_hash[:16],
        "nearest_neighbor_key_hash": nearest_hash,
        "nearest_neighbor_distance": nearest_distance,
        "nearest_neighbor_greedy_action": nearest_action,
        "selected_action": int(selected_action),
        "tabular_greedy_action": tabular_action,
        "neural_greedy_action": _action_or_blank(
            decision.get("neural_greedy_action")
        ),
        "support_score": _finite_or_blank(decision.get("support_score")),
        "optimal_action": optimal_action,
        "reference_control_action": reference_action,
        "reference_control_label": info.get("reference_control_label", ""),
        "tabular_action_correct": tabular_correct,
        "tabular_reference_action_agreement": reference_agreement,
        "localization_dropout": int(bool(flags.get("localization_dropout", False))),
        "range_dropout": int(bool(flags.get("range_dropout", False))),
        "camera_dropout": int(bool(flags.get("camera_dropout", False))),
        "sensor_dropout": int(bool(info.get("sensor_dropout", False))),
        "observation_perturbed": int(bool(info.get("observation_perturbed", False))),
        "observation_timestamp": _finite_or_blank(info.get("observation_timestamp")),
        "command_timestamp": _finite_or_blank(info.get("command_timestamp")),
        "effective_latency": _finite_or_blank(info.get("effective_latency")),
        "target_visibility": int(bool(info.get("target_visibility", False))),
    }
    if schema_version in {TRACE_SCHEMA_VERSION, SAFETY_TRACE_SCHEMA_VERSION}:
        outcome = outcome_info or {}
        row.update(
            {
                "post_action_trajectory_error": _finite_or_blank(
                    outcome.get("trajectory_error")
                ),
                "post_action_constraint_active": int(
                    bool(outcome.get("constraint_active", False))
                ),
                "post_action_risk_zone": int(
                    bool(outcome.get("risk_zone", False))
                ),
                "post_action_collision": int(
                    bool(outcome.get("collision", False))
                ),
                "post_action_motor_saturation": _finite_or_blank(
                    outcome.get("motor_saturation")
                ),
                "post_action_saturation_active": int(
                    bool(outcome.get("saturation_active", False))
                ),
                "post_action_target_visibility": int(
                    bool(outcome.get("target_visibility", False))
                ),
                "post_action_recovery_event": int(
                    bool(outcome.get("recovery_event", False))
                ),
                "post_action_terminated": int(
                    bool(outcome.get("terminated", False))
                ),
                "post_action_truncated": int(
                    bool(outcome.get("truncated", False))
                ),
                "post_action_success": int(
                    bool(outcome.get("success", False))
                ),
                "post_action_failure_stage": str(
                    outcome.get("failure_stage", "")
                ),
            }
        )
    if schema_version == SAFETY_TRACE_SCHEMA_VERSION:
        outcome = outcome_info or {}
        post_position = _vector(outcome, "latent_position", 3)
        reference = _vector(outcome, "nominal_reference_position", 3)
        row.update(
            {
                "perturbation_onset_step": int(
                    outcome.get("perturbation_onset_step", -1)
                ),
                "post_action_perturbation_active": int(
                    bool(outcome.get("perturbation_active", False))
                ),
                "nominal_reference_path_id": str(
                    outcome.get("nominal_reference_path_id", "")
                ),
                "control_timestep_seconds": _finite_or_blank(
                    outcome.get("control_timestep_seconds")
                ),
                "post_action_timestamp": _finite_or_blank(
                    outcome.get("post_action_timestamp")
                ),
                "post_action_latent_position_x": float(post_position[0]),
                "post_action_latent_position_y": float(post_position[1]),
                "post_action_latent_position_z": float(post_position[2]),
                "post_action_reference_x": float(reference[0]),
                "post_action_reference_y": float(reference[1]),
                "post_action_reference_z": float(reference[2]),
                "post_action_trajectory_deviation": _finite_or_blank(
                    outcome.get("trajectory_deviation")
                ),
                "post_action_minimum_obstacle_clearance": _finite_or_blank(
                    outcome.get("minimum_obstacle_clearance")
                ),
                "post_action_near_miss": int(
                    bool(outcome.get("near_miss", False))
                ),
            }
        )
    row.update(
        {column: float(sensor[index]) for index, column in enumerate(SENSOR_COLUMNS)}
    )
    if set(row) != set(trace_fieldnames(schema_version)):
        raise RuntimeError("Internal sensor trace schema mismatch")
    return row
