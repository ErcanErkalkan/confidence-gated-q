from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_sensor_aliasing import (
    SensorAliasingError,
    aliasing_rows,
    fragmentation_rows,
    nearest_neighbor_disagreement_rows,
    support_correctness_rows,
    validate_trace_schema,
)
from hybrid_q.trace_evidence import (
    SENSOR_COLUMNS,
    TRACE_FIELDNAMES,
    TRACE_SCHEMA_VERSION,
    build_sensor_trace_row,
    stable_trace_hash,
)


def _row(
    index: int,
    *,
    position_x: float = 0.0,
    sensor_00: float = 0.0,
    exact_hash: str | None = None,
) -> dict:
    row = {column: "" for column in TRACE_FIELDNAMES}
    row.update(
        {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "experiment_name": "controlled",
            "environment": "synthetic",
            "agent": "hybrid",
            "seed": 15000,
            "phase": "eval",
            "checkpoint": 40,
            "episode": 1,
            "step": index + 1,
            "latent_position_x": position_x,
            "latent_position_y": 0.0,
            "latent_position_z": 0.5,
            "latent_velocity_x": 0.0,
            "latent_velocity_y": 0.0,
            "latent_velocity_z": 0.0,
            "target_x": 0.5,
            "target_y": 0.0,
            "target_z": 0.5,
            "encoded_vector_hash": f"{index + 1:064x}",
            "exact_key_hash": exact_hash or f"{index + 11:064x}",
            "nearest_neighbor_key": f"{index + 21:016x}",
            "nearest_neighbor_key_hash": f"{index + 21:064x}",
            "nearest_neighbor_distance": 0.01,
            "nearest_neighbor_greedy_action": 2,
            "selected_action": 2,
            "tabular_greedy_action": 2,
            "neural_greedy_action": 3,
            "support_score": 0.6,
            "optimal_action": np.nan,
            "reference_control_action": 2,
            "reference_control_label": "nominal_free_space_latent_controller",
            "tabular_action_correct": np.nan,
            "tabular_reference_action_agreement": 1.0,
            "localization_dropout": 0,
            "range_dropout": 0,
            "camera_dropout": 0,
            "sensor_dropout": 0,
            "observation_perturbed": 0,
            "observation_timestamp": 0.1,
            "command_timestamp": 0.2,
            "effective_latency": 0.1,
            "target_visibility": 1,
            "post_action_trajectory_error": 0.5,
            "post_action_constraint_active": 0,
            "post_action_risk_zone": 0,
            "post_action_collision": 0,
            "post_action_motor_saturation": 0.0,
            "post_action_saturation_active": 0,
            "post_action_target_visibility": 1,
            "post_action_recovery_event": 0,
            "post_action_terminated": 0,
            "post_action_truncated": 0,
            "post_action_success": 0,
            "post_action_failure_stage": "ongoing",
        }
    )
    row.update({column: 0.0 for column in SENSOR_COLUMNS})
    row["raw_sensor_00"] = sensor_00
    return row


def test_controlled_fragmentation_recovers_known_case() -> None:
    frame = pd.DataFrame(
        [
            _row(0, position_x=0.0, exact_hash="a" * 64),
            _row(1, position_x=0.02, exact_hash="b" * 64),
            _row(2, position_x=0.8, exact_hash="c" * 64),
        ]
    )
    result = fragmentation_rows(frame, latent_radius=0.05)
    assert result.loc[0, "similar_latent_pair_count"] == 1
    assert result.loc[0, "different_exact_key_pair_count"] == 1
    assert result.loc[0, "fragmentation_rate"] == 1.0


def test_controlled_aliasing_recovers_known_case() -> None:
    frame = pd.DataFrame(
        [
            _row(0, position_x=0.0, sensor_00=0.0),
            _row(1, position_x=0.5, sensor_00=0.01),
            _row(2, position_x=0.9, sensor_00=0.9),
        ]
    )
    result = aliasing_rows(
        frame,
        observation_radius=0.05,
        material_latent_distance=0.30,
    )
    assert result.loc[0, "similar_observation_pair_count"] == 1
    assert result.loc[0, "materially_different_latent_pair_count"] == 1
    assert result.loc[0, "aliased_pair_count"] == 1
    assert result.loc[0, "aliasing_rate"] == 1.0


def test_pair_limit_fails_closed() -> None:
    frame = pd.DataFrame([_row(index) for index in range(4)])
    with pytest.raises(SensorAliasingError, match="exceeds max_pairs"):
        fragmentation_rows(frame, latent_radius=1.0, max_pairs=2)


def test_action_disagreement_and_support_reference_proxy() -> None:
    first = _row(0)
    second = _row(1)
    second["tabular_greedy_action"] = 1
    second["tabular_reference_action_agreement"] = 0.0
    frame = pd.DataFrame([first, second])
    disagreement = nearest_neighbor_disagreement_rows(frame)
    assert disagreement.loc[0, "eligible_decision_count"] == 2
    assert disagreement.loc[0, "disagreement_count"] == 1
    assert disagreement.loc[0, "nearest_neighbor_action_disagreement_rate"] == 0.5
    relationship = support_correctness_rows(frame)
    assert set(relationship["correctness_target"]) == {
        "nominal_free_space_reference_action"
    }
    assert relationship["inferential_status"].eq(
        "reference_agreement_proxy_not_optimality"
    ).all()
    assert relationship["eligible_decision_count"].sum() == 2


def test_trace_schema_validation_and_latency_consistency() -> None:
    frame = pd.DataFrame([_row(0), _row(1)])
    validate_trace_schema(frame)
    frame.loc[1, "effective_latency"] = 0.5
    with pytest.raises(SensorAliasingError, match="contradicts"):
        validate_trace_schema(frame)


def test_compact_trace_row_hashes_keys_without_storing_arrays() -> None:
    key = ((22,), "<f4", np.zeros(22, dtype=np.float32).tobytes())
    estimator = SimpleNamespace(
        diagnostics=lambda: {
            "nearest_key": key,
            "nearest_distance": 0.0,
        }
    )
    agent = SimpleNamespace(
        support_estimator=estimator,
        table={key: np.asarray([0.0, 1.0, 0.0])},
    )
    observation = np.zeros(22, dtype=np.float32)
    info = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "latent_position": (0.0, 0.0, 0.5),
        "latent_velocity": (0.0, 0.0, 0.0),
        "target_state": (0.5, 0.0, 0.5),
        "raw_sensor_observation": tuple(observation),
        "reference_control_action": 1,
        "reference_control_label": "nominal_free_space_latent_controller",
        "perturbation_flags": {},
        "observation_timestamp": 0.0,
        "command_timestamp": 0.1,
        "effective_latency": 0.1,
        "target_visibility": True,
    }
    row = build_sensor_trace_row(
        context={
            "experiment_name": "controlled",
            "environment": "synthetic",
            "agent": "hybrid",
            "seed": 15000,
            "phase": "eval",
            "checkpoint": 40,
            "episode": 1,
            "step": 1,
        },
        observation=observation,
        encoded_vector=observation,
        exact_key=key,
        selected_action=1,
        decision={
            "tabular_greedy_action": 1,
            "neural_greedy_action": 2,
            "support_score": 0.75,
        },
        info=info,
        agent=agent,
        outcome_info={
            "trajectory_error": 0.4,
            "constraint_active": False,
            "risk_zone": False,
            "collision": False,
            "motor_saturation": 0.0,
            "saturation_active": False,
            "target_visibility": True,
            "recovery_event": False,
            "terminated": False,
            "truncated": False,
            "success": False,
            "failure_stage": "ongoing",
        },
    )
    assert row["exact_key_hash"] == stable_trace_hash(key)
    assert row["nearest_neighbor_key_hash"] == stable_trace_hash(key)
    assert row["nearest_neighbor_greedy_action"] == 1
    assert row["post_action_trajectory_error"] == pytest.approx(0.4)
    assert all(not isinstance(value, (np.ndarray, list, tuple)) for value in row.values())
