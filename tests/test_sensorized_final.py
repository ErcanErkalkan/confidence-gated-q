from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from hybrid_q.trace_evidence import (
    LEGACY_TRACE_FIELDNAMES,
    LEGACY_TRACE_SCHEMA_VERSION,
)
from scripts.aggregate_sensor_aliasing import validate_trace_schema
from scripts.aggregate_sensorized_final import (
    AGENT_ORDER,
    CONDITION_ORDER,
    build_episode_outcomes,
    build_planned_contrasts,
)
from scripts.lock_sensor_final_protocol import (
    EXPECTED_SEEDS,
    load_and_validate,
)
from scripts.run_sensorized_final import DEFAULT_CONFIG, validate_config


def test_sensor_final_protocol_and_config_are_locked() -> None:
    protocol = load_and_validate()
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    manifest = validate_config(config, protocol)
    assert protocol["condition_selection"]["selected_isolated_condition"] == "latency_only"
    assert protocol["condition_selection"]["outcome_columns_prohibited"] == [
        "success_rate",
        "failure_rate",
        "return",
        "p_value",
    ]
    assert config["seeds"] == EXPECTED_SEEDS
    assert len(manifest) == 10
    assert manifest["interaction_budget_matched"].all()
    assert not manifest["compute_budget_identical"].any()


def test_legacy_v1_trace_schema_remains_readable() -> None:
    row = {column: "" for column in LEGACY_TRACE_FIELDNAMES}
    row.update(
        {
            "trace_schema_version": LEGACY_TRACE_SCHEMA_VERSION,
            "experiment_name": "legacy",
            "environment": "sensor",
            "agent": "agent",
            "seed": 1,
            "phase": "eval",
            "checkpoint": 1,
            "episode": 1,
            "step": 1,
            "latent_position_x": 0.0,
            "latent_position_y": 0.0,
            "latent_position_z": 0.5,
            "latent_velocity_x": 0.0,
            "latent_velocity_y": 0.0,
            "latent_velocity_z": 0.0,
            "target_x": 0.5,
            "target_y": 0.0,
            "target_z": 0.5,
            "encoded_vector_hash": "a" * 64,
            "exact_key_hash": "b" * 64,
            "observation_timestamp": 0.0,
            "command_timestamp": 0.1,
            "effective_latency": 0.1,
        }
    )
    for index in range(22):
        row[f"raw_sensor_{index:02d}"] = 0.0
    validate_trace_schema(pd.DataFrame([row]))


def _trace_row(step: int, *, final: bool) -> dict:
    return {
        "environment": "combined_executed_condition",
        "agent": "feed_forward_dqn",
        "seed": 16000,
        "phase": "eval",
        "checkpoint": 120,
        "episode": 1,
        "step": step,
        "latent_position_x": 0.0,
        "latent_position_y": 0.0,
        "latent_position_z": 0.5,
        "target_x": 0.5,
        "target_y": 0.0,
        "target_z": 0.5,
        "post_action_trajectory_error": 0.5 - 0.1 * step,
        "post_action_constraint_active": int(step == 1),
        "post_action_saturation_active": int(step == 1),
        "post_action_target_visibility": int(step == 2),
        "post_action_recovery_event": int(step == 2),
        "post_action_success": 0,
        "post_action_failure_stage": "timeout_partial_progress" if final else "ongoing",
    }


def test_episode_outcomes_recover_locked_durations_and_stage() -> None:
    result = build_episode_outcomes(
        pd.DataFrame([_trace_row(1, final=False), _trace_row(2, final=True)])
    )
    row = result.iloc[0]
    assert row["constraint_steps"] == 1
    assert row["constraint_duration_seconds"] == 1.0 / 60.0
    assert row["saturation_steps"] == 1
    assert row["recovery_event_count"] == 1
    assert row["target_visibility_fraction"] == 0.5
    assert row["failure_stage"] == "timeout_partial_progress"


def test_planned_contrasts_keep_nulls_and_apply_one_eight_row_holm_family() -> None:
    offsets = {
        "feed_forward_dqn": 0.0,
        "selected_temporal_drqn": 0.1,
        "fuzzy_relative_reliability": 0.0,
        "selected_approximate_support": -0.05,
        "sensorized_controller": 0.2,
    }
    rows = []
    for condition_index, condition in enumerate(CONDITION_ORDER):
        for agent in AGENT_ORDER:
            for seed in EXPECTED_SEEDS:
                rows.append(
                    {
                        "condition": condition,
                        "agent": agent,
                        "seed": seed,
                        "normalized_return_auc": (
                            offsets[agent]
                            + condition_index * 0.01
                            + (seed - 16000) * 0.0001
                        ),
                    }
                )
    result = build_planned_contrasts(pd.DataFrame(rows), load_and_validate())
    assert len(result) == 8
    assert result["n_pairs"].eq(30).all()
    assert result["report_holm_scope"].eq("all_eight_rows_together").all()
    null_rows = result[
        result["contrast"].eq("fuzzy_relative_reliability_vs_feed_forward_dqn")
    ]
    assert null_rows["ties"].eq(30).all()
    assert null_rows["paired_t_holm_p"].eq(1.0).all()
