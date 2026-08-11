from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from hybrid_q.safety_traces import (
    SafetyTraceMetricError,
    duration_from_samples,
    episode_safety_metrics,
    recovery_from_deviation,
    trajectory_deviation,
)
from scripts.lock_safety_trace_protocol import (
    DEFAULT_PROTOCOL,
    EXPECTED_SEEDS,
    load_and_validate,
)
from scripts.aggregate_safety_traces import build_episode_metrics
from scripts.run_safety_trace_reruns import DEFAULT_CONFIG, validate_config


def test_synthetic_trajectory_recovers_known_rmse_max_and_durations() -> None:
    deviations = np.asarray([0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    positions = np.column_stack((deviations, np.zeros((8, 2))))
    references = np.zeros((8, 3))
    metrics = episode_safety_metrics(
        positions=positions,
        reference_positions=references,
        timestamps=np.arange(1, 9, dtype=float) * 0.1,
        perturbation_active=[0, 1, 1, 1, 1, 1, 1, 1],
        saturation_active=[1, 1, 0, 1, 0, 0, 0, 0],
        constraint_active=[0, 1, 1, 0, 0, 0, 0, 0],
        near_miss_active=[0, 0, 1, 0, 0, 0, 0, 0],
        timestep_seconds=0.1,
        recovery_band_m=0.15,
        dwell_steps=6,
    )
    expected_rmse = math.sqrt((0.5**2 + 6 * 0.1**2) / 7)
    assert metrics["trajectory_deviation_rmse_m"] == pytest.approx(expected_rmse)
    assert metrics["maximum_trajectory_deviation_m"] == pytest.approx(0.5)
    assert metrics["recovered"] is True
    assert metrics["recovery_time_seconds"] == pytest.approx(0.1)
    assert math.isnan(float(metrics["censor_time_seconds"]))
    assert metrics["saturation_duration_seconds"] == pytest.approx(0.3)
    assert metrics["constraint_violation_duration_seconds"] == pytest.approx(0.2)
    assert metrics["near_miss_duration_seconds"] == pytest.approx(0.1)


def test_non_recovery_is_right_censored_without_horizon_imputation() -> None:
    result = recovery_from_deviation(
        [0.4, 0.3, 0.2, 0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        onset_index=1,
        recovery_band_m=0.15,
        dwell_steps=2,
    )
    assert result["recovered"] is False
    assert math.isnan(float(result["recovery_time_seconds"]))
    assert result["censor_time_seconds"] == pytest.approx(0.3)


def test_synthetic_deviation_and_saturation_helpers_are_exact() -> None:
    positions = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    references = np.zeros((2, 3))
    assert trajectory_deviation(positions, references).tolist() == [5.0, 2.0]
    assert duration_from_samples([0, 1, 1, 0], 1.0 / 60.0) == pytest.approx(
        2.0 / 60.0
    )
    with pytest.raises(SafetyTraceMetricError, match="binary"):
        duration_from_samples([0, 2], 0.1)


def test_protocol_digest_and_rerun_config_are_locked_to_new_seeds() -> None:
    protocol = load_and_validate()
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    design = validate_config(config, protocol)
    assert protocol["new_seeds"]["seeds"] == EXPECTED_SEEDS
    assert protocol["definitions"]["recovery"]["non_recovery_rule"] == (
        "right_censored_at_last_observed_post_onset_timestamp"
    )
    assert config["seeds"] == EXPECTED_SEEDS
    assert len(design) == 10
    assert design["audit_status"].eq("PASS").all()
    assert DEFAULT_PROTOCOL.with_suffix(".sha256").is_file()


def test_aggregator_recovers_known_synthetic_episode_metrics() -> None:
    deviations = [0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    rows = []
    for step, deviation in enumerate(deviations, start=1):
        rows.append(
            {
                "family": "synthetic_safety_family",
                "agent": "synthetic_agent",
                "seed": 16030,
                "checkpoint": 120,
                "episode": 1,
                "step": step,
                "post_action_latent_position_x": deviation,
                "post_action_latent_position_y": 0.0,
                "post_action_latent_position_z": 0.0,
                "post_action_reference_x": 0.0,
                "post_action_reference_y": 0.0,
                "post_action_reference_z": 0.0,
                "post_action_timestamp": step * 0.1,
                "post_action_perturbation_active": int(step >= 6),
                "post_action_saturation_active": int(step in {1, 2, 4}),
                "post_action_constraint_active": int(step in {2, 3}),
                "post_action_near_miss": int(step == 3),
                "control_timestep_seconds": 0.1,
                "perturbation_onset_step": 6,
                "nominal_reference_path_id": "linear_reset_position_to_episode_target_v1",
                "source_file": "synthetic.trace.csv.gz",
            }
        )
    result = build_episode_metrics(pd.DataFrame(rows), load_and_validate()).iloc[0]
    assert result["recovered"]
    assert result["recovery_time_seconds"] == pytest.approx(0.1)
    assert result["saturation_duration_seconds"] == pytest.approx(0.3)
    assert result["constraint_violation_duration_seconds"] == pytest.approx(0.2)
    assert result["near_miss_duration_seconds"] == pytest.approx(0.1)
