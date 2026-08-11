from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_sensor_nondegenerate_feasibility import (
    EXPECTED_SEEDS,
    load_and_validate,
    select_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sensor_nondegenerate_lock_and_config_match() -> None:
    protocol, config, digest = load_and_validate()
    assert config["analysis"]["protocol_sha256"] == digest
    assert config["seeds"] == EXPECTED_SEEDS
    assert not set(config["seeds"]).intersection(
        protocol["seed_lock"]["independent_final"]["seeds"]
    )


def _summary(candidate_values: dict[str, dict[tuple[str, str], float]]) -> pd.DataFrame:
    rows = []
    references = {
        "low_level": "sensorized_motor_controller",
        "high_level": "velocity_setpoint_controller",
    }
    for candidate_id, values in candidate_values.items():
        for (interface, agent), success in values.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "control_interface": interface,
                    "agent": agent,
                    "success_mean": success,
                }
            )
        for interface, reference in references.items():
            if (interface, reference) not in values:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "control_interface": interface,
                        "agent": reference,
                        "success_mean": 0.2,
                    }
                )
    return pd.DataFrame(rows)


def test_feasibility_selection_is_ordered_and_deterministic() -> None:
    protocol, _, _ = load_and_validate()
    values = {
        "horizon120_budget6000": {
            ("low_level", "feed_forward_dqn"): 0.00,
            ("low_level", "selected_temporal_drqn"): 0.00,
            ("high_level", "feed_forward_dqn"): 0.20,
            ("high_level", "selected_temporal_drqn"): 0.30,
        },
        "horizon240_budget12000": {
            ("low_level", "feed_forward_dqn"): 0.10,
            ("low_level", "selected_temporal_drqn"): 0.20,
            ("high_level", "feed_forward_dqn"): 0.40,
            ("high_level", "selected_temporal_drqn"): 0.50,
        },
        "horizon360_budget24000": {
            ("low_level", "feed_forward_dqn"): 0.20,
            ("low_level", "selected_temporal_drqn"): 0.30,
            ("high_level", "feed_forward_dqn"): 0.60,
            ("high_level", "selected_temporal_drqn"): 0.70,
        },
    }
    selected = select_candidate(_summary(values), protocol)
    assert selected["selected_candidate_id"] == "horizon120_budget6000"
    assert selected["selected_factorial_interface"] == "high_level"
    assert selected["p_values_used"] is False


def test_feasibility_selection_fails_closed_on_universal_floor() -> None:
    protocol, _, _ = load_and_validate()
    values = {
        candidate["candidate_id"]: {
            (interface, agent): 0.0
            for interface in ("low_level", "high_level")
            for agent in ("feed_forward_dqn", "selected_temporal_drqn")
        }
        for candidate in protocol["feasibility_phase"]["candidates"]
    }
    selected = select_candidate(_summary(values), protocol)
    assert selected["selection_status"] == "no_eligible_candidate"
    assert selected["selected_candidate_id"] is None
