from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.aggregate_sensor_factorial import build_budget_audit
from scripts.run_sensor_factorial_development import (
    CONDITION_FACTORS,
    DEFAULT_CONFIG,
    FACTOR_FIELDS,
    SensorFactorialError,
    validate_factorial_config,
)


def _config() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_all_eight_conditions_change_only_declared_sensor_factors() -> None:
    config = _config()
    manifest = validate_factorial_config(config)
    assert list(manifest["condition"]) == list(CONDITION_FACTORS)
    assert manifest["audit_status"].eq("PASS").all()
    for row in manifest.to_dict("records"):
        active = {field for field in FACTOR_FIELDS if row[field] is True}
        assert active == CONDITION_FACTORS[row["condition"]]
    fixed_columns = [
        "seed_set",
        "training_steps",
        "checkpoint_interval",
        "evaluation_episodes",
        "max_steps",
    ]
    assert all(manifest[column].nunique() == 1 for column in fixed_columns)


def test_factorial_validation_rejects_nonfactor_drift() -> None:
    config = copy.deepcopy(_config())
    condition = next(env for env in config["envs"] if env["name"] == "noise_only")
    condition["kwargs"]["wind_force_std"] = 0.5
    with pytest.raises(SensorFactorialError, match="non-factor settings differ"):
        validate_factorial_config(config)


def test_factorial_validation_rejects_cross_factor_contamination() -> None:
    config = copy.deepcopy(_config())
    condition = next(
        env for env in config["envs"] if env["name"] == "camera_dropout_only"
    )
    condition["kwargs"]["range_dropout_enabled"] = True
    with pytest.raises(SensorFactorialError, match="factor isolation mismatch"):
        validate_factorial_config(config)


def test_factorial_validation_rejects_final_seed() -> None:
    config = copy.deepcopy(_config())
    config["seeds"] = [15002, 16000]
    with pytest.raises(SensorFactorialError, match="seeds must"):
        validate_factorial_config(config)


def test_budget_audit_recovers_matched_synthetic_design() -> None:
    config = _config()
    rows = []
    for condition in CONDITION_FACTORS:
        for agent in (
            "dqn",
            "fuzzy_relative_reliability",
            "selected_approximate_support",
            "sensorized_controller",
        ):
            for seed in (15002, 15003):
                rows.append(
                    {
                        "environment": condition,
                        "agent": agent,
                        "seed": seed,
                        "phase": "train",
                        "checkpoint": 80,
                        "environment_steps": 80,
                    }
                )
                for checkpoint in (40, 80):
                    for _episode in (1, 2):
                        rows.append(
                            {
                                "environment": condition,
                                "agent": agent,
                                "seed": seed,
                                "phase": "eval",
                                "checkpoint": checkpoint,
                                "environment_steps": checkpoint,
                            }
                        )
    audit = build_budget_audit(pd.DataFrame(rows), config)
    assert len(audit) == 32
    assert audit["audit_status"].eq("PASS").all()


def test_factorial_config_path_is_development_only() -> None:
    path = Path(DEFAULT_CONFIG).as_posix()
    assert "sensor_factorial_development" in path
    config = _config()
    assert config["analysis"]["final_seed_access"] == "prohibited"
    assert config["analysis"]["planned_contrasts"] == []
