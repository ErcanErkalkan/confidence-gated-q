from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.run_reliability_calibration_independent import (
    CONFIG_PATH,
    EXPECTED_SEEDS,
    IndependentCalibrationError,
    LOCK_YAML,
    composite_sha256,
    validate_lock_and_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_independent_calibration_lock_and_config_match() -> None:
    config = _config()
    digest = validate_lock_and_config(config)
    assert digest == composite_sha256()
    assert config["seeds"] == EXPECTED_SEEDS
    assert not set(config["seeds"]).intersection(range(10000, 10100))


def test_independent_calibration_rejects_seed_substitution() -> None:
    config = deepcopy(_config())
    config["seeds"][-1] = 12089
    with pytest.raises(IndependentCalibrationError, match="seed set mismatch"):
        validate_lock_and_config(config)


def test_independent_calibration_keeps_targets_separate() -> None:
    protocol = yaml.safe_load(LOCK_YAML.read_text(encoding="utf-8"))
    assert set(protocol["targets"]) >= {
        "action_correctness",
        "value_error",
        "target_pooling",
    }
    assert protocol["targets"]["target_pooling"] == (
        "action_and_value_targets_must_never_be_pooled"
    )
    assert protocol["availability_rules"]["auroc"] == (
        "unavailable_when_either_binary_class_is_absent"
    )


def test_instrumentation_rerun_preserves_locked_scientific_design() -> None:
    original = _config()
    rerun_path = CONFIG_PATH.parent / "independent_focal_instrumentation_rerun.yaml"
    rerun = yaml.safe_load(rerun_path.read_text(encoding="utf-8"))
    validate_lock_and_config(rerun)
    assert rerun["seeds"] == original["seeds"]
    assert rerun["envs"] == original["envs"]
    assert rerun["agents"] == original["agents"]
    assert rerun["evaluation"] == original["evaluation"]
    assert rerun["analysis"]["calibration_targets"] == original["analysis"][
        "calibration_targets"
    ]
    assert rerun["output_dir"] != original["output_dir"]
