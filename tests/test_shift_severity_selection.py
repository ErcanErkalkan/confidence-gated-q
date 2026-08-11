from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.lock_protocol import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_SEED_REGISTRY,
    load_and_validate,
)
from scripts.run_shift_severity_development import (  # noqa: E402
    DEFAULT_CONFIG,
    SeverityDevelopmentError,
    _load_yaml,
    validate_config,
)
from scripts.select_shift_severities import (  # noqa: E402
    select_mechanism_severity,
)


def test_locked_severity_config_constructs_and_uses_development_seeds_only():
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    config = _load_yaml(DEFAULT_CONFIG)
    validate_config(config, protocol)
    assert len(config["envs"]) == 9
    assert {seed for env in config["envs"] for seed in env["seeds"]} == set(
        range(11000, 11030)
    )


def test_severity_config_rejects_final_seed():
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    config = deepcopy(_load_yaml(DEFAULT_CONFIG))
    config["envs"][0]["seeds"][0] = 12000
    with pytest.raises(SeverityDevelopmentError, match="seed mismatch|final"):
        validate_config(config, protocol)


def test_mechanical_selection_uses_median_distance_then_locked_order():
    rows = []
    for severity, value in (("low", 0.25), ("middle", 0.75), ("high", 0.9)):
        rows.extend(
            {
                "severity_id": severity,
                "post_shift_success_auc": value,
            }
            for _ in range(30)
        )
    selected, audit = select_mechanism_severity(
        pd.DataFrame(rows), ["low", "middle", "high"]
    )
    assert selected == "low"
    assert audit.loc[audit["selected"], "severity_id"].item() == "low"
    assert not audit.loc[audit["severity_id"] == "high", "eligible"].item()


def test_mechanical_selection_fails_closed_when_all_candidates_degenerate():
    frame = pd.DataFrame(
        [
            {"severity_id": severity, "post_shift_success_auc": value}
            for severity, value in (("floor", 0.0), ("ceiling", 1.0))
            for _ in range(30)
        ]
    )
    with pytest.raises(SeverityDevelopmentError, match="no eligible severity"):
        select_mechanism_severity(frame, ["floor", "ceiling"])
