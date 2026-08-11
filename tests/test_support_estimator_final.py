from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.aggregate_support_estimator_final import (  # noqa: E402
    build_budget_audit,
    build_calibration_summary,
    build_global_sensitivity,
    build_planned_contrasts,
)
from scripts.generate_support_final_configs import (  # noqa: E402
    CONFIG_ROOT,
    PROTOCOL,
    build_config,
    load_yaml,
)
from scripts.run_support_estimator_final import (  # noqa: E402
    SupportFinalError,
    validate_protocol,
)


def _synthetic_seed_metrics(protocol: dict) -> pd.DataFrame:
    offsets = {
        item["agent_id"]: index * 0.05
        for index, item in enumerate(protocol["agents"])
    }
    rows = []
    for environment_index, environment in enumerate(protocol["environments"]):
        for agent in protocol["agents"]:
            for seed_index, seed in enumerate(environment["seeds"]):
                score = (
                    environment_index
                    + offsets[agent["agent_id"]]
                    + 0.01 * seed_index
                    + 0.001 * offsets[agent["agent_id"]] * seed_index
                )
                rows.append(
                    {
                        "environment_key": environment["environment_key"],
                        "environment": environment["environment_name"],
                        "agent": agent["agent_id"],
                        "seed": seed,
                        "normalized_return_auc": score,
                        "success_auc": 0.5,
                        "failure_probability": 0.5,
                        "collision_rate": 0.1,
                        "risk_zone_rate": 0.1,
                        "mean_support_score": 0.5,
                        "mean_memory_action_correctness": 0.5,
                        "mean_neural_action_correctness": 0.5,
                        "mean_support_query_latency_seconds": 0.001,
                        "support_memory_bytes": 100.0,
                        "gradient_updates": 10,
                        "source_file": "synthetic.csv",
                        "source_column": "normalized_return_auc",
                    }
                )
    return pd.DataFrame(rows)


def test_protocol_has_disjoint_thirty_seed_blocks_and_generated_configs():
    protocol = load_yaml(PROTOCOL)
    validate_protocol(protocol)
    all_seeds = [
        int(seed)
        for environment in protocol["environments"]
        for seed in environment["seeds"]
    ]
    assert len(all_seeds) == len(set(all_seeds)) == 90
    for environment in protocol["environments"]:
        generated = load_yaml(
            CONFIG_ROOT / f"{environment['environment_key']}.yaml"
        )
        assert generated == build_config(protocol, environment)


def test_protocol_rejects_overlapping_final_seed_blocks():
    protocol = load_yaml(PROTOCOL)
    corrupted = deepcopy(protocol)
    corrupted["environments"][1]["seeds"][0] = corrupted["environments"][0][
        "seeds"
    ][0]
    with pytest.raises(SupportFinalError, match="overlap"):
        validate_protocol(corrupted)


def test_planned_holm_is_per_environment_and_global_is_primary_only():
    protocol = load_yaml(PROTOCOL)
    seed_metrics = _synthetic_seed_metrics(protocol)
    planned = build_planned_contrasts(seed_metrics, protocol)
    assert len(planned) == 36
    assert planned.groupby("environment_key").size().eq(12).all()
    assert np.isfinite(planned["paired_t_holm_p"]).all()
    global_sensitivity = build_global_sensitivity(planned)
    assert len(global_sensitivity) == 3
    assert global_sensitivity["does_not_replace_report_holm"].all()


def test_budget_audit_matches_all_agents_without_claiming_compute_identity():
    protocol = load_yaml(PROTOCOL)
    audit = build_budget_audit(_synthetic_seed_metrics(protocol), protocol)
    assert len(audit) == 27
    assert audit["audit_status"].eq("PASS").all()
    assert not audit["compute_budget_identical"].any()


def test_calibration_reports_single_class_and_missing_action_targets():
    rows = []
    for seed in (1, 2):
        for episode in range(4):
            rows.append(
                {
                    "environment_key": "application_goal_shift",
                    "environment": "application",
                    "agent": "exact_count_gate",
                    "seed": seed,
                    "support_score_mean": 0.2 + 0.1 * episode,
                    "failure_rate": 0.0,
                    "tabular_action_correct": np.nan,
                    "source_file": "synthetic.csv",
                }
            )
    result = build_calibration_summary(pd.DataFrame(rows))
    failure = result[result["target_type"].eq("episode_failure")].iloc[0]
    action = result[
        result["target_type"].eq("episode_memory_majority_correct")
    ].iloc[0]
    assert failure["availability"] == "not_available_single_class"
    assert np.isnan(failure["auroc"])
    assert action["availability"] == "not_available_no_analytic_optimal_action"
