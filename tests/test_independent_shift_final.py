from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_results import audit_independent_shifts
from scripts.aggregate_independent_shifts import (
    POST_SHIFT_CHECKPOINTS,
    build_planned_contrasts,
    detection_delay_rows,
    normalized_auc,
)
from scripts.lock_protocol import DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY, load_and_validate
from scripts.run_independent_shift_final import (
    CONFIGS,
    EXPECTED,
    EXPECTED_AGENTS,
    FinalShiftError,
    load_yaml,
    validate_final_config,
    validate_prerequisites,
)


def test_final_configs_match_locked_seeds_budgets_and_agents():
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    validate_prerequisites(protocol)
    observed_seeds: set[int] = set()
    for mechanism, path in CONFIGS.items():
        config = load_yaml(path)
        validate_final_config(config, mechanism)
        seeds = set(config["envs"][0]["seeds"])
        assert len(seeds) == 30
        assert not observed_seeds.intersection(seeds)
        observed_seeds.update(seeds)
        assert {agent["name"] for agent in config["agents"]} == EXPECTED_AGENTS
    assert observed_seeds == set(range(12000, 12090))


def test_final_config_tampering_fails_closed():
    mechanism = "observation_shift"
    config = copy.deepcopy(load_yaml(CONFIGS[mechanism]))
    config["envs"][0]["seeds"][-1] = 12060
    with pytest.raises(FinalShiftError, match="seed set mismatch"):
        validate_final_config(config, mechanism)


def _synthetic_seed_metrics() -> pd.DataFrame:
    rows = []
    offsets = {
        "tabular": -0.03,
        "dqn": -0.01,
        "count_gated_tau_20": 0.00,
        "relative_reliability_fuzzy": 0.02,
        "same_input_crisp": 0.01,
    }
    for mechanism_index, (mechanism, expected) in enumerate(EXPECTED.items()):
        for agent in sorted(EXPECTED_AGENTS):
            for index, seed in enumerate(expected["seeds"]):
                wobble = 0.002 * np.sin(index + mechanism_index)
                rows.append(
                    {
                        "mechanism_id": mechanism,
                        "agent": agent,
                        "seed": seed,
                        "normalized_return_auc": 0.5 + offsets[agent] + wobble,
                    }
                )
    return pd.DataFrame(rows)


def test_locked_holm_family_has_six_separate_generator_rows():
    contrasts = build_planned_contrasts(_synthetic_seed_metrics())
    assert len(contrasts) == 6
    assert contrasts["mechanism_id"].nunique() == 3
    assert contrasts.groupby("mechanism_id").size().eq(2).all()
    assert set(contrasts["report_family"]) == {"independent_shift_primary"}
    assert np.all(contrasts["paired_t_holm_p"] >= contrasts["paired_t_p"])
    assert np.all(contrasts["wilcoxon_holm_p"] >= contrasts["wilcoxon_p"])


def test_detection_delay_uses_first_of_two_consecutive_correct_checkpoints():
    rows = []
    for checkpoint, correct in zip(POST_SHIFT_CHECKPOINTS, [12, 13, 15] + [0] * 10):
        rows.append(
            {
                "mechanism_id": "transition_dynamics_shift",
                "agent": "relative_reliability_fuzzy",
                "seed": 12000,
                "checkpoint": checkpoint,
                "branch_eligible": 20,
                "branch_correct": correct,
                "branch_correctness_rate": correct / 20,
                "source_file": "synthetic.csv",
            }
        )
    result = detection_delay_rows(pd.DataFrame(rows)).iloc[0]
    assert bool(result["available"])
    assert bool(result["detected"])
    assert result["detection_checkpoint"] == 12000
    assert result["detection_delay_interactions"] == 0


def test_detection_delay_reports_unavailable_without_oracle_rows():
    frame = pd.DataFrame(
        {
            "mechanism_id": ["localized_multistep_reward_or_policy_shift"] * 13,
            "agent": ["relative_reliability_fuzzy"] * 13,
            "seed": [12060] * 13,
            "checkpoint": POST_SHIFT_CHECKPOINTS,
            "branch_eligible": [0] * 13,
            "branch_correct": [0] * 13,
            "branch_correctness_rate": [np.nan] * 13,
            "source_file": ["synthetic.csv"] * 13,
        }
    )
    result = detection_delay_rows(frame).iloc[0]
    assert not bool(result["available"])
    assert not bool(result["detected"])
    assert "unavailable" in result["reason"]


def test_detection_delay_identifies_missing_final_branch_logging():
    frame = pd.DataFrame(
        {
            "mechanism_id": ["transition_dynamics_shift"] * 13,
            "agent": ["relative_reliability_fuzzy"] * 13,
            "seed": [12000] * 13,
            "checkpoint": POST_SHIFT_CHECKPOINTS,
            "branch_eligible": [0] * 13,
            "branch_correct": [0] * 13,
            "branch_correctness_rate": [np.nan] * 13,
            "source_file": ["synthetic.csv"] * 13,
        }
    )
    result = detection_delay_rows(frame).iloc[0]
    assert not bool(result["available"])
    assert "did not log branch diagnostics" in result["reason"]


def test_normalized_auc_requires_exact_locked_post_shift_schedule():
    values = np.linspace(0.2, 0.8, len(POST_SHIFT_CHECKPOINTS))
    assert normalized_auc(POST_SHIFT_CHECKPOINTS, values) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="coverage mismatch"):
        normalized_auc(POST_SHIFT_CHECKPOINTS[:-1], values[:-1])


def test_combined_independent_shift_result_audit_passes():
    report = audit_independent_shifts(
        Path("results/diagnostic_extensions/final_shifts"),
        Path("tables/table_independent_shift_replication.csv"),
        Path("figures/fig_independent_shift_replication.pdf"),
    )
    assert report["status"] == "PASS", report["violations"]
