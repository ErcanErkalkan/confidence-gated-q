from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_q.agents import AgentConfig, DQNAgent, DuelingQNetwork, QNetwork
from hybrid_q.complexity import (
    approximate_inference_flops,
    approximate_inference_macs,
    mapping_operation_estimate,
    trainable_parameter_count,
)
from scripts.audit_budget_equivalence import (
    _comparison_row,
    build_dqn_candidate_grid,
)
from scripts.audit_results import audit_duplicate_rows, audit_stability_logging


ROOT = Path(__file__).resolve().parents[1]


def test_exact_parameters_and_approximate_macs_are_architecture_specific():
    dqn = QNetwork(4, 5, 128)
    dueling = DuelingQNetwork(4, 5, 128)

    assert trainable_parameter_count(dqn) == 17_797
    assert trainable_parameter_count(dueling) == 17_926
    assert approximate_inference_macs(dqn) == 17_536
    assert approximate_inference_macs(dueling) == 17_664
    assert approximate_inference_flops(dqn) == 2 * 17_536


def test_fuzzy_and_same_input_crisp_overheads_are_separately_labeled():
    fuzzy = mapping_operation_estimate("fuzzy_triangular_five_rule")
    crisp = mapping_operation_estimate("same_input_crisp_threshold")

    assert fuzzy.arithmetic_flops == 26
    assert fuzzy.comparisons > crisp.comparisons
    assert crisp.arithmetic_flops == 0
    assert "same two normalized scalar inputs" in crisp.definition


def test_dqn_training_stability_records_finite_and_nonfinite_losses():
    agent = DQNAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(batch_size=1, replay_warmup=1),
    )
    state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(state, "s", 0, 1.0, state, "s", True)
    stability = agent.training_stability()
    assert np.isfinite(stability["training_loss_mean"])
    assert np.isfinite(stability["training_loss_max"])
    assert stability["nonfinite_loss_count"] == 0

    agent._record_training_loss(float("inf"))
    assert agent.training_stability()["nonfinite_loss_count"] == 1


def test_historical_loss_scope_and_future_nonfinite_loss_failure():
    historical = pd.DataFrame({"gradient_updates": [1]})
    violations, warnings, scope = audit_stability_logging(historical)
    assert not violations
    assert warnings
    assert scope == "metric/checkpoint audit only"

    future = pd.DataFrame(
        {
            "gradient_updates": [1],
            "training_loss_mean": [0.5],
            "training_loss_max": [float("inf")],
            "nonfinite_loss_count": [1],
            "completed_checkpoint_count": [2],
            "expected_checkpoint_count": [2],
        }
    )
    violations, _, scope = audit_stability_logging(future)
    assert scope == "metric/checkpoint/loss audit"
    assert any("non-finite logged losses" in item for item in violations)
    assert any("training_loss_max" in item for item in violations)


def test_duplicate_seed_checkpoint_rows_fail_closed():
    raw = pd.DataFrame(
        [
            {
                "environment": "env",
                "agent": "a",
                "seed": 0,
                "phase": "eval",
                "checkpoint": 10,
                "episode": 1,
            },
            {
                "environment": "env",
                "agent": "a",
                "seed": 0,
                "phase": "eval",
                "checkpoint": 10,
                "episode": 1,
            },
        ]
    )
    seed_metrics = pd.DataFrame(
        [
            {"environment": "env", "agent": "a", "seed": 0, "checkpoint": 10},
            {"environment": "env", "agent": "a", "seed": 0, "checkpoint": 10},
        ]
    )
    violations = audit_duplicate_rows(raw, seed_metrics)
    assert any("duplicate raw rows" in item for item in violations)
    assert any("duplicate seed/checkpoint rows" in item for item in violations)


def test_complete_dqn_candidate_grid_is_reproducible():
    grid = build_dqn_candidate_grid(
        ROOT / "configs/dqn_tuning_development.json", ROOT
    )
    assert len(grid) == 7
    assert grid["candidate"].is_unique
    assert set(grid["candidate"]) == {
        "vanilla_dqn_h64",
        "vanilla_dqn_h128",
        "vanilla_dqn_lr3e4_h128",
        "vanilla_dqn_buffer100k",
        "vanilla_dqn_target250",
        "vanilla_dqn_target1000",
        "double_dqn_lr3e4_h128",
    }
    assert set(grid["development_environment_count"]) == {7}


class _FakeSource:
    def __init__(self, agent: str, signature: str):
        self.agent = agent
        self.signature = signature

    def has(self, environment: str, agent: str) -> bool:
        return environment == "env" and agent == self.agent

    def summarize(self, environment: str, agent: str):
        return {
            "actual_seeds": [1, 2],
            "declared_seeds": [1, 2],
            "declared_steps": 100,
            "observed_steps": {"1": 100, "2": 100},
            "declared_schedule": [50, 100],
            "observed_schedules": [[50, 100]],
            "declared_eval_episodes": 10,
            "observed_eval_episode_counts": [10],
            "gradient_updates": {"1": 7, "2": 7},
            "raw_complete": True,
            "seed_metrics_consistent": True,
            "loss_logged": False,
            "agent_signature": self.signature,
            "runtime_signature": "same-runtime",
            "source_files": [f"results/{agent}/raw.csv"],
        }


def test_matching_interactions_do_not_imply_compute_identity():
    record = pd.Series(
        {
            "report_family": "family",
            "contrast_name": "left_vs_right",
            "environment_or_severity": "env",
            "left": "left",
            "right": "right",
            "metric": "return_auc",
            "evidence_class": "confirmatory",
            "source_file": "results/family/planned_contrasts.csv",
        }
    )
    row = _comparison_row(
        record,
        [_FakeSource("left", "algorithm-a"), _FakeSource("right", "algorithm-b")],
    )
    assert row["interaction_budget_matched"] is True
    assert row["compute_budget_identical"] is False
    assert row["historical_audit_scope"] == "metric/checkpoint audit only"
