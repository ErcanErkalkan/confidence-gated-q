import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from hybrid_q.agents import AgentConfig, HybridQAgent
from hybrid_q.calibration import binary_auroc, first_sustained_detection
from hybrid_q.encoding import ObservationEncoder
from hybrid_q.envs import make_env
from hybrid_q.experiment import RELIABILITY_DIAGNOSTIC_FIELDS, evaluate, run_config
from scripts.aggregate_reliability_calibration import (
    aggregate_reliability_calibration,
)


def _bandit_spec():
    return {
        "name": "shift-bandit",
        "id": "ReliabilityShiftBandit-v0",
        "kwargs": {
            "context_count": 9,
            "regime": "switch",
            "shift_after": 10,
            "pre_boundary": 0.5,
            "post_boundary": 0.25,
        },
        "eval_kwargs": {"regime": "switch"},
        "training_steps": 20,
        "max_steps": 1,
        "success_mode": "positive_terminal",
    }


def test_evaluation_diagnostics_use_oracle_without_mutating_estimators():
    env_spec = _bandit_spec()
    env = make_env(env_spec)
    encoder = ObservationEncoder(env.observation_space)
    agent = HybridQAgent(
        input_dim=encoder.input_dim,
        action_dim=env.action_space.n,
        seed=4,
        config=AgentConfig(
            replay_warmup=100,
            reliability_beta=0.1,
            reliability_prior_strength=5,
            reliability_epsilon=1e-6,
        ),
        gate_kind="reliability",
    )
    before = {
        "tabular_error": dict(agent.tabular_error),
        "neural_error": dict(agent.neural_error),
        "error_counts": dict(agent.error_counts),
        "global_tabular_error": agent.global_tabular_error,
        "global_neural_error": agent.global_neural_error,
        "gate_sum": agent.gate_sum,
        "gate_queries": agent.gate_queries,
        "rng": copy.deepcopy(agent.rng.bit_generator.state),
        "online": {
            name: tensor.detach().clone()
            for name, tensor in agent.online.state_dict().items()
        },
    }

    rows = evaluate(
        env_spec=env_spec,
        encoder=encoder,
        agent=agent,
        base_seed=4,
        checkpoint=15,
        episodes=8,
    )

    assert rows
    assert all(row["post_shift"] == 1.0 for row in rows)
    assert all(np.isfinite(row["optimal_action"]) for row in rows)
    assert all(np.isfinite(row["tabular_q_error"]) for row in rows)
    assert all(np.isfinite(row["neural_q_error"]) for row in rows)
    assert all(0.0 <= row["relative_reliability_score"] <= 1.0 for row in rows)
    assert dict(agent.tabular_error) == before["tabular_error"]
    assert dict(agent.neural_error) == before["neural_error"]
    assert dict(agent.error_counts) == before["error_counts"]
    assert agent.global_tabular_error == before["global_tabular_error"]
    assert agent.global_neural_error == before["global_neural_error"]
    assert agent.gate_sum == before["gate_sum"]
    assert agent.gate_queries == before["gate_queries"]
    assert agent.rng.bit_generator.state == before["rng"]
    for name, tensor in agent.online.state_dict().items():
        assert torch.equal(tensor, before["online"][name])
    env.close()


def test_reliability_fields_are_evaluation_only_in_raw_output(tmp_path):
    config = {
        "experiment_name": "reliability_diagnostic_test",
        "output_dir": str(tmp_path / "result"),
        "runtime": {
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "workers": 1,
            "allow_dirty_execution_inputs": True,
        },
        "seeds": [10000],
        "evaluation": {"interval_steps": 2, "episodes": 3},
        "analysis": {
            "analysis_status": "development_only_not_confirmatory",
            "planned_contrasts": [],
        },
        "envs": [{**_bandit_spec(), "training_steps": 4}],
        "agents": [
            {
                "name": "reliability",
                "kind": "reliability_gated",
                "params": {
                    "batch_size": 2,
                    "replay_warmup": 100,
                    "reliability_epsilon": 1e-6,
                },
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw = pd.read_csv(run_config(config_path))
    assert raw.loc[raw["phase"] == "train", RELIABILITY_DIAGNOSTIC_FIELDS].isna().all().all()
    always_available = [
        field for field in RELIABILITY_DIAGNOSTIC_FIELDS
        if field != "steps_since_shift"
    ]
    assert raw.loc[raw["phase"] == "eval", always_available].notna().all().all()
    assert raw.loc[raw["phase"] == "eval", "steps_since_shift"].isna().all()


def test_auroc_is_unavailable_for_one_class_and_tie_aware_otherwise():
    assert np.isnan(binary_auroc([0.1, 0.9], [1, 1]))
    assert binary_auroc([0.1, 0.9], [0, 1]) == 1.0
    assert binary_auroc([0.5, 0.5], [0, 1]) == 0.5


def test_first_sustained_detection_uses_post_shift_persistence():
    first, delay = first_sustained_detection(
        [5, 10, 15, 20, 25],
        [5, 2, 4, 4, 1],
        [5, 5, 5, 5, 5],
        shift_step=10,
        minimum_rows=5,
        consecutive_checkpoints=2,
    )
    assert first == 15
    assert delay == 5
    missing = first_sustained_detection(
        [10, 15], [2, 2], [5, 5], shift_step=10
    )
    assert all(np.isnan(value) for value in missing)


def _write_synthetic_calibration_family(tmp_path: Path):
    input_dir = tmp_path / "development"
    input_dir.mkdir()
    config = {
        "seeds": [10000, 10001],
        "envs": [
            {
                "name": "env",
                "id": "ReliabilityShiftBandit-v0",
                "kwargs": {"shift_after": 10},
            }
        ],
        "agents": [
            {
                "name": "rel",
                "kind": "reliability_gated",
                "params": {
                    "reliability_beta": 0.05,
                    "reliability_prior_strength": 5,
                    "reliability_epsilon": 1e-8,
                },
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rows = []
    blank = {field: np.nan for field in RELIABILITY_DIAGNOSTIC_FIELDS}
    rows.append(
        {
            "environment": "env",
            "agent": "rel",
            "seed": 10000,
            "phase": "train",
            "checkpoint": 1,
            "episode": 1,
            **blank,
        }
    )
    for seed in config["seeds"]:
        for checkpoint in (5, 10, 15, 20):
            for episode in range(8):
                tabular_better = episode % 2 == 0
                rows.append(
                    {
                        "environment": "env",
                        "agent": "rel",
                        "seed": seed,
                        "phase": "eval",
                        "checkpoint": checkpoint,
                        "episode": episode,
                        "tabular_greedy_action": int(tabular_better),
                        "neural_greedy_action": int(not tabular_better),
                        "mixed_selected_action": int(tabular_better),
                        "optimal_action": 1,
                        "tabular_action_correct": float(tabular_better),
                        "neural_action_correct": float(not tabular_better),
                        "relative_reliability_score": 0.8 if tabular_better else 0.2,
                        "predicted_better_branch": "tabular" if tabular_better else "neural",
                        "actually_better_branch": "tabular" if tabular_better else "neural",
                        "actually_better_branch_value": "neural" if tabular_better else "tabular",
                        "tabular_q_error": 0.2 if not tabular_better else 0.1,
                        "neural_q_error": 0.1 if not tabular_better else 0.2,
                        "post_shift": float(checkpoint >= 10),
                        "shift_region": "changed_optimal_action",
                        "steps_since_shift": max(0, checkpoint - 10) if checkpoint >= 10 else np.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(input_dir / "raw.csv", index=False)
    return input_dir, config_path


def test_reliability_aggregator_outputs_separate_targets(tmp_path):
    input_dir, config_path = _write_synthetic_calibration_family(tmp_path)
    output_dir = tmp_path / "out"
    table = tmp_path / "table.csv"
    figure = tmp_path / "figure.pdf"
    counts = aggregate_reliability_calibration(
        input_dir, config_path, output_dir, table, figure
    )
    assert counts["evaluation_rows"] == 64
    expected = {
        "branch_discrimination.csv",
        "calibration_bins.csv",
        "selective_risk.csv",
        "parameter_sensitivity.csv",
        "detection_delay.csv",
    }
    assert {path.name for path in output_dir.glob("*.csv")} == expected
    branch = pd.read_csv(output_dir / "branch_discrimination.csv")
    assert set(branch["target_type"]) == {"action_correctness", "value_error"}
    assert (branch["auroc_availability"] == "available").all()
    assert table.exists()
    assert figure.read_bytes().startswith(b"%PDF-")
