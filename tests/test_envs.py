import json

import numpy as np
import pandas as pd

from hybrid_q.agents import AgentConfig, HybridQAgent
from hybrid_q.encoding import ObservationEncoder
from hybrid_q.envs import StructuredFourRoomsEnv
from hybrid_q.envs import make_env
from hybrid_q.experiment import evaluate, run_config


def test_structured_goal_splits_are_disjoint():
    train = StructuredFourRoomsEnv(size=7, goal_split="train")
    test = StructuredFourRoomsEnv(size=7, goal_split="test")
    assert set(train.goals)
    assert set(test.goals)
    assert set(train.goals).isdisjoint(test.goals)


def test_step_budget_is_exact(tmp_path):
    output_dir = tmp_path / "result"
    config = {
        "experiment_name": "budget_test",
        "output_dir": str(output_dir),
        "runtime": {"torch_threads": 1, "torch_interop_threads": 1},
        "seeds": [0],
        "evaluation": {"interval_steps": 10, "episodes": 2},
        "envs": [
            {
                "id": "StructuredFourRooms-v0",
                "kwargs": {"size": 7, "goal_split": "train", "max_steps": 20},
                "eval_kwargs": {"goal_split": "test"},
                "training_steps": 25,
                "max_steps": 20,
                "success_mode": "positive_terminal",
            }
        ],
        "agents": [
            {
                "name": "tabular",
                "kind": "tabular",
                "params": {"epsilon_decay_steps": 20},
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw_path = run_config(config_path)
    raw = pd.read_csv(raw_path)
    assert raw["environment_steps"].max() == 25
    evaluation = raw[raw["phase"] == "eval"]
    assert set(evaluation["checkpoint"]) == {10, 20, 25}
    assert (
        evaluation["environment_steps"] == evaluation["checkpoint"]
    ).all()


def test_evaluation_does_not_change_rng_or_exact_state_support():
    env_spec = {
        "id": "StructuredFourRooms-v0",
        "kwargs": {
            "size": 7,
            "goal_split": "train",
            "max_steps": 20,
        },
        "eval_kwargs": {"goal_split": "test"},
        "max_steps": 20,
        "success_mode": "positive_terminal",
    }
    env = make_env(env_spec)
    encoder = ObservationEncoder(env.observation_space)
    agent = HybridQAgent(
        input_dim=encoder.input_dim,
        action_dim=env.action_space.n,
        seed=7,
        config=AgentConfig(replay_warmup=100),
        gate_kind="count",
    )
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = agent.rng.bit_generator.state
    expected_next = expected_rng.random()

    evaluate(
        env_spec=env_spec,
        encoder=encoder,
        agent=agent,
        base_seed=7,
        checkpoint=10,
        episodes=3,
    )

    assert agent.rng.random() == expected_next
    assert len(agent.table) == 0
    assert len(agent.counts) == 0
    assert agent.gate_queries == 0
    env.close()


def test_evaluation_restores_support_abstention_diagnostics():
    env_spec = {
        "id": "StructuredFourRooms-v0",
        "kwargs": {
            "size": 7,
            "goal_split": "train",
            "max_steps": 20,
        },
        "eval_kwargs": {"goal_split": "test"},
        "max_steps": 20,
        "success_mode": "positive_terminal",
    }
    env = make_env(env_spec)
    encoder = ObservationEncoder(env.observation_space)
    agent = HybridQAgent(
        input_dim=encoder.input_dim,
        action_dim=env.action_space.n,
        seed=9,
        config=AgentConfig(replay_warmup=100),
        gate_kind="support_abstain",
    )
    evaluate(
        env_spec=env_spec,
        encoder=encoder,
        agent=agent,
        base_seed=9,
        checkpoint=10,
        episodes=2,
    )
    assert agent.support_queries == 0
    assert agent.support_abstentions == 0
    env.close()
