import json

import numpy as np
import pandas as pd
import pytest

from hybrid_q.agents import AgentConfig, HybridQAgent
from hybrid_q.encoding import ObservationEncoder
from hybrid_q.envs import (
    ApplicationNavigationSupportShiftEnv,
    PyBulletUAVWaypointSupportShiftEnv,
    ReliabilityShiftBanditEnv,
    StructuredFourRoomsEnv,
    has_uav_backend,
)
from hybrid_q.envs import make_env, resolve_env_id
from hybrid_q.experiment import evaluate, run_config


def test_structured_goal_splits_are_disjoint():
    train = StructuredFourRoomsEnv(size=7, goal_split="train")
    test = StructuredFourRoomsEnv(size=7, goal_split="test")
    assert set(train.goals)
    assert set(test.goals)
    assert set(train.goals).isdisjoint(test.goals)


def test_application_hold_penalty_and_risk_metadata():
    env = ApplicationNavigationSupportShiftEnv(
        goal_split="train",
        slip_probability=0.0,
        hold_penalty=0.05,
        lambda_collision=1.5,
        lambda_idle=0.2,
    )
    env.reset(seed=3)
    _, reward, _, _, info = env.step(4)
    assert np.isclose(reward, -0.07)
    assert info["idle"] is True
    assert info["lambda_collision"] == 1.5
    assert info["lambda_idle"] == 0.2
    env.close()


def test_application_navigation_goal_shift_and_seed_are_deterministic():
    train = ApplicationNavigationSupportShiftEnv(goal_split="train")
    test = ApplicationNavigationSupportShiftEnv(goal_split="test")
    assert set(train.goals).isdisjoint(test.goals)
    first_observation, first_info = train.reset(seed=13)
    second_observation, second_info = train.reset(seed=13)
    assert np.array_equal(first_observation, second_observation)
    assert first_info == second_info
    first_step = train.step(0)
    train.reset(seed=13)
    second_step = train.step(0)
    assert np.array_equal(first_step[0], second_step[0])
    assert first_step[1:] == second_step[1:]


def test_reliability_shift_bandit_changes_the_optimal_action():
    env = ReliabilityShiftBanditEnv(
        context_count=5,
        regime="switch",
        shift_after=1,
        pre_boundary=0.75,
        post_boundary=0.25,
    )
    env.context_index = 2
    _, pre_reward, _, _, pre_info = env.step(0)
    env.context_index = 2
    _, post_reward, _, _, post_info = env.step(1)
    assert pre_reward == 1.0
    assert post_reward == 1.0
    assert pre_info["post_shift"] is False
    assert post_info["post_shift"] is True


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


def test_application_support_shift_emits_extended_metrics(tmp_path):
    output_dir = tmp_path / "application"
    config = {
        "experiment_name": "application_metric_test",
        "output_dir": str(output_dir),
        "runtime": {"torch_threads": 1, "torch_interop_threads": 1},
        "seeds": [0],
        "evaluation": {"interval_steps": 10, "episodes": 4},
        "envs": [
            {
                "id": "ApplicationNavigationSupportShift-v0",
                "kwargs": {
                    "goal_split": "train",
                    "slip_probability": 0.0,
                    "max_steps": 20,
                },
                "eval_kwargs": {"goal_split": "test"},
                "training_steps": 20,
                "max_steps": 20,
                "success_mode": "positive_terminal",
            }
        ],
        "agents": [
            {
                "name": "fuzzy_support_adaptive",
                "kind": "fuzzy_support_adaptive_gate",
                "params": {
                    "batch_size": 2,
                    "replay_warmup": 100,
                    "fuzzy_abstain_zero_support": False,
                },
            }
        ],
    }
    config_path = tmp_path / "application.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw = pd.read_csv(run_config(config_path))
    evaluation = raw[raw["phase"] == "eval"]
    assert evaluation["unsupported_state_ratio"].gt(0).all()
    assert evaluation["adaptive_alpha_mean"].between(0, 1).all()
    assert evaluation["inference_time_us_per_decision_mean"].gt(0).all()
    assert evaluation["memory_cost_states"].ge(0).all()
    assert set(evaluation["selected_branch"]).issubset(
        {"memory", "neural", "mixed", "abstention"}
    )


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


def test_compact_environment_variants_reset_and_encode():
    specs = [
        {
            "id": "FrozenLake-v1",
            "kwargs": {"map_name": "4x4", "is_slippery": True},
        },
        {
            "id": "FrozenLake-v1",
            "kwargs": {"map_name": "8x8", "is_slippery": True},
        },
        {"id": "CliffWalking-v1"},
        {"id": "Taxi-v3"},
    ]
    for spec in specs:
        env = make_env(spec)
        observation, _ = env.reset(seed=3)
        encoded = ObservationEncoder(env.observation_space).encode(observation)
        assert encoded.vector.ndim == 1
        assert encoded.vector.size > 0
        env.close()
    assert resolve_env_id("Taxi-v3") in {"Taxi-v3", "Taxi-v4"}


def test_minigrid_variants_use_explicit_fully_observable_images():
    for env_id in (
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-6x6-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-FourRooms-v0",
    ):
        env = make_env(
            {
                "id": env_id,
                "observation": "fully_observable_image",
            }
        )
        observation, _ = env.reset(seed=4)
        assert isinstance(observation, np.ndarray)
        assert observation.ndim == 3
        encoded = ObservationEncoder(env.observation_space).encode(observation)
        assert encoded.vector.size == int(np.prod(observation.shape))
        env.close()


@pytest.mark.skipif(
    not has_uav_backend(), reason="optional UAV backend is not installed"
)
def test_pybullet_uav_support_shift_reset_step_and_seed():
    env = PyBulletUAVWaypointSupportShiftEnv(
        target_split="deployment",
        physics="pyb_drag",
        action_repeat=1,
        max_steps=2,
        wind_force_std=0.0,
    )
    first, first_info = env.reset(seed=21)
    second, second_info = env.reset(seed=21)
    assert np.array_equal(first, second)
    assert first_info == second_info
    assert first.shape == (15,)
    next_observation, reward, terminated, truncated, info = env.step(6)
    assert next_observation.shape == (15,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["physics_backend"] == "gym-pybullet-drones"
    env.close()
