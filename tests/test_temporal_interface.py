from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hybrid_q.agents import AgentConfig, DRQNAgent
from hybrid_q.encoding import ObservationEncoder
from hybrid_q.envs import make_env
from hybrid_q.experiment import evaluate
from hybrid_q.temporal import SequenceReplayBuffer
from scripts.aggregate_temporal_interface_development import (
    build_architecture_results,
    select_temporal_model,
)
from scripts.run_temporal_interface_development import (
    CAPACITY_CONTROLS,
    DEFAULT_CONFIG,
    DEVELOPMENT_SEEDS,
    TEMPORAL_AGENTS,
    TemporalInterfaceConfigError,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = DEFAULT_CONFIG


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_sequence_replay_samples_contiguous_episode_bounded_sequences() -> None:
    replay = SequenceReplayBuffer(capacity=100, sequence_length=3, seed=4)
    for episode_offset in (0.0, 100.0):
        for step in range(5):
            state = np.asarray([episode_offset + step], dtype=np.float32)
            next_state = np.asarray([episode_offset + step + 1], dtype=np.float32)
            replay.add((state, 0, float(step), next_state, step == 4))
    states, _, _, next_states, dones = replay.sample(batch_size=4)
    assert states.shape == (4, 3, 1)
    assert np.all(np.diff(states[..., 0], axis=1) == 1.0)
    assert np.all(next_states[..., 0] - states[..., 0] == 1.0)
    assert not np.any(
        (states[..., 0].min(axis=1) < 50.0)
        & (states[..., 0].max(axis=1) > 50.0)
    )
    assert np.all(dones[:, :-1] == 0.0)


def test_drqn_hidden_state_resets_at_episode_boundary() -> None:
    agent = DRQNAgent(
        input_dim=3,
        action_dim=2,
        seed=2,
        config=AgentConfig(
            sequence_length=2,
            recurrent_hidden_size=8,
            replay_warmup=100,
        ),
    )
    agent.act(np.ones(3, dtype=np.float32), "s", epsilon=0.0)
    assert agent.capture_temporal_state() is not None
    agent.reset_episode(training=True)
    assert agent.capture_temporal_state() is None


def test_evaluation_restores_recurrent_state_and_does_not_update_agent() -> None:
    env_spec = {
        "name": "temporal_eval_isolation",
        "id": "StructuredFourRooms-v0",
        "kwargs": {"size": 7, "goal_split": "train", "max_steps": 2},
        "eval_kwargs": {},
        "max_steps": 2,
        "success_mode": "positive_terminal",
    }
    env = make_env(env_spec)
    encoder = ObservationEncoder(env.observation_space)
    env.close()
    agent = DRQNAgent(
        encoder.input_dim,
        action_dim=4,
        seed=9,
        config=AgentConfig(
            sequence_length=2,
            recurrent_hidden_size=8,
            replay_warmup=100,
        ),
    )
    state = np.zeros(encoder.input_dim, dtype=np.float32)
    agent.act(state, "training", epsilon=0.0)
    hidden_before = agent.capture_temporal_state()
    parameters_before = [item.detach().clone() for item in agent.online.parameters()]
    rng_before = copy.deepcopy(agent.rng.bit_generator.state)

    rows = evaluate(
        env_spec,
        encoder,
        agent,
        base_seed=9,
        checkpoint=2,
        episodes=2,
    )

    assert len(rows) == 2
    assert agent.environment_steps == 0
    assert len(agent.replay) == 0
    assert agent.rng.bit_generator.state == rng_before
    assert torch.equal(agent.capture_temporal_state(), hidden_before)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, agent.online.parameters())
    )


def test_frame_stack_and_filter_history_reset_between_episodes() -> None:
    base = {
        "id": "StructuredFourRooms-v0",
        "kwargs": {"size": 7, "goal_split": "train", "max_steps": 2},
    }
    stacked = make_env({**base, "frame_stack": 3})
    first, _ = stacked.reset(seed=5)
    assert first.shape == (12,)
    assert np.array_equal(first[:4], first[4:8])
    stacked.step(1)
    reset, _ = stacked.reset(seed=5)
    assert np.array_equal(reset, first)
    stacked.close()

    filtered = make_env({**base, "temporal_filter_alpha": 0.25})
    initial, _ = filtered.reset(seed=6)
    filtered.step(1)
    reset_filtered, _ = filtered.reset(seed=6)
    assert np.array_equal(reset_filtered, initial)
    filtered.close()


def test_matched_interface_configuration_invariants() -> None:
    manifest = validate_config(_config())
    assert len(manifest) == 10
    assert manifest["audit_status"].eq("PASS").all()
    assert set(manifest["seed_set"]) == {"15004;15005"}
    assert set(manifest["training_steps"]) == {240}
    assert set(manifest["checkpoint_schedule"]) == {"120;240"}
    assert set(manifest["evaluation_episodes"]) == {3}
    assert len(TEMPORAL_AGENTS) == len(CAPACITY_CONTROLS) == 3


def test_interface_validator_fails_on_non_interface_drift() -> None:
    config = _config()
    target = next(
        env
        for env in config["envs"]
        if env["name"] == "interface_A_state_low_level"
    )
    target["kwargs"]["wind_force_std"] = 0.5
    with pytest.raises(TemporalInterfaceConfigError):
        validate_config(config)


def test_temporal_development_seed_isolation() -> None:
    config = _config()
    assert config["seeds"] == DEVELOPMENT_SEEDS
    assert all(15000 <= seed <= 15099 for seed in config["seeds"])
    assert not any(16000 <= seed <= 16099 for seed in config["seeds"])
    assert config["analysis"]["final_seed_access"] == "prohibited"


def test_temporal_selection_is_deterministic_and_excludes_controls() -> None:
    rows = []
    candidates = sorted(TEMPORAL_AGENTS | CAPACITY_CONTROLS)
    for candidate_index, candidate in enumerate(candidates):
        for seed in DEVELOPMENT_SEEDS:
            rows.append(
                {
                    "environment": "architecture_sensorized_low_level",
                    "agent": candidate,
                    "seed": seed,
                    "normalized_return_auc": float(candidate_index),
                    "success_rate": 0.0,
                    "failure_probability": 1.0,
                    "collision_rate": 0.0,
                    "localization_error_mean": 0.1,
                    "episode_lower_tail_return_0_10": -1.0,
                    "inference_time_us_per_decision_mean": 5.0,
                    "gradient_updates": 10,
                    "source_row_count": 6,
                    "source_file": "synthetic.csv",
                }
            )
    summary = build_architecture_results(pd.DataFrame(rows))
    first = select_temporal_model(summary)["candidate_id"]
    second = select_temporal_model(
        summary.sample(frac=1.0, random_state=11)
    )["candidate_id"]
    assert first == second
    assert first in TEMPORAL_AGENTS
