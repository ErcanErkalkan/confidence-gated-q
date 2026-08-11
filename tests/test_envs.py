import json

import numpy as np
import pandas as pd
import pytest

from hybrid_q.agents import AgentConfig, HybridQAgent
from hybrid_q.encoding import ObservationEncoder
from hybrid_q.envs import (
    ApplicationNavigationSupportShiftEnv,
    LocalizedRewardShiftNavigationEnv,
    ObservationShiftFourRoomsEnv,
    PyBulletUAVWaypointSupportShiftEnv,
    ReliabilityShiftBanditEnv,
    SensorizedPyBulletUAVWaypointEnv,
    StructuredFourRoomsEnv,
    TransitionDynamicsShiftFourRoomsEnv,
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
    assert evaluation["exact_support_coverage"].between(0, 1).all()
    assert evaluation["approximate_support_coverage"].between(0, 1).all()
    assert evaluation["neural_only_fallback_rate"].between(0, 1).all()
    assert "support_status" in evaluation
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


@pytest.mark.parametrize(
    ("environment", "action"),
    [
        (
            TransitionDynamicsShiftFourRoomsEnv(
                pre_shift_slip_probability=0.0,
                post_shift_slip_probability=0.0,
            ),
            0,
        ),
        (ObservationShiftFourRoomsEnv(slip_probability=0.0), 0),
        (
            LocalizedRewardShiftNavigationEnv(
                slip_probability=0.0,
            ),
            4,
        ),
    ],
)
def test_locked_shift_occurs_after_interaction_12000(environment, action):
    environment.reset(seed=1)
    environment.total_steps = 11999
    environment.steps = 0
    environment.agent_position = (1, 3)
    environment.goal_position = (7, 7)
    _, _, terminated, truncated, pre_info = environment.step(action)
    assert not terminated
    assert not truncated
    assert pre_info["post_shift"] is False
    assert environment.total_steps == 12000

    environment.steps = 0
    _, _, _, _, post_info = environment.step(action)
    assert post_info["post_shift"] is True
    assert post_info["steps_since_shift"] == 0
    assert environment.total_steps == 12001
    environment.close()


def test_shift_environment_pre_regimes_match_backward_compatible_baselines():
    transition_baseline = StructuredFourRoomsEnv(
        goal_split="all", slip_probability=0.05, max_steps=160
    )
    transition_shift = TransitionDynamicsShiftFourRoomsEnv()
    observation_baseline = StructuredFourRoomsEnv(
        goal_split="all", slip_probability=0.10, max_steps=160
    )
    observation_shift = ObservationShiftFourRoomsEnv()
    reward_baseline = ApplicationNavigationSupportShiftEnv(
        goal_split="deployment", slip_probability=0.05, max_steps=120
    )
    reward_shift = LocalizedRewardShiftNavigationEnv()

    for baseline, shifted in (
        (transition_baseline, transition_shift),
        (observation_baseline, observation_shift),
        (reward_baseline, reward_shift),
    ):
        baseline_observation, _ = baseline.reset(seed=23)
        shifted_observation, shifted_info = shifted.reset(seed=23)
        assert np.array_equal(baseline_observation, shifted_observation)
        assert shifted_info["post_shift"] is False
        baseline_step = baseline.step(0)
        shifted_step = shifted.step(0)
        assert np.array_equal(baseline_step[0], shifted_step[0])
        assert baseline_step[1:4] == shifted_step[1:4]
        baseline.close()
        shifted.close()


def test_transition_shift_changes_only_action_outcomes_and_states_recur():
    pre = TransitionDynamicsShiftFourRoomsEnv(
        shift_after=12000,
        pre_shift_slip_probability=0.0,
        post_shift_slip_probability=1.0,
    )
    post = TransitionDynamicsShiftFourRoomsEnv(
        shift_after=0,
        pre_shift_slip_probability=0.0,
        post_shift_slip_probability=1.0,
    )
    for environment in (pre, post):
        environment.reset(seed=0)
        environment.agent_position = (2, 2)
        environment.goal_position = (7, 7)
        environment.steps = 0
        environment.np_random = np.random.default_rng(0)

    pre_observation, pre_reward, _, _, pre_info = pre.step(0)
    post_observation, post_reward, _, _, post_info = post.step(0)
    assert pre_info["executed_action"] == 0
    assert post_info["executed_action"] != 0
    assert not np.array_equal(pre_observation, post_observation)
    assert pre_reward == post_reward == -0.01
    assert pre_info["latent_state_id"] == post_info["latent_state_id"]
    assert pre_info["shift_type"] == "transition_dynamics_shift"

    recurring = TransitionDynamicsShiftFourRoomsEnv(
        pre_shift_slip_probability=0.0,
        post_shift_slip_probability=0.0,
    )
    recurring.reset(seed=3)
    recurring.agent_position = (0, 0)
    recurring.goal_position = (7, 7)
    _, _, _, _, first = recurring.step(2)
    _, _, _, _, second = recurring.step(2)
    assert first["latent_state_id"] == second["latent_state_id"]
    assert first["next_latent_state_id"] == first["latent_state_id"]
    pre.close()
    post.close()
    recurring.close()


def test_observation_shift_preserves_latent_transition_law():
    latent = ObservationShiftFourRoomsEnv(
        shift_after=12000,
        slip_probability=0.25,
        post_shift_sensor_gain=0.55,
    )
    shifted = ObservationShiftFourRoomsEnv(
        shift_after=0,
        slip_probability=0.25,
        post_shift_sensor_gain=0.55,
    )
    for environment in (latent, shifted):
        environment.reset(seed=0)
        environment.agent_position = (2, 2)
        environment.goal_position = (7, 7)
        environment.steps = 0
        environment.np_random = np.random.default_rng(5)

    latent_step = latent.step(1)
    shifted_step = shifted.step(1)
    assert latent_step[1:4] == shifted_step[1:4]
    assert latent_step[4]["next_latent_state_id"] == shifted_step[4][
        "next_latent_state_id"
    ]
    assert not np.array_equal(latent_step[0], shifted_step[0])
    assert shifted_step[4]["shift_type"] == "observation_shift"
    assert np.any(
        np.asarray(shifted_step[4]["observation_perturbation"]) != 0.0
    )
    latent.close()
    shifted.close()


def test_localized_reward_shift_leaves_unaffected_regions_unchanged():
    pre = LocalizedRewardShiftNavigationEnv(
        shift_after=12000,
        slip_probability=0.0,
        pre_shift_risk_penalty=0.08,
        post_shift_risk_penalty=0.16,
    )
    post = LocalizedRewardShiftNavigationEnv(
        shift_after=0,
        slip_probability=0.0,
        pre_shift_risk_penalty=0.08,
        post_shift_risk_penalty=0.16,
    )
    for environment in (pre, post):
        environment.reset(seed=7)
        environment.agent_position = (1, 3)
        environment.goal_position = (7, 7)
        environment.steps = 0
        environment.np_random = np.random.default_rng(9)

    pre_unaffected = pre.step(2)
    post_unaffected = post.step(2)
    assert np.array_equal(pre_unaffected[0], post_unaffected[0])
    assert pre_unaffected[1] == post_unaffected[1] == -0.02
    assert post_unaffected[4]["shift_region"] == "unaffected"
    assert post_unaffected[4]["localized_shift_applied"] is False

    for environment in (pre, post):
        environment.agent_position = (1, 2)
        environment.goal_position = (7, 7)
        environment.steps = 0
        environment.np_random = np.random.default_rng(11)
    pre_localized = pre.step(1)
    post_localized = post.step(1)
    assert np.array_equal(pre_localized[0], post_localized[0])
    assert np.isclose(pre_localized[1], -0.10)
    assert np.isclose(post_localized[1], -0.18)
    assert post_localized[4]["shift_region"] == "localized_risk_cells"
    assert post_localized[4]["localized_shift_applied"] is True
    pre.close()
    post.close()


def test_locked_shift_ids_register_and_expose_required_metadata():
    ids = {
        "TransitionDynamicsShift-v0": TransitionDynamicsShiftFourRoomsEnv,
        "ObservationShift-v0": ObservationShiftFourRoomsEnv,
        "LocalizedRewardShift-v0": LocalizedRewardShiftNavigationEnv,
    }
    required = {
        "post_shift",
        "shift_type",
        "shift_severity",
        "shift_region",
        "latent_state_id",
        "optimal_action",
    }
    for environment_id, environment_type in ids.items():
        environment = make_env({"id": environment_id})
        assert isinstance(environment, environment_type)
        _, info = environment.reset(seed=17)
        assert required.issubset(info)
        environment.close()


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


@pytest.mark.skipif(
    not has_uav_backend(), reason="optional UAV backend is not installed"
)
def test_sensorized_uav_uses_sensor_estimates_and_low_level_commands():
    env = SensorizedPyBulletUAVWaypointEnv(
        target_split="deployment",
        physics="pyb_drag",
        action_repeat=1,
        max_steps=2,
        localization_latency_steps=1,
        localization_dropout_probability=0.0,
        range_dropout_probability=0.0,
        camera_dropout_probability=0.0,
        wind_force_std=0.0,
    )
    first, first_info = env.reset(seed=31)
    second, second_info = env.reset(seed=31)
    assert np.array_equal(first, second)
    assert first_info == second_info
    assert first.shape == (22,)
    assert first_info["observation_source"] == (
        "delayed_vio_imu_lidar_pinhole_target_detector"
    )
    assert first_info["control_interface"] == "attitude_collective_to_motor_rpm"
    assert first_info["trace_schema_version"] == "sensorized_sil_trace_v2"
    assert len(first_info["raw_sensor_observation"]) == 22
    assert len(first_info["latent_position"]) == 3
    assert len(first_info["latent_velocity"]) == 3
    assert len(first_info["target_state"]) == 3
    next_observation, reward, terminated, truncated, info = env.step(13)
    assert next_observation.shape == (22,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert np.isfinite(info["localization_error"])
    assert 0.0 <= info["motor_saturation"] <= 1.0
    assert info["control_interface"] == "attitude_collective_to_motor_rpm"
    assert info["selected_action"] == 13
    assert info["command_timestamp"] >= info["observation_timestamp"]
    assert info["effective_latency"] == pytest.approx(
        info["command_timestamp"] - info["observation_timestamp"]
    )
    assert isinstance(info["target_visibility"], bool)
    assert info["trajectory_error"] == pytest.approx(info["distance_to_goal"])
    assert isinstance(info["constraint_active"], bool)
    assert isinstance(info["recovery_event"], bool)
    assert isinstance(info["saturation_active"], bool)
    assert info["failure_stage"] in {
        "ongoing",
        "success",
        "collision_out_of_bounds",
        "collision_unstable",
        "collision_contact",
        "timeout_no_progress",
        "timeout_near_target",
        "timeout_partial_progress",
    }
    assert info["sensor_factor_noise"] is True
    assert info["sensor_factor_latency"] is True
    assert info["sensor_factor_localization_dropout"] is True
    assert info["sensor_factor_range_dropout"] is True
    assert info["sensor_factor_camera_dropout"] is True
    assert info["sensor_factor_visibility_occlusion"] is True
    env.close()


@pytest.mark.skipif(
    not has_uav_backend(), reason="optional UAV backend is not installed"
)
def test_sensorized_uav_factor_switches_disable_nonzero_magnitudes():
    env = SensorizedPyBulletUAVWaypointEnv(
        action_repeat=1,
        max_steps=1,
        lidar_noise_std=0.02,
        vio_position_noise_std=0.02,
        vio_velocity_noise_std=0.03,
        imu_attitude_noise_std=0.008,
        localization_latency_steps=2,
        localization_dropout_probability=0.03,
        range_dropout_probability=0.02,
        sensor_bias_walk_std=0.0007,
        camera_dropout_probability=0.05,
        sensor_noise_enabled=False,
        sensor_latency_enabled=False,
        localization_dropout_enabled=False,
        range_dropout_enabled=False,
        camera_dropout_enabled=False,
        visibility_occlusion_enabled=False,
    )
    _, info = env.reset(seed=41)
    assert env.lidar_noise_std == 0.0
    assert env.vio_position_noise_std == 0.0
    assert env.vio_velocity_noise_std == 0.0
    assert env.imu_attitude_noise_std == 0.0
    assert env.sensor_bias_walk_std == 0.0
    assert env.localization_latency_steps == 0
    assert env.localization_dropout_probability == 0.0
    assert env.range_dropout_probability == 0.0
    assert env.camera_dropout_probability == 0.0
    assert info["target_visibility"] is True
    assert not any(
        info[field]
        for field in (
            "sensor_factor_noise",
            "sensor_factor_latency",
            "sensor_factor_localization_dropout",
            "sensor_factor_range_dropout",
            "sensor_factor_camera_dropout",
            "sensor_factor_visibility_occlusion",
        )
    )
    env.close()


@pytest.mark.skipif(
    not has_uav_backend(), reason="optional UAV backend is not installed"
)
@pytest.mark.parametrize(
    ("observation_mode", "control_mode", "observation_size", "action_count"),
    [
        ("state_accessible", "low_level", 15, 27),
        ("sensorized", "high_level", 22, 7),
        ("state_accessible", "high_level", 15, 7),
        ("sensorized", "low_level", 22, 27),
    ],
)
def test_sensorized_uav_observation_and_control_modes_are_independent(
    observation_mode: str,
    control_mode: str,
    observation_size: int,
    action_count: int,
) -> None:
    env = SensorizedPyBulletUAVWaypointEnv(
        action_repeat=1,
        max_steps=1,
        wind_force_std=0.0,
        observation_mode=observation_mode,
        control_interface_mode=control_mode,
    )
    observation, info = env.reset(seed=51)
    assert observation.shape == (observation_size,)
    assert env.action_space.n == action_count
    assert info["learning_observation_mode"] == observation_mode
    assert info["control_interface_mode"] == control_mode
    action = 13 if control_mode == "low_level" else 6
    next_observation, reward, _, _, next_info = env.step(action)
    assert next_observation.shape == (observation_size,)
    assert np.isfinite(reward)
    assert next_info["learning_observation_mode"] == observation_mode
    assert next_info["control_interface_mode"] == control_mode
    env.close()
