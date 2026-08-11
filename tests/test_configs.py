import json
from pathlib import Path

from hybrid_q.config import load_config
from hybrid_q.agents import create_agent
from hybrid_q.encoding import ObservationEncoder
from hybrid_q.envs import has_uav_backend, make_env


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "confirmatory_extended_compact.json",
    "minigrid_extended_diagnostic.json",
    "support_abstention_replication.json",
    "dqn_tuning_development.json",
    "dqn_strong_validation.json",
    "smoke_application_navigation_case_study.json",
    "application_navigation_case_study.json",
    "adaptive_gate_compact_validation.json",
    "cost_support_metrics.json",
    "approximate_support_baseline_validation.json",
    "stronger_baseline_validation.json",
    "fuzzy_sensitivity_ablation.json",
    "quick_reproduction_smoke.yaml",
    "strong_baselines/double_dqn_30seed.yaml",
    "strong_baselines/dueling_double_dqn_30seed.yaml",
    "strong_baselines/a2c_or_ppo_protocol.yaml",
    "approx_support/knn_support_30seed.yaml",
    "approx_support/feature_distance_support_30seed.yaml",
    "fuzzy_ablation/fuzzy_ablation_30seed.yaml",
    "application_risk_variants_30seed.yaml",
    "uav_pybullet_30seed.yaml",
    "uav_pybullet_smoke.yaml",
    "uav_sensorized_motor_30seed.yaml",
    "uav_sensorized_motor_smoke.yaml",
    "fuzzy_reliability_confirmatory_30seed.yaml",
    "fuzzy_reliability_shift_confirmatory_30seed.yaml",
    "development/fuzzy_risk_selection.yaml",
    "development/fuzzy_reliability_selection.yaml",
    "development/fuzzy_rule_selection.yaml",
    "development/fuzzy_component_screen.yaml",
    "development/fuzzy_reliability_shift_selection.yaml",
    "diagnostic_extensions/reliability_calibration/development_fractional.yaml",
    "diagnostic_extensions/smoke/transition_dynamics_shift.yaml",
    "diagnostic_extensions/smoke/observation_shift.yaml",
    "diagnostic_extensions/smoke/localized_reward_shift.yaml",
    "diagnostic_extensions/smoke/support_estimators.yaml",
    "diagnostic_extensions/smoke/sensor_aliasing.yaml",
    "diagnostic_extensions/sensor_factorial_development/matched_conditions.yaml",
    "diagnostic_extensions/temporal_interface_development/matched_design.yaml",
    "diagnostic_extensions/support_development/matched_candidates.yaml",
)


def test_asoc_configs_load_with_unique_environment_and_agent_names():
    for name in CONFIGS:
        config = load_config(ROOT / "configs" / name)
        assert config["experiment_name"]
        assert config["output_dir"].startswith(
            ("results/", ".quick_repro/", ".uav_smoke/")
        )
        assert config["analysis"]["analysis_status"]
        environment_names = [
            spec.get("name", spec["id"]) for spec in config["envs"]
        ]
        agent_names = [spec["name"] for spec in config["agents"]]
        assert len(environment_names) == len(set(environment_names))
        assert len(agent_names) == len(set(agent_names))


def test_every_added_environment_can_reset_with_declared_encoding():
    seen = set()
    for name in CONFIGS:
        config = load_config(ROOT / "configs" / name)
        for spec in config["envs"]:
            if (
                spec["id"]
                in {
                    "PyBulletUAVWaypointSupportShift-v0",
                    "SensorizedPyBulletUAVWaypoint-v0",
                }
                and not has_uav_backend()
            ):
                continue
            key = (
                spec["id"],
                json.dumps(spec.get("kwargs", {}), sort_keys=True),
                spec.get("observation"),
            )
            if key in seen:
                continue
            seen.add(key)
            env = make_env(spec)
            observation, _ = env.reset(seed=17)
            assert observation is not None
            env.close()


def test_yaml_config_loader_supports_quick_smoke():
    config = load_config(ROOT / "configs" / "quick_reproduction_smoke.yaml")
    assert config["experiment_name"] == "quick_reproduction_smoke"
    assert config["seeds"] == [42, 43]


def test_support_estimator_smoke_is_development_only() -> None:
    config = load_config(
        ROOT / "configs/diagnostic_extensions/smoke/support_estimators.yaml"
    )
    assert config["seeds"] == [13000, 13001]
    assert all(13000 <= seed <= 13099 for seed in config["seeds"])
    assert not any(14000 <= seed <= 14099 for seed in config["seeds"])
    assert config["analysis"]["planned_contrasts"] == []
    assert "smoke_only" in config["analysis"]["analysis_status"]
    env = make_env(config["envs"][0])
    encoder = ObservationEncoder(env.observation_space)
    for spec in config["agents"]:
        agent = create_agent(
            spec["kind"],
            input_dim=encoder.input_dim,
            action_dim=env.action_space.n,
            seed=config["seeds"][0],
            params=spec["params"],
        )
        assert agent is not None
    env.close()


def test_sensor_aliasing_smoke_uses_only_registered_development_seeds() -> None:
    config = load_config(
        ROOT / "configs/diagnostic_extensions/smoke/sensor_aliasing.yaml"
    )
    assert config["seeds"] == [15000, 15001]
    assert all(15000 <= seed <= 15099 for seed in config["seeds"])
    assert not any(16000 <= seed <= 16099 for seed in config["seeds"])
    assert config["trace_logging"] == {
        "enabled": True,
        "phases": ["eval"],
        "compression": "gzip",
        "schema_version": "sensorized_sil_trace_v1",
    }
    assert config["analysis"]["planned_contrasts"] == []


def test_reliability_calibration_fraction_uses_only_reserved_development_seeds():
    config = load_config(
        ROOT
        / "configs/diagnostic_extensions/reliability_calibration/development_fractional.yaml"
    )
    assert config["seeds"] == [10000, 10001, 10002, 10003, 10004]
    assert all(10000 <= seed <= 10099 for seed in config["seeds"])
    assert not any(12000 <= seed <= 12099 for seed in config["seeds"])
    assert len(config["agents"]) == 24
    for agent in config["agents"]:
        params = agent["params"]
        beta_index = [0.02, 0.05, 0.10, 0.20].index(
            params["reliability_beta"]
        )
        lambda_index = [1, 5, 10, 20].index(
            params["reliability_prior_strength"]
        )
        epsilon_index = [1e-8, 1e-6, 1e-4].index(
            params["reliability_epsilon"]
        )
        assert (beta_index + lambda_index + epsilon_index) % 2 == 0


def test_public_uav_terminology_uses_bounded_simulation_language() -> None:
    public_files = (
        "README.md",
        "REPRODUCIBILITY.md",
        "PROVENANCE.md",
        ".zenodo.json",
    )
    prohibited = (
        "sensorized software-in-the-loop validation",
        "sensorized crazyflie sil validation",
        "sensorized uav sil validation",
        "sensorized sil validation",
        "physics-based crazyflie validation",
        "physics-based uav validation",
        "physics-based uav external validation",
        "uav external validation",
        "intermediate validation layer between grid diagnostics and hardware flight",
        "main sensorized sil validation",
    )
    for relative in public_files:
        text = (ROOT / relative).read_text(encoding="utf-8").lower()
        for phrase in prohibited:
            assert phrase not in text, f"{relative}: {phrase}"

    combined = " ".join(
        (ROOT / relative).read_text(encoding="utf-8").lower()
        for relative in public_files
    )
    assert "diagnostic" in combined
    assert "hardware" in combined or "flight" in combined

