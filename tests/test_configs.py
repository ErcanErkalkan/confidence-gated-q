import json
from pathlib import Path

from hybrid_q.config import load_config
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
    "fuzzy_reliability_confirmatory_30seed.yaml",
    "fuzzy_reliability_shift_confirmatory_30seed.yaml",
    "development/fuzzy_risk_selection.yaml",
    "development/fuzzy_reliability_selection.yaml",
    "development/fuzzy_rule_selection.yaml",
    "development/fuzzy_component_screen.yaml",
    "development/fuzzy_reliability_shift_selection.yaml",
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
                spec["id"] == "PyBulletUAVWaypointSupportShift-v0"
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
