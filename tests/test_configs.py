import json
from pathlib import Path

from src.hybrid_q.envs import make_env


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
)


def test_asoc_configs_load_with_unique_environment_and_agent_names():
    for name in CONFIGS:
        config = json.loads(
            (ROOT / "configs" / name).read_text(encoding="utf-8")
        )
        assert config["experiment_name"]
        assert config["output_dir"].startswith("results/")
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
        config = json.loads(
            (ROOT / "configs" / name).read_text(encoding="utf-8")
        )
        for spec in config["envs"]:
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
