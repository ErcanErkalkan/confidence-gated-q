from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.experiment import run_config  # noqa: E402
from hybrid_q.statistics import aggregate as aggregate_standard  # noqa: E402
from scripts.audit_results import audit  # noqa: E402


DEFAULT_CONFIG = ROOT / (
    "configs/diagnostic_extensions/temporal_interface_development/"
    "matched_design.yaml"
)
DEVELOPMENT_SEEDS = [15004, 15005]
FINAL_SEED_RANGE = range(16000, 16100)
TEMPORAL_AGENTS = {
    "temporal_frame_stack_4",
    "temporal_drqn_seq4",
    "temporal_filtered_alpha035",
}
CAPACITY_CONTROLS = {
    "feedforward_capacity_h064",
    "feedforward_capacity_h096",
    "feedforward_capacity_h128",
}
INTERFACE_CONDITIONS = {
    "interface_A_state_low_level": ("state_accessible", "low_level"),
    "interface_B_sensorized_high_level": ("sensorized", "high_level"),
    "interface_C_state_high_level": ("state_accessible", "high_level"),
    "interface_D_sensorized_low_level": ("sensorized", "low_level"),
}
EXPECTED_ASSIGNMENTS = {
    "temporal_frame_stack_4": {"architecture_frame_stack"},
    "temporal_drqn_seq4": {"architecture_sensorized_low_level"},
    "temporal_filtered_alpha035": {"architecture_filtered_belief"},
    "feedforward_capacity_h064": {"architecture_sensorized_low_level"},
    "feedforward_capacity_h096": {"architecture_sensorized_low_level"},
    "feedforward_capacity_h128": {"architecture_sensorized_low_level"},
    "interface_dqn_h064": set(INTERFACE_CONDITIONS),
}


class TemporalInterfaceConfigError(ValueError):
    pass


def _without(mapping: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in excluded}


def validate_config(config: dict[str, Any]) -> pd.DataFrame:
    analysis = config.get("analysis", {})
    if analysis.get("analysis_status") != (
        "development_diagnostic_not_final_confirmation"
    ):
        raise TemporalInterfaceConfigError("config is not development diagnostic")
    seeds = [int(seed) for seed in config.get("seeds", [])]
    if seeds != DEVELOPMENT_SEEDS:
        raise TemporalInterfaceConfigError(
            f"development seeds must be {DEVELOPMENT_SEEDS}"
        )
    if any(seed in FINAL_SEED_RANGE for seed in seeds):
        raise TemporalInterfaceConfigError("reserved final sensor seed detected")
    if analysis.get("planned_contrasts") != []:
        raise TemporalInterfaceConfigError("development config must be descriptive")
    if analysis.get("temporal_candidate_count") != len(TEMPORAL_AGENTS):
        raise TemporalInterfaceConfigError("temporal candidate count mismatch")
    if analysis.get("feed_forward_capacity_control_count") != len(
        CAPACITY_CONTROLS
    ):
        raise TemporalInterfaceConfigError("capacity control count mismatch")

    agents = {agent["name"]: agent for agent in config.get("agents", [])}
    if set(agents) != set(EXPECTED_ASSIGNMENTS):
        raise TemporalInterfaceConfigError("unexpected agent registry")
    for name, expected in EXPECTED_ASSIGNMENTS.items():
        observed = set(agents[name].get("applies_to_envs", []))
        if observed != expected:
            raise TemporalInterfaceConfigError(
                f"environment assignment mismatch for {name}"
            )

    envs = {env["name"]: env for env in config.get("envs", [])}
    expected_envs = {
        "architecture_sensorized_low_level",
        "architecture_frame_stack",
        "architecture_filtered_belief",
        *INTERFACE_CONDITIONS,
    }
    if set(envs) != expected_envs or len(envs) != len(config["envs"]):
        raise TemporalInterfaceConfigError("environment names are incomplete")

    base = envs["architecture_sensorized_low_level"]
    fixed_spec = _without(
        base, {"name", "frame_stack", "temporal_filter_alpha"}
    )
    frame_spec = _without(
        envs["architecture_frame_stack"],
        {"name", "frame_stack", "temporal_filter_alpha"},
    )
    filtered_spec = _without(
        envs["architecture_filtered_belief"],
        {"name", "frame_stack", "temporal_filter_alpha"},
    )
    if frame_spec != fixed_spec or filtered_spec != fixed_spec:
        raise TemporalInterfaceConfigError("architecture budgets or plant differ")
    if envs["architecture_frame_stack"].get("frame_stack") != 4:
        raise TemporalInterfaceConfigError("frame-stack candidate is not stack-4")
    if envs["architecture_filtered_belief"].get(
        "temporal_filter_alpha"
    ) != 0.35:
        raise TemporalInterfaceConfigError("belief-filter alpha mismatch")

    baseline_interface = envs["interface_D_sensorized_low_level"]
    fixed_interface_spec = _without(baseline_interface, {"name", "kwargs"})
    fixed_interface_kwargs = _without(
        baseline_interface["kwargs"],
        {"observation_mode", "control_interface_mode"},
    )
    for name, (observation_mode, control_mode) in INTERFACE_CONDITIONS.items():
        env = envs[name]
        if _without(env, {"name", "kwargs"}) != fixed_interface_spec:
            raise TemporalInterfaceConfigError(
                f"non-interface environment field differs for {name}"
            )
        if _without(
            env["kwargs"],
            {"observation_mode", "control_interface_mode"},
        ) != fixed_interface_kwargs:
            raise TemporalInterfaceConfigError(
                f"non-interface plant field differs for {name}"
            )
        if env["kwargs"].get("observation_mode") != observation_mode:
            raise TemporalInterfaceConfigError(
                f"observation mode mismatch for {name}"
            )
        if env["kwargs"].get("control_interface_mode") != control_mode:
            raise TemporalInterfaceConfigError(
                f"control interface mismatch for {name}"
            )

    rows = []
    for agent_name, environments in EXPECTED_ASSIGNMENTS.items():
        family = (
            "temporal_candidate"
            if agent_name in TEMPORAL_AGENTS
            else (
                "feedforward_capacity_control"
                if agent_name in CAPACITY_CONTROLS
                else "interface_ablation"
            )
        )
        for environment in sorted(environments):
            env = envs[environment]
            rows.append(
                {
                    "family": family,
                    "candidate_or_agent": agent_name,
                    "environment": environment,
                    "observation_mode": env["kwargs"]["observation_mode"],
                    "control_interface_mode": env["kwargs"][
                        "control_interface_mode"
                    ],
                    "frame_stack": int(env.get("frame_stack", 1)),
                    "temporal_filter_alpha": env.get(
                        "temporal_filter_alpha", ""
                    ),
                    "seed_set": ";".join(map(str, seeds)),
                    "training_steps": int(env["training_steps"]),
                    "checkpoint_schedule": "120;240",
                    "evaluation_episodes": int(
                        config["evaluation"]["episodes"]
                    ),
                    "audit_status": "PASS",
                }
            )
    return pd.DataFrame(rows)


def audit_execution(raw_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    evaluation = raw[raw["phase"].eq("eval")]
    runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    expected_runs = 10 * len(DEVELOPMENT_SEEDS)
    expected_eval_rows = expected_runs * 2 * int(
        config["evaluation"]["episodes"]
    )
    violations = []
    if len(runs) != expected_runs:
        violations.append(f"run count {len(runs)} != {expected_runs}")
    if len(evaluation) != expected_eval_rows:
        violations.append(
            f"evaluation rows {len(evaluation)} != {expected_eval_rows}"
        )
    if set(raw["seed"].astype(int)) != set(DEVELOPMENT_SEEDS):
        violations.append("development seed coverage mismatch")
    if raw["seed"].astype(int).isin(FINAL_SEED_RANGE).any():
        violations.append("reserved final seed row detected")
    if set(evaluation["checkpoint"].astype(int)) != {120, 240}:
        violations.append("checkpoint coverage mismatch")
    return {
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "development_diagnostic",
        "expected_agent_seed_environment_runs": expected_runs,
        "observed_agent_seed_environment_runs": len(runs),
        "evaluation_rows": len(evaluation),
        "development_seeds": DEVELOPMENT_SEEDS,
        "final_seed_rows": int(
            raw["seed"].astype(int).isin(FINAL_SEED_RANGE).sum()
        ),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = validate_config(config)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "design_manifest.csv", index=False)
    raw_path = run_config(config_path)
    aggregate_standard(raw_path, output_dir)
    execution = audit_execution(raw_path, config)
    standard = audit(config_path, output_dir)
    execution["standard_result_audit"] = standard["status"]
    execution["standard_provenance_audit"] = standard["provenance_status"]
    (output_dir / "execution_audit.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )
    if execution["status"] != "PASS" or standard["status"] != "PASS":
        raise SystemExit(json.dumps(execution, indent=2))
    evaluation_rows = len(pd.read_csv(raw_path).query("phase == 'eval'"))
    print(
        "TEMPORAL_INTERFACE_DEVELOPMENT_PASS "
        f"runs={len(manifest) * len(DEVELOPMENT_SEEDS)} "
        f"eval_rows={evaluation_rows}"
    )


if __name__ == "__main__":
    main()
