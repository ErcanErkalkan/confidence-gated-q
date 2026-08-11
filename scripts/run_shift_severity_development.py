from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.envs import make_env  # noqa: E402
from hybrid_q.experiment import run_config  # noqa: E402
from scripts.run_fuzzy_crisp_development import (  # noqa: E402
    deterministic_gzip_copy,
)


PROTOCOL = ROOT / "configs/diagnostic_extensions/INDEPENDENT_SHIFT_PROTOCOL.yaml"
DEFAULT_CONFIG = ROOT / "configs/diagnostic_extensions/final_shifts/development_severity_selection.yaml"
FINAL_SEED_MIN = 12000
FINAL_SEED_MAX = 12099
EXPECTED_AGENTS = {
    "relative_reliability_fuzzy",
    "count_gated_tau_20",
    "same_input_crisp",
}
ENVIRONMENT_IDS = {
    "transition_dynamics_shift": "TransitionDynamicsShift-v0",
    "observation_shift": "ObservationShift-v0",
    "localized_multistep_reward_or_policy_shift": ("LocalizedRewardShift-v0"),
}


class SeverityDevelopmentError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SeverityDevelopmentError(f"YAML root must be a mapping: {path}")
    return value


def _constructor_severity(
    mechanism_id: str,
    locked_mechanism: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    if mechanism_id == "transition_dynamics_shift":
        return {
            "pre_shift_slip_probability": float(
                locked_mechanism["environment_definition"]["pre_shift_slip_probability"]
            ),
            "post_shift_slip_probability": float(
                candidate["post_shift_slip_probability"]
            ),
        }
    if mechanism_id == "observation_shift":
        return {
            "slip_probability": float(
                locked_mechanism["environment_definition"][
                    "transition_slip_probability"
                ]
            ),
            "pre_shift_sensor_gain": float(
                locked_mechanism["environment_definition"]["pre_shift_sensor_gain"]
            ),
            "post_shift_sensor_gain": float(candidate["post_shift_sensor_gain"]),
        }
    if mechanism_id == "localized_multistep_reward_or_policy_shift":
        return {
            "slip_probability": float(
                locked_mechanism["environment_definition"][
                    "transition_slip_probability"
                ]
            ),
            "pre_shift_risk_penalty": float(
                locked_mechanism["environment_definition"]["pre_shift_risk_penalty"]
            ),
            "post_shift_risk_penalty": float(candidate["post_shift_risk_penalty"]),
        }
    raise SeverityDevelopmentError(f"unknown mechanism: {mechanism_id}")


def validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    analysis = config.get("analysis", {})
    if analysis.get("evidence_class") != "development":
        raise SeverityDevelopmentError("severity screen must be development-only")
    if analysis.get("p_values_prohibited") is not True:
        raise SeverityDevelopmentError("severity selection must prohibit p-values")
    if config.get("output_dir") != (
        "results/diagnostic_extensions/final_shifts/severity_development/execution"
    ):
        raise SeverityDevelopmentError("unexpected severity output directory")
    if int(config["evaluation"]["interval_steps"]) != 1000:
        raise SeverityDevelopmentError("checkpoint interval differs from lock")
    if int(config["evaluation"]["episodes"]) != 200:
        raise SeverityDevelopmentError("evaluation episode count differs from lock")
    if set(agent.get("name") for agent in config.get("agents", [])) != EXPECTED_AGENTS:
        raise SeverityDevelopmentError("severity agent set differs from lock")
    agents = {agent["name"]: agent for agent in config["agents"]}
    expected_kinds = {
        "relative_reliability_fuzzy": "fuzzy_reliability_gate",
        "count_gated_tau_20": "count_gated",
        "same_input_crisp": "fuzzy_reliability_gate",
    }
    locked_common = protocol["agent_lock"]["common_parameters"]
    for name, agent in agents.items():
        if agent.get("kind") != expected_kinds[name]:
            raise SeverityDevelopmentError(f"agent kind differs from lock: {name}")
        params = agent.get("params", {})
        for key, expected in locked_common.items():
            if params.get(key) != expected:
                raise SeverityDevelopmentError(
                    f"{name} common parameter {key} differs from lock"
                )
    fuzzy_expected = protocol["agent_lock"]["relative_reliability_fuzzy_parameters"][
        "fuzzy_reliability_consequents"
    ]
    if (
        agents["relative_reliability_fuzzy"]["params"].get(
            "fuzzy_reliability_consequents"
        )
        != fuzzy_expected
    ):
        raise SeverityDevelopmentError("fuzzy consequents differ from lock")
    crisp = agents["same_input_crisp"]["params"]
    if (
        crisp.get("fuzzy_risk_ablation_mode") != "crisp_threshold"
        or float(crisp.get("fuzzy_crisp_support_threshold", -1)) != 0.5
        or float(crisp.get("fuzzy_crisp_reliability_threshold", -1)) != 0.5
    ):
        raise SeverityDevelopmentError("crisp mapping differs from lock")

    locked_by_id = {item["mechanism_id"]: item for item in protocol["mechanisms"]}
    observed: dict[str, list[dict[str, Any]]] = {}
    for env in config.get("envs", []):
        mechanism_id = env.get("mechanism_id")
        if mechanism_id not in locked_by_id:
            raise SeverityDevelopmentError(f"unknown mechanism: {mechanism_id}")
        observed.setdefault(mechanism_id, []).append(env)
        locked = locked_by_id[mechanism_id]
        if env.get("id") != ENVIRONMENT_IDS[mechanism_id]:
            raise SeverityDevelopmentError("environment ID differs from lock")
        development = locked["development_seeds"]
        expected_seeds = list(
            range(int(development["start"]), int(development["end"]) + 1)
        )
        seeds = [int(seed) for seed in env.get("seeds", [])]
        if seeds != expected_seeds:
            raise SeverityDevelopmentError(
                f"development seed mismatch for {mechanism_id}"
            )
        if any(FINAL_SEED_MIN <= seed <= FINAL_SEED_MAX for seed in seeds):
            raise SeverityDevelopmentError("reserved final seed detected")
        budget = locked["training_interaction_budget"]
        if int(env.get("training_steps", -1)) != int(budget["per_agent_seed"]):
            raise SeverityDevelopmentError("training budget differs from lock")
        if int(env["kwargs"].get("shift_after", -1)) != int(
            locked["shift_onset"]["training_interaction"]
        ):
            raise SeverityDevelopmentError("shift onset differs from lock")

        candidates = {
            item["severity_id"]: item
            for item in locked["development_severity_candidates"]
        }
        severity_id = env.get("severity_id")
        if severity_id not in candidates:
            raise SeverityDevelopmentError(
                f"unknown severity {severity_id} for {mechanism_id}"
            )
        expected_kwargs = _constructor_severity(
            mechanism_id, locked, candidates[severity_id]
        )
        for key, expected in expected_kwargs.items():
            if float(env["kwargs"].get(key, np.nan)) != expected:
                raise SeverityDevelopmentError(
                    f"{mechanism_id}/{severity_id} field {key} differs from lock"
                )
        instance = make_env(env)
        instance.close()

    if set(observed) != set(locked_by_id):
        raise SeverityDevelopmentError("severity mechanism coverage mismatch")
    for mechanism_id, envs in observed.items():
        locked_order = [
            item["severity_id"]
            for item in locked_by_id[mechanism_id]["development_severity_candidates"]
        ]
        if [env["severity_id"] for env in envs] != locked_order:
            raise SeverityDevelopmentError(
                f"severity order differs from lock for {mechanism_id}"
            )


def audit_execution(raw_path: Path, config: dict[str, Any]) -> None:
    columns = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        "success",
        "environment_steps",
        "completed_checkpoint_count",
        "expected_checkpoint_count",
    ]
    raw = pd.read_csv(raw_path, usecols=columns)
    if raw["seed"].astype(int).between(FINAL_SEED_MIN, FINAL_SEED_MAX).any():
        raise SeverityDevelopmentError("reserved final result row detected")
    evaluation = raw[raw["phase"] == "eval"].copy()
    key = ["environment", "agent", "seed", "checkpoint", "episode"]
    if evaluation.duplicated(key).any():
        raise SeverityDevelopmentError("duplicate evaluation rows")
    expected_checkpoints = set(range(1000, 24001, 1000))
    expected_runs = 9 * 3 * 10
    run_key = ["environment", "agent", "seed"]
    if evaluation.groupby(run_key).ngroups != expected_runs:
        raise SeverityDevelopmentError("severity run coverage mismatch")
    for run, frame in evaluation.groupby(run_key, sort=False):
        if set(frame["checkpoint"].astype(int)) != expected_checkpoints:
            raise SeverityDevelopmentError(f"checkpoint mismatch for {run}")
        if not frame.groupby("checkpoint").size().eq(200).all():
            raise SeverityDevelopmentError(f"evaluation episode mismatch for {run}")
        success = pd.to_numeric(frame["success"], errors="coerce")
        if success.isna().any() or not np.isfinite(success).all():
            raise SeverityDevelopmentError(f"invalid success rows for {run}")
    if len(evaluation) != expected_runs * 24 * 200:
        raise SeverityDevelopmentError("evaluation row count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the immutable-lock development severity screen."
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol = _load_yaml(PROTOCOL)
    config = _load_yaml(DEFAULT_CONFIG)
    validate_config(config, protocol)
    if args.validate_only:
        print("SHIFT_SEVERITY_DEVELOPMENT_LOCK_PASS final_seed_rows=0")
        return
    raw_path = run_config(DEFAULT_CONFIG)
    audit_execution(raw_path, config)
    compressed = deterministic_gzip_copy(raw_path)
    print(
        "SHIFT_SEVERITY_DEVELOPMENT_PASS "
        f"runs={9 * 3 * 10} final_seed_rows=0 raw={compressed}"
    )


if __name__ == "__main__":
    main()
