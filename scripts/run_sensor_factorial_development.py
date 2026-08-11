from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_CONFIG = (
    ROOT
    / "configs/diagnostic_extensions/sensor_factorial_development/matched_conditions.yaml"
)
SELECTED_ESTIMATORS = (
    ROOT / "results/diagnostic_extensions/support_development/selected_estimators.yaml"
)
SELECTED_ESTIMATORS_DIGEST = SELECTED_ESTIMATORS.with_suffix(".sha256")

FACTOR_FIELDS = (
    "sensor_noise_enabled",
    "sensor_latency_enabled",
    "localization_dropout_enabled",
    "range_dropout_enabled",
    "camera_dropout_enabled",
    "visibility_occlusion_enabled",
)
CONDITION_FACTORS = {
    "no_noise_no_delay_no_dropout": set(),
    "noise_only": {"sensor_noise_enabled"},
    "latency_only": {"sensor_latency_enabled"},
    "localization_dropout_only": {"localization_dropout_enabled"},
    "range_dropout_only": {"range_dropout_enabled"},
    "camera_dropout_only": {"camera_dropout_enabled"},
    "visibility_occlusion_only": {"visibility_occlusion_enabled"},
    "combined_executed_condition": set(FACTOR_FIELDS),
}
EXPECTED_AGENTS = {
    "dqn",
    "fuzzy_relative_reliability",
    "selected_approximate_support",
    "sensorized_controller",
}


class SensorFactorialError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_selected_estimator() -> dict[str, Any]:
    expected = SELECTED_ESTIMATORS_DIGEST.read_text(encoding="utf-8").split()[0]
    observed = _sha256(SELECTED_ESTIMATORS)
    if observed != expected:
        raise SensorFactorialError("selected estimator SHA-256 mismatch")
    selected = yaml.safe_load(SELECTED_ESTIMATORS.read_text(encoding="utf-8"))
    definition = selected["overall_selected_estimator"]["candidate_definition"]
    return definition


def validate_factorial_config(config: dict[str, Any]) -> pd.DataFrame:
    if config.get("analysis", {}).get("analysis_status") != (
        "development_diagnostic_not_final_confirmation"
    ):
        raise SensorFactorialError("factorial config is not development diagnostic")
    seeds = [int(seed) for seed in config.get("seeds", [])]
    if seeds != [15002, 15003]:
        raise SensorFactorialError("factorial seeds must be [15002, 15003]")
    if any(16000 <= seed <= 16099 for seed in seeds):
        raise SensorFactorialError("reserved sensorized final seed detected")
    if set(agent["name"] for agent in config.get("agents", [])) != EXPECTED_AGENTS:
        raise SensorFactorialError("unexpected factorial agent set")

    selected = verify_selected_estimator()
    approximate = next(
        agent
        for agent in config["agents"]
        if agent["name"] == "selected_approximate_support"
    )["params"]
    selected_pairs = {
        "support_estimator_type": selected["estimator_type"],
        "support_representation_type": selected["representation"],
        "support_index_type": selected["requested_index"],
        "approximate_support_k": int(selected["k"]),
        "approximate_support_bandwidth": float(selected["h"]),
        "approximate_support_tau": float(selected["tau_approx"]),
        "support_covariance_regularization": float(selected["regularization"]),
    }
    for key, value in selected_pairs.items():
        if approximate.get(key) != value:
            raise SensorFactorialError(f"selected estimator mismatch: {key}")

    envs = config.get("envs", [])
    by_name = {env["name"]: env for env in envs}
    if set(by_name) != set(CONDITION_FACTORS) or len(by_name) != len(envs):
        raise SensorFactorialError("condition names are missing or duplicated")
    baseline = by_name["no_noise_no_delay_no_dropout"]
    fixed_env = {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "kwargs"}
    }
    fixed_kwargs = {
        key: value
        for key, value in baseline["kwargs"].items()
        if key not in FACTOR_FIELDS
    }
    rows = []
    for name, expected_active in CONDITION_FACTORS.items():
        env = by_name[name]
        observed_fixed_env = {
            key: value for key, value in env.items() if key not in {"name", "kwargs"}
        }
        observed_fixed_kwargs = {
            key: value
            for key, value in env["kwargs"].items()
            if key not in FACTOR_FIELDS
        }
        if observed_fixed_env != fixed_env or observed_fixed_kwargs != fixed_kwargs:
            raise SensorFactorialError(f"non-factor settings differ for {name}")
        active = {field for field in FACTOR_FIELDS if env["kwargs"].get(field) is True}
        if active != expected_active:
            raise SensorFactorialError(
                f"factor isolation mismatch for {name}: {sorted(active)}"
            )
        rows.append(
            {
                "condition": name,
                **{field: field in active for field in FACTOR_FIELDS},
                "seed_set": ";".join(str(seed) for seed in seeds),
                "training_steps": int(env["training_steps"]),
                "checkpoint_interval": int(config["evaluation"]["interval_steps"]),
                "evaluation_episodes": int(config["evaluation"]["episodes"]),
                "max_steps": int(env["max_steps"]),
                "audit_status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def audit_execution(raw_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    evaluation = raw[raw["phase"].eq("eval")]
    expected_runs = len(CONDITION_FACTORS) * len(EXPECTED_AGENTS) * 2
    runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    violations = []
    if len(runs) != expected_runs:
        violations.append(f"run count {len(runs)} != {expected_runs}")
    if set(raw["seed"].astype(int)) != {15002, 15003}:
        violations.append("seed set mismatch")
    if raw["seed"].astype(int).between(16000, 16099).any():
        violations.append("final sensorized seed present")
    expected_eval_rows = expected_runs * 2 * int(config["evaluation"]["episodes"])
    if len(evaluation) != expected_eval_rows:
        violations.append(
            f"evaluation rows {len(evaluation)} != {expected_eval_rows}"
        )
    if set(evaluation["checkpoint"].astype(int)) != {40, 80}:
        violations.append("checkpoint set mismatch")
    return {
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "development_diagnostic",
        "expected_agent_seed_condition_runs": expected_runs,
        "observed_agent_seed_condition_runs": len(runs),
        "evaluation_rows": len(evaluation),
        "development_seeds": [15002, 15003],
        "final_seed_rows": int(raw["seed"].astype(int).between(16000, 16099).sum()),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = validate_factorial_config(config)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "condition_manifest.csv", index=False)
    raw_path = run_config(config_path)
    aggregate_standard(raw_path, output_dir)
    execution = audit_execution(raw_path, config)
    existing = audit(config_path, output_dir)
    execution["standard_result_audit"] = existing["status"]
    execution["standard_provenance_audit"] = existing["provenance_status"]
    (output_dir / "execution_audit.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )
    if execution["status"] != "PASS" or existing["status"] != "PASS":
        raise SystemExit(json.dumps(execution, indent=2))
    print(
        "SENSOR_FACTORIAL_DEVELOPMENT_PASS "
        f"runs={execution['observed_agent_seed_condition_runs']} "
        f"eval_rows={execution['evaluation_rows']}"
    )


if __name__ == "__main__":
    main()
