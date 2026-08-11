from __future__ import annotations

import argparse
import gzip
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
from scripts.lock_sensor_final_protocol import (  # noqa: E402
    EXPECTED_AGENTS,
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    load_and_validate,
    sha256,
)


DEFAULT_CONFIG = (
    ROOT / "configs/diagnostic_extensions/sensorized_final/matched_final.yaml"
)
PROTOCOL_PATH = (
    ROOT / "configs/diagnostic_extensions/sensorized_final/protocol_lock.yaml"
)
DIGEST_PATH = PROTOCOL_PATH.with_suffix(".sha256")
FACTOR_FIELDS = {
    "sensor_noise_enabled",
    "sensor_latency_enabled",
    "localization_dropout_enabled",
    "range_dropout_enabled",
    "camera_dropout_enabled",
    "visibility_occlusion_enabled",
}


class SensorFinalExecutionError(ValueError):
    pass


def _hash_from_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").split()[0]


def assert_final_seeds_unused(output_dir: Path) -> None:
    if output_dir.exists() and any((output_dir / "runs").glob("*.csv")):
        return
    for metadata_path in (ROOT / "results/diagnostic_extensions").rglob(
        "metadata.json"
    ):
        if output_dir in metadata_path.parents:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        declared = {int(seed) for seed in config.get("seeds", [])}
        for env in config.get("envs", []):
            declared.update(int(seed) for seed in env.get("seeds", []))
        collision = sorted(declared.intersection(EXPECTED_SEEDS))
        if collision:
            raise SensorFinalExecutionError(
                f"reserved final seeds already declared by {metadata_path}: {collision}"
            )
    filenames = [
        path
        for path in (ROOT / "results/diagnostic_extensions").rglob("*")
        if path.is_file()
        and any(f"seed_{seed}" in path.name for seed in EXPECTED_SEEDS)
    ]
    if filenames:
        raise SensorFinalExecutionError(
            f"reserved final seed shards already exist: {filenames[:3]}"
        )


def validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> pd.DataFrame:
    protocol_digest = sha256(PROTOCOL_PATH)
    if _hash_from_file(DIGEST_PATH) != protocol_digest:
        raise SensorFinalExecutionError("sensor final protocol hash is invalid")
    analysis = config.get("analysis", {})
    if analysis.get("protocol_sha256") != protocol_digest:
        raise SensorFinalExecutionError("config protocol digest mismatch")
    if [int(seed) for seed in config.get("seeds", [])] != EXPECTED_SEEDS:
        raise SensorFinalExecutionError("config seed set mismatch")
    if config.get("trace_logging", {}).get("schema_version") != (
        protocol["trace_lock"]["schema_version"]
    ):
        raise SensorFinalExecutionError("trace schema mismatch")

    envs = {item["name"]: item for item in config.get("envs", [])}
    if set(envs) != EXPECTED_CONDITIONS or len(envs) != 2:
        raise SensorFinalExecutionError("condition registry mismatch")
    locked_conditions = {
        item["condition_id"]: item["factor_flags"]
        for item in protocol["final_conditions"]
    }
    base = envs["combined_executed_condition"]
    fixed_env = {key: value for key, value in base.items() if key not in {"name", "kwargs"}}
    fixed_kwargs = {key: value for key, value in base["kwargs"].items() if key not in FACTOR_FIELDS}
    for name, env in envs.items():
        if {key: value for key, value in env.items() if key not in {"name", "kwargs"}} != fixed_env:
            raise SensorFinalExecutionError(f"unmatched environment budget: {name}")
        if {key: value for key, value in env["kwargs"].items() if key not in FACTOR_FIELDS} != fixed_kwargs:
            raise SensorFinalExecutionError(f"unmatched plant configuration: {name}")
        observed = {field: bool(env["kwargs"].get(field, False)) for field in FACTOR_FIELDS}
        if observed != locked_conditions[name]:
            raise SensorFinalExecutionError(f"factor isolation mismatch: {name}")

    agents = {item["name"]: item for item in config.get("agents", [])}
    if set(agents) != EXPECTED_AGENTS or len(agents) != 5:
        raise SensorFinalExecutionError("agent registry mismatch")
    selected_temporal = yaml.safe_load(
        (ROOT / protocol["prerequisites"]["selected_temporal_model"]["source_file"]).read_text(encoding="utf-8")
    )
    expected_temporal = selected_temporal["agent_params"]
    for key, value in expected_temporal.items():
        if agents["selected_temporal_drqn"]["params"].get(key) != value:
            raise SensorFinalExecutionError(f"frozen temporal parameter mismatch: {key}")
    selected_support = yaml.safe_load(
        (ROOT / protocol["prerequisites"]["selected_support_estimator"]["source_file"]).read_text(encoding="utf-8")
    )["overall_selected_estimator"]["candidate_definition"]
    support_params = agents["selected_approximate_support"]["params"]
    support_mapping = {
        "support_estimator_type": selected_support["estimator_type"],
        "support_representation_type": selected_support["representation"],
        "support_index_type": selected_support["requested_index"],
        "approximate_support_k": selected_support["k"],
        "approximate_support_bandwidth": selected_support["h"],
        "approximate_support_tau": selected_support["tau_approx"],
        "support_covariance_regularization": selected_support["regularization"],
    }
    for key, value in support_mapping.items():
        if support_params.get(key) != value:
            raise SensorFinalExecutionError(f"frozen support parameter mismatch: {key}")

    locked_contrasts = [
        {
            "name": item["contrast_id"],
            "status": item["status"],
            "left": item["left"],
            "right": item["right"],
        }
        for item in protocol["analysis"]["planned_contrasts"]
    ]
    if analysis.get("planned_contrasts") != locked_contrasts:
        raise SensorFinalExecutionError("planned contrast mismatch")

    rows = []
    schedule = ";".join(map(str, protocol["budget"]["checkpoint_schedule"]))
    seed_set = ";".join(map(str, EXPECTED_SEEDS))
    for condition in sorted(envs):
        for agent in sorted(agents):
            rows.append(
                {
                    "condition": condition,
                    "agent": agent,
                    "seed_set": seed_set,
                    "training_steps": 240,
                    "checkpoint_schedule": schedule,
                    "evaluation_episodes": 4,
                    "interaction_budget_matched": True,
                    "compute_budget_identical": False,
                    "audit_status": "PASS",
                }
            )
    return pd.DataFrame(rows)


def audit_execution(raw_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    evaluation = raw[raw["phase"].eq("eval")]
    expected_runs = 2 * 5 * 30
    expected_eval_rows = expected_runs * 2 * 4
    trace_paths = sorted((raw_path.parent / "trace_shards").glob("*.csv.gz"))
    violations = []
    if len(runs) != expected_runs:
        violations.append(f"run count {len(runs)} != {expected_runs}")
    if set(raw["seed"].astype(int)) != set(EXPECTED_SEEDS):
        violations.append("final seed coverage mismatch")
    if set(evaluation["checkpoint"].astype(int)) != {120, 240}:
        violations.append("checkpoint coverage mismatch")
    if len(evaluation) != expected_eval_rows:
        violations.append(f"evaluation rows {len(evaluation)} != {expected_eval_rows}")
    if len(trace_paths) != expected_runs:
        violations.append(f"trace shard count {len(trace_paths)} != {expected_runs}")
    for path in trace_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = handle.readline()
        if "post_action_failure_stage" not in header:
            violations.append(f"v2 trace fields missing: {path.name}")
            break
    return {
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "replication",
        "expected_agent_seed_condition_runs": expected_runs,
        "observed_agent_seed_condition_runs": len(runs),
        "evaluation_rows": len(evaluation),
        "trace_shards": len(trace_paths),
        "seed_range": "16000-16029",
        "excluded_seeds": [],
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    protocol = load_and_validate(PROTOCOL_PATH)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = validate_config(config, protocol)
    output_dir = ROOT / config["output_dir"]
    assert_final_seeds_unused(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "design_manifest.csv", index=False)
    raw_path = run_config(config_path)
    aggregate_standard(raw_path, output_dir)
    execution = audit_execution(raw_path, config)
    standard = audit(config_path, output_dir)
    execution["standard_result_audit"] = standard["status"]
    execution["standard_provenance_audit"] = standard["provenance_status"]
    execution["protocol_sha256"] = sha256(PROTOCOL_PATH)
    (output_dir / "execution_audit.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )
    if execution["status"] != "PASS" or standard["status"] != "PASS":
        raise SystemExit(json.dumps(execution, indent=2))
    print(
        "SENSOR_FINAL_EXECUTION_PASS "
        f"runs={execution['observed_agent_seed_condition_runs']} "
        f"eval_rows={execution['evaluation_rows']}"
    )


if __name__ == "__main__":
    main()
