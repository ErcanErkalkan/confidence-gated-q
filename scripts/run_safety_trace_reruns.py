from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for path in (ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_q.experiment import run_config  # noqa: E402
from hybrid_q.statistics import aggregate as aggregate_standard  # noqa: E402
from scripts.audit_results import audit  # noqa: E402
from scripts.lock_safety_trace_protocol import (  # noqa: E402
    EXPECTED_AGENTS,
    EXPECTED_FAMILIES,
    EXPECTED_SEEDS,
    DEFAULT_DIGEST,
    DEFAULT_PROTOCOL,
    load_and_validate,
    sha256,
)


DEFAULT_CONFIG = ROOT / "configs/diagnostic_extensions/safety_traces/matched_rerun.yaml"


class SafetyTraceExecutionError(ValueError):
    pass


def assert_seeds_unused(output_dir: Path) -> None:
    if output_dir.exists() and any((output_dir / "runs").glob("*.csv")):
        return
    for metadata_path in (ROOT / "results/diagnostic_extensions").rglob("metadata.json"):
        if output_dir in metadata_path.parents:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        declared = {int(seed) for seed in config.get("seeds", [])}
        for env in config.get("envs", []):
            declared.update(int(seed) for seed in env.get("seeds", []))
        collision = sorted(declared.intersection(EXPECTED_SEEDS))
        if collision:
            raise SafetyTraceExecutionError(
                f"new safety seeds already declared by {metadata_path}: {collision}"
            )
    collisions = [
        path
        for path in (ROOT / "results/diagnostic_extensions").rglob("*")
        if path.is_file()
        and any(f"seed_{seed}" in path.name for seed in EXPECTED_SEEDS)
    ]
    if collisions:
        raise SafetyTraceExecutionError(
            f"new safety seed shards already exist: {collisions[:3]}"
        )


def validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> pd.DataFrame:
    digest = sha256(DEFAULT_PROTOCOL)
    if DEFAULT_DIGEST.read_text(encoding="utf-8").split()[0] != digest:
        raise SafetyTraceExecutionError("safety protocol digest mismatch")
    if config.get("analysis", {}).get("protocol_sha256") != digest:
        raise SafetyTraceExecutionError("config does not reference locked digest")
    if config.get("seeds") != EXPECTED_SEEDS:
        raise SafetyTraceExecutionError("config seed mismatch")
    if config.get("trace_logging", {}).get("schema_version") != "sensorized_sil_trace_v3":
        raise SafetyTraceExecutionError("safety trace schema mismatch")
    envs = {env["name"]: env for env in config.get("envs", [])}
    if set(envs) != EXPECTED_FAMILIES:
        raise SafetyTraceExecutionError("rerun family mismatch")
    locked = {item["family_id"]: item for item in protocol["rerun_families"]}
    factor_fields = set(next(iter(locked.values()))["factor_flags"])
    base = envs["combined_executed_condition_safety_trace"]
    fixed_kwargs = {
        key: value for key, value in base["kwargs"].items() if key not in factor_fields
    }
    rows = []
    for family, env in envs.items():
        observed = {
            key: bool(env["kwargs"].get(key, False)) for key in factor_fields
        }
        if observed != locked[family]["factor_flags"]:
            raise SafetyTraceExecutionError(f"factor mismatch: {family}")
        comparable = {
            key: value for key, value in env["kwargs"].items() if key not in factor_fields
        }
        if comparable != fixed_kwargs:
            raise SafetyTraceExecutionError(f"unmatched plant: {family}")
        if env["kwargs"].get("perturbation_onset_step") != 6:
            raise SafetyTraceExecutionError("perturbation onset mismatch")
    agents = {agent["name"] for agent in config.get("agents", [])}
    if agents != EXPECTED_AGENTS:
        raise SafetyTraceExecutionError("agent set mismatch")
    for family in sorted(envs):
        for agent in sorted(agents):
            rows.append(
                {
                    "family": family,
                    "agent": agent,
                    "seed_set": ";".join(map(str, EXPECTED_SEEDS)),
                    "training_steps": 240,
                    "checkpoint_schedule": "120;240",
                    "evaluation_episodes": 4,
                    "episode_horizon": 30,
                    "interaction_budget_matched": True,
                    "compute_budget_identical": False,
                    "audit_status": "PASS",
                }
            )
    return pd.DataFrame(rows)


def audit_execution(raw_path: Path) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    evaluation = raw[raw["phase"].eq("eval")]
    expected_runs = len(EXPECTED_FAMILIES) * len(EXPECTED_AGENTS) * len(EXPECTED_SEEDS)
    expected_eval_rows = expected_runs * 2 * 4
    traces = sorted((raw_path.parent / "trace_shards").glob("*.csv.gz"))
    violations = []
    if len(runs) != expected_runs:
        violations.append(f"run count {len(runs)} != {expected_runs}")
    if set(raw["seed"].astype(int)) != set(EXPECTED_SEEDS):
        violations.append("seed coverage mismatch")
    if set(evaluation["checkpoint"].astype(int)) != {120, 240}:
        violations.append("checkpoint coverage mismatch")
    if len(evaluation) != expected_eval_rows:
        violations.append(f"evaluation rows {len(evaluation)} != {expected_eval_rows}")
    if len(traces) != expected_runs:
        violations.append(f"trace shard count {len(traces)} != {expected_runs}")
    for path in traces:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = handle.readline()
        if "post_action_trajectory_deviation" not in header:
            violations.append(f"v3 fields missing: {path.name}")
            break
    return {
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "descriptive_safety_rerun",
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "evaluation_rows": len(evaluation),
        "trace_shards": len(traces),
        "trace_schema_version": "sensorized_sil_trace_v3",
        "seed_range": "16030-16039",
        "excluded_seeds": [],
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    protocol = load_and_validate(DEFAULT_PROTOCOL)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    design = validate_config(config, protocol)
    output_dir = ROOT / config["output_dir"]
    assert_seeds_unused(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design.to_csv(output_dir / "design_manifest.csv", index=False)
    raw_path = run_config(config_path)
    aggregate_standard(raw_path, output_dir)
    execution = audit_execution(raw_path)
    standard = audit(config_path, output_dir)
    execution["standard_result_audit"] = standard["status"]
    execution["standard_provenance_audit"] = standard["provenance_status"]
    execution["protocol_sha256"] = sha256(DEFAULT_PROTOCOL)
    (output_dir / "execution_audit.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )
    if execution["status"] != "PASS" or standard["status"] != "PASS":
        raise SystemExit(json.dumps(execution, indent=2))
    print(
        "SAFETY_TRACE_EXECUTION_PASS "
        f"runs={execution['observed_runs']} eval_rows={execution['evaluation_rows']}"
    )


if __name__ == "__main__":
    main()
