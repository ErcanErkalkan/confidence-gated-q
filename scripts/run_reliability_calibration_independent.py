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
from scripts.aggregate_reliability_calibration_independent import (  # noqa: E402
    aggregate_independent,
)
from scripts.audit_results import audit  # noqa: E402


CONFIG_PATH = ROOT / "configs/diagnostic_extensions/reliability_calibration/independent_focal.yaml"
AMENDMENT_PATH = ROOT / "configs/diagnostic_extensions/reliability_calibration/CALIBRATION_MEASUREMENT_AMENDMENT_2026-08-05.md"
FAILED_INSTRUMENTATION_DIR = ROOT / "results/diagnostic_extensions/reliability_calibration_independent/execution"
LOCK_YAML = ROOT / "configs/diagnostic_extensions/reliability_calibration/final_lock.yaml"
LOCK_MD = LOCK_YAML.with_suffix(".md")
LOCK_DIGEST = LOCK_YAML.with_suffix(".sha256")
EXPECTED_SEEDS = list(range(12090, 12100))
EXPECTED_AGENTS = {"relative_reliability_fuzzy", "same_input_crisp"}
EXPECTED_ENVIRONMENTS = {
    "ReliabilityShift-boundary-030-independent-calibration",
    "ReliabilityShift-boundary-020-independent-calibration",
}


class IndependentCalibrationError(ValueError):
    pass


def composite_sha256() -> str:
    digest = hashlib.sha256()
    for path in (LOCK_YAML, LOCK_MD):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_lock_and_config(config: dict[str, Any]) -> str:
    observed = composite_sha256()
    expected = LOCK_DIGEST.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise IndependentCalibrationError("protocol lock SHA-256 mismatch")
    protocol = yaml.safe_load(LOCK_YAML.read_text(encoding="utf-8"))
    if protocol.get("status") != "immutable_before_execution":
        raise IndependentCalibrationError("protocol is not immutable")
    if protocol["seeds"]["values"] != EXPECTED_SEEDS:
        raise IndependentCalibrationError("protocol seed set mismatch")
    if [int(seed) for seed in config.get("seeds", [])] != EXPECTED_SEEDS:
        raise IndependentCalibrationError("config seed set mismatch")
    analysis = config.get("analysis", {})
    if analysis.get("protocol_sha256") != observed:
        raise IndependentCalibrationError("config protocol digest mismatch")
    if analysis.get("evidence_class") != "replication_calibration":
        raise IndependentCalibrationError("invalid evidence class")
    if analysis.get("seed_registry_key") != "new_shift_final":
        raise IndependentCalibrationError("invalid seed registry key")
    agents = {item["name"] for item in config.get("agents", [])}
    envs = {item["name"] for item in config.get("envs", [])}
    if agents != EXPECTED_AGENTS:
        raise IndependentCalibrationError("focal agent registry mismatch")
    if envs != EXPECTED_ENVIRONMENTS:
        raise IndependentCalibrationError("analytic severity registry mismatch")
    amendment = analysis.get("instrumentation_amendment")
    if amendment is not None:
        if (ROOT / str(amendment)).resolve() != AMENDMENT_PATH.resolve():
            raise IndependentCalibrationError("unknown instrumentation amendment")
        if not AMENDMENT_PATH.exists():
            raise IndependentCalibrationError("instrumentation amendment is missing")
    selected_path = ROOT / protocol["prerequisites"]["frozen_mapping_selection"]
    selected_digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    if selected_digest != protocol["prerequisites"]["frozen_mapping_selection_sha256"]:
        raise IndependentCalibrationError("frozen mapping selection hash mismatch")
    return observed


def compare_initial_outcomes(rerun_raw_path: Path) -> dict[str, Any]:
    initial_raw_path = FAILED_INSTRUMENTATION_DIR / "raw.csv"
    columns = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        "return",
        "length",
        "success",
        "environment_steps",
        "gradient_updates",
        "training_loss_mean",
        "training_loss_max",
        "nonfinite_loss_count",
        "mean_gate",
        "global_tabular_error",
        "global_neural_error",
        "visited_states",
        "failure_rate",
        "memory_branch_usage_ratio",
        "neural_branch_usage_ratio",
        "abstention_ratio",
        "adaptive_alpha_mean",
        "support_score_mean",
        "uncertainty_score_mean",
        "selected_branch",
        "mixed_selected_action",
        "optimal_action",
        "post_shift",
        "shift_region",
    ]
    initial = pd.read_csv(initial_raw_path, usecols=columns)
    rerun = pd.read_csv(rerun_raw_path, usecols=columns)
    equal = initial.equals(rerun)
    mismatched_columns = [
        column
        for column in columns
        if len(initial) != len(rerun)
        or not initial[column].equals(rerun[column])
    ]
    return {
        "initial_instrumentation_attempt": str(
            initial_raw_path.relative_to(ROOT)
        ),
        "non_diagnostic_outcomes_equal": bool(equal),
        "compared_rows": int(min(len(initial), len(rerun))),
        "mismatched_columns": mismatched_columns,
    }


def assert_seeds_unused(output_dir: Path) -> None:
    if output_dir.exists() and any((output_dir / "runs").glob("*.csv")):
        return
    expected = set(EXPECTED_SEEDS)
    for metadata_path in (ROOT / "results").rglob("metadata.json"):
        if output_dir in metadata_path.parents:
            continue
        if metadata_path.parent.resolve() == FAILED_INSTRUMENTATION_DIR.resolve():
            if not AMENDMENT_PATH.exists():
                raise IndependentCalibrationError(
                    "instrumentation rerun requires its pre-run amendment"
                )
            failed_raw = FAILED_INSTRUMENTATION_DIR / "raw.csv"
            if not failed_raw.exists():
                raise IndependentCalibrationError(
                    "declared failed instrumentation output is incomplete"
                )
            failed_eval = pd.read_csv(
                failed_raw,
                usecols=["phase", "relative_reliability_score"],
            )
            failed_eval = failed_eval.loc[failed_eval["phase"].eq("eval")]
            if len(failed_eval) != 52_480 or failed_eval[
                "relative_reliability_score"
            ].notna().any():
                raise IndependentCalibrationError(
                    "prior output does not match the amended failure signature"
                )
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        declared = {int(seed) for seed in config.get("seeds", [])}
        for environment in config.get("envs", []):
            declared.update(int(seed) for seed in environment.get("seeds", []))
        collision = sorted(expected.intersection(declared))
        if collision:
            raise IndependentCalibrationError(
                f"independent calibration seed already used by {metadata_path}: {collision}"
            )


def audit_execution(raw_path: Path, digest: str) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    evaluation = raw.loc[raw["phase"].eq("eval")]
    runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    expected_runs = 2 * 2 * len(EXPECTED_SEEDS)
    expected_checkpoints = set(range(250, 4001, 250))
    expected_eval_rows = expected_runs * len(expected_checkpoints) * 82
    violations: list[str] = []
    if len(runs) != expected_runs:
        violations.append(f"run count {len(runs)} != {expected_runs}")
    if set(raw["seed"].astype(int)) != set(EXPECTED_SEEDS):
        violations.append("seed coverage mismatch")
    if set(evaluation["checkpoint"].astype(int)) != expected_checkpoints:
        violations.append("checkpoint coverage mismatch")
    if len(evaluation) != expected_eval_rows:
        violations.append(
            f"evaluation rows {len(evaluation)} != {expected_eval_rows}"
        )
    diagnostic = [
        "relative_reliability_score",
        "tabular_action_correct",
        "neural_action_correct",
        "tabular_q_error",
        "neural_q_error",
    ]
    if evaluation[diagnostic].isna().any().any():
        violations.append("analytic evaluation diagnostics contain missing values")
    training = raw.loc[raw["phase"].eq("train")]
    if not training[diagnostic].isna().all().all():
        violations.append("evaluation-only diagnostics leaked into training rows")
    return {
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "replication_calibration",
        "protocol_sha256": digest,
        "expected_agent_seed_severity_runs": expected_runs,
        "observed_agent_seed_severity_runs": len(runs),
        "evaluation_rows": len(evaluation),
        "seed_range": "12090-12099",
        "excluded_seeds": [],
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    digest = validate_lock_and_config(config)
    execution_dir = ROOT / config["output_dir"]
    assert_seeds_unused(execution_dir)
    raw_path = run_config(config_path)
    aggregate_standard(raw_path, execution_dir)
    standard = audit(config_path, execution_dir)
    report = audit_execution(raw_path, digest)
    if config.get("analysis", {}).get("instrumentation_amendment"):
        equivalence = compare_initial_outcomes(raw_path)
        report["instrumentation_rerun_equivalence"] = equivalence
        if not equivalence["non_diagnostic_outcomes_equal"]:
            report["status"] = "FAIL"
            report["violations"].append(
                "instrumentation rerun changed non-diagnostic outcomes"
            )
    report["standard_result_audit"] = standard["status"]
    report["standard_provenance_audit"] = standard["provenance_status"]
    result_dir = ROOT / "results/diagnostic_extensions/reliability_calibration_independent"
    counts = aggregate_independent(
        execution_dir,
        config_path,
        result_dir,
        ROOT / "tables/table_reliability_calibration_independent.csv",
        ROOT / "figures/fig_reliability_calibration_independent.pdf",
    )
    report.update(counts)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "execution_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "PASS" or standard["status"] != "PASS":
        raise SystemExit(json.dumps(report, indent=2))
    print(
        "RELIABILITY_CALIBRATION_INDEPENDENT_EXECUTION_PASS "
        f"runs={report['observed_agent_seed_severity_runs']} "
        f"evaluation_rows={report['evaluation_rows']}"
    )


if __name__ == "__main__":
    main()
