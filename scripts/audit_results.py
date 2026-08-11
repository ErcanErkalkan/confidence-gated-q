from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.config import load_config
from hybrid_q.provenance import (
    execution_input_manifest,
    execution_snapshot_sha256,
)
from hybrid_q.statistics import holm_adjust


REQUIRED_COLUMNS = {
    "experiment_name",
    "experiment",
    "config_file",
    "environment",
    "environment_id",
    "resolved_environment_id",
    "observation_representation",
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
    "elapsed_seconds",
    "training_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "git_commit_hash",
    "package_version",
    "python_version",
    "torch_version",
    "numpy_version",
    "gymnasium_version",
    "minigrid_version",
}

STABILITY_COLUMNS = {
    "training_loss_mean",
    "training_loss_max",
    "nonfinite_loss_count",
    "completed_checkpoint_count",
    "expected_checkpoint_count",
}

TAIL_RISK_SCHEMAS = {
    "seed_tail_risk.csv": {
        "result_family",
        "environment",
        "agent",
        "aggregation_level",
        "n_finite_returns",
        "minimum_return",
        "return_quantile_0_05",
        "worst_decile_mean_return",
        "cvar_0_10_return",
        "cvar_0_05_return",
        "failure_probability",
        "collision_rate",
        "risk_zone_rate",
        "motor_saturation_rate",
        "source_file",
        "source_column",
    },
    "episode_tail_risk.csv": {
        "result_family",
        "environment",
        "agent",
        "aggregation_level",
        "n_finite_returns",
        "minimum_return",
        "return_quantile_0_05",
        "worst_decile_mean_return",
        "cvar_0_10_return",
        "cvar_0_05_return",
        "source_file",
        "source_column",
    },
    "checkpoint_tail_risk.csv": {
        "result_family",
        "environment",
        "agent",
        "checkpoint",
        "n_seeds",
        "minimum_return",
        "return_quantile_0_05",
        "cvar_0_10_return",
        "source_file",
        "source_column",
    },
    "tail_risk_summary.csv": {
        "result_family",
        "environment",
        "agent",
        "seed_minimum_return",
        "seed_return_quantile_0_05",
        "seed_worst_decile_mean_return",
        "seed_cvar_0_10_return",
        "seed_cvar_0_05_return",
        "worst_checkpoint_return",
        "maximum_learning_curve_drawdown",
        "seed_failure_probability",
        "seed_collision_rate",
        "seed_risk_zone_rate",
        "seed_motor_saturation_rate",
        "seed_source_file",
        "checkpoint_source_file",
    },
    "safety_metrics_manifest.csv": {
        "result_family",
        "environment",
        "agent",
        "metric_name",
        "definition",
        "direction",
        "aggregation_level",
        "denominator",
        "alpha",
        "source_file",
        "source_column",
        "availability",
        "reason_if_unavailable",
    },
}

TAIL_RISK_TABLE_COLUMNS = {
    "result_family",
    "environment",
    "agent",
    "seed_minimum_return",
    "seed_return_quantile_0_05",
    "seed_worst_decile_mean_return",
    "seed_cvar_0_10_return",
    "seed_cvar_0_05_return",
    "episode_minimum_return",
    "episode_cvar_0_10_return",
    "worst_checkpoint_return",
    "maximum_learning_curve_drawdown",
    "seed_failure_probability",
    "seed_collision_rate",
    "seed_risk_zone_rate",
    "seed_motor_saturation_rate",
}

INDEPENDENT_SHIFT_SCHEMAS = {
    "checkpoint_metrics.csv": {
        "mechanism_id", "agent", "seed", "checkpoint", "mean_return",
        "mean_success", "mean_failure", "source_file",
    },
    "seed_metrics.csv": {
        "mechanism_id", "agent", "seed", "normalized_return_auc",
        "success_auc", "failure_probability", "collision_rate",
        "risk_zone_rate", "branch_correctness", "source_file",
    },
    "planned_contrasts.csv": {
        "report_family", "evidence_class", "mechanism_id", "contrast",
        "left", "right", "metric", "n_pairs", "mean_difference",
        "median_difference", "bootstrap_ci_low", "bootstrap_ci_high",
        "cohen_dz", "rank_biserial", "paired_t_p", "paired_t_holm_p",
        "wilcoxon_p", "wilcoxon_holm_p", "wins", "losses", "ties",
        "source_file",
    },
    "lower_tail_metrics.csv": {
        "mechanism_id", "agent", "aggregation_level", "n_finite",
        "minimum_return", "return_quantile_0_05", "worst_decile_mean",
        "cvar_0_10", "cvar_0_05", "source_file", "source_column",
    },
    "detection_delay.csv": {
        "mechanism_id", "agent", "seed", "available", "detected",
        "right_censored", "detection_checkpoint",
        "detection_delay_interactions", "criterion", "reason",
        "source_file", "source_column",
    },
    "descriptive_agent_summary.csv": {
        "mechanism_id", "agent", "metric", "n_seeds", "mean",
        "bootstrap_ci_low", "bootstrap_ci_high", "available",
        "reason_if_unavailable", "inferential_status", "source_file",
    },
    "execution_audit.csv": {
        "mechanism_id", "severity_id", "expected_seeds",
        "completed_seeds_per_agent", "agents",
        "completed_agent_seed_runs", "expected_checkpoints_per_run",
        "evaluation_episodes_per_checkpoint", "audit_status", "source_file",
    },
}

INDEPENDENT_SHIFT_EXPECTED_ROWS = {
    "checkpoint_metrics.csv": 5850,
    "seed_metrics.csv": 450,
    "planned_contrasts.csv": 6,
    "lower_tail_metrics.csv": 30,
    "detection_delay.csv": 450,
    "descriptive_agent_summary.csv": 90,
    "execution_audit.csv": 3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_tail_risk(
    result_dir: Path,
    table_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict:
    violations: list[str] = []
    files: dict[str, dict[str, object]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, required_columns in TAIL_RISK_SCHEMAS.items():
        path = result_dir / name
        if name == "episode_tail_risk.csv" and not path.exists():
            continue
        if not path.exists():
            violations.append(f"missing tail-risk file: {name}")
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            violations.append(f"unreadable tail-risk file {name}: {error}")
            continue
        frames[name] = frame
        missing = required_columns - set(frame.columns)
        if missing:
            violations.append(f"{name} missing columns: {sorted(missing)}")
        if frame.empty:
            violations.append(f"{name} is empty")
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    identity = ["result_family", "environment", "agent"]
    summary = frames.get("tail_risk_summary.csv")
    if summary is not None:
        present_identity = [column for column in identity if column in summary]
        if len(present_identity) == len(identity):
            if summary[present_identity].isna().any().any():
                violations.append("tail_risk_summary.csv has null identity fields")
            if summary.duplicated(identity).any():
                violations.append("tail_risk_summary.csv has duplicate group rows")

    manifest = frames.get("safety_metrics_manifest.csv")
    if manifest is not None and TAIL_RISK_SCHEMAS[
        "safety_metrics_manifest.csv"
    ].issubset(manifest.columns):
        invalid_availability = set(manifest["availability"].dropna()) - {
            "available",
            "not_available",
        }
        if invalid_availability:
            violations.append(
                f"manifest has invalid availability values: {sorted(invalid_availability)}"
            )
        invalid_direction = set(manifest["direction"].dropna()) - {
            "higher_is_better",
            "lower_is_better",
        }
        if invalid_direction:
            violations.append(
                f"manifest has invalid direction values: {sorted(invalid_direction)}"
            )
        unavailable = manifest[manifest["availability"] == "not_available"]
        if unavailable["reason_if_unavailable"].fillna("").str.strip().eq("").any():
            violations.append("unavailable manifest rows require a reason")
        available = manifest[manifest["availability"] == "available"]
        if available["source_file"].fillna("").str.strip().eq("").any():
            violations.append("available manifest rows require a source_file")
        cvar = manifest[manifest["metric_name"].str.contains("cvar", case=False)]
        definition = cvar["definition"].fillna("").str.lower()
        if cvar.empty or not (
            definition.str.contains("arithmetic mean").all()
            and definition.str.contains(r"max\(1, ceil", regex=True).all()
            and definition.str.contains("finite").all()
        ):
            violations.append("manifest CVaR definitions do not match the locked estimator")
        alpha = set(pd.to_numeric(cvar["alpha"], errors="coerce").dropna())
        if not alpha.issubset({0.05, 0.10}) or not {0.05, 0.10}.issubset(alpha):
            violations.append(f"manifest CVaR alpha values are invalid: {sorted(alpha)}")
        episode_available = (
            (manifest["aggregation_level"] == "episode")
            & (manifest["availability"] == "available")
        ).any()
        if episode_available and "episode_tail_risk.csv" not in frames:
            violations.append(
                "episode-level evidence is available but episode_tail_risk.csv is missing"
            )
        forbidden_substitutions = manifest[
            manifest["metric_name"].isin(["recovery_time", "trajectory_deviation"])
        ]
        if not forbidden_substitutions.empty and (
            forbidden_substitutions["availability"] != "not_available"
        ).any():
            violations.append(
                "recovery time or trajectory deviation was synthesized without trace fields"
            )

    if table_path is not None:
        if not table_path.exists():
            violations.append(f"missing tail-risk table: {table_path}")
        else:
            table = pd.read_csv(table_path)
            missing = TAIL_RISK_TABLE_COLUMNS - set(table.columns)
            if missing:
                violations.append(f"tail-risk table missing columns: {sorted(missing)}")
            if table.empty:
                violations.append("tail-risk table is empty")
            files[display_name(table_path)] = {
                "sha256": sha256(table_path),
                "bytes": table_path.stat().st_size,
            }

    if figure_path is not None:
        if not figure_path.exists():
            violations.append(f"missing tail-risk figure: {figure_path}")
        else:
            with figure_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    violations.append("tail-risk figure does not have a PDF signature")
            files[display_name(figure_path)] = {
                "sha256": sha256(figure_path),
                "bytes": figure_path.stat().st_size,
            }

    return {
        "status": "PASS" if not violations else "FAIL",
        "result_dir": str(result_dir),
        "violations": violations,
        "files": files,
    }


def display_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def expected_checkpoints(config: dict, env_spec: dict) -> set[int]:
    budget = int(env_spec["training_steps"])
    interval = int(config["evaluation"]["interval_steps"])
    checkpoints = set(range(interval, budget + 1, interval))
    checkpoints.add(budget)
    return checkpoints


def raw_result_paths(result_dir: Path) -> list[Path]:
    for name in ("raw.csv", "raw.csv.gz", "raw.csv.xz"):
        candidate = result_dir / name
        if candidate.exists():
            return [candidate]
    parts = sorted((result_dir / "raw_parts").glob("*.csv*"))
    if parts:
        return parts
    raise FileNotFoundError(
        f"Missing raw.csv, compressed raw.csv, or raw_parts in {result_dir}"
    )


def audit_duplicate_rows(
    raw: pd.DataFrame, seed_metrics: pd.DataFrame | None = None
) -> list[str]:
    violations: list[str] = []
    raw_key = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
    ]
    if set(raw_key).issubset(raw.columns):
        duplicate_count = int(raw.duplicated(raw_key).sum())
        if duplicate_count:
            violations.append(f"duplicate raw rows: {duplicate_count}")
    if seed_metrics is not None:
        seed_key = ["environment", "agent", "seed", "checkpoint"]
        if set(seed_key).issubset(seed_metrics.columns):
            duplicate_count = int(seed_metrics.duplicated(seed_key).sum())
            if duplicate_count:
                violations.append(
                    f"duplicate seed/checkpoint rows: {duplicate_count}"
                )
    return violations


def audit_stability_logging(
    raw: pd.DataFrame,
) -> tuple[list[str], list[str], str]:
    present = STABILITY_COLUMNS & set(raw.columns)
    if not present:
        return (
            [],
            [
                "historical result: training loss was not logged; finite-loss "
                "behavior cannot be audited"
            ],
            "metric/checkpoint audit only",
        )
    if present != STABILITY_COLUMNS:
        missing = sorted(STABILITY_COLUMNS - present)
        return (
            [f"partial stability logging schema; missing columns: {missing}"],
            [],
            "invalid partial stability log",
        )

    violations: list[str] = []
    loss_count = pd.to_numeric(raw["nonfinite_loss_count"], errors="coerce")
    if loss_count.isna().any() or (~np.isfinite(loss_count)).any():
        violations.append("nonfinite_loss_count contains missing/non-finite values")
    elif (loss_count < 0).any() or (loss_count % 1 != 0).any():
        violations.append("nonfinite_loss_count must contain non-negative integers")
    elif (loss_count > 0).any():
        violations.append(
            f"non-finite logged losses: {int(loss_count.max())} cumulative"
        )

    updates = pd.to_numeric(raw["gradient_updates"], errors="coerce")
    for column in ("training_loss_mean", "training_loss_max"):
        values = pd.to_numeric(raw[column], errors="coerce")
        logged = updates > 0
        if values[logged].isna().any() or (~np.isfinite(values[logged])).any():
            violations.append(
                f"{column} is missing/non-finite after gradient updates"
            )

    for column in ("completed_checkpoint_count", "expected_checkpoint_count"):
        values = pd.to_numeric(raw[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            violations.append(f"{column} contains missing/non-finite values")
        elif (values < 0).any() or (values % 1 != 0).any():
            violations.append(f"{column} must contain non-negative integers")
    return violations, [], "metric/checkpoint/loss audit"


def audit(config_path: Path, result_dir: Path) -> dict:
    config = load_config(config_path)
    metadata = json.loads(
        (result_dir / "metadata.json").read_text(encoding="utf-8")
    )
    violations = []
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if metadata.get("config_sha256") != config_hash:
        violations.append("metadata config hash does not match config")
    warnings = []
    provenance_status = "LEGACY_COMMIT_ONLY"
    recorded_snapshot = metadata.get("source_snapshot_sha256")
    if recorded_snapshot:
        current_manifest = execution_input_manifest(config_path)
        current_snapshot = execution_snapshot_sha256(current_manifest)
        provenance_status = "STRICT_PASS"
        if metadata.get("execution_inputs_clean") is not True:
            violations.append("execution inputs were not clean at run start")
            provenance_status = "STRICT_FAIL"
        if recorded_snapshot != current_snapshot:
            violations.append(
                "recorded source snapshot does not match current execution inputs"
            )
            provenance_status = "STRICT_FAIL"
        if metadata.get("execution_input_manifest") != current_manifest:
            violations.append(
                "recorded execution input manifest does not match current inputs"
            )
            provenance_status = "STRICT_FAIL"
    else:
        warnings.append(
            "legacy result: source snapshot unavailable; commit-only provenance"
        )

    runs_dir = result_dir / "runs"
    temporary_files = list(runs_dir.glob("*.tmp"))
    if temporary_files:
        violations.append(f"incomplete run shards: {len(temporary_files)}")

    expected_runs = sum(
        len(spec.get("seeds", config["seeds"]))
        * sum(
            agent.get("applies_to_envs") is None
            or spec.get("name", spec["id"]) in agent["applies_to_envs"]
            for agent in config["agents"]
        )
        for spec in config["envs"]
    )
    shards = list(runs_dir.glob("*.csv"))
    if (shards or temporary_files) and len(shards) != expected_runs:
        violations.append(
            f"run shard count {len(shards)} != expected {expected_runs}"
        )

    raw_paths = raw_result_paths(result_dir)
    raw = pd.concat(
        [pd.read_csv(path) for path in raw_paths],
        ignore_index=True,
        sort=False,
    )
    seed_metrics_path = result_dir / "seed_metrics.csv"
    seed_metrics = (
        pd.read_csv(seed_metrics_path) if seed_metrics_path.exists() else None
    )
    observed_runs = len(
        raw[["environment", "agent", "seed"]].drop_duplicates()
    )
    if observed_runs != expected_runs:
        violations.append(
            f"raw run coverage {observed_runs} != expected {expected_runs}"
        )
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        violations.append(
            f"missing raw columns: {sorted(missing_columns)}"
        )
    if recorded_snapshot:
        for column, expected in (
            ("source_snapshot_sha256", recorded_snapshot),
            ("execution_inputs_clean", True),
            ("git_commit_hash", metadata.get("git_commit_hash")),
            ("package_version", metadata.get("package_version")),
        ):
            if column not in raw:
                violations.append(f"missing strict provenance column: {column}")
                provenance_status = "STRICT_FAIL"
                continue
            observed = set(raw[column].dropna().astype(str).str.lower())
            if observed != {str(expected).lower()}:
                violations.append(
                    f"raw {column} values do not match metadata"
                )
                provenance_status = "STRICT_FAIL"

    critical = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        "return",
        "environment_steps",
    ]
    if raw[critical].isna().any().any():
        violations.append("null values in critical raw columns")

    violations.extend(audit_duplicate_rows(raw, seed_metrics))
    stability_violations, stability_warnings, historical_audit_scope = (
        audit_stability_logging(raw)
    )
    violations.extend(stability_violations)
    warnings.extend(stability_warnings)

    env_by_name = {
        spec.get("name", spec["id"]): spec for spec in config["envs"]
    }
    for environment, env_spec in env_by_name.items():
        env_data = raw[raw["environment"] == environment]
        expected_agents = {
            agent["name"]
            for agent in config["agents"]
            if agent.get("applies_to_envs") is None
            or environment in agent["applies_to_envs"]
        }
        expected_seeds = {
            int(seed) for seed in env_spec.get("seeds", config["seeds"])
        }
        if set(env_data["agent"].unique()) != expected_agents:
            violations.append(f"{environment}: method coverage mismatch")
        budget = int(env_spec["training_steps"])
        checkpoints = expected_checkpoints(config, env_spec)
        for agent in expected_agents:
            method_data = env_data[env_data["agent"] == agent]
            if set(method_data["seed"].unique()) != expected_seeds:
                violations.append(
                    f"{environment}/{agent}: seed coverage mismatch"
                )
            for seed in expected_seeds:
                run = method_data[method_data["seed"] == seed]
                if int(run["environment_steps"].max()) != budget:
                    violations.append(
                        f"{environment}/{agent}/{seed}: step budget mismatch"
                    )
                observed = set(
                    run[run["phase"] == "eval"]["checkpoint"].astype(int)
                )
                if observed != checkpoints:
                    violations.append(
                        f"{environment}/{agent}/{seed}: checkpoint mismatch"
                    )
                evaluation = run[run["phase"] == "eval"]
                expected_episodes = int(config["evaluation"]["episodes"])
                episode_counts = evaluation.groupby("checkpoint")[
                    "episode"
                ].nunique()
                if not episode_counts.eq(expected_episodes).all():
                    violations.append(
                        f"{environment}/{agent}/{seed}: evaluation episode "
                        "count mismatch"
                    )
                mismatched_steps = evaluation[
                    evaluation["environment_steps"]
                    != evaluation["checkpoint"]
                ]
                if not mismatched_steps.empty:
                    violations.append(
                        f"{environment}/{agent}/{seed}: "
                        "evaluation step/checkpoint mismatch"
                    )
                if STABILITY_COLUMNS.issubset(raw.columns):
                    final_evaluation = evaluation[
                        evaluation["checkpoint"] == max(checkpoints)
                    ]
                    completed = pd.to_numeric(
                        final_evaluation["completed_checkpoint_count"],
                        errors="coerce",
                    )
                    expected_logged = pd.to_numeric(
                        final_evaluation["expected_checkpoint_count"],
                        errors="coerce",
                    )
                    if (
                        final_evaluation.empty
                        or not completed.eq(len(checkpoints)).all()
                        or not expected_logged.eq(len(checkpoints)).all()
                    ):
                        violations.append(
                            f"{environment}/{agent}/{seed}: completed/expected "
                            "checkpoint count mismatch"
                        )

    result_files = {}
    for name in (
        "seed_metrics.csv",
        "summary.csv",
        "pairwise.csv",
        "planned_contrasts.csv",
        "heavy_tail_diagnostics.csv",
        "cross_environment.csv",
        "metadata.json",
    ):
        path = result_dir / name
        if not path.exists():
            violations.append(f"missing result file: {name}")
        else:
            result_files[name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    for raw_path in raw_paths:
        logical_path = raw_path.relative_to(result_dir).as_posix()
        result_files[logical_path] = {
            "sha256": sha256(raw_path),
            "bytes": raw_path.stat().st_size,
        }
    raw_manifest = result_dir / "raw_parts" / "manifest.json"
    if raw_manifest.exists():
        result_files["raw_parts/manifest.json"] = {
            "sha256": sha256(raw_manifest),
            "bytes": raw_manifest.stat().st_size,
        }

    return {
        "status": "PASS" if not violations else "FAIL",
        "config": str(config_path),
        "result_dir": str(result_dir),
        "expected_runs": expected_runs,
        "observed_runs": observed_runs,
        "run_shards_present": len(shards),
        "raw_files": len(raw_paths),
        "raw_rows": len(raw),
        "violations": violations,
        "warnings": warnings,
        "provenance_status": provenance_status,
        "historical_audit_scope": historical_audit_scope,
        "files": result_files,
    }


def audit_fuzzy_crisp(
    config_path: Path, result_dir: Path, aggregate_dir: Path
) -> dict:
    """Audit Step 07 execution coverage and its matched-search artifacts."""

    report = audit(config_path, result_dir)
    standard_outputs = (
        "seed_metrics.csv",
        "summary.csv",
        "pairwise.csv",
        "planned_contrasts.csv",
        "heavy_tail_diagnostics.csv",
        "cross_environment.csv",
    )
    ignored = {f"missing result file: {name}" for name in standard_outputs}
    report["violations"] = [
        item for item in report["violations"] if item not in ignored
    ]
    canonical_raw = result_dir / "raw.csv.gz"
    if canonical_raw.exists():
        report["files"].pop("raw.csv", None)
        report["files"]["raw.csv.gz"] = {
            "sha256": sha256(canonical_raw),
            "bytes": canonical_raw.stat().st_size,
        }

    schemas = {
        "candidate_results.csv": {
            "environment", "mapping_family", "candidate_id", "seed",
            "primary_metric", "source_file", "execution_source_commit",
        },
        "candidate_ranks.csv": {
            "mapping_family", "candidate_id", "environment_rank",
            "mean_environment_rank", "worst_environment_rank",
            "median_latency_ns", "selected",
        },
        "budget_match_audit.csv": {
            "candidate_counts_equal", "seed_sets_equal",
            "training_steps_equal", "checkpoint_schedule_equal",
            "evaluation_episodes_equal", "gate_bounds_equal",
            "final_seed_rows", "audit_status",
        },
        "decision_surface_metrics.csv": {
            "local_lipschitz_style_mean", "local_lipschitz_style_max",
            "boundary_output_change_mean", "boundary_output_change_max",
            "perturbation_stability_rate",
        },
        "latency_complexity.csv": {
            "median_latency_ns", "cpu_threads", "warmup_calls",
            "arithmetic_flops_approximate", "comparisons",
        },
    }
    frames = {}
    for name, required in schemas.items():
        path = aggregate_dir / name
        if not path.exists():
            report["violations"].append(
                f"missing fuzzy/crisp result file: {name}"
            )
            continue
        frame = pd.read_csv(path)
        frames[name] = frame
        missing = sorted(required - set(frame.columns))
        if missing:
            report["violations"].append(
                f"{name} missing columns: {missing}"
            )
        report["files"][name] = {
            "sha256": sha256(path), "bytes": path.stat().st_size
        }

    candidates = frames.get("candidate_results.csv")
    if candidates is not None:
        if len(candidates) != 240:
            report["violations"].append(
                f"candidate_results.csv rows {len(candidates)} != 240"
            )
        seeds = pd.to_numeric(candidates["seed"], errors="coerce")
        if seeds.between(12000, 12099).any():
            report["violations"].append(
                "reserved final seed in candidate results"
            )
        counts = candidates.groupby("mapping_family")[
            "candidate_id"
        ].nunique().to_dict()
        if counts != {"crisp": 4, "fuzzy": 4}:
            report["violations"].append(
                f"fuzzy/crisp candidate count mismatch: {counts}"
            )
        for source in candidates["source_file"].unique():
            if not (ROOT / str(source)).exists():
                report["violations"].append(
                    f"candidate source file does not exist: {source}"
                )

    budget = frames.get("budget_match_audit.csv")
    if budget is not None:
        if len(budget) != 12 or not budget["audit_status"].eq("PASS").all():
            report["violations"].append(
                "budget match audit is not 12/12 PASS"
            )
        equal_columns = [
            "candidate_counts_equal", "seed_sets_equal",
            "training_steps_equal", "checkpoint_schedule_equal",
            "evaluation_episodes_equal", "gate_bounds_equal",
        ]
        for column in equal_columns:
            if column in budget and not budget[column].all():
                report["violations"].append(f"budget mismatch in {column}")
        if not budget["final_seed_rows"].eq(0).all():
            report["violations"].append(
                "budget audit reports final-seed rows"
            )

    selected = aggregate_dir / "selected_candidates.yaml"
    digest = aggregate_dir / "selected_candidates.sha256"
    if not selected.exists() or not digest.exists():
        report["violations"].append(
            "missing selected-candidate freeze files"
        )
    elif digest.read_text(encoding="utf-8").split()[0] != sha256(selected):
        report["violations"].append("selected-candidate SHA-256 mismatch")
    else:
        for path in (selected, digest):
            report["files"][path.name] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }

    report["analysis_type"] = "matched_fuzzy_crisp_development"
    report["aggregate_dir"] = str(aggregate_dir)
    report["status"] = (
        "PASS" if not report["violations"] else "FAIL"
    )
    return report


def audit_independent_shifts(
    result_dir: Path,
    table_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict:
    """Schema-check the combined locked Step 08 aggregation."""

    violations: list[str] = []
    files: dict[str, dict[str, object]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, required in INDEPENDENT_SHIFT_SCHEMAS.items():
        path = result_dir / name
        if not path.exists():
            violations.append(f"missing independent-shift file: {name}")
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            violations.append(f"unreadable independent-shift file {name}: {error}")
            continue
        frames[name] = frame
        missing = sorted(required - set(frame.columns))
        if missing:
            violations.append(f"{name} missing columns: {missing}")
        expected_rows = INDEPENDENT_SHIFT_EXPECTED_ROWS[name]
        if len(frame) != expected_rows:
            violations.append(f"{name} rows {len(frame)} != {expected_rows}")
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    key_specs = {
        "checkpoint_metrics.csv": ["mechanism_id", "agent", "seed", "checkpoint"],
        "seed_metrics.csv": ["mechanism_id", "agent", "seed"],
        "planned_contrasts.csv": ["mechanism_id", "contrast"],
        "lower_tail_metrics.csv": ["mechanism_id", "agent", "aggregation_level"],
        "detection_delay.csv": ["mechanism_id", "agent", "seed"],
        "descriptive_agent_summary.csv": ["mechanism_id", "agent", "metric"],
        "execution_audit.csv": ["mechanism_id"],
    }
    for name, keys in key_specs.items():
        frame = frames.get(name)
        if frame is not None and set(keys).issubset(frame.columns):
            if frame[keys].isna().any().any():
                violations.append(f"{name} has null identity fields")
            if frame.duplicated(keys).any():
                violations.append(f"{name} has duplicate rows at {keys}")

    seed = frames.get("seed_metrics.csv")
    if seed is not None:
        required = {
            "mechanism_id", "agent", "seed", "normalized_return_auc",
            "success_auc", "failure_probability", "collision_rate", "risk_zone_rate",
        }
        if required.issubset(seed.columns):
            groups = seed.groupby(["mechanism_id", "agent"]).size()
            if len(groups) != 15 or not groups.eq(30).all():
                violations.append("seed_metrics.csv is not 15 groups x 30 seeds")
            finite_columns = [
                "normalized_return_auc", "success_auc", "failure_probability",
                "collision_rate", "risk_zone_rate",
            ]
            numeric = seed[finite_columns].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy()).all():
                violations.append("seed_metrics.csv has non-finite required metrics")
            for column in finite_columns[1:]:
                if not numeric[column].between(0.0, 1.0).all():
                    violations.append(f"seed_metrics.csv {column} lies outside [0,1]")

    planned = frames.get("planned_contrasts.csv")
    if planned is not None and INDEPENDENT_SHIFT_SCHEMAS[
        "planned_contrasts.csv"
    ].issubset(planned.columns):
        if set(planned["report_family"]) != {"reviewer1_comment6_shift_primary"}:
            violations.append("planned contrasts use an unexpected report family")
        if set(planned["evidence_class"]) != {"replication"}:
            violations.append("planned contrasts are not replication evidence")
        if set(planned["n_pairs"].astype(int)) != {30}:
            violations.append("planned contrasts do not all contain 30 pairs")
        if planned["mechanism_id"].nunique() != 3 or not planned.groupby(
            "mechanism_id"
        ).size().eq(2).all():
            violations.append("planned contrasts are not two rows per generator")
        for raw, adjusted in (
            ("paired_t_p", "paired_t_holm_p"),
            ("wilcoxon_p", "wilcoxon_holm_p"),
        ):
            expected = np.asarray(holm_adjust(planned[raw].tolist()), dtype=float)
            observed = pd.to_numeric(planned[adjusted], errors="coerce").to_numpy()
            if not np.allclose(expected, observed, rtol=1e-8, atol=1e-12):
                violations.append(f"{adjusted} is inconsistent with six-row Holm")

    execution = frames.get("execution_audit.csv")
    if execution is not None and "audit_status" in execution:
        if not execution["audit_status"].eq("PASS").all():
            violations.append("execution_audit.csv contains a non-PASS mechanism")
        expected = {
            "expected_seeds": 30,
            "completed_seeds_per_agent": 30,
            "agents": 5,
            "completed_agent_seed_runs": 150,
            "expected_checkpoints_per_run": 24,
            "evaluation_episodes_per_checkpoint": 200,
        }
        for column, value in expected.items():
            if column in execution and not pd.to_numeric(
                execution[column], errors="coerce"
            ).eq(value).all():
                violations.append(f"execution_audit.csv {column} differs from lock")

    for frame_name, frame in frames.items():
        if "source_file" not in frame:
            continue
        for source in frame["source_file"].dropna().astype(str).unique():
            for item in source.split(";"):
                if item and not (ROOT / item).exists():
                    violations.append(f"{frame_name} source does not exist: {item}")

    if table_path is not None:
        if not table_path.exists():
            violations.append(f"missing independent-shift table: {table_path}")
        else:
            table = pd.read_csv(table_path)
            if len(table) != 6:
                violations.append("independent-shift table does not contain six rows")
            files[display_name(table_path)] = {
                "sha256": sha256(table_path), "bytes": table_path.stat().st_size
            }
    if figure_path is not None:
        if not figure_path.exists():
            violations.append(f"missing independent-shift figure: {figure_path}")
        else:
            with figure_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    violations.append("independent-shift figure is not a PDF")
            files[display_name(figure_path)] = {
                "sha256": sha256(figure_path), "bytes": figure_path.stat().st_size
            }

    return {
        "status": "PASS" if not violations else "FAIL",
        "analysis_type": "locked_independent_shift_replication",
        "result_dir": str(result_dir),
        "violations": violations,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--result-dir")
    parser.add_argument("--tail-risk-dir")
    parser.add_argument("--fuzzy-crisp-dir")
    parser.add_argument("--independent-shift-dir")
    parser.add_argument("--table")
    parser.add_argument("--figure")
    parser.add_argument("--output")
    args = parser.parse_args()
    explicit_mode = any(
        (
            args.config,
            args.result_dir,
            args.tail_risk_dir,
            args.fuzzy_crisp_dir,
            args.independent_shift_dir,
            args.table,
            args.figure,
        )
    )
    if not explicit_mode:
        required_csv = (
            "results/diagnostic_extensions/tail_risk/tail_risk_summary.csv",
            "results/diagnostic_extensions/multiplicity/multiplicity_family_manifest.csv",
            "results/diagnostic_extensions/multiplicity/significance_claim_audit.csv",
            "results/diagnostic_extensions/budgets/budget_equivalence_audit.csv",
            "results/diagnostic_extensions/final_shifts/planned_contrasts.csv",
            "results/diagnostic_extensions/support_final/planned_contrasts.csv",
            "results/diagnostic_extensions/support_scaling/scaling.csv",
            "results/diagnostic_extensions/sensor_factorial_development/sensor_factor_summary.csv",
            "results/diagnostic_extensions/temporal_interface_development/architecture_results.csv",
            "results/diagnostic_extensions/sensorized_final/planned_contrasts.csv",
            "results/diagnostic_extensions/safety_traces/safety_trace_summary.csv",
        )
        required_audits = (
            "results/diagnostic_extensions/final_shifts/audit.json",
            "results/diagnostic_extensions/sensor_factorial_development/execution_audit.json",
            "results/diagnostic_extensions/sensor_factorial_development/aggregation_audit.json",
            "results/diagnostic_extensions/temporal_interface_development/execution_audit.json",
            "results/diagnostic_extensions/sensorized_final/execution_audit.json",
            "results/diagnostic_extensions/sensorized_final/aggregation_audit.json",
            "results/diagnostic_extensions/safety_traces/execution_audit.json",
            "results/diagnostic_extensions/safety_traces/aggregation_audit.json",
        )
        violations: list[str] = []
        files: dict[str, dict[str, object]] = {}
        for relative in required_csv:
            path = ROOT / relative
            if not path.exists() or path.stat().st_size == 0:
                violations.append(f"missing reviewer1 result: {relative}")
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                violations.append(f"empty reviewer1 result: {relative}")
            files[relative] = {
                "rows": int(len(frame)),
                "sha256": sha256(path),
            }
        for relative in required_audits:
            path = ROOT / relative
            if not path.exists():
                violations.append(f"missing component audit: {relative}")
                continue
            component = json.loads(path.read_text(encoding="utf-8"))
            if component.get("status") != "PASS":
                violations.append(
                    f"component audit not PASS: {relative}="
                    f"{component.get('status')}"
                )
            files[relative] = {"status": component.get("status"), "sha256": sha256(path)}
        report = {
            "status": "PASS" if not violations else "FAIL",
            "analysis_type": "diagnostic_extensions_comprehensive_read_only_audit",
            "violations": violations,
            "files": files,
        }
    elif args.independent_shift_dir:
        if args.config or args.result_dir or args.tail_risk_dir or args.fuzzy_crisp_dir:
            parser.error(
                "--independent-shift-dir cannot be combined with other audit modes"
            )
        report = audit_independent_shifts(
            Path(args.independent_shift_dir),
            Path(args.table) if args.table else None,
            Path(args.figure) if args.figure else None,
        )
    elif args.fuzzy_crisp_dir:
        if args.tail_risk_dir or not args.config or not args.result_dir:
            parser.error(
                "--fuzzy-crisp-dir requires --config/--result-dir and cannot "
                "be combined with --tail-risk-dir"
            )
        report = audit_fuzzy_crisp(
            Path(args.config),
            Path(args.result_dir),
            Path(args.fuzzy_crisp_dir),
        )
    elif args.tail_risk_dir:
        if args.config or args.result_dir:
            parser.error("--tail-risk-dir cannot be combined with --config/--result-dir")
        report = audit_tail_risk(
            Path(args.tail_risk_dir),
            Path(args.table) if args.table else None,
            Path(args.figure) if args.figure else None,
        )
    else:
        if not args.config or not args.result_dir:
            parser.error("--config and --result-dir are required for experiment audit")
        report = audit(Path(args.config), Path(args.result_dir))
    output = Path(args.output) if args.output else (
        ROOT / "audits/final_results_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(report["status"])
    for violation in report["violations"]:
        print(violation)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
