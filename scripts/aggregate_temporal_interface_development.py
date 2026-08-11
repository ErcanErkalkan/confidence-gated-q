from __future__ import annotations

import argparse
import hashlib
import json
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

from hybrid_q.statistics import empirical_lower_cvar  # noqa: E402
from scripts.run_temporal_interface_development import (  # noqa: E402
    CAPACITY_CONTROLS,
    DEFAULT_CONFIG,
    DEVELOPMENT_SEEDS,
    EXPECTED_ASSIGNMENTS,
    INTERFACE_CONDITIONS,
    TEMPORAL_AGENTS,
    validate_config,
)


OUTPUT_DIR = ROOT / "results/diagnostic_extensions/temporal_interface_development"
RAW_SOURCE = (
    "results/diagnostic_extensions/temporal_interface_development/raw.csv"
)


class TemporalInterfaceAggregationError(ValueError):
    pass


def _return_auc(group: pd.DataFrame) -> float:
    checkpoint = (
        group.groupby("checkpoint", as_index=False)["eval_return"]
        .mean()
        .sort_values("checkpoint")
    )
    x = checkpoint["checkpoint"].to_numpy(dtype=float)
    y = checkpoint["eval_return"].to_numpy(dtype=float)
    if len(x) < 2 or not np.all(np.diff(x) > 0.0):
        raise TemporalInterfaceAggregationError("invalid checkpoint AUC support")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _seed_summaries(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (environment, agent, seed), group in evaluation.groupby(
        ["environment", "agent", "seed"], sort=True
    ):
        returns = pd.to_numeric(group["eval_return"], errors="coerce")
        rows.append(
            {
                "environment": environment,
                "agent": agent,
                "seed": int(seed),
                "normalized_return_auc": _return_auc(group),
                "success_rate": float(group["success"].mean()),
                "failure_probability": float(1.0 - group["success"].mean()),
                "collision_rate": float(group["collision_rate"].mean()),
                "localization_error_mean": float(
                    group["localization_error_mean"].mean()
                ),
                "episode_lower_tail_return_0_10": empirical_lower_cvar(
                    returns, 0.10
                ),
                "inference_time_us_per_decision_mean": float(
                    group["inference_time_us_per_decision_mean"].mean()
                ),
                "gradient_updates": int(group["gradient_updates"].max()),
                "source_row_count": len(group),
                "source_file": RAW_SOURCE,
            }
        )
    return pd.DataFrame(rows)


def build_architecture_results(seed_rows: pd.DataFrame) -> pd.DataFrame:
    candidates = TEMPORAL_AGENTS | CAPACITY_CONTROLS
    scoped = seed_rows[seed_rows["agent"].isin(candidates)]
    rows = []
    for candidate, group in scoped.groupby("agent", sort=True):
        family = (
            "temporal_candidate"
            if candidate in TEMPORAL_AGENTS
            else "feedforward_capacity_control"
        )
        rows.append(
            {
                "candidate_id": candidate,
                "candidate_family": family,
                "temporal_memory": candidate in TEMPORAL_AGENTS,
                "environment": str(group["environment"].iloc[0]),
                "n_seeds": group["seed"].nunique(),
                "seed_set": ";".join(
                    map(str, sorted(group["seed"].astype(int).unique()))
                ),
                "mean_normalized_return_auc": float(
                    group["normalized_return_auc"].mean()
                ),
                "worst_seed_return_auc": float(
                    group["normalized_return_auc"].min()
                ),
                "mean_success_rate": float(group["success_rate"].mean()),
                "failure_probability": float(
                    group["failure_probability"].mean()
                ),
                "collision_rate": float(group["collision_rate"].mean()),
                "localization_error_mean": float(
                    group["localization_error_mean"].mean()
                ),
                "episode_lower_tail_return_0_10": float(
                    group["episode_lower_tail_return_0_10"].mean()
                ),
                "inference_time_us_per_decision_mean": float(
                    group["inference_time_us_per_decision_mean"].mean()
                ),
                "mean_gradient_updates": float(
                    group["gradient_updates"].mean()
                ),
                "source_row_count": int(group["source_row_count"].sum()),
                "source_file": RAW_SOURCE,
                "evidence_class": "development_diagnostic",
            }
        )
    result = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    if len(result) != 6 or result["candidate_id"].duplicated().any():
        raise TemporalInterfaceAggregationError(
            "architecture candidate coverage mismatch"
        )
    if set(result["n_seeds"].astype(int)) != {len(DEVELOPMENT_SEEDS)}:
        raise TemporalInterfaceAggregationError("architecture seed mismatch")
    return result


def select_temporal_model(architecture: pd.DataFrame) -> pd.Series:
    eligible = architecture[
        architecture["candidate_family"].eq("temporal_candidate")
    ].copy()
    if set(eligible["candidate_id"]) != TEMPORAL_AGENTS:
        raise TemporalInterfaceAggregationError("temporal selection scope mismatch")
    return eligible.sort_values(
        [
            "mean_normalized_return_auc",
            "failure_probability",
            "inference_time_us_per_decision_mean",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]


def build_interface_ablation(seed_rows: pd.DataFrame) -> pd.DataFrame:
    scoped = seed_rows[seed_rows["environment"].isin(INTERFACE_CONDITIONS)]
    rows = []
    for condition, group in scoped.groupby("environment", sort=True):
        observation_mode, control_mode = INTERFACE_CONDITIONS[condition]
        rows.append(
            {
                "condition": condition,
                "observation_mode": observation_mode,
                "control_interface_mode": control_mode,
                "n_seeds": group["seed"].nunique(),
                "seed_set": ";".join(
                    map(str, sorted(group["seed"].astype(int).unique()))
                ),
                "mean_normalized_return_auc": float(
                    group["normalized_return_auc"].mean()
                ),
                "mean_success_rate": float(group["success_rate"].mean()),
                "failure_probability": float(
                    group["failure_probability"].mean()
                ),
                "collision_rate": float(group["collision_rate"].mean()),
                "localization_error_mean": float(
                    group["localization_error_mean"].mean()
                ),
                "episode_lower_tail_return_0_10": float(
                    group["episode_lower_tail_return_0_10"].mean()
                ),
                "inference_time_us_per_decision_mean": float(
                    group["inference_time_us_per_decision_mean"].mean()
                ),
                "source_row_count": int(group["source_row_count"].sum()),
                "source_file": RAW_SOURCE,
                "evidence_class": "development_diagnostic",
            }
        )
    result = pd.DataFrame(rows)
    if set(result["condition"]) != set(INTERFACE_CONDITIONS):
        raise TemporalInterfaceAggregationError("interface coverage mismatch")
    baseline = result[
        result["condition"].eq("interface_D_sensorized_low_level")
    ].iloc[0]
    for metric in (
        "mean_normalized_return_auc",
        "mean_success_rate",
        "failure_probability",
        "collision_rate",
        "episode_lower_tail_return_0_10",
    ):
        result[f"delta_{metric}_vs_D"] = result[metric] - baseline[metric]
    return result.sort_values("condition").reset_index(drop=True)


def build_budget_audit(
    raw: pd.DataFrame,
    design: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for record in design.to_dict(orient="records"):
        group = raw[
            raw["environment"].eq(record["environment"])
            & raw["agent"].eq(record["candidate_or_agent"])
        ]
        evaluation = group[group["phase"].eq("eval")]
        seeds = sorted(group["seed"].astype(int).unique())
        checkpoints = sorted(evaluation["checkpoint"].astype(int).unique())
        episode_counts = evaluation.groupby(["seed", "checkpoint"]).size()
        final_rows = evaluation[evaluation["checkpoint"].eq(240)]
        final_gradient_updates = sorted(
            final_rows["gradient_updates"].astype(int).unique()
        )
        nonfinite_loss_count = int(
            pd.to_numeric(group["nonfinite_loss_count"], errors="coerce")
            .fillna(0)
            .max()
        )
        status = (
            seeds == DEVELOPMENT_SEEDS
            and checkpoints == [120, 240]
            and episode_counts.eq(config["evaluation"]["episodes"]).all()
            and int(group["environment_steps"].max()) == 240
            and final_gradient_updates == [53]
            and nonfinite_loss_count == 0
        )
        rows.append(
            {
                **record,
                "observed_run_count": group[
                    ["environment", "agent", "seed"]
                ].drop_duplicates().shape[0],
                "observed_evaluation_rows": len(evaluation),
                "final_gradient_updates": ";".join(
                    map(str, final_gradient_updates)
                ),
                "nonfinite_loss_count": nonfinite_loss_count,
                "final_seed_rows": int(
                    group["seed"].astype(int).between(16000, 16099).sum()
                ),
                "audit_status": "PASS" if status else "FAIL",
            }
        )
    result = pd.DataFrame(rows)
    family_counts = result[
        result["family"].isin(
            ["temporal_candidate", "feedforward_capacity_control"]
        )
    ].groupby("family")["candidate_or_agent"].nunique()
    if family_counts.to_dict() != {
        "feedforward_capacity_control": 3,
        "temporal_candidate": 3,
    }:
        raise TemporalInterfaceAggregationError("search budget count mismatch")
    if not result["audit_status"].eq("PASS").all():
        raise TemporalInterfaceAggregationError("matched budget audit failed")
    return result


def _selected_definition(
    selected: pd.Series, config: dict[str, Any]
) -> dict[str, Any]:
    agent = next(
        agent
        for agent in config["agents"]
        if agent["name"] == selected["candidate_id"]
    )
    environment_name = next(iter(EXPECTED_ASSIGNMENTS[agent["name"]]))
    environment = next(
        env for env in config["envs"] if env["name"] == environment_name
    )
    return {
        "schema_version": 1,
        "evidence_class": "development_diagnostic",
        "selection_scope": "temporal_candidates_only",
        "selection_rule": config["analysis"]["selection_rule"],
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_mean_normalized_return_auc": float(
            selected["mean_normalized_return_auc"]
        ),
        "selected_failure_probability": float(
            selected["failure_probability"]
        ),
        "selected_inference_time_us_per_decision_mean": float(
            selected["inference_time_us_per_decision_mean"]
        ),
        "agent_kind": agent["kind"],
        "agent_params": agent["params"],
        "environment_transform": {
            "frame_stack": int(environment.get("frame_stack", 1)),
            "temporal_filter_alpha": environment.get(
                "temporal_filter_alpha"
            ),
        },
        "development_seeds": DEVELOPMENT_SEEDS,
        "final_seed_results_used": False,
        "source_file": RAW_SOURCE,
    }


def aggregate(
    output_dir: Path = OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    design = validate_config(config)
    raw = pd.read_csv(output_dir / "raw.csv")
    if raw["seed"].astype(int).between(16000, 16099).any():
        raise TemporalInterfaceAggregationError("reserved final seed row detected")
    evaluation = raw[raw["phase"].eq("eval")].copy()
    seed_rows = _seed_summaries(evaluation)
    architecture = build_architecture_results(seed_rows)
    interface = build_interface_ablation(seed_rows)
    budget = build_budget_audit(raw, design, config)
    selected = select_temporal_model(architecture)
    selected_definition = _selected_definition(selected, config)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    selected_definition["execution_source_commit"] = metadata["git_commit_hash"]
    selected_definition["source_snapshot_sha256"] = metadata[
        "source_snapshot_sha256"
    ]
    selected_definition["config_sha256"] = metadata["config_sha256"]

    output_dir.mkdir(parents=True, exist_ok=True)
    architecture.to_csv(output_dir / "architecture_results.csv", index=False)
    interface.to_csv(output_dir / "interface_ablation.csv", index=False)
    budget.to_csv(output_dir / "budget_audit.csv", index=False)
    selected_path = output_dir / "selected_temporal_model.yaml"
    selected_path.write_text(
        yaml.safe_dump(selected_definition, sort_keys=False), encoding="utf-8"
    )
    digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    selected_path.with_suffix(".sha256").write_text(
        f"{digest}  {selected_path.name}\n", encoding="utf-8"
    )
    print(
        "TEMPORAL_INTERFACE_AGGREGATION_PASS "
        f"selected={selected['candidate_id']} budget_rows={len(budget)}"
    )
    return {
        "architecture_results": architecture,
        "interface_ablation": interface,
        "budget_audit": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    aggregate(args.output_dir.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
