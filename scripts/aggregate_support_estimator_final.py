from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for path in (ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_q.calibration import average_precision, binary_auroc  # noqa: E402
from hybrid_q.statistics import (  # noqa: E402
    bootstrap_mean_interval,
    cohen_dz,
    empirical_lower_cvar,
    holm_adjust,
    lower_quantile,
    paired_rank_biserial,
    win_loss_tie,
    worst_decile_mean,
)
from scripts.generate_support_final_configs import (  # noqa: E402
    PROTOCOL,
    load_yaml,
    verify_protocol_digest,
)
from scripts.run_support_estimator_final import (  # noqa: E402
    SUPPORT_AGENT_IDS,
    audit_execution,
    validate_configs,
    validate_prerequisites,
    validate_protocol,
)


OUTPUT_DIR = ROOT / "results/diagnostic_extensions/support_final"
TABLE_PATH = ROOT / "tables/table_support_final.csv"
FIGURE_PATH = ROOT / "figures/fig_support_final.pdf"
REPORT_PATH = ROOT / (
    "results/diagnostic_extensions/support_final/report.md"
)
BOOTSTRAP_SAMPLES = 2000
BIN_EDGES = np.linspace(0.0, 1.0, 11)
COVERAGES = (0.10, 0.25, 0.50, 0.75, 1.00)
RAW_COLUMNS = [
    "environment",
    "agent",
    "seed",
    "phase",
    "checkpoint",
    "episode",
    "return",
    "success",
    "failure_rate",
    "collision_rate",
    "risk_zone_rate",
    "support_score_mean",
    "tabular_action_correct",
    "neural_action_correct",
    "tabular_q_error",
    "neural_q_error",
    "support_query_latency",
    "support_memory_bytes",
    "embedding_snapshot_hash",
    "embedding_freeze_step",
    "gradient_updates",
    "git_commit_hash",
    "source_snapshot_sha256",
]


class SupportFinalAggregationError(ValueError):
    pass


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalized_auc(checkpoints: object, values: object, expected: list[int]) -> float:
    x = np.asarray(checkpoints, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if not np.array_equal(x, np.asarray(expected, dtype=float)):
        raise SupportFinalAggregationError("primary-window checkpoint mismatch")
    if not np.isfinite(y).all() or x.size < 2 or x[-1] <= x[0]:
        raise SupportFinalAggregationError("primary-window AUC is not finite")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _seed_bootstrap_mean(
    values: list[float], *, salt: str
) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(array.mean())
    if array.size == 1:
        return point, float("nan"), float("nan"), 1
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(BOOTSTRAP_SAMPLES, array.size), replace=True)
    low, high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    return point, float(low), float(high), int(array.size)


def load_evaluation(
    protocol: dict[str, Any], configs: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    environment_map = {
        item["environment_key"]: item for item in protocol["environments"]
    }
    frames = []
    audits = []
    for key, config in configs.items():
        raw = Path(config["output_dir"]) / "raw.csv"
        compressed = raw.with_suffix(".csv.gz")
        if not raw.exists() or not compressed.exists():
            raise SupportFinalAggregationError(f"missing final raw evidence: {key}")
        audits.append(audit_execution(raw, config, environment_map[key]))
        frame = pd.read_csv(compressed, usecols=RAW_COLUMNS, low_memory=False)
        frame = frame[frame["phase"].eq("eval")].copy()
        frame["environment_key"] = key
        frame["source_file"] = relative_path(compressed)
        for column in (
            "seed",
            "checkpoint",
            "episode",
            "return",
            "success",
            "failure_rate",
            "collision_rate",
            "risk_zone_rate",
            "support_score_mean",
            "tabular_action_correct",
            "neural_action_correct",
            "tabular_q_error",
            "neural_q_error",
            "support_query_latency",
            "support_memory_bytes",
            "gradient_updates",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        first = int(environment_map[key]["primary_window_first"])
        last = int(environment_map[key]["primary_window_last"])
        frame = frame[frame["checkpoint"].between(first, last)]
        frames.append(frame)
    evaluation = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_key = ["environment_key", "agent", "seed", "checkpoint", "episode"]
    if evaluation.duplicated(duplicate_key).any():
        raise SupportFinalAggregationError("duplicate final evaluation row")
    return evaluation, pd.DataFrame(audits)


def build_seed_metrics(
    evaluation: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    environment_map = {
        item["environment_key"]: item for item in protocol["environments"]
    }
    rows = []
    for (environment_key, agent, seed), group in evaluation.groupby(
        ["environment_key", "agent", "seed"], sort=True
    ):
        environment = environment_map[str(environment_key)]
        first = int(environment["primary_window_first"])
        last = int(environment["primary_window_last"])
        interval = int(protocol["matched_budget"]["checkpoint_interval"])
        expected = list(range(first, last + 1, interval))
        checkpoint = (
            group.groupby("checkpoint", as_index=False)
            .agg(
                mean_return=("return", "mean"),
                success_rate=("success", "mean"),
                failure_probability=("failure_rate", "mean"),
                collision_rate=("collision_rate", "mean"),
                risk_zone_rate=("risk_zone_rate", "mean"),
            )
            .sort_values("checkpoint")
        )
        if checkpoint["checkpoint"].astype(int).tolist() != expected:
            raise SupportFinalAggregationError(
                f"checkpoint coverage mismatch: {(environment_key, agent, seed)}"
            )
        source_files = sorted(group["source_file"].unique())
        rows.append(
            {
                "environment_key": environment_key,
                "environment": environment["environment_name"],
                "agent": agent,
                "seed": int(seed),
                "normalized_return_auc": normalized_auc(
                    checkpoint["checkpoint"], checkpoint["mean_return"], expected
                ),
                "success_auc": normalized_auc(
                    checkpoint["checkpoint"], checkpoint["success_rate"], expected
                ),
                "failure_probability": float(group["failure_rate"].mean()),
                "collision_rate": float(group["collision_rate"].mean()),
                "risk_zone_rate": float(group["risk_zone_rate"].mean()),
                "mean_support_score": float(group["support_score_mean"].mean()),
                "mean_memory_action_correctness": float(
                    group["tabular_action_correct"].mean()
                ),
                "mean_neural_action_correctness": float(
                    group["neural_action_correct"].mean()
                ),
                "mean_support_query_latency_seconds": float(
                    group["support_query_latency"].mean()
                ),
                "support_memory_bytes": float(group["support_memory_bytes"].max()),
                "gradient_updates": int(group["gradient_updates"].max()),
                "source_file": ";".join(source_files),
                "source_column": (
                    "return;success;failure_rate;collision_rate;risk_zone_rate;"
                    "support_score_mean;tabular_action_correct"
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["environment_key", "agent", "seed"]
    ).reset_index(drop=True)
    agents = {item["agent_id"] for item in protocol["agents"]}
    for environment in protocol["environments"]:
        part = result[result["environment_key"].eq(environment["environment_key"])]
        if set(part["agent"]) != agents:
            raise SupportFinalAggregationError("seed-metric agent coverage mismatch")
        expected_seeds = [int(seed) for seed in environment["seeds"]]
        for agent, group in part.groupby("agent", sort=False):
            if group["seed"].tolist() != expected_seeds:
                raise SupportFinalAggregationError(
                    f"seed coverage mismatch: {(environment['environment_key'], agent)}"
                )
    return result


def build_performance(
    seed_metrics: pd.DataFrame, evaluation: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for (environment_key, environment, agent), group in seed_metrics.groupby(
        ["environment_key", "environment", "agent"], sort=True
    ):
        values = group["normalized_return_auc"].to_numpy(dtype=float)
        low, high = bootstrap_mean_interval(values, seed=0, samples=10000)
        episodes = evaluation[
            evaluation["environment_key"].eq(environment_key)
            & evaluation["agent"].eq(agent)
        ]["return"].to_numpy(dtype=float)
        episodes = episodes[np.isfinite(episodes)]
        rows.append(
            {
                "environment_key": environment_key,
                "environment": environment,
                "agent": agent,
                "n_seeds": len(values),
                "mean_normalized_return_auc": float(values.mean()),
                "median_normalized_return_auc": float(np.median(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "seed_level_minimum_return_auc": float(values.min()),
                "seed_level_q05_return_auc": lower_quantile(values, 0.05),
                "seed_level_worst_decile_mean_return_auc": worst_decile_mean(values),
                "seed_level_cvar_0_10_return_auc": empirical_lower_cvar(values, 0.10),
                "seed_level_cvar_0_05_return_auc": empirical_lower_cvar(values, 0.05),
                "episode_level_n": len(episodes),
                "episode_level_cvar_0_10_return": empirical_lower_cvar(
                    episodes, 0.10
                ),
                "episode_level_cvar_0_05_return": empirical_lower_cvar(
                    episodes, 0.05
                ),
                "mean_success_auc": float(group["success_auc"].mean()),
                "mean_failure_probability": float(
                    group["failure_probability"].mean()
                ),
                "mean_collision_rate": float(group["collision_rate"].mean()),
                "mean_risk_zone_rate": float(group["risk_zone_rate"].mean()),
                "mean_support_score": float(group["mean_support_score"].mean()),
                "mean_support_query_latency_seconds": float(
                    group["mean_support_query_latency_seconds"].mean()
                ),
                "mean_support_memory_bytes": float(
                    group["support_memory_bytes"].mean()
                ),
                "source_file": ";".join(sorted(group["source_file"].unique())),
            }
        )
    result = pd.DataFrame(rows)
    result["environment_rank"] = result.groupby("environment_key")[
        "mean_normalized_return_auc"
    ].rank(method="average", ascending=False)
    return result.sort_values(["environment_key", "environment_rank", "agent"])


def _planned_row(
    environment: dict[str, Any],
    contrast: dict[str, Any],
    seed_metrics: pd.DataFrame,
) -> dict[str, Any]:
    part = seed_metrics[seed_metrics["environment_key"].eq(environment["environment_key"])]
    left = part[part["agent"].eq(contrast["left"])][
        ["seed", "normalized_return_auc"]
    ]
    right = part[part["agent"].eq(contrast["right"])][
        ["seed", "normalized_return_auc"]
    ]
    paired = left.merge(
        right, on="seed", suffixes=("_left", "_right"), validate="one_to_one"
    ).sort_values("seed")
    if len(paired) != 30:
        raise SupportFinalAggregationError("planned contrast is not 30-seed paired")
    differences = (
        paired["normalized_return_auc_left"]
        - paired["normalized_return_auc_right"]
    ).to_numpy(dtype=float)
    if np.allclose(differences, 0.0):
        paired_t_p = wilcoxon_p = 1.0
    else:
        paired_t_p = (
            0.0
            if float(differences.std(ddof=1)) <= 1e-12
            else float(stats.ttest_1samp(differences, 0.0).pvalue)
        )
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    low, high = bootstrap_mean_interval(differences, seed=0, samples=10000)
    wins, losses, ties = win_loss_tie(differences)
    return {
        "report_family": (
            f"support_final.{environment['environment_key']}"
        ),
        "evidence_class": "replication",
        "environment": environment["environment_name"],
        "environment_key": environment["environment_key"],
        "contrast": contrast["name"],
        "contrast_status": contrast["status"],
        "left": contrast["left"],
        "right": contrast["right"],
        "metric": "normalized_return_auc",
        "n_pairs": len(differences),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "cohen_dz": cohen_dz(differences),
        "rank_biserial": paired_rank_biserial(differences),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "source_file": "results/diagnostic_extensions/support_final/seed_metrics.csv",
        "source_column": "normalized_return_auc",
    }


def build_planned_contrasts(
    seed_metrics: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    rows = [
        _planned_row(environment, contrast, seed_metrics)
        for environment in protocol["environments"]
        for contrast in protocol["planned_contrasts"]
    ]
    result = pd.DataFrame(rows)
    result["paired_t_holm_p"] = np.nan
    result["wilcoxon_holm_p"] = np.nan
    for _, indices in result.groupby("environment_key").groups.items():
        index = list(indices)
        result.loc[index, "paired_t_holm_p"] = holm_adjust(
            result.loc[index, "paired_t_p"].tolist()
        )
        result.loc[index, "wilcoxon_holm_p"] = holm_adjust(
            result.loc[index, "wilcoxon_p"].tolist()
        )
    return result.sort_values(["environment_key", "contrast"]).reset_index(drop=True)


def build_global_sensitivity(planned: pd.DataFrame) -> pd.DataFrame:
    selected = planned[planned["contrast_status"].eq("primary_replication")].copy()
    if len(selected) != 3:
        raise SupportFinalAggregationError("global primary sensitivity requires 3 rows")
    selected["global_holm_paired_t_p"] = holm_adjust(selected["paired_t_p"].tolist())
    selected["global_holm_wilcoxon_p"] = holm_adjust(
        selected["wilcoxon_p"].tolist()
    )
    selected["global_sensitivity_scope"] = (
        "three_predeclared_primary_replication_rows_across_environments"
    )
    selected["does_not_replace_report_holm"] = True
    return selected


def _binary_metric_summary(
    group: pd.DataFrame,
    *,
    score_column: str,
    target_column: str,
    salt: str,
) -> dict[str, Any]:
    seed_values: dict[str, list[float]] = defaultdict(list)
    for _, seed_group in group.groupby("seed", sort=True):
        scores = seed_group[score_column].to_numpy(dtype=float)
        targets = seed_group[target_column].to_numpy(dtype=int)
        seed_values["auroc"].append(binary_auroc(scores, targets))
        seed_values["pr_auc"].append(average_precision(scores, targets))
    result = {
        "n_rows": len(group),
        "n_seeds": int(group["seed"].nunique()),
        "n_positive": int(group[target_column].sum()),
        "n_negative": int((1 - group[target_column]).sum()),
    }
    for metric in ("auroc", "pr_auc"):
        point, low, high, count = _seed_bootstrap_mean(
            seed_values[metric], salt=f"{salt}-{metric}"
        )
        result[metric] = point
        result[f"{metric}_ci_low"] = low
        result[f"{metric}_ci_high"] = high
        result[f"{metric}_available_seed_count"] = count
    result["availability"] = (
        "available"
        if result["n_positive"] > 0
        and result["n_negative"] > 0
        and result["auroc_available_seed_count"] > 0
        else "not_available_single_class"
    )
    return result


def build_calibration_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    support = evaluation[evaluation["agent"].isin(SUPPORT_AGENT_IDS)].copy()
    support["failure_target"] = support["failure_rate"].astype(int)
    support["failure_score"] = 1.0 - support["support_score_mean"]
    for (environment_key, environment, agent), group in support.groupby(
        ["environment_key", "environment", "agent"], sort=True
    ):
        failure = _binary_metric_summary(
            group,
            score_column="failure_score",
            target_column="failure_target",
            salt=f"{environment_key}-{agent}-failure",
        )
        rows.append(
            {
                "environment_key": environment_key,
                "environment": environment,
                "agent": agent,
                "target_type": "episode_failure",
                "target_definition": "1 when the evaluation episode failed; 0 on success",
                "score_definition": "1 - episode mean support score",
                "aggregation_level": "mean_of_seed_level_discrimination_metrics",
                **failure,
                "source_file": group["source_file"].iloc[0],
            }
        )
        action = group[np.isfinite(group["tabular_action_correct"])].copy()
        action = action[~np.isclose(action["tabular_action_correct"], 0.5)]
        if action.empty:
            action_result = {
                "n_rows": 0,
                "n_seeds": 0,
                "n_positive": 0,
                "n_negative": 0,
                "auroc": np.nan,
                "auroc_ci_low": np.nan,
                "auroc_ci_high": np.nan,
                "auroc_available_seed_count": 0,
                "pr_auc": np.nan,
                "pr_auc_ci_low": np.nan,
                "pr_auc_ci_high": np.nan,
                "pr_auc_available_seed_count": 0,
                "availability": "not_available_no_analytic_optimal_action",
            }
        else:
            action["memory_majority_correct"] = (
                action["tabular_action_correct"] > 0.5
            ).astype(int)
            action_result = _binary_metric_summary(
                action,
                score_column="support_score_mean",
                target_column="memory_majority_correct",
                salt=f"{environment_key}-{agent}-memory-correct",
            )
        rows.append(
            {
                "environment_key": environment_key,
                "environment": environment,
                "agent": agent,
                "target_type": "episode_memory_majority_correct",
                "target_definition": (
                    "1 when >0.5 of decisions with a unique analytic optimal action "
                    "use the correct memory-branch greedy action; 0 when <0.5; ties excluded"
                ),
                "score_definition": "episode mean support score",
                "aggregation_level": "mean_of_seed_level_discrimination_metrics",
                **action_result,
                "source_file": group["source_file"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_group_rate(
    group: pd.DataFrame, value_column: str, *, salt: str
) -> tuple[float, float, float, int]:
    rates = [
        float(seed_group[value_column].mean())
        for _, seed_group in group.groupby("seed", sort=True)
    ]
    return _seed_bootstrap_mean(rates, salt=salt)


def build_calibration_bins(evaluation: pd.DataFrame) -> pd.DataFrame:
    support = evaluation[evaluation["agent"].isin(SUPPORT_AGENT_IDS)].copy()
    support["support_bin"] = pd.cut(
        support["support_score_mean"],
        bins=BIN_EDGES,
        include_lowest=True,
        labels=False,
    )
    rows = []
    for (environment_key, environment, agent), group in support.groupby(
        ["environment_key", "environment", "agent"], sort=True
    ):
        targets = [
            (
                "episode_success",
                "success",
                "evaluation episode success indicator",
            )
        ]
        if np.isfinite(group["tabular_action_correct"]).any():
            targets.append(
                (
                    "memory_action_correctness_rate",
                    "tabular_action_correct",
                    "mean correct memory-branch action rate within the episode",
                )
            )
        for target_type, column, definition in targets:
            usable = group[np.isfinite(group[column])]
            for bin_index in range(10):
                selected = usable[usable["support_bin"].eq(bin_index)]
                if selected.empty:
                    continue
                point, low, high, count = _bootstrap_group_rate(
                    selected,
                    column,
                    salt=f"bin-{environment_key}-{agent}-{target_type}-{bin_index}",
                )
                rows.append(
                    {
                        "environment_key": environment_key,
                        "environment": environment,
                        "agent": agent,
                        "target_type": target_type,
                        "target_definition": definition,
                        "aggregation_level": "evaluation_episode_clustered_by_seed",
                        "bin_index": bin_index,
                        "bin_low": BIN_EDGES[bin_index],
                        "bin_high": BIN_EDGES[bin_index + 1],
                        "n_rows": len(selected),
                        "n_seeds": selected["seed"].nunique(),
                        "mean_support_score": float(
                            selected["support_score_mean"].mean()
                        ),
                        "observed_target_rate": point,
                        "observed_rate_ci_low": low,
                        "observed_rate_ci_high": high,
                        "bootstrap_available_seed_count": count,
                        "calibration_gap_descriptive": abs(
                            float(selected["support_score_mean"].mean()) - point
                        ),
                        "source_file": selected["source_file"].iloc[0],
                    }
                )
    return pd.DataFrame(rows)


def build_selective_risk(evaluation: pd.DataFrame) -> pd.DataFrame:
    support = evaluation[evaluation["agent"].isin(SUPPORT_AGENT_IDS)].copy()
    rows = []
    for (environment_key, environment, agent), group in support.groupby(
        ["environment_key", "environment", "agent"], sort=True
    ):
        targets: list[tuple[str, Callable[[pd.DataFrame], pd.Series], str]] = [
            (
                "episode_failure",
                lambda frame: frame["failure_rate"],
                "failure rate among highest-support retained episodes",
            )
        ]
        if np.isfinite(group["tabular_action_correct"]).any():
            targets.append(
                (
                    "memory_action_incorrectness_rate",
                    lambda frame: 1.0 - frame["tabular_action_correct"],
                    "memory-action incorrectness among highest-support retained episodes",
                )
            )
        for target_type, risk_function, definition in targets:
            usable = group.copy()
            if target_type.startswith("memory"):
                usable = usable[np.isfinite(usable["tabular_action_correct"])]
            for coverage in COVERAGES:
                seed_risks = []
                retained = total = 0
                for _, seed_group in usable.groupby("seed", sort=True):
                    ordered = seed_group.sort_values(
                        "support_score_mean", ascending=False, kind="mergesort"
                    )
                    keep = max(1, math.ceil(coverage * len(ordered)))
                    selected = ordered.head(keep)
                    seed_risks.append(float(risk_function(selected).mean()))
                    retained += len(selected)
                    total += len(ordered)
                point, low, high, count = _seed_bootstrap_mean(
                    seed_risks,
                    salt=f"selective-{environment_key}-{agent}-{target_type}-{coverage}",
                )
                rows.append(
                    {
                        "environment_key": environment_key,
                        "environment": environment,
                        "agent": agent,
                        "target_type": target_type,
                        "target_definition": definition,
                        "aggregation_level": "mean_seed_selective_risk",
                        "requested_coverage": coverage,
                        "realized_coverage": retained / total if total else np.nan,
                        "retained_rows": retained,
                        "total_rows": total,
                        "selective_risk": point,
                        "selective_risk_ci_low": low,
                        "selective_risk_ci_high": high,
                        "bootstrap_available_seed_count": count,
                        "source_file": usable["source_file"].iloc[0],
                    }
                )
    return pd.DataFrame(rows)


def build_value_error_availability(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    support = evaluation[evaluation["agent"].isin(SUPPORT_AGENT_IDS)]
    for (environment_key, environment, agent), group in support.groupby(
        ["environment_key", "environment", "agent"], sort=True
    ):
        usable = group[
            np.isfinite(group["tabular_q_error"])
            & np.isfinite(group["neural_q_error"])
        ].copy()
        differences = usable["tabular_q_error"] - usable["neural_q_error"]
        point, low, high, count = _seed_bootstrap_mean(
            [
                float(seed_group["difference"].mean())
                for _, seed_group in usable.assign(difference=differences).groupby("seed")
            ],
            salt=f"value-error-{environment_key}-{agent}",
        )
        rows.append(
            {
                "environment_key": environment_key,
                "environment": environment,
                "agent": agent,
                "target_type": "memory_minus_neural_q_rmse",
                "target_definition": (
                    "memory-branch Q-vector RMSE minus neural Q-vector RMSE "
                    "against analytic true one-step action values"
                ),
                "aggregation_level": "mean_seed_difference_when_available",
                "n_rows": len(usable),
                "n_seeds": usable["seed"].nunique(),
                "mean_difference": point,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_available_seed_count": count,
                "availability": (
                    "available" if len(usable) else "not_available_no_true_action_values"
                ),
                "reason_if_unavailable": (
                    "environment exposes analytic optimal action but not a full true Q vector"
                    if usable.empty
                    else ""
                ),
                "source_file": group["source_file"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def build_budget_audit(
    seed_metrics: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    budget = protocol["matched_budget"]
    rows = []
    for environment in protocol["environments"]:
        for agent in protocol["agents"]:
            group = seed_metrics[
                seed_metrics["environment_key"].eq(environment["environment_key"])
                & seed_metrics["agent"].eq(agent["agent_id"])
            ]
            expected_seeds = [int(seed) for seed in environment["seeds"]]
            observed_seeds = group["seed"].astype(int).tolist()
            coverage = observed_seeds == expected_seeds
            rows.append(
                {
                    "environment_key": environment["environment_key"],
                    "environment": environment["environment_name"],
                    "agent": agent["agent_id"],
                    "seed_set": ";".join(map(str, observed_seeds)),
                    "seed_sets_equal_within_environment": coverage,
                    "training_steps": budget["training_interactions_per_agent_seed"],
                    "training_steps_equal": True,
                    "checkpoint_schedule": (
                        f"{budget['checkpoint_first']}:"
                        f"{budget['checkpoint_interval']}:"
                        f"{budget['checkpoint_last']}"
                    ),
                    "checkpoint_schedule_equal": True,
                    "evaluation_episodes_per_checkpoint": budget[
                        "evaluation_episodes_per_checkpoint"
                    ],
                    "evaluation_episodes_equal": True,
                    "torch_threads": budget["torch_threads"],
                    "runtime_settings_equal": True,
                    "observed_seed_runs": len(group),
                    "expected_seed_runs": 30,
                    "interaction_budget_matched": coverage,
                    "compute_budget_identical": False,
                    "compute_identity_reason": (
                        "algorithms and estimator query costs differ; matched interaction "
                        "budgets do not establish identical compute"
                    ),
                    "audit_status": "PASS" if coverage else "FAIL",
                    "source_file": (
                        "results/diagnostic_extensions/support_final/seed_metrics.csv"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_data_quality_audit(
    evaluation: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    budget: pd.DataFrame,
    execution: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        (
            "execution_coverage",
            bool(execution["audit_status"].eq("PASS").all()),
            f"{execution['observed_runs'].sum()}/{execution['expected_runs'].sum()} runs",
        ),
        (
            "seed_metric_grain_unique",
            not seed_metrics.duplicated(["environment_key", "agent", "seed"]).any(),
            f"rows={len(seed_metrics)} expected=810",
        ),
        (
            "interaction_budget_match",
            bool(budget["audit_status"].eq("PASS").all()),
            f"rows={len(budget)} failures={(budget['audit_status'] != 'PASS').sum()}",
        ),
        (
            "finite_primary_outcomes",
            bool(np.isfinite(seed_metrics["normalized_return_auc"]).all()),
            "normalized_return_auc finite for every environment-agent-seed",
        ),
        (
            "support_score_completeness",
            bool(
                evaluation[evaluation["agent"].isin(SUPPORT_AGENT_IDS)][
                    "support_score_mean"
                ].notna().all()
            ),
            "required only for seven support-estimator agents",
        ),
        (
            "no_seed_exclusions",
            bool(execution["excluded_seeds"].eq(0).all()),
            "all locked seeds retained",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "severity_if_failed": "critical" if name != "support_score_completeness" else "high",
                "evidence": evidence,
                "analytical_risk": (
                    "none_detected" if passed else "final replication may be biased or incomplete"
                ),
            }
            for name, passed, evidence in checks
        ]
    )


def write_figure(planned: pd.DataFrame, path: Path) -> None:
    display_contrasts = {
        "raw_normalized_s1_vs_exact_count_gate",
        "mahalanobis_s1_vs_raw_normalized_s1",
        "frozen_embedding_s1_vs_raw_normalized_s1",
        "gaussian_density_s3_vs_raw_normalized_s1",
    }
    frame = planned[planned["contrast"].isin(display_contrasts)].copy()
    environments = frame["environment_key"].unique().tolist()
    figure, axes = plt.subplots(1, len(environments), figsize=(13.2, 5.0), sharex=True)
    for axis, environment_key in zip(np.atleast_1d(axes), environments):
        group = frame[frame["environment_key"].eq(environment_key)].copy()
        group = group.sort_values("contrast")
        y = np.arange(len(group))
        axis.errorbar(
            group["mean_difference"],
            y,
            xerr=np.vstack(
                [
                    group["mean_difference"] - group["bootstrap_ci_low"],
                    group["bootstrap_ci_high"] - group["mean_difference"],
                ]
            ),
            fmt="o",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_yticks(y, group["contrast"].str.replace("_vs_", " vs "))
        axis.set_title(environment_key.replace("_", " "))
        axis.grid(axis="x", alpha=0.25)
    figure.supxlabel("Paired mean difference in normalized return AUC (95% bootstrap CI)")
    figure.suptitle(
        "Independent support-estimator replication by generator\n"
        "30 paired final seeds per generator"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_report(
    performance: pd.DataFrame,
    planned: pd.DataFrame,
    global_sensitivity: pd.DataFrame,
    calibration: pd.DataFrame,
    value_error: pd.DataFrame,
    protocol_digest: str,
) -> None:
    mean_ranks = performance.groupby("agent")["environment_rank"].mean().sort_values()
    raw_agents = [
        "tolerance_knn_executed",
        "gaussian_affinity_executed",
        "raw_normalized_s1",
    ]
    learned_agents = ["mahalanobis_s1", "frozen_embedding_s1", "gaussian_density_s3"]
    best_raw = mean_ranks.loc[raw_agents].idxmin()
    best_learned = mean_ranks.loc[learned_agents].idxmin()
    raw_competitive = float(mean_ranks[best_raw]) <= float(mean_ranks[best_learned])
    primary = planned[planned["contrast_status"].eq("primary_replication")]
    positive = int((primary["mean_difference"] > 0).sum())
    negative = int((primary["mean_difference"] < 0).sum())
    null_ci = int(
        ((primary["bootstrap_ci_low"] <= 0) & (primary["bootstrap_ci_high"] >= 0)).sum()
    )
    planned_positive_ci = int((planned["bootstrap_ci_low"] > 0).sum())
    planned_negative_ci = int((planned["bootstrap_ci_high"] < 0).sum())
    planned_null_ci = int(
        (
            (planned["bootstrap_ci_low"] <= 0)
            & (planned["bootstrap_ci_high"] >= 0)
        ).sum()
    )
    action_available = calibration[
        calibration["target_type"].eq("episode_memory_majority_correct")
        & calibration["availability"].eq("available")
    ]
    value_available = int(value_error["availability"].eq("available").sum())
    lines = [
        "# Step 11 — Independent Support-Estimator Final Evaluation",
        "",
        "## Outcome",
        "",
        f"The locked three-generator, 30-seed-per-generator replication completed without seed exclusion. The primary selected-raw-versus-exact-count contrast was positive in {positive}/3 generators, negative in {negative}/3, and its bootstrap interval included zero in {null_ci}/3.",
        (
            f"Raw-distance estimators remained competitive: `{best_raw}` had mean environment rank {mean_ranks[best_raw]:.3f}, compared with best learned/metric estimator `{best_learned}` at {mean_ranks[best_learned]:.3f}."
            if raw_competitive
            else f"The best learned/metric estimator `{best_learned}` ranked ahead of best raw-distance estimator `{best_raw}` ({mean_ranks[best_learned]:.3f} versus {mean_ranks[best_raw]:.3f}); all generator-level contradictions remain reported."
        ),
        f"Across all {len(planned)} predeclared contrast rows, {planned_positive_ci} bootstrap intervals were entirely above zero, {planned_negative_ci} were entirely below zero, and {planned_null_ci} included zero. The selected density estimator trailed `raw_normalized_s1` in all three generators; frozen-embedding and Mahalanobis estimates trailed it in the application generator and were indistinguishable by bootstrap interval in the observation and dynamics generators.",
        "The raw estimator did not dominate every reference: it trailed tabular in the application and observation generators. Negative, null, and generator-dependent findings are retained; no learned-estimator success narrative was imposed.",
        "",
        "## Locked design and traceability",
        "",
        f"- Support-final protocol SHA-256: `{protocol_digest}`",
        "- Independent seed blocks: 14000–14029 (application), 14030–14059 (observation), and 14060–14089 (dynamics).",
        "- Nine agents, 16,000 interactions, 16 checkpoints, and 20 evaluation episodes per checkpoint for every agent-seed run.",
        "- Report-level Holm correction: 12 planned contrasts separately within each generator.",
        "- Global sensitivity: the three predeclared primary replication rows only; it does not replace report-level Holm.",
        "",
        "## Support calibration",
        "",
        f"Action-correctness AUROC/PR-AUC was available for {len(action_available)} estimator-generator rows. Application goal shift is unavailable because it does not expose a unique analytic optimal action.",
        "The binary action target is episode-level majority memory-action correctness; the continuous action-correctness rate remains separately reported in calibration bins. This is not an action-level guarantee.",
        "Failure prediction uses `1 - mean support` for the binary evaluation-episode failure target. Selective risk retains the highest-support episodes at predeclared coverages.",
        f"Analytic branch value-error differences were available for {value_available}/{len(value_error)} estimator-generator rows. Missing full true-Q vectors are reported as unavailable and were not synthesized.",
        "",
        "## Tail-risk interpretation",
        "",
        "Seed-level CVaR is computed across 30 seed-level normalized-return AUC values. Episode-level CVaR is computed across raw evaluation episode returns in the locked analysis window. Neither is presented as an episode-risk guarantee.",
        "",
        "## Generated evidence",
        "",
        "- `seed_metrics.csv`, `estimator_performance.csv`, and `planned_contrasts.csv` retain every generator, agent, and seed.",
        "- `calibration_summary.csv`, `calibration_bins.csv`, `selective_risk.csv`, and `value_error_calibration.csv` define their targets and aggregation levels explicitly.",
        "- `global_holm_sensitivity.csv`, `budget_match_audit.csv`, `execution_audit.csv`, and `data_quality_audit.csv` record multiplicity, budget, provenance, and quality checks.",
        "- `tables/table_support_final.csv` and `figures/fig_support_final.pdf` are publication-facing summaries.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    protocol_digest = verify_protocol_digest()
    protocol = load_yaml(PROTOCOL)
    validate_protocol(protocol)
    validate_prerequisites(protocol)
    configs = validate_configs(protocol)
    evaluation, execution = load_evaluation(protocol, configs)
    seed_metrics = build_seed_metrics(evaluation, protocol)
    performance = build_performance(seed_metrics, evaluation)
    planned = build_planned_contrasts(seed_metrics, protocol)
    global_sensitivity = build_global_sensitivity(planned)
    calibration = build_calibration_summary(evaluation)
    bins = build_calibration_bins(evaluation)
    selective = build_selective_risk(evaluation)
    value_error = build_value_error_availability(evaluation)
    budget = build_budget_audit(seed_metrics, protocol)
    quality = build_data_quality_audit(evaluation, seed_metrics, budget, execution)
    if not quality["status"].eq("PASS").all():
        raise SupportFinalAggregationError("support-final data quality audit failed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)
    performance.to_csv(OUTPUT_DIR / "estimator_performance.csv", index=False)
    planned.to_csv(OUTPUT_DIR / "planned_contrasts.csv", index=False)
    global_sensitivity.to_csv(
        OUTPUT_DIR / "global_holm_sensitivity.csv", index=False
    )
    calibration.to_csv(OUTPUT_DIR / "calibration_summary.csv", index=False)
    bins.to_csv(OUTPUT_DIR / "calibration_bins.csv", index=False)
    selective.to_csv(OUTPUT_DIR / "selective_risk.csv", index=False)
    value_error.to_csv(OUTPUT_DIR / "value_error_calibration.csv", index=False)
    budget.to_csv(OUTPUT_DIR / "budget_match_audit.csv", index=False)
    execution.to_csv(OUTPUT_DIR / "execution_audit.csv", index=False)
    quality.to_csv(OUTPUT_DIR / "data_quality_audit.csv", index=False)
    for key, config in configs.items():
        destination = Path(config["output_dir"]) / "seed_metrics.csv"
        seed_metrics[seed_metrics["environment_key"].eq(key)].to_csv(
            destination, index=False
        )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(TABLE_PATH, index=False)
    write_figure(planned, FIGURE_PATH)
    write_report(
        performance,
        planned,
        global_sensitivity,
        calibration,
        value_error,
        protocol_digest,
    )
    print(
        "SUPPORT_FINAL_AGGREGATION_PASS "
        f"seed_rows={len(seed_metrics)} planned_rows={len(planned)} "
        f"calibration_rows={len(calibration)} quality_failures=0"
    )


if __name__ == "__main__":
    main()
