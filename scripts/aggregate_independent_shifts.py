from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

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
from scripts.run_independent_shift_final import (  # noqa: E402
    CONFIGS,
    EXPECTED,
    EXPECTED_AGENTS,
    audit_execution,
    load_yaml,
    validate_final_config,
    validate_prerequisites,
)
from scripts.lock_protocol import DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY, load_and_validate  # noqa: E402


OUTPUT_DIR = ROOT / "results/diagnostic_extensions/final_shifts"
TABLE_PATH = ROOT / "tables/table_independent_shift_replication.csv"
FIGURE_PATH = ROOT / "figures/fig_independent_shift_replication.pdf"
POST_SHIFT_CHECKPOINTS = list(range(12000, 24001, 1000))
PRIMARY_AGENTS = {
    "relative_reliability_fuzzy",
    "count_gated_tau_20",
    "same_input_crisp",
}
ENVIRONMENT_NAMES = {
    "transition_dynamics_shift": "final-transition-dynamics-shift",
    "observation_shift": "final-observation-shift",
    "localized_multistep_reward_or_policy_shift": (
        "final-localized-multistep-reward-or-policy-shift"
    ),
}
RAW_COLUMNS = [
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
    "predicted_better_branch",
    "actually_better_branch",
]


class IndependentShiftAggregationError(ValueError):
    pass


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalized_auc(checkpoints: object, values: object) -> float:
    x = np.asarray(checkpoints, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise IndependentShiftAggregationError("AUC inputs must be equal vectors")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    expected = np.asarray(POST_SHIFT_CHECKPOINTS, dtype=float)
    if not np.array_equal(x, expected) or not np.isfinite(y).all():
        raise IndependentShiftAggregationError("post-shift AUC coverage mismatch")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _numeric_sum_count(frame: pd.DataFrame, column: str) -> tuple[float, int]:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.sum()), int(finite.size)


def aggregate_raw_file(
    path: Path, mechanism: str
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(RAW_COLUMNS) - header
    if missing:
        raise IndependentShiftAggregationError(
            f"{path} is missing columns {sorted(missing)}"
        )
    partial_rows: list[dict[str, Any]] = []
    episode_values: dict[str, list[np.ndarray]] = defaultdict(list)
    source = relative_path(path)
    for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, chunksize=200_000):
        chunk = chunk[chunk["phase"].eq("eval")].copy()
        chunk["checkpoint"] = pd.to_numeric(chunk["checkpoint"], errors="coerce")
        chunk = chunk[chunk["checkpoint"].isin(POST_SHIFT_CHECKPOINTS)]
        if chunk.empty:
            continue
        chunk["seed"] = pd.to_numeric(chunk["seed"], errors="raise").astype(int)
        chunk["return"] = pd.to_numeric(chunk["return"], errors="coerce")
        for agent, values in chunk.groupby("agent", sort=False)["return"]:
            finite = values.to_numpy(dtype=float)
            episode_values[str(agent)].append(finite[np.isfinite(finite)])
        actual = chunk["actually_better_branch"].astype(str)
        predicted = chunk["predicted_better_branch"].astype(str)
        chunk["branch_eligible"] = actual.isin(["tabular", "neural"]).astype(int)
        chunk["branch_correct"] = (
            chunk["branch_eligible"].eq(1) & predicted.eq(actual)
        ).astype(int)
        for (agent, seed, checkpoint), group in chunk.groupby(
            ["agent", "seed", "checkpoint"], sort=False
        ):
            row: dict[str, Any] = {
                "mechanism_id": mechanism,
                "agent": str(agent),
                "seed": int(seed),
                "checkpoint": int(checkpoint),
                "source_file": source,
            }
            for metric, column in (
                ("return", "return"),
                ("success", "success"),
                ("failure", "failure_rate"),
                ("collision", "collision_rate"),
                ("risk_zone", "risk_zone_rate"),
            ):
                total, count = _numeric_sum_count(group, column)
                row[f"{metric}_sum"] = total
                row[f"{metric}_count"] = count
            row["branch_eligible"] = int(group["branch_eligible"].sum())
            row["branch_correct"] = int(group["branch_correct"].sum())
            partial_rows.append(row)
    if not partial_rows:
        raise IndependentShiftAggregationError(f"no post-shift evaluation rows: {path}")
    partial = pd.DataFrame(partial_rows)
    keys = ["mechanism_id", "agent", "seed", "checkpoint", "source_file"]
    sums = [column for column in partial if column.endswith("_sum") or column.endswith("_count")]
    sums += ["branch_eligible", "branch_correct"]
    checkpoint = partial.groupby(keys, as_index=False, sort=True)[sums].sum()
    for metric in ("return", "success", "failure", "collision", "risk_zone"):
        count = checkpoint[f"{metric}_count"]
        checkpoint[f"mean_{metric}"] = checkpoint[f"{metric}_sum"] / count.where(count > 0)
    checkpoint["branch_correctness_rate"] = (
        checkpoint["branch_correct"]
        / checkpoint["branch_eligible"].where(checkpoint["branch_eligible"] > 0)
    )
    episode_arrays = {
        agent: np.concatenate(parts) if parts else np.asarray([], dtype=float)
        for agent, parts in episode_values.items()
    }
    return checkpoint, episode_arrays


def build_seed_metrics(checkpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mechanism, agent, seed), group in checkpoint.groupby(
        ["mechanism_id", "agent", "seed"], sort=True
    ):
        group = group.sort_values("checkpoint")
        if len(group) != len(POST_SHIFT_CHECKPOINTS):
            raise IndependentShiftAggregationError(
                f"checkpoint count mismatch for {(mechanism, agent, seed)}"
            )
        rows.append(
            {
                "mechanism_id": mechanism,
                "environment": ENVIRONMENT_NAMES[mechanism],
                "agent": agent,
                "seed": int(seed),
                "normalized_return_auc": normalized_auc(group["checkpoint"], group["mean_return"]),
                "success_auc": normalized_auc(group["checkpoint"], group["mean_success"]),
                "failure_probability": float(group["failure_sum"].sum() / group["failure_count"].sum()),
                "collision_rate": float(group["collision_sum"].sum() / group["collision_count"].sum()),
                "risk_zone_rate": float(group["risk_zone_sum"].sum() / group["risk_zone_count"].sum()),
                "branch_correctness": (
                    float(group["branch_correct"].sum() / group["branch_eligible"].sum())
                    if group["branch_eligible"].sum() > 0
                    else np.nan
                ),
                "branch_correctness_denominator": int(group["branch_eligible"].sum()),
                "source_file": str(group["source_file"].iloc[0]),
                "source_column": "return;success;failure_rate;collision_rate;risk_zone_rate;predicted_better_branch;actually_better_branch",
            }
        )
    result = pd.DataFrame(rows).sort_values(["mechanism_id", "agent", "seed"])
    for mechanism, expected in EXPECTED.items():
        part = result[result["mechanism_id"] == mechanism]
        if set(part["agent"]) != EXPECTED_AGENTS:
            raise IndependentShiftAggregationError(f"agent coverage mismatch: {mechanism}")
        for agent, group in part.groupby("agent"):
            if group["seed"].tolist() != expected["seeds"]:
                raise IndependentShiftAggregationError(
                    f"seed coverage mismatch: {(mechanism, agent)}"
                )
    return result.reset_index(drop=True)


def _paired_test_row(
    mechanism: str,
    contrast: str,
    status: str,
    left: str,
    right: str,
    seed_metrics: pd.DataFrame,
) -> dict[str, Any]:
    part = seed_metrics[seed_metrics["mechanism_id"] == mechanism]
    left_values = part[part["agent"] == left][["seed", "normalized_return_auc"]]
    right_values = part[part["agent"] == right][["seed", "normalized_return_auc"]]
    paired = left_values.merge(
        right_values, on="seed", suffixes=("_left", "_right"), validate="one_to_one"
    ).sort_values("seed")
    if len(paired) != 30 or not np.isfinite(paired.filter(like="normalized_return_auc")).all().all():
        raise IndependentShiftAggregationError(f"paired coverage mismatch: {mechanism}/{contrast}")
    differences = (
        paired["normalized_return_auc_left"] - paired["normalized_return_auc_right"]
    ).to_numpy(dtype=float)
    std = float(differences.std(ddof=1))
    if np.allclose(differences, 0.0):
        paired_t_p = wilcoxon_p = 1.0
    elif std <= 1e-12:
        paired_t_p = 0.0
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
    else:
        paired_t_p = float(stats.ttest_1samp(differences, 0.0).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    low, high = bootstrap_mean_interval(differences, seed=0, samples=10_000)
    wins, losses, ties = win_loss_tie(differences)
    return {
        "report_family": "independent_shift_primary",
        "evidence_class": "replication",
        "environment": ENVIRONMENT_NAMES[mechanism],
        "mechanism_id": mechanism,
        "contrast": contrast,
        "contrast_status": status,
        "left": left,
        "right": right,
        "metric": "normalized_return_auc",
        "n_pairs": 30,
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
        "source_file": "results/diagnostic_extensions/final_shifts/seed_metrics.csv",
        "source_column": "normalized_return_auc",
    }


def build_planned_contrasts(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mechanism in EXPECTED:
        rows.append(
            _paired_test_row(
                mechanism,
                "relative_reliability_fuzzy_vs_count_gated_tau_20",
                "primary",
                "relative_reliability_fuzzy",
                "count_gated_tau_20",
                seed_metrics,
            )
        )
        rows.append(
            _paired_test_row(
                mechanism,
                "same_input_crisp_vs_relative_reliability_fuzzy",
                "co_primary_mechanism_contrast",
                "same_input_crisp",
                "relative_reliability_fuzzy",
                seed_metrics,
            )
        )
    if len(rows) != 6:
        raise IndependentShiftAggregationError("locked Holm family must contain six rows")
    paired_adjusted = holm_adjust([row["paired_t_p"] for row in rows])
    wilcoxon_adjusted = holm_adjust([row["wilcoxon_p"] for row in rows])
    for row, t_value, w_value in zip(rows, paired_adjusted, wilcoxon_adjusted):
        row["paired_t_holm_p"] = t_value
        row["wilcoxon_holm_p"] = w_value
        row["report_holm_scope"] = "independent_shift_primary_six_rows"
    return pd.DataFrame(rows)


def _tail_row(
    mechanism: str,
    agent: str,
    level: str,
    values: np.ndarray,
    source_file: str,
    source_column: str,
) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise IndependentShiftAggregationError("tail metric has no finite values")
    alpha05_count = max(1, int(np.ceil(0.05 * finite.size)))
    alpha05_valid = alpha05_count >= 2
    return {
        "mechanism_id": mechanism,
        "agent": agent,
        "aggregation_level": level,
        "n_finite": int(finite.size),
        "minimum_return": float(finite.min()),
        "return_quantile_0_05": lower_quantile(finite, 0.05),
        "worst_decile_mean": worst_decile_mean(finite),
        "cvar_0_10": empirical_lower_cvar(finite, 0.10),
        "cvar_0_10_tail_count": max(1, int(np.ceil(0.10 * finite.size))),
        "cvar_0_05": empirical_lower_cvar(finite, 0.05) if alpha05_valid else np.nan,
        "cvar_0_05_tail_count": alpha05_count,
        "cvar_0_05_available": alpha05_valid,
        "source_file": source_file,
        "source_column": source_column,
    }


def build_lower_tail(
    seed_metrics: pd.DataFrame,
    episode_values: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mechanism, agent), group in seed_metrics.groupby(["mechanism_id", "agent"], sort=True):
        rows.append(
            _tail_row(
                mechanism,
                agent,
                "seed_normalized_return_auc",
                group["normalized_return_auc"].to_numpy(dtype=float),
                "results/diagnostic_extensions/final_shifts/seed_metrics.csv",
                "normalized_return_auc",
            )
        )
        episode = episode_values[(mechanism, agent)]
        source = str(group["source_file"].iloc[0])
        rows.append(
            _tail_row(
                mechanism,
                agent,
                "post_shift_evaluation_episode_return",
                episode,
                source,
                "return",
            )
        )
    return pd.DataFrame(rows)


def detection_delay_rows(checkpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mechanism, agent, seed), group in checkpoint.groupby(
        ["mechanism_id", "agent", "seed"], sort=True
    ):
        group = group.sort_values("checkpoint")
        eligible = group["branch_eligible"].to_numpy(dtype=int)
        rates = group["branch_correctness_rate"].to_numpy(dtype=float)
        qualifies = (eligible >= 20) & np.isfinite(rates) & (rates > 0.50)
        detection_checkpoint: int | None = None
        for index in range(len(group) - 1):
            consecutive = int(group.iloc[index + 1]["checkpoint"]) - int(
                group.iloc[index]["checkpoint"]
            ) == 1000
            if qualifies[index] and qualifies[index + 1] and consecutive:
                detection_checkpoint = int(group.iloc[index]["checkpoint"])
                break
        available = bool((eligible >= 20).any())
        detected = detection_checkpoint is not None
        reason = ""
        if not available and agent in {"tabular", "dqn"}:
            reason = "not applicable to a single-branch descriptive comparator"
        elif not available and mechanism == "localized_multistep_reward_or_policy_shift":
            reason = "analytic optimal action is unavailable for the localized multi-step environment"
        elif not available:
            reason = (
                "final evaluation rows contain no eligible branch target because "
                "this agent path did not log branch diagnostics"
            )
        elif not detected:
            reason = "no two consecutive checkpoints exceeded branch correctness 0.50"
        rows.append(
            {
                "mechanism_id": mechanism,
                "agent": agent,
                "seed": int(seed),
                "available": available,
                "detected": detected,
                "right_censored": bool(available and not detected),
                "detection_checkpoint": detection_checkpoint if detected else np.nan,
                "detection_delay_interactions": detection_checkpoint - 12000 if detected else np.nan,
                "censoring_time_interactions": 12000 if available and not detected else np.nan,
                "criterion": "first of two consecutive checkpoints with >=20 eligible episodes and correctness>0.50",
                "reason": reason,
                "source_file": str(group["source_file"].iloc[0]),
                "source_column": "predicted_better_branch;actually_better_branch",
            }
        )
    return pd.DataFrame(rows)


def build_descriptive_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "normalized_return_auc",
        "success_auc",
        "failure_probability",
        "collision_rate",
        "risk_zone_rate",
        "branch_correctness",
    ]
    for (mechanism, agent), group in seed_metrics.groupby(["mechanism_id", "agent"], sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                low = high = mean = np.nan
                available = False
                if metric == "branch_correctness" and agent in {"tabular", "dqn"}:
                    reason = "not applicable to a single-branch descriptive comparator"
                elif (
                    metric == "branch_correctness"
                    and mechanism == "localized_multistep_reward_or_policy_shift"
                ):
                    reason = "analytic optimal action is unavailable for the localized multi-step environment"
                elif metric == "branch_correctness":
                    reason = "final evaluation rows did not log eligible branch diagnostics for this agent path"
                else:
                    reason = "no finite source values"
            else:
                mean = float(values.mean())
                low, high = bootstrap_mean_interval(values, seed=0, samples=10_000)
                available = True
                reason = ""
            rows.append(
                {
                    "mechanism_id": mechanism,
                    "agent": agent,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "available": available,
                    "reason_if_unavailable": reason,
                    "inferential_status": (
                        "locked_primary_agents_descriptive_summary"
                        if agent in PRIMARY_AGENTS
                        else "descriptive_comparator_only"
                    ),
                    "source_file": "results/diagnostic_extensions/final_shifts/seed_metrics.csv",
                    "source_column": metric,
                }
            )
    return pd.DataFrame(rows)


def write_forest_figure(contrasts: pd.DataFrame, path: Path) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    mechanism_labels = {
        "transition_dynamics_shift": "Transition dynamics",
        "observation_shift": "Observation",
        "localized_multistep_reward_or_policy_shift": "Localized reward",
    }
    contrast_labels = {
        "relative_reliability_fuzzy_vs_count_gated_tau_20": "Fuzzy - count gate",
        "same_input_crisp_vs_relative_reliability_fuzzy": "Crisp - fuzzy",
    }
    labels = [
        f"{mechanism_labels[row.mechanism_id]} | {contrast_labels[row.contrast]}"
        for row in contrasts.itertuples(index=False)
    ]
    y = np.arange(len(contrasts))
    mean = contrasts["mean_difference"].to_numpy(dtype=float)
    low = contrasts["bootstrap_ci_low"].to_numpy(dtype=float)
    high = contrasts["bootstrap_ci_high"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    for index, row in enumerate(contrasts.itertuples(index=False)):
        primary = row.contrast_status == "primary"
        color = "#2563a6" if primary else "#d97706"
        marker = "o" if primary else "s"
        axis.errorbar(
            mean[index],
            y[index],
            xerr=np.asarray(
                [[mean[index] - low[index]], [high[index] - mean[index]]]
            ),
            fmt=marker,
            color=color,
            markerfacecolor=color if primary else "white",
            markeredgecolor=color,
            markersize=6,
            capsize=4,
            linewidth=1.6,
            zorder=3,
        )
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired difference in normalized return AUC (left - right)")
    figure.text(
        0.36,
        0.955,
        "Independent locked shift replications",
        fontsize=14,
        color="#111827",
        va="top",
    )
    figure.text(
        0.36,
        0.915,
        "Normalized return AUC; n=30 paired seeds per row; 95% seed-bootstrap CI; generators not pooled",
        fontsize=9,
        color="#4b5563",
        va="top",
    )
    axis.grid(axis="x", alpha=0.25)
    figure.subplots_adjust(left=0.36, right=0.97, top=0.84, bottom=0.14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def aggregate(output_dir: Path = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    validate_prerequisites(protocol)
    checkpoint_parts: list[pd.DataFrame] = []
    episode_values: dict[tuple[str, str], np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    for mechanism, config_path in CONFIGS.items():
        config = load_yaml(config_path)
        validate_final_config(config, mechanism)
        raw = ROOT / config["output_dir"] / "raw.csv"
        compressed = raw.with_suffix(".csv.gz")
        if not raw.exists() or not compressed.exists():
            raise IndependentShiftAggregationError(f"missing complete raw outputs: {mechanism}")
        audit_execution(raw, config, mechanism)
        checkpoint, episodes = aggregate_raw_file(compressed, mechanism)
        checkpoint_parts.append(checkpoint)
        for agent, values in episodes.items():
            episode_values[(mechanism, agent)] = values
        audit_rows.append(
            {
                "mechanism_id": mechanism,
                "severity_id": EXPECTED[mechanism]["severity_id"],
                "expected_seeds": 30,
                "completed_seeds_per_agent": 30,
                "agents": 5,
                "completed_agent_seed_runs": 150,
                "expected_checkpoints_per_run": 24,
                "evaluation_episodes_per_checkpoint": 200,
                "audit_status": "PASS",
                "source_file": relative_path(compressed),
            }
        )
    checkpoint = pd.concat(checkpoint_parts, ignore_index=True).sort_values(
        ["mechanism_id", "agent", "seed", "checkpoint"]
    )
    seed_metrics = build_seed_metrics(checkpoint)
    contrasts = build_planned_contrasts(seed_metrics)
    tails = build_lower_tail(seed_metrics, episode_values)
    delays = detection_delay_rows(checkpoint)
    descriptive = build_descriptive_summary(seed_metrics)
    audit = pd.DataFrame(audit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    contrasts.to_csv(output_dir / "planned_contrasts.csv", index=False)
    tails.to_csv(output_dir / "lower_tail_metrics.csv", index=False)
    delays.to_csv(output_dir / "detection_delay.csv", index=False)
    descriptive.to_csv(output_dir / "descriptive_agent_summary.csv", index=False)
    audit.to_csv(output_dir / "execution_audit.csv", index=False)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(TABLE_PATH, index=False)
    write_forest_figure(contrasts, FIGURE_PATH)
    return {
        "checkpoint_metrics": checkpoint,
        "seed_metrics": seed_metrics,
        "planned_contrasts": contrasts,
        "lower_tail_metrics": tails,
        "detection_delay": delays,
        "descriptive_agent_summary": descriptive,
        "execution_audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the locked independent shift replications.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    outputs = aggregate(args.output_dir)
    print(
        "INDEPENDENT_SHIFT_AGGREGATION_PASS "
        f"seed_rows={len(outputs['seed_metrics'])} contrast_rows={len(outputs['planned_contrasts'])}"
    )


if __name__ == "__main__":
    main()
