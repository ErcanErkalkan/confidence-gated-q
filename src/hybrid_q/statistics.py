from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        value = float(values[0]) if values.size else float("nan")
        return value, value
    mean = float(values.mean())
    sem = stats.sem(values)
    radius = float(stats.t.ppf((1 + confidence) / 2, values.size - 1) * sem)
    return mean - radius, mean + radius


def cohen_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    std = differences.std(ddof=1)
    mean = differences.mean()
    if std > 1e-12:
        return float(mean / std)
    if np.isclose(mean, 0):
        return 0.0
    return float(np.copysign(np.inf, mean))


def paired_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[~np.isclose(differences, 0)]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def bootstrap_mean_interval(
    values: np.ndarray,
    seed: int = 0,
    samples: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite, non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    clean = [value if np.isfinite(value) else 1.0 for value in p_values]
    count = len(clean)
    order = np.argsort(clean)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * clean[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def win_loss_tie(
    differences: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[int, int, int]:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError("differences must be a finite one-dimensional vector")
    wins = int((differences > tolerance).sum())
    losses = int((differences < -tolerance).sum())
    ties = int(differences.size - wins - losses)
    return wins, losses, ties


def paired_differences(
    left_values: pd.DataFrame,
    right_values: pd.DataFrame,
    metric: str,
) -> np.ndarray:
    for label, values in (("left", left_values), ("right", right_values)):
        if values["seed"].duplicated().any():
            raise ValueError(f"{label} paired input contains duplicate seeds")
        if values[metric].isna().any():
            raise ValueError(f"{label} paired input contains missing values")
    left_seeds = set(left_values["seed"])
    right_seeds = set(right_values["seed"])
    if left_seeds != right_seeds:
        raise ValueError("paired inputs must contain identical seed sets")
    if len(left_seeds) < 2:
        raise ValueError("paired tests require at least two matched seeds")
    paired = left_values.merge(
        right_values, on="seed", suffixes=("_left", "_right"), validate="one_to_one"
    ).sort_values("seed")
    return (
        paired[f"{metric}_left"] - paired[f"{metric}_right"]
    ).to_numpy(dtype=float)


def robust_outlier_counts(values: np.ndarray) -> tuple[int, int]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-12:
        scale = 1.4826 * mad
        return (
            int((values < median - 6.0 * scale).sum()),
            int((values > median + 6.0 * scale).sum()),
        )
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr <= 1e-12:
        return 0, 0
    return (
        int((values < q25 - 3.0 * iqr).sum()),
        int((values > q75 + 3.0 * iqr).sum()),
    )


def sample_skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or np.allclose(values, values[0]):
        return 0.0
    return float(stats.skew(values, bias=False))


def sample_excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 4 or np.allclose(values, values[0]):
        return 0.0
    return float(stats.kurtosis(values, bias=False))


def seed_level_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "support_abstention_rate",
        "training_elapsed_seconds",
        "evaluation_elapsed_seconds",
    ):
        if column not in raw:
            raw[column] = np.nan
    evaluation = raw[raw["phase"] == "eval"].copy()
    grouped = (
        evaluation.groupby(["environment", "agent", "seed", "checkpoint"])
        .agg(
            mean_return=("return", "mean"),
            success_rate=("success", "mean"),
            mean_length=("length", "mean"),
            elapsed_seconds=("elapsed_seconds", "max"),
            environment_steps=("environment_steps", "max"),
            gradient_updates=("gradient_updates", "max"),
            training_elapsed_seconds=(
                "training_elapsed_seconds",
                "max",
            ),
            evaluation_elapsed_seconds=(
                "evaluation_elapsed_seconds",
                "max",
            ),
            mean_gate=("mean_gate", "max"),
            support_abstention_rate=(
                "support_abstention_rate",
                "max",
            ),
            global_tabular_error=("global_tabular_error", "max"),
            global_neural_error=("global_neural_error", "max"),
            visited_states=("visited_states", "max"),
        )
        .reset_index()
        .sort_values(["environment", "agent", "seed", "checkpoint"])
    )
    def normalized_auc(group: pd.DataFrame, metric: str) -> float:
        checkpoints = group["checkpoint"].to_numpy(dtype=float)
        values = group[metric].to_numpy(dtype=float)
        width = checkpoints[-1] - checkpoints[0]
        return float(np.trapezoid(values, checkpoints) / width) if width > 0 else float(values[-1])

    return_auc = (
        grouped.groupby(["environment", "agent", "seed"])
        .apply(normalized_auc, "mean_return", include_groups=False)
        .rename("return_auc")
        .reset_index()
    )
    success_auc = (
        grouped.groupby(["environment", "agent", "seed"])
        .apply(normalized_auc, "success_rate", include_groups=False)
        .rename("success_rate_auc")
        .reset_index()
    )
    final = grouped.groupby(["environment", "agent", "seed"]).tail(1).copy()
    final["eval_checkpoint"] = final["checkpoint"]
    final["eval_return"] = final["mean_return"]
    final["train_steps"] = final["environment_steps"]
    final["seconds_per_1000_steps"] = (
        1000.0
        * final["training_elapsed_seconds"]
        / final["environment_steps"]
    )
    result = final.merge(
        return_auc, on=["environment", "agent", "seed"]
    ).merge(success_auc, on=["environment", "agent", "seed"])
    provenance_columns = [
        "experiment_name",
        "config_file",
        "environment_id",
        "resolved_environment_id",
        "observation_representation",
        "git_commit_hash",
        "package_version",
        "python_version",
        "torch_version",
        "numpy_version",
        "gymnasium_version",
        "minigrid_version",
    ]
    available = [column for column in provenance_columns if column in evaluation]
    if available:
        provenance = (
            evaluation.groupby(["environment", "agent", "seed"], dropna=False)[
                available
            ]
            .first()
            .reset_index()
        )
        result = result.merge(
            provenance, on=["environment", "agent", "seed"], how="left"
        )
    return result


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (environment, agent), group in seed_metrics.groupby(
        ["environment", "agent"]
    ):
        for metric in (
            "mean_return",
            "success_rate",
            "return_auc",
            "success_rate_auc",
            "seconds_per_1000_steps",
            "gradient_updates",
            "mean_gate",
            "support_abstention_rate",
            "global_tabular_error",
            "global_neural_error",
            "visited_states",
        ):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            ci_low, ci_high = t_interval(values)
            boot_low, boot_high = bootstrap_mean_interval(values)
            rows.append(
                {
                    "environment": environment,
                    "agent": agent,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "median": np.median(values),
                    "q25": np.quantile(values, 0.25),
                    "q75": np.quantile(values, 0.75),
                    "t_ci_low": ci_low,
                    "t_ci_high": ci_high,
                    "bootstrap_ci_low": boot_low,
                    "bootstrap_ci_high": boot_high,
                }
            )
    return pd.DataFrame(rows)


def _comparison_row(
    environment: str,
    metric: str,
    left: str,
    right: str,
    env_group: pd.DataFrame,
) -> dict | None:
    left_values = env_group[env_group["agent"] == left][["seed", metric]]
    right_values = env_group[env_group["agent"] == right][["seed", metric]]
    if left_values.empty or right_values.empty:
        return None
    differences = paired_differences(left_values, right_values, metric)
    difference_std = differences.std(ddof=1)
    if np.allclose(differences, 0):
        paired_t_p = 1.0
        wilcoxon_p = 1.0
    elif difference_std <= 1e-12:
        paired_t_p = 0.0
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
    else:
        paired_t_p = float(
            stats.ttest_1samp(differences, popmean=0.0).pvalue
        )
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    wins, losses, ties = win_loss_tie(differences)
    sign_test_p = (
        float(stats.binomtest(wins, wins + losses, p=0.5).pvalue)
        if wins + losses
        else 1.0
    )
    ci_low, ci_high = bootstrap_mean_interval(differences)
    catastrophic_low, catastrophic_high = robust_outlier_counts(differences)
    return {
        "environment": environment,
        "metric": metric,
        "left": left,
        "right": right,
        "n_pairs": len(differences),
        "mean_difference": differences.mean(),
        "median_difference": np.median(differences),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohen_dz": cohen_dz(differences),
        "rank_biserial": paired_rank_biserial(differences),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_test_p,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "difference_skew": sample_skew(differences),
        "difference_excess_kurtosis": sample_excess_kurtosis(differences),
        "catastrophic_low_outliers": catastrophic_low,
        "catastrophic_high_outliers": catastrophic_high,
        "win_rate_non_ties": (
            wins / (wins + losses) if wins + losses else 0.5
        ),
    }


def pairwise(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("mean_return", "success_rate", "return_auc")
    for environment, env_group in seed_metrics.groupby("environment"):
        agents = sorted(env_group["agent"].unique())
        for metric in metrics:
            metric_rows = []
            for left, right in itertools.combinations(agents, 2):
                row = _comparison_row(
                    environment, metric, left, right, env_group
                )
                if row is not None:
                    metric_rows.append(row)
            adjusted = holm_adjust([row["paired_t_p"] for row in metric_rows])
            wilcoxon_adjusted = holm_adjust(
                [row["wilcoxon_p"] for row in metric_rows]
            )
            sign_adjusted = holm_adjust(
                [row["sign_test_p"] for row in metric_rows]
            )
            for row, adjusted_p, wilcoxon_p, sign_p in zip(
                metric_rows,
                adjusted,
                wilcoxon_adjusted,
                sign_adjusted,
            ):
                row["paired_t_holm_p"] = adjusted_p
                row["wilcoxon_holm_p"] = wilcoxon_p
                row["sign_test_holm_p"] = sign_p
                rows.append(row)
    return pd.DataFrame(rows)


def planned_contrasts(
    seed_metrics: pd.DataFrame,
    contrasts: list[dict],
) -> pd.DataFrame:
    rows = []
    for contrast in contrasts:
        left = contrast["left"]
        right = contrast["right"]
        metric = contrast.get("metric", "return_auc")
        for environment, env_group in seed_metrics.groupby("environment"):
            row = _comparison_row(
                environment, metric, left, right, env_group
            )
            if row is not None:
                row["contrast"] = contrast.get(
                    "name", f"{left}_vs_{right}_{metric}"
                )
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    adjusted = holm_adjust([row["paired_t_p"] for row in rows])
    wilcoxon_adjusted = holm_adjust(
        [row["wilcoxon_p"] for row in rows]
    )
    sign_adjusted = holm_adjust(
        [row["sign_test_p"] for row in rows]
    )
    for row, adjusted_p, wilcoxon_p, sign_p in zip(
        rows, adjusted, wilcoxon_adjusted, sign_adjusted
    ):
        row["paired_t_holm_p"] = adjusted_p
        row["wilcoxon_holm_p"] = wilcoxon_p
        row["sign_test_holm_p"] = sign_p
    return pd.DataFrame(rows)


def heavy_tail_diagnostics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (environment, agent), group in seed_metrics.groupby(
        ["environment", "agent"]
    ):
        for metric in ("return_auc", "mean_return"):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 2:
                continue
            low_outliers, high_outliers = robust_outlier_counts(values)
            rows.append(
                {
                    "environment": environment,
                    "agent": agent,
                    "metric": metric,
                    "n_seeds": values.size,
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "median": np.median(values),
                    "mad": np.median(np.abs(values - np.median(values))),
                    "skew": sample_skew(values),
                    "excess_kurtosis": sample_excess_kurtosis(values),
                    "catastrophic_low_outliers": low_outliers,
                    "catastrophic_high_outliers": high_outliers,
                    "heavy_tail_flag": bool(
                        low_outliers
                        or high_outliers
                        or (
                            values.size >= 4
                            and abs(sample_excess_kurtosis(values)) > 3
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def cross_environment_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("return_auc", "success_rate_auc", "mean_return"):
        means = (
            seed_metrics.groupby(["environment", "agent"])[metric]
            .mean()
            .reset_index()
        )
        means["environment_rank"] = means.groupby("environment")[metric].rank(
            ascending=False, method="average"
        )
        for agent, group in means.groupby("agent"):
            rows.append(
                {
                    "agent": agent,
                    "metric": metric,
                    "n_environments": group["environment"].nunique(),
                    "mean_environment_rank": group["environment_rank"].mean(),
                    "median_environment_rank": group[
                        "environment_rank"
                    ].median(),
                    "best_environment_rank": group["environment_rank"].min(),
                    "worst_environment_rank": group["environment_rank"].max(),
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curves(raw: pd.DataFrame, output_dir: Path) -> None:
    display_order = [
        "tabular",
        "dqn",
        "fixed_hybrid",
        "count_gated_tau_20",
        "reliability_gated",
    ]
    display_names = {
        "tabular": "Tabular Q",
        "dqn": "DQN",
        "fixed_hybrid": "Fixed mixture",
        "count_gated_tau_20": "Count gate",
        "reliability_gated": "TD-reliability gate",
    }
    evaluation = raw[raw["phase"] == "eval"]
    for environment, env_group in evaluation.groupby("environment"):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        checkpoint_seed = (
            env_group.groupby(["agent", "seed", "checkpoint"])["return"]
            .mean()
            .reset_index()
        )
        for agent in display_order:
            agent_group = checkpoint_seed[
                checkpoint_seed["agent"] == agent
            ]
            if agent_group.empty:
                continue
            curve = (
                agent_group.groupby("checkpoint")["return"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            axis.plot(
                curve["checkpoint"],
                curve["mean"],
                linewidth=2,
                label=display_names[agent],
            )
            axis.fill_between(
                curve["checkpoint"],
                curve["mean"] - 1.96 * curve["sem"].fillna(0),
                curve["mean"] + 1.96 * curve["sem"].fillna(0),
                alpha=0.2,
            )
        axis.set_title(environment)
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Held-out evaluation return")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, frameon=False)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        safe_name = environment.replace("/", "_")
        figure.savefig(output_dir / f"learning_curve_{safe_name}.png", dpi=300)
        plt.close(figure)


def aggregate(input_path: str | Path, output_dir: str | Path) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path)
    seed_metrics = seed_level_metrics(raw)
    seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    summarize(seed_metrics).to_csv(output_dir / "summary.csv", index=False)
    pairwise(seed_metrics).to_csv(output_dir / "pairwise.csv", index=False)
    heavy_tail_diagnostics(seed_metrics).to_csv(
        output_dir / "heavy_tail_diagnostics.csv", index=False
    )
    cross_environment_summary(seed_metrics).to_csv(
        output_dir / "cross_environment.csv", index=False
    )
    metadata_path = input_path.parent / "metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contrasts = metadata.get("config", {}).get("analysis", {}).get(
            "planned_contrasts", []
        )
        planned = planned_contrasts(seed_metrics, contrasts)
        analysis_status = metadata.get("config", {}).get("analysis", {}).get(
            "analysis_status", "unspecified"
        )
        if not planned.empty:
            planned["analysis_status"] = analysis_status
        planned.to_csv(output_dir / "planned_contrasts.csv", index=False)
    plot_learning_curves(raw, output_dir)
