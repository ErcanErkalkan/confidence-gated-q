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
    def normalized_auc(group: pd.DataFrame) -> float:
        checkpoints = group["checkpoint"].to_numpy(dtype=float)
        values = group["mean_return"].to_numpy(dtype=float)
        width = checkpoints[-1] - checkpoints[0]
        return float(np.trapezoid(values, checkpoints) / width) if width > 0 else float(values[-1])

    auc = (
        grouped.groupby(["environment", "agent", "seed"])
        .apply(normalized_auc, include_groups=False)
        .rename("return_auc")
        .reset_index()
    )
    final = grouped.groupby(["environment", "agent", "seed"]).tail(1).copy()
    final["seconds_per_1000_steps"] = (
        1000.0
        * final["training_elapsed_seconds"]
        / final["environment_steps"]
    )
    return final.merge(auc, on=["environment", "agent", "seed"])


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (environment, agent), group in seed_metrics.groupby(
        ["environment", "agent"]
    ):
        for metric in (
            "mean_return",
            "success_rate",
            "return_auc",
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
    paired = left_values.merge(
        right_values, on="seed", suffixes=("_left", "_right")
    )
    differences = (
        paired[f"{metric}_left"] - paired[f"{metric}_right"]
    ).to_numpy()
    if len(differences) < 2:
        return None
    difference_std = differences.std(ddof=1)
    if np.allclose(differences, 0):
        paired_t_p = 1.0
        wilcoxon_p = 1.0
    elif difference_std <= 1e-12:
        paired_t_p = 0.0
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
    else:
        paired_t_p = float(
            stats.ttest_rel(
                paired[f"{metric}_left"],
                paired[f"{metric}_right"],
            ).pvalue
        )
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    wins = int((differences > 0).sum())
    losses = int((differences < 0).sum())
    ties = int(len(differences) - wins - losses)
    sign_test_p = (
        float(stats.binomtest(wins, wins + losses, p=0.5).pvalue)
        if wins + losses
        else 1.0
    )
    ci_low, ci_high = bootstrap_mean_interval(differences)
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
    metadata_path = input_path.parent / "metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contrasts = metadata.get("config", {}).get("analysis", {}).get(
            "planned_contrasts", []
        )
        planned_contrasts(seed_metrics, contrasts).to_csv(
            output_dir / "planned_contrasts.csv", index=False
        )
    plot_learning_curves(raw, output_dir)
