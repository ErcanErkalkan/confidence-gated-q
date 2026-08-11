from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.aggregate_reliability_calibration import (  # noqa: E402
    build_branch_discrimination,
    build_calibration_bins,
    build_detection_delay,
    build_selective_risk,
    load_and_validate_evaluation,
)


def build_summary(
    branch: pd.DataFrame,
    bins: pd.DataFrame,
    delay: pd.DataFrame,
) -> pd.DataFrame:
    primary = branch.loc[branch["scope"].eq("post_shift_all")].copy()
    keys = ["environment", "agent", "target_type"]
    ece = (
        bins.assign(weighted_gap=bins["calibration_gap"] * bins["n_rows"])
        .groupby(keys, sort=True)
        .apply(
            lambda group: group["weighted_gap"].sum() / group["n_rows"].sum(),
            include_groups=False,
        )
        .rename("expected_calibration_error")
        .reset_index()
    )
    delay_summary = (
        delay.groupby(["environment", "agent"], sort=True)
        .agg(
            detection_rate=("detected", "mean"),
            detected_seed_count=("detected", "sum"),
            adaptation_delay_median=("adaptation_delay_steps", "median"),
        )
        .reset_index()
    )
    result = primary.merge(ece, on=keys, how="left").merge(
        delay_summary, on=["environment", "agent"], how="left"
    )
    result["class_sufficient"] = (
        result["n_positive"].gt(0) & result["n_negative"].gt(0)
    )
    result["evidence_class"] = "replication_calibration"
    result["performance_claim_scope"] = "diagnostic_only"
    columns = [
        "environment",
        "agent",
        "target_type",
        "target_definition",
        "aggregation_level",
        "n_rows",
        "n_seeds",
        "n_positive",
        "n_negative",
        "class_sufficient",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "auroc_availability",
        "average_precision",
        "average_precision_ci_low",
        "average_precision_ci_high",
        "balanced_accuracy_0_5",
        "balanced_accuracy_0_5_ci_low",
        "balanced_accuracy_0_5_ci_high",
        "brier_score",
        "brier_score_ci_low",
        "brier_score_ci_high",
        "expected_calibration_error",
        "detection_rate",
        "detected_seed_count",
        "adaptation_delay_median",
        "evidence_class",
        "performance_claim_scope",
        "source_file",
    ]
    return result[columns].sort_values(keys).reset_index(drop=True)


def make_figure(
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    selective: pd.DataFrame,
    output: Path,
) -> None:
    agents = sorted(summary["agent"].unique())
    environments = sorted(summary["environment"].unique())
    targets = ["action_correctness", "value_error"]
    colors = {agents[0]: "#3568A8", agents[-1]: "#C58A18"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    x = np.arange(len(environments) * len(targets))
    width = 0.36
    labels = [
        f"{environment.split('-')[2]}\n{target.replace('_', ' ')}"
        for environment in environments
        for target in targets
    ]
    for offset, agent in enumerate(agents):
        values = []
        for environment in environments:
            for target in targets:
                row = summary[
                    summary["environment"].eq(environment)
                    & summary["agent"].eq(agent)
                    & summary["target_type"].eq(target)
                ]
                values.append(float(row["auroc"].iloc[0]) if len(row) else np.nan)
        axes[0, 0].bar(
            x + (offset - (len(agents) - 1) / 2) * width,
            values,
            width,
            label=agent.replace("_", " "),
            color=colors[agent],
        )
    axes[0, 0].axhline(0.5, color="#30343B", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 0].set_ylabel("Mean seed-level AUROC")
    axes[0, 0].set_title("Independent branch discrimination")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for agent in agents:
        selected = bins[
            bins["agent"].eq(agent)
            & bins["target_type"].eq("action_correctness")
        ]
        grouped = selected.groupby("bin_index").agg(
            score=("mean_score", "mean"),
            observed=("observed_tabular_better_rate", "mean"),
        )
        axes[0, 1].plot(
            grouped["score"],
            grouped["observed"],
            marker="o",
            label=agent.replace("_", " "),
            color=colors[agent],
        )
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#30343B")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Mean relative-reliability score")
    axes[0, 1].set_ylabel("Observed tabular-better rate")
    axes[0, 1].set_title("Action-target calibration")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for agent in agents:
        selected = selective[
            selective["agent"].eq(agent)
            & selective["target_type"].eq("action_correctness")
        ]
        grouped = selected.groupby("requested_coverage")["selective_risk"].mean()
        axes[1, 0].plot(
            grouped.index,
            grouped.values,
            marker="o",
            label=agent.replace("_", " "),
            color=colors[agent],
        )
    axes[1, 0].set_xlim(0.1, 1.0)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Retained coverage")
    axes[1, 0].set_ylabel("Prediction error rate")
    axes[1, 0].set_title("Selective action risk")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for offset, agent in enumerate(agents):
        selected = summary[
            summary["agent"].eq(agent)
            & summary["target_type"].eq("action_correctness")
        ]
        axes[1, 1].bar(
            np.arange(len(selected))
            + (offset - (len(agents) - 1) / 2) * width,
            selected["brier_score"],
            width,
            color=colors[agent],
            label=agent.replace("_", " "),
        )
    axes[1, 1].set_xticks(
        np.arange(len(environments)),
        [item.split("-")[2] for item in environments],
        rotation=20,
        ha="right",
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Brier score")
    axes[1, 1].set_title("Action-target probabilistic calibration")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", color="#D8DCE2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Independent focal relative-reliability calibration\n"
        "Analytic action and value targets; final seeds 12090--12099",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_independent(
    input_dir: Path,
    config_path: Path,
    output_dir: Path,
    table_path: Path,
    figure_path: Path,
) -> dict[str, int]:
    evaluation, config = load_and_validate_evaluation(input_dir, config_path)
    branch = build_branch_discrimination(evaluation)
    bins = build_calibration_bins(evaluation, scope="post_shift_all")
    selective = build_selective_risk(evaluation, scope="post_shift_all")
    delay = build_detection_delay(evaluation, config)
    summary = build_summary(branch, bins, delay)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    branch.to_csv(output_dir / "branch_discrimination.csv", index=False)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    selective.to_csv(output_dir / "selective_risk.csv", index=False)
    delay.to_csv(output_dir / "detection_delay.csv", index=False)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    summary.to_csv(table_path, index=False)
    make_figure(summary, bins, selective, figure_path)
    return {
        "evaluation_rows": len(evaluation),
        "branch_rows": len(branch),
        "calibration_bin_rows": len(bins),
        "selective_risk_rows": len(selective),
        "delay_rows": len(delay),
        "summary_rows": len(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results/diagnostic_extensions/reliability_calibration_independent/execution",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/diagnostic_extensions/reliability_calibration/independent_focal.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/diagnostic_extensions/reliability_calibration_independent",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "tables/table_reliability_calibration_independent.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures/fig_reliability_calibration_independent.pdf",
    )
    args = parser.parse_args()
    counts = aggregate_independent(
        args.input_dir,
        args.config,
        args.output_dir,
        args.table,
        args.figure,
    )
    print(
        "RELIABILITY_CALIBRATION_INDEPENDENT_PASS "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )


if __name__ == "__main__":
    main()
