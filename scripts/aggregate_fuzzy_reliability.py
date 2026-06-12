from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid_q.statistics import planned_contrasts, summarize
from result_reporting import ROOT, read_seed_metrics


def write_multi_environment_report(
    seed_metrics: pd.DataFrame,
    result_dir: Path,
    table_path: Path,
    figure_path: Path,
    *,
    agents: list[str],
    names: dict[str, str],
    table_metrics: list[str],
    contrasts: list[dict],
    figure_metric: str = "return_auc",
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    seed_metrics = seed_metrics[seed_metrics["agent"].isin(agents)].copy()
    seed_metrics.to_csv(result_dir / "seed_metrics.csv", index=False)
    summary = summarize(seed_metrics)
    summary.to_csv(result_dir / "summary.csv", index=False)
    planned_contrasts(seed_metrics, contrasts).to_csv(
        result_dir / "planned_contrasts.csv", index=False
    )
    table = summary[summary["metric"].isin(table_metrics)].copy()
    table["method"] = table["agent"].map(names)
    table.to_csv(table_path, index=False)

    selected = summary[summary["metric"] == figure_metric]
    environments = list(selected["environment"].drop_duplicates())
    x = np.arange(len(environments), dtype=float)
    width = 0.8 / len(agents)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for index, agent in enumerate(agents):
        rows = selected[selected["agent"] == agent].set_index("environment")
        means = rows.loc[environments, "mean"].to_numpy(dtype=float)
        low = rows.loc[
            environments, "bootstrap_ci_low"
        ].to_numpy(dtype=float)
        high = rows.loc[
            environments, "bootstrap_ci_high"
        ].to_numpy(dtype=float)
        positions = x - 0.4 + width / 2 + index * width
        axis.bar(
            positions,
            means,
            width,
            yerr=np.vstack((means - low, high - means)),
            capsize=2,
            label=names[agent],
        )
    axis.set_xticks(x, environments, rotation=15, ha="right")
    axis.set_ylabel(figure_metric.replace("_", " "))
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    write_multi_environment_report(
        read_seed_metrics(
            ROOT / "results/fuzzy_reliability_confirmatory/seed_metrics.csv"
        ),
        ROOT / "results/fuzzy_reliability_confirmatory/reported",
        ROOT / "tables/table_fuzzy_reliability_stationary.csv",
        ROOT / "figures/fig_fuzzy_reliability_stationary.pdf",
        agents=[
            "count_gate_tau_20",
            "original_fuzzy",
            "fuzzy_reliability_full",
            "crisp_relative_gate",
        ],
        names={
            "count_gate_tau_20": "Count gate",
            "original_fuzzy": "Original fuzzy gate",
            "fuzzy_reliability_full": "Fuzzy relative-reliability gate",
            "crisp_relative_gate": "Crisp relative-reliability gate",
        },
        table_metrics=["return_auc"],
        contrasts=[
            {
                "name": "fuzzy_reliability_vs_count",
                "left": "fuzzy_reliability_full",
                "right": "count_gate_tau_20",
                "metric": "return_auc",
            },
            {
                "name": "fuzzy_reliability_vs_crisp",
                "left": "fuzzy_reliability_full",
                "right": "crisp_relative_gate",
                "metric": "return_auc",
            },
        ],
    )
    write_multi_environment_report(
        read_seed_metrics(
            ROOT
            / "results/fuzzy_reliability_shift_confirmatory/seed_metrics.csv"
        ),
        ROOT / "results/fuzzy_reliability_shift_confirmatory/reported",
        ROOT / "tables/table_fuzzy_reliability_shift.csv",
        ROOT / "figures/fig_fuzzy_reliability_shift.pdf",
        agents=[
            "count_gate",
            "original_fuzzy",
            "fuzzy_shift_full",
            "fuzzy_shift_crisp",
            "fuzzy_shift_no_reliability",
        ],
        names={
            "count_gate": "Count gate",
            "original_fuzzy": "Original fuzzy gate",
            "fuzzy_shift_full": "Fuzzy relative-reliability gate",
            "fuzzy_shift_crisp": "Crisp relative-reliability gate",
            "fuzzy_shift_no_reliability": "No-reliability ablation",
        },
        table_metrics=[
            "return_auc",
            "success_rate",
            "adaptive_alpha_mean",
        ],
        contrasts=[
            {
                "name": "fuzzy_shift_vs_count",
                "left": "fuzzy_shift_full",
                "right": "count_gate",
                "metric": "return_auc",
            },
            {
                "name": "fuzzy_shift_vs_original",
                "left": "fuzzy_shift_full",
                "right": "original_fuzzy",
                "metric": "return_auc",
            },
            {
                "name": "fuzzy_shift_vs_crisp",
                "left": "fuzzy_shift_full",
                "right": "fuzzy_shift_crisp",
                "metric": "return_auc",
            },
        ],
    )


if __name__ == "__main__":
    main()
