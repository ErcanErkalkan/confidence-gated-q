from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from result_reporting import ROOT, read_seed_metrics, write_result_report


def main() -> None:
    metrics = read_seed_metrics(
        ROOT / "results/application_risk_variants/seed_metrics.csv"
    )
    write_result_report(
        metrics,
        ROOT / "results/application_risk_variants/combined",
        ROOT / "tables/table_application_risk_adjusted.csv",
        ROOT / "figures/fig_application_tradeoff.pdf",
        agents=[
            "support_hold_penalized",
            "support_no_hold",
            "support_shortest_path",
            "support_verified_safe",
        ],
        names={
            "support_hold_penalized": "Hold fallback + penalty",
            "support_no_hold": "No-hold fallback",
            "support_shortest_path": "Shortest-path fallback",
            "support_verified_safe": "Verified-safe fallback",
        },
        table_metrics=[
            "risk_adjusted_score",
            "success_rate",
            "collision_rate",
            "idle_rate",
            "return_auc",
        ],
        contrasts=[
            {
                "name": "no_hold_vs_hold_penalized",
                "left": "support_no_hold",
                "right": "support_hold_penalized",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "shortest_path_vs_hold_penalized",
                "left": "support_shortest_path",
                "right": "support_hold_penalized",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "verified_safe_vs_hold_penalized",
                "left": "support_verified_safe",
                "right": "support_hold_penalized",
                "metric": "risk_adjusted_score",
            },
        ],
        figure_metric="risk_adjusted_score",
    )
    summary = pd.read_csv(
        ROOT / "results/application_risk_variants/combined/summary.csv"
    )
    values = summary[
        summary["metric"].isin(
            ["success_rate", "collision_rate", "idle_rate"]
        )
    ].pivot(index="agent", columns="metric", values="mean")
    names = {
        "support_hold_penalized": "Hold + penalty",
        "support_no_hold": "No hold",
        "support_shortest_path": "Shortest path",
        "support_verified_safe": "Verified safe",
    }
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for agent, row in values.iterrows():
        axis.scatter(
            row["collision_rate"],
            row["success_rate"],
            s=80 + 900 * row["idle_rate"],
            label=names[agent],
        )
        axis.annotate(
            names[agent],
            (row["collision_rate"], row["success_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Collision rate (lower is better)")
    axis.set_ylabel("Success rate (higher is better)")
    axis.set_title("Application fallback risk-success trade-off")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        ROOT / "figures/fig_application_tradeoff.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
