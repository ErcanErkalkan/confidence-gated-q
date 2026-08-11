from __future__ import annotations

import pandas as pd

from result_reporting import ROOT, read_seed_metrics, write_result_report


def main() -> None:
    application = pd.read_csv(
        ROOT / "results/application_navigation_case_study/seed_metrics.csv"
    )
    application = application[
        application["agent"].isin(
            ["tabular", "count_gated_tau_20", "support_abstain_tau_20"]
        )
    ]
    new = read_seed_metrics(
        ROOT / "results/approx_support/knn_support/seed_metrics.csv",
        ROOT / "results/approx_support/feature_distance_support/seed_metrics.csv",
    )
    combined = pd.concat([application, new], ignore_index=True, sort=False)
    write_result_report(
        combined,
        ROOT / "results/approx_support",
        ROOT / "tables/table_approx_support.csv",
        ROOT / "figures/fig_approx_support.pdf",
        agents=[
            "tabular",
            "count_gated_tau_20",
            "support_abstain_tau_20",
            "knn_support_gate",
            "feature_distance_support_gate",
        ],
        names={
            "tabular": "Tabular Q",
            "count_gated_tau_20": "Exact count gate",
            "support_abstain_tau_20": "Exact support abstention",
            "knn_support_gate": "kNN support gate",
            "feature_distance_support_gate": "Feature-distance gate",
        },
        table_metrics=[
            "return_auc",
            "success_rate",
            "unsupported_state_ratio",
            "abstention_ratio",
            "collision_rate",
        ],
        contrasts=[
            {
                "name": "knn_vs_exact_count",
                "left": "knn_support_gate",
                "right": "count_gated_tau_20",
                "metric": "return_auc",
            },
            {
                "name": "feature_distance_vs_exact_count",
                "left": "feature_distance_support_gate",
                "right": "count_gated_tau_20",
                "metric": "return_auc",
            },
            {
                "name": "knn_vs_tabular",
                "left": "knn_support_gate",
                "right": "tabular",
                "metric": "return_auc",
            },
            {
                "name": "feature_distance_vs_tabular",
                "left": "feature_distance_support_gate",
                "right": "tabular",
                "metric": "return_auc",
            },
        ],
    )


if __name__ == "__main__":
    main()
