from __future__ import annotations

import pandas as pd

from result_reporting import ROOT, write_result_report


def main() -> None:
    baseline = pd.read_csv(
        ROOT / "results/application_navigation_case_study/seed_metrics.csv"
    )
    baseline = baseline[baseline["agent"] == "fuzzy_support_adaptive"]
    ablations = pd.read_csv(
        ROOT / "results/fuzzy_ablation/seed_metrics.csv"
    )
    combined = pd.concat([baseline, ablations], ignore_index=True, sort=False)
    write_result_report(
        combined,
        ROOT / "results/fuzzy_ablation/combined",
        ROOT / "tables/table_fuzzy_ablation.csv",
        ROOT / "figures/fig_fuzzy_ablation.pdf",
        agents=[
            "fuzzy_support_adaptive",
            "fuzzy_no_td_residual",
            "fuzzy_no_count",
            "fuzzy_no_confidence",
            "crisp_threshold_gate",
        ],
        names={
            "fuzzy_support_adaptive": "Full fuzzy gate",
            "fuzzy_no_td_residual": "No TD residual",
            "fuzzy_no_count": "No count",
            "fuzzy_no_confidence": "No confidence",
            "crisp_threshold_gate": "Crisp threshold",
        },
        table_metrics=[
            "return_auc",
            "success_rate",
            "collision_rate",
            "adaptive_alpha_mean",
        ],
        contrasts=[
            {
                "name": "no_td_residual_vs_full",
                "left": "fuzzy_no_td_residual",
                "right": "fuzzy_support_adaptive",
                "metric": "return_auc",
            },
            {
                "name": "no_count_vs_full",
                "left": "fuzzy_no_count",
                "right": "fuzzy_support_adaptive",
                "metric": "return_auc",
            },
            {
                "name": "no_confidence_vs_full",
                "left": "fuzzy_no_confidence",
                "right": "fuzzy_support_adaptive",
                "metric": "return_auc",
            },
            {
                "name": "crisp_threshold_vs_full",
                "left": "crisp_threshold_gate",
                "right": "fuzzy_support_adaptive",
                "metric": "return_auc",
            },
        ],
    )


if __name__ == "__main__":
    main()
