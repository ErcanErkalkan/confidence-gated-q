from __future__ import annotations

from result_reporting import ROOT, read_seed_metrics, write_result_report


def main() -> None:
    application = read_seed_metrics(
        ROOT / "results/application_navigation_case_study/seed_metrics.csv"
    )
    application = application[application["agent"] == "validated_dqn"]
    combined = read_seed_metrics(
        ROOT / "results/strong_baselines/double_dqn/seed_metrics.csv",
        ROOT / "results/strong_baselines/dueling_double_dqn/seed_metrics.csv",
    )
    combined = read_seed_metrics_from_frames(application, combined)
    write_result_report(
        combined,
        ROOT / "results/strong_baselines",
        ROOT / "tables/table_strong_baselines.csv",
        ROOT / "figures/fig_strong_baselines.pdf",
        agents=["validated_dqn", "double_dqn", "dueling_double_dqn"],
        names={
            "validated_dqn": "Validated DQN",
            "double_dqn": "Double DQN",
            "dueling_double_dqn": "Dueling Double DQN",
        },
        table_metrics=["return_auc", "success_rate", "collision_rate"],
        contrasts=[
            {
                "name": "double_dqn_vs_validated_dqn",
                "left": "double_dqn",
                "right": "validated_dqn",
                "metric": "return_auc",
            },
            {
                "name": "dueling_double_dqn_vs_validated_dqn",
                "left": "dueling_double_dqn",
                "right": "validated_dqn",
                "metric": "return_auc",
            },
        ],
    )


def read_seed_metrics_from_frames(*frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True, sort=False)


if __name__ == "__main__":
    main()
