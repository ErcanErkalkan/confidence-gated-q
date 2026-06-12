from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from result_reporting import ROOT, read_seed_metrics, write_result_report


AGENTS = [
    "pid_safe_controller",
    "dqn",
    "double_dqn",
    "fuzzy_support",
    "feature_support",
]
NAMES = {
    "pid_safe_controller": "Obstacle-aware PID",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "fuzzy_support": "Fuzzy support",
    "feature_support": "Feature-distance support",
}


def main() -> None:
    metrics = read_seed_metrics(
        ROOT / "results/uav_pybullet_validation/seed_metrics.csv"
    )
    write_result_report(
        metrics,
        ROOT / "results/uav_pybullet_validation/combined",
        ROOT / "tables/table_uav_pybullet_validation.csv",
        ROOT / "figures/fig_uav_pybullet_validation.pdf",
        agents=AGENTS,
        names=NAMES,
        table_metrics=[
            "risk_adjusted_score",
            "success_rate",
            "collision_rate",
            "return_auc",
            "unsupported_state_ratio",
        ],
        contrasts=[
            {
                "name": "double_dqn_vs_dqn",
                "left": "double_dqn",
                "right": "dqn",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "fuzzy_support_vs_dqn",
                "left": "fuzzy_support",
                "right": "dqn",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "feature_support_vs_dqn",
                "left": "feature_support",
                "right": "dqn",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "feature_support_vs_pid_safe_controller",
                "left": "feature_support",
                "right": "pid_safe_controller",
                "metric": "risk_adjusted_score",
            },
        ],
        figure_metric="risk_adjusted_score",
    )
    summary = pd.read_csv(
        ROOT / "results/uav_pybullet_validation/combined/summary.csv"
    )
    values = summary[
        summary["metric"].isin(
            ["success_rate", "collision_rate", "risk_adjusted_score"]
        )
    ].pivot(index="agent", columns="metric", values="mean")
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for agent, row in values.iterrows():
        axis.scatter(
            row["collision_rate"],
            row["success_rate"],
            s=90,
            label=NAMES[agent],
        )
        axis.annotate(
            NAMES[agent],
            (row["collision_rate"], row["success_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Collision rate (lower is better)")
    axis.set_ylabel("Waypoint success rate (higher is better)")
    axis.set_title("Physics-based Crazyflie deployment shift")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        ROOT / "figures/fig_uav_pybullet_tradeoff.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
