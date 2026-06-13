from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from result_reporting import ROOT, read_seed_metrics, write_result_report


AGENTS = [
    "sensorized_controller",
    "dqn",
    "double_dqn",
    "fuzzy_reliability",
]
NAMES = {
    "sensorized_controller": "Sensorized flight controller",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "fuzzy_reliability": "Fuzzy reliability gate",
}


def main() -> None:
    source = ROOT / "results/uav_sensorized_motor_validation/seed_metrics.csv"
    result_dir = ROOT / "results/uav_sensorized_motor_validation/reported"
    write_result_report(
        read_seed_metrics(source),
        result_dir,
        ROOT / "tables/table_uav_sensorized_validation.csv",
        ROOT / "figures/fig_uav_sensorized_validation.pdf",
        agents=AGENTS,
        names=NAMES,
        table_metrics=[
            "risk_adjusted_score",
            "success_rate",
            "collision_rate",
            "localization_error_mean",
            "sensor_dropout_rate",
            "motor_saturation_rate",
        ],
        contrasts=[
            {
                "name": "double_dqn_vs_dqn",
                "left": "double_dqn",
                "right": "dqn",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "fuzzy_reliability_vs_dqn",
                "left": "fuzzy_reliability",
                "right": "dqn",
                "metric": "risk_adjusted_score",
            },
            {
                "name": "fuzzy_reliability_vs_sensorized_controller",
                "left": "fuzzy_reliability",
                "right": "sensorized_controller",
                "metric": "risk_adjusted_score",
            },
        ],
        figure_metric="risk_adjusted_score",
    )
    summary = pd.read_csv(result_dir / "summary.csv")
    values = summary[
        summary["metric"].isin(["success_rate", "collision_rate"])
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
    axis.set_title("Sensorized Crazyflie SIL deployment shift")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        ROOT / "figures/fig_uav_sensorized_tradeoff.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
