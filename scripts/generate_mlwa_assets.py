from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GENERATED = ROOT / "paper" / "generated"
FIGURES = ROOT / "paper" / "figures"

COLORS = {
    "random": "#7f7f7f",
    "tabular": "#1f77b4",
    "original_selected_dqn": "#9467bd",
    "validated_dqn": "#d62728",
    "fixed_hybrid": "#8c564b",
    "count_gated_tau_20": "#2ca02c",
    "reliability_gated": "#ff7f0e",
    "support_abstain_tau_20": "#17becf",
}
NAMES = {
    "random": "Random",
    "tabular": "Tabular Q",
    "original_selected_dqn": "Original DQN",
    "validated_dqn": "Validated DQN",
    "fixed_hybrid": "Fixed mixture",
    "count_gated_tau_20": "Count gate",
    "reliability_gated": "TD-reliability gate",
    "support_abstain_tau_20": "Support-abstain",
}
PLOT_AGENTS = (
    "tabular",
    "validated_dqn",
    "count_gated_tau_20",
    "support_abstain_tau_20",
)

COMPACT_ENVS = (
    "FrozenLake-4x4-slippery",
    "FrozenLake-8x8-slippery",
    "CliffWalking-v1",
    "Taxi-v3-compatible",
    "FourRooms-7-unseen-goals",
    "FourRooms-9-slip-unseen-goals",
    "FourRooms-11-slip-unseen-goals",
)
MINIGRID_ENVS = (
    "MiniGrid-Empty-5x5",
    "MiniGrid-Empty-6x6",
    "MiniGrid-Empty-8x8",
    "MiniGrid-DoorKey-5x5",
    "MiniGrid-DoorKey-6x6",
    "MiniGrid-FourRooms",
)
ENV_LABELS = {
    "FrozenLake-4x4-slippery": "FrozenLake 4x4",
    "FrozenLake-8x8-slippery": "FrozenLake 8x8",
    "CliffWalking-v1": "CliffWalking",
    "Taxi-v3-compatible": "Taxi",
    "FourRooms-7-unseen-goals": "FourRooms 7",
    "FourRooms-9-slip-unseen-goals": "FourRooms 9 slip",
    "FourRooms-11-slip-unseen-goals": "FourRooms 11 slip",
    "MiniGrid-Empty-5x5": "Empty 5x5",
    "MiniGrid-Empty-6x6": "Empty 6x6",
    "MiniGrid-Empty-8x8": "Empty 8x8",
    "MiniGrid-DoorKey-5x5": "DoorKey 5x5",
    "MiniGrid-DoorKey-6x6": "DoorKey 6x6",
    "MiniGrid-FourRooms": "FourRooms",
}


def seed_metrics(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name / "seed_metrics.csv")


def planned(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name / "planned_contrasts.csv")


def latex_table(frame: pd.DataFrame, stem: str, column_format: str) -> None:
    text = frame.to_latex(
        index=False,
        escape=True,
        column_format=column_format,
        float_format=lambda value: f"{value:.3g}",
    )
    (GENERATED / f"{stem}.tex").write_text(text, encoding="utf-8")


def interval(row: pd.Series) -> str:
    return (
        f"{row['mean']:.3f} "
        f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"
    )


def write_performance_tables() -> None:
    compact = pd.read_csv(
        RESULTS / "confirmatory_extended_compact" / "summary.csv"
    )
    compact = compact[compact["metric"] == "return_auc"].copy()
    compact = compact[
        compact["agent"].isin(
            ("tabular", "validated_dqn", "count_gated_tau_20", "support_abstain_tau_20")
        )
    ]
    compact["Environment"] = compact["environment"].map(ENV_LABELS)
    compact["Method"] = compact["agent"].map(NAMES)
    compact["Mean AUC [bootstrap 95% CI]"] = compact.apply(interval, axis=1)
    compact["Median AUC"] = compact["median"]
    latex_table(
        compact[
            ["Environment", "Method", "n_seeds", "Mean AUC [bootstrap 95% CI]", "Median AUC"]
        ].rename(columns={"n_seeds": "Seeds"}),
        "mlwa_recurrent_confirmatory",
        "llclc",
    )

    replication = planned("support_abstention_replication")
    replication = replication[
        replication["contrast"] == "support_abstain_vs_count"
    ].copy()
    replication["Environment"] = replication["environment"].replace(
        {
            "FourRooms-7-unseen-goals": "FourRooms 7 held-out",
            "FourRooms-9-slip-unseen-goals": "FourRooms 9 slip held-out",
            "FourRooms-11-slip-unseen-goals": "FourRooms 11 slip held-out",
            "MiniGrid-Empty-5x5-replication": "MiniGrid Empty 5x5",
        }
    )
    replication["Mean difference [bootstrap 95% CI]"] = replication.apply(
        lambda row: (
            f"{row['mean_difference']:.3f} "
            f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"
        ),
        axis=1,
    )
    replication["Wins/losses/ties"] = replication.apply(
        lambda row: f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])}",
        axis=1,
    )
    latex_table(
        replication[
            [
                "Environment",
                "n_pairs",
                "Mean difference [bootstrap 95% CI]",
                "paired_t_holm_p",
                "wilcoxon_holm_p",
                "Wins/losses/ties",
            ]
        ].rename(
            columns={
                "n_pairs": "Pairs",
                "paired_t_holm_p": "t-Holm p",
                "wilcoxon_holm_p": "Wilcoxon-Holm p",
            }
        ),
        "mlwa_support_replication",
        "lclccc",
    )

    minigrid = pd.read_csv(
        RESULTS / "minigrid_extended_diagnostic" / "summary.csv"
    )
    minigrid = minigrid[minigrid["metric"] == "return_auc"].copy()
    minigrid = minigrid[
        minigrid["agent"].isin(
            ("tabular", "validated_dqn", "count_gated_tau_20", "support_abstain_tau_20")
        )
    ]
    minigrid["Environment"] = minigrid["environment"].map(ENV_LABELS)
    minigrid["Method"] = minigrid["agent"].map(NAMES)
    minigrid["Mean AUC [bootstrap 95% CI]"] = minigrid.apply(interval, axis=1)
    latex_table(
        minigrid[
            ["Environment", "Method", "n_seeds", "Mean AUC [bootstrap 95% CI]", "median"]
        ].rename(columns={"n_seeds": "Seeds", "median": "Median AUC"}),
        "mlwa_minigrid_extended",
        "llclc",
    )

    validation = planned("dqn_strong_validation").copy()
    validation["Environment"] = validation["environment"].replace(
        {**ENV_LABELS, "MiniGrid-Empty-5x5-validation": "MiniGrid Empty 5x5"}
    )
    validation["Mean difference [bootstrap 95% CI]"] = validation.apply(
        lambda row: (
            f"{row['mean_difference']:.3f} "
            f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"
        ),
        axis=1,
    )
    validation["Wins/losses/ties"] = validation.apply(
        lambda row: f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])}",
        axis=1,
    )
    latex_table(
        validation[
            [
                "Environment",
                "n_pairs",
                "Mean difference [bootstrap 95% CI]",
                "paired_t_holm_p",
                "Wins/losses/ties",
            ]
        ].rename(
            columns={"n_pairs": "Pairs", "paired_t_holm_p": "t-Holm p"}
        ),
        "mlwa_dqn_validation",
        "lclcc",
    )

    rows = []
    for config_name in (
        "dqn_tuning_development",
        "dqn_strong_validation",
        "confirmatory_extended_compact",
        "support_abstention_replication",
        "minigrid_extended_diagnostic",
    ):
        config = json.loads(
            (ROOT / "configs" / f"{config_name}.json").read_text(
                encoding="utf-8"
            )
        )
        audit = json.loads(
            (RESULTS / config_name / "audit.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "Result family": config_name,
                "Status": audit["status"],
                "Environments": len(config["envs"]),
                "Agents": len(config["agents"]),
                "Runs": audit["observed_runs"],
                "Raw rows": audit["raw_rows"],
                "Analysis": config["analysis"]["analysis_status"],
            }
        )
    latex_table(
        pd.DataFrame(rows),
        "mlwa_benchmark_status",
        "llrrrrl",
    )


def evaluation_curves(name: str) -> pd.DataFrame:
    path = RESULTS / name / "raw.csv"
    if not path.exists():
        path = RESULTS / name / "raw.csv.gz"
    columns = ["phase", "environment", "agent", "seed", "checkpoint", "return"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        chunk = chunk[
            (chunk["phase"] == "eval")
            & chunk["agent"].isin(PLOT_AGENTS)
        ]
        if not chunk.empty:
            chunks.append(
                chunk.groupby(
                    ["environment", "agent", "seed", "checkpoint"],
                    as_index=False,
                )["return"].mean()
            )
    return pd.concat(chunks, ignore_index=True).groupby(
        ["environment", "agent", "seed", "checkpoint"], as_index=False
    )["return"].mean()


def curve_panel(
    frame: pd.DataFrame,
    environments: tuple[str, ...],
    stem: str,
    columns: int,
) -> None:
    rows = int(np.ceil(len(environments) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.5 * columns, 3.0 * rows), squeeze=False
    )
    for axis, environment in zip(axes.flat, environments):
        env = frame[frame["environment"] == environment]
        for agent in PLOT_AGENTS:
            values = env[env["agent"] == agent]
            if values.empty:
                continue
            curve = values.groupby("checkpoint")["return"].agg(
                median="median",
                q25=lambda series: series.quantile(0.25),
                q75=lambda series: series.quantile(0.75),
            )
            axis.plot(
                curve.index,
                curve["median"],
                label=NAMES[agent],
                color=COLORS[agent],
                linewidth=2.2,
            )
            axis.fill_between(
                curve.index,
                curve["q25"],
                curve["q75"],
                color=COLORS[agent],
                alpha=0.16,
            )
        axis.set_title(ENV_LABELS[environment], fontsize=11)
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Evaluation return")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(environments) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def support_boundary_figure() -> None:
    frame = seed_metrics("support_abstention_replication")
    environments = (
        "FourRooms-7-unseen-goals",
        "FourRooms-9-slip-unseen-goals",
        "FourRooms-11-slip-unseen-goals",
    )
    methods = ("tabular", "validated_dqn", "count_gated_tau_20", "support_abstain_tau_20")
    figure, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    rng = np.random.default_rng(27)
    for axis, environment in zip(axes[:3], environments):
        env = frame[frame["environment"] == environment]
        values = [
            env[env["agent"] == agent]["return_auc"].to_numpy()
            for agent in methods
        ]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, agent in zip(boxes["boxes"], methods):
            patch.set_facecolor(COLORS[agent])
            patch.set_alpha(0.3)
        for position, (agent, data) in enumerate(zip(methods, values), start=1):
            x = position + rng.uniform(-0.1, 0.1, len(data))
            axis.scatter(x, data, s=13, color=COLORS[agent], alpha=0.55)
        axis.set_xticks(range(1, 5), ["Tabular", "DQN", "Count", "Abstain"], rotation=25)
        axis.set_title(ENV_LABELS[environment])
        axis.set_ylabel("Seed-level return AUC")
        axis.grid(axis="y", alpha=0.2)

    counts = np.arange(0, 101)
    gate = counts / (counts + 20.0)
    axes[3].plot(counts, gate, color=COLORS["count_gated_tau_20"], linewidth=2.5)
    axes[3].scatter([0], [0], color=COLORS["count_gated_tau_20"], s=50)
    axes[3].scatter([0], [1], color=COLORS["support_abstain_tau_20"], s=50)
    axes[3].annotate(
        "Count gate: full neural delegation",
        (0, 0),
        xytext=(12, 0.18),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    axes[3].annotate(
        "Abstention: neutral tabular fallback",
        (0, 1),
        xytext=(12, 0.82),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    axes[3].set_xlabel("Exact-state visit count N(s)")
    axes[3].set_ylabel("Tabular weight")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Support boundary at N(s)=0")
    axes[3].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "Fig2_support_boundary.pdf", bbox_inches="tight")
    plt.close(figure)


def dqn_validation_figure() -> None:
    tuning = pd.read_csv(
        RESULTS / "dqn_tuning_development" / "cross_environment.csv"
    )
    tuning = tuning[tuning["metric"] == "return_auc"].sort_values(
        "mean_environment_rank"
    )
    validation = planned("dqn_strong_validation")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].barh(
        tuning["agent"],
        tuning["mean_environment_rank"],
        color="#4c78a8",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean environment rank (lower is better)")
    axes[0].set_title("Development-only DQN selection")
    axes[0].grid(axis="x", alpha=0.2)

    labels = [
        ENV_LABELS.get(environment, environment.replace("-validation", ""))
        for environment in validation["environment"]
    ]
    differences = validation["mean_difference"].to_numpy()
    low = differences - validation["bootstrap_ci_low"].to_numpy()
    high = validation["bootstrap_ci_high"].to_numpy() - differences
    axes[1].errorbar(
        differences,
        np.arange(len(labels)),
        xerr=np.vstack([low, high]),
        fmt="o",
        color=COLORS["validated_dqn"],
        ecolor="#777777",
        capsize=3,
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].set_xlabel("Validated minus original DQN return AUC")
    axes[1].set_title("Independent validation")
    axes[1].grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "Fig3_dqn_validation.pdf", bbox_inches="tight")
    plt.close(figure)


def extended_benchmark_figure() -> None:
    frames = [
        seed_metrics("confirmatory_extended_compact"),
        seed_metrics("minigrid_extended_diagnostic"),
    ]
    data = pd.concat(frames, ignore_index=True)
    agents = ("tabular", "validated_dqn", "count_gated_tau_20", "support_abstain_tau_20")
    means = data[data["agent"].isin(agents)].groupby(
        ["environment", "agent"], as_index=False
    )["return_auc"].mean()
    means["rank"] = means.groupby("environment")["return_auc"].rank(
        ascending=False, method="average"
    )
    matrix = means.pivot(index="environment", columns="agent", values="rank")
    order = list(COMPACT_ENVS) + list(MINIGRID_ENVS)
    matrix = matrix.reindex(order)[list(agents)]
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.imshow(matrix.to_numpy(), cmap="viridis_r", vmin=1, vmax=4)
    axis.set_xticks(range(len(agents)), [NAMES[agent] for agent in agents], rotation=20)
    axis.set_yticks(
        range(len(matrix)),
        [ENV_LABELS.get(environment, environment) for environment in matrix.index],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.1f}", ha="center", va="center")
    axis.set_title("Per-environment return-AUC rank; lower is better")
    figure.colorbar(image, ax=axis, label="Rank")
    figure.tight_layout()
    figure.savefig(FIGURES / "Fig4_extended_benchmark.pdf", bbox_inches="tight")
    plt.close(figure)


def write_verified_claims() -> None:
    rows = []
    for family in (
        "dqn_strong_validation",
        "confirmatory_extended_compact",
        "support_abstention_replication",
        "minigrid_extended_diagnostic",
    ):
        frame = planned(family)
        frame.insert(0, "result_family", family)
        rows.append(frame)
    claims = pd.concat(rows, ignore_index=True)
    claims.to_csv(GENERATED / "verified_claims.csv", index=False)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_performance_tables()
    compact_curves = evaluation_curves("confirmatory_extended_compact")
    minigrid_curves = evaluation_curves("minigrid_extended_diagnostic")
    curve_panel(
        compact_curves,
        COMPACT_ENVS,
        "Fig1A_compact_learning_curves",
        columns=3,
    )
    curve_panel(
        minigrid_curves,
        MINIGRID_ENVS,
        "Fig1B_minigrid_learning_curves",
        columns=3,
    )
    support_boundary_figure()
    dqn_validation_figure()
    extended_benchmark_figure()
    write_verified_claims()
    print("Generated MLWA tables, figures, and verified_claims.csv")


if __name__ == "__main__":
    main()
