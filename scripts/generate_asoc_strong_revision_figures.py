from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
GRAPHICAL_ABSTRACT = ROOT / "paper" / "graphical_abstract"

ORDER = [
    "tabular",
    "validated_dqn",
    "fixed_hybrid",
    "count_gated_tau_20",
    "support_abstain_tau_20",
    "fuzzy_support_adaptive",
]
COST_ORDER = [agent for agent in ORDER if agent != "fixed_hybrid"]
NAMES = {
    "tabular": "Tabular Q",
    "validated_dqn": "DQN",
    "fixed_hybrid": "Fixed",
    "count_gated_tau_20": "Count",
    "support_abstain_tau_20": "Abstain",
    "fuzzy_support_adaptive": "Fuzzy",
}
COLORS = {
    "tabular": "#1f77b4",
    "validated_dqn": "#d62728",
    "fixed_hybrid": "#8c564b",
    "count_gated_tau_20": "#2ca02c",
    "support_abstain_tau_20": "#17becf",
    "fuzzy_support_adaptive": "#9467bd",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_application_map(axis) -> None:
    size = 9
    walls = {(4, y) for y in range(size)} | {(x, 4) for x in range(size)}
    walls -= {(4, 1), (4, 7), (1, 4), (7, 4)}
    starts = {(1, 3), (3, 1), (5, 7), (7, 5)}
    train_goals = {(1, 1), (7, 1), (1, 7)}
    shifted_goals = {(7, 7), (5, 5)}
    risk_cells = {(2, 2), (2, 6), (6, 2), (6, 6)}
    for y in range(size):
        for x in range(size):
            face = "white"
            label = ""
            if (x, y) in walls:
                face = "#455a64"
            elif (x, y) in risk_cells:
                face = "#ffe0b2"
                label = "R"
            elif (x, y) in train_goals:
                face = "#c8e6c9"
                label = "T"
            elif (x, y) in shifted_goals:
                face = "#ffcdd2"
                label = "S"
            elif (x, y) in starts:
                face = "#bbdefb"
                label = "A"
            axis.add_patch(
                Rectangle(
                    (x, size - 1 - y),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#37474f",
                    linewidth=0.6,
                )
            )
            if label:
                axis.text(
                    x + 0.5,
                    size - 1 - y + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
    axis.set_xlim(0, size)
    axis.set_ylim(0, size)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("(a) Application layout")
    axis.text(
        0.0,
        -0.9,
        "A=start, T=training goal, S=shifted goal, R=risk, dark=wall",
        fontsize=7.5,
        transform=axis.transData,
    )


def application_figure() -> None:
    result_dir = RESULTS / "application_navigation_case_study"
    raw_path = result_dir / "raw.csv"
    if not raw_path.exists():
        raw_path = result_dir / "raw.csv.gz"
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(
        RESULTS / "application_navigation_case_study" / "summary.csv"
    )
    evaluation = raw[raw["phase"] == "eval"]
    checkpoint = (
        evaluation.groupby(["agent", "seed", "checkpoint"])["return"]
        .mean()
        .reset_index()
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.0))
    _draw_application_map(axes[0])
    for agent in ORDER:
        selected = checkpoint[checkpoint["agent"] == agent]
        curve = (
            selected.groupby("checkpoint")["return"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        axes[1].plot(
            curve["checkpoint"],
            curve["mean"],
            label=NAMES[agent],
            color=COLORS[agent],
            linewidth=1.8,
        )
        radius = 1.96 * curve["sem"].fillna(0)
        axes[1].fill_between(
            curve["checkpoint"],
            curve["mean"] - radius,
            curve["mean"] + radius,
            color=COLORS[agent],
            alpha=0.14,
        )
    axes[1].set_title("(b) Deployment-shift learning curves")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Evaluation return")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=2)

    final = summary[
        summary["metric"].isin(("success_rate", "collision_rate"))
    ].pivot(index="agent", columns="metric", values="mean")
    x = np.arange(len(ORDER))
    width = 0.36
    axes[2].bar(
        x - width / 2,
        final.loc[ORDER, "success_rate"],
        width,
        label="Success",
        color="#2ca02c",
    )
    axes[2].bar(
        x + width / 2,
        final.loc[ORDER, "collision_rate"],
        width,
        label="Collision",
        color="#d62728",
    )
    axes[2].set_title("(c) Final deployment outcomes")
    axes[2].set_ylabel("Rate")
    axes[2].set_xticks(x, [NAMES[item] for item in ORDER], rotation=30)
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        FIGURES / "Figure_05_Application_Case_Study.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def cost_support_figure() -> None:
    summary = pd.read_csv(
        RESULTS / "cost_support_metrics" / "summary.csv"
    )
    environments = [
        "ApplicationNavigation-cost-support",
        "FourRooms-9-cost-support",
    ]
    titles = ["Application navigation", "FourRooms support shift"]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    metrics = [
        ("unsupported_state_ratio", "Unsupported-state ratio"),
        (
            "inference_time_us_per_decision_mean",
            "Decision time (microseconds)",
        ),
        ("memory_cost_bytes_estimated", "Exact-memory estimate (KiB)"),
        ("memory_branch_usage_ratio", "Memory-branch usage"),
    ]
    x = np.arange(len(COST_ORDER))
    width = 0.36
    for axis, (metric_name, ylabel) in zip(axes.flat, metrics):
        selected = summary[summary["metric"] == metric_name]
        for index, (environment, title) in enumerate(
            zip(environments, titles)
        ):
            values = (
                selected[selected["environment"] == environment]
                .set_index("agent")
                .loc[COST_ORDER, "mean"]
                .to_numpy()
            )
            if metric_name == "memory_cost_bytes_estimated":
                values = values / 1024.0
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=title,
                color=("#4c78a8", "#f58518")[index],
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            x, [NAMES[item] for item in COST_ORDER], rotation=30
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        FIGURES / "Figure_07_Cost_Support_and_Unsupported_Ratio.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _fuzzy_alpha_grid(
    tau_n: float,
    kappa: float,
    consequents: tuple[float, float, float, float, float],
    shape: str = "triangular",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.linspace(0, 80, 121)
    errors = np.linspace(0, 4, 121)
    x, y = np.meshgrid(counts, errors)
    support = 1.0 - np.exp(-x / tau_n)
    uncertainty = y / (y + kappa + 1e-12)
    if shape == "shoulder":
        low = 1.0 / (1.0 + np.exp(12.0 * (support - 0.35)))
        high = 1.0 / (1.0 + np.exp(-12.0 * (support - 0.65)))
        medium = np.maximum(0.0, 1.0 - np.maximum(low, high))
    else:
        low = np.clip(1.0 - 2.0 * support, 0.0, 1.0)
        medium = np.clip(1.0 - np.abs(2.0 * support - 1.0), 0.0, 1.0)
        high = np.clip(2.0 * support - 1.0, 0.0, 1.0)
    low_unc = 1.0 - uncertainty
    high_unc = uncertainty
    weights = [
        low,
        medium * low_unc,
        medium * high_unc,
        high * low_unc,
        high * high_unc,
    ]
    numerator = sum(weight * value for weight, value in zip(weights, consequents))
    denominator = sum(weights)
    alpha = numerator / np.maximum(denominator, 1e-12)
    alpha = np.clip(alpha, 0.05, 0.95)
    alpha[x == 0] = 0.0
    return x, y, alpha


def fuzzy_sensitivity_figure() -> None:
    settings = [
        (5.0, 1.0, (0.0, 0.35, 0.65, 0.75, 0.95), "triangular", "Fast support"),
        (20.0, 1.0, (0.0, 0.35, 0.65, 0.75, 0.95), "triangular", "Default"),
        (80.0, 1.0, (0.0, 0.35, 0.65, 0.75, 0.95), "triangular", "Slow support"),
        (20.0, 0.25, (0.0, 0.35, 0.65, 0.75, 0.95), "triangular", "Low kappa"),
        (20.0, 4.0, (0.0, 0.35, 0.65, 0.75, 0.95), "triangular", "High kappa"),
        (20.0, 1.0, (0.0, 0.20, 0.55, 0.65, 0.90), "shoulder", "Shape/consequent"),
    ]
    figure, axes = plt.subplots(
        3, 2, figsize=(10.4, 10.2), constrained_layout=True
    )
    last = None
    for index, (axis, (tau_n, kappa, consequents, shape, title)) in enumerate(
        zip(axes.flat, settings)
    ):
        x, y, alpha = _fuzzy_alpha_grid(tau_n, kappa, consequents, shape)
        last = axis.contourf(
            x, y, alpha, levels=np.linspace(0, 1, 11), vmin=0, vmax=1
        )
        axis.set_title(f"{title}: tau_N={tau_n:g}, kappa={kappa:g}", fontsize=9)
        axis.axvline(0, color="black", linewidth=0.9)
        if index >= 4:
            axis.set_xlabel("Exact-state count N(s)")
        else:
            axis.set_xlabel("")
        if index % 2 == 0:
            axis.set_ylabel("Neural TD-error proxy")
        else:
            axis.set_ylabel("")
    figure.colorbar(
        last, ax=axes.ravel().tolist(), label="Fuzzy memory weight alpha", shrink=0.82
    )
    figure.suptitle("Fuzzy gate sensitivity and zero-support behavior", fontsize=12)
    figure.savefig(
        FIGURES / "Figure_06_Fuzzy_Gate_Sensitivity.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def graphical_abstract() -> None:
    figure, axis = plt.subplots(figsize=(16, 6))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 6)
    axis.axis("off")
    colors = ("#e8f1fb", "#eee8f8", "#fff0e5", "#e8f6ee")
    titles = (
        "1  Neuro-memory agent",
        "2  Soft arbitration",
        "3  Deployment support shift",
        "4  Measured boundary",
    )
    for index, (color, title) in enumerate(zip(colors, titles)):
        x = 0.25 + index * 3.93
        panel = FancyBboxPatch(
            (x, 1.15),
            3.65,
            3.95,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#263238",
            linewidth=1.2,
        )
        axis.add_patch(panel)
        axis.text(
            x + 0.18,
            4.75,
            title,
            fontsize=12,
            fontweight="bold",
            color="#102a43",
        )

    axis.add_patch(
        Rectangle((0.65, 2.95), 1.25, 1.05, fill=False, linewidth=1.4)
    )
    axis.text(1.275, 3.48, "Exact-state\nmemory", ha="center", va="center")
    axis.add_patch(
        Rectangle((2.15, 2.95), 1.25, 1.05, fill=False, linewidth=1.4)
    )
    axis.text(2.775, 3.48, "Deep Q-\nnetwork", ha="center", va="center")
    axis.annotate(
        "",
        xy=(2.05, 2.5),
        xytext=(1.35, 2.95),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5},
    )
    axis.annotate(
        "",
        xy=(2.05, 2.5),
        xytext=(2.75, 2.95),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5},
    )
    axis.text(2.05, 2.25, "support-aware\naction values", ha="center")

    axis.text(4.55, 3.85, r"$u_N=1-\exp[-N(s)/\tau]$", fontsize=11)
    axis.text(4.55, 3.25, r"$u_E=\widehat{E}/(\widehat{E}+\kappa)$", fontsize=11)
    axis.text(
        4.55,
        2.5,
        "fuzzy rules\nlow / medium / high",
        ha="left",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "white"},
    )
    axis.annotate(
        r"$\alpha(s)\in[0,1]$",
        xy=(7.35, 2.7),
        xytext=(6.4, 2.7),
        arrowprops={"arrowstyle": "->"},
        va="center",
    )

    grid_x, grid_y = 8.25, 2.05
    for row in range(5):
        for column in range(5):
            face = "white"
            if (row, column) in {(1, 2), (2, 2), (3, 2)}:
                face = "#607d8b"
            if (row, column) in {(0, 4), (4, 4)}:
                face = "#ef5350"
            axis.add_patch(
                Rectangle(
                    (grid_x + 0.45 * column, grid_y + 0.45 * row),
                    0.45,
                    0.45,
                    facecolor=face,
                    edgecolor="#455a64",
                    linewidth=0.6,
                )
            )
    axis.text(8.35, 4.45, "train goals", color="#2e7d32")
    axis.text(9.9, 4.45, "held-out goals", color="#c62828")
    axis.text(
        10.55,
        1.55,
        "39.2% unsupported\nevaluation decisions",
        ha="center",
        fontweight="bold",
    )

    labels = ["DQN", "Fuzzy", "Abstain", "Tabular"]
    values = [-3.976, -3.401, -0.881, -1.102]
    y = np.asarray([3.75, 3.15, 2.55, 1.95])
    axis.barh(
        y,
        np.asarray(values) + 4.2,
        left=12.1,
        height=0.35,
        color=("#d62728", "#9467bd", "#17becf", "#1f77b4"),
    )
    for position, label, value in zip(y, labels, values):
        axis.text(11.9, position, label, ha="right", va="center")
        axis.text(
            15.55,
            position,
            f"AUC {value:.3f}",
            ha="right",
            va="center",
        )
    axis.text(
        13.9,
        4.25,
        "Fuzzy > DQN,\nbut < safe fallback / table",
        ha="center",
        fontweight="bold",
    )

    axis.text(
        8,
        0.55,
        "Boundary result: adaptive arbitration helps only when reliability "
        "signals are informative; explicit fallback remains necessary.",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#102a43",
    )
    axis.text(
        8,
        5.65,
        "Support-boundary diagnostic under exact-state support shift",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#102a43",
    )
    figure.tight_layout()
    GRAPHICAL_ABSTRACT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        GRAPHICAL_ABSTRACT
        / "Graphical_Abstract_Neuro_Memory_Support_Shift.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    application_figure()
    cost_support_figure()
    fuzzy_sensitivity_figure()
    graphical_abstract()
    print("Generated ASOC strong-revision figures.")


if __name__ == "__main__":
    main()
