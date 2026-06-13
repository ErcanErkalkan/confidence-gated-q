from __future__ import annotations

from pathlib import Path
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
PAPER_GENERATED = ROOT / "paper/generated"
PAPER_FIGURES = ROOT / "paper/figures"


def fuzzy_rule_base() -> None:
    rows = [
        ("Low", "Any", 0.00, "Prefer neural estimate; no memory evidence"),
        ("Medium", "Low", 0.35, "Mostly neural"),
        ("Medium", "High", 0.65, "Mostly memory"),
        ("High", "Low", 0.75, "Memory-led mixture"),
        ("High", "High", 0.95, "Strong memory preference"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "support_membership",
            "uncertainty_membership",
            "memory_weight",
            "interpretation",
        ],
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "table_fuzzy_rule_base.csv", index=False)
    latex = frame.to_latex(index=False, escape=True, column_format="llcl")
    (PAPER_GENERATED / "table_fuzzy_rule_base.tex").write_text(
        latex, encoding="utf-8"
    )


def fuzzy_memberships() -> None:
    support = np.linspace(0.0, 1.0, 401)
    low = np.clip(1.0 - 2.0 * support, 0.0, 1.0)
    medium = np.clip(1.0 - np.abs(2.0 * support - 1.0), 0.0, 1.0)
    high = np.clip(2.0 * support - 1.0, 0.0, 1.0)
    uncertainty = np.linspace(0.0, 1.0, 401)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(support, low, label="Low support")
    axes[0].plot(support, medium, label="Medium support")
    axes[0].plot(support, high, label="High support")
    axes[0].set_xlabel("Normalized support")
    axes[0].set_ylabel("Membership")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(uncertainty, 1.0 - uncertainty, label="Low uncertainty")
    axes[1].plot(uncertainty, uncertainty, label="High uncertainty")
    axes[1].set_xlabel("TD-residual uncertainty proxy")
    axes[1].set_ylabel("Membership")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "fig_fuzzy_memberships.pdf"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(output, PAPER_FIGURES / output.name)


def copy_submission_figures() -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for name in (
        "fig_strong_baselines.pdf",
        "fig_approx_support.pdf",
        "fig_fuzzy_ablation.pdf",
        "fig_application_tradeoff.pdf",
        "fig_uav_pybullet_validation.pdf",
        "fig_uav_pybullet_tradeoff.pdf",
        "fig_uav_sensorized_validation.pdf",
        "fig_uav_sensorized_tradeoff.pdf",
        "fig_fuzzy_reliability_stationary.pdf",
        "fig_fuzzy_reliability_shift.pdf",
    ):
        shutil.copy2(FIGURES / name, PAPER_FIGURES / name)


def submission_result_tables() -> None:
    specifications = (
        (
            TABLES / "table_strong_baselines.csv",
            "table_strong_baselines.tex",
        ),
        (
            TABLES / "table_approx_support.csv",
            "table_approx_support.tex",
        ),
        (
            TABLES / "table_fuzzy_ablation.csv",
            "table_fuzzy_ablation.tex",
        ),
        (
            TABLES / "table_application_risk_adjusted.csv",
            "table_application_risk_adjusted.tex",
        ),
        (
            TABLES / "table_uav_pybullet_validation.csv",
            "table_uav_pybullet_validation.tex",
        ),
        (
            TABLES / "table_uav_sensorized_validation.csv",
            "table_uav_sensorized_validation.tex",
        ),
        (
            TABLES / "table_fuzzy_reliability_stationary.csv",
            "table_fuzzy_reliability_stationary.tex",
        ),
        (
            TABLES / "table_fuzzy_reliability_shift.csv",
            "table_fuzzy_reliability_shift.tex",
        ),
    )
    for source, target in specifications:
        frame = pd.read_csv(source)
        compact = frame[
            [
                "environment",
                "method",
                "metric",
                "n_seeds",
                "mean",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            ]
        ].copy()
        compact["estimate"] = compact.apply(
            lambda row: (
                f"{row['mean']:.3f} "
                f"[{row['bootstrap_ci_low']:.3f}, "
                f"{row['bootstrap_ci_high']:.3f}]"
            ),
            axis=1,
        )
        columns = ["method", "metric", "n_seeds", "estimate"]
        if frame["environment"].nunique() > 1:
            columns.insert(0, "environment")
        compact = compact[columns]
        latex = compact.to_latex(
            index=False,
            escape=True,
            column_format="lllrl" if len(columns) == 5 else "llrl",
        )
        (PAPER_GENERATED / target).write_text(latex, encoding="utf-8")


def benchmark_status() -> None:
    rows = []
    legacy = (
        "dqn_tuning_development",
        "dqn_strong_validation",
        "confirmatory_extended_compact",
        "support_abstention_replication",
        "minigrid_extended_diagnostic",
        "application_navigation_case_study",
        "adaptive_gate_compact_validation",
        "cost_support_metrics",
    )
    for name in legacy:
        audit = json.loads(
            (ROOT / "results" / name / "audit.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append((name, "executed main/diagnostic", audit["observed_runs"]))
    additions = (
        ("strong neural baselines", "main confirmatory", 60),
        ("approximate support", "main confirmatory", 60),
        ("fuzzy component ablation", "main confirmatory", 120),
        ("application fallback ablation", "main confirmatory", 120),
        (
            "physics-based state-accessible UAV benchmark",
            "main simulator benchmark",
            150,
        ),
        (
            "sensorized low-level-control UAV SIL",
            "main sensorized SIL validation",
            120,
        ),
        ("fuzzy reliability stationary", "independent confirmatory", 540),
        ("reliability shift", "independent confirmatory", 360),
    )
    rows.extend(additions)
    frame = pd.DataFrame(
        rows, columns=["Result family", "Evidence class", "Runs"]
    )
    latex = frame.to_latex(index=False, escape=True, column_format="llr")
    (PAPER_GENERATED / "asoc_benchmark_status.tex").write_text(
        latex, encoding="utf-8"
    )


def main() -> None:
    fuzzy_rule_base()
    fuzzy_memberships()
    copy_submission_figures()
    submission_result_tables()
    benchmark_status()
    print("Generated submission figures and result tables.")


if __name__ == "__main__":
    main()
