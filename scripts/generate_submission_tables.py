from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GENERATED = ROOT / "paper" / "generated"

NAMES = {
    "tabular": "Tabular Q",
    "validated_dqn": "Validated DQN",
    "fixed_hybrid": "Fixed mixture",
    "count_gated_tau_20": "Count gate",
    "support_abstain_tau_20": "Support abstention",
    "fuzzy_support_adaptive": "Fuzzy adaptive",
}

ENV_NAMES = {
    "ApplicationNavigation-deployment-goal-shift": "Application navigation",
    "ApplicationNavigation-cost-support": "Application navigation",
    "FourRooms-9-cost-support": "FourRooms 9 support shift",
    "CliffWalking-adaptive": "CliffWalking",
    "FourRooms-9-support-shift-adaptive": "FourRooms 9 support shift",
    "FrozenLake-4x4-slippery-adaptive": "FrozenLake 4x4",
}


def summary(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name / "summary.csv")


def metric(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["metric"] == name].set_index(
        ["environment", "agent"]
    )


def write_latex(frame: pd.DataFrame, name: str, column_format: str) -> None:
    text = frame.to_latex(
        index=False,
        escape=True,
        column_format=column_format,
    )
    (GENERATED / name).write_text(text, encoding="utf-8")


def interval(row: pd.Series) -> str:
    return (
        f"{row['mean']:.3f} "
        f"[{row['bootstrap_ci_low']:.3f}, "
        f"{row['bootstrap_ci_high']:.3f}]"
    )


def application_table() -> None:
    frame = summary("application_navigation_case_study")
    auc = metric(frame, "return_auc")
    success = metric(frame, "success_rate")
    collision = metric(frame, "collision_rate")
    unsupported = metric(frame, "unsupported_state_ratio")
    abstention = metric(frame, "abstention_ratio")
    rows = []
    for key, row in auc.iterrows():
        environment, agent = key
        rows.append(
            {
                "Method": NAMES[agent],
                "Return AUC [95% bootstrap CI]": interval(row),
                "Success": f"{success.loc[key, 'mean']:.3f}",
                "Failure": f"{1.0 - success.loc[key, 'mean']:.3f}",
                "Collision": f"{collision.loc[key, 'mean']:.3f}",
                "Unsupported": f"{unsupported.loc[key, 'mean']:.3f}",
                "Abstention": f"{abstention.loc[key, 'mean']:.3f}",
            }
        )
    write_latex(
        pd.DataFrame(rows),
        "asoc_application_case_study.tex",
        "lcccccc",
    )


def adaptive_table() -> None:
    frame = summary("adaptive_gate_compact_validation")
    auc = frame[frame["metric"] == "return_auc"].copy()
    auc["Environment"] = auc["environment"].map(ENV_NAMES)
    auc["Method"] = auc["agent"].map(NAMES)
    auc["Return AUC [95% bootstrap CI]"] = auc.apply(interval, axis=1)
    output = auc[
        ["Environment", "Method", "Return AUC [95% bootstrap CI]"]
    ]
    write_latex(
        output,
        "asoc_adaptive_gate_summary.tex",
        "lll",
    )


def cost_table() -> None:
    frame = summary("cost_support_metrics")
    metrics = {
        name: metric(frame, name)
        for name in (
            "unsupported_state_ratio",
            "memory_branch_usage_ratio",
            "neural_branch_usage_ratio",
            "abstention_ratio",
            "memory_cost_states",
            "memory_cost_bytes_estimated",
            "inference_time_us_per_decision_mean",
        )
    }
    rows = []
    for key, row in metrics["unsupported_state_ratio"].iterrows():
        environment, agent = key
        rows.append(
            {
                "Environment": ENV_NAMES[environment],
                "Method": NAMES[agent],
                "Unsupported": f"{row['mean']:.3f}",
                "Memory branch": (
                    f"{metrics['memory_branch_usage_ratio'].loc[key, 'mean']:.3f}"
                ),
                "Neural branch": (
                    f"{metrics['neural_branch_usage_ratio'].loc[key, 'mean']:.3f}"
                ),
                "Abstention": (
                    f"{metrics['abstention_ratio'].loc[key, 'mean']:.3f}"
                ),
                "Stored states": (
                    f"{metrics['memory_cost_states'].loc[key, 'mean']:.1f}"
                ),
                "Estimated KiB": (
                    f"{metrics['memory_cost_bytes_estimated'].loc[key, 'mean'] / 1024:.2f}"
                ),
                "Decision time (us)": (
                    f"{metrics['inference_time_us_per_decision_mean'].loc[key, 'mean']:.1f}"
                ),
            }
        )
    write_latex(
        pd.DataFrame(rows),
        "asoc_cost_and_support_metrics.tex",
        "llccccccc",
    )


def claims_registry() -> None:
    app_summary = summary("application_navigation_case_study")
    app_contrasts = pd.read_csv(
        RESULTS
        / "application_navigation_case_study"
        / "planned_contrasts.csv"
    )
    adaptive = pd.read_csv(
        RESULTS
        / "adaptive_gate_compact_validation"
        / "planned_contrasts.csv"
    )
    rows = []
    for _, row in app_contrasts.iterrows():
        rows.append(
            {
                "claim_id": f"app_{row['contrast']}",
                "value": row["mean_difference"],
                "ci_low": row["bootstrap_ci_low"],
                "ci_high": row["bootstrap_ci_high"],
                "source": (
                    "results/application_navigation_case_study/"
                    "planned_contrasts.csv"
                ),
                "filter": row["contrast"],
            }
        )
    for agent in NAMES:
        selected = app_summary[
            (app_summary["agent"] == agent)
            & (
                app_summary["metric"].isin(
                    (
                        "return_auc",
                        "success_rate",
                        "unsupported_state_ratio",
                        "inference_time_us_per_decision_mean",
                    )
                )
            )
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "claim_id": f"app_{agent}_{row['metric']}",
                    "value": row["mean"],
                    "ci_low": row["bootstrap_ci_low"],
                    "ci_high": row["bootstrap_ci_high"],
                    "source": (
                        "results/application_navigation_case_study/"
                        "summary.csv"
                    ),
                    "filter": f"agent={agent};metric={row['metric']}",
                }
            )
    for _, row in adaptive.iterrows():
        rows.append(
            {
                "claim_id": (
                    f"adaptive_{row['environment']}_{row['contrast']}"
                ),
                "value": row["mean_difference"],
                "ci_low": row["bootstrap_ci_low"],
                "ci_high": row["bootstrap_ci_high"],
                "source": (
                    "results/adaptive_gate_compact_validation/"
                    "planned_contrasts.csv"
                ),
                "filter": (
                    f"environment={row['environment']};"
                    f"contrast={row['contrast']}"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        GENERATED / "verified_claims.csv", index=False
    )


def benchmark_status() -> None:
    families = (
        "dqn_tuning_development",
        "dqn_strong_validation",
        "confirmatory_extended_compact",
        "support_abstention_replication",
        "minigrid_extended_diagnostic",
        "application_navigation_case_study",
        "adaptive_gate_compact_validation",
        "cost_support_metrics",
    )
    rows = []
    for name in families:
        config = json.loads(
            (ROOT / "configs" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        audit = json.loads(
            (RESULTS / name / "audit.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "Result family": name.replace("_", " "),
                "Status": audit["status"],
                "Environments": len(config["envs"]),
                "Methods": len(config["agents"]),
                "Runs": audit["observed_runs"],
                "Analysis": config["analysis"]["analysis_status"].replace(
                    "_", " "
                ),
            }
        )
    write_latex(
        pd.DataFrame(rows),
        "asoc_benchmark_status.tex",
        "llrrrl",
    )


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    application_table()
    adaptive_table()
    cost_table()
    claims_registry()
    benchmark_status()
    print("Generated submission tables and claims registry.")


if __name__ == "__main__":
    main()
