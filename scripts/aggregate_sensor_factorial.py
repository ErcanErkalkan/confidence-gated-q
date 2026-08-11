from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.statistics import empirical_lower_cvar  # noqa: E402
from scripts.aggregate_sensor_aliasing import (  # noqa: E402
    DEFAULT_LATENT_RADIUS,
    DEFAULT_MATERIAL_LATENT_DISTANCE,
    DEFAULT_MAX_PAIRS,
    DEFAULT_OBSERVATION_RADIUS,
    aliasing_rows,
    fragmentation_rows,
    validate_trace_schema,
)
from scripts.run_sensor_factorial_development import (  # noqa: E402
    CONDITION_FACTORS,
    DEFAULT_CONFIG,
    EXPECTED_AGENTS,
    validate_factorial_config,
)


OUTPUT_DIR = ROOT / "results/diagnostic_extensions/sensor_factorial_development"
TABLE_PATH = ROOT / "tables/table_sensor_factorial.csv"
FIGURE_PATH = ROOT / "figures/fig_sensor_factorial.pdf"
CONDITION_ORDER = list(CONDITION_FACTORS)
AGENT_ORDER = [
    "dqn",
    "fuzzy_relative_reliability",
    "selected_approximate_support",
    "sensorized_controller",
]
SEED_METRICS = [
    "success_rate",
    "failure_rate",
    "collision_rate",
    "localization_error_mean",
    "exact_support_coverage",
    "approximate_support_coverage",
    "memory_branch_usage_ratio",
    "neural_branch_usage_ratio",
    "seed_lower_tail_return_0_10",
    "seed_minimum_return",
]


class SensorFactorialAggregationError(ValueError):
    pass


def _finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def build_seed_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, agent, seed), group in evaluation.groupby(
        ["environment", "agent", "seed"], sort=True
    ):
        returns = pd.to_numeric(group["return"], errors="coerce").to_numpy(
            dtype=float
        )
        finite_returns = returns[np.isfinite(returns)]
        if finite_returns.size != 4:
            raise SensorFactorialAggregationError(
                f"{condition}/{agent}/{seed} has {finite_returns.size} eval returns"
            )
        rows.append(
            {
                "condition": condition,
                "agent": agent,
                "seed": int(seed),
                "success_rate": _finite_mean(group["success"]),
                "failure_rate": _finite_mean(group["failure_rate"]),
                "collision_rate": _finite_mean(group["collision_rate"]),
                "localization_error_mean": _finite_mean(
                    group["localization_error_mean"]
                ),
                "exact_support_coverage": _finite_mean(
                    group["exact_support_coverage"]
                ),
                "approximate_support_coverage": _finite_mean(
                    group["approximate_support_coverage"]
                ),
                "memory_branch_usage_ratio": _finite_mean(
                    group["memory_branch_usage_ratio"]
                ),
                "neural_branch_usage_ratio": _finite_mean(
                    group["neural_branch_usage_ratio"]
                ),
                "seed_lower_tail_return_0_10": empirical_lower_cvar(
                    finite_returns, 0.10
                ),
                "seed_minimum_return": float(finite_returns.min()),
                "evaluation_episode_rows": int(finite_returns.size),
                "source_file": (
                    "results/diagnostic_extensions/"
                    "sensor_factorial_development/raw.csv"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 64 or result.duplicated(
        ["condition", "agent", "seed"]
    ).any():
        raise SensorFactorialAggregationError("seed metric grain mismatch")
    return result


def _trace_metrics(trace_dir: Path) -> pd.DataFrame:
    paths = sorted(trace_dir.glob("*.csv.gz"))
    if len(paths) != 64:
        raise SensorFactorialAggregationError(
            f"trace shard count {len(paths)} != 64"
        )
    traces = pd.concat(
        [pd.read_csv(path, compression="gzip") for path in paths],
        ignore_index=True,
    )
    validate_trace_schema(traces)
    exposure = (
        traces.groupby(
            ["environment", "agent", "seed", "phase"], sort=True
        )
        .agg(
            trace_row_count=("step", "size"),
            effective_latency_mean=("effective_latency", "mean"),
            localization_dropout_rate=("localization_dropout", "mean"),
            range_dropout_rate=("range_dropout", "mean"),
            camera_dropout_rate=("camera_dropout", "mean"),
            target_visibility_rate=("target_visibility", "mean"),
        )
        .reset_index()
        .rename(columns={"environment": "condition"})
    )
    fragmentation = fragmentation_rows(
        traces,
        latent_radius=DEFAULT_LATENT_RADIUS,
        max_pairs=DEFAULT_MAX_PAIRS,
    ).rename(columns={"environment": "condition"})
    aliasing = aliasing_rows(
        traces,
        observation_radius=DEFAULT_OBSERVATION_RADIUS,
        material_latent_distance=DEFAULT_MATERIAL_LATENT_DISTANCE,
        max_pairs=DEFAULT_MAX_PAIRS,
    ).rename(columns={"environment": "condition"})
    keys = ["condition", "agent", "seed", "phase"]
    merged = fragmentation.merge(aliasing, on=keys, validate="one_to_one")
    return merged.merge(exposure, on=keys, validate="one_to_one")


def build_summary(
    seed_metrics: pd.DataFrame,
    trace_metrics: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        for agent in AGENT_ORDER:
            seeds = seed_metrics[
                seed_metrics["condition"].eq(condition)
                & seed_metrics["agent"].eq(agent)
            ]
            trace = trace_metrics[
                trace_metrics["condition"].eq(condition)
                & trace_metrics["agent"].eq(agent)
            ]
            episodes = evaluation[
                evaluation["environment"].eq(condition)
                & evaluation["agent"].eq(agent)
            ]
            if len(seeds) != 2 or len(trace) != 2 or len(episodes) != 8:
                raise SensorFactorialAggregationError(
                    f"incomplete summary group {condition}/{agent}"
                )
            similar_latent = int(trace["similar_latent_pair_count"].sum())
            fragmented = int(trace["different_exact_key_pair_count"].sum())
            similar_observation = int(
                trace["similar_observation_pair_count"].sum()
            )
            aliased = int(trace["aliased_pair_count"].sum())
            returns = pd.to_numeric(episodes["return"], errors="coerce").to_numpy(
                dtype=float
            )
            row: dict[str, Any] = {
                "condition": condition,
                "agent": agent,
                "n_seeds": 2,
                "evaluation_episode_rows": len(returns),
            }
            row.update({metric: _finite_mean(seeds[metric]) for metric in SEED_METRICS})
            row.update(
                {
                    "episode_lower_tail_return_0_10": empirical_lower_cvar(
                        returns, 0.10
                    ),
                    "fragmentation_rate": (
                        fragmented / similar_latent if similar_latent else np.nan
                    ),
                    "fragmentation_denominator": similar_latent,
                    "aliasing_rate": (
                        aliased / similar_observation
                        if similar_observation
                        else np.nan
                    ),
                    "aliasing_denominator": similar_observation,
                    "effective_latency_mean": _finite_mean(
                        trace["effective_latency_mean"]
                    ),
                    "localization_dropout_rate": _finite_mean(
                        trace["localization_dropout_rate"]
                    ),
                    "range_dropout_rate": _finite_mean(
                        trace["range_dropout_rate"]
                    ),
                    "camera_dropout_rate": _finite_mean(
                        trace["camera_dropout_rate"]
                    ),
                    "target_visibility_rate": _finite_mean(
                        trace["target_visibility_rate"]
                    ),
                    "source_file": (
                        "results/diagnostic_extensions/"
                        "sensor_factorial_development/raw.csv;"
                        "results/diagnostic_extensions/"
                        "sensor_factorial_development/trace_shards/*.csv.gz"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_effects(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[
        summary["condition"].eq("no_noise_no_delay_no_dropout")
    ].set_index("agent")
    metrics = [
        "success_rate",
        "failure_rate",
        "collision_rate",
        "localization_error_mean",
        "exact_support_coverage",
        "approximate_support_coverage",
        "memory_branch_usage_ratio",
        "neural_branch_usage_ratio",
        "episode_lower_tail_return_0_10",
        "fragmentation_rate",
        "aliasing_rate",
        "effective_latency_mean",
        "localization_dropout_rate",
        "range_dropout_rate",
        "camera_dropout_rate",
        "target_visibility_rate",
    ]
    rows = []
    for _, row in summary.iterrows():
        reference = baseline.loc[row["agent"]]
        result = {"condition": row["condition"], "agent": row["agent"]}
        for metric in metrics:
            result[f"delta_{metric}_vs_baseline"] = (
                float(row[metric] - reference[metric])
                if np.isfinite(row[metric]) and np.isfinite(reference[metric])
                else np.nan
            )
        rows.append(result)
    return pd.DataFrame(rows)


def build_manipulation_audit(summary: pd.DataFrame) -> pd.DataFrame:
    condition = summary.groupby("condition", sort=False).agg(
        localization_error_mean=("localization_error_mean", "mean"),
        effective_latency_mean=("effective_latency_mean", "mean"),
        localization_dropout_rate=("localization_dropout_rate", "mean"),
        range_dropout_rate=("range_dropout_rate", "mean"),
        camera_dropout_rate=("camera_dropout_rate", "mean"),
        target_visibility_rate=("target_visibility_rate", "mean"),
    )
    rows = []
    for name in CONDITION_ORDER:
        values = condition.loc[name]
        if name == "no_noise_no_delay_no_dropout":
            checks = {
                "noise_effect": values["localization_error_mean"] <= 1e-12,
                "latency_effect": values["effective_latency_mean"] <= 1e-12,
                "localization_dropout_effect": values[
                    "localization_dropout_rate"
                ] <= 1e-12,
                "range_dropout_effect": values["range_dropout_rate"] <= 1e-12,
                "camera_dropout_effect": values["camera_dropout_rate"] <= 1e-12,
                "visibility_effect": values["target_visibility_rate"] >= 1.0,
            }
        else:
            checks = {
                "noise_effect": (
                    values["localization_error_mean"] > 0.0
                    if name in {"noise_only", "combined_executed_condition"}
                    else True
                ),
                "latency_effect": (
                    values["effective_latency_mean"] > 0.0
                    if name in {
                        "latency_only",
                        "localization_dropout_only",
                        "combined_executed_condition",
                    }
                    else True
                ),
                "localization_dropout_effect": (
                    values["localization_dropout_rate"] > 0.0
                    if name in {
                        "localization_dropout_only",
                        "combined_executed_condition",
                    }
                    else values["localization_dropout_rate"] <= 1e-12
                ),
                "range_dropout_effect": (
                    values["range_dropout_rate"] > 0.0
                    if name in {"range_dropout_only", "combined_executed_condition"}
                    else values["range_dropout_rate"] <= 1e-12
                ),
                "camera_dropout_effect": (
                    values["camera_dropout_rate"] > 0.0
                    if name in {"camera_dropout_only", "combined_executed_condition"}
                    else values["camera_dropout_rate"] <= 1e-12
                ),
                "visibility_effect": (
                    values["target_visibility_rate"] < 1.0
                    if name in {
                        "visibility_occlusion_only",
                        "camera_dropout_only",
                        "combined_executed_condition",
                    }
                    else values["target_visibility_rate"] >= 1.0
                ),
            }
        rows.append(
            {
                "condition": name,
                **{key: bool(value) for key, value in checks.items()},
                **{key: float(value) for key, value in values.items()},
                "audit_status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_metric_manifest() -> pd.DataFrame:
    source_raw = (
        "results/diagnostic_extensions/sensor_factorial_development/raw.csv"
    )
    source_trace = (
        "results/diagnostic_extensions/sensor_factorial_development/"
        "trace_shards/*.csv.gz"
    )
    definitions = [
        (
            "success_rate",
            "mean episode success indicator",
            "higher",
            "episode",
            "evaluation episodes",
            source_raw,
        ),
        (
            "failure_rate",
            "mean one-minus-success indicator",
            "lower",
            "episode",
            "evaluation episodes",
            source_raw,
        ),
        (
            "collision_rate",
            "mean per-episode collision-step fraction",
            "lower",
            "episode",
            "evaluation steps",
            source_raw,
        ),
        (
            "localization_error_mean",
            "mean Euclidean latent-versus-estimated position error",
            "lower",
            "decision",
            "evaluation decisions",
            source_raw,
        ),
        (
            "exact_support_coverage",
            "mean exact-key support indicator",
            "higher",
            "decision",
            "evaluation decisions",
            source_raw,
        ),
        (
            "approximate_support_coverage",
            "mean approximate-support indicator",
            "higher",
            "decision",
            "evaluation decisions",
            source_raw,
        ),
        (
            "fragmentation_rate",
            "similar latent-state pairs with different exact keys",
            "lower",
            "unordered_pair",
            "pairs within latent radius 0.10",
            source_trace,
        ),
        (
            "aliasing_rate",
            "similar observations with materially different latent state or optimal label",
            "lower",
            "unordered_pair",
            "pairs within observation radius 0.15",
            source_trace,
        ),
        (
            "memory_branch_usage_ratio",
            "mean memory-branch decision weight",
            "descriptive",
            "decision",
            "evaluation decisions",
            source_raw,
        ),
        (
            "neural_branch_usage_ratio",
            "mean neural-branch decision weight",
            "descriptive",
            "decision",
            "evaluation decisions",
            source_raw,
        ),
        (
            "episode_lower_tail_return_0_10",
            "arithmetic mean of the lowest max(1, ceil(0.10*n)) finite episode returns",
            "higher",
            "episode",
            "eight evaluation episodes per condition-agent",
            source_raw,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "definition": definition,
                "direction": direction,
                "aggregation_level": level,
                "denominator": denominator,
                "evidence_class": "development_diagnostic",
                "source_file": source,
            }
            for name, definition, direction, level, denominator, source in definitions
        ]
    )


def build_budget_audit(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        for agent in AGENT_ORDER:
            group = raw[
                raw["environment"].eq(condition) & raw["agent"].eq(agent)
            ]
            evaluation = group[group["phase"].eq("eval")]
            seeds = sorted(group["seed"].astype(int).unique())
            checkpoints = sorted(evaluation["checkpoint"].astype(int).unique())
            counts = evaluation.groupby(["seed", "checkpoint"]).size()
            status = (
                seeds == [15002, 15003]
                and checkpoints == [40, 80]
                and counts.eq(2).all()
                and int(group["environment_steps"].max()) == 80
            )
            rows.append(
                {
                    "condition": condition,
                    "agent": agent,
                    "seed_set": ";".join(map(str, seeds)),
                    "training_steps": int(group["environment_steps"].max()),
                    "checkpoint_schedule": ";".join(map(str, checkpoints)),
                    "evaluation_episodes_per_checkpoint": (
                        int(counts.min()) if len(counts) else 0
                    ),
                    "cpu_threads": int(config["runtime"]["torch_threads"]),
                    "audit_status": "PASS" if status else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def render_figure(effects: pd.DataFrame, destination: Path) -> None:
    condition_labels = {
        "no_noise_no_delay_no_dropout": "Baseline",
        "noise_only": "Noise only",
        "latency_only": "Latency only",
        "localization_dropout_only": "Localization dropout",
        "range_dropout_only": "Range dropout",
        "camera_dropout_only": "Camera dropout",
        "visibility_occlusion_only": "Visibility/occlusion",
        "combined_executed_condition": "Combined executed",
    }
    agent_labels = {
        "dqn": "DQN",
        "fuzzy_relative_reliability": "Fuzzy reliability",
        "selected_approximate_support": "Approx. support",
        "sensorized_controller": "Model-based controller",
    }
    panels = [
        (
            "delta_localization_error_mean_vs_baseline",
            "Localization-error change",
            ".3f",
        ),
        (
            "delta_approximate_support_coverage_vs_baseline",
            "Approximate-support coverage change",
            ".2f",
        ),
        (
            "delta_fragmentation_rate_vs_baseline",
            "Fragmentation-rate change",
            ".2f",
        ),
        (
            "delta_episode_lower_tail_return_0_10_vs_baseline",
            "Lower-tail return change",
            ".2f",
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11.0))
    figure.subplots_adjust(
        left=0.13,
        right=0.94,
        bottom=0.13,
        top=0.86,
        hspace=0.48,
        wspace=0.43,
    )
    for axis, (metric, title, number_format) in zip(axes.flat, panels):
        matrix = (
            effects.pivot(index="condition", columns="agent", values=metric)
            .reindex(index=CONDITION_ORDER, columns=AGENT_ORDER)
            .to_numpy(dtype=float)
        )
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit = max(float(finite.max()) if finite.size else 0.0, 1e-9)
        image = axis.imshow(matrix, cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xticks(range(len(AGENT_ORDER)))
        axis.set_xticklabels(
            [agent_labels[item] for item in AGENT_ORDER], rotation=28, ha="right"
        )
        axis.set_yticks(range(len(CONDITION_ORDER)))
        axis.set_yticklabels([condition_labels[item] for item in CONDITION_ORDER])
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                label = "NA" if not np.isfinite(value) else format(value, number_format)
                tone = "white" if np.isfinite(value) and abs(value) > 0.55 * limit else "#222222"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=tone,
                )
        axis.tick_params(length=0, labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.suptitle(
        "Sensor-factor development diagnostics",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.943,
        "Mean change from matched no-noise/no-delay/no-dropout baseline; two development seeds",
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )
    figure.text(
        0.01,
        0.005,
        "Descriptive development evidence. Positive values mean more of the named metric; lower-tail return uses empirical CVaR_0.10.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="pdf", bbox_inches="tight")
    plt.close(figure)


def aggregate(
    output_dir: Path = OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG,
    table_path: Path = TABLE_PATH,
    figure_path: Path = FIGURE_PATH,
) -> dict[str, pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    condition_manifest = validate_factorial_config(config)
    raw = pd.read_csv(output_dir / "raw.csv")
    evaluation = raw[raw["phase"].eq("eval")].copy()
    if set(evaluation["environment"]) != set(CONDITION_ORDER):
        raise SensorFactorialAggregationError("evaluation condition coverage mismatch")
    if set(evaluation["agent"]) != EXPECTED_AGENTS:
        raise SensorFactorialAggregationError("evaluation agent coverage mismatch")
    if evaluation["seed"].astype(int).between(16000, 16099).any():
        raise SensorFactorialAggregationError("reserved final seed row detected")
    seed_metrics = build_seed_metrics(evaluation)
    trace_metrics = _trace_metrics(output_dir / "trace_shards")
    summary = build_summary(seed_metrics, trace_metrics, evaluation)
    effects = build_effects(summary)
    budget = build_budget_audit(raw, config)
    manipulation = build_manipulation_audit(summary)
    metric_manifest = build_metric_manifest()
    if not budget["audit_status"].eq("PASS").all():
        raise SensorFactorialAggregationError("matched budget audit failed")
    if not condition_manifest["audit_status"].eq("PASS").all():
        raise SensorFactorialAggregationError("factor isolation audit failed")
    if not manipulation["audit_status"].eq("PASS").all():
        raise SensorFactorialAggregationError("factor manipulation audit failed")
    merged_table = summary.merge(
        effects, on=["condition", "agent"], validate="one_to_one"
    )
    outputs = {
        "seed_metrics.csv": seed_metrics,
        "trace_pair_metrics.csv": trace_metrics,
        "sensor_factor_summary.csv": summary,
        "factor_effects.csv": effects,
        "budget_match_audit.csv": budget,
        "factor_isolation_audit.csv": condition_manifest,
        "manipulation_check_audit.csv": manipulation,
        "metric_manifest.csv": metric_manifest,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    merged_table.to_csv(table_path, index=False)
    render_figure(effects, figure_path)
    receipt = {
        "status": "PASS",
        "evidence_class": "development_diagnostic",
        "conditions": len(CONDITION_ORDER),
        "agents": len(AGENT_ORDER),
        "development_seeds": [15002, 15003],
        "seed_metric_rows": len(seed_metrics),
        "summary_rows": len(summary),
        "budget_rows_passed": int(budget["audit_status"].eq("PASS").sum()),
        "factor_isolation_rows_passed": int(
            condition_manifest["audit_status"].eq("PASS").sum()
        ),
        "manipulation_rows_passed": int(
            manipulation["audit_status"].eq("PASS").sum()
        ),
        "final_seed_rows": 0,
    }
    (output_dir / "aggregation_audit.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return {**outputs, "table": merged_table}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()
    outputs = aggregate(args.output_dir, args.config, args.table, args.figure)
    print(
        "SENSOR_FACTORIAL_AGGREGATION_PASS "
        f"seed_rows={len(outputs['seed_metrics.csv'])} "
        f"summary_rows={len(outputs['sensor_factor_summary.csv'])}"
    )


if __name__ == "__main__":
    main()
