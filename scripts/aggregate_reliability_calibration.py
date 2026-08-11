from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.calibration import (  # noqa: E402
    average_precision,
    balanced_accuracy_at_half,
    binary_auroc,
    brier_score,
    first_sustained_detection,
)


DEVELOPMENT_SEED_MIN = 10000
DEVELOPMENT_SEED_MAX = 10099
FINAL_SEED_MIN = 12000
FINAL_SEED_MAX = 12099
BOOTSTRAP_SAMPLES = 2000
BIN_EDGES = np.linspace(0.0, 1.0, 11)
COVERAGES = np.linspace(1.0, 0.1, 10)
TARGETS = {
    "action_correctness": {
        "branch_column": "actually_better_branch",
        "definition": (
            "Evaluation rows where exactly one branch greedy action equals the "
            "analytic optimal action; target=1 means tabular is correct and "
            "neural is incorrect. Ties are excluded."
        ),
    },
    "value_error": {
        "branch_column": "actually_better_branch_value",
        "definition": (
            "Evaluation rows with analytic full-action Q*: target=1 means the "
            "tabular Q-vector RMSE is lower than the neural Q-vector RMSE. "
            "Numerical ties are excluded."
        ),
    },
}
SCOPES = {
    "pre_shift_all": lambda frame: frame[frame["post_shift"] < 0.5],
    "post_shift_all": lambda frame: frame[frame["post_shift"] >= 0.5],
    "post_shift_changed_region": lambda frame: frame[
        (frame["post_shift"] >= 0.5)
        & (frame["shift_region"] == "changed_optimal_action")
    ],
}


class ReliabilityCalibrationError(ValueError):
    pass


def _bootstrap_mean_ci(
    values: list[float], *, salt: str
) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(array.mean())
    if array.size == 1:
        return point, float("nan"), float("nan"), 1
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(BOOTSTRAP_SAMPLES, array.size), replace=True)
    means = sampled.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return point, float(low), float(high), int(array.size)


def _metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    scores = frame["relative_reliability_score"].to_numpy(dtype=float)
    targets = frame["binary_target"].to_numpy(dtype=int)
    return {
        "auroc": binary_auroc(scores, targets),
        "average_precision": average_precision(scores, targets),
        "balanced_accuracy_0_5": balanced_accuracy_at_half(scores, targets),
        "brier_score": brier_score(scores, targets),
    }


def _raw_paths(input_dir: Path) -> list[Path]:
    parts = sorted((input_dir / "raw_parts").glob("*.csv*"))
    if parts:
        return parts
    for name in ("raw.csv", "raw.csv.gz", "raw.csv.xz"):
        path = input_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"No raw result found under {input_dir}")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_and_validate_evaluation(
    input_dir: Path, config_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_seeds = {int(seed) for seed in config["seeds"]}
    registry_key = config.get("analysis", {}).get(
        "seed_registry_key", "reliability_calibration_development"
    )
    if registry_key == "reliability_calibration_development":
        if not configured_seeds or not all(
            DEVELOPMENT_SEED_MIN <= seed <= DEVELOPMENT_SEED_MAX
            for seed in configured_seeds
        ):
            raise ReliabilityCalibrationError(
                "config seeds are outside reliability_calibration_development"
            )
        if any(
            FINAL_SEED_MIN <= seed <= FINAL_SEED_MAX
            for seed in configured_seeds
        ):
            raise ReliabilityCalibrationError("reserved final seed detected")
    elif registry_key == "new_shift_final":
        expected = set(range(12090, 12100))
        if configured_seeds != expected:
            raise ReliabilityCalibrationError(
                "independent focal calibration must use seeds 12090-12099"
            )
    else:
        raise ReliabilityCalibrationError(
            f"unsupported reliability calibration seed registry: {registry_key}"
        )
    diagnostic_columns = [
        "tabular_greedy_action",
        "neural_greedy_action",
        "mixed_selected_action",
        "optimal_action",
        "tabular_action_correct",
        "neural_action_correct",
        "relative_reliability_score",
        "predicted_better_branch",
        "actually_better_branch",
        "actually_better_branch_value",
        "tabular_q_error",
        "neural_q_error",
        "post_shift",
        "shift_region",
        "steps_since_shift",
    ]
    needed = {
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        *diagnostic_columns,
    }
    frames = []
    for path in _raw_paths(input_dir):
        frame = pd.read_csv(path, usecols=lambda column: column in needed)
        frame["source_file"] = _display_path(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise ReliabilityCalibrationError(f"raw diagnostics missing: {missing}")
    training = raw[raw["phase"] == "train"]
    if not training[diagnostic_columns].isna().all().all():
        raise ReliabilityCalibrationError(
            "reliability diagnostics must be blank on training rows"
        )
    evaluation = raw[raw["phase"] == "eval"].copy()
    key = ["environment", "agent", "seed", "checkpoint", "episode"]
    if evaluation.duplicated(key).any():
        raise ReliabilityCalibrationError("duplicate evaluation diagnostic rows")
    observed_seeds = {int(seed) for seed in evaluation["seed"].unique()}
    if observed_seeds != configured_seeds:
        raise ReliabilityCalibrationError(
            f"seed coverage mismatch: observed={sorted(observed_seeds)}"
        )
    if registry_key == "reliability_calibration_development" and any(
        FINAL_SEED_MIN <= seed <= FINAL_SEED_MAX for seed in observed_seeds
    ):
        raise ReliabilityCalibrationError("reserved final result row detected")
    score = pd.to_numeric(
        evaluation["relative_reliability_score"], errors="coerce"
    )
    if score.isna().any() or ((score < 0.0) | (score > 1.0)).any():
        raise ReliabilityCalibrationError("reliability score is missing or invalid")
    for branch in ("tabular", "neural"):
        action = pd.to_numeric(
            evaluation[f"{branch}_greedy_action"], errors="coerce"
        )
        correct = pd.to_numeric(
            evaluation[f"{branch}_action_correct"], errors="coerce"
        )
        expected = (action == evaluation["optimal_action"]).astype(float)
        if action.isna().any() or not np.array_equal(correct.to_numpy(), expected.to_numpy()):
            raise ReliabilityCalibrationError(
                f"{branch} action-correctness diagnostic is inconsistent"
            )
        q_error = pd.to_numeric(evaluation[f"{branch}_q_error"], errors="coerce")
        if q_error.isna().any() or (q_error < 0.0).any():
            raise ReliabilityCalibrationError(f"{branch} analytic Q error is invalid")
    allowed_branches = {"tabular", "neural", "tie"}
    for column in ("actually_better_branch", "actually_better_branch_value"):
        observed = set(evaluation[column].astype(str))
        if not observed.issubset(allowed_branches):
            raise ReliabilityCalibrationError(
                f"{column} contains invalid labels: {sorted(observed)}"
            )

    params = {
        spec["name"]: {
            "beta": float(spec["params"]["reliability_beta"]),
            "lambda": float(spec["params"]["reliability_prior_strength"]),
            "epsilon": float(spec["params"]["reliability_epsilon"]),
        }
        for spec in config["agents"]
    }
    if set(evaluation["agent"].unique()) != set(params):
        raise ReliabilityCalibrationError("agent coverage differs from locked design")
    for name in ("beta", "lambda", "epsilon"):
        evaluation[name] = evaluation["agent"].map(
            {agent: values[name] for agent, values in params.items()}
        )
    return evaluation, config


def _target_frame(
    frame: pd.DataFrame, target_type: str
) -> pd.DataFrame:
    branch_column = TARGETS[target_type]["branch_column"]
    result = frame[frame[branch_column].isin(["tabular", "neural"])].copy()
    result["binary_target"] = (result[branch_column] == "tabular").astype(int)
    result["prediction_correct"] = (
        result["predicted_better_branch"] == result[branch_column]
    ).astype(int)
    return result


def build_branch_discrimination(evaluation: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["environment", "agent", "beta", "lambda", "epsilon"]
    rows: list[dict[str, Any]] = []
    for scope, scope_filter in SCOPES.items():
        scoped = scope_filter(evaluation)
        for target_type, target_spec in TARGETS.items():
            informative = _target_frame(scoped, target_type)
            for keys, group in informative.groupby(group_columns, sort=True):
                seed_metrics = [
                    _metric_bundle(seed_group)
                    for _, seed_group in group.groupby("seed")
                ]
                row = dict(zip(group_columns, keys))
                row.update(
                    {
                        "scope": scope,
                        "target_type": target_type,
                        "target_definition": target_spec["definition"],
                        "positive_class": "tabular_better",
                        "score_definition": (
                            "neural residual error divided by tabular plus neural "
                            "residual error plus epsilon; higher predicts tabular"
                        ),
                        "aggregation_level": "mean_of_seed_metrics",
                        "n_rows": len(group),
                        "n_seeds": group["seed"].nunique(),
                        "n_positive": int(group["binary_target"].sum()),
                        "n_negative": int((1 - group["binary_target"]).sum()),
                    }
                )
                for metric in (
                    "auroc",
                    "average_precision",
                    "balanced_accuracy_0_5",
                    "brier_score",
                ):
                    point, low, high, available = _bootstrap_mean_ci(
                        [item[metric] for item in seed_metrics],
                        salt=f"{keys}-{scope}-{target_type}-{metric}",
                    )
                    row[metric] = point
                    row[f"{metric}_ci_low"] = low
                    row[f"{metric}_ci_high"] = high
                    row[f"{metric}_available_seed_count"] = available
                row["auroc_availability"] = (
                    "available"
                    if row["n_positive"] > 0
                    and row["n_negative"] > 0
                    and row["auroc_available_seed_count"] > 0
                    else "not_available_single_class"
                )
                row["source_file"] = group["source_file"].iloc[0]
                rows.append(row)
    return pd.DataFrame(rows)


def build_calibration_bins(
    evaluation: pd.DataFrame,
    scope: str = "post_shift_changed_region",
) -> pd.DataFrame:
    if scope not in SCOPES:
        raise ReliabilityCalibrationError(f"unknown calibration scope: {scope}")
    group_columns = ["environment", "agent", "beta", "lambda", "epsilon"]
    rows = []
    scoped = SCOPES[scope](evaluation)
    for target_type, target_spec in TARGETS.items():
        informative = _target_frame(scoped, target_type)
        informative["calibration_bin"] = pd.cut(
            informative["relative_reliability_score"],
            bins=BIN_EDGES,
            include_lowest=True,
            labels=False,
        )
        for keys, group in informative.groupby(group_columns, sort=True):
            for bin_index in range(10):
                selected = group[group["calibration_bin"] == bin_index]
                if selected.empty:
                    continue
                seed_rates = [
                    float(seed_group["binary_target"].mean())
                    for _, seed_group in selected.groupby("seed")
                ]
                observed, low, high, available = _bootstrap_mean_ci(
                    seed_rates,
                    salt=f"bins-{keys}-{target_type}-{bin_index}",
                )
                mean_score = float(selected["relative_reliability_score"].mean())
                rows.append(
                    {
                        **dict(zip(group_columns, keys)),
                        "scope": scope,
                        "target_type": target_type,
                        "target_definition": target_spec["definition"],
                        "aggregation_level": "evaluation_rows_clustered_by_seed",
                        "bin_index": bin_index,
                        "bin_low": BIN_EDGES[bin_index],
                        "bin_high": BIN_EDGES[bin_index + 1],
                        "n_rows": len(selected),
                        "n_seeds": selected["seed"].nunique(),
                        "mean_score": mean_score,
                        "observed_tabular_better_rate": observed,
                        "observed_rate_ci_low": low,
                        "observed_rate_ci_high": high,
                        "calibration_gap": abs(mean_score - observed),
                        "bootstrap_available_seed_count": available,
                        "source_file": selected["source_file"].iloc[0],
                    }
                )
    return pd.DataFrame(rows)


def build_selective_risk(
    evaluation: pd.DataFrame,
    scope: str = "post_shift_changed_region",
) -> pd.DataFrame:
    if scope not in SCOPES:
        raise ReliabilityCalibrationError(f"unknown selective-risk scope: {scope}")
    group_columns = ["environment", "agent", "beta", "lambda", "epsilon"]
    rows = []
    scoped = SCOPES[scope](evaluation)
    for target_type, target_spec in TARGETS.items():
        informative = _target_frame(scoped, target_type)
        informative["confidence"] = (
            2.0 * (informative["relative_reliability_score"] - 0.5).abs()
        )
        for keys, group in informative.groupby(group_columns, sort=True):
            for requested_coverage in COVERAGES:
                seed_risks = []
                retained = 0
                total = 0
                for _, seed_group in group.groupby("seed"):
                    ordered = seed_group.sort_values(
                        "confidence", ascending=False, kind="mergesort"
                    )
                    keep = max(1, math.ceil(requested_coverage * len(ordered)))
                    selected = ordered.head(keep)
                    seed_risks.append(1.0 - float(selected["prediction_correct"].mean()))
                    retained += len(selected)
                    total += len(ordered)
                risk, low, high, available = _bootstrap_mean_ci(
                    seed_risks,
                    salt=f"risk-{keys}-{target_type}-{requested_coverage}",
                )
                rows.append(
                    {
                        **dict(zip(group_columns, keys)),
                        "scope": scope,
                        "target_type": target_type,
                        "target_definition": target_spec["definition"],
                        "aggregation_level": "mean_seed_selective_error_rate",
                        "requested_coverage": requested_coverage,
                        "realized_coverage": retained / total if total else float("nan"),
                        "retained_rows": retained,
                        "total_informative_rows": total,
                        "selective_risk": risk,
                        "selective_risk_ci_low": low,
                        "selective_risk_ci_high": high,
                        "bootstrap_available_seed_count": available,
                        "source_file": group["source_file"].iloc[0],
                    }
                )
    return pd.DataFrame(rows)


def build_detection_delay(
    evaluation: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    shift_by_environment = {
        spec.get("name", spec["id"]): int(spec["kwargs"]["shift_after"])
        for spec in config["envs"]
    }
    scoped = SCOPES["post_shift_changed_region"](evaluation)
    informative = _target_frame(scoped, "action_correctness")
    rows = []
    group_columns = [
        "environment",
        "agent",
        "beta",
        "lambda",
        "epsilon",
        "seed",
    ]
    for keys, group in informative.groupby(group_columns, sort=True):
        checkpoint = (
            group.groupby("checkpoint")["prediction_correct"]
            .agg(["sum", "count"])
            .reset_index()
        )
        shift_step = shift_by_environment[str(keys[0])]
        first, delay = first_sustained_detection(
            checkpoint["checkpoint"],
            checkpoint["sum"],
            checkpoint["count"],
            shift_step=shift_step,
            minimum_rows=5,
            consecutive_checkpoints=2,
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "target_type": "action_correctness",
                "target_definition": TARGETS["action_correctness"]["definition"],
                "aggregation_level": "seed",
                "detection_rule": (
                    "First of two consecutive post-shift evaluation checkpoints "
                    "with >0.5 prediction accuracy and >=5 informative changed-region rows"
                ),
                "shift_step": shift_step,
                "first_correct_detection_step": first,
                "adaptation_delay_steps": delay,
                "detected": bool(np.isfinite(first)),
                "last_observed_checkpoint": int(group["checkpoint"].max()),
                "source_file": group["source_file"].iloc[0],
            }
        )
    result = pd.DataFrame(rows)
    summary_keys = ["environment", "agent", "beta", "lambda", "epsilon"]
    summaries = {}
    for keys, group in result.groupby(summary_keys):
        point, low, high, available = _bootstrap_mean_ci(
            group["adaptation_delay_steps"].tolist(), salt=f"delay-{keys}"
        )
        summaries[keys] = {
            "detection_rate": float(group["detected"].mean()),
            "adaptation_delay_mean": point,
            "adaptation_delay_ci_low": low,
            "adaptation_delay_ci_high": high,
            "detected_seed_count": available,
        }
    for index, row in result.iterrows():
        key = tuple(row[column] for column in summary_keys)
        for column, value in summaries[key].items():
            result.at[index, column] = value
    return result


def build_parameter_sensitivity(
    branch: pd.DataFrame,
    bins: pd.DataFrame,
    delay: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["environment", "agent", "beta", "lambda", "epsilon"]
    primary = branch[branch["scope"] == "post_shift_all"].copy()
    wide = primary.pivot(index=keys, columns="target_type").reset_index()
    wide.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    changed = branch[branch["scope"] == "post_shift_changed_region"].copy()
    changed = changed.pivot(index=keys, columns="target_type").reset_index()
    changed.columns = [
        (
            "changed_region_"
            + "_".join(str(item) for item in column if str(item))
        )
        if isinstance(column, tuple) and len(column) > 1 and str(column[1])
        else (str(column[0]) if isinstance(column, tuple) else str(column))
        for column in changed.columns
    ]
    calibration = (
        bins.assign(weighted_gap=bins["calibration_gap"] * bins["n_rows"])
        .groupby(keys + ["target_type"])
        .apply(
            lambda group: group["weighted_gap"].sum() / group["n_rows"].sum(),
            include_groups=False,
        )
        .rename("expected_calibration_error")
        .reset_index()
        .pivot(index=keys, columns="target_type", values="expected_calibration_error")
        .add_prefix("ece_")
        .reset_index()
    )
    delay_summary = (
        delay.groupby(keys)
        .agg(
            detection_rate=("detected", "mean"),
            adaptation_delay_mean=("adaptation_delay_steps", "mean"),
            adaptation_delay_median=("adaptation_delay_steps", "median"),
        )
        .reset_index()
    )
    result = (
        wide.merge(changed, on=keys, how="left")
        .merge(calibration, on=keys, how="left")
        .merge(delay_summary, on=keys, how="left")
    )
    result["design_class"] = "predeclared_half_fraction_development"
    result["aggregation_level"] = "environment_parameter_cell"
    result["source_file"] = branch["source_file"].iloc[0]
    action_column = "auroc_action_correctness"
    result["development_rank_within_environment"] = result.groupby(
        "environment"
    )[action_column].rank(method="min", ascending=False)
    return result.sort_values(
        ["environment", "development_rank_within_environment", "agent"]
    )


def build_table(sensitivity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "environment",
        "agent",
        "beta",
        "lambda",
        "epsilon",
        "auroc_action_correctness",
        "auroc_ci_low_action_correctness",
        "auroc_ci_high_action_correctness",
        "average_precision_action_correctness",
        "balanced_accuracy_0_5_action_correctness",
        "brier_score_action_correctness",
        "changed_region_auroc_action_correctness",
        "changed_region_auroc_availability_action_correctness",
        "changed_region_brier_score_action_correctness",
        "ece_action_correctness",
        "auroc_value_error",
        "auroc_ci_low_value_error",
        "auroc_ci_high_value_error",
        "brier_score_value_error",
        "changed_region_auroc_value_error",
        "changed_region_brier_score_value_error",
        "ece_value_error",
        "detection_rate",
        "adaptation_delay_median",
        "development_rank_within_environment",
        "aggregation_level",
        "source_file",
    ]
    return sensitivity[columns]


def make_figure(
    branch: pd.DataFrame,
    bins: pd.DataFrame,
    selective: pd.DataFrame,
    delay: pd.DataFrame,
    output: Path,
) -> None:
    reference = "rel_b005_l5_e1e8"
    primary_action = branch[
        (branch["scope"] == "post_shift_all")
        & (branch["target_type"] == "action_correctness")
    ]
    beta_summary = (
        primary_action.groupby("beta")["auroc"].mean().sort_index()
    )
    reference_bins = bins[bins["agent"] == reference]
    reference_risk = selective[selective["agent"] == reference]

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    blue = "#3568A8"
    gold = "#C58A18"
    ink = "#30343B"
    grid = "#D8DCE2"

    axes[0, 0].bar(
        [f"{value:.2f}" for value in beta_summary.index],
        beta_summary.values,
        color=blue,
        edgecolor=ink,
        linewidth=0.6,
    )
    axes[0, 0].axhline(0.5, color=ink, linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Post-shift action discrimination by beta")
    axes[0, 0].set_xlabel("Residual EMA beta")
    axes[0, 0].set_ylabel("Mean seed-level AUROC")

    for target_type, color, marker in (
        ("action_correctness", blue, "o"),
        ("value_error", gold, "s"),
    ):
        selected = reference_bins[reference_bins["target_type"] == target_type]
        aggregated = selected.groupby("bin_index").agg(
            mean_score=("mean_score", "mean"),
            observed=("observed_tabular_better_rate", "mean"),
        )
        axes[0, 1].plot(
            aggregated["mean_score"],
            aggregated["observed"],
            marker=marker,
            color=color,
            label=target_type.replace("_", " "),
        )
    axes[0, 1].plot([0, 1], [0, 1], color=ink, linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Calibration of the reference cell")
    axes[0, 1].set_xlabel("Mean predicted tabular-better score")
    axes[0, 1].set_ylabel("Observed tabular-better rate")
    axes[0, 1].legend(frameon=False)

    for target_type, color, marker in (
        ("action_correctness", blue, "o"),
        ("value_error", gold, "s"),
    ):
        selected = reference_risk[reference_risk["target_type"] == target_type]
        aggregated = selected.groupby("requested_coverage")["selective_risk"].mean()
        axes[1, 0].plot(
            aggregated.index,
            aggregated.values,
            marker=marker,
            color=color,
            label=target_type.replace("_", " "),
        )
    axes[1, 0].set_xlim(0.1, 1.0)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Selective risk of the reference cell")
    axes[1, 0].set_xlabel("Retained coverage")
    axes[1, 0].set_ylabel("Prediction error rate")
    axes[1, 0].legend(frameon=False)

    delay_groups = [
        delay.loc[delay["beta"] == beta, "adaptation_delay_steps"].dropna()
        for beta in sorted(delay["beta"].unique())
    ]
    axes[1, 1].boxplot(
        delay_groups,
        tick_labels=[f"{value:.2f}" for value in sorted(delay["beta"].unique())],
        patch_artist=True,
        boxprops={"facecolor": "#D9E5F3", "edgecolor": ink},
        medianprops={"color": gold, "linewidth": 1.5},
        whiskerprops={"color": ink},
        capprops={"color": ink},
        flierprops={"markeredgecolor": ink, "marker": "."},
    )
    axes[1, 1].set_title("Post-shift adaptation-delay distribution")
    axes[1, 1].set_xlabel("Residual EMA beta")
    axes[1, 1].set_ylabel("Delay (training steps)")

    for axis in axes.flat:
        axis.grid(axis="y", color=grid, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Development-only residual reliability calibration\n"
        "Two analytic one-step shift severities; seeds 10000-10004; "
        "reference cell beta=0.05, lambda=5, epsilon=1e-8",
        fontsize=12,
        color=ink,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_reliability_calibration(
    input_dir: Path,
    config_path: Path,
    output_dir: Path,
    table_path: Path,
    figure_path: Path,
) -> dict[str, int]:
    evaluation, config = load_and_validate_evaluation(input_dir, config_path)
    branch = build_branch_discrimination(evaluation)
    bins = build_calibration_bins(evaluation)
    selective = build_selective_risk(evaluation)
    delay = build_detection_delay(evaluation, config)
    sensitivity = build_parameter_sensitivity(branch, bins, delay)
    table = build_table(sensitivity)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    branch.to_csv(output_dir / "branch_discrimination.csv", index=False)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    selective.to_csv(output_dir / "selective_risk.csv", index=False)
    sensitivity.to_csv(output_dir / "parameter_sensitivity.csv", index=False)
    delay.to_csv(output_dir / "detection_delay.csv", index=False)
    table.to_csv(table_path, index=False)
    make_figure(branch, bins, selective, delay, figure_path)
    return {
        "evaluation_rows": len(evaluation),
        "branch_rows": len(branch),
        "calibration_bin_rows": len(bins),
        "selective_risk_rows": len(selective),
        "parameter_rows": len(sensitivity),
        "delay_seed_rows": len(delay),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate development-only relative-reliability calibration."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results/reviewer1_remaining/reliability_calibration/development",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/reviewer1_remaining/reliability_calibration/development_fractional.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/reviewer1_remaining/reliability_calibration",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "tables/table_reliability_calibration.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures/fig_reliability_calibration.pdf",
    )
    args = parser.parse_args()
    counts = aggregate_reliability_calibration(
        args.input_dir,
        args.config,
        args.output_dir,
        args.table,
        args.figure,
    )
    print("RELIABILITY_CALIBRATION_PASS " + " ".join(
        f"{key}={value}" for key, value in counts.items()
    ))


if __name__ == "__main__":
    main()
