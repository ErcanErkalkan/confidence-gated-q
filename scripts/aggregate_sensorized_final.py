from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.statistics import (  # noqa: E402
    bootstrap_mean_interval,
    cohen_dz,
    empirical_lower_cvar,
    holm_adjust,
    lower_quantile,
    paired_rank_biserial,
    win_loss_tie,
    worst_decile_mean,
)
from hybrid_q.trace_evidence import TRACE_SCHEMA_VERSION  # noqa: E402
from scripts.aggregate_sensor_aliasing import (  # noqa: E402
    aliasing_rows,
    fragmentation_rows,
    support_correctness_rows,
    validate_trace_schema,
)
from scripts.lock_sensor_final_protocol import (  # noqa: E402
    EXPECTED_AGENTS,
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    load_and_validate,
    sha256,
)


OUTPUT_DIR = ROOT / "results/diagnostic_extensions/sensorized_final"
TABLE_PATH = ROOT / "tables/table_sensorized_final.csv"
FIGURE_PATH = ROOT / "figures/fig_sensorized_final.pdf"
PROTOCOL_PATH = (
    ROOT / "configs/diagnostic_extensions/sensorized_final/protocol_lock.yaml"
)
CONFIG_PATH = ROOT / "configs/diagnostic_extensions/sensorized_final/matched_final.yaml"
AGENT_ORDER = [
    "feed_forward_dqn",
    "selected_temporal_drqn",
    "fuzzy_relative_reliability",
    "selected_approximate_support",
    "sensorized_controller",
]
CONDITION_ORDER = ["combined_executed_condition", "latency_only"]
GROUP = ["environment", "agent", "seed", "phase"]
EPISODE_GROUP = [*GROUP, "checkpoint", "episode"]


class SensorizedFinalAggregationError(ValueError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uncompressed_size(path: Path) -> int:
    total = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
    return total


def load_traces(trace_paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifests = []
    for path in sorted(trace_paths):
        frame = pd.read_csv(path, compression="gzip")
        validate_trace_schema(frame)
        if set(frame["trace_schema_version"].astype(str)) != {TRACE_SCHEMA_VERSION}:
            raise SensorizedFinalAggregationError(f"non-v2 final trace: {path}")
        frames.append(frame)
        manifests.append(
            {
                "source_file": path.relative_to(ROOT).as_posix(),
                "sha256": _file_sha256(path),
                "compressed_bytes": path.stat().st_size,
                "uncompressed_bytes": _uncompressed_size(path),
                "trace_rows": len(frame),
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "environment": str(frame["environment"].iloc[0]),
                "agent": str(frame["agent"].iloc[0]),
                "seed": int(frame["seed"].iloc[0]),
                "phase_set": ";".join(sorted(set(frame["phase"].astype(str)))),
                "outcome_metadata_complete": bool(
                    frame[
                        [
                            "post_action_trajectory_error",
                            "post_action_motor_saturation",
                            "post_action_failure_stage",
                        ]
                    ].notna().all().all()
                ),
            }
        )
    if len(frames) != 300:
        raise SensorizedFinalAggregationError(
            f"expected 300 trace shards, found {len(frames)}"
        )
    traces = pd.concat(frames, ignore_index=True)
    return traces, pd.DataFrame(manifests)


def build_episode_outcomes(
    traces: pd.DataFrame, *, seconds_per_step: float = 1.0 / 60.0
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, group in traces.groupby(EPISODE_GROUP, sort=True):
        group = group.sort_values("step")
        first = group.iloc[0]
        last = group.iloc[-1]
        initial_position = first[
            ["latent_position_x", "latent_position_y", "latent_position_z"]
        ].to_numpy(dtype=float)
        target = first[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
        initial_error = float(np.linalg.norm(initial_position - target))
        trajectory = pd.to_numeric(
            group["post_action_trajectory_error"], errors="coerce"
        ).to_numpy(dtype=float)
        constraint = pd.to_numeric(
            group["post_action_constraint_active"], errors="coerce"
        ).to_numpy(dtype=int)
        saturation = pd.to_numeric(
            group["post_action_saturation_active"], errors="coerce"
        ).to_numpy(dtype=int)
        visibility = pd.to_numeric(
            group["post_action_target_visibility"], errors="coerce"
        ).to_numpy(dtype=int)
        recovery = pd.to_numeric(
            group["post_action_recovery_event"], errors="coerce"
        ).to_numpy(dtype=int)
        if not np.isfinite(trajectory).all():
            raise SensorizedFinalAggregationError("non-finite trajectory outcome")
        stage = str(last["post_action_failure_stage"])
        if stage == "ongoing":
            raise SensorizedFinalAggregationError("episode ended with ongoing stage")
        rows.append(
            {
                **dict(zip(EPISODE_GROUP, identity)),
                "trace_steps": len(group),
                "initial_trajectory_error": initial_error,
                "trajectory_error_mean": float(trajectory.mean()),
                "trajectory_error_min": float(trajectory.min()),
                "trajectory_error_max": float(trajectory.max()),
                "trajectory_error_final": float(trajectory[-1]),
                "recovery_event_count": int(recovery.sum()),
                "recovery_event_occurred": int(recovery.sum() > 0),
                "constraint_steps": int(constraint.sum()),
                "constraint_duration_seconds": float(constraint.sum() * seconds_per_step),
                "saturation_steps": int(saturation.sum()),
                "saturation_duration_seconds": float(saturation.sum() * seconds_per_step),
                "target_visibility_fraction": float(visibility.mean()),
                "target_visible_any": int(visibility.any()),
                "target_visible_all": int(visibility.all()),
                "episode_success": int(last["post_action_success"]),
                "failure_stage": stage,
                "source_file": (
                    "results/diagnostic_extensions/sensorized_final/trace_shards/*.csv.gz"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(EPISODE_GROUP).any():
        raise SensorizedFinalAggregationError("episode trace grain mismatch")
    return result


def build_pair_metrics(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fragmentation = fragmentation_rows(traces).rename(
        columns={"status": "fragmentation_status", "reason_if_unavailable": "fragmentation_reason"}
    )
    aliasing = aliasing_rows(traces).rename(
        columns={"status": "aliasing_status", "reason_if_unavailable": "aliasing_reason"}
    )
    pair_metrics = fragmentation.merge(aliasing, on=GROUP, validate="one_to_one")
    correctness = support_correctness_rows(traces)
    return pair_metrics, correctness


def _weighted_correctness(group: pd.DataFrame) -> tuple[float, int, str, str]:
    available = group[group["status"].eq("available")]
    denominator = int(available["eligible_decision_count"].sum())
    if not denominator:
        return np.nan, 0, "unavailable", "not_available"
    numerator = float(
        (available["correctness_rate"] * available["eligible_decision_count"]).sum()
    )
    targets = sorted(set(available["correctness_target"].astype(str)))
    statuses = sorted(set(available["inferential_status"].astype(str)))
    return numerator / denominator, denominator, ";".join(targets), ";".join(statuses)


def build_sensor_seed_metrics(
    standard_seed_metrics: pd.DataFrame,
    episode_outcomes: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    correctness: pd.DataFrame,
) -> pd.DataFrame:
    base = standard_seed_metrics.rename(
        columns={"environment": "condition", "return_auc": "normalized_return_auc"}
    ).copy()
    if len(base) != 300:
        raise SensorizedFinalAggregationError("standard seed metric coverage mismatch")
    episode = (
        episode_outcomes.groupby(["environment", "agent", "seed"], sort=True)
        .agg(
            trajectory_error_mean=("trajectory_error_mean", "mean"),
            trajectory_error_final_mean=("trajectory_error_final", "mean"),
            recovery_events=("recovery_event_count", "sum"),
            recovery_episode_rate=("recovery_event_occurred", "mean"),
            constraint_duration_seconds_mean=("constraint_duration_seconds", "mean"),
            saturation_duration_seconds_mean=("saturation_duration_seconds", "mean"),
            target_visible_episode_any_rate=("target_visible_any", "mean"),
            target_visible_episode_all_rate=("target_visible_all", "mean"),
        )
        .reset_index()
        .rename(columns={"environment": "condition"})
    )
    pair = pair_metrics.rename(columns={"environment": "condition"}).copy()
    pair = pair[pair["phase"].eq("eval")]
    correct_rows = []
    for identity, group in correctness[correctness["phase"].eq("eval")].groupby(
        ["environment", "agent", "seed"], sort=True
    ):
        rate, denominator, target, status = _weighted_correctness(group)
        correct_rows.append(
            {
                "condition": identity[0],
                "agent": identity[1],
                "seed": identity[2],
                "support_correctness_rate": rate,
                "support_correctness_denominator": denominator,
                "support_correctness_target": target,
                "support_correctness_status": status,
            }
        )
    correct = pd.DataFrame(correct_rows)
    keys = ["condition", "agent", "seed"]
    result = base.merge(episode, on=keys, validate="one_to_one")
    result = result.merge(
        pair[
            keys
            + [
                "similar_latent_pair_count",
                "different_exact_key_pair_count",
                "fragmentation_rate",
                "similar_observation_pair_count",
                "aliased_pair_count",
                "aliasing_rate",
            ]
        ],
        on=keys,
        validate="one_to_one",
    )
    result = result.merge(correct, on=keys, validate="one_to_one")
    result["source_file"] = (
        "results/diagnostic_extensions/sensorized_final/seed_metrics.csv;"
        "results/diagnostic_extensions/sensorized_final/trace_shards/*.csv.gz"
    )
    return result.sort_values(keys).reset_index(drop=True)


def build_summary(
    seed_metrics: pd.DataFrame,
    raw: pd.DataFrame,
    episode_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    final_eval = raw[
        raw["phase"].eq("eval") & raw["checkpoint"].eq(240)
    ]
    rows = []
    for condition in CONDITION_ORDER:
        for agent in AGENT_ORDER:
            seeds = seed_metrics[
                seed_metrics["condition"].eq(condition) & seed_metrics["agent"].eq(agent)
            ]
            episodes = raw[
                raw["phase"].eq("eval")
                & raw["environment"].eq(condition)
                & raw["agent"].eq(agent)
            ]
            final = final_eval[
                final_eval["environment"].eq(condition) & final_eval["agent"].eq(agent)
            ]
            trace_episodes = episode_outcomes[
                episode_outcomes["environment"].eq(condition)
                & episode_outcomes["agent"].eq(agent)
            ]
            if len(seeds) != 30 or len(episodes) != 240 or len(trace_episodes) != 240:
                raise SensorizedFinalAggregationError(f"incomplete summary {condition}/{agent}")
            seed_return = seeds["normalized_return_auc"].to_numpy(dtype=float)
            episode_return = episodes["return"].to_numpy(dtype=float)
            similar_latent = int(seeds["similar_latent_pair_count"].sum())
            fragmented = int(seeds["different_exact_key_pair_count"].sum())
            similar_observation = int(seeds["similar_observation_pair_count"].sum())
            aliased = int(seeds["aliased_pair_count"].sum())
            correct_denominator = int(seeds["support_correctness_denominator"].sum())
            correct_rate = (
                float(
                    np.nansum(
                        seeds["support_correctness_rate"]
                        * seeds["support_correctness_denominator"]
                    )
                    / correct_denominator
                )
                if correct_denominator
                else np.nan
            )
            stages = trace_episodes["failure_stage"].value_counts()
            rows.append(
                {
                    "condition": condition,
                    "agent": agent,
                    "n_seeds": 30,
                    "evaluation_episode_rows": 240,
                    "mean_normalized_return_auc": float(seed_return.mean()),
                    "median_normalized_return_auc": float(np.median(seed_return)),
                    "seed_minimum_normalized_return_auc": float(seed_return.min()),
                    "seed_q05_normalized_return_auc": lower_quantile(seed_return, 0.05),
                    "seed_worst_decile_mean_normalized_return_auc": worst_decile_mean(seed_return),
                    "seed_cvar_0_10_normalized_return_auc": empirical_lower_cvar(seed_return, 0.10),
                    "seed_cvar_0_05_normalized_return_auc": empirical_lower_cvar(seed_return, 0.05),
                    "episode_minimum_return": float(episode_return.min()),
                    "episode_q05_return": lower_quantile(episode_return, 0.05),
                    "episode_cvar_0_10_return": empirical_lower_cvar(episode_return, 0.10),
                    "episode_cvar_0_05_return": empirical_lower_cvar(episode_return, 0.05),
                    "final_checkpoint_success_rate": float(final["success"].mean()),
                    "final_checkpoint_failure_probability": float(1.0 - final["success"].mean()),
                    "final_checkpoint_collision_rate": float(final["collision_rate"].mean()),
                    "final_checkpoint_risk_zone_rate": float(final["risk_zone_rate"].mean()),
                    "final_checkpoint_motor_saturation_rate": float(final["motor_saturation_rate"].mean()),
                    "trajectory_error_mean": float(trace_episodes["trajectory_error_mean"].mean()),
                    "trajectory_error_final_mean": float(trace_episodes["trajectory_error_final"].mean()),
                    "recovery_event_count": int(trace_episodes["recovery_event_count"].sum()),
                    "recovery_episode_rate": float(trace_episodes["recovery_event_occurred"].mean()),
                    "constraint_duration_seconds_mean": float(trace_episodes["constraint_duration_seconds"].mean()),
                    "saturation_duration_seconds_mean": float(trace_episodes["saturation_duration_seconds"].mean()),
                    "target_visible_episode_any_rate": float(trace_episodes["target_visible_any"].mean()),
                    "target_visible_episode_all_rate": float(trace_episodes["target_visible_all"].mean()),
                    "fragmentation_rate": fragmented / similar_latent if similar_latent else np.nan,
                    "fragmentation_denominator": similar_latent,
                    "aliasing_rate": aliased / similar_observation if similar_observation else np.nan,
                    "aliasing_denominator": similar_observation,
                    "support_correctness_rate": correct_rate,
                    "support_correctness_denominator": correct_denominator,
                    "support_correctness_target": (
                        "nominal_free_space_reference_action_proxy"
                        if correct_denominator
                        else "not_available"
                    ),
                    "dominant_failure_stage": str(stages.index[0]),
                    "dominant_failure_stage_count": int(stages.iloc[0]),
                    "zero_success_retained": bool(final["success"].sum() == 0),
                    "source_file": (
                        "results/diagnostic_extensions/sensorized_final/raw.csv;"
                        "results/diagnostic_extensions/sensorized_final/trace_shards/*.csv.gz"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _planned_row(
    condition: str, contrast: dict[str, Any], seed_metrics: pd.DataFrame
) -> dict[str, Any]:
    part = seed_metrics[seed_metrics["condition"].eq(condition)]
    left = part[part["agent"].eq(contrast["left"])][["seed", "normalized_return_auc"]]
    right = part[part["agent"].eq(contrast["right"])][["seed", "normalized_return_auc"]]
    paired = left.merge(right, on="seed", suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) != 30:
        raise SensorizedFinalAggregationError("planned contrast is not 30-seed paired")
    differences = (
        paired["normalized_return_auc_left"] - paired["normalized_return_auc_right"]
    ).to_numpy(dtype=float)
    if np.allclose(differences, 0.0):
        paired_t_p = wilcoxon_p = 1.0
    else:
        paired_t_p = (
            0.0
            if float(differences.std(ddof=1)) <= 1e-12
            else float(stats.ttest_1samp(differences, 0.0).pvalue)
        )
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    low, high = bootstrap_mean_interval(differences, seed=0, samples=10000)
    wins, losses, ties = win_loss_tie(differences)
    return {
        "report_family": "sensorized_final_eight_rows",
        "evidence_class": "replication",
        "environment": condition,
        "contrast": contrast["name"],
        "contrast_status": contrast["status"],
        "left": contrast["left"],
        "right": contrast["right"],
        "metric": "normalized_return_auc",
        "n_pairs": 30,
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "cohen_dz": cohen_dz(differences),
        "rank_biserial": paired_rank_biserial(differences),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "source_file": "results/diagnostic_extensions/sensorized_final/sensor_seed_metrics.csv",
        "source_column": "normalized_return_auc",
    }


def build_planned_contrasts(
    seed_metrics: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    contrasts = [
        {
            "name": item["contrast_id"],
            "status": item["status"],
            "left": item["left"],
            "right": item["right"],
        }
        for item in protocol["analysis"]["planned_contrasts"]
    ]
    rows = [_planned_row(condition, contrast, seed_metrics) for condition in CONDITION_ORDER for contrast in contrasts]
    if len(rows) != 8:
        raise SensorizedFinalAggregationError("Holm family must contain eight rows")
    paired = holm_adjust([row["paired_t_p"] for row in rows])
    wilcoxon = holm_adjust([row["wilcoxon_p"] for row in rows])
    for row, t_value, w_value in zip(rows, paired, wilcoxon):
        row["paired_t_holm_p"] = t_value
        row["wilcoxon_holm_p"] = w_value
        row["report_holm_scope"] = "all_eight_rows_together"
    return pd.DataFrame(rows)


def build_budget_audit(
    seed_metrics: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    seed_set = ";".join(map(str, EXPECTED_SEEDS))
    for condition in CONDITION_ORDER:
        part = seed_metrics[seed_metrics["condition"].eq(condition)]
        for agent in AGENT_ORDER:
            group = part[part["agent"].eq(agent)]
            updates = ";".join(
                f"{int(row.seed)}:{int(row.gradient_updates)}"
                for row in group.sort_values("seed").itertuples()
            )
            rows.append(
                {
                    "condition": condition,
                    "agent": agent,
                    "seed_set": seed_set,
                    "seed_sets_equal": True,
                    "training_steps": 240,
                    "training_steps_equal": True,
                    "checkpoint_schedule": "120;240",
                    "checkpoint_schedule_equal": True,
                    "evaluation_episodes_per_checkpoint": 4,
                    "evaluation_episodes_equal": True,
                    "observed_seed_runs": len(group),
                    "gradient_updates_by_seed": updates,
                    "interaction_budget_matched": len(group) == 30,
                    "compute_budget_identical": False,
                    "compute_identity_reason": protocol["budget"]["compute_identity_reason"],
                    "audit_status": "PASS" if len(group) == 30 else "FAIL",
                    "source_file": "results/diagnostic_extensions/sensorized_final/sensor_seed_metrics.csv",
                }
            )
    return pd.DataFrame(rows)


def build_table(summary: pd.DataFrame, planned: pd.DataFrame) -> pd.DataFrame:
    summary_rows = summary.copy()
    summary_rows.insert(0, "record_type", "agent_summary")
    contrast_rows = planned.copy()
    contrast_rows.insert(0, "record_type", "planned_contrast")
    return pd.concat([summary_rows, contrast_rows], ignore_index=True, sort=False)


def plot_figure(summary: pd.DataFrame, planned: pd.DataFrame, path: Path) -> None:
    """Static chart contract: interval forest plus condition-coded lower tail."""

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), constrained_layout=True)
    ink = "#252525"
    blue = "#2F6B9A"
    orange = "#D28B26"
    condition_style = {
        "combined_executed_condition": (blue, "o", "Combined"),
        "latency_only": (orange, "s", "Latency only"),
    }

    forest = planned.copy().reset_index(drop=True)
    labels = [
        f"{row.contrast.replace('_vs_', ' vs ').replace('_', ' ')}\n{condition_style[row.environment][2]}"
        for row in forest.itertuples()
    ]
    y = np.arange(len(forest))[::-1]
    for index, row in forest.iterrows():
        color, marker, _ = condition_style[row["environment"]]
        axes[0].errorbar(
            row["mean_difference"],
            y[index],
            xerr=np.asarray(
                [[row["mean_difference"] - row["bootstrap_ci_low"]], [row["bootstrap_ci_high"] - row["mean_difference"]]]
            ),
            fmt=marker,
            color=color,
            markerfacecolor=(color if marker == "o" else "white"),
            markeredgecolor=color,
            capsize=2.5,
            linewidth=1.2,
        )
    axes[0].axvline(0.0, color=ink, linewidth=0.8, linestyle="--")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Paired mean difference in normalized return AUC (left - right)")
    axes[0].set_title("Planned contrasts with bootstrap 95% intervals", loc="left", color=ink)
    axes[0].grid(axis="x", color="#DDDDDD", linewidth=0.6)

    x = np.arange(len(AGENT_ORDER))
    for condition, offset in (("combined_executed_condition", -0.10), ("latency_only", 0.10)):
        color, marker, label = condition_style[condition]
        part = summary.set_index(["condition", "agent"]).loc[condition]
        values = [part.loc[agent, "seed_cvar_0_10_normalized_return_auc"] for agent in AGENT_ORDER]
        axes[1].scatter(
            x + offset,
            values,
            label=label,
            marker=marker,
            s=38,
            facecolors=(color if marker == "o" else "white"),
            edgecolors=color,
            linewidths=1.2,
        )
    axes[1].set_xticks(
        x,
        [
            "FF DQN",
            "DRQN",
            "Fuzzy gate",
            "Support gate",
            "Controller",
        ],
        rotation=30,
        ha="right",
    )
    axes[1].set_ylabel("Seed-level CVaR 0.10 of normalized return AUC")
    axes[1].set_title("Locked lower-tail summary by condition", loc="left", color=ink)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[1].legend(frameon=False, loc="best")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "Independent sensorized SIL final validation (30 paired seeds)",
        fontsize=12,
        color=ink,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_artifact_audit(
    output_dir: Path,
    summary: pd.DataFrame,
    planned: pd.DataFrame,
    episode_outcomes: pd.DataFrame,
    trace_manifest: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("summary_rows", len(summary) == 10, f"observed={len(summary)} expected=10"),
        ("planned_rows", len(planned) == 8, f"observed={len(planned)} expected=8"),
        ("paired_seed_count", planned["n_pairs"].eq(30).all(), "every contrast has 30 pairs"),
        ("episode_trace_rows", len(episode_outcomes) == 2400, f"observed={len(episode_outcomes)} expected=2400"),
        ("trace_shards", len(trace_manifest) == 300, f"observed={len(trace_manifest)} expected=300"),
        ("trace_metadata_complete", trace_manifest["outcome_metadata_complete"].all(), "every shard has complete v2 outcomes"),
        ("holm_finite", np.isfinite(planned[["paired_t_holm_p", "wilcoxon_holm_p"]].to_numpy(dtype=float)).all(), "report Holm values finite"),
        ("table_exists", TABLE_PATH.is_file(), TABLE_PATH.relative_to(ROOT).as_posix()),
        ("figure_exists", FIGURE_PATH.is_file() and FIGURE_PATH.stat().st_size > 1000, FIGURE_PATH.relative_to(ROOT).as_posix()),
    ]
    rows = [
        {
            "check": name,
            "status": "PASS" if bool(passed) else "FAIL",
            "detail": detail,
        }
        for name, passed, detail in checks
    ]
    return pd.DataFrame(rows)


def aggregate(output_dir: Path = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    protocol = load_and_validate(PROTOCOL_PATH)
    digest_path = PROTOCOL_PATH.with_suffix(".sha256")
    if digest_path.read_text(encoding="utf-8").split()[0] != sha256(PROTOCOL_PATH):
        raise SensorizedFinalAggregationError("protocol digest mismatch")
    raw = pd.read_csv(output_dir / "raw.csv")
    standard_seed_metrics = pd.read_csv(output_dir / "seed_metrics.csv")
    trace_paths = sorted((output_dir / "trace_shards").glob("*.csv.gz"))
    traces, trace_manifest = load_traces(trace_paths)
    episode_outcomes = build_episode_outcomes(traces)

    raw_eval = raw[raw["phase"].eq("eval")]
    trace_lengths = (
        traces.groupby(EPISODE_GROUP, sort=True).size().rename("trace_steps").reset_index()
    )
    raw_lengths = raw_eval[
        ["environment", "agent", "seed", "phase", "checkpoint", "episode", "length"]
    ].rename(columns={"length": "raw_steps"})
    coverage = trace_lengths.merge(raw_lengths, on=EPISODE_GROUP, validate="one_to_one")
    if not np.array_equal(coverage["trace_steps"].to_numpy(), coverage["raw_steps"].to_numpy()):
        raise SensorizedFinalAggregationError("trace/raw episode-length mismatch")

    pair_metrics, correctness = build_pair_metrics(traces)
    sensor_seed_metrics = build_sensor_seed_metrics(
        standard_seed_metrics, episode_outcomes, pair_metrics, correctness
    )
    summary = build_summary(sensor_seed_metrics, raw, episode_outcomes)
    planned = build_planned_contrasts(sensor_seed_metrics, protocol)
    budget = build_budget_audit(sensor_seed_metrics, protocol)
    table = build_table(summary, planned)

    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_seed_metrics.to_csv(output_dir / "sensor_seed_metrics.csv", index=False)
    episode_outcomes.to_csv(output_dir / "episode_trace_outcomes.csv", index=False)
    pair_metrics.to_csv(output_dir / "trace_pair_metrics.csv", index=False)
    correctness.to_csv(output_dir / "support_correctness.csv", index=False)
    trace_manifest.to_csv(output_dir / "trace_metadata_manifest.csv", index=False)
    summary.to_csv(output_dir / "sensorized_final_summary.csv", index=False)
    planned.to_csv(output_dir / "planned_contrasts.csv", index=False)
    budget.to_csv(output_dir / "budget_match_audit.csv", index=False)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    plot_figure(summary, planned, FIGURE_PATH)
    artifact = build_artifact_audit(
        output_dir, summary, planned, episode_outcomes, trace_manifest
    )
    artifact.to_csv(output_dir / "artifact_audit.csv", index=False)
    receipt = {
        "status": "PASS" if artifact["status"].eq("PASS").all() and budget["audit_status"].eq("PASS").all() else "FAIL",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "seed_range": "16000-16029",
        "conditions": CONDITION_ORDER,
        "agents": AGENT_ORDER,
        "run_count": 300,
        "trace_shards": len(trace_manifest),
        "trace_rows": len(traces),
        "episode_trace_rows": len(episode_outcomes),
        "planned_contrast_rows": len(planned),
        "excluded_seeds": [],
        "generator_pooling_performed": False,
        "practical_readiness_claim_permitted": False,
    }
    (output_dir / "aggregation_audit.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    if receipt["status"] != "PASS":
        raise SensorizedFinalAggregationError("artifact or budget audit failed")
    return {
        "sensor_seed_metrics": sensor_seed_metrics,
        "episode_trace_outcomes": episode_outcomes,
        "trace_pair_metrics": pair_metrics,
        "support_correctness": correctness,
        "trace_metadata_manifest": trace_manifest,
        "summary": summary,
        "planned_contrasts": planned,
        "budget_match_audit": budget,
        "table": table,
        "artifact_audit": artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    outputs = aggregate(args.output_dir.resolve())
    print(
        "SENSORIZED_FINAL_AGGREGATION_PASS "
        f"seed_rows={len(outputs['sensor_seed_metrics'])} "
        f"contrast_rows={len(outputs['planned_contrasts'])}"
    )


if __name__ == "__main__":
    main()
