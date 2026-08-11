from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for path in (ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_q.safety_traces import episode_safety_metrics  # noqa: E402
from hybrid_q.statistics import bootstrap_mean_interval  # noqa: E402
from hybrid_q.trace_evidence import SAFETY_TRACE_SCHEMA_VERSION  # noqa: E402
from scripts.aggregate_sensor_aliasing import validate_trace_schema  # noqa: E402
from scripts.lock_safety_trace_protocol import (  # noqa: E402
    DEFAULT_PROTOCOL,
    load_and_validate,
    sha256,
)


DEFAULT_RESULTS = ROOT / "results/diagnostic_extensions/safety_traces"
DEFAULT_TABLE = ROOT / "tables/table_safety_traces.csv"
DEFAULT_FIGURE = ROOT / "figures/fig_safety_traces.pdf"
GLOBAL_MANIFEST = (
    ROOT / "results/diagnostic_extensions/tail_risk/safety_metrics_manifest.csv"
)
IDENTITY = ["family", "agent", "seed", "checkpoint", "episode"]
POSITION_COLUMNS = [
    "post_action_latent_position_x",
    "post_action_latent_position_y",
    "post_action_latent_position_z",
]
REFERENCE_COLUMNS = [
    "post_action_reference_x",
    "post_action_reference_y",
    "post_action_reference_z",
]
SUMMARY_METRICS = {
    "trajectory_deviation_rmse_m": "lower_is_better",
    "maximum_trajectory_deviation_m": "lower_is_better",
    "recovery_probability": "higher_is_better",
    "recovery_time_seconds_recovered_only": "lower_is_better",
    "censor_time_seconds_nonrecovered_only": "descriptive",
    "saturation_duration_seconds": "lower_is_better",
    "constraint_violation_duration_seconds": "lower_is_better",
    "near_miss_duration_seconds": "lower_is_better",
}


class SafetyTraceAggregationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_trace_shards(trace_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(trace_dir.glob("*.csv.gz"))
    if not paths:
        raise SafetyTraceAggregationError("no safety trace shards found")
    frames = []
    manifest_rows = []
    for path in paths:
        frame = pd.read_csv(path, compression="gzip")
        validate_trace_schema(frame)
        versions = set(frame["trace_schema_version"].astype(str))
        if versions != {SAFETY_TRACE_SCHEMA_VERSION}:
            raise SafetyTraceAggregationError(f"non-v3 trace shard: {path.name}")
        frame.rename(columns={"environment": "family"}, inplace=True)
        frame["source_file"] = relative(path)
        frames.append(frame)
        with gzip.open(path, "rb") as handle:
            uncompressed_bytes = sum(
                len(block) for block in iter(lambda: handle.read(1024 * 1024), b"")
            )
        manifest_rows.append(
            {
                "source_file": relative(path),
                "sha256": file_sha256(path),
                "compressed_bytes": path.stat().st_size,
                "uncompressed_bytes": uncompressed_bytes,
                "decision_rows": len(frame),
                "trace_schema_version": SAFETY_TRACE_SCHEMA_VERSION,
                "audit_status": "PASS",
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated([*IDENTITY, "step"]).any():
        raise SafetyTraceAggregationError("duplicate trace decisions")
    return combined, pd.DataFrame(manifest_rows)


def build_episode_metrics(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    recovery = protocol["definitions"]["recovery"]
    rows = []
    for identity, group in frame.groupby(IDENTITY, sort=True, dropna=False):
        group = group.sort_values("step")
        timestep = pd.to_numeric(
            group["control_timestep_seconds"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.allclose(timestep, timestep[0], atol=1e-12):
            raise SafetyTraceAggregationError("episode timestep is not fixed")
        metrics = episode_safety_metrics(
            positions=group[POSITION_COLUMNS].to_numpy(dtype=float),
            reference_positions=group[REFERENCE_COLUMNS].to_numpy(dtype=float),
            timestamps=pd.to_numeric(
                group["post_action_timestamp"], errors="raise"
            ).to_numpy(dtype=float),
            perturbation_active=pd.to_numeric(
                group["post_action_perturbation_active"], errors="raise"
            ).to_numpy(dtype=int),
            saturation_active=pd.to_numeric(
                group["post_action_saturation_active"], errors="raise"
            ).to_numpy(dtype=int),
            constraint_active=pd.to_numeric(
                group["post_action_constraint_active"], errors="raise"
            ).to_numpy(dtype=int),
            near_miss_active=pd.to_numeric(
                group["post_action_near_miss"], errors="raise"
            ).to_numpy(dtype=int),
            timestep_seconds=float(timestep[0]),
            recovery_band_m=float(recovery["band_m"]),
            dwell_steps=int(recovery["required_dwell_steps"]),
        )
        onset_steps = set(
            pd.to_numeric(group["perturbation_onset_step"], errors="raise").astype(int)
        )
        path_ids = set(group["nominal_reference_path_id"].astype(str))
        if onset_steps != {6} or path_ids != {"linear_reset_position_to_episode_target_v1"}:
            raise SafetyTraceAggregationError("episode protocol metadata mismatch")
        active = pd.to_numeric(
            group["post_action_perturbation_active"], errors="raise"
        ).astype(bool)
        first_active_step = int(group.loc[active, "step"].min())
        if first_active_step != 6:
            raise SafetyTraceAggregationError("perturbation onset step mismatch")
        rows.append(
            {
                **dict(zip(IDENTITY, identity)),
                **metrics,
                "recovery_band_m": float(recovery["band_m"]),
                "required_dwell_steps": int(recovery["required_dwell_steps"]),
                "non_recovery_rule": recovery["non_recovery_rule"],
                "source_file": group["source_file"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(IDENTITY).reset_index(drop=True)


def build_seed_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    final_checkpoint = int(episodes["checkpoint"].max())
    final = episodes[episodes["checkpoint"].eq(final_checkpoint)].copy()
    final["recovered_float"] = final["recovered"].astype(float)
    final["recovery_time_for_mean"] = pd.to_numeric(
        final["recovery_time_seconds"], errors="coerce"
    )
    final["censor_time_for_mean"] = pd.to_numeric(
        final["censor_time_seconds"], errors="coerce"
    )
    rows = []
    for identity, group in final.groupby(["family", "agent", "seed"], sort=True):
        recovered = group[group["recovered"].astype(bool)]
        censored = group[~group["recovered"].astype(bool)]
        rows.append(
            {
                **dict(zip(["family", "agent", "seed"], identity)),
                "checkpoint": final_checkpoint,
                "episode_count": len(group),
                "trajectory_deviation_rmse_m": group[
                    "trajectory_deviation_rmse_m"
                ].mean(),
                "maximum_trajectory_deviation_m": group[
                    "maximum_trajectory_deviation_m"
                ].mean(),
                "recovery_probability": group["recovered_float"].mean(),
                "recovery_time_seconds_recovered_only": (
                    recovered["recovery_time_seconds"].mean()
                    if len(recovered)
                    else np.nan
                ),
                "recovered_episode_count": len(recovered),
                "censor_time_seconds_nonrecovered_only": (
                    censored["censor_time_seconds"].mean()
                    if len(censored)
                    else np.nan
                ),
                "censored_episode_count": len(censored),
                "saturation_duration_seconds": group[
                    "saturation_duration_seconds"
                ].mean(),
                "constraint_violation_duration_seconds": group[
                    "constraint_violation_duration_seconds"
                ].mean(),
                "near_miss_duration_seconds": group[
                    "near_miss_duration_seconds"
                ].mean(),
                "source_files": ";".join(sorted(set(group["source_file"]))),
            }
        )
    return pd.DataFrame(rows)


def build_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, agent), group in seeds.groupby(["family", "agent"], sort=True):
        for metric, direction in SUMMARY_METRICS.items():
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = bootstrap_mean_interval(
                    finite,
                    seed=int(hashlib.sha256(f"{family}|{agent}|{metric}".encode()).hexdigest()[:8], 16),
                    samples=10_000,
                )
                status = "available"
                reason = ""
                mean = float(finite.mean())
                median = float(np.median(finite))
            else:
                low = high = mean = median = np.nan
                status = "not_available"
                reason = "no finite seed-level values under the locked conditional denominator"
            rows.append(
                {
                    "result_family": "diagnostic_extensions/safety_traces",
                    "family": family,
                    "agent": agent,
                    "metric": metric,
                    "direction": direction,
                    "n_seeds_total": group["seed"].nunique(),
                    "n_seeds_finite": int(finite.size),
                    "mean": mean,
                    "median": median,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "availability": status,
                    "reason_if_unavailable": reason,
                    "source_file": relative(DEFAULT_RESULTS / "safety_trace_seed_metrics.csv"),
                    "source_column": metric,
                }
            )
    return pd.DataFrame(rows)


def build_metric_manifest(summary: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "trajectory_deviation_rmse_m": "RMSE of post-onset Euclidean latent-position deviation from the locked timestamp-aligned nominal path.",
        "maximum_trajectory_deviation_m": "Maximum post-onset Euclidean latent-position deviation from the locked timestamp-aligned nominal path.",
        "recovery_probability": "Fraction of final-checkpoint evaluation episodes satisfying the locked 0.15 m, six-sample dwell rule.",
        "recovery_time_seconds_recovered_only": "Elapsed onset-to-first-qualifying-band-entry time, summarized only for recovered episodes; non-recovery is not imputed.",
        "censor_time_seconds_nonrecovered_only": "Observed onset-to-last-sample time for right-censored non-recovered episodes.",
        "saturation_duration_seconds": "Mean episode duration with at least one motor at the locked low/high RPM threshold.",
        "constraint_violation_duration_seconds": "Mean episode duration with collision or obstacle clearance at or below 0.18 m.",
        "near_miss_duration_seconds": "Mean episode duration without collision and with obstacle clearance at or below 0.10 m.",
    }
    denominator = {
        "recovery_time_seconds_recovered_only": "finite seed means from recovered final-checkpoint episodes only",
        "censor_time_seconds_nonrecovered_only": "finite seed means from non-recovered final-checkpoint episodes only",
        "recovery_probability": "all final-checkpoint evaluation episodes per seed",
    }
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "result_family": row.result_family,
                "environment": row.family,
                "agent": row.agent,
                "metric_name": row.metric,
                "definition": definitions[row.metric],
                "direction": row.direction,
                "aggregation_level": "seed_from_episode_trace",
                "denominator": denominator.get(
                    row.metric, "all final-checkpoint evaluation episodes per seed"
                ),
                "alpha": "",
                "source_file": row.source_file,
                "source_column": row.source_column,
                "availability": row.availability,
                "reason_if_unavailable": row.reason_if_unavailable,
            }
        )
    return pd.DataFrame(rows)


def update_global_manifest(step_manifest: pd.DataFrame) -> None:
    columns = list(step_manifest.columns)
    existing = pd.read_csv(GLOBAL_MANIFEST, keep_default_na=False)
    if list(existing.columns) != columns:
        raise SafetyTraceAggregationError("global safety manifest schema mismatch")
    existing = existing[
        existing["result_family"].astype(str) != "diagnostic_extensions/safety_traces"
    ]
    updated = pd.concat([existing, step_manifest], ignore_index=True)
    updated.sort_values(
        ["result_family", "environment", "agent", "metric_name"], inplace=True
    )
    updated.to_csv(GLOBAL_MANIFEST, index=False)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    agents = [
        "feed_forward_dqn",
        "selected_temporal_drqn",
        "fuzzy_relative_reliability",
        "selected_approximate_support",
        "sensorized_controller",
    ]
    labels = {
        "feed_forward_dqn": "Feed-forward DQN",
        "selected_temporal_drqn": "Selected DRQN",
        "fuzzy_relative_reliability": "Fuzzy reliability",
        "selected_approximate_support": "Selected support",
        "sensorized_controller": "Model-based controller",
    }
    families = [
        "combined_executed_condition_safety_trace",
        "latency_only_safety_trace",
    ]
    styles = {
        families[0]: ("#2F6B9A", "o", "Combined"),
        families[1]: ("#D9822B", "s", "Latency only"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    for axis, metric, title, xlabel in (
        (axes[0], "trajectory_deviation_rmse_m", "Trajectory-deviation RMSE", "Metres (lower is better)"),
        (axes[1], "recovery_probability", "Recovery probability", "Episode proportion (higher is better)"),
    ):
        metric_rows = summary[summary["metric"].eq(metric)]
        for offset, family in zip((-0.12, 0.12), families):
            color, marker, legend = styles[family]
            rows = metric_rows[metric_rows["family"].eq(family)].set_index("agent")
            values = np.asarray([rows.loc[agent, "mean"] for agent in agents], dtype=float)
            low = np.asarray([rows.loc[agent, "bootstrap_ci_low"] for agent in agents], dtype=float)
            high = np.asarray([rows.loc[agent, "bootstrap_ci_high"] for agent in agents], dtype=float)
            y = np.arange(len(agents)) + offset
            axis.errorbar(
                values,
                y,
                xerr=np.vstack((values - low, high - values)),
                fmt=marker,
                color=color,
                markerfacecolor="white" if family == families[1] else color,
                markeredgecolor=color,
                linewidth=1.4,
                capsize=2.5,
                label=legend,
            )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_yticks(np.arange(len(agents)), [labels[agent] for agent in agents])
        axis.grid(axis="x", color="#D7DCE2", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.invert_yaxis()
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle("Locked safety-trace outcomes", x=0.07, ha="left", fontsize=14, fontweight="bold")
    figure.text(
        0.07,
        0.925,
        "Final checkpoint; 10 paired new seeds per family; points are seed means with bootstrap 95% intervals",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def write_artifact_audit(output_dir: Path, expected_traces: int) -> None:
    checks = {
        "protocol_digest_valid": sha256(DEFAULT_PROTOCOL)
        == DEFAULT_PROTOCOL.with_suffix(".sha256").read_text(encoding="utf-8").split()[0],
        "trace_shard_count": len(list((output_dir / "trace_shards").glob("*.csv.gz")))
        == expected_traces,
        "episode_metrics_present": (output_dir / "safety_trace_episode_metrics.csv").is_file(),
        "seed_metrics_present": (output_dir / "safety_trace_seed_metrics.csv").is_file(),
        "summary_present": (output_dir / "safety_trace_summary.csv").is_file(),
        "metric_manifest_present": (output_dir / "safety_metrics_manifest.csv").is_file(),
        "table_present": DEFAULT_TABLE.is_file(),
        "figure_present": DEFAULT_FIGURE.is_file(),
    }
    rows = [
        {"check": key, "status": "PASS" if value else "FAIL", "reason": "" if value else "missing or inconsistent artifact"}
        for key, value in checks.items()
    ]
    pd.DataFrame(rows).to_csv(output_dir / "artifact_audit.csv", index=False)
    if not all(checks.values()):
        raise SafetyTraceAggregationError("artifact audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    protocol = load_and_validate(DEFAULT_PROTOCOL)
    output_dir = args.results.resolve()
    frame, trace_manifest = load_trace_shards(output_dir / "trace_shards")
    episodes = build_episode_metrics(frame, protocol)
    seeds = build_seed_metrics(episodes)
    summary = build_summary(seeds)
    metric_manifest = build_metric_manifest(summary)
    trace_manifest.to_csv(output_dir / "trace_manifest.csv", index=False)
    episodes.to_csv(output_dir / "safety_trace_episode_metrics.csv", index=False)
    seeds.to_csv(output_dir / "safety_trace_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "safety_trace_summary.csv", index=False)
    metric_manifest.to_csv(output_dir / "safety_metrics_manifest.csv", index=False)
    update_global_manifest(metric_manifest)
    DEFAULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEFAULT_TABLE, index=False)
    plot_summary(summary, DEFAULT_FIGURE)
    write_artifact_audit(output_dir, expected_traces=100)
    audit = {
        "status": "PASS",
        "protocol_sha256": sha256(DEFAULT_PROTOCOL),
        "trace_schema_version": SAFETY_TRACE_SCHEMA_VERSION,
        "trace_shards": len(trace_manifest),
        "trace_rows": len(frame),
        "episode_rows": len(episodes),
        "seed_rows": len(seeds),
        "summary_rows": len(summary),
        "excluded_seeds": [],
        "non_recovery_rule": protocol["definitions"]["recovery"]["non_recovery_rule"],
        "generator_pooling_performed": False,
    }
    (output_dir / "aggregation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "SAFETY_TRACE_AGGREGATION_PASS "
        f"episodes={len(episodes)} seed_rows={len(seeds)}"
    )


if __name__ == "__main__":
    main()
