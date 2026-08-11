from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.statistics import (  # noqa: E402
    empirical_lower_cvar,
    lower_quantile,
    maximum_drawdown,
    worst_checkpoint,
    worst_decile_mean,
)


PRIMARY_ALPHA = 0.10
SENSITIVITY_ALPHA = 0.05
MIN_SENSITIVITY_TAIL_OBSERVATIONS = 2
IDENTITY = ["result_family", "environment", "agent"]
RAW_REQUIRED = {"environment", "agent", "seed", "phase", "checkpoint", "return"}
SAFETY_COLUMNS = {
    "failure_probability": "failure_rate",
    "collision_rate": "collision_rate",
    "risk_zone_rate": "risk_zone_rate",
    "motor_saturation_rate": "motor_saturation_rate",
}


@dataclass(frozen=True)
class CompletedFamily:
    name: str
    directory: Path
    seed_metrics: Path
    raw_paths: tuple[Path, ...]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def raw_result_paths(directory: Path) -> tuple[Path, ...]:
    for name in ("raw.csv", "raw.csv.gz", "raw.csv.xz"):
        candidate = directory / name
        if candidate.exists():
            return (candidate,)
    return tuple(sorted((directory / "raw_parts").glob("*.csv*")))


def discover_completed_families(
    results_root: Path, output_dir: Path
) -> list[CompletedFamily]:
    families: list[CompletedFamily] = []
    output_resolved = output_dir.resolve()
    for seed_metrics in sorted(results_root.rglob("seed_metrics.csv")):
        directory = seed_metrics.parent
        directory_resolved = directory.resolve()
        if directory_resolved == output_resolved or output_resolved in directory_resolved.parents:
            continue
        metadata = directory / "metadata.json"
        audit_path = directory / "audit.json"
        if not metadata.exists() or not audit_path.exists():
            continue
        try:
            audit_status = json.loads(audit_path.read_text(encoding="utf-8")).get(
                "status"
            )
        except (json.JSONDecodeError, OSError):
            continue
        if audit_status != "PASS":
            continue
        families.append(
            CompletedFamily(
                name=directory.relative_to(results_root).as_posix(),
                directory=directory,
                seed_metrics=seed_metrics,
                raw_paths=raw_result_paths(directory),
            )
        )
    if not families:
        raise RuntimeError(f"no completed result families found under {results_root}")
    return families


def tail_statistics(values: object) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("tail statistics require at least one finite observation")
    primary_count = max(1, int(np.ceil(PRIMARY_ALPHA * finite.size)))
    sensitivity_count = max(1, int(np.ceil(SENSITIVITY_ALPHA * finite.size)))
    sensitivity_available = (
        sensitivity_count >= MIN_SENSITIVITY_TAIL_OBSERVATIONS
    )
    return {
        "n_total": int(array.size),
        "n_finite": int(finite.size),
        "nonfinite_dropped": int(array.size - finite.size),
        "mean_return": float(finite.mean()),
        "minimum_return": float(finite.min()),
        "return_quantile_0_05": lower_quantile(finite, 0.05),
        "worst_decile_mean_return": worst_decile_mean(finite),
        "cvar_0_10_return": empirical_lower_cvar(finite, PRIMARY_ALPHA),
        "cvar_0_10_tail_count": primary_count,
        "cvar_0_05_return": (
            empirical_lower_cvar(finite, SENSITIVITY_ALPHA)
            if sensitivity_available
            else np.nan
        ),
        "cvar_0_05_tail_count": sensitivity_count,
        "cvar_0_05_available": sensitivity_available,
        "cvar_0_05_reason": (
            ""
            if sensitivity_available
            else "fewer than two observations fall in the alpha=0.05 empirical tail"
        ),
    }


def finite_mean(frame: pd.DataFrame, column: str) -> tuple[float, int, str]:
    if column not in frame.columns:
        return np.nan, 0, f"source column {column} is absent"
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, 0, f"source column {column} has no finite observations"
    return float(finite.mean()), int(finite.size), ""


def analyze_seed_metrics(family: CompletedFamily) -> pd.DataFrame:
    frame = pd.read_csv(family.seed_metrics)
    required = {"environment", "agent", "seed", "mean_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{family.seed_metrics} is missing seed-level columns {sorted(missing)}"
        )
    if frame.duplicated(["environment", "agent", "seed"]).any():
        raise ValueError(f"{family.seed_metrics} contains duplicate seed-level rows")
    rows: list[dict[str, object]] = []
    for (environment, agent), group in frame.groupby(
        ["environment", "agent"], sort=True, dropna=False
    ):
        stats = tail_statistics(pd.to_numeric(group["mean_return"], errors="coerce"))
        row: dict[str, object] = {
            "result_family": family.name,
            "environment": environment,
            "agent": agent,
            "aggregation_level": "seed",
            "source_file": display_path(family.seed_metrics),
            "source_column": "mean_return",
            "n_seed_rows": stats.pop("n_total"),
            "n_finite_returns": stats.pop("n_finite"),
            "nonfinite_returns_dropped": stats.pop("nonfinite_dropped"),
            **stats,
        }
        for metric_name, source_column in SAFETY_COLUMNS.items():
            value, denominator, reason = finite_mean(group, source_column)
            row[metric_name] = value
            row[f"{metric_name}_n"] = denominator
            row[f"{metric_name}_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows).sort_values(IDENTITY).reset_index(drop=True)


def _raw_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def analyze_raw_outputs(
    family: CompletedFamily, seed_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if not family.raw_paths:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "raw episode file is absent"
    for path in family.raw_paths:
        missing = RAW_REQUIRED - set(_raw_header(path))
        if missing:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                f"{display_path(path)} is missing raw columns {sorted(missing)}",
            )

    final_keys = seed_frame[["environment", "agent", "seed", "checkpoint"]].copy()
    for column in ("seed", "checkpoint"):
        final_keys[column] = pd.to_numeric(final_keys[column], errors="raise")
    if final_keys.duplicated().any():
        raise ValueError(f"{family.seed_metrics} has duplicate final checkpoint keys")

    episode_values: dict[tuple[object, object], list[np.ndarray]] = defaultdict(list)
    episode_seeds: dict[tuple[object, object], set[int]] = defaultdict(set)
    episode_checkpoints: dict[tuple[object, object], set[float]] = defaultdict(set)
    episode_total: dict[tuple[object, object], int] = defaultdict(int)
    checkpoint_parts: list[pd.DataFrame] = []

    for raw_path in family.raw_paths:
        for chunk in pd.read_csv(
            raw_path,
            usecols=lambda column: column in RAW_REQUIRED,
            chunksize=250_000,
        ):
            evaluation = chunk[chunk["phase"].astype(str).str.lower() == "eval"].copy()
            if evaluation.empty:
                continue
            for column in ("seed", "checkpoint", "return"):
                evaluation[column] = pd.to_numeric(evaluation[column], errors="coerce")

            final_rows = evaluation.merge(
                final_keys,
                on=["environment", "agent", "seed", "checkpoint"],
                how="inner",
                validate="many_to_one",
            )
            for (environment, agent), group in final_rows.groupby(
                ["environment", "agent"], sort=False, dropna=False
            ):
                key = (environment, agent)
                episode_total[key] += len(group)
                values = group["return"].to_numpy(dtype=float)
                episode_values[key].append(values[np.isfinite(values)])
                episode_seeds[key].update(
                    group.loc[np.isfinite(group["seed"]), "seed"].astype(int).tolist()
                )
                episode_checkpoints[key].update(
                    group.loc[
                        np.isfinite(group["checkpoint"]), "checkpoint"
                    ].astype(float)
                )

            finite_eval = evaluation[np.isfinite(evaluation["return"])].copy()
            if finite_eval.empty:
                continue
            partial = (
                finite_eval.groupby(
                    ["environment", "agent", "seed", "checkpoint"],
                    sort=False,
                    dropna=False,
                )["return"]
                .agg(return_sum="sum", episode_count="count")
                .reset_index()
            )
            checkpoint_parts.append(partial)

    source_files = ";".join(display_path(path) for path in family.raw_paths)
    episode_rows: list[dict[str, object]] = []
    for key in sorted(episode_values, key=lambda item: (str(item[0]), str(item[1]))):
        finite_arrays = [array for array in episode_values[key] if array.size]
        if not finite_arrays:
            continue
        values = np.concatenate(finite_arrays)
        stats = tail_statistics(values)
        stats.pop("n_total")
        stats.pop("nonfinite_dropped")
        checkpoints = episode_checkpoints[key]
        episode_rows.append(
            {
                "result_family": family.name,
                "environment": key[0],
                "agent": key[1],
                "aggregation_level": "episode",
                "source_file": source_files,
                "source_column": "return",
                "n_episode_rows": episode_total[key],
                "n_finite_returns": stats.pop("n_finite"),
                "nonfinite_returns_dropped": episode_total[key] - len(values),
                "n_seeds": len(episode_seeds[key]),
                "final_checkpoint_min": min(checkpoints) if checkpoints else np.nan,
                "final_checkpoint_max": max(checkpoints) if checkpoints else np.nan,
                **stats,
            }
        )

    if not checkpoint_parts:
        reason = "raw files contain no finite phase=eval return observations"
        return pd.DataFrame(episode_rows), pd.DataFrame(), pd.DataFrame(), reason

    checkpoint_seed = pd.concat(checkpoint_parts, ignore_index=True)
    checkpoint_seed = (
        checkpoint_seed.groupby(
            ["environment", "agent", "seed", "checkpoint"],
            sort=True,
            dropna=False,
        )[["return_sum", "episode_count"]]
        .sum()
        .reset_index()
    )
    checkpoint_seed["mean_return"] = (
        checkpoint_seed["return_sum"] / checkpoint_seed["episode_count"]
    )

    checkpoint_rows: list[dict[str, object]] = []
    for (environment, agent, checkpoint), group in checkpoint_seed.groupby(
        ["environment", "agent", "checkpoint"], sort=True, dropna=False
    ):
        stats = tail_statistics(group["mean_return"])
        checkpoint_rows.append(
            {
                "result_family": family.name,
                "environment": environment,
                "agent": agent,
                "checkpoint": checkpoint,
                "aggregation_level": "seed_at_checkpoint",
                "source_file": source_files,
                "source_column": "return",
                "n_seeds": stats.pop("n_total"),
                "n_finite_returns": stats.pop("n_finite"),
                "nonfinite_returns_dropped": stats.pop("nonfinite_dropped"),
                **stats,
            }
        )
    checkpoint_frame = pd.DataFrame(checkpoint_rows).sort_values(
        IDENTITY + ["checkpoint"]
    )

    curve_rows: list[dict[str, object]] = []
    for (environment, agent), curve in checkpoint_frame.groupby(
        ["environment", "agent"], sort=True, dropna=False
    ):
        curve = curve.sort_values("checkpoint")
        checkpoints = curve["checkpoint"].to_numpy(dtype=float)
        values = curve["mean_return"].to_numpy(dtype=float)
        worst_value = worst_checkpoint(values)
        tied = np.isclose(values, worst_value, rtol=1e-12, atol=1e-12)
        curve_rows.append(
            {
                "result_family": family.name,
                "environment": environment,
                "agent": agent,
                "checkpoint_source_file": source_files,
                "checkpoint_source_column": "return",
                "n_checkpoints": len(curve),
                "worst_checkpoint": float(np.min(checkpoints[tied])),
                "worst_checkpoint_return": worst_value,
                "maximum_learning_curve_drawdown": maximum_drawdown(
                    checkpoints, values
                ),
            }
        )
    return (
        pd.DataFrame(episode_rows).sort_values(IDENTITY).reset_index(drop=True),
        checkpoint_frame.reset_index(drop=True),
        pd.DataFrame(curve_rows).sort_values(IDENTITY).reset_index(drop=True),
        "",
    )


def merge_summary(
    seed: pd.DataFrame, episode: pd.DataFrame, curves: pd.DataFrame
) -> pd.DataFrame:
    seed_rename = {
        column: f"seed_{column}"
        for column in seed.columns
        if column not in IDENTITY
    }
    summary = seed.rename(columns=seed_rename)
    if not episode.empty:
        episode_rename = {
            column: f"episode_{column}"
            for column in episode.columns
            if column not in IDENTITY
        }
        summary = summary.merge(
            episode.rename(columns=episode_rename), on=IDENTITY, how="left"
        )
    if not curves.empty:
        summary = summary.merge(curves, on=IDENTITY, how="left")
    return summary.sort_values(IDENTITY).reset_index(drop=True)


def _available(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def build_manifest(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        identity: dict[str, object],
        metric_name: str,
        definition: str,
        direction: str,
        aggregation_level: str,
        denominator: str,
        alpha: float | None,
        source_file: object,
        source_column: str,
        value: object,
        reason: str = "",
    ) -> None:
        available = _available(value)
        rows.append(
            {
                **identity,
                "metric_name": metric_name,
                "definition": definition,
                "direction": direction,
                "aggregation_level": aggregation_level,
                "denominator": denominator,
                "alpha": alpha if alpha is not None else np.nan,
                "source_file": source_file,
                "source_column": source_column,
                "availability": "available" if available else "not_available",
                "reason_if_unavailable": "" if available else reason,
            }
        )

    tail_definitions = {
        "minimum_return": "Minimum finite {unit} return.",
        "return_quantile_0_05": (
            "Linear empirical 5th percentile of finite {unit} returns."
        ),
        "worst_decile_mean_return": (
            "Arithmetic mean of the lowest max(1, ceil(0.10 * n)) finite "
            "{unit} returns; identical to the locked empirical CVaR_0.10 estimator."
        ),
        "cvar_0_10_return": (
            "Arithmetic mean of the lowest max(1, ceil(0.10 * n)) finite "
            "{unit} returns."
        ),
        "cvar_0_05_return": (
            "Sensitivity metric: arithmetic mean of the lowest "
            "max(1, ceil(0.05 * n)) finite {unit} returns; available only when "
            "the empirical tail contains at least two observations."
        ),
    }

    for _, row in summary.iterrows():
        identity = {column: row[column] for column in IDENTITY}
        seed_source = row.get("seed_source_file", "")
        for metric, definition in tail_definitions.items():
            reason = "no finite seed-level final-checkpoint returns"
            if metric == "cvar_0_05_return":
                reason = str(row.get("seed_cvar_0_05_reason", reason))
            add(
                identity,
                f"seed_{metric}",
                definition.format(unit="final-checkpoint seed-mean"),
                "higher_is_better",
                "seed",
                "finite final-checkpoint seed means",
                0.10 if "0_10" in metric or metric == "worst_decile_mean_return" else (
                    0.05 if "0_05" in metric else None
                ),
                seed_source,
                "mean_return",
                row.get(f"seed_{metric}"),
                reason,
            )

        episode_source = row.get("episode_source_file", "")
        episode_missing = "no matching finite final-checkpoint raw episode returns"
        for metric, definition in tail_definitions.items():
            reason = episode_missing
            if metric == "cvar_0_05_return" and row.get("episode_cvar_0_05_reason"):
                reason = str(row.get("episode_cvar_0_05_reason"))
            add(
                identity,
                f"episode_{metric}",
                definition.format(unit="pooled final-checkpoint evaluation-episode"),
                "higher_is_better",
                "episode",
                "finite final-checkpoint evaluation episodes pooled across seeds",
                0.10 if "0_10" in metric or metric == "worst_decile_mean_return" else (
                    0.05 if "0_05" in metric else None
                ),
                episode_source,
                "return",
                row.get(f"episode_{metric}"),
                reason,
            )

        for metric_name, source_column in SAFETY_COLUMNS.items():
            add(
                identity,
                metric_name,
                (
                    f"Arithmetic mean of finite final-checkpoint per-seed "
                    f"{source_column} values."
                ),
                "lower_is_better",
                "seed",
                f"finite seeds with {source_column}",
                None,
                seed_source,
                source_column,
                row.get(f"seed_{metric_name}"),
                str(row.get(f"seed_{metric_name}_reason", "metric unavailable")),
            )

        checkpoint_source = row.get("checkpoint_source_file", "")
        add(
            identity,
            "worst_checkpoint_return",
            "Minimum checkpoint return on the learning curve after averaging real evaluation episodes within seed and then averaging seed means.",
            "higher_is_better",
            "checkpoint_curve",
            "finite learning-curve checkpoints",
            None,
            checkpoint_source,
            "return",
            row.get("worst_checkpoint_return"),
            "no finite evaluation checkpoint curve",
        )
        add(
            identity,
            "maximum_learning_curve_drawdown",
            "Largest decline from any prior peak checkpoint mean return to a later checkpoint mean return, in return units.",
            "lower_is_better",
            "checkpoint_curve",
            "ordered finite learning-curve checkpoints",
            None,
            checkpoint_source,
            "return",
            row.get("maximum_learning_curve_drawdown"),
            "no finite evaluation checkpoint curve",
        )
        add(
            identity,
            "recovery_time",
            "Elapsed trace time from a pre-specified adverse event to a pre-specified recovered state.",
            "lower_is_better",
            "episode_trace",
            "adverse events with timestamped recovery traces",
            None,
            episode_source,
            "not_present",
            np.nan,
            "required timestamped adverse-event and recovery-state trace fields are absent; no value was synthesized",
        )
        add(
            identity,
            "trajectory_deviation",
            "Distance between timestamp-aligned realized and reference trajectory positions.",
            "lower_is_better",
            "episode_trace",
            "timestamped trajectory samples with realized and reference positions",
            None,
            episode_source,
            "not_present",
            np.nan,
            "required timestamped realized-position and reference-trajectory fields are absent; localization_error_mean is not substituted",
        )
    columns = [
        *IDENTITY,
        "metric_name",
        "definition",
        "direction",
        "aggregation_level",
        "denominator",
        "alpha",
        "source_file",
        "source_column",
        "availability",
        "reason_if_unavailable",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        IDENTITY + ["aggregation_level", "metric_name"]
    )


def publication_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *IDENTITY,
        "seed_n_finite_returns",
        "seed_minimum_return",
        "seed_return_quantile_0_05",
        "seed_worst_decile_mean_return",
        "seed_cvar_0_10_return",
        "seed_cvar_0_05_return",
        "episode_n_finite_returns",
        "episode_minimum_return",
        "episode_return_quantile_0_05",
        "episode_worst_decile_mean_return",
        "episode_cvar_0_10_return",
        "episode_cvar_0_05_return",
        "worst_checkpoint",
        "worst_checkpoint_return",
        "maximum_learning_curve_drawdown",
        "seed_failure_probability",
        "seed_collision_rate",
        "seed_risk_zone_rate",
        "seed_motor_saturation_rate",
        "seed_source_file",
        "episode_source_file",
        "checkpoint_source_file",
    ]
    for column in columns:
        if column not in summary:
            summary[column] = np.nan
    return summary[columns].copy()


def plot_tail_risk(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.1), constrained_layout=True)
    panels = (
        (
            axes[0],
            "seed_mean_return",
            "seed_cvar_0_10_return",
            "Seed-level lower tail",
            "Each point is a result-family/environment/agent group",
            "#2463A3",
        ),
        (
            axes[1],
            "episode_mean_return",
            "episode_cvar_0_10_return",
            "Episode-level lower tail",
            "Final-checkpoint evaluation episodes pooled across seeds",
            "#D28C18",
        ),
    )
    for axis, x_column, y_column, title, subtitle, color in panels:
        if x_column not in summary or y_column not in summary:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            axis.set_axis_off()
            continue
        valid = summary[[x_column, y_column]].apply(pd.to_numeric, errors="coerce")
        valid = valid[np.isfinite(valid[x_column]) & np.isfinite(valid[y_column])]
        if valid.empty:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            axis.set_axis_off()
            continue
        low = float(valid.min().min())
        high = float(valid.max().max())
        span = max(high - low, 1.0)
        bounds = (low - 0.04 * span, high + 0.04 * span)
        axis.scatter(
            valid[x_column],
            valid[y_column],
            s=25,
            color=color,
            alpha=0.58,
            edgecolor="#17212B",
            linewidth=0.35,
        )
        axis.plot(bounds, bounds, color="#4C5663", linestyle="--", linewidth=1.1)
        axis.set_xlim(bounds)
        axis.set_ylim(bounds)
        if max(abs(bounds[0]), abs(bounds[1])) > 100:
            axis.set_xscale("symlog", linthresh=1.0)
            axis.set_yscale("symlog", linthresh=1.0)
        axis.set_title(
            title,
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=22,
        )
        axis.text(
            0.0,
            1.005,
            f"{subtitle}; n={len(valid)} groups",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#4C5663",
        )
        axis.set_xlabel("Mean return")
        axis.set_ylabel("Empirical lower CVaR (alpha=0.10)")
        axis.grid(color="#D9DEE5", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Mean and empirical lower-tail return by aggregation level",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "Dashed line: CVaR = mean. Lower-tail summaries describe observed samples and are not episode-risk guarantees.",
        ha="center",
        fontsize=8.5,
        color="#4C5663",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Tail-risk reanalysis",
            "Creator": "scripts/aggregate_tail_risk.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", na_rep="")


def aggregate_tail_risk(
    results_root: Path,
    output_dir: Path,
    table_output: Path,
    figure_output: Path,
) -> dict[str, int]:
    families = discover_completed_families(results_root, output_dir)
    seed_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    checkpoint_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for family in families:
        seed_source = pd.read_csv(family.seed_metrics)
        seed_result = analyze_seed_metrics(family)
        episode_result, checkpoint_result, curve_result, _ = analyze_raw_outputs(
            family, seed_source
        )
        seed_frames.append(seed_result)
        if not episode_result.empty:
            episode_frames.append(episode_result)
        if not checkpoint_result.empty:
            checkpoint_frames.append(checkpoint_result)
        if not curve_result.empty:
            curve_frames.append(curve_result)

    seed = pd.concat(seed_frames, ignore_index=True).sort_values(IDENTITY)
    episode = (
        pd.concat(episode_frames, ignore_index=True).sort_values(IDENTITY)
        if episode_frames
        else pd.DataFrame()
    )
    checkpoint = (
        pd.concat(checkpoint_frames, ignore_index=True).sort_values(
            IDENTITY + ["checkpoint"]
        )
        if checkpoint_frames
        else pd.DataFrame()
    )
    curves = (
        pd.concat(curve_frames, ignore_index=True).sort_values(IDENTITY)
        if curve_frames
        else pd.DataFrame()
    )
    summary = merge_summary(seed, episode, curves)
    manifest = build_manifest(summary)
    table = publication_table(summary.copy())

    write_csv(seed, output_dir / "seed_tail_risk.csv")
    if not episode.empty:
        write_csv(episode, output_dir / "episode_tail_risk.csv")
    write_csv(checkpoint, output_dir / "checkpoint_tail_risk.csv")
    write_csv(summary, output_dir / "tail_risk_summary.csv")
    write_csv(manifest, output_dir / "safety_metrics_manifest.csv")
    write_csv(table, table_output)
    plot_tail_risk(summary, figure_output)
    return {
        "families": len(families),
        "groups": len(summary),
        "episode_groups": len(episode),
        "checkpoint_rows": len(checkpoint),
        "manifest_rows": len(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reanalyze existing completed outputs for locked lower-tail risk."
    )
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/reviewer1_remaining/tail_risk",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=ROOT / "tables/table_tail_risk.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "figures/fig_tail_risk.pdf",
    )
    args = parser.parse_args()
    counts = aggregate_tail_risk(
        args.results_root, args.output_dir, args.table_output, args.figure_output
    )
    print(
        "TAIL_RISK_REANALYSIS_PASS "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )


if __name__ == "__main__":
    main()
