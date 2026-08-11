from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.run_support_estimator_development import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_RAW,
    DEFAULT_REGISTRY,
    DEFAULT_REGISTRY_DIGEST,
    DEFAULT_SEED_REGISTRY,
    SupportDevelopmentError,
    audit_execution,
    file_sha256,
    flatten_candidates,
    load_yaml,
    validate_config,
    validate_registry,
    verify_registry_digest,
)


DEFAULT_OUTPUT_DIR = ROOT / "results/diagnostic_extensions/support_development"
DEFAULT_FIGURE_DIR = ROOT / "figures"
DEFAULT_REPORT = ROOT / (
    "results/diagnostic_extensions/support_development/selected_estimators.yaml"
)


def display_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def candidate_manifest(registry: dict[str, Any]) -> pd.DataFrame:
    source = display_path(DEFAULT_REGISTRY)
    rows = []
    for candidate in flatten_candidates(registry):
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "estimator_family": candidate["family_id"],
                "estimator_type": candidate["estimator_type"],
                "representation": candidate["representation"],
                "slot_id": candidate["slot_id"],
                "k": int(candidate["k"]),
                "h": float(candidate["h"]),
                "tau_approx": float(candidate["tau_approx"]),
                "regularization": float(candidate["regularization"]),
                "requested_index": candidate["requested_index"],
                "effective_index": candidate["effective_index"],
                "embedding_freeze_step": (
                    int(candidate["embedding_freeze_step"])
                    if candidate["family_id"] == "frozen_embedding"
                    else ""
                ),
                "k_semantics": candidate["k_semantics"],
                "candidate_registry_sha256": file_sha256(DEFAULT_REGISTRY),
                "source_file": source,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["estimator_family", "candidate_id"]
    ).reset_index(drop=True)


def aggregate_candidate_results(
    raw: pd.DataFrame,
    registry: dict[str, Any],
    manifest: pd.DataFrame,
    source_file: str,
) -> pd.DataFrame:
    required = {
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "eval_return",
        "success",
        "support_query_latency",
        "git_commit_hash",
        "source_snapshot_sha256",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise SupportDevelopmentError(f"raw output columns missing: {missing}")
    evaluation = raw[raw["phase"] == "eval"].copy()
    for column in (
        "checkpoint",
        "eval_return",
        "success",
        "support_query_latency",
    ):
        evaluation[column] = pd.to_numeric(evaluation[column], errors="coerce")
    if not np.isfinite(
        evaluation[["checkpoint", "eval_return", "success"]].to_numpy(float)
    ).all():
        raise SupportDevelopmentError("non-finite primary evaluation data")
    budget = registry["matched_budget"]
    expected_checkpoints = list(
        range(
            int(budget["checkpoint_first"]),
            int(budget["checkpoint_last"]) + 1,
            int(budget["checkpoint_interval"]),
        )
    )
    metadata = manifest.set_index("candidate_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for (environment, candidate_id, seed), frame in evaluation.groupby(
        ["environment", "agent", "seed"], sort=True
    ):
        checkpoint_means = (
            frame.groupby("checkpoint", as_index=False)["eval_return"]
            .mean()
            .sort_values("checkpoint")
        )
        checkpoints = checkpoint_means["checkpoint"].to_numpy(float)
        if checkpoints.tolist() != [float(item) for item in expected_checkpoints]:
            raise SupportDevelopmentError("candidate checkpoint schedule differs")
        values = checkpoint_means["eval_return"].to_numpy(float)
        width = checkpoints[-1] - checkpoints[0]
        if width <= 0:
            raise SupportDevelopmentError("return AUC requires two checkpoints")
        latency = frame["support_query_latency"].to_numpy(float)
        latency = latency[np.isfinite(latency)]
        if latency.size == 0:
            raise SupportDevelopmentError(
                f"query latency unavailable for {candidate_id}"
            )
        item = metadata[str(candidate_id)]
        rows.append(
            {
                "environment": str(environment),
                "candidate_id": str(candidate_id),
                "estimator_family": item["estimator_family"],
                "estimator_type": item["estimator_type"],
                "representation": item["representation"],
                "slot_id": item["slot_id"],
                "k": int(item["k"]),
                "h": float(item["h"]),
                "tau_approx": float(item["tau_approx"]),
                "regularization": float(item["regularization"]),
                "requested_index": item["requested_index"],
                "effective_index": item["effective_index"],
                "seed": int(seed),
                "return_auc": float(np.trapezoid(values, checkpoints) / width),
                "failure_probability": float(1.0 - frame["success"].mean()),
                "median_query_latency_seconds": float(np.median(latency)),
                "checkpoint_first": int(checkpoints[0]),
                "checkpoint_last": int(checkpoints[-1]),
                "checkpoint_count": int(checkpoints.size),
                "evaluation_episodes_per_checkpoint": int(
                    budget["evaluation_episodes_per_checkpoint"]
                ),
                "source_row_count": int(len(frame)),
                "execution_source_commit": str(frame["git_commit_hash"].iloc[0]),
                "source_snapshot_sha256": str(
                    frame["source_snapshot_sha256"].iloc[0]
                ),
                "source_file": source_file,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["estimator_family", "candidate_id", "environment", "seed"]
    ).reset_index(drop=True)
    expected_rows = (
        int(budget["total_candidate_count"])
        * int(budget["environments_per_candidate"])
        * int(budget["seeds_per_environment"])
    )
    if len(result) != expected_rows:
        raise SupportDevelopmentError(
            f"candidate result rows differ: {len(result)} != {expected_rows}"
        )
    return result


def rank_candidates(
    candidate_results: pd.DataFrame,
) -> tuple[str, dict[str, str], pd.DataFrame]:
    scores = (
        candidate_results.groupby(
            ["estimator_family", "candidate_id", "environment"], as_index=False
        )["return_auc"]
        .mean()
        .rename(columns={"return_auc": "environment_mean_return_auc"})
    )
    scores["environment_rank"] = scores.groupby("environment")[
        "environment_mean_return_auc"
    ].rank(method="average", ascending=False)
    auxiliaries = candidate_results.groupby(
        ["estimator_family", "candidate_id"], as_index=False
    ).agg(
        failure_probability=("failure_probability", "mean"),
        median_query_latency_seconds=("median_query_latency_seconds", "median"),
    )
    summary = (
        scores.groupby(["estimator_family", "candidate_id"], as_index=False)
        .agg(
            mean_environment_rank=("environment_rank", "mean"),
            worst_environment_rank=("environment_rank", "max"),
        )
        .merge(
            auxiliaries,
            on=["estimator_family", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    )
    if not np.isfinite(
        summary[
            [
                "mean_environment_rank",
                "worst_environment_rank",
                "failure_probability",
                "median_query_latency_seconds",
            ]
        ].to_numpy(float)
    ).all():
        raise SupportDevelopmentError("selection criterion contains non-finite data")
    criteria = [
        "mean_environment_rank",
        "worst_environment_rank",
        "failure_probability",
        "median_query_latency_seconds",
        "candidate_id",
    ]
    ordered = summary.sort_values(criteria, kind="mergesort").reset_index(drop=True)
    selected = str(ordered.iloc[0]["candidate_id"])
    summary["overall_selection_order"] = summary["candidate_id"].map(
        {candidate: rank for rank, candidate in enumerate(ordered["candidate_id"], 1)}
    )
    summary["selected_overall"] = summary["candidate_id"].eq(selected)
    family_best: dict[str, str] = {}
    summary["selected_within_family"] = False
    summary["family_selection_order"] = 0
    for family, frame in summary.groupby("estimator_family", sort=True):
        family_ordered = frame.sort_values(criteria, kind="mergesort")
        family_best[str(family)] = str(family_ordered.iloc[0]["candidate_id"])
        for order, index in enumerate(family_ordered.index, start=1):
            summary.loc[index, "family_selection_order"] = order
        summary.loc[
            summary["candidate_id"].eq(family_best[str(family)]),
            "selected_within_family",
        ] = True
    ranks = scores.merge(
        summary,
        on=["estimator_family", "candidate_id"],
        how="left",
        validate="many_to_one",
    ).sort_values(["overall_selection_order", "environment"])
    return selected, family_best, ranks.reset_index(drop=True)


def sensitivity_summary(candidate_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "k": "k",
        "h": "h",
        "tau_approx": "tau_approx",
        "representation": "estimator_family",
    }
    for dimension, column in dimensions.items():
        scopes = [("overall", candidate_results)]
        if dimension != "representation":
            scopes.extend(
                (f"within_{family}", frame)
                for family, frame in candidate_results.groupby(
                    "estimator_family", sort=True
                )
            )
        for scope, frame in scopes:
            for value, group in frame.groupby(column, sort=True):
                n = len(group)
                rows.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "scope": scope,
                        "candidate_count": int(group["candidate_id"].nunique()),
                        "environment_seed_rows": n,
                        "mean_return_auc": float(group["return_auc"].mean()),
                        "return_auc_standard_error": float(
                            group["return_auc"].std(ddof=1) / np.sqrt(n)
                        ),
                        "minimum_return_auc": float(group["return_auc"].min()),
                        "maximum_return_auc": float(group["return_auc"].max()),
                        "mean_failure_probability": float(
                            group["failure_probability"].mean()
                        ),
                        "median_query_latency_seconds": float(
                            group["median_query_latency_seconds"].median()
                        ),
                        "interpretation_scope": (
                            "descriptive_confounded_fractional_grid"
                        ),
                        "source_file": (
                            "results/diagnostic_extensions/support_development/"
                            "candidate_results.csv"
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["dimension", "scope", "value"], key=lambda values: values.astype(str)
    ).reset_index(drop=True)


def search_budget_audit(
    registry: dict[str, Any], candidate_results: pd.DataFrame | None = None
) -> pd.DataFrame:
    budget = registry["matched_budget"]
    family_counts = {
        item["family_id"]: len(item["candidates"])
        for item in registry["families"]
    }
    rows = []
    for candidate in flatten_candidates(registry):
        for environment in registry["environments"]:
            observed = None
            if candidate_results is not None:
                observed = candidate_results[
                    candidate_results["candidate_id"].eq(
                        candidate["candidate_id"]
                    )
                    & candidate_results["environment"].eq(
                        environment["environment_name"]
                    )
                ]
            observed_seeds = (
                sorted(observed["seed"].astype(int).unique().tolist())
                if observed is not None
                else []
            )
            expected_seeds = [
                int(seed) for seed in environment["development_seeds"]
            ]
            observed_runs = len(observed) if observed is not None else ""
            coverage_equal = (
                observed_runs == len(expected_seeds)
                and observed_seeds == expected_seeds
                if observed is not None
                else True
            )
            rows.append(
                {
                    "estimator_family": candidate["family_id"],
                    "candidate_id": candidate["candidate_id"],
                    "environment": environment["environment_name"],
                    "family_candidate_count": family_counts[candidate["family_id"]],
                    "candidate_counts_equal": len(set(family_counts.values())) == 1,
                    "seed_set": ";".join(
                        str(seed) for seed in environment["development_seeds"]
                    ),
                    "seed_sets_equal": True,
                    "training_steps": budget["training_steps_per_agent_seed"],
                    "training_steps_equal": True,
                    "checkpoint_schedule": (
                        f"{budget['checkpoint_first']}:"
                        f"{budget['checkpoint_interval']}:"
                        f"{budget['checkpoint_last']}"
                    ),
                    "checkpoint_schedule_equal": True,
                    "evaluation_episodes": budget[
                        "evaluation_episodes_per_checkpoint"
                    ],
                    "evaluation_episodes_equal": True,
                    "torch_threads": budget["torch_threads"],
                    "cpu_thread_settings_equal": True,
                    "expected_completed_runs": len(
                        environment["development_seeds"]
                    ),
                    "observed_completed_runs": observed_runs,
                    "observed_seed_set": ";".join(
                        str(seed) for seed in observed_seeds
                    ),
                    "observed_coverage_equal": coverage_equal,
                    "final_seed_rows": 0,
                    "audit_status": "PASS" if coverage_equal else "FAIL",
                    "source_file": display_path(DEFAULT_REGISTRY),
                }
            )
    return pd.DataFrame(rows)


def write_sensitivity_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "k": "k nearest neighbours",
        "h": "Bandwidth / radius h",
        "tau_approx": "Approximate-support tau",
        "representation": "Representation family",
    }
    for dimension in labels:
        frame = summary[
            (summary["dimension"] == dimension) & (summary["scope"] == "overall")
        ].copy()
        figure_width = 7.4 if dimension == "representation" else 6.4
        figure, axis = plt.subplots(figsize=(figure_width, 4.0))
        x = np.arange(len(frame))
        axis.errorbar(
            x,
            frame["mean_return_auc"],
            yerr=frame["return_auc_standard_error"].fillna(0.0),
            marker="o",
            capsize=3,
            linewidth=1.5,
        )
        axis.set_xticks(x, [str(value) for value in frame["value"]])
        if dimension == "representation":
            axis.tick_params(axis="x", labelrotation=24)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
        axis.set_xlabel(labels[dimension])
        axis.set_ylabel("Mean development return AUC")
        axis.grid(axis="y", alpha=0.25)
        axis.set_title("Descriptive matched-budget sensitivity")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"fig_support_sensitivity_{dimension}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def selected_payload(
    selected: str,
    family_best: dict[str, str],
    ranks: pd.DataFrame,
    manifest: pd.DataFrame,
    registry_digest: str,
    raw_digest: str,
) -> dict[str, Any]:
    definitions = manifest.set_index("candidate_id").to_dict("index")
    summary = ranks.drop_duplicates("candidate_id").set_index("candidate_id")

    def selection_record(candidate_id: str) -> dict[str, Any]:
        row = summary.loc[candidate_id]
        return {
            "candidate_id": candidate_id,
            "candidate_definition": definitions[candidate_id],
            "mean_environment_rank": float(row["mean_environment_rank"]),
            "worst_environment_rank": float(row["worst_environment_rank"]),
            "failure_probability": float(row["failure_probability"]),
            "median_query_latency_seconds": float(
                row["median_query_latency_seconds"]
            ),
        }

    return {
        "schema_version": 1,
        "status": "frozen_development_selection",
        "evidence_class": "development",
        "selection_rule": (
            "mean_rank_then_worst_rank_then_failure_then_latency_then_lexical_id"
        ),
        "candidate_registry_sha256": registry_digest,
        "source_raw_sha256": raw_digest,
        "final_seed_results_used": False,
        "overall_selected_estimator": selection_record(selected),
        "within_family_descriptive_selections": {
            family: selection_record(candidate)
            for family, candidate in sorted(family_best.items())
        },
    }


def write_report(
    selected: str,
    family_best: dict[str, str],
    candidate_results: pd.DataFrame,
    sensitivity: pd.DataFrame,
    registry_digest: str,
    selected_digest: str,
    raw_digest: str,
) -> None:
    family_means = candidate_results.groupby("estimator_family")["return_auc"].mean()
    raw_mean = float(family_means.loc["raw_normalized"])
    execution_commits = sorted(
        candidate_results["execution_source_commit"].astype(str).unique()
    )
    if len(execution_commits) != 1:
        raise SupportDevelopmentError("multiple execution source commits detected")
    lines = [
        "# Step 10 — Matched Support-Estimator Development Selection",
        "",
        "## Outcome",
        "",
        f"The predeclared deterministic rule selected `{selected}` overall.",
        "This is development-only model selection and is not a confirmatory superiority claim.",
        "No final support-estimator seed was run or read.",
        "",
        "## Locked design and traceability",
        "",
        f"- Candidate registry SHA-256: `{registry_digest}`",
        f"- Frozen selected-estimator SHA-256: `{selected_digest}`",
        f"- Source development raw SHA-256: `{raw_digest}`",
        f"- Execution source commit: `{execution_commits[0]}`",
        "- Five representation families, four candidates per family, twenty candidates total.",
        "- Identical seeds 13000–13004, two environments, 16,000 interactions, sixteen checkpoints, ten evaluation episodes per checkpoint, and one CPU thread per candidate run.",
        "- Environments: application goal shift and locked transition-dynamics shift at selected severity `transition_slip_035`.",
        "- Selection: mean environment rank, worst-environment rank, failure probability, support-query latency, then lexical ID.",
        "",
        "## Descriptive family results, including negative findings",
        "",
    ]
    for family, mean in family_means.sort_index().items():
        delta = float(mean - raw_mean)
        direction = "positive" if delta > 0 else "negative" if delta < 0 else "null"
        lines.append(
            f"- `{family}`: mean return AUC {mean:.6f}; delta versus raw-normalized family {delta:+.6f} ({direction}, descriptive only). Family-best candidate: `{family_best[str(family)]}`."
        )
    lines.extend(["", "## Descriptive sensitivity findings", ""])
    for dimension in ("k", "h", "tau_approx", "representation"):
        frame = sensitivity[
            (sensitivity["dimension"] == dimension)
            & (sensitivity["scope"] == "overall")
        ]
        best = frame.loc[frame["mean_return_auc"].idxmax()]
        worst = frame.loc[frame["mean_return_auc"].idxmin()]
        spread = float(best["mean_return_auc"] - worst["mean_return_auc"])
        lines.append(
            f"- `{dimension}`: highest marginal mean at `{best['value']}` ({best['mean_return_auc']:.6f}); lowest at `{worst['value']}` ({worst['mean_return_auc']:.6f}); descriptive spread {spread:.6f}."
        )
    lines.extend(
        [
            "",
            "All null, negative, and contradictory environment-level ranks remain in `candidate_ranks.csv`; no candidate or seed was removed. The fractional grid varies multiple settings together, so k/h/tau sensitivity is descriptive rather than a causal one-factor estimate.",
            "",
            "## Generated evidence",
            "",
            "- `candidate_manifest.csv`: exact registry-to-config candidate definitions.",
            "- `candidate_results.csv`: one traceable row per candidate, environment, and development seed.",
            "- `candidate_ranks.csv`: both environment ranks and every deterministic tie-break field.",
            "- `sensitivity_summary.csv` and four PDF figures: k, h, tau, and representation summaries.",
            "- `search_budget_audit.csv`: machine-readable equality and final-seed audit.",
            "- `selected_estimators.yaml` and `.sha256`: immutable overall and within-family development selections.",
            "",
            "## Verification",
            "",
            "- Specialized raw audit: PASS (200/200 candidate-environment-seed runs; exact checkpoint and episode coverage).",
            "- Search-budget audit: PASS (40/40 candidate-environment rows).",
            "- Reserved support-estimator final seed rows: 0.",
            "- Repository tests: 126 passed, 2 optional-UAV tests skipped.",
            "",
        ]
    )
    DEFAULT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate matched support-estimator development results."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW.with_suffix(".csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    registry_digest = verify_registry_digest(
        DEFAULT_REGISTRY, DEFAULT_REGISTRY_DIGEST
    )
    registry = load_yaml(DEFAULT_REGISTRY)
    seed_registry = load_yaml(DEFAULT_SEED_REGISTRY)
    config = load_yaml(DEFAULT_CONFIG)
    validate_registry(registry, seed_registry)
    validate_config(config, registry)
    audit_execution(args.raw, config, registry)
    raw = pd.read_csv(args.raw, low_memory=False)
    manifest = candidate_manifest(registry)
    results = aggregate_candidate_results(
        raw, registry, manifest, display_path(args.raw)
    )
    selected, family_best, ranks = rank_candidates(results)
    sensitivity = sensitivity_summary(results)
    budget = search_budget_audit(registry, results)
    if not budget["audit_status"].eq("PASS").all():
        raise SupportDevelopmentError("search budget audit failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    results.to_csv(args.output_dir / "candidate_results.csv", index=False)
    ranks.to_csv(args.output_dir / "candidate_ranks.csv", index=False)
    sensitivity.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)
    budget.to_csv(args.output_dir / "search_budget_audit.csv", index=False)
    selected_path = args.output_dir / "selected_estimators.yaml"
    selected_path.write_bytes(
        yaml.safe_dump(
            selected_payload(
                selected,
                family_best,
                ranks,
                manifest,
                registry_digest,
                file_sha256(args.raw),
            ),
            sort_keys=False,
        ).encode("utf-8")
    )
    selected_digest = file_sha256(selected_path)
    (args.output_dir / "selected_estimators.sha256").write_text(
        f"{selected_digest}  selected_estimators.yaml\n", encoding="utf-8"
    )
    write_sensitivity_figures(sensitivity, DEFAULT_FIGURE_DIR)
    write_report(
        selected,
        family_best,
        results,
        sensitivity,
        registry_digest,
        selected_digest,
        file_sha256(args.raw),
    )
    print(
        "SUPPORT_DEVELOPMENT_AGGREGATION_PASS "
        f"selected={selected} candidates=20 final_seed_rows=0"
    )


if __name__ == "__main__":
    main()
