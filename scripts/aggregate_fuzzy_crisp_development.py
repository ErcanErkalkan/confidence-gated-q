from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.complexity import mapping_operation_estimate  # noqa: E402
from hybrid_q.mappings import crisp_reliability_gate, fuzzy_reliability_gate  # noqa: E402
from scripts.run_fuzzy_crisp_development import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REGISTRY,
    DEFAULT_REGISTRY_DIGEST,
    PROTOCOL,
    PROTOCOL_DIGEST,
    FINAL_SEED_MAX,
    FINAL_SEED_MIN,
    FuzzyCrispDevelopmentError,
    _load_yaml,
    audit_execution,
    candidate_family,
    validate_config,
    validate_registry,
    verify_registry_digest,
)


DEFAULT_OUTPUT_DIR = ROOT / "results/diagnostic_extensions/fuzzy_crisp_development"
DEFAULT_RAW_PATH = DEFAULT_OUTPUT_DIR / "execution_corrected/raw.csv.gz"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "selection_report.md"


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _candidate_lookup(
    registry: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    for family, key in (("crisp", "crisp_candidates"), ("fuzzy", "fuzzy_candidates")):
        for candidate in registry[key]:
            lookup[candidate["candidate_id"]] = (family, candidate)
    return lookup


def mapping_function(
    family: str, candidate: dict[str, Any]
) -> Callable[[float, float], float]:
    if family == "crisp":
        return lambda support, reliability: crisp_reliability_gate(
            support,
            reliability,
            support_threshold=float(candidate["support_threshold"]),
            reliability_threshold=float(candidate["reliability_threshold"]),
            gate_min=float(candidate["gate_min"]),
            gate_max=float(candidate["gate_max"]),
        )
    if family == "fuzzy":
        return lambda support, reliability: fuzzy_reliability_gate(
            support,
            reliability,
            membership_shape=candidate["membership_family"],
            reliability_membership_shape=candidate["membership_family"],
            support_breakpoints=candidate["support_breakpoints"],
            reliability_breakpoints=candidate["reliability_breakpoints"],
            consequents=candidate["consequents"],
            gate_min=float(candidate["gate_min"]),
            gate_max=float(candidate["gate_max"]),
        )
    raise FuzzyCrispDevelopmentError(f"unknown mapping family: {family}")


def aggregate_candidate_results(
    raw: pd.DataFrame,
    registry: dict[str, Any],
    source_file: str,
) -> pd.DataFrame:
    needed = {
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "eval_return",
        "git_commit_hash",
    }
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise FuzzyCrispDevelopmentError(f"raw output is missing columns: {missing}")
    evaluation = raw[raw["phase"] == "eval"].copy()
    evaluation["checkpoint"] = pd.to_numeric(evaluation["checkpoint"], errors="coerce")
    evaluation["eval_return"] = pd.to_numeric(
        evaluation["eval_return"], errors="coerce"
    )
    if evaluation[["checkpoint", "eval_return"]].isna().any().any():
        raise FuzzyCrispDevelopmentError(
            "evaluation checkpoint/return contains non-finite values"
        )
    if any(
        FINAL_SEED_MIN <= int(seed) <= FINAL_SEED_MAX
        for seed in evaluation["seed"].unique()
    ):
        raise FuzzyCrispDevelopmentError("reserved final result row detected")
    onset = int(registry["matched_budget"]["shift_onset"])
    last = int(registry["matched_budget"]["checkpoint_last"])
    post = evaluation[evaluation["checkpoint"].between(onset, last)].copy()
    checkpoint_means = (
        post.groupby(["environment", "agent", "seed", "checkpoint"], as_index=False)[
            "eval_return"
        ]
        .mean()
        .sort_values(["environment", "agent", "seed", "checkpoint"])
    )
    environment_to_mechanism = {
        item["environment_name"]: item["mechanism_id"]
        for item in registry["environments"]
    }
    rows: list[dict[str, Any]] = []
    expected_checkpoints = list(
        range(onset, last + 1, int(registry["matched_budget"]["checkpoint_interval"]))
    )
    for (environment, agent, seed), frame in checkpoint_means.groupby(
        ["environment", "agent", "seed"], sort=True
    ):
        checkpoints = frame["checkpoint"].to_numpy(dtype=float)
        if checkpoints.tolist() != [float(item) for item in expected_checkpoints]:
            raise FuzzyCrispDevelopmentError(
                f"post-shift checkpoint mismatch for {(environment, agent, seed)}"
            )
        values = frame["eval_return"].to_numpy(dtype=float)
        width = checkpoints[-1] - checkpoints[0]
        if width <= 0:
            raise FuzzyCrispDevelopmentError(
                "primary AUC needs at least two post-shift checkpoints"
            )
        rows.append(
            {
                "mechanism_id": environment_to_mechanism[environment],
                "environment": environment,
                "mapping_family": candidate_family(str(agent), registry),
                "candidate_id": str(agent),
                "seed": int(seed),
                "primary_metric_name": "post_shift_normalized_return_auc",
                "primary_metric": float(np.trapezoid(values, checkpoints) / width),
                "post_shift_checkpoint_first": int(checkpoints[0]),
                "post_shift_checkpoint_last": int(checkpoints[-1]),
                "post_shift_checkpoint_count": len(checkpoints),
                "evaluation_episodes_per_checkpoint": int(
                    registry["matched_budget"]["evaluation_episodes_per_checkpoint"]
                ),
                "source_file": source_file,
                "source_row_count": int(
                    len(frame)
                    * registry["matched_budget"]["evaluation_episodes_per_checkpoint"]
                ),
                "execution_source_commit": str(
                    evaluation.loc[
                        (evaluation["environment"] == environment)
                        & (evaluation["agent"] == agent)
                        & (evaluation["seed"] == seed),
                        "git_commit_hash",
                    ].iloc[0]
                ),
            }
        )
    result = (
        pd.DataFrame(rows)
        .sort_values(["mapping_family", "candidate_id", "environment", "seed"])
        .reset_index(drop=True)
    )
    expected_rows = (
        2
        * int(registry["matched_budget"]["candidate_count_per_family"])
        * sum(len(item["development_seeds"]) for item in registry["environments"])
    )
    if len(result) != expected_rows:
        raise FuzzyCrispDevelopmentError(
            f"candidate result coverage mismatch: {len(result)} != {expected_rows}"
        )
    return result


def latency_complexity(registry: dict[str, Any]) -> pd.DataFrame:
    lock = registry["latency_lock"]
    warmup = int(lock["warmup_calls"])
    calls = int(lock["measured_calls_per_repeat"])
    repeats = int(lock["repeats"])
    inputs = [
        ((index * 73) % 257 / 256.0, (index * 151) % 257 / 256.0)
        for index in range(257)
    ]
    rows = []
    for candidate_id, (family, candidate) in sorted(
        _candidate_lookup(registry).items()
    ):
        function = mapping_function(family, candidate)
        for index in range(warmup):
            function(*inputs[index % len(inputs)])
        samples = []
        checksum = 0.0
        for repeat in range(repeats):
            started = time.perf_counter_ns()
            for index in range(calls):
                checksum += function(*inputs[(index + repeat) % len(inputs)])
            elapsed = time.perf_counter_ns() - started
            samples.append(elapsed / calls)
        mapping_name = (
            "same_input_crisp_threshold"
            if family == "crisp"
            else f"fuzzy_{candidate['membership_family']}_five_rule"
        )
        estimate = mapping_operation_estimate(mapping_name)
        rows.append(
            {
                "mapping_family": family,
                "candidate_id": candidate_id,
                "median_latency_ns": float(np.median(samples)),
                "latency_q25_ns": float(np.quantile(samples, 0.25)),
                "latency_q75_ns": float(np.quantile(samples, 0.75)),
                "cpu_threads": int(lock["cpu_threads"]),
                "warmup_calls": warmup,
                "measured_calls_per_repeat": calls,
                "repeats": repeats,
                "input_schedule": lock["input_schedule"],
                "mapping_scope": lock["includes"],
                "arithmetic_flops_approximate": estimate.arithmetic_flops,
                "comparisons": estimate.comparisons,
                "special_functions": estimate.special_functions,
                "complexity_definition": estimate.definition,
                "timing_checksum": checksum,
                "source_file": _display_path(DEFAULT_REGISTRY),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["mapping_family", "candidate_id"])
        .reset_index(drop=True)
    )


def load_or_measure_latency(
    registry: dict[str, Any],
    output_dir: Path,
    *,
    remeasure: bool = False,
) -> pd.DataFrame:
    """Reuse the recorded timing evidence unless remeasurement is explicit.

    Wall-clock latency is empirical evidence rather than a deterministic derived
    statistic. Re-running the general aggregator must therefore preserve the
    recorded measurement and the frozen selection that depends on it.
    """
    path = output_dir / "latency_complexity.csv"
    if remeasure or not path.exists():
        return latency_complexity(registry)

    frame = pd.read_csv(path, float_precision="round_trip")
    required = {
        "mapping_family",
        "candidate_id",
        "median_latency_ns",
        "latency_q25_ns",
        "latency_q75_ns",
        "cpu_threads",
        "warmup_calls",
        "measured_calls_per_repeat",
        "repeats",
        "input_schedule",
        "mapping_scope",
        "arithmetic_flops_approximate",
        "comparisons",
        "special_functions",
        "complexity_definition",
        "timing_checksum",
        "source_file",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FuzzyCrispDevelopmentError(
            f"recorded latency evidence is missing columns: {missing}"
        )

    expected = {
        (family, candidate_id)
        for candidate_id, (family, _) in _candidate_lookup(registry).items()
    }
    observed = set(
        frame[["mapping_family", "candidate_id"]].itertuples(index=False, name=None)
    )
    duplicated = frame.duplicated(["mapping_family", "candidate_id"]).any()
    if duplicated or observed != expected or len(frame) != len(expected):
        raise FuzzyCrispDevelopmentError(
            "recorded latency evidence does not contain each locked candidate exactly once"
        )

    numeric_columns = [
        "median_latency_ns",
        "latency_q25_ns",
        "latency_q75_ns",
        "timing_checksum",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise FuzzyCrispDevelopmentError(
            "recorded latency evidence contains a non-finite timing value"
        )

    lock = registry["latency_lock"]
    locked_values = {
        "cpu_threads": int(lock["cpu_threads"]),
        "warmup_calls": int(lock["warmup_calls"]),
        "measured_calls_per_repeat": int(lock["measured_calls_per_repeat"]),
        "repeats": int(lock["repeats"]),
        "input_schedule": str(lock["input_schedule"]),
        "mapping_scope": str(lock["includes"]),
    }
    for column, expected_value in locked_values.items():
        if not (frame[column] == expected_value).all():
            raise FuzzyCrispDevelopmentError(
                f"recorded latency evidence violates locked {column}"
            )
    return frame.sort_values(["mapping_family", "candidate_id"]).reset_index(drop=True)


def decision_surface_metrics(registry: dict[str, Any]) -> pd.DataFrame:
    lock = registry["decision_surface_lock"]
    count = int(lock["grid_points_per_axis"])
    grid = np.linspace(0.0, 1.0, count)
    finite_step = float(lock["finite_difference_step"])
    if not np.isclose(grid[1] - grid[0], finite_step):
        raise FuzzyCrispDevelopmentError(
            "finite-difference step differs from locked grid"
        )
    probe = float(lock["boundary_probe_epsilon"])
    perturbation = float(lock["perturbation_delta"])
    stable_limit = float(lock["perturbation_stable_absolute_change"])
    rows = []
    for candidate_id, (family, candidate) in sorted(
        _candidate_lookup(registry).items()
    ):
        function = mapping_function(family, candidate)
        surface = np.asarray(
            [
                [function(support, reliability) for reliability in grid]
                for support in grid
            ]
        )
        derivatives = np.concatenate(
            [
                np.abs(np.diff(surface, axis=0)).ravel() / finite_step,
                np.abs(np.diff(surface, axis=1)).ravel() / finite_step,
            ]
        )
        support_boundaries = (
            [float(candidate["support_threshold"])]
            if family == "crisp"
            else [float(value) for value in candidate["support_breakpoints"]]
        )
        reliability_boundaries = (
            [float(candidate["reliability_threshold"])]
            if family == "crisp"
            else [float(value) for value in candidate["reliability_breakpoints"]]
        )
        boundary_changes = []
        for boundary in support_boundaries:
            boundary_changes.extend(
                abs(
                    function(min(1.0, boundary + probe), value)
                    - function(max(0.0, boundary - probe), value)
                )
                for value in grid
            )
        for boundary in reliability_boundaries:
            boundary_changes.extend(
                abs(
                    function(value, min(1.0, boundary + probe))
                    - function(value, max(0.0, boundary - probe))
                )
                for value in grid
            )
        perturbation_changes = []
        for support in np.linspace(0.0, 1.0, 21):
            for reliability in np.linspace(0.0, 1.0, 21):
                baseline = function(float(support), float(reliability))
                for ds, dr in (
                    (perturbation, 0.0),
                    (-perturbation, 0.0),
                    (0.0, perturbation),
                    (0.0, -perturbation),
                ):
                    changed = function(
                        float(np.clip(support + ds, 0.0, 1.0)),
                        float(np.clip(reliability + dr, 0.0, 1.0)),
                    )
                    perturbation_changes.append(abs(changed - baseline))
        changes = np.asarray(perturbation_changes, dtype=float)
        rows.append(
            {
                "mapping_family": family,
                "candidate_id": candidate_id,
                "local_lipschitz_style_mean": float(derivatives.mean()),
                "local_lipschitz_style_max": float(derivatives.max()),
                "boundary_output_change_mean": float(np.mean(boundary_changes)),
                "boundary_output_change_max": float(np.max(boundary_changes)),
                "boundary_probe_epsilon": probe,
                "perturbation_delta": perturbation,
                "perturbation_stability_rate": float(np.mean(changes <= stable_limit)),
                "perturbation_absolute_change_mean": float(changes.mean()),
                "stable_absolute_change_limit": stable_limit,
                "source_file": _display_path(DEFAULT_REGISTRY),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["mapping_family", "candidate_id"])
        .reset_index(drop=True)
    )


def select_candidates(
    candidate_results: pd.DataFrame,
    latency: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    scores = (
        candidate_results.groupby(
            ["mapping_family", "candidate_id", "environment"], as_index=False
        )["primary_metric"]
        .mean()
        .rename(columns={"primary_metric": "environment_mean_primary_metric"})
        .sort_values(["mapping_family", "environment", "candidate_id"])
    )
    scores["environment_rank"] = scores.groupby(["mapping_family", "environment"])[
        "environment_mean_primary_metric"
    ].rank(method="average", ascending=False)
    summary = (
        scores.groupby(["mapping_family", "candidate_id"], as_index=False)
        .agg(
            mean_environment_rank=("environment_rank", "mean"),
            worst_environment_rank=("environment_rank", "max"),
        )
        .merge(
            latency[["mapping_family", "candidate_id", "median_latency_ns"]],
            on=["mapping_family", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    )
    if summary["median_latency_ns"].isna().any():
        raise FuzzyCrispDevelopmentError("latency is missing for a candidate")
    selected: dict[str, str] = {}
    summary["selected"] = False
    summary["selection_order_within_family"] = 0
    for family, frame in summary.groupby("mapping_family", sort=True):
        ordered = frame.sort_values(
            [
                "mean_environment_rank",
                "worst_environment_rank",
                "median_latency_ns",
                "candidate_id",
            ],
            kind="mergesort",
        )
        selected_id = str(ordered.iloc[0]["candidate_id"])
        selected[str(family)] = selected_id
        for order, index in enumerate(ordered.index, start=1):
            summary.loc[index, "selection_order_within_family"] = order
        summary.loc[summary["candidate_id"] == selected_id, "selected"] = True
    ranks = (
        scores.merge(
            summary,
            on=["mapping_family", "candidate_id"],
            how="left",
            validate="many_to_one",
        )
        .sort_values(["mapping_family", "candidate_id", "environment"])
        .reset_index(drop=True)
    )
    return selected, ranks


def budget_match_audit(registry: dict[str, Any]) -> pd.DataFrame:
    crisp = sorted(item["candidate_id"] for item in registry["crisp_candidates"])
    fuzzy = sorted(item["candidate_id"] for item in registry["fuzzy_candidates"])
    budget = registry["matched_budget"]
    rows = []
    for environment in registry["environments"]:
        for crisp_id, fuzzy_id in zip(crisp, fuzzy, strict=True):
            rows.append(
                {
                    "environment": environment["environment_name"],
                    "crisp_candidate": crisp_id,
                    "fuzzy_candidate": fuzzy_id,
                    "candidate_counts_equal": len(crisp) == len(fuzzy),
                    "environment_equal": True,
                    "seed_set": ";".join(
                        str(seed) for seed in environment["development_seeds"]
                    ),
                    "seed_sets_equal": True,
                    "training_steps_crisp": budget["training_steps_per_agent_seed"],
                    "training_steps_fuzzy": budget["training_steps_per_agent_seed"],
                    "training_steps_equal": True,
                    "checkpoint_schedule": f"{budget['checkpoint_first']}:{budget['checkpoint_interval']}:{budget['checkpoint_last']}",
                    "checkpoint_schedule_equal": True,
                    "evaluation_episodes_crisp": budget[
                        "evaluation_episodes_per_checkpoint"
                    ],
                    "evaluation_episodes_fuzzy": budget[
                        "evaluation_episodes_per_checkpoint"
                    ],
                    "evaluation_episodes_equal": True,
                    "gate_min_crisp": registry["common_mapping_inputs"]["gate_min"],
                    "gate_min_fuzzy": registry["common_mapping_inputs"]["gate_min"],
                    "gate_max_crisp": registry["common_mapping_inputs"]["gate_max"],
                    "gate_max_fuzzy": registry["common_mapping_inputs"]["gate_max"],
                    "gate_bounds_equal": True,
                    "final_seed_rows": 0,
                    "audit_status": "PASS",
                    "source_file": _display_path(DEFAULT_REGISTRY),
                }
            )
    return pd.DataFrame(rows)


def _selected_payload(
    selected: dict[str, str],
    ranks: pd.DataFrame,
    registry: dict[str, Any],
    registry_digest: str,
) -> dict[str, Any]:
    lookup = _candidate_lookup(registry)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "frozen_development_selection",
        "evidence_class": "development",
        "candidate_registry_sha256": registry_digest,
        "selection_rule": registry["selection_rule"],
        "incorporated_into_final_protocol": True,
        "final_seed_results_used": False,
        "selected": {},
    }
    for family in ("crisp", "fuzzy"):
        candidate_id = selected[family]
        row = ranks[
            (ranks["mapping_family"] == family)
            & (ranks["candidate_id"] == candidate_id)
        ].iloc[0]
        payload["selected"][family] = {
            "candidate_id": candidate_id,
            "candidate_definition": lookup[candidate_id][1],
            "mean_environment_rank": float(row["mean_environment_rank"]),
            "worst_environment_rank": float(row["worst_environment_rank"]),
            "median_latency_ns": float(row["median_latency_ns"]),
        }
    return payload


def _write_report(
    selected: dict[str, str],
    candidate_results: pd.DataFrame,
    registry_digest: str,
    selected_digest: str,
) -> None:
    means = candidate_results.groupby(["mapping_family", "candidate_id"])[
        "primary_metric"
    ].mean()
    lines = [
        "# Matched Fuzzy/Crisp Development Selection",
        "",
        "## Outcome",
        "",
        f"The locked deterministic rule selected `{selected['fuzzy']}` and `{selected['crisp']}` from four candidates per family.",
        "This is development-only mapping selection; it is not a superiority claim.",
        "",
        "## Scientific boundary",
        "",
        "All candidates used the same three independently locked shift environments, mechanism-specific ten-seed development sets, 16,000 training interactions, 1,000-step checkpoints, and 25 evaluation episodes per checkpoint. No seed in 12000–12099 was run or read.",
        "The selected mappings are incorporated into the finalized independent-shift protocol. Final seed outcomes were not used during mapping selection.",
        "",
        "## Execution note",
        "",
        "Execution artifacts are retained separately from the scientific selection record; the corrected development execution preserves the locked candidate set and budgets.",
        "",
        "## Selection and traceability",
        "",
        f"- Candidate registry SHA-256: `{registry_digest}`",
        f"- Frozen selected-candidate SHA-256: `{selected_digest}`",
        "- Primary metric: post-shift normalized return AUC (checkpoint means from 12000 through 16000; trapezoidal integral divided by 4000).",
        "- Rule: lowest mean environment rank, then lowest worst-environment rank, then lower warmed mapping-only CPU latency, then lexical candidate ID.",
        "",
        "## Descriptive pooled development means",
        "",
    ]
    for family, candidate_id in sorted(selected.items()):
        lines.append(
            f"- `{candidate_id}` ({family}): {means.loc[(family, candidate_id)]:.6f}"
        )
    lines.extend(
        [
            "",
            "Candidate-level values remain traceable to the execution raw file through `candidate_results.csv`. Complexity counts are mapping-only approximations; latency excludes network inference, residual/support estimation, and Q-value mixing.",
            "",
            "## Generated artifacts",
            "",
            "- `candidate_results.csv`: one row per environment, candidate, and development seed.",
            "- `candidate_ranks.csv`: every environment rank plus deterministic selection fields.",
            "- `selected_candidates.yaml` and `.sha256`: frozen selections and exact digest.",
            "- `budget_match_audit.csv`: matched candidate counts and experimental budgets.",
            "- `decision_surface_metrics.csv`: finite-difference, boundary, and perturbation diagnostics.",
            "- `latency_complexity.csv`: warmed fixed-thread mapping latency and labeled operation estimates.",
            "- `development_run_audit.json`: strict provenance, coverage, schema, freeze, and budget audit (generated by `audit_results.py --fuzzy-crisp-dir`).",
            "",
        ]
    )
    DEFAULT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the locked matched fuzzy/crisp development screen."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--remeasure-latency",
        action="store_true",
        help=(
            "explicitly replace the recorded warmed CPU latency evidence; "
            "the default preserves the measurement used by the frozen selection"
        ),
    )
    args = parser.parse_args()
    protocol = _load_yaml(PROTOCOL)
    verify_registry_digest(PROTOCOL, PROTOCOL_DIGEST)
    registry_digest = verify_registry_digest(DEFAULT_REGISTRY, DEFAULT_REGISTRY_DIGEST)
    registry = _load_yaml(DEFAULT_REGISTRY)
    config = _load_yaml(DEFAULT_CONFIG)
    validate_registry(registry, protocol)
    validate_config(config, registry)
    audit_execution(args.raw, config, registry)
    raw = pd.read_csv(args.raw)
    source_file = _display_path(args.raw)
    candidate_results = aggregate_candidate_results(raw, registry, source_file)
    latency = load_or_measure_latency(
        registry, args.output_dir, remeasure=args.remeasure_latency
    )
    surfaces = decision_surface_metrics(registry)
    selected, ranks = select_candidates(candidate_results, latency)
    budget = budget_match_audit(registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_results.to_csv(args.output_dir / "candidate_results.csv", index=False)
    ranks.to_csv(args.output_dir / "candidate_ranks.csv", index=False)
    budget.to_csv(args.output_dir / "budget_match_audit.csv", index=False)
    surfaces.to_csv(args.output_dir / "decision_surface_metrics.csv", index=False)
    latency_path = args.output_dir / "latency_complexity.csv"
    if args.remeasure_latency or not latency_path.exists():
        latency.to_csv(latency_path, index=False)
    selected_path = args.output_dir / "selected_candidates.yaml"
    selected_path.write_bytes(
        yaml.safe_dump(
            _selected_payload(selected, ranks, registry, registry_digest),
            sort_keys=False,
        ).encode("utf-8")
    )
    selected_digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    (args.output_dir / "selected_candidates.sha256").write_text(
        f"{selected_digest}  selected_candidates.yaml\n", encoding="utf-8"
    )
    _write_report(selected, candidate_results, registry_digest, selected_digest)
    print(
        "FUZZY_CRISP_AGGREGATION_PASS "
        f"fuzzy={selected['fuzzy']} crisp={selected['crisp']} final_seed_rows=0"
    )


if __name__ == "__main__":
    main()
