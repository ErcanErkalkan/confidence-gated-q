from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.run_shift_severity_development import (  # noqa: E402
    DEFAULT_CONFIG,
    FINAL_SEED_MAX,
    FINAL_SEED_MIN,
    SeverityDevelopmentError,
    _load_yaml,
    audit_execution,
    validate_config,
)


PROTOCOL = ROOT / "configs/diagnostic_extensions/INDEPENDENT_SHIFT_PROTOCOL.yaml"
PROTOCOL_DIGEST = PROTOCOL.with_suffix(".sha256")
DEFAULT_RESULT_DIR = ROOT / "results/diagnostic_extensions/final_shifts/severity_development"
DEFAULT_RAW = DEFAULT_RESULT_DIR / "execution/raw.csv.gz"
DEFAULT_RECORD = ROOT / "configs/diagnostic_extensions/selected_shift_severities.yaml"


def select_mechanism_severity(
    candidates: pd.DataFrame,
    locked_order: list[str],
) -> tuple[str, pd.DataFrame]:
    """Apply the immutable non-inferential difficulty rule."""

    rows = []
    for order, severity_id in enumerate(locked_order):
        frame = candidates[candidates["severity_id"] == severity_id]
        values = pd.to_numeric(frame["post_shift_success_auc"], errors="coerce")
        values = values[np.isfinite(values)]
        complete = len(values) == 30
        median = float(values.median()) if len(values) else float("nan")
        floor_share = float((values <= 0.05).mean()) if len(values) else 1.0
        ceiling_share = float((values >= 0.95).mean()) if len(values) else 1.0
        nondegenerate = bool(np.isfinite(median) and 0.15 <= median <= 0.85)
        avoids_floor = floor_share < 0.90
        avoids_ceiling = ceiling_share < 0.90
        eligible = complete and nondegenerate and avoids_floor and avoids_ceiling
        rows.append(
            {
                "severity_id": severity_id,
                "locked_order": order,
                "n_agent_seed": int(len(values)),
                "pooled_median_success_auc": median,
                "distance_from_0_5": abs(median - 0.5),
                "floor_share_le_0_05": floor_share,
                "ceiling_share_ge_0_95": ceiling_share,
                "complete": complete,
                "nondegenerate_interval": nondegenerate,
                "avoids_universal_floor": avoids_floor,
                "avoids_universal_ceiling": avoids_ceiling,
                "eligible": eligible,
            }
        )
    audit = pd.DataFrame(rows).sort_values("locked_order").reset_index(drop=True)
    eligible = audit[audit["eligible"]].sort_values(
        ["distance_from_0_5", "locked_order"], kind="mergesort"
    )
    if eligible.empty:
        raise SeverityDevelopmentError("no eligible severity under the locked selection rule")
    selected = str(eligible.iloc[0]["severity_id"])
    audit["selected"] = audit["severity_id"].eq(selected)
    return selected, audit


def aggregate_candidate_results(
    raw_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        "success",
        "git_commit_hash",
    ]
    raw = pd.read_csv(raw_path, usecols=columns)
    if raw["seed"].astype(int).between(FINAL_SEED_MIN, FINAL_SEED_MAX).any():
        raise SeverityDevelopmentError("reserved final result row detected")
    evaluation = raw[raw["phase"] == "eval"].copy()
    evaluation["success"] = pd.to_numeric(evaluation["success"], errors="coerce")
    post = evaluation[evaluation["checkpoint"].between(12000, 24000)]
    means = (
        post.groupby(["environment", "agent", "seed", "checkpoint"], as_index=False)[
            "success"
        ]
        .mean()
        .sort_values(["environment", "agent", "seed", "checkpoint"])
    )
    env_lookup = {env["name"]: env for env in config["envs"]}
    source = raw_path.resolve().relative_to(ROOT.resolve()).as_posix()
    rows = []
    expected = np.arange(12000, 24001, 1000, dtype=float)
    for (environment, agent, seed), frame in means.groupby(
        ["environment", "agent", "seed"], sort=True
    ):
        checkpoints = frame["checkpoint"].to_numpy(dtype=float)
        if not np.array_equal(checkpoints, expected):
            raise SeverityDevelopmentError(
                f"post-shift checkpoint mismatch for {(environment, agent, seed)}"
            )
        values = frame["success"].to_numpy(dtype=float)
        env = env_lookup[environment]
        commit = evaluation.loc[
            (evaluation["environment"] == environment)
            & (evaluation["agent"] == agent)
            & (evaluation["seed"] == seed),
            "git_commit_hash",
        ].iloc[0]
        rows.append(
            {
                "mechanism_id": env["mechanism_id"],
                "severity_id": env["severity_id"],
                "environment": environment,
                "agent": agent,
                "seed": int(seed),
                "post_shift_success_auc": float(
                    np.trapezoid(values, checkpoints) / 12000.0
                ),
                "source_file": source,
                "execution_source_commit": str(commit),
            }
        )
    result = (
        pd.DataFrame(rows)
        .sort_values(["mechanism_id", "severity_id", "agent", "seed"])
        .reset_index(drop=True)
    )
    if len(result) != 270:
        raise SeverityDevelopmentError(
            f"severity candidate coverage mismatch: {len(result)} != 270"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the locked mechanical severity selection."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    protocol = _load_yaml(PROTOCOL)
    config = _load_yaml(DEFAULT_CONFIG)
    validate_config(config, protocol)
    # Audit the source explicitly requested by the caller.  A stale or partially
    # materialized uncompressed sibling must never shadow the locked gzip archive.
    audit_execution(args.raw, config)
    results = aggregate_candidate_results(args.raw, config)
    selected: dict[str, str] = {}
    audits = []
    locked_by_id = {item["mechanism_id"]: item for item in protocol["mechanisms"]}
    for mechanism_id, mechanism in locked_by_id.items():
        order = [
            item["severity_id"] for item in mechanism["development_severity_candidates"]
        ]
        choice, audit = select_mechanism_severity(
            results[results["mechanism_id"] == mechanism_id], order
        )
        selected[mechanism_id] = choice
        audit.insert(0, "mechanism_id", mechanism_id)
        audits.append(audit)
    audit_frame = pd.concat(audits, ignore_index=True)
    DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_RESULT_DIR / "severity_candidate_results.csv", index=False)
    audit_frame.to_csv(DEFAULT_RESULT_DIR / "severity_selection_audit.csv", index=False)
    payload = {
        "schema_version": 2,
        "status": "frozen_development_selection",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "selection_metric": "post_shift_success_auc",
        "selection_uses_p_values": False,
        "final_seed_results_used": False,
        "selected_severities": selected,
        "source_files": [
            "results/diagnostic_extensions/final_shifts/severity_development/"
            "severity_candidate_results.csv",
            "results/diagnostic_extensions/final_shifts/severity_development/"
            "severity_selection_audit.csv",
        ],
    }
    DEFAULT_RECORD.write_bytes(yaml.safe_dump(payload, sort_keys=False).encode("utf-8"))
    digest = hashlib.sha256(DEFAULT_RECORD.read_bytes()).hexdigest()
    DEFAULT_RECORD.with_suffix(".sha256").write_bytes(
        f"{digest}  {DEFAULT_RECORD.name}\n".encode("utf-8")
    )
    print(
        "SHIFT_SEVERITY_SELECTION_PASS "
        + " ".join(f"{key}={value}" for key, value in selected.items())
    )


if __name__ == "__main__":
    main()
