from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from hybrid_q.config import load_config
from hybrid_q.provenance import (
    execution_input_manifest,
    execution_snapshot_sha256,
)


REQUIRED_COLUMNS = {
    "experiment_name",
    "experiment",
    "config_file",
    "environment",
    "environment_id",
    "resolved_environment_id",
    "observation_representation",
    "agent",
    "seed",
    "phase",
    "checkpoint",
    "episode",
    "return",
    "length",
    "success",
    "environment_steps",
    "gradient_updates",
    "elapsed_seconds",
    "training_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "git_commit_hash",
    "package_version",
    "python_version",
    "torch_version",
    "numpy_version",
    "gymnasium_version",
    "minigrid_version",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checkpoints(config: dict, env_spec: dict) -> set[int]:
    budget = int(env_spec["training_steps"])
    interval = int(config["evaluation"]["interval_steps"])
    checkpoints = set(range(interval, budget + 1, interval))
    checkpoints.add(budget)
    return checkpoints


def raw_result_paths(result_dir: Path) -> list[Path]:
    for name in ("raw.csv", "raw.csv.gz", "raw.csv.xz"):
        candidate = result_dir / name
        if candidate.exists():
            return [candidate]
    parts = sorted((result_dir / "raw_parts").glob("*.csv*"))
    if parts:
        return parts
    raise FileNotFoundError(
        f"Missing raw.csv, compressed raw.csv, or raw_parts in {result_dir}"
    )


def audit(config_path: Path, result_dir: Path) -> dict:
    config = load_config(config_path)
    metadata = json.loads(
        (result_dir / "metadata.json").read_text(encoding="utf-8")
    )
    violations = []
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if metadata.get("config_sha256") != config_hash:
        violations.append("metadata config hash does not match config")
    warnings = []
    provenance_status = "LEGACY_COMMIT_ONLY"
    recorded_snapshot = metadata.get("source_snapshot_sha256")
    if recorded_snapshot:
        current_manifest = execution_input_manifest(config_path)
        current_snapshot = execution_snapshot_sha256(current_manifest)
        provenance_status = "STRICT_PASS"
        if metadata.get("execution_inputs_clean") is not True:
            violations.append("execution inputs were not clean at run start")
            provenance_status = "STRICT_FAIL"
        if recorded_snapshot != current_snapshot:
            violations.append(
                "recorded source snapshot does not match current execution inputs"
            )
            provenance_status = "STRICT_FAIL"
        if metadata.get("execution_input_manifest") != current_manifest:
            violations.append(
                "recorded execution input manifest does not match current inputs"
            )
            provenance_status = "STRICT_FAIL"
    else:
        warnings.append(
            "legacy result: source snapshot unavailable; commit-only provenance"
        )

    runs_dir = result_dir / "runs"
    temporary_files = list(runs_dir.glob("*.tmp"))
    if temporary_files:
        violations.append(f"incomplete run shards: {len(temporary_files)}")

    expected_runs = sum(
        len(spec.get("seeds", config["seeds"])) * len(config["agents"])
        for spec in config["envs"]
    )
    shards = list(runs_dir.glob("*.csv"))
    if (shards or temporary_files) and len(shards) != expected_runs:
        violations.append(
            f"run shard count {len(shards)} != expected {expected_runs}"
        )

    raw_paths = raw_result_paths(result_dir)
    raw = pd.concat(
        [pd.read_csv(path) for path in raw_paths],
        ignore_index=True,
        sort=False,
    )
    observed_runs = len(
        raw[["environment", "agent", "seed"]].drop_duplicates()
    )
    if observed_runs != expected_runs:
        violations.append(
            f"raw run coverage {observed_runs} != expected {expected_runs}"
        )
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        violations.append(
            f"missing raw columns: {sorted(missing_columns)}"
        )
    if recorded_snapshot:
        for column, expected in (
            ("source_snapshot_sha256", recorded_snapshot),
            ("execution_inputs_clean", True),
            ("git_commit_hash", metadata.get("git_commit_hash")),
            ("package_version", metadata.get("package_version")),
        ):
            if column not in raw:
                violations.append(f"missing strict provenance column: {column}")
                provenance_status = "STRICT_FAIL"
                continue
            observed = set(raw[column].dropna().astype(str).str.lower())
            if observed != {str(expected).lower()}:
                violations.append(
                    f"raw {column} values do not match metadata"
                )
                provenance_status = "STRICT_FAIL"

    critical = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        "return",
        "environment_steps",
    ]
    if raw[critical].isna().any().any():
        violations.append("null values in critical raw columns")

    duplicate_key = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
    ]
    duplicate_count = int(raw.duplicated(duplicate_key).sum())
    if duplicate_count:
        violations.append(f"duplicate raw rows: {duplicate_count}")

    env_by_name = {
        spec.get("name", spec["id"]): spec for spec in config["envs"]
    }
    expected_agents = {agent["name"] for agent in config["agents"]}
    for environment, env_spec in env_by_name.items():
        env_data = raw[raw["environment"] == environment]
        expected_seeds = {
            int(seed) for seed in env_spec.get("seeds", config["seeds"])
        }
        if set(env_data["agent"].unique()) != expected_agents:
            violations.append(f"{environment}: method coverage mismatch")
        budget = int(env_spec["training_steps"])
        checkpoints = expected_checkpoints(config, env_spec)
        for agent in expected_agents:
            method_data = env_data[env_data["agent"] == agent]
            if set(method_data["seed"].unique()) != expected_seeds:
                violations.append(
                    f"{environment}/{agent}: seed coverage mismatch"
                )
            for seed in expected_seeds:
                run = method_data[method_data["seed"] == seed]
                if int(run["environment_steps"].max()) != budget:
                    violations.append(
                        f"{environment}/{agent}/{seed}: step budget mismatch"
                    )
                observed = set(
                    run[run["phase"] == "eval"]["checkpoint"].astype(int)
                )
                if observed != checkpoints:
                    violations.append(
                        f"{environment}/{agent}/{seed}: checkpoint mismatch"
                    )
                evaluation = run[run["phase"] == "eval"]
                mismatched_steps = evaluation[
                    evaluation["environment_steps"]
                    != evaluation["checkpoint"]
                ]
                if not mismatched_steps.empty:
                    violations.append(
                        f"{environment}/{agent}/{seed}: "
                        "evaluation step/checkpoint mismatch"
                    )

    result_files = {}
    for name in (
        "seed_metrics.csv",
        "summary.csv",
        "pairwise.csv",
        "planned_contrasts.csv",
        "heavy_tail_diagnostics.csv",
        "cross_environment.csv",
        "metadata.json",
    ):
        path = result_dir / name
        if not path.exists():
            violations.append(f"missing result file: {name}")
        else:
            result_files[name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    for raw_path in raw_paths:
        logical_path = raw_path.relative_to(result_dir).as_posix()
        result_files[logical_path] = {
            "sha256": sha256(raw_path),
            "bytes": raw_path.stat().st_size,
        }
    raw_manifest = result_dir / "raw_parts" / "manifest.json"
    if raw_manifest.exists():
        result_files["raw_parts/manifest.json"] = {
            "sha256": sha256(raw_manifest),
            "bytes": raw_manifest.stat().st_size,
        }

    return {
        "status": "PASS" if not violations else "FAIL",
        "config": str(config_path),
        "result_dir": str(result_dir),
        "expected_runs": expected_runs,
        "observed_runs": observed_runs,
        "run_shards_present": len(shards),
        "raw_files": len(raw_paths),
        "raw_rows": len(raw),
        "violations": violations,
        "warnings": warnings,
        "provenance_status": provenance_status,
        "files": result_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(Path(args.config), Path(args.result_dir))
    Path(args.output).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(report["status"])
    for violation in report["violations"]:
        print(violation)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
