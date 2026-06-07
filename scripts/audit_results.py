from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "experiment",
    "environment",
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


def audit(config_path: Path, result_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (result_dir / "metadata.json").read_text(encoding="utf-8")
    )
    violations = []
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if metadata.get("config_sha256") != config_hash:
        violations.append("metadata config hash does not match config")

    temporary_files = list((result_dir / "runs").glob("*.tmp"))
    if temporary_files:
        violations.append(f"incomplete run shards: {len(temporary_files)}")

    expected_runs = (
        len(config["envs"]) * len(config["agents"]) * len(config["seeds"])
    )
    shards = list((result_dir / "runs").glob("*.csv"))
    if len(shards) != expected_runs:
        violations.append(
            f"run shard count {len(shards)} != expected {expected_runs}"
        )

    raw_path = result_dir / "raw.csv"
    raw = pd.read_csv(raw_path)
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        violations.append(
            f"missing raw columns: {sorted(missing_columns)}"
        )

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
    expected_seeds = {int(seed) for seed in config["seeds"]}
    for environment, env_spec in env_by_name.items():
        env_data = raw[raw["environment"] == environment]
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
        "raw.csv",
        "seed_metrics.csv",
        "summary.csv",
        "pairwise.csv",
        "planned_contrasts.csv",
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

    return {
        "status": "PASS" if not violations else "FAIL",
        "config": str(config_path),
        "result_dir": str(result_dir),
        "expected_runs": expected_runs,
        "observed_runs": len(shards),
        "raw_rows": len(raw),
        "violations": violations,
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
