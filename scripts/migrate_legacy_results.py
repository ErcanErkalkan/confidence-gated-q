from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_q.envs import resolve_env_id


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESULTS = (
    "confirmatory",
    "external_minigrid",
    "support_abstention_confirmatory",
    "minigrid_supplemental",
    "tau_sensitivity_compact",
    "tau_sensitivity_minigrid",
    "dqn_sensitivity_compact",
    "dqn_sensitivity_minigrid",
    "dqn_validation_confirmatory",
    "dqn_validation_minigrid",
)


def config_for_output(output_dir: str) -> Path:
    for path in (ROOT / "configs").glob("*.json"):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("output_dir") == output_dir:
            return path
    raise FileNotFoundError(f"No config declares output_dir={output_dir}")


def introduction_commit(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--",
            relative,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commits = [line for line in result.stdout.splitlines() if line]
    if not commits:
        raise RuntimeError(f"Could not find introduction commit for {relative}")
    return commits[-1]


def migrate(name: str) -> None:
    result_dir = ROOT / "results" / name
    source = result_dir / "raw.csv"
    destination = result_dir / "raw.csv.gz"
    if not source.exists():
        if destination.exists():
            print(f"{name}: already migrated")
            return
        raise FileNotFoundError(source)

    config_path = config_for_output(f"results/{name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    packages = metadata.get("packages", {})
    commit = introduction_commit(source)
    env_specs = {
        spec.get("name", spec["id"]): spec for spec in config["envs"]
    }

    temporary = destination.with_suffix(".tmp")
    first = True
    for chunk in pd.read_csv(source, chunksize=100_000):
        chunk["experiment_name"] = chunk["experiment"]
        chunk["config_file"] = config_path.relative_to(ROOT).as_posix()
        chunk["environment_id"] = chunk["environment"].map(
            lambda value: env_specs[value]["id"]
        )
        chunk["resolved_environment_id"] = chunk["environment_id"].map(
            resolve_env_id
        )
        chunk["observation_representation"] = chunk["environment"].map(
            lambda value: env_specs[value].get(
                "observation", "native_gymnasium_observation"
            )
        )
        chunk["train_steps"] = chunk["environment_steps"]
        is_evaluation = chunk["phase"] == "eval"
        chunk["eval_checkpoint"] = np.where(
            is_evaluation, chunk["checkpoint"], np.nan
        )
        chunk["eval_return"] = np.where(
            is_evaluation, chunk["return"], np.nan
        )
        chunk["wall_clock_training_time"] = chunk[
            "training_elapsed_seconds"
        ]
        chunk["wall_clock_evaluation_time"] = chunk[
            "evaluation_elapsed_seconds"
        ]
        chunk["git_commit_hash"] = commit
        chunk["package_version"] = "1.2.0"
        chunk["python_version"] = metadata.get("python", "unknown")
        chunk["torch_version"] = metadata.get(
            "torch", packages.get("torch", "unknown")
        )
        chunk["numpy_version"] = metadata.get(
            "numpy", packages.get("numpy", "unknown")
        )
        chunk["gymnasium_version"] = metadata.get(
            "gymnasium", packages.get("gymnasium", "unknown")
        )
        chunk["minigrid_version"] = packages.get(
            "minigrid", "not-installed"
        )
        chunk.to_csv(
            temporary,
            mode="w" if first else "a",
            header=first,
            index=False,
            compression="gzip",
        )
        first = False

    metadata["schema_migration"] = {
        "source_file": "raw.csv",
        "output_file": "raw.csv.gz",
        "numeric_results_changed": False,
        "provenance_commit": commit,
        "package_version": "1.2.0",
        "config_file": config_path.relative_to(ROOT).as_posix(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    source.unlink()
    print(f"{name}: migrated to {destination.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        default=LEGACY_RESULTS,
        help="Result directory names; defaults to all legacy families.",
    )
    args = parser.parse_args()
    for name in args.names:
        migrate(name)


if __name__ == "__main__":
    main()
