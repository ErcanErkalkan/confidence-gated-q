from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "dqn_tuning_development.json",
    "dqn_strong_validation.json",
    "confirmatory_extended_compact.json",
    "support_abstention_replication.json",
    "minigrid_extended_diagnostic.json",
]


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def result_dir(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return ROOT / config["output_dir"]


def raw_result(output: Path) -> Path:
    for name in ("raw.csv", "raw.csv.gz"):
        candidate = output / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing raw.csv or raw.csv.gz in {output}. "
        "Run with --full to create experiment data."
    )


def run_experiments() -> None:
    for name in CONFIGS:
        run(
            "scripts/run_benchmark.py",
            "--config",
            f"configs/{name}",
        )


def aggregate_and_audit() -> None:
    for name in CONFIGS:
        config_path = ROOT / "configs" / name
        output = result_dir(config_path)
        raw = raw_result(output)
        run(
            "scripts/aggregate_results.py",
            "--input",
            str(raw.relative_to(ROOT)),
            "--output",
            str(output.relative_to(ROOT)),
        )
        run(
            "scripts/audit_results.py",
            "--config",
            str(config_path.relative_to(ROOT)),
            "--result-dir",
            str(output.relative_to(ROOT)),
            "--output",
            str((output / "audit.json").relative_to(ROOT)),
        )


def quick_check() -> None:
    run("-m", "pytest", "-q")
    aggregate_and_audit()
    run(
        "scripts/audit_artifact.py",
        "--root",
        ".",
        "--output",
        "artifact_audit.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and audit the confidence-gated Q artifact."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Reaggregate existing data, regenerate assets, and run audits.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run all experiments before the quick checks.",
    )
    args = parser.parse_args()
    if args.full:
        run_experiments()
    quick_check()


if __name__ == "__main__":
    main()
