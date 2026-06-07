from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROHIBITED_BASENAMES = {
    "analysis.py",
    "autograder.py",
    "game.py",
    "pacman.py",
    "qlearningAgents.py",
    "reinforcementTestClasses.py",
    "valueIterationAgents.py",
}

PROHIBITED_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:game|pacman|qlearningAgents|valueIterationAgents)\b",
    re.MULTILINE,
)

BUILD_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".out", ".spl"}
MLWA_RESULT_SETS = (
    "dqn_tuning_development",
    "dqn_strong_validation",
    "confirmatory_extended_compact",
    "support_abstention_replication",
    "minigrid_extended_diagnostic",
)
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "confidence_gated_q.egg-info",
    "development",
    "paper",
    "release",
    "runs",
    "submission_clean_mlwa",
}
EXCLUDED_FILES = {
    "FINAL_SUBMISSION_CHECK.md",
    "artifact_audit.json",
    "submission_audit.json",
    "submission_audit.md",
    "submission_clean_mlwa.zip",
    "submission_clean_mlwa.zip.sha256",
    "scripts/audit_submission.py",
    "scripts/generate_mlwa_assets.py",
    "scripts/package_release.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict:
    violations = []
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            relative.as_posix() in EXCLUDED_FILES
            or path.suffix in BUILD_SUFFIXES
            or any(part in EXCLUDED_PARTS for part in relative.parts)
        ):
            continue
        if path.name in PROHIBITED_BASENAMES:
            violations.append(f"prohibited assignment filename: {relative}")
        if path.suffix == ".py":
            text = path.read_text(encoding="utf-8")
            if PROHIBITED_IMPORT.search(text):
                violations.append(f"prohibited assignment import: {relative}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    required = [
        "LICENSE",
        "README.md",
        "PROVENANCE.md",
        "REPRODUCIBILITY.md",
        "pyproject.toml",
        "requirements.txt",
        "src/hybrid_q/agents.py",
        "scripts/reproduce_all.py",
        "tests/test_agents.py",
        *[f"results/{name}/audit.json" for name in MLWA_RESULT_SETS],
    ]
    for name in required:
        if not (root / name).exists():
            violations.append(f"missing required file: {name}")
    for name in MLWA_RESULT_SETS:
        result_dir = root / "results" / name
        if not (
            (result_dir / "raw.csv").exists()
            or (result_dir / "raw.csv.gz").exists()
        ):
            violations.append(f"{name}: missing raw.csv or raw.csv.gz")

    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifact_audit.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = audit(root)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["status"])
    if report["violations"]:
        for violation in report["violations"]:
            print(violation)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
