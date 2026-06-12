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
BUILD_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".spl",
    ".synctex.gz",
}
RESULT_DIRS = (
    "results/dqn_tuning_development",
    "results/dqn_strong_validation",
    "results/confirmatory_extended_compact",
    "results/support_abstention_replication",
    "results/minigrid_extended_diagnostic",
    "results/application_navigation_case_study",
    "results/adaptive_gate_compact_validation",
    "results/cost_support_metrics",
    "results/strong_baselines/double_dqn",
    "results/strong_baselines/dueling_double_dqn",
    "results/approx_support/knn_support",
    "results/approx_support/feature_distance_support",
    "results/fuzzy_ablation",
    "results/application_risk_variants",
    "results/uav_pybullet_validation",
)
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".quick_repro",
    ".uav_smoke",
    ".venv",
    "__pycache__",
    "build",
    "confidence_gated_q.egg-info",
    "development",
    "dist",
    "paper",
    "release",
    "runs",
    "submission_clean_asoc",
}
EXCLUDED_FILES = {
    "artifact_audit.json",
    "submission_readiness_audit.json",
    "MANIFEST.sha256",
}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(root: Path, violations: list[str]) -> None:
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        violations.append("missing required file: MANIFEST.sha256")
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            violations.append(f"invalid manifest line: {line}")
            continue
        path = root / relative
        if not path.is_file():
            violations.append(f"manifest references missing file: {relative}")
        elif sha256(path) != expected:
            violations.append(f"manifest checksum mismatch: {relative}")


def audit(root: Path) -> dict:
    violations: list[str] = []
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
        if path.suffix.lower() == ".zip":
            violations.append(f"nested archive in artifact tree: {relative}")
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
        "MANIFEST.sha256",
        "pyproject.toml",
        "requirements.txt",
        "src/hybrid_q/agents.py",
        "scripts/reproduce_all.py",
        "tests/test_agents.py",
        "tables/table_strong_baselines.csv",
        "tables/table_approx_support.csv",
        "tables/table_application_risk_adjusted.csv",
        "tables/table_uav_pybullet_validation.csv",
    ]
    for name in required:
        if not (root / name).exists():
            violations.append(f"missing required file: {name}")
    for relative in RESULT_DIRS:
        result_dir = root / relative
        audit_path = result_dir / "audit.json"
        if not audit_path.exists():
            violations.append(f"{relative}: missing audit.json")
            continue
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            violations.append(f"{relative}: result audit is not PASS")
        if not (
            (result_dir / "raw.csv").exists()
            or (result_dir / "raw.csv.gz").exists()
        ):
            violations.append(f"{relative}: missing raw.csv or raw.csv.gz")

    audit_manifest(root, violations)
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
    for violation in report["violations"]:
        print(violation)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
