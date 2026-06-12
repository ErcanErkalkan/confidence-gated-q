from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hybrid_q.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TITLE = (
    "A Reproducible Support-Boundary Diagnostic for Tabular--Neural "
    "Reinforcement Learning under Exact-State and Approximate Support Shift"
)
NEW_RESULTS = (
    (
        "configs/strong_baselines/double_dqn_30seed.yaml",
        "results/strong_baselines/double_dqn",
    ),
    (
        "configs/strong_baselines/dueling_double_dqn_30seed.yaml",
        "results/strong_baselines/dueling_double_dqn",
    ),
    (
        "configs/approx_support/knn_support_30seed.yaml",
        "results/approx_support/knn_support",
    ),
    (
        "configs/approx_support/feature_distance_support_30seed.yaml",
        "results/approx_support/feature_distance_support",
    ),
    (
        "configs/fuzzy_ablation/fuzzy_ablation_30seed.yaml",
        "results/fuzzy_ablation",
    ),
    (
        "configs/application_risk_variants_30seed.yaml",
        "results/application_risk_variants",
    ),
    (
        "configs/uav_pybullet_30seed.yaml",
        "results/uav_pybullet_validation",
    ),
)
PAPER_FILES = (
    "paper/generated/table_strong_baselines.tex",
    "paper/generated/table_approx_support.tex",
    "paper/generated/table_fuzzy_rule_base.tex",
    "paper/generated/table_fuzzy_ablation.tex",
    "paper/generated/table_application_risk_adjusted.tex",
    "paper/generated/table_uav_pybullet_validation.tex",
    "paper/figures/fig_strong_baselines.pdf",
    "paper/figures/fig_approx_support.pdf",
    "paper/figures/fig_fuzzy_memberships.pdf",
    "paper/figures/fig_fuzzy_ablation.pdf",
    "paper/figures/fig_application_tradeoff.pdf",
    "paper/figures/fig_uav_pybullet_validation.pdf",
    "paper/figures/fig_uav_pybullet_tradeoff.pdf",
)
def raw_path(result_dir: Path) -> Path:
    for name in ("raw.csv", "raw.csv.gz"):
        candidate = result_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing raw results in {result_dir}")


def audit() -> dict:
    violations = []
    new_runs = 0
    execution_commits = set()
    package_versions = set()
    source_snapshots = set()
    for config_relative, result_relative in NEW_RESULTS:
        config_path = ROOT / config_relative
        result_dir = ROOT / result_relative
        config = load_config(config_path)
        report = json.loads(
            (result_dir / "audit.json").read_text(encoding="utf-8")
        )
        if report.get("status") != "PASS":
            violations.append(f"{result_relative}: result audit is not PASS")
        if report.get("provenance_status") != "STRICT_PASS":
            violations.append(
                f"{result_relative}: strict source provenance is not PASS"
            )
        metadata = json.loads(
            (result_dir / "metadata.json").read_text(encoding="utf-8")
        )
        execution_commits.add(metadata.get("git_commit_hash"))
        package_versions.add(metadata.get("package_version"))
        source_snapshots.add(metadata.get("source_snapshot_sha256"))
        new_runs += int(report["observed_runs"])
        raw = pd.read_csv(raw_path(result_dir), nrows=5)
        for metric in ("collision_rate", "unsupported_state_ratio"):
            if metric not in raw:
                violations.append(f"{result_relative}: missing {metric}")
        if (
            config["analysis"]["analysis_status"]
            .lower()
            .startswith("preregistered")
        ):
            violations.append(f"executed result marked protocol-only: {result_relative}")
    if new_runs != 510:
        violations.append(f"focused experiment coverage {new_runs} != 510")
    if len(execution_commits) != 1 or None in execution_commits:
        violations.append(
            "focused experiments do not share one execution commit"
        )
    if package_versions != {"1.5.0"}:
        violations.append(
            "focused experiment package versions are not exactly v1.5.0"
        )
    if len(source_snapshots) != len(NEW_RESULTS) or None in source_snapshots:
        violations.append(
            "focused experiments lack config-specific source snapshots"
        )

    protocol = load_config(
        ROOT / "configs/strong_baselines/a2c_or_ppo_protocol.yaml"
    )
    if "not_executed" not in protocol["analysis"]["analysis_status"]:
        violations.append("A2C/PPO protocol is not explicitly unexecuted")
    if (ROOT / protocol["output_dir"] / "raw.csv").exists():
        violations.append("protocol-only A2C/PPO has a result file")

    for relative in PAPER_FILES:
        path = ROOT / relative
        if not path.exists() or path.stat().st_size == 0:
            violations.append(f"missing paper asset: {relative}")

    manuscript = ROOT / "paper/manuscript.tex"
    text = manuscript.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    if TITLE not in normalized:
        violations.append("manuscript title mismatch")
    required_phrases = (
        "Main confirmatory experiments",
        "Auxiliary smoke checks",
        "Pre-registered extension protocols",
        "Fuzzy arbitration is evaluated as an interpretable soft-computing support proxy",
        "not a universally superior controller",
        "not a real-world validated UAV or robotics deployment",
        "\\section{Limitations}",
        "protocol-only",
        "fig_application_tradeoff.pdf",
        "physics-based Crazyflie",
        "not flight-hardware validation",
    )
    for phrase in required_phrases:
        if phrase.lower() not in normalized.lower():
            violations.append(f"manuscript missing required content: {phrase}")
    for relative in ("README.md", "paper/ARTIFACT_README.md"):
        document = (ROOT / relative).read_text(encoding="utf-8")
        for command in (
            "python -m pip install -e .",
            "python scripts/reproduce_all.py --quick",
            "pytest",
        ):
            if command not in document:
                violations.append(f"{relative}: missing command {command}")

    return {
        "status": "PASS" if not violations else "FAIL",
        "focused_experiment_runs": new_runs,
        "execution_commits": sorted(execution_commits - {None}),
        "package_versions": sorted(package_versions - {None}),
        "source_snapshot_count": len(source_snapshots - {None}),
        "violations": violations,
    }


def main() -> None:
    report = audit()
    output = ROOT / "submission_readiness_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["status"])
    for violation in report["violations"]:
        print(violation)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
