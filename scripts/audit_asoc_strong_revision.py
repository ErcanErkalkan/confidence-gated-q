from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TITLE = (
    "A Reproducible Support-Boundary Diagnostic for Tabular--Neural "
    "Reinforcement Learning under Exact-State Support Shift"
)
FAMILIES = (
    "application_navigation_case_study",
    "adaptive_gate_compact_validation",
    "cost_support_metrics",
)
REQUIRED_METRICS = {
    "failure_rate",
    "collision_rate",
    "risk_zone_rate",
    "unsupported_state_ratio",
    "memory_branch_usage_ratio",
    "neural_branch_usage_ratio",
    "abstention_ratio",
    "adaptive_alpha_mean",
    "adaptive_alpha_iqr",
    "support_score_mean",
    "uncertainty_score_mean",
    "inference_time_us_per_decision_mean",
    "inference_time_us_per_decision_median",
    "memory_cost_states",
    "memory_cost_entries",
    "memory_cost_bytes_estimated",
}
PAPER_FILES = (
    "paper/generated/asoc_application_case_study.tex",
    "paper/generated/asoc_adaptive_gate_summary.tex",
    "paper/generated/asoc_cost_and_support_metrics.tex",
    "paper/generated/asoc_strong_revision_claims.csv",
    "paper/figures/Figure_05_Application_Case_Study.pdf",
    "paper/figures/Figure_06_Fuzzy_Gate_Sensitivity.pdf",
    "paper/figures/Figure_07_Cost_Support_and_Unsupported_Ratio.pdf",
    "paper/graphical_abstract/Graphical_Abstract_Neuro_Memory_Support_Shift.png",
)


def raw_path(result_dir: Path) -> Path:
    for name in ("raw.csv", "raw.csv.gz"):
        candidate = result_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing raw results in {result_dir}")


def audit() -> dict:
    violations = []
    run_count = 0
    for family in FAMILIES:
        result_dir = ROOT / "results" / family
        report = json.loads(
            (result_dir / "audit.json").read_text(encoding="utf-8")
        )
        if report["status"] != "PASS":
            violations.append(f"{family}: result audit is not PASS")
        run_count += int(report["observed_runs"])
        raw = pd.read_csv(raw_path(result_dir))
        missing = REQUIRED_METRICS - set(raw.columns)
        if missing:
            violations.append(f"{family}: missing metrics {sorted(missing)}")
            continue
        evaluation = raw[raw["phase"] == "eval"]
        for column in (
            "failure_rate",
            "collision_rate",
            "risk_zone_rate",
            "unsupported_state_ratio",
            "memory_branch_usage_ratio",
            "neural_branch_usage_ratio",
            "abstention_ratio",
        ):
            values = evaluation[column].dropna()
            if not values.between(0, 1).all():
                violations.append(f"{family}: {column} outside [0, 1]")
        for column in (
            "inference_time_us_per_decision_mean",
            "inference_time_us_per_decision_median",
            "memory_cost_states",
            "memory_cost_entries",
            "memory_cost_bytes_estimated",
        ):
            values = evaluation[column].dropna()
            if values.empty or (values < 0).any():
                violations.append(
                    f"{family}: invalid nonnegative metric {column}"
                )
    if run_count != 730:
        violations.append(f"new full-run coverage {run_count} != 730")

    if (ROOT / "paper").exists():
        for relative in PAPER_FILES:
            path = ROOT / relative
            if not path.exists() or path.stat().st_size == 0:
                violations.append(f"missing paper asset: {relative}")

    manuscript = ROOT / "paper" / "manuscript.tex"
    if manuscript.exists():
        text = manuscript.read_text(encoding="utf-8")
        if TITLE not in " ".join(text.split()):
            violations.append("manuscript title mismatch")
        for required in (
            "ApplicationNavigationSupportShift",
            "fuzzy",
            "unsupported-state ratio",
            "Figure_05_Application_Case_Study.pdf",
            "Figure_06_Fuzzy_Gate_Sensitivity.pdf",
            "Figure_07_Cost_Support_and_Unsupported_Ratio.pdf",
            "Declaration of generative AI",
            "dependency-free smoke",
        ):
            if required.lower() not in text.lower():
                violations.append(
                    f"manuscript missing required content: {required}"
                )

    return {
        "status": "PASS" if not violations else "FAIL",
        "new_full_runs": run_count,
        "families": list(FAMILIES),
        "violations": violations,
    }


def main() -> None:
    report = audit()
    output = ROOT / "asoc_strong_revision_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["status"])
    for violation in report["violations"]:
        print(violation)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
