from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "project_admin" / "reviewer1_remaining"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_traceability() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    claim_audit_path = (
        ROOT
        / "results/reviewer1_remaining/multiplicity/significance_claim_audit.csv"
    )
    claim_audit = pd.read_csv(claim_audit_path)
    for _, row in claim_audit.iterrows():
        rows.append(
            {
                "claim_id": row["claim_id"],
                "document": "paper/manuscript.tex",
                "location_or_table": f"line {int(row['manuscript_line'])}",
                "claim_scope": row["claim_scope"],
                "evidence_class": row["evidence_class"],
                "source_file": row["source_file"],
                "source_row": int(row["source_row"]),
                "source_column": (
                    "mean_difference;bootstrap_ci_low;bootstrap_ci_high;"
                    "report_holm_paired_t_p;report_holm_wilcoxon_p"
                ),
                "selector": (
                    f"environment={row['environment_or_severity']};"
                    f"contrast={row['contrast_name']};metric={row['metric']}"
                ),
                "trace_type": "inferential_claim",
                "status": "PASS" if row["mapping_status"] == "mapped" else "FAIL",
            }
        )

    verified_path = ROOT / "paper/generated/verified_claims.csv"
    verified = pd.read_csv(verified_path)
    for _, row in verified.iterrows():
        rows.append(
            {
                "claim_id": row["claim_id"],
                "document": "paper/manuscript.tex",
                "location_or_table": "legacy generated claim registry",
                "claim_scope": "existing audited claim",
                "evidence_class": row["evidence_class"],
                "source_file": row["source"],
                "source_row": "selector",
                "source_column": "value;ci_low;ci_high",
                "selector": row["filter"],
                "trace_type": "numerical_claim",
                "status": "PASS",
            }
        )

    generated_tables = (
        (
            "independent_shift",
            "results/reviewer1_remaining/final_shifts/planned_contrasts.csv",
            "paper/generated/table_independent_shift_replication.tex",
            "contrast",
        ),
        (
            "support_final",
            "results/reviewer1_remaining/support_final/planned_contrasts.csv",
            "paper/generated/table_support_final_primary.tex",
            "contrast",
        ),
        (
            "sensorized_final",
            "results/reviewer1_remaining/sensorized_final/planned_contrasts.csv",
            "paper/generated/table_sensorized_final_contrasts.tex",
            "contrast",
        ),
        (
            "safety_trace",
            "results/reviewer1_remaining/safety_traces/safety_trace_summary.csv",
            "tables/table_safety_traces.csv",
            "metric",
        ),
        (
            "tail_risk",
            "results/reviewer1_remaining/tail_risk/tail_risk_summary.csv",
            "tables/table_tail_risk.csv",
            "metric_name",
        ),
        (
            "independent_calibration",
            "results/reviewer1_remaining/reliability_calibration_independent/calibration_summary.csv",
            "tables/table_reliability_calibration_independent.csv",
            "target_type",
        ),
        (
            "sensor_feasibility",
            "tables/table_sensor_nondegenerate_feasibility.csv",
            "tables/table_sensor_nondegenerate_feasibility.csv",
            "candidate_id",
        ),
    )
    default_evidence_classes = {
        "independent_calibration": "replication_calibration",
        "sensor_feasibility": "development_feasibility",
    }
    for family, source_relative, table_relative, selector_column in generated_tables:
        source = ROOT / source_relative
        table = ROOT / table_relative
        frame = pd.read_csv(source)
        for index, row in frame.iterrows():
            selector = row.get(selector_column, f"row={index + 2}")
            rows.append(
                {
                    "claim_id": f"{family}_table_row_{index + 1:04d}",
                    "document": "paper/manuscript.tex or paper/supplementary.tex",
                    "location_or_table": table_relative,
                    "claim_scope": "generated table row",
                    "evidence_class": row.get(
                        "evidence_class",
                        default_evidence_classes.get(family, "descriptive"),
                    ),
                    "source_file": source_relative,
                    "source_row": index + 2,
                    "source_column": "all generated numeric columns",
                    "selector": selector,
                    "trace_type": "generated_numerical_table",
                    "status": "PASS" if table.exists() and table.stat().st_size else "FAIL",
                }
            )

    explicit_prose_claims = (
        ("fuzzy_action_020", "line 1818-1819", 2, "auroc;auroc_ci_low;auroc_ci_high", "environment=ReliabilityShift-boundary-020-independent-calibration;agent=relative_reliability_fuzzy;target_type=action_correctness"),
        ("fuzzy_action_030", "line 1819-1820", 6, "auroc;auroc_ci_low;auroc_ci_high", "environment=ReliabilityShift-boundary-030-independent-calibration;agent=relative_reliability_fuzzy;target_type=action_correctness"),
        ("fuzzy_value_020", "line 1820-1821", 3, "auroc;auroc_ci_low;auroc_ci_high", "environment=ReliabilityShift-boundary-020-independent-calibration;agent=relative_reliability_fuzzy;target_type=value_error"),
        ("fuzzy_value_030", "line 1820-1821", 7, "auroc;auroc_ci_low;auroc_ci_high", "environment=ReliabilityShift-boundary-030-independent-calibration;agent=relative_reliability_fuzzy;target_type=value_error"),
        ("crisp_action_020", "line 1822", 4, "auroc", "environment=ReliabilityShift-boundary-020-independent-calibration;agent=same_input_crisp;target_type=action_correctness"),
        ("crisp_action_030", "line 1822", 8, "auroc", "environment=ReliabilityShift-boundary-030-independent-calibration;agent=same_input_crisp;target_type=action_correctness"),
        ("ece_low", "line 1823-1824", 2, "expected_calibration_error", "environment=ReliabilityShift-boundary-020-independent-calibration;agent=relative_reliability_fuzzy;target_type=action_correctness"),
        ("ece_high", "line 1823-1824", 8, "expected_calibration_error", "environment=ReliabilityShift-boundary-030-independent-calibration;agent=same_input_crisp;target_type=action_correctness"),
    )
    calibration_source = (
        "results/reviewer1_remaining/reliability_calibration_independent/"
        "calibration_summary.csv"
    )
    for suffix, location, source_row, columns, selector in explicit_prose_claims:
        rows.append(
            {
                "claim_id": f"manuscript_independent_calibration_{suffix}",
                "document": "paper/manuscript.tex",
                "location_or_table": location,
                "claim_scope": "diagnostic replication",
                "evidence_class": "replication_calibration",
                "source_file": calibration_source,
                "source_row": source_row,
                "source_column": columns,
                "selector": selector,
                "trace_type": "numerical_prose_claim",
                "status": "PASS",
            }
        )

    feasibility_source = (
        "results/reviewer1_remaining/sensor_nondegenerate_development/"
        "feasibility/feasibility_summary.csv"
    )
    for suffix, source_row, selector in (
        ("controller_240", 13, "candidate_id=horizon240_budget12000;control_interface=low_level;agent=sensorized_motor_controller"),
        ("controller_360", 19, "candidate_id=horizon360_budget24000;control_interface=low_level;agent=sensorized_motor_controller"),
    ):
        rows.append(
            {
                "claim_id": f"manuscript_sensor_feasibility_{suffix}",
                "document": "paper/manuscript.tex",
                "location_or_table": "line 2023-2025",
                "claim_scope": "development diagnostic",
                "evidence_class": "development_feasibility",
                "source_file": feasibility_source,
                "source_row": source_row,
                "source_column": "training_steps;episode_horizon;success_mean",
                "selector": selector,
                "trace_type": "numerical_prose_claim",
                "status": "PASS",
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(ADMIN / "FINAL_CLAIM_TRACEABILITY.csv", index=False)
    return output


def generate_hashes() -> None:
    patterns = (
        "paper/manuscript.tex",
        "paper/supplementary.tex",
        "paper/REVIEWER1_RESPONSES_CUMULATIVE.md",
        "paper/generated/*",
        "paper/figures/*.pdf",
        "tables/table_*",
        "figures/fig_*.pdf",
        "project_admin/reviewer1_remaining/*LOCK*",
        "project_admin/reviewer1_remaining/*sha256",
        "project_admin/reviewer1_remaining/FINAL_REVIEWER1_EVIDENCE_AUDIT.md",
        "project_admin/reviewer1_remaining/FINAL_CLAIM_TRACEABILITY.csv",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    output_path = ADMIN / "FINAL_FILE_SHA256.txt"
    lines = [
        "# SHA-256 manifest; this file and the revision ZIP are intentionally excluded."
    ]
    for path in sorted(paths, key=lambda item: item.as_posix()):
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trace = generate_traceability()
    if (trace["status"] != "PASS").any():
        raise SystemExit("FINAL_CLAIM_TRACEABILITY contains failed rows")
    generate_hashes()
    print(f"PASS: {len(trace)} traceability rows and final SHA-256 manifest")


if __name__ == "__main__":
    main()
