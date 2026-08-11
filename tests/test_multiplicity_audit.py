from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_multiplicity import (
    ALLOWED_EVIDENCE_CLASSES,
    MultiplicityAuditError,
    audit_claim_map,
    build_global_holm_sensitivity,
    build_multiplicity_manifest,
    validate_evidence_registry,
)


def _source_row(contrast: str = "a_vs_b") -> dict:
    return {
        "environment": "env",
        "metric": "return_auc",
        "left": "a",
        "right": "b",
        "n_pairs": 10,
        "mean_difference": 1.0,
        "median_difference": 1.0,
        "bootstrap_ci_low": 0.5,
        "bootstrap_ci_high": 1.5,
        "cohen_dz": 1.0,
        "rank_biserial": 1.0,
        "paired_t_p": 0.01,
        "paired_t_holm_p": 0.01,
        "wilcoxon_p": 0.02,
        "wilcoxon_holm_p": 0.02,
        "wins": 10,
        "losses": 0,
        "ties": 0,
        "contrast": contrast,
    }


def _registry(sources: list[dict]) -> dict:
    return {
        "allowed_evidence_classes": sorted(ALLOWED_EVIDENCE_CLASSES),
        "sources": sources,
    }


def _entry(path: str, source_id: str = "source") -> dict:
    return {
        "source_id": source_id,
        "source_file": path,
        "source_kind": "planned",
        "evidence_class": "confirmatory",
        "include": True,
        "report_family": f"{source_id}.planned",
        "report_holm_scope": "file",
        "principal_planned": True,
        "primary_confirmatory": True,
        "expected_contrasts": ["a_vs_b"],
        "expected_included_rows": 1,
    }


def _write_source(root: Path, relative: str, contrast: str = "a_vs_b") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_source_row(contrast)]).to_csv(path, index=False)


def test_global_holm_is_additional_and_preserves_report_values():
    frame = pd.DataFrame(
        {
            "primary_confirmatory": [True, True, True, False],
            "raw_paired_t_p": [0.01, 0.04, 0.2, 0.001],
            "raw_wilcoxon_p": [0.02, 0.03, 0.5, 0.001],
            "report_holm_paired_t_p": [0.02, 0.04, 0.2, 0.001],
        }
    )
    result = build_global_holm_sensitivity(frame)
    assert result["global_holm_paired_t_p"].tolist() == pytest.approx(
        [0.03, 0.08, 0.2]
    )
    assert result["report_holm_paired_t_p"].tolist() == [0.02, 0.04, 0.2]


def test_registry_rejects_invalid_classes_duplicate_families_and_missing_files(
    tmp_path,
):
    first = "results/a/planned_contrasts.csv"
    second = "results/b/planned_contrasts.csv"
    _write_source(tmp_path, first)
    _write_source(tmp_path, second)

    invalid = _entry(first)
    invalid["evidence_class"] = "independent_validation"
    with pytest.raises(MultiplicityAuditError, match="invalid evidence_class"):
        validate_evidence_registry(_registry([invalid]), tmp_path)

    one = _entry(first, "one")
    two = _entry(second, "two")
    two["report_family"] = one["report_family"]
    with pytest.raises(MultiplicityAuditError, match="duplicate included"):
        validate_evidence_registry(_registry([one, two]), tmp_path)

    missing = _entry("results/missing/planned_contrasts.csv")
    with pytest.raises(MultiplicityAuditError, match="missing source file"):
        validate_evidence_registry(_registry([missing]), tmp_path)


def test_registry_fails_when_expected_planned_source_row_is_missing(tmp_path):
    relative = "results/a/planned_contrasts.csv"
    _write_source(tmp_path, relative, contrast="different_contrast")
    entry = _entry(relative)
    with pytest.raises(MultiplicityAuditError, match="missing expected source rows"):
        build_multiplicity_manifest(tmp_path, _registry([entry]))


def test_claim_mapping_is_exact_and_unmapped_claims_fail_closed(tmp_path):
    manuscript = tmp_path / "paper" / "manuscript.tex"
    manuscript.parent.mkdir(parents=True)
    manuscript.write_text(
        "Method A significantly improves over Method B after Holm correction.",
        encoding="utf-8",
    )
    manifest = pd.DataFrame(
        [
            {
                "registry_source_id": "source",
                "report_family": "source.planned",
                "environment_or_severity": "env",
                "contrast_name": "a_vs_b",
                "metric": "return_auc",
                "evidence_class": "confirmatory",
                "primary_confirmatory": True,
                "report_holm_paired_t_p": 0.02,
                "report_holm_wilcoxon_p": 0.03,
                "source_file": "results/a/planned_contrasts.csv",
                "source_row": 2,
                "source_kind": "planned",
            }
        ]
    )
    claim = {
        "claim_id": "claim",
        "manuscript_anchor": "significantly improves over Method B",
        "line_hint": 1,
        "claim_scope": "primary_confirmatory",
        "evidence_class": "confirmatory",
        "planned_contrast": {
            "registry_source_id": "source",
            "environment_or_severity": "env",
            "contrast_name": "a_vs_b",
        },
    }
    audit, violations = audit_claim_map(
        manifest,
        {"claims": [claim], "non_inferential_exclusions": []},
        manuscript,
    )
    assert not violations
    assert audit.loc[0, "mapping_status"] == "mapped"

    audit, violations = audit_claim_map(
        manifest,
        {"claims": [], "non_inferential_exclusions": []},
        manuscript,
    )
    assert violations
    assert audit.loc[0, "mapping_status"] == "ambiguous_unmapped"
