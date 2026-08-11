from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.statistics import holm_adjust  # noqa: E402


ALLOWED_EVIDENCE_CLASSES = {
    "development",
    "confirmatory",
    "replication",
    "exploratory",
    "descriptive",
}
INPUT_COLUMNS = {
    "environment",
    "metric",
    "left",
    "right",
    "n_pairs",
    "paired_t_p",
    "paired_t_holm_p",
    "wilcoxon_p",
    "wilcoxon_holm_p",
    "mean_difference",
    "median_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "cohen_dz",
    "rank_biserial",
    "wins",
    "losses",
    "ties",
}
MANIFEST_COLUMNS = [
    "report_family",
    "evidence_class",
    "environment_or_severity",
    "contrast_name",
    "left",
    "right",
    "metric",
    "n_pairs",
    "raw_paired_t_p",
    "report_holm_paired_t_p",
    "raw_wilcoxon_p",
    "report_holm_wilcoxon_p",
    "mean_difference",
    "median_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "cohen_dz",
    "rank_biserial",
    "wins",
    "losses",
    "ties",
    "source_file",
    "source_row",
    "source_kind",
    "principal_planned",
    "primary_confirmatory",
    "registry_source_id",
]
DISCOVERED_GLOBS = ("**/planned_contrasts.csv", "**/pairwise*.csv")
INFERENTIAL_PATTERN = re.compile(
    r"(?:\bsignific|\bnonsignific|\bnot\s+superior\b|\bholm\b|"
    r"\bp\s*[=<>]|surviv(?:e|es|ed)\s+(?:the\s+)?"
    r"(?:mean-based\s+)?holm|\bnot\s+different\b|\bdo(?:es)?\s+not\s+"
    r"(?:beat|differ)|\bimprov(?:e|es|ed|ing)\b|\bexceed(?:s|ed)?\b|"
    r"\bworse\s+than\b|\bbetter\s+than\b|\btrails?\b|\boutperform|"
    r"\ball\s+\d+\s+(?:paired\s+)?(?:seeds|pairs)\b|\b\d+/\d+\s+paired\s+wins\b)",
    re.IGNORECASE,
)


class MultiplicityAuditError(ValueError):
    pass


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MultiplicityAuditError(f"{path}: registry must be a mapping")
    return data


def validate_evidence_registry(
    registry: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    declared = set(registry.get("allowed_evidence_classes", []))
    if declared != ALLOWED_EVIDENCE_CLASSES:
        raise MultiplicityAuditError(
            "allowed_evidence_classes must be exactly "
            f"{sorted(ALLOWED_EVIDENCE_CLASSES)}"
        )
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise MultiplicityAuditError("registry sources must be a non-empty list")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_family_specs: set[str] = set()
    for entry in sources:
        if not isinstance(entry, dict):
            raise MultiplicityAuditError("each registry source must be a mapping")
        source_id = str(entry.get("source_id", ""))
        source_file = str(entry.get("source_file", ""))
        evidence_class = str(entry.get("evidence_class", ""))
        source_kind = str(entry.get("source_kind", ""))
        if not source_id or source_id in seen_ids:
            raise MultiplicityAuditError(
                f"duplicate or missing source_id: {source_id!r}"
            )
        if not source_file or source_file in seen_paths:
            raise MultiplicityAuditError(
                f"duplicate or missing source_file: {source_file!r}"
            )
        if evidence_class not in ALLOWED_EVIDENCE_CLASSES:
            raise MultiplicityAuditError(
                f"{source_id}: invalid evidence_class {evidence_class!r}"
            )
        if source_kind not in {"planned", "pairwise"}:
            raise MultiplicityAuditError(
                f"{source_id}: source_kind must be planned or pairwise"
            )
        if not (root / source_file).is_file():
            raise MultiplicityAuditError(
                f"{source_id}: missing source file {source_file}"
            )
        included = bool(entry.get("include", False))
        if included:
            family = str(entry.get("report_family", ""))
            template = str(entry.get("report_family_template", ""))
            if bool(family) == bool(template):
                raise MultiplicityAuditError(
                    f"{source_id}: set exactly one of report_family or "
                    "report_family_template"
                )
            family_spec = family or template
            if family_spec in seen_family_specs:
                raise MultiplicityAuditError(
                    f"duplicate included report_family: {family_spec}"
                )
            seen_family_specs.add(family_spec)
            if bool(entry.get("primary_confirmatory", False)) and (
                evidence_class != "confirmatory" or source_kind != "planned"
            ):
                raise MultiplicityAuditError(
                    f"{source_id}: primary_confirmatory requires a planned "
                    "confirmatory source"
                )
        elif not str(entry.get("reason_if_excluded", "")).strip():
            raise MultiplicityAuditError(
                f"{source_id}: excluded sources require reason_if_excluded"
            )
        seen_ids.add(source_id)
        seen_paths.add(source_file)

    discovered: set[str] = set()
    results_root = root / "results"
    for pattern in DISCOVERED_GLOBS:
        discovered.update(
            _relative(path, root) for path in results_root.glob(pattern)
        )
    missing_registry = sorted(discovered - seen_paths)
    stale_registry = sorted(seen_paths - discovered)
    if missing_registry or stale_registry:
        raise MultiplicityAuditError(
            "registry/source inventory mismatch; unregistered="
            f"{missing_registry}; not_discovered={stale_registry}"
        )
    return sources


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _check_report_holm(frame: pd.DataFrame, source_id: str, scope: str) -> None:
    groups: list[pd.DataFrame]
    if scope == "file":
        groups = [frame]
    elif scope == "environment_metric":
        groups = [group for _, group in frame.groupby(["environment", "metric"])]
    else:
        raise MultiplicityAuditError(
            f"{source_id}: unsupported report_holm_scope {scope!r}"
        )
    for group in groups:
        for raw, adjusted in (
            ("paired_t_p", "paired_t_holm_p"),
            ("wilcoxon_p", "wilcoxon_holm_p"),
        ):
            expected = np.asarray(holm_adjust(group[raw].tolist()), dtype=float)
            observed = pd.to_numeric(group[adjusted], errors="coerce").to_numpy()
            if not np.allclose(expected, observed, rtol=1e-8, atol=1e-12):
                raise MultiplicityAuditError(
                    f"{source_id}: {adjusted} is inconsistent with {raw} "
                    f"for report_holm_scope={scope}"
                )


def build_multiplicity_manifest(
    root: Path, registry: dict[str, Any]
) -> pd.DataFrame:
    sources = validate_evidence_registry(registry, root)
    rows: list[dict[str, Any]] = []
    for entry in sources:
        if not entry.get("include", False):
            continue
        source_file = str(entry["source_file"])
        frame = _read_csv(root / source_file)
        source_id = str(entry["source_id"])
        if frame.empty:
            raise MultiplicityAuditError(f"{source_id}: included source is empty")
        missing = sorted(INPUT_COLUMNS - set(frame.columns))
        if missing:
            raise MultiplicityAuditError(
                f"{source_id}: missing required source columns {missing}"
            )
        if entry["source_kind"] == "planned" and "contrast" not in frame:
            raise MultiplicityAuditError(
                f"{source_id}: planned source is missing contrast"
            )
        _check_report_holm(
            frame,
            source_id,
            str(entry.get("report_holm_scope", "file")),
        )
        expected_contrasts = set(entry.get("expected_contrasts", []))
        if expected_contrasts:
            observed_contrasts = set(frame["contrast"].astype(str))
            missing_contrasts = sorted(expected_contrasts - observed_contrasts)
            if missing_contrasts:
                raise MultiplicityAuditError(
                    f"{source_id}: missing expected source rows for contrasts "
                    f"{missing_contrasts}"
                )
        include_contrasts = set(entry.get("include_contrasts", []))
        if include_contrasts:
            frame = frame[frame["contrast"].isin(include_contrasts)]
            missing_selected = sorted(
                include_contrasts - set(frame["contrast"].astype(str))
            )
            if missing_selected:
                raise MultiplicityAuditError(
                    f"{source_id}: selected contrasts not found {missing_selected}"
                )
        expected_rows = entry.get("expected_included_rows")
        if expected_rows is not None and len(frame) != int(expected_rows):
            raise MultiplicityAuditError(
                f"{source_id}: included row count {len(frame)} != {expected_rows}"
            )

        for index, row in frame.iterrows():
            environment = str(row["environment"])
            metric = str(row["metric"])
            if entry.get("report_family"):
                report_family = str(entry["report_family"])
            else:
                report_family = str(entry["report_family_template"]).format(
                    environment=environment, metric=metric
                )
            contrast = (
                str(row["contrast"])
                if entry["source_kind"] == "planned"
                else f"{row['left']}_vs_{row['right']}"
            )
            rows.append(
                {
                    "report_family": report_family,
                    "evidence_class": entry["evidence_class"],
                    "environment_or_severity": environment,
                    "contrast_name": contrast,
                    "left": row["left"],
                    "right": row["right"],
                    "metric": metric,
                    "n_pairs": int(row["n_pairs"]),
                    "raw_paired_t_p": row["paired_t_p"],
                    "report_holm_paired_t_p": row["paired_t_holm_p"],
                    "raw_wilcoxon_p": row["wilcoxon_p"],
                    "report_holm_wilcoxon_p": row["wilcoxon_holm_p"],
                    "mean_difference": row["mean_difference"],
                    "median_difference": row["median_difference"],
                    "bootstrap_ci_low": row["bootstrap_ci_low"],
                    "bootstrap_ci_high": row["bootstrap_ci_high"],
                    "cohen_dz": row["cohen_dz"],
                    "rank_biserial": row["rank_biserial"],
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "ties": int(row["ties"]),
                    "source_file": source_file,
                    "source_row": int(index) + 2,
                    "source_kind": entry["source_kind"],
                    "principal_planned": bool(
                        entry.get("principal_planned", False)
                    ),
                    "primary_confirmatory": bool(
                        entry.get("primary_confirmatory", False)
                    ),
                    "registry_source_id": source_id,
                }
            )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        raise MultiplicityAuditError("no inferential rows were included")
    family_sources = manifest.groupby("report_family")[
        "registry_source_id"
    ].nunique()
    collided_families = family_sources[family_sources > 1].index.tolist()
    if collided_families:
        raise MultiplicityAuditError(
            f"duplicate realized report families: {collided_families[:5]}"
        )
    duplicate_key = [
        "report_family",
        "environment_or_severity",
        "contrast_name",
        "left",
        "right",
        "metric",
        "source_kind",
    ]
    duplicates = manifest.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        duplicate_rows = manifest.loc[duplicates, duplicate_key].to_dict("records")
        raise MultiplicityAuditError(
            f"duplicate inferential rows in included families: {duplicate_rows[:5]}"
        )
    return manifest.sort_values(
        [
            "evidence_class",
            "report_family",
            "environment_or_severity",
            "contrast_name",
            "metric",
            "source_row",
        ]
    ).reset_index(drop=True)


def build_global_holm_sensitivity(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = manifest[manifest["primary_confirmatory"]].copy()
    if selected.empty:
        raise MultiplicityAuditError("no rows are explicitly primary_confirmatory")
    selected["global_holm_paired_t_p"] = holm_adjust(
        selected["raw_paired_t_p"].tolist()
    )
    selected["global_holm_wilcoxon_p"] = holm_adjust(
        selected["raw_wilcoxon_p"].tolist()
    )
    selected["global_sensitivity_scope"] = (
        "all_registry_rows_explicitly_marked_primary_confirmatory"
    )
    return selected


def build_principal_table(
    manifest: pd.DataFrame, global_holm: pd.DataFrame
) -> pd.DataFrame:
    principal = manifest[manifest["principal_planned"]].copy()
    key = ["source_file", "source_row"]
    sensitivity = global_holm[
        key + ["global_holm_paired_t_p", "global_holm_wilcoxon_p"]
    ]
    principal = principal.merge(sensitivity, on=key, how="left")
    return principal[
        [
            "report_family",
            "evidence_class",
            "primary_confirmatory",
            "environment_or_severity",
            "contrast_name",
            "left",
            "right",
            "metric",
            "n_pairs",
            "mean_difference",
            "median_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "cohen_dz",
            "rank_biserial",
            "wins",
            "losses",
            "ties",
            "raw_paired_t_p",
            "report_holm_paired_t_p",
            "global_holm_paired_t_p",
            "raw_wilcoxon_p",
            "report_holm_wilcoxon_p",
            "global_holm_wilcoxon_p",
            "source_file",
            "source_row",
        ]
    ]


def _sentence_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paragraphs: list[tuple[int, int, str]] = []
    buffer: list[str] = []
    start = 1
    lines = text.splitlines()
    for line_no, line in enumerate(lines + [""], start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            if not buffer:
                start = line_no
            buffer.append(stripped)
            continue
        if buffer:
            paragraphs.append((start, line_no - 1, " ".join(buffer)))
            buffer = []
    for start_line, end_line, paragraph in paragraphs:
        offset = 0
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", paragraph):
            normalized = " ".join(sentence.split())
            if normalized:
                records.append(
                    {
                        "sentence_id": f"L{start_line:04d}_{offset:02d}",
                        "line_start": start_line,
                        "line_end": end_line,
                        "sentence_text": normalized,
                    }
                )
                offset += 1
    return records


def discover_inferential_claims(manuscript_path: Path) -> list[dict[str, Any]]:
    text = manuscript_path.read_text(encoding="utf-8")
    records = _sentence_records(text)
    return [
        record
        for record in records
        if INFERENTIAL_PATTERN.search(record["sentence_text"])
    ]


def _anchor_matches(record: dict[str, Any], item: dict[str, Any]) -> bool:
    anchor = " ".join(str(item.get("manuscript_anchor", "")).split())
    # Line hints are documentation only: red revisions legitimately move text.
    # Exact anchors must still resolve to one and only one discovered sentence,
    # preserving the fail-closed ambiguity behavior without stale line coupling.
    return bool(anchor) and anchor in record["sentence_text"]


def audit_claim_map(
    manifest: pd.DataFrame,
    claim_map: dict[str, Any],
    manuscript_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    claims = claim_map.get("claims", [])
    exclusions = claim_map.get("non_inferential_exclusions", [])
    if not isinstance(claims, list) or not isinstance(exclusions, list):
        raise MultiplicityAuditError("claim map lists are malformed")
    discovered = discover_inferential_claims(manuscript_path)
    audit_rows: list[dict[str, Any]] = []
    violations: list[str] = []
    seen_claim_ids: set[str] = set()
    allowed_scopes = {
        "primary_confirmatory",
        "secondary_confirmatory",
        "replication",
        "exploratory",
        "exploratory_secondary",
        "exploratory_unplanned",
        "descriptive",
    }

    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        if not claim_id or claim_id in seen_claim_ids:
            raise MultiplicityAuditError(
                f"duplicate or missing claim_id: {claim_id!r}"
            )
        seen_claim_ids.add(claim_id)
        matching_sentences = [
            record for record in discovered if _anchor_matches(record, claim)
        ]
        selectors = claim.get("planned_contrasts")
        if selectors is None:
            selectors = [claim.get("planned_contrast")]
        if (
            not isinstance(selectors, list)
            or not selectors
            or not all(isinstance(selector, dict) for selector in selectors)
        ):
            raise MultiplicityAuditError(
                f"{claim_id}: planned_contrast(s) must contain mappings"
            )
        for selector_index, selector in enumerate(selectors, start=1):
            atomic_id = (
                claim_id
                if len(selectors) == 1
                else f"{claim_id}.{selector_index:02d}"
            )
            selected = manifest.copy()
            for column, value in selector.items():
                if column not in selected.columns:
                    raise MultiplicityAuditError(
                        f"{atomic_id}: invalid selector column {column}"
                    )
                selected = selected[selected[column].astype(str) == str(value)]
            expected_class = str(claim.get("evidence_class", ""))
            scope = str(claim.get("claim_scope", ""))
            if scope not in allowed_scopes:
                raise MultiplicityAuditError(
                    f"{atomic_id}: unsupported claim_scope {scope!r}"
                )
            reason = ""
            status = "mapped"
            if len(matching_sentences) != 1:
                status = "ambiguous_unmapped"
                reason = (
                    f"manuscript anchor matched {len(matching_sentences)} "
                    "discovered inferential sentences"
                )
            elif len(selected) != 1:
                status = "ambiguous_unmapped"
                reason = (
                    f"inferential-row selector matched {len(selected)} rows"
                )
            elif selected.iloc[0]["evidence_class"] != expected_class:
                status = "ambiguous_unmapped"
                reason = "claim evidence_class does not match registry row"
            elif scope == "primary_confirmatory" and not bool(
                selected.iloc[0]["primary_confirmatory"]
            ):
                status = "ambiguous_unmapped"
                reason = "primary_confirmatory claim maps to a non-primary row"
            if status != "mapped":
                violations.append(f"{atomic_id}: {reason}")
            sentence = matching_sentences[0] if matching_sentences else {}
            source = (
                selected.iloc[0]
                if len(selected) == 1
                else pd.Series(dtype=object)
            )
            audit_rows.append(
                {
                    "claim_id": atomic_id,
                    "manuscript_line": sentence.get(
                        "line_start", claim.get("line_hint")
                    ),
                    "claim_text": sentence.get(
                        "sentence_text", claim.get("manuscript_anchor")
                    ),
                    "claim_scope": scope,
                    "evidence_class": expected_class,
                    "mapping_status": status,
                    "matched_inferential_rows": len(selected),
                    "matched_planned_rows": (
                        len(selected)
                        if len(selected) == 1
                        and source.get("source_kind", "") == "planned"
                        else 0
                    ),
                    "report_family": source.get("report_family", ""),
                    "environment_or_severity": source.get(
                        "environment_or_severity", ""
                    ),
                    "contrast_name": source.get("contrast_name", ""),
                    "metric": source.get("metric", ""),
                    "report_holm_paired_t_p": source.get(
                        "report_holm_paired_t_p", np.nan
                    ),
                    "report_holm_wilcoxon_p": source.get(
                        "report_holm_wilcoxon_p", np.nan
                    ),
                    "source_file": source.get("source_file", ""),
                    "source_row": source.get("source_row", ""),
                    "source_kind": source.get("source_kind", ""),
                    "reason_if_unresolved": reason,
                }
            )

    for exclusion_index, item in enumerate(exclusions, start=1):
        matches = [record for record in discovered if _anchor_matches(record, item)]
        if len(matches) != 1:
            violations.append(
                f"exclusion_{exclusion_index}: anchor matched {len(matches)} "
                "discovered inferential sentences"
            )
        if not str(item.get("reason", "")).strip():
            violations.append(
                f"exclusion_{exclusion_index}: missing non-inferential reason"
            )
        overlaps_mapped_claim = bool(matches) and any(
            _anchor_matches(matches[0], claim) for claim in claims
        )
        if overlaps_mapped_claim:
            violations.append(
                f"exclusion_{exclusion_index}: sentence is both mapped and excluded"
            )

    for record in discovered:
        mapped = any(_anchor_matches(record, claim) for claim in claims)
        excluded = any(_anchor_matches(record, item) for item in exclusions)
        if mapped or excluded:
            continue
        reason = "inferential-looking sentence has no exact claim-map entry"
        violations.append(f"{record['sentence_id']}: {reason}")
        audit_rows.append(
            {
                "claim_id": record["sentence_id"],
                "manuscript_line": record["line_start"],
                "claim_text": record["sentence_text"],
                "claim_scope": "manual_resolution",
                "evidence_class": "",
                "mapping_status": "ambiguous_unmapped",
                "matched_inferential_rows": 0,
                "matched_planned_rows": 0,
                "report_family": "",
                "environment_or_severity": "",
                "contrast_name": "",
                "metric": "",
                "report_holm_paired_t_p": np.nan,
                "report_holm_wilcoxon_p": np.nan,
                "source_file": "",
                "source_row": "",
                "source_kind": "",
                "reason_if_unresolved": reason,
            }
        )
    return pd.DataFrame(audit_rows), violations


def write_principal_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    display["environment_or_severity"] = display[
        "environment_or_severity"
    ].replace(
        {
            "ApplicationNavigation-deployment-goal-shift":
                "ApplicationNavigation-held-out-evaluation-shift"
        }
    )
    display = display.rename(
        columns={
            "report_family": "Report family",
            "evidence_class": "Evidence class",
            "environment_or_severity": "Environment/severity",
            "contrast_name": "Contrast",
            "n_pairs": "n",
            "mean_difference": "Mean diff.",
            "bootstrap_ci_low": "CI low",
            "bootstrap_ci_high": "CI high",
            "report_holm_paired_t_p": "Report t-Holm",
            "global_holm_paired_t_p": "Global sensitivity t-Holm",
            "report_holm_wilcoxon_p": "Report W-Holm",
            "global_holm_wilcoxon_p": "Global sensitivity W-Holm",
        }
    )
    display = display[
        [
            "Report family",
            "Evidence class",
            "Environment/severity",
            "Contrast",
            "n",
            "Mean diff.",
            "CI low",
            "CI high",
            "Report t-Holm",
            "Global sensitivity t-Holm",
            "Report W-Holm",
            "Global sensitivity W-Holm",
        ]
    ]
    text = display.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.6g}",
        na_rep="--",
        longtable=True,
        caption=(
            "All principal planned comparisons. Report-level Holm values are "
            "the scientific family corrections; global values are a "
            "conservative manuscript-wide sensitivity analysis."
        ),
        label="tab:principal-comparisons",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_audit(
    root: Path = ROOT,
    registry_path: Path | None = None,
    claim_map_path: Path | None = None,
    output_dir: Path | None = None,
    table_csv: Path | None = None,
    table_tex: Path | None = None,
    fail_on_claims: bool = True,
) -> dict[str, Any]:
    registry_path = registry_path or (
        root / "project_admin/reviewer1_remaining/EVIDENCE_CLASS_REGISTRY.yaml"
    )
    claim_map_path = claim_map_path or (
        root / "project_admin/reviewer1_remaining/CLAIM_MAP.yaml"
    )
    output_dir = output_dir or (
        root / "results/reviewer1_remaining/multiplicity"
    )
    table_csv = table_csv or (root / "tables/table_principal_comparisons.csv")
    table_tex = table_tex or (root / "tables/table_principal_comparisons.tex")
    registry = _load_yaml(registry_path)
    claim_map = _load_yaml(claim_map_path)
    manifest = build_multiplicity_manifest(root, registry)
    global_holm = build_global_holm_sensitivity(manifest)
    principal = build_principal_table(manifest, global_holm)
    claim_audit, claim_violations = audit_claim_map(
        manifest, claim_map, root / claim_map["manuscript_file"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "multiplicity_family_manifest.csv", index=False)
    global_holm.to_csv(output_dir / "global_holm_sensitivity.csv", index=False)
    claim_audit.to_csv(
        output_dir / "significance_claim_audit.csv", index=False
    )
    principal.to_csv(table_csv, index=False)
    write_principal_latex(principal, table_tex)
    result = {
        "manifest_rows": len(manifest),
        "principal_rows": len(principal),
        "primary_confirmatory_rows": len(global_holm),
        "claim_rows": len(claim_audit),
        "unresolved_claims": len(claim_violations),
        "claim_violations": claim_violations,
    }
    if fail_on_claims and claim_violations:
        raise MultiplicityAuditError(
            "claim audit failed closed: " + "; ".join(claim_violations[:10])
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--claim-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--table-csv", type=Path)
    parser.add_argument("--table-tex", type=Path)
    parser.add_argument(
        "--allow-unresolved-claims",
        action="store_true",
        help="write fail-closed diagnostics without returning a failing status",
    )
    args = parser.parse_args()
    result = run_audit(
        registry_path=args.registry,
        claim_map_path=args.claim_map,
        output_dir=args.output_dir,
        table_csv=args.table_csv,
        table_tex=args.table_tex,
        fail_on_claims=not args.allow_unresolved_claims,
    )
    print(
        "MULTIPLICITY_AUDIT_PASS "
        f"manifest_rows={result['manifest_rows']} "
        f"principal_rows={result['principal_rows']} "
        f"primary_confirmatory_rows={result['primary_confirmatory_rows']} "
        f"claim_rows={result['claim_rows']}"
    )


if __name__ == "__main__":
    main()
