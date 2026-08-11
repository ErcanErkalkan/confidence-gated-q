from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


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
    ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out",
    ".spl", ".synctex.gz",
}
EXCLUDED_PARTS = {
    ".ruff_cache", ".git", ".pytest_cache", ".quick_repro", ".uav_smoke",
    ".venv", "__pycache__", "build", "confidence_gated_q.egg-info", "dist",
}
EXCLUDED_FILES = {
    "artifact_audit.json",
    "submission_readiness_audit.json",
    "MANIFEST.sha256",
}
PRIVATE_TREE_NAMES = {
    "01_PRIVATE_PAPER",
    "03_PRIVATE_HISTORICAL_ARCHIVE",
    "paper",
    "project_admin",
    "release",
    "submission_clean_asoc",
}
PRIVATE_ROOT_PATTERNS = (
    re.compile(r"^manuscript", re.I),
    re.compile(r"^supplementary", re.I),
    re.compile(r"^reviewer", re.I),
    re.compile(r"^response_to_reviewers", re.I),
    re.compile(r"^cover_letter", re.I),
    re.compile(r"^title_page", re.I),
    re.compile(r"^highlights", re.I),
    re.compile(r"^release_(?:checklist|notes)", re.I),
)
FORBIDDEN_REGISTRY_ANCHORS = (
    "paper/", "manuscript", "project_admin/", "reviewer1_remaining",
    "reviewer1_comment", "tab:", "fig:",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, violations: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive audit path
        violations.append(f"invalid JSON {path}: {exc}")
        return None


def load_yaml(path: Path, violations: list[str]) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("top-level YAML value is not a mapping")
        return payload
    except Exception as exc:  # pragma: no cover - defensive audit path
        violations.append(f"invalid YAML {path}: {exc}")
        return None


def parse_sha_manifest(manifest: Path, base_dir: Path, violations: list[str]) -> int:
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            violations.append(f"invalid SHA-256 manifest line in {manifest}: {line}")
            continue
        path = base_dir / relative
        if not path.is_file():
            violations.append(f"SHA-256 manifest references missing file: {path}")
            continue
        checked += 1
        if sha256(path) != expected:
            violations.append(f"SHA-256 mismatch: {path}")
    return checked


def _is_auditable_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return not (
        relative.as_posix() in EXCLUDED_FILES
        or path.suffix in BUILD_SUFFIXES
        or any(part in EXCLUDED_PARTS for part in relative.parts)
    )


def _manifest_listed_paths(manifest: Path, violations: list[str]) -> set[str]:
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            _expected, relative = line.split("  ", 1)
        except ValueError:
            continue  # parse_sha_manifest reports the malformed line.
        relative = relative.strip().lstrip("*")
        if relative in listed:
            violations.append(f"duplicate SHA-256 manifest entry: {relative}")
        listed.add(relative)
    return listed


def audit_repository_manifest(root: Path, violations: list[str], require_manifest: bool) -> dict[str, Any]:
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        if require_manifest:
            violations.append("missing required file: MANIFEST.sha256")
            return {"status": "MISSING", "required": True, "entries_checked": 0}
        return {"status": "NOT_REQUIRED_PREPUBLICATION", "required": False, "entries_checked": 0}
    before = len(violations)
    checked = parse_sha_manifest(manifest, root, violations)
    listed = _manifest_listed_paths(manifest, violations)
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest and _is_auditable_file(root, path)
    }
    for relative in sorted(expected - listed):
        violations.append(f"manifest omits public file: {relative}")
    for relative in sorted(listed - expected):
        violations.append(f"manifest contains non-public or unexpected entry: {relative}")
    return {
        "status": "PASS" if len(violations) == before else "FAIL",
        "required": require_manifest,
        "entries_checked": checked,
        "expected_entries": len(expected),
        "listed_entries": len(listed),
        "missing_entries": len(expected - listed),
        "unexpected_entries": len(listed - expected),
    }


def audit_public_boundary(root: Path, violations: list[str]) -> dict[str, Any]:
    findings: list[str] = []
    for name in PRIVATE_TREE_NAMES:
        if (root / name).exists():
            findings.append(name)
            violations.append(f"private/admin tree present in public artifact: {name}")
    for child in root.iterdir():
        if any(pattern.search(child.name) for pattern in PRIVATE_ROOT_PATTERNS):
            findings.append(child.name)
            violations.append(f"private/submission file present at public root: {child.name}")
    public_scripts = root / "scripts"
    if public_scripts.is_dir():
        for path in public_scripts.glob("generate_paper_*.py"):
            findings.append(path.relative_to(root).as_posix())
            violations.append(f"manuscript-generation script present in public scripts: {path.relative_to(root)}")
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def split_registry_sources(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def audit_evidence_registry(root: Path, violations: list[str]) -> dict[str, Any]:
    path = root / "configs" / "evidence_registry.json"
    payload = load_json(path, violations)
    if payload is None:
        return {"status": "FAIL", "family_count": 0}
    families = payload.get("families", [])
    scope = payload.get("revision_scope", {})
    ids = [row.get("id") for row in families if isinstance(row, dict)]
    expected_ids = [f"E{index:02d}" for index in range(1, 29)]
    if ids != expected_ids:
        violations.append(f"evidence registry family IDs are not exactly E01-E28: {ids}")
    if scope.get("family_count") != 28:
        violations.append("evidence registry revision_scope.family_count must equal 28")
    if scope.get("first_family_id") != "E01" or scope.get("last_family_id") != "E28":
        violations.append("evidence registry revision scope must span E01-E28")
    if scope.get("artifact_version") != "1.0.0":
        violations.append("evidence registry artifact_version must equal 1.0.0")

    source_paths: list[str] = []
    missing_sources: list[str] = []
    for row in families:
        if not isinstance(row, dict):
            violations.append("evidence registry contains a non-mapping family entry")
            continue
        for required in ("id", "family", "class", "scope", "source"):
            if required not in row:
                violations.append(f"evidence family {row.get('id', '?')} missing field: {required}")
        source = str(row.get("source", ""))
        if any(anchor.lower() in source.lower() for anchor in FORBIDDEN_REGISTRY_ANCHORS):
            violations.append(f"evidence family {row.get('id')} contains private/stale source anchor: {source}")
        for relative in split_registry_sources(source):
            source_paths.append(relative)
            if not (root / relative).exists():
                missing_sources.append(relative)
                violations.append(f"evidence family {row.get('id')} source missing: {relative}")

    return {
        "status": "PASS" if ids == expected_ids and not missing_sources else "FAIL",
        "family_count": len(families),
        "family_range": f"{ids[0]}-{ids[-1]}" if ids else None,
        "source_paths_checked": len(source_paths),
        "missing_sources": missing_sources,
        "artifact_version": scope.get("artifact_version"),
    }


def audit_claim_registry(root: Path, valid_family_ids: set[str], violations: list[str]) -> dict[str, Any]:
    path = root / "configs" / "claim_evidence_index.yaml"
    payload = load_yaml(path, violations)
    if payload is None:
        return {"status": "FAIL", "claim_count": 0}
    if payload.get("schema_version") != 3:
        violations.append("claim-evidence registry schema_version must equal 3")
    if payload.get("artifact_version") != "1.0.0":
        violations.append("claim-evidence registry artifact_version must equal 1.0.0")
    if payload.get("evidence_registry") != "configs/evidence_registry.json":
        violations.append("claim-evidence registry must point to configs/evidence_registry.json")

    registered_scope = payload.get("registered_scope", {})
    if registered_scope.get("family_range") != "E01-E28":
        violations.append("claim-evidence registered scope must be E01-E28")

    auxiliary = registered_scope.get("auxiliary_public_diagnostics", {})
    auxiliary_names = set(auxiliary)
    artifact_paths: list[str] = []
    for name, spec in auxiliary.items():
        for relative in spec.get("artifacts", []):
            artifact_paths.append(relative)
            if not (root / relative).exists():
                violations.append(f"auxiliary program {name} artifact missing: {relative}")

    claims = payload.get("claims", [])
    claim_ids: list[str] = []
    pass_count = 0
    invalid_family_refs: list[str] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        claim_ids.append(claim_id)
        if claim.get("status") == "PASS":
            pass_count += 1
        else:
            violations.append(f"claim {claim_id} status is not PASS")
        refs = claim.get("evidence_families", []) or []
        for family_id in refs:
            if family_id not in valid_family_ids:
                invalid_family_refs.append(str(family_id))
                violations.append(f"claim {claim_id} references unknown evidence family: {family_id}")
        for aux_name in claim.get("auxiliary_programs", []) or []:
            if aux_name not in auxiliary_names:
                violations.append(f"claim {claim_id} references unknown auxiliary program: {aux_name}")
        for field in ("primary_artifacts", "supporting_artifacts"):
            for relative in claim.get(field, []) or []:
                artifact_paths.append(relative)
                lowered = str(relative).lower()
                if any(anchor.lower() in lowered for anchor in FORBIDDEN_REGISTRY_ANCHORS):
                    violations.append(f"claim {claim_id} contains private/LaTeX anchor in {field}: {relative}")
                if not (root / relative).exists():
                    violations.append(f"claim {claim_id} {field} path missing: {relative}")

    if len(claim_ids) != len(set(claim_ids)):
        violations.append("claim-evidence registry contains duplicate claim IDs")
    if len(claims) != 16:
        violations.append(f"claim-evidence registry must contain 16 claims, observed {len(claims)}")

    return {
        "status": "PASS" if len(claims) == 16 and pass_count == 16 and not invalid_family_refs else "FAIL",
        "claim_count": len(claims),
        "pass_count": pass_count,
        "artifact_paths_checked": len(artifact_paths),
        "invalid_family_references": invalid_family_refs,
        "auxiliary_programs": sorted(auxiliary_names),
    }


def audit_continuous_control(root: Path, violations: list[str]) -> dict[str, Any]:
    protocol_path = root / "configs" / "continuous_control" / "CONTINUOUS_CONTROL_PROTOCOL.yaml"
    protocol_sha = root / "configs" / "continuous_control" / "CONTINUOUS_CONTROL_PROTOCOL_SHA256.txt"
    analysis_dir = root / "results" / "continuous_control" / "analysis"
    audit_path = analysis_dir / "audit.json"
    s1_path = analysis_dir / "S1_supplemental_controller_contrasts.csv"
    s2_path = analysis_dir / "S2_supplemental_support_contrasts.csv"
    required = [protocol_path, protocol_sha, audit_path, s1_path, s2_path]
    for path in required:
        if not path.exists():
            violations.append(f"continuous-control required artifact missing: {path.relative_to(root)}")
    if any(not path.exists() for path in required):
        return {"status": "FAIL"}

    before = len(violations)
    protocol = load_yaml(protocol_path, violations) or {}
    parse_sha_manifest(protocol_sha, protocol_sha.parent, violations)
    report = load_json(audit_path, violations) or {}
    expected_s1 = protocol.get("inference", {}).get("controller_contrast_family_S1", {}).get("n_contrasts")
    expected_s2 = protocol.get("inference", {}).get("support_contrast_family_S2", {}).get("n_contrasts")
    expected_runs = protocol.get("budget", {}).get("planned_trained_agents")

    if report.get("status") != "PASS":
        violations.append("continuous-control analysis audit status is not PASS")
    if report.get("inferential_status") != "SUPPLEMENTAL_NONCONFIRMATORY":
        violations.append("continuous-control inferential_status must be SUPPLEMENTAL_NONCONFIRMATORY")
    if report.get("complete_runs") != expected_runs:
        violations.append(f"continuous-control complete_runs mismatch: {report.get('complete_runs')} != {expected_runs}")
    if report.get("S1_tests") != expected_s1:
        violations.append(f"continuous-control S1_tests mismatch: {report.get('S1_tests')} != {expected_s1}")
    if report.get("S2_tests") != expected_s2:
        violations.append(f"continuous-control S2_tests mismatch: {report.get('S2_tests')} != {expected_s2}")
    if report.get("final_episode_rows") != 3600:
        violations.append("continuous-control final_episode_rows must equal 3600")
    if report.get("checkpoint_episode_rows") != 600:
        violations.append("continuous-control checkpoint_episode_rows must equal 600")

    def csv_family_check(path: Path, expected_family: str, expected_rows: int) -> None:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            violations.append(f"{path.relative_to(root)} row count mismatch: {len(rows)} != {expected_rows}")
        families = {row.get("family") for row in rows}
        if families != {expected_family}:
            violations.append(f"{path.relative_to(root)} family values mismatch: {sorted(families)}")

    csv_family_check(s1_path, "S1", int(expected_s1 or 0))
    csv_family_check(s2_path, "S2", int(expected_s2 or 0))

    direct_req = root / "requirements-continuous-control.txt"
    if not direct_req.exists():
        violations.append("continuous-control direct requirements file is missing")
    else:
        unpinned = []
        for raw_line in direct_req.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" not in line:
                unpinned.append(line)
        if unpinned:
            violations.append(f"continuous-control direct requirements contain unpinned entries: {unpinned}")

    tested_req = root / "requirements-tested-continuous-control.txt"
    tested_sha = root / "requirements-tested-continuous-control.sha256"
    if not tested_req.exists() or not tested_sha.exists():
        violations.append("continuous-control tested environment freeze or SHA-256 is missing")
    else:
        parse_sha_manifest(tested_sha, root, violations)

    return {
        "status": "PASS" if len(violations) == before else "FAIL",
        "complete_runs": report.get("complete_runs"),
        "final_episode_rows": report.get("final_episode_rows"),
        "checkpoint_episode_rows": report.get("checkpoint_episode_rows"),
        "S1_tests": report.get("S1_tests"),
        "S2_tests": report.get("S2_tests"),
        "inferential_status": report.get("inferential_status"),
    }


def audit_result_audits(root: Path, violations: list[str]) -> dict[str, Any]:
    audit_paths = sorted((root / "results").rglob("audit.json")) if (root / "results").exists() else []
    checked = 0
    failed: list[str] = []
    for path in audit_paths:
        report = load_json(path, violations)
        if report is None:
            failed.append(path.relative_to(root).as_posix())
            continue
        if "status" not in report:
            continue
        checked += 1
        if report.get("status") != "PASS":
            relative = path.relative_to(root).as_posix()
            failed.append(relative)
            violations.append(f"result audit is not PASS: {relative}")
    return {"status": "PASS" if not failed else "FAIL", "audits_checked": checked, "failed": failed}


def audit_files(root: Path, violations: list[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not _is_auditable_file(root, path):
            continue
        if path.suffix.lower() in {".zip", ".7z", ".rar"}:
            violations.append(f"nested archive in artifact tree: {relative}")
        if path.name in PROHIBITED_BASENAMES:
            violations.append(f"prohibited assignment filename: {relative}")
        if path.suffix == ".py":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            if PROHIBITED_IMPORT.search(text):
                violations.append(f"prohibited assignment import: {relative}")
        files.append({"path": relative.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    return files


def audit(root: Path, *, require_manifest: bool = False) -> dict[str, Any]:
    violations: list[str] = []
    required = [
        "LICENSE", "README.md", "PROVENANCE.md", "REPRODUCIBILITY.md",
        "CITATION.cff", ".zenodo.json", ".gitignore", "pyproject.toml",
        "requirements.txt", "requirements-tested.txt", "requirements-uav.txt",
        "requirements-tested-uav.txt", "requirements-continuous-control.txt",
        "requirements-tested-continuous-control.txt",
        "requirements-tested-continuous-control.sha256",
        "configs/evidence_registry.json", "configs/claim_evidence_index.yaml",
        "configs/continuous_control/CONTINUOUS_CONTROL_PROTOCOL.yaml",
        "configs/continuous_control/CONTINUOUS_CONTROL_PROTOCOL_SHA256.txt",
        "src/hybrid_q/agents.py", "scripts/reproduce_all.py", "scripts/audit_artifact.py",
        "scripts/preflight_release.py", "scripts/generate_tables.py", "scripts/generate_figures.py",
        "tests/test_agents.py", "tests/test_artifact_audit.py", "tests/test_protocol_integrity.py",
        "tables/table_principal_comparisons.csv",
        "results/continuous_control/analysis/audit.json",
        "results/continuous_control/analysis/S1_supplemental_controller_contrasts.csv",
        "results/continuous_control/analysis/S2_supplemental_support_contrasts.csv",
    ]
    for name in required:
        if not (root / name).exists():
            violations.append(f"missing required file: {name}")

    boundary = audit_public_boundary(root, violations)
    evidence = audit_evidence_registry(root, violations)
    family_ids = {f"E{i:02d}" for i in range(1, 29)} if evidence.get("family_count") == 28 else set()
    claims = audit_claim_registry(root, family_ids, violations)
    continuous = audit_continuous_control(root, violations)
    result_audits = audit_result_audits(root, violations)
    manifest = audit_repository_manifest(root, violations, require_manifest=require_manifest)
    files = audit_files(root, violations)

    return {
        "status": "PASS" if not violations else "FAIL",
        "mode": "frozen_release" if require_manifest else "prepublication",
        "violations": violations,
        "summary": {
            "evidence": evidence,
            "claims": claims,
            "continuous_control": continuous,
            "result_audits": result_audits,
            "public_boundary": boundary,
            "repository_manifest": manifest,
            "files_hashed": len(files),
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the public confidence-gated-q artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifact_audit.json")
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Require and verify MANIFEST.sha256 for a frozen release snapshot.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = audit(root, require_manifest=args.require_manifest)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(report["status"])
    print(
        "EVIDENCE_COVERAGE "
        f"families={summary['evidence'].get('family_count')} "
        f"range={summary['evidence'].get('family_range')} "
        f"claims={summary['claims'].get('pass_count')}/{summary['claims'].get('claim_count')}"
    )
    cc = summary["continuous_control"]
    print(
        "CONTINUOUS_CONTROL "
        f"status={cc.get('status')} runs={cc.get('complete_runs')} "
        f"S1={cc.get('S1_tests')} S2={cc.get('S2_tests')}"
    )
    print(f"MANIFEST {summary['repository_manifest'].get('status')}")
    for violation in report["violations"]:
        print(violation)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
