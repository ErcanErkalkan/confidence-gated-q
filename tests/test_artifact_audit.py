from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = ROOT / "scripts" / "audit_artifact.py"
    spec = importlib.util.spec_from_file_location("artifact_audit_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch_path(root: Path, relative: str) -> None:
    path = root / relative
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_sha_manifest(path: Path, targets: list[Path], base: Path) -> None:
    rows = []
    for target in targets:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append(f"{digest}  {target.relative_to(base).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_minimal_public_tree(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()

    # Canonical registries are copied from the real checkout so this fixture tracks
    # the release schema rather than duplicating 28 evidence families by hand.
    evidence = json.loads((ROOT / "configs/evidence_registry.json").read_text(encoding="utf-8"))
    claims = yaml.safe_load((ROOT / "configs/claim_evidence_index.yaml").read_text(encoding="utf-8"))
    (root / "configs").mkdir()
    (root / "configs/evidence_registry.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (root / "configs/claim_evidence_index.yaml").write_text(yaml.safe_dump(claims, sort_keys=False), encoding="utf-8")

    for family in evidence["families"]:
        for relative in family["source"].split(";"):
            _touch_path(root, relative.strip())

    for spec in claims["registered_scope"]["auxiliary_public_diagnostics"].values():
        for relative in spec.get("artifacts", []):
            _touch_path(root, relative)
    for claim in claims["claims"]:
        for field in ("primary_artifacts", "supporting_artifacts"):
            for relative in claim.get(field, []) or []:
                _touch_path(root, relative)

    # Continuous-control frozen protocol and exact sidecar.
    cc_src = ROOT / "configs/continuous_control"
    cc_dst = root / "configs/continuous_control"
    cc_dst.mkdir(parents=True, exist_ok=True)
    for name in ("CONTINUOUS_CONTROL_PROTOCOL.yaml", "CONTINUOUS_CONTROL_PROTOCOL.md"):
        (cc_dst / name).write_bytes((cc_src / name).read_bytes())
    _write_sha_manifest(
        cc_dst / "CONTINUOUS_CONTROL_PROTOCOL_SHA256.txt",
        [cc_dst / "CONTINUOUS_CONTROL_PROTOCOL.yaml", cc_dst / "CONTINUOUS_CONTROL_PROTOCOL.md"],
        cc_dst,
    )
    protocol = yaml.safe_load((cc_dst / "CONTINUOUS_CONTROL_PROTOCOL.yaml").read_text(encoding="utf-8"))
    n_runs = protocol["budget"]["planned_trained_agents"]
    n_s1 = protocol["inference"]["controller_contrast_family_S1"]["n_contrasts"]
    n_s2 = protocol["inference"]["support_contrast_family_S2"]["n_contrasts"]

    analysis = root / "results/continuous_control/analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "audit.json").write_text(
        json.dumps({
            "status": "PASS",
            "inferential_status": "SUPPLEMENTAL_NONCONFIRMATORY",
            "complete_runs": n_runs,
            "final_episode_rows": 3600,
            "checkpoint_episode_rows": 600,
            "S1_tests": n_s1,
            "S2_tests": n_s2,
            "analysis_bootstrap_base_seed": 56001,
        }, indent=2),
        encoding="utf-8",
    )
    for name, family, count in (
        ("S1_supplemental_controller_contrasts.csv", "S1", n_s1),
        ("S2_supplemental_support_contrasts.csv", "S2", n_s2),
    ):
        with (analysis / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["family", "contrast_id"])
            writer.writeheader()
            for i in range(count):
                writer.writerow({"family": family, "contrast_id": i})

    direct = root / "requirements-continuous-control.txt"
    direct.write_text("numpy==2.5.1\npandas==3.0.5\nPyYAML==6.0.3\n", encoding="utf-8")
    tested = root / "requirements-tested-continuous-control.txt"
    tested.write_text("numpy==2.5.1\npandas==3.0.5\nPyYAML==6.0.3\n", encoding="utf-8")
    _write_sha_manifest(root / "requirements-tested-continuous-control.sha256", [tested], root)

    required = [
        ".gitignore", "LICENSE", "README.md", "PROVENANCE.md", "REPRODUCIBILITY.md", "CITATION.cff",
        ".zenodo.json", "pyproject.toml", "requirements.txt", "requirements-tested.txt",
        "requirements-uav.txt", "requirements-tested-uav.txt", "src/hybrid_q/agents.py",
        "scripts/reproduce_all.py", "scripts/audit_artifact.py", "scripts/preflight_release.py",
        "scripts/generate_tables.py", "scripts/generate_figures.py", "tests/test_agents.py",
        "tests/test_artifact_audit.py", "tests/test_protocol_integrity.py",
        "tables/table_principal_comparisons.csv",
    ]
    for relative in required:
        if not (root / relative).exists():
            _touch_path(root, relative)
    return root


def test_prepublication_audit_accepts_complete_e01_e28_tree(tmp_path: Path):
    audit_module = _load_audit_module()
    root = build_minimal_public_tree(tmp_path)
    report = audit_module.audit(root, require_manifest=False)
    assert report["status"] == "PASS", report["violations"]
    assert report["summary"]["evidence"]["family_count"] == 28
    assert report["summary"]["claims"]["pass_count"] == 16
    assert report["summary"]["continuous_control"]["S1_tests"] == 8
    assert report["summary"]["continuous_control"]["S2_tests"] == 12


def test_audit_rejects_missing_e28_source(tmp_path: Path):
    audit_module = _load_audit_module()
    root = build_minimal_public_tree(tmp_path)
    registry = json.loads((root / "configs/evidence_registry.json").read_text(encoding="utf-8"))
    e28 = next(row for row in registry["families"] if row["id"] == "E28")
    target = root / e28["source"].split(";")[0].strip()
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    report = audit_module.audit(root)
    assert report["status"] == "FAIL"
    assert any("E28 source missing" in item for item in report["violations"])


def test_audit_rejects_private_tree_and_legacy_family_label(tmp_path: Path):
    audit_module = _load_audit_module()
    root = build_minimal_public_tree(tmp_path)
    (root / "paper").mkdir()
    s1 = root / "results/continuous_control/analysis/S1_supplemental_controller_contrasts.csv"
    text = s1.read_text(encoding="utf-8").replace("S1,", "P1,", 1)
    s1.write_text(text, encoding="utf-8")
    report = audit_module.audit(root)
    assert report["status"] == "FAIL"
    assert any("private/admin tree" in item for item in report["violations"])
    assert any("family values mismatch" in item for item in report["violations"])


def test_frozen_manifest_requires_complete_public_file_coverage(tmp_path: Path):
    audit_module = _load_audit_module()
    root = build_minimal_public_tree(tmp_path)

    no_manifest = audit_module.audit(root, require_manifest=True)
    assert no_manifest["status"] == "FAIL"
    assert any("MANIFEST.sha256" in item for item in no_manifest["violations"])

    # An incomplete-but-correct manifest must still fail.
    target = root / "README.md"
    _write_sha_manifest(root / "MANIFEST.sha256", [target], root)
    incomplete = audit_module.audit(root, require_manifest=True)
    assert incomplete["status"] == "FAIL"
    assert any("manifest omits public file" in item for item in incomplete["violations"])

    targets = [
        path for path in root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and audit_module._is_auditable_file(root, path)
    ]
    _write_sha_manifest(root / "MANIFEST.sha256", targets, root)
    complete = audit_module.audit(root, require_manifest=True)
    assert complete["status"] == "PASS", complete["violations"]
    assert complete["summary"]["repository_manifest"]["missing_entries"] == 0
