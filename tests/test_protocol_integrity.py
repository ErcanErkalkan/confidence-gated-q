from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_hash_target(sidecar: Path, token: str | None) -> Path:
    if token:
        local = sidecar.parent / token
        if local.exists():
            return local
        rooted = ROOT / token
        if rooted.exists():
            return rooted
        return local
    # Single-token sidecars conventionally protect the same path without .sha256.
    return sidecar.with_suffix("")


def test_evidence_registry_is_exactly_e01_e28_v1():
    payload = json.loads((ROOT / "configs/evidence_registry.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["revision_scope"]["artifact_version"] == "1.0.0"
    assert payload["revision_scope"]["family_count"] == 28
    assert [row["id"] for row in payload["families"]] == [f"E{i:02d}" for i in range(1, 29)]


def test_all_config_sha256_sidecars_match_their_targets():
    sidecars = sorted(ROOT.glob("configs/**/*.sha256"))
    assert sidecars, "expected at least one config SHA-256 sidecar"
    for sidecar in sidecars:
        checked = 0
        for raw in sidecar.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            assert HASH_RE.match(parts[0]), f"invalid SHA-256 record in {sidecar}: {line}"
            target_token = parts[1].lstrip("*") if len(parts) == 2 else None
            if target_token and "+" in target_token:
                targets = [sidecar.parent / item for item in target_token.split("+")]
                for target in targets:
                    assert target.is_file(), f"missing hash target for {sidecar}: {target}"
                digest = hashlib.sha256()
                for target in targets:
                    digest.update(target.read_bytes())
                    digest.update(b"\0")
                assert digest.hexdigest() == parts[0], (
                    f"composite SHA-256 mismatch for {sidecar}: {target_token}"
                )
            else:
                target = resolve_hash_target(sidecar, target_token)
                assert target.is_file(), f"missing hash target for {sidecar}: {target}"
                assert sha256(target) == parts[0], f"SHA-256 mismatch for {target}"
            checked += 1
        assert checked > 0, f"no SHA-256 records found in {sidecar}"


def test_continuous_control_protocol_and_analysis_counts_are_locked():
    protocol = yaml.safe_load((ROOT / "configs/continuous_control/CONTINUOUS_CONTROL_PROTOCOL.yaml").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "results/continuous_control/analysis/audit.json").read_text(encoding="utf-8"))
    assert protocol["budget"]["planned_trained_agents"] == 30
    assert protocol["inference"]["controller_contrast_family_S1"]["n_contrasts"] == 8
    assert protocol["inference"]["support_contrast_family_S2"]["n_contrasts"] == 12
    assert analysis["status"] == "PASS"
    assert analysis["inferential_status"] == "SUPPLEMENTAL_NONCONFIRMATORY"
    assert analysis["complete_runs"] == 30
    assert analysis["final_episode_rows"] == 3600
    assert analysis["checkpoint_episode_rows"] == 600
    assert analysis["S1_tests"] == 8
    assert analysis["S2_tests"] == 12


def test_continuous_control_environment_freeze_sha_matches():
    lock = ROOT / "requirements-tested-continuous-control.sha256"
    line = lock.read_text(encoding="utf-8").strip()
    expected, relative = line.split(maxsplit=1)
    target = ROOT / relative.lstrip("*")
    assert HASH_RE.match(expected)
    assert target.is_file()
    assert sha256(target) == expected
    for raw in (ROOT / "requirements-continuous-control.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            assert "==" in line, f"unpinned continuous-control dependency: {line}"


def test_public_generators_have_no_private_manuscript_dependency():
    scripts = ROOT / "scripts"
    assert not list(scripts.glob("generate_paper_*.py"))
    for name in ("generate_tables.py", "generate_figures.py"):
        text = (scripts / name).read_text(encoding="utf-8").lower()
        assert "01_private_paper" not in text
        assert 'root / "paper"' not in text
        assert "root / 'paper'" not in text
