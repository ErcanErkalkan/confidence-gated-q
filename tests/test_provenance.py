from __future__ import annotations

from pathlib import Path

from hybrid_q.provenance import (
    execution_input_manifest,
    execution_snapshot_sha256,
    repository_root,
)


def test_execution_snapshot_changes_with_config_content(tmp_path: Path):
    root = tmp_path
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    package = root / "src" / "hybrid_q"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__='1.0'\n")
    config = root / "config.yaml"
    config.write_text("seeds: [1]\n")

    first = execution_snapshot_sha256(
        execution_input_manifest(config, root)
    )
    config.write_text("seeds: [1, 2]\n")
    second = execution_snapshot_sha256(
        execution_input_manifest(config, root)
    )

    assert first != second


def test_repository_root_from_checkout_root():
    expected = Path(__file__).resolve().parents[1]
    assert repository_root(expected) == expected


def test_repository_root_from_nested_directory():
    expected = Path(__file__).resolve().parents[1]
    nested = expected / "src" / "hybrid_q"
    assert repository_root(nested) == expected


def test_repository_root_uses_environment_override(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    outside = tmp_path / "external" / "config.yaml"
    outside.parent.mkdir()
    outside.write_text("seeds: [1]\n")
    monkeypatch.setenv("HYBRID_Q_REPO_ROOT", str(root))
    assert repository_root(outside) == root
