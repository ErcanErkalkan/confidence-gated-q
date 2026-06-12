from __future__ import annotations

from pathlib import Path

from hybrid_q.provenance import (
    execution_input_manifest,
    execution_snapshot_sha256,
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
