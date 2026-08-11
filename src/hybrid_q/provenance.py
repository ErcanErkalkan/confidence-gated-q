from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_MARKERS = ("pyproject.toml", ".git", "PROVENANCE.md")


def _find_repository_root(candidate: Path) -> Path | None:
    candidate = candidate.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if any((parent / marker).exists() for marker in PROJECT_MARKERS):
            return parent
    return None


def repository_root(start: Path | None = None) -> Path:
    override = os.environ.get("HYBRID_Q_REPO_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                "HYBRID_Q_REPO_ROOT does not name an existing directory: "
                f"{root}"
            )
        if not any((root / marker).exists() for marker in PROJECT_MARKERS):
            raise RuntimeError(
                "HYBRID_Q_REPO_ROOT does not contain a recognized project "
                f"marker ({', '.join(PROJECT_MARKERS)}): {root}"
            )
        return root

    candidates = []
    if start is not None:
        candidates.append(Path(start))
    candidates.extend((Path.cwd(), Path(__file__)))
    for candidate in candidates:
        root = _find_repository_root(candidate)
        if root is not None:
            return root

    searched = ", ".join(str(Path(item).resolve()) for item in candidates)
    raise RuntimeError(
        "Could not locate the confidence-gated-q repository root. Searched "
        f"from: {searched}. Set HYBRID_Q_REPO_ROOT to the checkout directory."
    )


def _logical_path(path: Path, root: Path, *, config: bool = False) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        prefix = "external-config" if config else "external-input"
        return f"{prefix}/{path.name}"


def execution_input_files(
    config_path: Path,
    root: Path | None = None,
) -> list[tuple[str, Path]]:
    root = (root or repository_root(config_path)).resolve()
    config_path = config_path.resolve()
    candidates: set[Path] = {
        root / "pyproject.toml",
        root / "environment.yml",
        config_path,
    }
    candidates.update(root.glob("requirements*.txt"))
    candidates.update((root / "src" / "hybrid_q").rglob("*.py"))
    files = [
        (_logical_path(path, root, config=path == config_path), path)
        for path in candidates
        if path.is_file()
    ]
    return sorted(files, key=lambda item: item[0])


def execution_input_manifest(
    config_path: Path,
    root: Path | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "path": logical_path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for logical_path, path in execution_input_files(config_path, root)
    ]


def execution_snapshot_sha256(manifest: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for entry in manifest:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without normalizing its bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable, compact representation."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hil_execution_input_files(
    protocol_path: Path,
    root: Path | None = None,
) -> list[tuple[str, Path]]:
    """Return source, schema, and protocol inputs for a future HIL run."""

    root = (root or repository_root(protocol_path)).resolve()
    protocol_path = Path(protocol_path).resolve()
    candidates: set[Path] = {
        root / "pyproject.toml",
        protocol_path,
    }
    candidates.update(root.glob("requirements*.txt"))
    candidates.update((root / "src" / "hybrid_q").rglob("*.py"))
    hil_root = root / "hil"
    if hil_root.is_dir():
        for suffix in ("*.py", "*.json", "*.yaml", "*.md"):
            candidates.update(hil_root.rglob(suffix))
    files = [
        (_logical_path(path, root, config=path == protocol_path), path)
        for path in candidates
        if path.is_file()
    ]
    return sorted(files, key=lambda item: item[0])


def hil_execution_input_manifest(
    protocol_path: Path,
    root: Path | None = None,
) -> list[dict[str, str]]:
    """Build a content-addressed manifest for future HIL log provenance."""

    return [
        {"path": logical_path, "sha256": file_sha256(path)}
        for logical_path, path in hil_execution_input_files(protocol_path, root)
    ]


def hil_provenance_record(
    *,
    protocol_path: Path,
    adapter_name: str,
    adapter_version: str,
    source_trace_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Create traceable provenance for a dry-run or future supervised HIL log.

    The returned mapping contains no claim about physical execution. Callers
    must separately record the run mode and physical-evidence status.
    """

    if not adapter_name or not adapter_version:
        raise ValueError("adapter_name and adapter_version must be non-empty")
    protocol_path = Path(protocol_path).resolve()
    root = (root or repository_root(protocol_path)).resolve()
    manifest = hil_execution_input_manifest(protocol_path, root)
    source = None if source_trace_path is None else Path(source_trace_path)
    return deepcopy(
        {
            "repository_commit": git_commit_hash(root),
            "hil_input_snapshot_sha256": execution_snapshot_sha256(manifest),
            "protocol_sha256": file_sha256(protocol_path),
            "source_trace_path": (
                None
                if source is None
                else _logical_path(source.resolve(), root)
            ),
            "source_trace_sha256": (
                None if source is None else file_sha256(source)
            ),
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "input_manifest": manifest,
        }
    )


def git_commit_hash(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def execution_inputs_clean(
    config_path: Path,
    root: Path | None = None,
) -> bool | None:
    root = (root or repository_root(config_path)).resolve()
    tracked_paths = []
    for _, path in execution_input_files(config_path, root):
        try:
            tracked_paths.append(path.resolve().relative_to(root).as_posix())
        except ValueError:
            continue
    if not tracked_paths:
        return None
    try:
        output = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *tracked_paths,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return not bool(output.strip())
