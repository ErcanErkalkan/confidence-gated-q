from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


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
