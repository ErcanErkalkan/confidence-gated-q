from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".quick_repro",
    ".uav_smoke",
    ".venv",
    "__pycache__",
    "build",
    "confidence_gated_q.egg-info",
    "dist",
    "paper",
    "runs",
}
EXCLUDED_NAMES = {
    "artifact_audit.json",
    "submission_readiness_audit.json",
    "MANIFEST.sha256",
}
BUILD_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if (
            path.name in EXCLUDED_NAMES
            or path.suffix in BUILD_SUFFIXES
            or any(part in EXCLUDED_PARTS for part in relative.parts)
            or path.suffix.lower() == ".zip"
            or (
                "results" in relative.parts
                and path.name.startswith("learning_curve_")
                and path.suffix.lower() == ".png"
            )
        ):
            continue
        files.append(path)
    return files


def write_manifest() -> Path:
    manifest = ROOT / "MANIFEST.sha256"
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in artifact_files()
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def submission_files() -> list[Path]:
    files = []
    for path in sorted((ROOT / "paper").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT / "paper")
        if (
            path.suffix in BUILD_SUFFIXES
            or any(
                part in {".git", ".pytest_cache", "__pycache__"}
                for part in relative.parts
            )
            or path.name == "manuscript.pdf"
        ):
            continue
        files.append(path)
    return files


def write_zip(path: Path, entries: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, archive_name in entries:
            archive.write(source, archive_name)
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if (
                name.lower().endswith(".zip")
                or any(
                    part in {".git", ".pytest_cache", "__pycache__", "dist"}
                    for part in parts
                )
            ):
                raise RuntimeError(f"forbidden package entry: {name}")


def build_packages() -> list[Path]:
    DIST.mkdir(parents=True, exist_ok=True)
    compiled = ROOT / "paper/manuscript.pdf"
    if not compiled.exists():
        raise FileNotFoundError("paper/manuscript.pdf must be built first")
    final_pdf = DIST / "manuscript.pdf"
    shutil.copy2(compiled, final_pdf)

    submission_zip = DIST / "submission.zip"
    submission_entries = [
        (path, path.relative_to(ROOT / "paper").as_posix())
        for path in submission_files()
        if path != compiled
    ]
    submission_entries.append((final_pdf, final_pdf.name))
    write_zip(submission_zip, submission_entries)

    artifact_zip = DIST / "research_artifact.zip"
    artifact_entries = [
        (path, path.relative_to(ROOT).as_posix())
        for path in artifact_files()
    ]
    for extra in (
        ROOT / "MANIFEST.sha256",
        ROOT / "artifact_audit.json",
        ROOT / "submission_readiness_audit.json",
    ):
        artifact_entries.append((extra, extra.name))
    write_zip(artifact_zip, artifact_entries)
    return [submission_zip, artifact_zip, final_pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest = write_manifest()
    print(manifest.relative_to(ROOT))
    if args.manifest_only:
        return
    for path in build_packages():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
