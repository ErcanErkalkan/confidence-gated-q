from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


SEED_SUFFIX = re.compile(r"__seed_\d+\.csv$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(result_dir: Path) -> Path:
    runs_dir = result_dir / "runs"
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(runs_dir.glob("*.csv")):
        group = SEED_SUFFIX.sub("", path.name)
        groups[group].append(path)
    if not groups:
        raise FileNotFoundError(f"No run shards found in {runs_dir}")

    output_dir = result_dir / "raw_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for group, paths in sorted(groups.items()):
        output = output_dir / f"{group}.csv.gz"
        with gzip.open(output, "wb", compresslevel=9) as target:
            for index, path in enumerate(paths):
                with path.open("rb") as source:
                    if index:
                        source.readline()
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        entries.append(
            {
                "path": output.relative_to(result_dir).as_posix(),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
                "run_shards": len(paths),
            }
        )

    manifest = {
        "format": "gzip_csv_parts_by_environment_and_agent",
        "parts": entries,
        "total_parts": len(entries),
        "total_run_shards": sum(len(paths) for paths in groups.values()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    print(pack(Path(args.result_dir)))


if __name__ == "__main__":
    main()
