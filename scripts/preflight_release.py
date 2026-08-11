from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-training release preflight checks for the public artifact.")
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Require complete MANIFEST.sha256 coverage for the frozen-release preflight.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="confidence_gated_q_preflight_") as temp_name:
        temp = Path(temp_name)
        env = os.environ.copy()
        env["PYTHONPYCACHEPREFIX"] = str(temp / "pycache")
        source = str(ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(part for part in (source, env.get("PYTHONPATH", "")) if part)

        run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"], env=env)
        run([
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--basetemp", str(temp / "pytest"),
        ], env=env)

        report_path = temp / "artifact_audit.json"
        audit_command = [
            sys.executable,
            "scripts/audit_artifact.py",
            "--root", ".",
            "--output", str(report_path),
        ]
        if args.require_manifest:
            audit_command.append("--require-manifest")
        run(audit_command, env=env)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise SystemExit("artifact audit did not report PASS")

    mode = "FROZEN_RELEASE" if args.require_manifest else "PREPUBLICATION"
    print(f"{mode}_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
