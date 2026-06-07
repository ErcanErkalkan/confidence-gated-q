from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_results import audit
from hybrid_q.experiment import run_config
from hybrid_q.statistics import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/adaptive_gate_compact_validation.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_dir = Path(config["output_dir"])
    raw_path = run_config(config_path)
    aggregate(raw_path, result_dir)
    report = audit(config_path, result_dir)
    audit_path = result_dir / "audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"{report['status']}: {audit_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
