from __future__ import annotations

import json
from pathlib import Path

from audit_results import audit
from hybrid_q.config import load_config
from hybrid_q.experiment import run_config
from hybrid_q.statistics import aggregate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/application_risk_variants_30seed.yaml"


def main() -> None:
    config = load_config(CONFIG)
    result_dir = ROOT / config["output_dir"]
    raw_path = run_config(CONFIG)
    aggregate(raw_path, result_dir)
    report = audit(CONFIG, result_dir)
    (result_dir / "audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(report["status"])
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
