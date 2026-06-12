from __future__ import annotations

import json
from pathlib import Path

from audit_results import audit
from hybrid_q.config import load_config
from hybrid_q.experiment import run_config
from hybrid_q.statistics import aggregate


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/strong_baselines/double_dqn_30seed.yaml",
    ROOT / "configs/strong_baselines/dueling_double_dqn_30seed.yaml",
)


def main() -> None:
    for config_path in CONFIGS:
        config = load_config(config_path)
        result_dir = ROOT / config["output_dir"]
        raw_path = run_config(config_path)
        aggregate(raw_path, result_dir)
        report = audit(config_path, result_dir)
        (result_dir / "audit.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"{config_path.name}: {report['status']}")
        if report["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
