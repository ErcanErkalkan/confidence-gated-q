from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"


def fuzzy_rule_base() -> None:
    """Generate the public fuzzy-rule-base CSV used by the artifact."""
    rows = [
        ("Low", "Any", 0.00, "Prefer neural estimate; no memory evidence"),
        ("Medium", "Low", 0.35, "Mostly neural"),
        ("Medium", "High", 0.65, "Mostly memory"),
        ("High", "Low", 0.75, "Memory-led mixture"),
        ("High", "High", 0.95, "Strong memory preference"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "support_membership",
            "uncertainty_membership",
            "memory_weight",
            "interpretation",
        ],
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "table_fuzzy_rule_base.csv", index=False)


def sensor_nondegenerate_feasibility() -> None:
    """Regenerate the public E28 development-feasibility summary table."""
    source = (
        RESULTS
        / "diagnostic_extensions"
        / "sensor_nondegenerate_development"
        / "feasibility"
        / "feasibility_selection.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["candidate_diagnostics"])[
        [
            "candidate_id",
            "training_interactions",
            "episode_horizon",
            "reference_success_max",
            "learned_success_min",
            "learned_success_max",
            "eligible",
        ]
    ].copy()
    frame["evidence_class"] = "development_feasibility"
    frame["source_file"] = (
        "results/diagnostic_extensions/sensor_nondegenerate_development/"
        "feasibility/feasibility_selection.yaml"
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "table_sensor_nondegenerate_feasibility.csv", index=False)


def main() -> None:
    fuzzy_rule_base()
    sensor_nondegenerate_feasibility()
    print("Generated public artifact tables.")


if __name__ == "__main__":
    main()
