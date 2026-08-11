import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_tail_risk import aggregate_tail_risk
from scripts.audit_results import audit_tail_risk


ROOT = Path(__file__).resolve().parents[1]


def _write_completed_family(results_root: Path) -> None:
    family = results_root / "completed_family"
    family.mkdir(parents=True)
    (family / "metadata.json").write_text("{}", encoding="utf-8")
    (family / "audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    seed_rows = []
    raw_rows = []
    for agent, seeds in (("full", range(30)), ("sparse", range(2))):
        for seed in seeds:
            seed_rows.append(
                {
                    "environment": "env",
                    "agent": agent,
                    "seed": seed,
                    "checkpoint": 100,
                    "mean_return": seed - 4.0,
                    "failure_rate": seed % 2,
                    "collision_rate": 0.1,
                    "risk_zone_rate": 0.2,
                    "motor_saturation_rate": 0.3,
                }
            )
            for checkpoint, offset in ((50, 0.0), (100, -5.0)):
                for episode in range(3):
                    raw_rows.append(
                        {
                            "environment": "env",
                            "agent": agent,
                            "seed": seed,
                            "phase": "eval",
                            "checkpoint": checkpoint,
                            "episode": episode,
                            "return": seed + offset + episode,
                        }
                    )
    pd.DataFrame(seed_rows).to_csv(family / "seed_metrics.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(family / "raw.csv", index=False)


def test_tail_risk_aggregator_and_schema_audit(tmp_path):
    results_root = tmp_path / "results"
    output_dir = results_root / "reviewer1_remaining" / "tail_risk"
    table_path = tmp_path / "tables" / "table_tail_risk.csv"
    figure_path = tmp_path / "figures" / "fig_tail_risk.pdf"
    audit_path = tmp_path / "tail_risk_audit.json"
    _write_completed_family(results_root)

    counts = aggregate_tail_risk(
        results_root,
        output_dir,
        table_path,
        figure_path,
    )
    assert counts == {
        "families": 1,
        "groups": 2,
        "episode_groups": 2,
        "checkpoint_rows": 4,
        "manifest_rows": 36,
    }
    expected = {
        "seed_tail_risk.csv",
        "episode_tail_risk.csv",
        "checkpoint_tail_risk.csv",
        "tail_risk_summary.csv",
        "safety_metrics_manifest.csv",
    }
    assert expected == {path.name for path in output_dir.glob("*.csv")}
    assert table_path.exists()
    assert figure_path.read_bytes().startswith(b"%PDF-")

    seed = pd.read_csv(output_dir / "seed_tail_risk.csv")
    full = seed[seed["agent"] == "full"].iloc[0]
    sparse = seed[seed["agent"] == "sparse"].iloc[0]
    assert full["cvar_0_10_tail_count"] == 3
    assert full["cvar_0_10_return"] == -3.0
    assert full["worst_decile_mean_return"] == full["cvar_0_10_return"]
    assert full["cvar_0_05_tail_count"] == 2
    assert full["cvar_0_05_return"] == -3.5
    assert bool(full["cvar_0_05_available"])
    assert not bool(sparse["cvar_0_05_available"])
    assert np.isnan(sparse["cvar_0_05_return"])

    episode = pd.read_csv(output_dir / "episode_tail_risk.csv")
    assert episode.loc[episode["agent"] == "full", "n_finite_returns"].item() == 90
    summary = pd.read_csv(output_dir / "tail_risk_summary.csv")
    assert summary.loc[
        summary["agent"] == "full", "maximum_learning_curve_drawdown"
    ].item() == 5.0

    manifest = pd.read_csv(output_dir / "safety_metrics_manifest.csv")
    cvar_definitions = manifest[
        manifest["metric_name"].str.contains("cvar", case=False)
    ]["definition"]
    assert cvar_definitions.str.contains("Arithmetic mean", case=False).all()
    trace_metrics = manifest[
        manifest["metric_name"].isin(["recovery_time", "trajectory_deviation"])
    ]
    assert (trace_metrics["availability"] == "not_available").all()
    assert trace_metrics["reason_if_unavailable"].notna().all()

    report = audit_tail_risk(output_dir, table_path, figure_path)
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["status"] == "PASS"
    assert json.loads(audit_path.read_text(encoding="utf-8"))["status"] == "PASS"
