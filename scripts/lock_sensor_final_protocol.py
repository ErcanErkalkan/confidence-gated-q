from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs/diagnostic_extensions/sensorized_final/protocol_lock.yaml"
)
DEFAULT_DIGEST = DEFAULT_PROTOCOL.with_suffix(".sha256")
EXPECTED_SEEDS = list(range(16000, 16030))
EXPECTED_CONDITIONS = {"combined_executed_condition", "latency_only"}
EXPECTED_AGENTS = {
    "feed_forward_dqn",
    "selected_temporal_drqn",
    "fuzzy_relative_reliability",
    "selected_approximate_support",
    "sensorized_controller",
}


class SensorFinalProtocolError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_hash(root: Path, entry: dict[str, Any]) -> None:
    path = root / str(entry["source_file"])
    if not path.is_file() or sha256(path) != str(entry["sha256"]):
        raise SensorFinalProtocolError(f"prerequisite hash mismatch: {path}")


def validate_protocol(protocol: dict[str, Any], root: Path = ROOT) -> None:
    if protocol.get("protocol_id") != "temporal_sensorized_final":
        raise SensorFinalProtocolError("unexpected protocol_id")
    prerequisites = protocol.get("prerequisites", {})
    for key in (
        "selected_temporal_model",
        "selected_support_estimator",
    ):
        _check_hash(root, prerequisites[key])
    original = prerequisites["original_shift_protocol"]
    if original.get("amendment_created") is not False:
        raise SensorFinalProtocolError("original lock must remain unamended")

    selection = protocol["condition_selection"]
    source = root / selection["evidence_source"]
    if sha256(source) != selection["evidence_sha256"]:
        raise SensorFinalProtocolError("condition-selection evidence changed")
    audit = pd.read_csv(source).set_index("condition")
    selected = str(selection["selected_isolated_condition"])
    eligible = []
    metric_by_condition = {
        "latency_only": "effective_latency_mean",
        "localization_dropout_only": "localization_dropout_rate",
        "visibility_occlusion_only": "target_visibility_rate",
        "range_dropout_only": "range_dropout_rate",
        "camera_dropout_only": "camera_dropout_rate",
        "noise_only": "localization_error_mean",
    }
    for condition in selection["priority_order"]:
        row = audit.loc[condition]
        metric = metric_by_condition[condition]
        baseline = float(audit.loc["no_noise_no_delay_no_dropout", metric])
        value = float(row[metric])
        changed = value < baseline if condition == "visibility_occlusion_only" else value > baseline
        if row["audit_status"] == "PASS" and changed:
            eligible.append(condition)
    if not eligible or selected != eligible[0] or selected != "latency_only":
        raise SensorFinalProtocolError("isolated-condition selection rule mismatch")

    conditions = protocol["final_conditions"]
    if {item["condition_id"] for item in conditions} != EXPECTED_CONDITIONS:
        raise SensorFinalProtocolError("final condition set mismatch")
    combined = next(x for x in conditions if x["condition_id"] == "combined_executed_condition")
    latency = next(x for x in conditions if x["condition_id"] == "latency_only")
    if not all(combined["factor_flags"].values()):
        raise SensorFinalProtocolError("combined condition must enable all factors")
    if {k for k, v in latency["factor_flags"].items() if v} != {"sensor_latency_enabled"}:
        raise SensorFinalProtocolError("latency-only factor isolation mismatch")

    if protocol["final_seeds"]["seeds"] != EXPECTED_SEEDS:
        raise SensorFinalProtocolError("final seeds must be exactly 16000-16029")
    if {item["agent_id"] for item in protocol["agents"]} != EXPECTED_AGENTS:
        raise SensorFinalProtocolError("agent set mismatch")
    budget = protocol["budget"]
    if (
        budget["training_interactions_per_agent_seed_condition"] != 240
        or budget["checkpoint_schedule"] != [120, 240]
        or budget["evaluation_episodes_per_checkpoint"] != 4
        or budget["episode_horizon"] != 30
    ):
        raise SensorFinalProtocolError("final budget mismatch")
    contrasts = protocol["analysis"]["planned_contrasts"]
    if len(contrasts) != 4 or contrasts[0]["status"] != "primary_replication":
        raise SensorFinalProtocolError("planned contrast family mismatch")
    if protocol["analysis"]["report_holm_scope"] != "all_eight_rows_together":
        raise SensorFinalProtocolError("Holm family scope mismatch")
    if protocol["trace_lock"]["schema_version"] != "sensorized_sil_trace_v2":
        raise SensorFinalProtocolError("trace schema lock mismatch")


def load_and_validate(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise SensorFinalProtocolError("protocol must be a mapping")
    validate_protocol(protocol, ROOT)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--digest", type=Path, default=DEFAULT_DIGEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    path = args.protocol.resolve()
    load_and_validate(path)
    digest = sha256(path)
    if args.validate_only:
        if args.digest.exists():
            expected = args.digest.read_text(encoding="utf-8").split()[0]
            if expected != digest:
                raise SensorFinalProtocolError("protocol digest mismatch")
        print("SENSOR_FINAL_PROTOCOL_VALIDATION_PASS")
        return
    args.digest.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    print(f"SENSOR_FINAL_PROTOCOL_LOCKED sha256={digest}")


if __name__ == "__main__":
    main()
