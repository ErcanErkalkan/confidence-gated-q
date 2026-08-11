from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/diagnostic_extensions/safety_traces/protocol.yaml"
DEFAULT_DIGEST = DEFAULT_PROTOCOL.with_suffix(".sha256")
EXPECTED_SEEDS = list(range(16030, 16040))
EXPECTED_FAMILIES = {
    "combined_executed_condition_safety_trace",
    "latency_only_safety_trace",
}
EXPECTED_AGENTS = {
    "feed_forward_dqn",
    "selected_temporal_drqn",
    "fuzzy_relative_reliability",
    "selected_approximate_support",
    "sensorized_controller",
}


class SafetyTraceProtocolError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(protocol: dict[str, Any], root: Path = ROOT) -> None:
    if protocol.get("protocol_id") != "safety_trace_rerun":
        raise SafetyTraceProtocolError("unexpected protocol_id")
    scope = protocol.get("scope", {})
    source = scope.get("source_sensor_protocol", {})
    source_path = root / str(source.get("file", ""))
    if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
        raise SafetyTraceProtocolError("source sensor protocol hash mismatch")
    if scope.get("prior_outcomes_used_to_choose_definitions") is not False:
        raise SafetyTraceProtocolError("definitions must be outcome-independent")
    if scope.get("inferential_superiority_claims_permitted") is not False:
        raise SafetyTraceProtocolError("safety rerun must remain descriptive")

    families = protocol.get("rerun_families", [])
    if {item.get("family_id") for item in families} != EXPECTED_FAMILIES:
        raise SafetyTraceProtocolError("unexpected safety rerun family set")
    if protocol.get("new_seeds", {}).get("seeds") != EXPECTED_SEEDS:
        raise SafetyTraceProtocolError("new seed set must be exactly 16030-16039")
    if set(protocol.get("agents", [])) != EXPECTED_AGENTS:
        raise SafetyTraceProtocolError("unexpected agent set")

    budget = protocol.get("budget", {})
    expected_budget = {
        "training_interactions_per_agent_seed_family": 240,
        "checkpoint_schedule": [120, 240],
        "evaluation_episodes_per_checkpoint": 4,
        "episode_horizon_steps": 30,
        "control_frequency_hz": 60,
    }
    for key, expected in expected_budget.items():
        if budget.get(key) != expected:
            raise SafetyTraceProtocolError(f"budget mismatch: {key}")

    definitions = protocol.get("definitions", {})
    onset = definitions.get("perturbation_onset", {})
    recovery = definitions.get("recovery", {})
    if onset.get("onset_step") != 6 or onset.get("step_indexing") != "one_based_post_action":
        raise SafetyTraceProtocolError("invalid perturbation onset")
    if recovery.get("band_m") != 0.15 or recovery.get("required_dwell_steps") != 6:
        raise SafetyTraceProtocolError("invalid recovery definition")
    if recovery.get("non_recovery_rule") != "right_censored_at_last_observed_post_onset_timestamp":
        raise SafetyTraceProtocolError("non-recovery rule must be locked right censoring")
    if definitions.get("near_miss", {}).get("availability") != "available_for_declared_obstacle_geometry":
        raise SafetyTraceProtocolError("near-miss availability must be explicit")
    if protocol.get("trace", {}).get("schema_version") != "sensorized_sil_trace_v3":
        raise SafetyTraceProtocolError("trace schema mismatch")


def load_and_validate(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise SafetyTraceProtocolError("protocol must be a mapping")
    validate_protocol(protocol)
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
        if not args.digest.is_file():
            raise SafetyTraceProtocolError("protocol digest file is missing")
        expected = args.digest.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise SafetyTraceProtocolError("protocol digest mismatch")
        print("SAFETY_TRACE_PROTOCOL_VALIDATION_PASS")
        return
    args.digest.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    print(f"SAFETY_TRACE_PROTOCOL_LOCKED sha256={digest}")


if __name__ == "__main__":
    main()
