from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/diagnostic_extensions/INDEPENDENT_SHIFT_PROTOCOL.yaml"
DEFAULT_COMPANION = None
DEFAULT_DIGEST = ROOT / "configs/diagnostic_extensions/INDEPENDENT_SHIFT_PROTOCOL.sha256"
DEFAULT_SEED_REGISTRY = ROOT / "configs/diagnostic_extensions/seed_registry.yaml"

EXPECTED_MECHANISMS = {
    "transition_dynamics_shift",
    "observation_shift",
    "localized_multistep_reward_or_policy_shift",
}
REQUIRED_MECHANISM_FIELDS = {
    "mechanism_id",
    "environment_definition",
    "what_changes",
    "what_remains_fixed",
    "shift_onset",
    "development_severity_candidates",
    "development_seeds",
    "final_seeds",
    "training_interaction_budget",
    "checkpoint_schedule",
    "evaluation_episodes",
    "primary_metric",
    "primary_contrast",
    "co_primary_contrast",
    "secondary_diagnostics",
    "report_level_holm_family",
    "exclusion_rules",
    "stopping_rules",
}
REQUIRED_SECONDARY_DIAGNOSTICS = {
    "success_auc",
    "failure_probability",
    "detection_delay",
    "branch_correctness",
    "lower_tail_summaries",
}
PRIMARY_CONTRAST = {
    "contrast_id": "relative_reliability_fuzzy_vs_count_gated_tau_20",
    "left": "relative_reliability_fuzzy",
    "right": "count_gated_tau_20",
}
CO_PRIMARY_CONTRAST = {
    "contrast_id": "same_input_crisp_vs_relative_reliability_fuzzy",
    "left": "same_input_crisp",
    "right": "relative_reliability_fuzzy",
}


class ProtocolLockError(ValueError):
    pass


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolLockError(f"{label} must be a mapping")
    return value


def _require_fields(mapping: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ProtocolLockError(f"{label} missing required fields: {missing}")


def _seed_interval(value: Any, label: str) -> tuple[int, int, str]:
    seeds = _mapping(value, label)
    _require_fields(seeds, {"start", "end", "count", "registry_key"}, label)
    start = seeds["start"]
    end = seeds["end"]
    count = seeds["count"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end, count)):
        raise ProtocolLockError(f"{label} start/end/count must be integers")
    if start > end or count != end - start + 1:
        raise ProtocolLockError(f"{label} is not a closed contiguous interval")
    registry_key = seeds["registry_key"]
    if not isinstance(registry_key, str) or not registry_key:
        raise ProtocolLockError(f"{label} registry_key must be non-empty")
    return start, end, registry_key


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _validate_contrast(value: Any, expected: dict[str, str], label: str) -> None:
    contrast = _mapping(value, label)
    _require_fields(
        contrast,
        {"contrast_id", "left", "right", "alternative", "status"},
        label,
    )
    for field, expected_value in expected.items():
        if contrast[field] != expected_value:
            raise ProtocolLockError(
                f"{label} {field} must be {expected_value!r}"
            )
    if contrast["alternative"] != "two_sided":
        raise ProtocolLockError(f"{label} must be two-sided")


def validate_protocol(protocol: dict[str, Any], seed_registry: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_id",
        "lock_status",
        "final_results_inspected",
        "immutability",
        "seed_registry",
        "analysis_lock",
        "agent_lock",
        "development_selection",
        "report_level_holm_families",
        "mechanisms",
    }
    _require_fields(protocol, required, "protocol")
    if (
        protocol["schema_version"] != 2
        or protocol["protocol_id"] != "independent_shift_replication"
    ):
        raise ProtocolLockError(
            "unexpected public independent-shift protocol identity"
        )
    if (
        protocol["lock_status"] != "finalized_locked_design"
        or protocol["final_results_inspected"] is not False
    ):
        raise ProtocolLockError(
            "independent-shift protocol is not a pre-outcome finalized lock"
        )

    imm = _mapping(protocol["immutability"], "immutability")
    if imm.get("status") != "locked":
        raise ProtocolLockError("immutability status must be locked")

    analysis = _mapping(protocol["analysis_lock"], "analysis_lock")
    primary = _mapping(analysis["primary_metric"], "primary_metric")
    if (
        primary.get("metric_name") != "normalized_return_auc"
        or primary.get("window") != "post_shift_inclusive"
    ):
        raise ProtocolLockError("primary metric lock mismatch")
    _validate_contrast(
        analysis["primary_contrast"], PRIMARY_CONTRAST, "primary_contrast"
    )
    _validate_contrast(
        analysis["co_primary_contrast"],
        CO_PRIMARY_CONTRAST,
        "co_primary_contrast",
    )
    if analysis.get("secondary_inference_policy", {}).get("p_values") != "prohibited":
        raise ProtocolLockError(
            "secondary diagnostics must remain non-inferential"
        )

    mechanisms = protocol["mechanisms"]
    if (
        not isinstance(mechanisms, list)
        or {m.get("mechanism_id") for m in mechanisms} != EXPECTED_MECHANISMS
    ):
        raise ProtocolLockError("mechanism registry mismatch")

    reserved = _mapping(seed_registry.get("reserved_ranges"), "reserved_ranges")
    expected_ranges = {
        "new_shift_development": (11000, 11099),
        "new_shift_final": (12000, 12099),
    }
    for key, (expected_start, expected_end) in expected_ranges.items():
        reservation = _mapping(reserved.get(key), key)
        observed = (int(reservation.get("start")), int(reservation.get("end")))
        if observed != (expected_start, expected_end):
            raise ProtocolLockError(f"seed reservation mismatch: {key}")

    development_intervals: list[tuple[int, int, str]] = []
    final_intervals: list[tuple[int, int, str]] = []
    expected_final = {
        "transition_dynamics_shift": (
            12000,
            12029,
            "transition_slip_035",
            "TransitionDynamicsShift-v0",
        ),
        "observation_shift": (
            12030,
            12059,
            "observation_gain_085",
            "ObservationShift-v0",
        ),
        "localized_multistep_reward_or_policy_shift": (
            12060,
            12089,
            "localized_risk_penalty_064",
            "LocalizedRewardShift-v0",
        ),
    }

    for mechanism in mechanisms:
        mechanism_id = str(mechanism.get("mechanism_id"))
        _require_fields(
            mechanism,
            REQUIRED_MECHANISM_FIELDS
            | {"selected_final_severity", "final_environment_id"},
            mechanism_id,
        )
        if (
            mechanism["primary_metric"] != "normalized_return_auc"
            or int(mechanism["shift_onset"]["training_interaction"]) != 12000
        ):
            raise ProtocolLockError("mechanism metric/onset mismatch")
        _validate_contrast(
            mechanism["primary_contrast"],
            PRIMARY_CONTRAST,
            f"{mechanism_id}.primary_contrast",
        )
        _validate_contrast(
            mechanism["co_primary_contrast"],
            CO_PRIMARY_CONTRAST,
            f"{mechanism_id}.co_primary_contrast",
        )

        dev_start, dev_end, dev_key = _seed_interval(
            mechanism["development_seeds"],
            f"{mechanism_id}.development_seeds",
        )
        if dev_key != "new_shift_development" or not (
            11000 <= dev_start <= dev_end <= 11099
        ):
            raise ProtocolLockError("development seed reservation mismatch")
        development_intervals.append((dev_start, dev_end, mechanism_id))

        final_start, final_end, final_key = _seed_interval(
            mechanism["final_seeds"], f"{mechanism_id}.final_seeds"
        )
        if final_key != "new_shift_final":
            raise ProtocolLockError("final seed reservation mismatch")
        expected_start, expected_end, severity, environment_id = expected_final[
            mechanism_id
        ]
        if (final_start, final_end) != (expected_start, expected_end):
            raise ProtocolLockError("final seed block mismatch")
        final_intervals.append((final_start, final_end, mechanism_id))

        if (
            mechanism["selected_final_severity"] != severity
            or mechanism["final_environment_id"] != environment_id
        ):
            raise ProtocolLockError(
                "final severity/environment lock mismatch"
            )
        if (
            int(mechanism["training_interaction_budget"]["per_agent_seed"])
            != 24000
            or int(mechanism["evaluation_episodes"]) != 200
        ):
            raise ProtocolLockError("budget mismatch")
        if mechanism["report_level_holm_family"] != "independent_shift_primary":
            raise ProtocolLockError("Holm family label mismatch")

    for intervals, label in (
        (development_intervals, "development seed overlap"),
        (final_intervals, "final seed overlap"),
    ):
        for index, left in enumerate(intervals):
            for right in intervals[index + 1 :]:
                if _overlap((left[0], left[1]), (right[0], right[1])):
                    raise ProtocolLockError(
                        f"{label}: {left[2]} and {right[2]}"
                    )

    families = protocol["report_level_holm_families"]
    if not isinstance(families, list) or len(families) != 1:
        raise ProtocolLockError("invalid family definition")
    family = _mapping(families[0], "report_level_holm_families[0]")
    required_family = {
        "family_id",
        "scope",
        "correction",
        "evidence_class",
        "expected_member_count",
        "members",
    }
    _require_fields(family, required_family, "report_level_holm_families[0]")
    if (
        family["family_id"] != "independent_shift_primary"
        or family["scope"] != "report_level"
        or family["correction"] != "holm"
        or family["evidence_class"] != "replication"
        or int(family["expected_member_count"]) != 6
    ):
        raise ProtocolLockError("invalid family definition")

    members = family["members"]
    if not isinstance(members, list) or len(members) != 6:
        raise ProtocolLockError("duplicate or malformed family members")
    observed_members = []
    for member in members:
        mapped = _mapping(member, "Holm family member")
        _require_fields(
            mapped, {"mechanism_id", "contrast_id", "metric"}, "Holm family member"
        )
        observed_members.append(
            (mapped["mechanism_id"], mapped["contrast_id"], mapped["metric"])
        )
    if len(set(observed_members)) != len(observed_members):
        raise ProtocolLockError("duplicate or malformed family members")
    expected_members = {
        (mechanism_id, contrast["contrast_id"], "normalized_return_auc")
        for mechanism_id in EXPECTED_MECHANISMS
        for contrast in (PRIMARY_CONTRAST, CO_PRIMARY_CONTRAST)
    }
    if set(observed_members) != expected_members:
        raise ProtocolLockError("duplicate or malformed family members")

def load_and_validate(protocol_path: Path, seed_registry_path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    seed_registry = yaml.safe_load(seed_registry_path.read_text(encoding="utf-8"))
    validate_protocol(
        _mapping(protocol, "protocol"), _mapping(seed_registry, "seed registry")
    )
    return protocol


def composite_digest(protocol_path: Path, companion_path: Path | None = None) -> str:
    """Backward-compatible name: the public lock digest is the exact YAML SHA-256."""
    return hashlib.sha256(protocol_path.read_bytes()).hexdigest()


def verify_or_write_digest(protocol_path: Path, companion_path: Path | None, digest_path: Path) -> str:
    current = composite_digest(protocol_path, companion_path)
    line = f"{current}  {protocol_path.name}\n"
    if digest_path.exists():
        recorded = digest_path.read_text(encoding="utf-8")
        if recorded != line:
            raise ProtocolLockError("public protocol digest mismatch")
        return "verified"
    digest_path.write_text(line, encoding="utf-8")
    return "written"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and immutably digest the independent-shift protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--companion", type=Path, default=DEFAULT_COMPANION)
    parser.add_argument("--digest", type=Path, default=DEFAULT_DIGEST)
    parser.add_argument("--seed-registry", type=Path, default=DEFAULT_SEED_REGISTRY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    load_and_validate(args.protocol, args.seed_registry)
    if args.validate_only:
        print("SHIFT_PROTOCOL_VALIDATION_PASS")
        return
    state = verify_or_write_digest(args.protocol, args.companion, args.digest)
    print(f"SHIFT_PROTOCOL_LOCK_PASS digest_status={state}")


if __name__ == "__main__":
    main()
