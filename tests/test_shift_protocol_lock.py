from copy import deepcopy

import pytest
import yaml

from scripts.lock_protocol import (
    DEFAULT_DIGEST,
    DEFAULT_PROTOCOL,
    DEFAULT_SEED_REGISTRY,
    ProtocolLockError,
    composite_digest,
    validate_protocol,
    verify_or_write_digest,
)


def _locked_inputs() -> tuple[dict, dict]:
    protocol = yaml.safe_load(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    registry = yaml.safe_load(DEFAULT_SEED_REGISTRY.read_text(encoding="utf-8"))
    return protocol, registry


def test_checked_in_protocol_validates_and_has_three_independent_mechanisms():
    protocol, registry = _locked_inputs()
    validate_protocol(protocol, registry)
    assert {item["mechanism_id"] for item in protocol["mechanisms"]} == {
        "transition_dynamics_shift",
        "observation_shift",
        "localized_multistep_reward_or_policy_shift",
    }
    assert protocol["final_results_inspected"] is False
    expected = DEFAULT_DIGEST.read_text(encoding="utf-8").split()[0]
    assert composite_digest(DEFAULT_PROTOCOL) == expected


def test_protocol_rejects_development_seed_overlap():
    protocol, registry = _locked_inputs()
    invalid = deepcopy(protocol)
    seeds = invalid["mechanisms"][1]["development_seeds"]
    seeds.update(start=11005, end=11014, count=10)
    with pytest.raises(ProtocolLockError, match="development seed overlap"):
        validate_protocol(invalid, registry)


def test_protocol_rejects_missing_mechanism_fields():
    protocol, registry = _locked_inputs()
    invalid = deepcopy(protocol)
    del invalid["mechanisms"][0]["stopping_rules"]
    with pytest.raises(ProtocolLockError, match="missing required fields"):
        validate_protocol(invalid, registry)


def test_protocol_rejects_invalid_holm_family_definition():
    protocol, registry = _locked_inputs()
    invalid = deepcopy(protocol)
    invalid["report_level_holm_families"][0]["correction"] = "none"
    with pytest.raises(ProtocolLockError, match="invalid family definition"):
        validate_protocol(invalid, registry)


def test_protocol_rejects_duplicate_holm_family_members():
    protocol, registry = _locked_inputs()
    invalid = deepcopy(protocol)
    members = invalid["report_level_holm_families"][0]["members"]
    members[-1] = deepcopy(members[0])
    with pytest.raises(
        ProtocolLockError, match="duplicate or malformed family members"
    ):
        validate_protocol(invalid, registry)


def test_digest_locks_yaml_and_refuses_silent_rewrite(tmp_path):
    protocol = tmp_path / DEFAULT_PROTOCOL.name
    digest = tmp_path / DEFAULT_DIGEST.name
    protocol.write_bytes(DEFAULT_PROTOCOL.read_bytes())

    assert verify_or_write_digest(protocol, None, digest) == "written"
    assert composite_digest(protocol) in digest.read_text(encoding="utf-8")
    assert verify_or_write_digest(protocol, None, digest) == "verified"

    protocol.write_text(
        protocol.read_text(encoding="utf-8") + "\n# silent change\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolLockError, match="public protocol digest mismatch"):
        verify_or_write_digest(protocol, None, digest)
