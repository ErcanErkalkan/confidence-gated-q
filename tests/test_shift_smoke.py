from copy import deepcopy

import pytest

from scripts.lock_protocol import (
    DEFAULT_PROTOCOL,
    DEFAULT_SEED_REGISTRY,
    load_and_validate,
)
from scripts.run_shift_smoke import (
    SMOKE_CONFIGS,
    ShiftSmokeError,
    _load_config,
    validate_smoke_config,
)


PROTOCOL = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)


def test_smoke_configs_use_two_registered_development_seeds_and_locked_onset():
    observed = set()
    for path in SMOKE_CONFIGS:
        config = _load_config(path)
        mechanism = validate_smoke_config(config, PROTOCOL)
        observed.add(mechanism["mechanism_id"])
        assert len(config["seeds"]) == 2
        assert not any(12000 <= int(seed) <= 12099 for seed in config["seeds"])
        assert config["envs"][0]["kwargs"]["shift_after"] == 12000
        assert config["envs"][0]["training_steps"] == 12020
    assert observed == {
        "transition_dynamics_shift",
        "observation_shift",
        "localized_multistep_reward_or_policy_shift",
    }


def test_smoke_validator_rejects_a_reserved_final_seed():
    config = deepcopy(_load_config(SMOKE_CONFIGS[0]))
    config["seeds"] = [12000, 12001]
    with pytest.raises(ShiftSmokeError, match="reserved final seed detected"):
        validate_smoke_config(config, PROTOCOL)
