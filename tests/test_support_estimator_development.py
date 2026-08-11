from __future__ import annotations

import copy

import pandas as pd
import pytest

from scripts.aggregate_support_estimator_development import rank_candidates
from scripts.run_support_estimator_development import (
    DEFAULT_CONFIG,
    DEFAULT_REGISTRY,
    DEFAULT_SEED_REGISTRY,
    SupportDevelopmentError,
    flatten_candidates,
    load_yaml,
    validate_config,
    validate_registry,
    verify_registry_digest,
)


def _locked_inputs():
    registry = load_yaml(DEFAULT_REGISTRY)
    seed_registry = load_yaml(DEFAULT_SEED_REGISTRY)
    config = load_yaml(DEFAULT_CONFIG)
    return registry, seed_registry, config


def test_candidate_counts_are_equal_and_digest_is_valid() -> None:
    registry, seed_registry, _ = _locked_inputs()
    validate_registry(registry, seed_registry)
    counts = {
        family["family_id"]: len(family["candidates"])
        for family in registry["families"]
    }
    assert set(counts.values()) == {4}
    assert len(flatten_candidates(registry)) == 20
    assert len(verify_registry_digest()) == 64


def test_candidate_budgets_are_identical() -> None:
    registry, _, config = _locked_inputs()
    validate_config(config, registry)
    assert {env["training_steps"] for env in config["envs"]} == {16000}
    assert {tuple(env["seeds"]) for env in config["envs"]} == {
        (13000, 13001, 13002, 13003, 13004)
    }
    assert config["evaluation"] == {"interval_steps": 1000, "episodes": 10}
    assert config["runtime"]["torch_threads"] == 1
    assert config["runtime"]["torch_interop_threads"] == 1


def test_final_seed_in_registry_fails_closed() -> None:
    registry, seed_registry, _ = _locked_inputs()
    modified = copy.deepcopy(registry)
    modified["seed_lock"]["development_seeds"][0] = 14000
    for environment in modified["environments"]:
        environment["development_seeds"][0] = 14000
    with pytest.raises(SupportDevelopmentError, match="outside registered range"):
        validate_registry(modified, seed_registry)


def test_unequal_candidate_count_fails_closed() -> None:
    registry, seed_registry, _ = _locked_inputs()
    modified = copy.deepcopy(registry)
    modified["families"][0]["candidates"].pop()
    with pytest.raises(SupportDevelopmentError, match="counts differ"):
        validate_registry(modified, seed_registry)


def test_selection_is_deterministic_through_all_tie_breakers() -> None:
    rows = []
    specifications = {
        "candidate_a": (0.20, 0.10),
        "candidate_b": (0.10, 0.20),
        "candidate_c": (0.10, 0.20),
    }
    for candidate_id, (failure, latency) in specifications.items():
        for environment in ("application", "transition"):
            for seed in (13000, 13001):
                rows.append(
                    {
                        "estimator_family": "family",
                        "candidate_id": candidate_id,
                        "environment": environment,
                        "seed": seed,
                        "return_auc": 1.0,
                        "failure_probability": failure,
                        "median_query_latency_seconds": latency,
                    }
                )
    frame = pd.DataFrame(rows)
    first, family_first, ranks_first = rank_candidates(frame)
    second, family_second, ranks_second = rank_candidates(
        frame.sample(frac=1.0, random_state=42)
    )
    assert first == second == "candidate_b"
    assert family_first == family_second == {"family": "candidate_b"}
    pd.testing.assert_frame_equal(
        ranks_first.reset_index(drop=True),
        ranks_second.reset_index(drop=True),
    )
