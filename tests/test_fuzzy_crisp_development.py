from __future__ import annotations

from copy import deepcopy
import gzip
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_q.mappings import (  # noqa: E402
    crisp_reliability_gate,
    fuzzy_reliability_gate,
)
from hybrid_q.envs import make_env  # noqa: E402
from scripts.aggregate_fuzzy_crisp_development import (  # noqa: E402
    load_or_measure_latency,
    select_candidates,
)
from scripts.lock_protocol import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_SEED_REGISTRY,
    load_and_validate,
)
from scripts.run_fuzzy_crisp_development import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REGISTRY,
    FuzzyCrispDevelopmentError,
    _load_yaml,
    deterministic_gzip_copy,
    validate_config,
    validate_registry,
)


def test_candidate_counts_and_declared_budgets_are_equal():
    registry = _load_yaml(DEFAULT_REGISTRY)
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    validate_registry(registry, protocol)
    config = _load_yaml(DEFAULT_CONFIG)
    validate_config(config, registry)
    assert len(registry["crisp_candidates"]) == len(registry["fuzzy_candidates"]) == 4
    assert {item["membership_family"] for item in registry["fuzzy_candidates"]} == {
        "triangular",
        "shoulder",
    }
    assert all(
        agent["params"]["fuzzy_fallback_risk_mode"] == "disabled"
        for agent in config["agents"]
    )


def test_registry_and_config_fail_closed_on_final_seed_use():
    registry = _load_yaml(DEFAULT_REGISTRY)
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    corrupted_registry = deepcopy(registry)
    corrupted_registry["environments"][0]["development_seeds"][0] = 12000
    with pytest.raises(
        FuzzyCrispDevelopmentError, match="development seed mismatch|final"
    ):
        validate_registry(corrupted_registry, protocol)

    config = _load_yaml(DEFAULT_CONFIG)
    corrupted_config = deepcopy(config)
    corrupted_config["envs"][0]["seeds"][0] = 12000
    with pytest.raises(FuzzyCrispDevelopmentError, match="seed set"):
        validate_config(corrupted_config, registry)


def test_all_declared_development_environments_construct():
    config = _load_yaml(DEFAULT_CONFIG)
    for env_spec in config["envs"]:
        env = make_env(env_spec)
        try:
            observation, info = env.reset(seed=int(env_spec["seeds"][0]))
            assert observation is not None
            assert info["post_shift"] is False
        finally:
            env.close()


def test_selection_is_deterministic_and_uses_lexical_last_tie_break():
    results = pd.DataFrame(
        [
            {
                "mapping_family": family,
                "candidate_id": candidate,
                "environment": env,
                "primary_metric": 1.0,
            }
            for family in ("crisp", "fuzzy")
            for candidate in (f"{family}_a", f"{family}_b")
            for env in ("env_a", "env_b")
        ]
    )
    latency = pd.DataFrame(
        [
            {
                "mapping_family": family,
                "candidate_id": candidate,
                "median_latency_ns": 10.0,
            }
            for family in ("crisp", "fuzzy")
            for candidate in (f"{family}_a", f"{family}_b")
        ]
    )
    first, ranks_first = select_candidates(results, latency)
    second, ranks_second = select_candidates(
        results.sample(frac=1.0, random_state=8), latency
    )
    assert first == second == {"crisp": "crisp_a", "fuzzy": "fuzzy_a"}
    pd.testing.assert_frame_equal(ranks_first, ranks_second)


def test_recorded_latency_is_reused_and_validated(tmp_path):
    registry = _load_yaml(DEFAULT_REGISTRY)
    source = ROOT / "results/diagnostic_extensions/fuzzy_crisp_development/latency_complexity.csv"
    recorded = pd.read_csv(source, float_precision="round_trip")
    recorded.to_csv(tmp_path / "latency_complexity.csv", index=False)

    reused = load_or_measure_latency(registry, tmp_path)
    pd.testing.assert_frame_equal(reused, recorded)

    corrupted = recorded.iloc[:-1]
    corrupted.to_csv(tmp_path / "latency_complexity.csv", index=False)
    with pytest.raises(FuzzyCrispDevelopmentError, match="exactly once"):
        load_or_measure_latency(registry, tmp_path)


def test_mapping_thresholds_and_declared_breakpoints_are_effective():
    assert (
        crisp_reliability_gate(
            0.29,
            0.8,
            support_threshold=0.3,
            reliability_threshold=0.7,
            gate_min=0.05,
            gate_max=0.95,
        )
        == 0.05
    )
    assert (
        crisp_reliability_gate(
            0.3,
            0.7,
            support_threshold=0.3,
            reliability_threshold=0.7,
            gate_min=0.05,
            gate_max=0.95,
        )
        == 0.95
    )
    low = fuzzy_reliability_gate(
        0.8,
        0.2,
        membership_shape="triangular",
        reliability_membership_shape="triangular",
        support_breakpoints=[0.0, 0.5, 1.0],
        reliability_breakpoints=[0.0, 0.5, 1.0],
        consequents=[0.0, 0.2, 0.7, 0.1, 0.95],
        gate_min=0.05,
        gate_max=0.95,
    )
    high = fuzzy_reliability_gate(
        0.8,
        0.8,
        membership_shape="triangular",
        reliability_membership_shape="triangular",
        support_breakpoints=[0.0, 0.5, 1.0],
        reliability_breakpoints=[0.0, 0.5, 1.0],
        consequents=[0.0, 0.2, 0.7, 0.1, 0.95],
        gate_min=0.05,
        gate_max=0.95,
    )
    assert high > low


def test_raw_gzip_copy_is_deterministic_and_non_destructive(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_bytes(b"seed,value\n11000,1.25\n")
    first = deterministic_gzip_copy(source).read_bytes()
    second = deterministic_gzip_copy(source).read_bytes()
    assert first == second
    assert source.read_bytes() == b"seed,value\n11000,1.25\n"
    with gzip.open(source.with_suffix(".csv.gz"), "rb") as handle:
        assert handle.read() == source.read_bytes()
