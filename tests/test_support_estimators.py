from __future__ import annotations

import numpy as np
import pytest
import torch

import hybrid_q.support_estimators as support_module
from hybrid_q.agents import (
    AgentConfig,
    DuelingQNetwork,
    HybridQAgent,
    QNetwork,
    create_agent,
)
from hybrid_q.support_estimators import (
    ExactCountSupportEstimator,
    FrozenEmbeddingSupportEstimator,
    GaussianAffinitySupportEstimator,
    MahalanobisSupportEstimator,
    RegularizedGaussianDensitySupportEstimator,
    ToleranceKNNSupportEstimator,
    ZScoreKNNSupportEstimator,
    cKDTree,
    create_support_estimator,
)


STATES = (
    np.asarray([0.0, 0.0], dtype=np.float32),
    np.asarray([0.2, 0.1], dtype=np.float32),
    np.asarray([1.0, 1.0], dtype=np.float32),
)


def _observe_three(estimator) -> None:
    for index, state in enumerate(STATES):
        estimator.observe(state, f"s{index}")


def test_exact_count_tracks_key_visits_without_state_aliasing() -> None:
    estimator = ExactCountSupportEstimator(support_tau=2.0)
    state = np.asarray([1.0, 2.0], dtype=np.float32)
    estimator.observe(state, "same")
    state[:] = -100.0
    estimator.observe(np.asarray([3.0, 4.0]), "same")
    result = estimator.query(np.asarray([9.0, 9.0]), key="same")
    assert result.effective_sample_mass == 2.0
    assert result.support_score == 0.5
    np.testing.assert_array_equal(
        estimator.raw_states["same"], np.asarray([1.0, 2.0])
    )


def test_tolerance_knn_is_exactly_the_executed_formula() -> None:
    estimator = ToleranceKNNSupportEstimator(
        k=2, bandwidth=0.25, support_tau=2.0
    )
    _observe_three(estimator)
    query = np.asarray([0.1, 0.0], dtype=np.float32)
    result = estimator.query(query)

    matrix = np.stack(STATES)
    distances = np.linalg.norm(matrix - query[None, :], axis=1)
    nearest = np.argpartition(distances, 1)[:2]
    expected = (distances[nearest] <= 0.25).astype(np.float64)
    assert result.neighbor_keys == tuple(f"s{i}" for i in nearest)
    np.testing.assert_array_equal(result.weights, expected)
    assert result.effective_sample_mass == expected.sum()


def test_gaussian_affinity_is_exactly_the_executed_formula() -> None:
    estimator = GaussianAffinitySupportEstimator(
        k=2, bandwidth=0.25, support_tau=2.0
    )
    _observe_three(estimator)
    query = np.asarray([0.1, 0.0], dtype=np.float32)
    result = estimator.query(query)

    matrix = np.stack(STATES)
    distances = np.linalg.norm(matrix - query[None, :], axis=1)
    nearest = np.argpartition(distances, 1)[:2]
    expected = np.exp(-0.5 * (distances[nearest] / 0.25) ** 2)
    assert result.neighbor_keys == tuple(f"s{i}" for i in nearest)
    np.testing.assert_array_equal(result.weights, expected)


def test_agent_legacy_gaussian_path_matches_historical_calculation() -> None:
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=3,
        config=AgentConfig(
            replay_warmup=100,
            approximate_support_k=2,
            approximate_support_bandwidth=0.25,
            approximate_support_tau=2.0,
        ),
        gate_kind="feature_distance_support",
    )
    for index, state in enumerate(STATES):
        agent.state_vectors[f"s{index}"] = state.copy()
        agent.table[f"s{index}"][:] = np.asarray([index, -index])
    query = np.asarray([0.1, 0.0], dtype=np.float32)
    mass, values = agent._approximate_support_and_values(
        query, "feature_distance_support"
    )

    distances = np.linalg.norm(np.stack(STATES) - query[None, :], axis=1)
    nearest = np.argpartition(distances, 1)[:2]
    weights = np.exp(-0.5 * (distances[nearest] / 0.25) ** 2)
    expected = sum(
        float(weight) * agent.table[f"s{int(index)}"]
        for weight, index in zip(weights, nearest)
    ) / weights.sum()
    assert mass == float(weights.sum())
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=0.0)


def test_approximate_support_records_branches_without_changing_values() -> None:
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=3,
        config=AgentConfig(
            replay_warmup=100,
            approximate_support_k=2,
            approximate_support_bandwidth=0.25,
            approximate_support_tau=2.0,
        ),
        gate_kind="feature_distance_support",
    )
    for index, state in enumerate(STATES):
        agent.state_vectors[f"s{index}"] = state.copy()
        agent.table[f"s{index}"][:] = np.asarray([index, -index])
    agent.neural_q_values = lambda state: np.asarray([-2.0, 2.0])
    query = np.asarray([0.1, 0.0], dtype=np.float32)

    gate, memory_values, _ = agent.approximate_gate(
        query, "feature_distance_support"
    )
    actual = agent.q_values(query, "unseen")
    expected = gate * memory_values + (1.0 - gate) * np.asarray([-2.0, 2.0])

    np.testing.assert_allclose(actual, expected)
    diagnostics = agent.decision_diagnostics()
    assert diagnostics["tabular_greedy_action"] == int(np.argmax(memory_values))
    assert diagnostics["neural_greedy_action"] == 1


def test_zscore_knn_scales_features_and_handles_constant_columns() -> None:
    estimator = ZScoreKNNSupportEstimator(k=2, bandwidth=1.1)
    estimator.observe(np.asarray([0.0, 7.0]), "low")
    estimator.observe(np.asarray([10.0, 7.0]), "high")
    result = estimator.query(np.asarray([1.0, 7.0]))
    weights = dict(zip(result.neighbor_keys, result.weights))
    assert weights["low"] == 1.0
    assert weights["high"] == 0.0
    assert np.isfinite(result.nearest_distance)


def test_regularized_mahalanobis_handles_singular_covariance() -> None:
    estimator = MahalanobisSupportEstimator(
        k=3, bandwidth=2.0, covariance_regularization=1e-4
    )
    for index in range(3):
        estimator.observe(np.asarray([index, index], dtype=np.float32), index)
    result = estimator.query(np.asarray([0.5, 0.5], dtype=np.float32))
    assert np.isfinite(result.nearest_distance)
    assert np.all(np.isfinite(result.weights))
    assert result.effective_sample_mass > 0.0


def test_covariance_regularization_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MahalanobisSupportEstimator(covariance_regularization=0.0)


def test_regularized_gaussian_density_has_explicit_finite_log_score() -> None:
    estimator = RegularizedGaussianDensitySupportEstimator(
        k=1,
        bandwidth=0.5,
        covariance_regularization=1e-3,
    )
    estimator.observe(np.asarray([1.0, 1.0]), "a")
    estimator.observe(np.asarray([1.0, 1.0]), "b")
    result = estimator.query(np.asarray([1.0, 1.0]))
    assert result.neighbor_count == 2
    assert np.isfinite(result.density_log_score)
    assert result.effective_sample_mass == pytest.approx(2.0)


def test_frozen_embedding_is_immutable_and_traceable() -> None:
    torch.manual_seed(7)
    live_network = QNetwork(2, 2, 4)
    estimator = create_support_estimator(
        "gaussian_affinity",
        k=1,
        bandwidth=1.0,
        representation_type="frozen_dqn_penultimate",
    )
    assert isinstance(estimator, FrozenEmbeddingSupportEstimator)
    estimator.observe(np.asarray([0.0, 1.0], dtype=np.float32), "seen")
    assert estimator.query(np.asarray([0.0, 1.0])).neighbor_count == 0

    digest = estimator.freeze(live_network, freeze_step=25)
    before = estimator.query(np.asarray([0.2, 0.8], dtype=np.float32))
    with torch.no_grad():
        for parameter in live_network.parameters():
            parameter.add_(100.0)
    after = estimator.query(np.asarray([0.2, 0.8], dtype=np.float32))

    np.testing.assert_array_equal(before.weights, after.weights)
    diagnostics = estimator.diagnostics()
    assert diagnostics["embedding_snapshot_hash"] == digest
    assert len(digest) == 64
    assert diagnostics["embedding_freeze_step"] == 25
    assert diagnostics["embedding_frozen"] is True


@pytest.mark.skipif(cKDTree is None, reason="scipy cKDTree unavailable")
def test_ckdtree_and_brute_force_are_consistent() -> None:
    brute = GaussianAffinitySupportEstimator(
        k=3, bandwidth=0.7, index_type="brute_force"
    )
    indexed = GaussianAffinitySupportEstimator(
        k=3, bandwidth=0.7, index_type="ckdtree"
    )
    rng = np.random.default_rng(42)
    for index, state in enumerate(rng.normal(size=(12, 4)).astype(np.float32)):
        brute.observe(state, index)
        indexed.observe(state, index)
    query = rng.normal(size=4).astype(np.float32)
    brute_result = brute.query(query)
    indexed_result = indexed.query(query)
    brute_pairs = sorted(zip(brute_result.neighbor_keys, brute_result.weights))
    indexed_pairs = sorted(zip(indexed_result.neighbor_keys, indexed_result.weights))
    assert [key for key, _ in brute_pairs] == [key for key, _ in indexed_pairs]
    np.testing.assert_allclose(
        [weight for _, weight in brute_pairs],
        [weight for _, weight in indexed_pairs],
        rtol=1e-6,
    )


def test_ckdtree_request_has_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(support_module, "cKDTree", None)
    estimator = GaussianAffinitySupportEstimator(index_type="ckdtree")
    assert estimator.index_type == "brute_force"


def test_estimator_state_round_trip_and_config_loading() -> None:
    original = create_support_estimator(
        "zscore_knn", k=2, bandwidth=1.5, scale_epsilon=1e-6
    )
    _observe_three(original)
    restored = create_support_estimator(
        "zscore_knn", k=2, bandwidth=1.5, scale_epsilon=1e-6
    )
    restored.load_state_dict(original.state_dict())
    query = np.asarray([0.3, 0.2], dtype=np.float32)
    first = original.query(query)
    second = restored.query(query)
    assert first.neighbor_keys == second.neighbor_keys
    np.testing.assert_array_equal(first.weights, second.weights)

    agent = create_agent(
        "support_estimator_gate",
        input_dim=2,
        action_dim=2,
        seed=0,
        params={
            "support_estimator_type": "regularized_mahalanobis",
            "support_covariance_regularization": 1e-2,
            "support_index_type": "brute_force",
            "replay_warmup": 100,
        },
    )
    assert isinstance(agent, HybridQAgent)
    assert agent.support_estimator.estimator_type == "regularized_mahalanobis"


def test_feature_extraction_preserves_qnetwork_forward_values() -> None:
    torch.manual_seed(11)
    network = QNetwork(3, 2, 5)
    states = torch.randn(4, 3)
    expected = network.layers(states)
    actual = network(states)
    np.testing.assert_array_equal(
        actual.detach().numpy(), expected.detach().numpy()
    )
    assert network.extract_features(states).shape == (4, 5)

    dueling = DuelingQNetwork(3, 2, 5)
    features = dueling.extract_features(states)
    expected_dueling = (
        dueling.value(features)
        + dueling.advantage(features)
        - dueling.advantage(features).mean(dim=1, keepdim=True)
    )
    np.testing.assert_array_equal(
        dueling(states).detach().numpy(),
        expected_dueling.detach().numpy(),
    )


def test_agent_freezes_embedding_at_declared_environment_step() -> None:
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=5,
        config=AgentConfig(
            replay_warmup=100,
            support_estimator_type="gaussian_affinity",
            support_representation_type="frozen_dqn_penultimate",
            support_embedding_freeze_step=1,
        ),
        gate_kind="support_estimator",
    )
    state = np.asarray([0.0, 1.0], dtype=np.float32)
    agent.observe(state, "s", 0, 1.0, state, "s", True)
    diagnostics = agent.support_estimator.diagnostics()
    assert diagnostics["embedding_frozen"] is True
    assert diagnostics["embedding_freeze_step"] == 1
    assert len(diagnostics["embedding_snapshot_hash"]) == 64


def test_diagnostics_expose_required_fields() -> None:
    estimator = GaussianAffinitySupportEstimator(k=1, bandwidth=1.0)
    estimator.observe(np.asarray([0.0]), "zero")
    estimator.query(np.asarray([0.1]))
    required = {
        "support_score",
        "neighbor_count",
        "nearest_distance",
        "effective_sample_mass",
        "density_log_score",
        "estimator_type",
        "representation_type",
        "index_type",
        "query_latency",
        "memory_bytes",
    }
    assert required <= estimator.diagnostics().keys()
