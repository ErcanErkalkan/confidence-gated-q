"""Deterministic support estimators used by hybrid agents.

The legacy estimators deliberately retain their historical numerical path:
one float32 state per key, Euclidean distance, ``numpy.argpartition`` nearest
neighbours, and either tolerance or Gaussian-affinity weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Hashable, Mapping

import numpy as np
import torch
from torch import nn

try:  # Optional acceleration; scipy itself remains a project dependency.
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised through monkeypatching.
    cKDTree = None


@dataclass(frozen=True)
class SupportEstimate:
    """A support query plus the weights needed to retrieve stored values."""

    support_score: float
    neighbor_keys: tuple[Hashable, ...]
    weights: np.ndarray
    neighbor_count: int
    nearest_distance: float
    effective_sample_mass: float
    density_log_score: float
    nearest_key: Hashable | None = None


class SupportEstimator(ABC):
    """Common state-support interface.

    ``key`` is optional on queries because distance estimators do not need it;
    exact-count support does.  ``optional_embedding`` lets callers provide a
    declared, versioned representation without changing the raw-state record.
    """

    estimator_type = "abstract"
    representation_type = "raw_state"
    index_type = "none"

    def __init__(self, *, support_tau: float = 2.0) -> None:
        self.support_tau = max(float(support_tau), 1e-8)
        self._last_diagnostics = self._empty_diagnostics()

    @abstractmethod
    def observe(
        self,
        state: np.ndarray,
        key: Hashable,
        optional_embedding: np.ndarray | None = None,
    ) -> None:
        """Record a training observation."""

    @abstractmethod
    def query(
        self,
        state: np.ndarray,
        optional_embedding: np.ndarray | None = None,
        *,
        key: Hashable | None = None,
    ) -> SupportEstimate:
        """Query support without updating estimator state."""

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self._last_diagnostics)
        diagnostics["memory_bytes"] = int(self.memory_bytes())
        return diagnostics

    @abstractmethod
    def memory_bytes(self) -> int:
        """Return bytes in stored numeric arrays (Python overhead excluded)."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a deterministic Python-serializable estimator state."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

    def _empty_diagnostics(self) -> dict[str, Any]:
        return {
            "support_score": 0.0,
            "neighbor_count": 0,
            "nearest_distance": float("nan"),
            "nearest_key": None,
            "effective_sample_mass": 0.0,
            "density_log_score": float("nan"),
            "estimator_type": self.estimator_type,
            "representation_type": self.representation_type,
            "index_type": self.index_type,
            "query_latency": float("nan"),
            "memory_bytes": 0,
        }

    def _record(self, estimate: SupportEstimate, started_ns: int) -> None:
        self._last_diagnostics = {
            "support_score": float(estimate.support_score),
            "neighbor_count": int(estimate.neighbor_count),
            "nearest_distance": float(estimate.nearest_distance),
            "nearest_key": estimate.nearest_key,
            "effective_sample_mass": float(estimate.effective_sample_mass),
            "density_log_score": float(estimate.density_log_score),
            "estimator_type": self.estimator_type,
            "representation_type": self.representation_type,
            "index_type": self.index_type,
            "query_latency": (time.perf_counter_ns() - started_ns) / 1e9,
            "memory_bytes": int(self.memory_bytes()),
        }

    def _score(self, mass: float) -> float:
        return float(mass / (mass + self.support_tau))


def _empty_estimate() -> SupportEstimate:
    return SupportEstimate(
        support_score=0.0,
        neighbor_keys=(),
        weights=np.empty(0, dtype=np.float64),
        neighbor_count=0,
        nearest_distance=float("nan"),
        effective_sample_mass=0.0,
        density_log_score=float("nan"),
        nearest_key=None,
    )


class ExactCountSupportEstimator(SupportEstimator):
    estimator_type = "exact_count"

    def __init__(self, *, support_tau: float = 20.0) -> None:
        super().__init__(support_tau=support_tau)
        self.counts: dict[Hashable, int] = {}
        self.raw_states: dict[Hashable, np.ndarray] = {}

    def observe(self, state, key, optional_embedding=None) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        if key not in self.raw_states:
            self.raw_states[key] = np.asarray(state, dtype=np.float32).copy()

    def query(self, state, optional_embedding=None, *, key=None) -> SupportEstimate:
        started = time.perf_counter_ns()
        count = int(self.counts.get(key, 0)) if key is not None else 0
        estimate = (
            SupportEstimate(
                support_score=self._score(float(count)),
                neighbor_keys=(key,),
                weights=np.asarray([float(count)], dtype=np.float64),
                neighbor_count=1,
                nearest_distance=0.0,
                effective_sample_mass=float(count),
                density_log_score=float("nan"),
                nearest_key=key,
            )
            if count
            else _empty_estimate()
        )
        self._record(estimate, started)
        return estimate

    def memory_bytes(self) -> int:
        vector_bytes = sum(item.nbytes for item in self.raw_states.values())
        return int(vector_bytes + len(self.counts) * np.dtype(np.int64).itemsize)

    def state_dict(self) -> dict[str, Any]:
        return {
            "estimator_type": self.estimator_type,
            "support_tau": self.support_tau,
            "items": [
                (key, self.counts[key], self.raw_states[key].copy())
                for key in self.counts
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.counts.clear()
        self.raw_states.clear()
        for key, count, raw_state in state.get("items", []):
            self.counts[key] = int(count)
            self.raw_states[key] = np.asarray(raw_state, dtype=np.float32).copy()


class _VectorSupportEstimator(SupportEstimator):
    def __init__(
        self,
        *,
        k: int = 5,
        bandwidth: float = 0.25,
        support_tau: float = 2.0,
        index_type: str = "brute_force",
        representation_type: str = "raw_state",
    ) -> None:
        super().__init__(support_tau=support_tau)
        self.k = max(1, int(k))
        self.bandwidth = max(float(bandwidth), 1e-8)
        requested = str(index_type)
        if requested not in {"brute_force", "ckdtree", "auto"}:
            raise ValueError(f"Unknown support index type: {requested}")
        if requested == "ckdtree" and cKDTree is None:
            self.index_type = "brute_force"
        elif requested == "auto":
            self.index_type = "ckdtree" if cKDTree is not None else "brute_force"
        else:
            self.index_type = requested
        self.representation_type = representation_type
        self.state_vectors: dict[Hashable, np.ndarray] = {}
        self.raw_states: dict[Hashable, np.ndarray] = {}
        self._tree: Any | None = None
        self._tree_matrix: np.ndarray | None = None

    def _representation(self, state, optional_embedding) -> np.ndarray:
        value = optional_embedding if optional_embedding is not None else state
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def observe(self, state, key, optional_embedding=None) -> None:
        if key in self.state_vectors:
            return
        self.raw_states[key] = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        self.state_vectors[key] = self._representation(
            state, optional_embedding
        ).copy()
        self._tree = None
        self._tree_matrix = None

    def _matrix_and_keys(self) -> tuple[np.ndarray, list[Hashable]]:
        keys = list(self.state_vectors)
        return np.stack([self.state_vectors[key] for key in keys]), keys

    def _transform_matrix_and_query(
        self, matrix: np.ndarray, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return matrix, query

    def _distance_vector(self, matrix, query) -> np.ndarray:
        return np.linalg.norm(matrix - query[None, :], axis=1)

    def _nearest(self, matrix, query, distances, k) -> np.ndarray:
        if self.index_type == "brute_force":
            # This is the exact historical selection path.
            return np.argpartition(distances, k - 1)[:k]
        if self._tree is None or self._tree_matrix is not matrix:
            self._tree = cKDTree(matrix)
            self._tree_matrix = matrix
        tree_distances, tree_indices = self._tree.query(query, k=k)
        tree_distances = np.atleast_1d(tree_distances)
        tree_indices = np.atleast_1d(tree_indices).astype(int)
        radius = float(np.max(tree_distances))
        candidates = np.asarray(
            self._tree.query_ball_point(query, r=np.nextafter(radius, np.inf)),
            dtype=int,
        )
        order = np.lexsort((candidates, distances[candidates]))
        selected = candidates[order[:k]]
        if selected.size < k:  # Defensive fallback for unusual scipy builds.
            return tree_indices[np.argsort(tree_distances, kind="stable")][:k]
        return selected

    @abstractmethod
    def _weights_and_density(
        self, distances: np.ndarray, nearest: np.ndarray, dimension: int
    ) -> tuple[np.ndarray, float]:
        pass

    def query(self, state, optional_embedding=None, *, key=None) -> SupportEstimate:
        started = time.perf_counter_ns()
        if not self.state_vectors:
            estimate = _empty_estimate()
            self._record(estimate, started)
            return estimate
        matrix, keys = self._matrix_and_keys()
        query = self._representation(state, optional_embedding)
        if query.size != matrix.shape[1]:
            raise ValueError(
                f"Support query dimension {query.size} != stored dimension {matrix.shape[1]}"
            )
        transformed, transformed_query = self._transform_matrix_and_query(
            matrix, query
        )
        distances = self._distance_vector(transformed, transformed_query)
        k = max(1, min(self.k, len(keys)))
        nearest = self._nearest(transformed, transformed_query, distances, k)
        weights, density_log_score = self._weights_and_density(
            distances, nearest, transformed.shape[1]
        )
        mass = float(weights.sum())
        positive = int(np.count_nonzero(weights > 0.0))
        if mass <= 1e-12:
            nearest_index = int(np.argmin(distances))
            estimate = SupportEstimate(
                support_score=0.0,
                neighbor_keys=tuple(keys[int(index)] for index in nearest),
                weights=weights,
                neighbor_count=0,
                nearest_distance=float(np.min(distances)),
                effective_sample_mass=0.0,
                density_log_score=float(density_log_score),
                nearest_key=keys[nearest_index],
            )
        else:
            nearest_index = int(np.argmin(distances))
            estimate = SupportEstimate(
                support_score=self._score(mass),
                neighbor_keys=tuple(keys[int(index)] for index in nearest),
                weights=weights,
                neighbor_count=positive,
                nearest_distance=float(np.min(distances)),
                effective_sample_mass=mass,
                density_log_score=float(density_log_score),
                nearest_key=keys[nearest_index],
            )
        self._record(estimate, started)
        return estimate

    def memory_bytes(self) -> int:
        representations = sum(item.nbytes for item in self.state_vectors.values())
        raw = sum(item.nbytes for item in self.raw_states.values())
        return int(representations + (0 if self.representation_type == "raw_state" else raw))

    def state_dict(self) -> dict[str, Any]:
        return {
            "estimator_type": self.estimator_type,
            "k": self.k,
            "bandwidth": self.bandwidth,
            "support_tau": self.support_tau,
            "index_type": self.index_type,
            "representation_type": self.representation_type,
            "items": [
                (key, self.raw_states[key].copy(), vector.copy())
                for key, vector in self.state_vectors.items()
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.state_vectors.clear()
        self.raw_states.clear()
        for key, raw, vector in state.get("items", []):
            self.raw_states[key] = np.asarray(raw, dtype=np.float32).copy()
            self.state_vectors[key] = np.asarray(vector, dtype=np.float32).copy()
        self._tree = None
        self._tree_matrix = None


class ToleranceKNNSupportEstimator(_VectorSupportEstimator):
    estimator_type = "tolerance_knn"

    def _weights_and_density(self, distances, nearest, dimension):
        return (
            (distances[nearest] <= self.bandwidth).astype(np.float64),
            float("nan"),
        )


class GaussianAffinitySupportEstimator(_VectorSupportEstimator):
    estimator_type = "gaussian_affinity"

    def _weights_and_density(self, distances, nearest, dimension):
        weights = np.exp(-0.5 * (distances[nearest] / self.bandwidth) ** 2)
        return weights, float("nan")


class ZScoreKNNSupportEstimator(ToleranceKNNSupportEstimator):
    estimator_type = "zscore_knn"
    representation_type = "zscore_normalized_raw_state"

    def __init__(self, *, scale_epsilon: float = 1e-8, **kwargs) -> None:
        kwargs["representation_type"] = self.representation_type
        super().__init__(**kwargs)
        self.scale_epsilon = max(float(scale_epsilon), 0.0)

    def _transform_matrix_and_query(self, matrix, query):
        mean = matrix.mean(axis=0, dtype=np.float64)
        scale = matrix.std(axis=0, dtype=np.float64)
        scale = np.where(scale > self.scale_epsilon, scale, 1.0)
        return (matrix - mean) / scale, (query - mean) / scale


class MahalanobisSupportEstimator(GaussianAffinitySupportEstimator):
    estimator_type = "regularized_mahalanobis"

    def __init__(self, *, covariance_regularization: float = 1e-3, **kwargs) -> None:
        # cKDTree is Euclidean; using it before Mahalanobis whitening would
        # select the wrong neighbours.  The deterministic exact fallback is
        # therefore mandatory for this metric.
        kwargs["index_type"] = "brute_force"
        super().__init__(**kwargs)
        if covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive")
        self.covariance_regularization = float(covariance_regularization)

    def _distance_vector(self, matrix, query):
        dimension = matrix.shape[1]
        if matrix.shape[0] < 2:
            covariance = np.zeros((dimension, dimension), dtype=np.float64)
        else:
            covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
        covariance = covariance + self.covariance_regularization * np.eye(dimension)
        inverse = np.linalg.pinv(covariance, hermitian=True)
        delta = matrix - query[None, :]
        squared = np.einsum("ni,ij,nj->n", delta, inverse, delta)
        return np.sqrt(np.maximum(squared, 0.0))


class RegularizedGaussianDensitySupportEstimator(_VectorSupportEstimator):
    """Gaussian-kernel density support with a regularized global covariance."""

    estimator_type = "regularized_gaussian_density"

    def __init__(self, *, covariance_regularization: float = 1e-3, **kwargs) -> None:
        kwargs["index_type"] = "brute_force"
        super().__init__(**kwargs)
        if covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive")
        self.covariance_regularization = float(covariance_regularization)
        self._query_covariance_logdet = 0.0

    def _distance_vector(self, matrix, query):
        dimension = matrix.shape[1]
        covariance = (
            np.zeros((dimension, dimension), dtype=np.float64)
            if matrix.shape[0] < 2
            else np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
        )
        covariance += self.covariance_regularization * np.eye(dimension)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Regularized covariance is not positive definite")
        self._query_covariance_logdet = float(logdet)
        inverse = np.linalg.pinv(covariance, hermitian=True)
        delta = matrix - query[None, :]
        squared = np.einsum("ni,ij,nj->n", delta, inverse, delta)
        return np.sqrt(np.maximum(squared, 0.0))

    def _weights_and_density(self, distances, nearest, dimension):
        # query() sets k to the ledger size, so every sample is represented.
        log_weights = -0.5 * (distances / self.bandwidth) ** 2
        maximum = float(np.max(log_weights))
        log_kernel_sum = maximum + math.log(float(np.exp(log_weights - maximum).sum()))
        normalization = (
            math.log(distances.size)
            + dimension * math.log(self.bandwidth)
            + 0.5 * dimension * math.log(2.0 * math.pi)
            + 0.5 * self._query_covariance_logdet
        )
        return np.exp(log_weights), log_kernel_sum - normalization

    def query(self, state, optional_embedding=None, *, key=None):
        # Temporarily include all points rather than truncating a density to k.
        original_k = self.k
        self.k = max(1, len(self.state_vectors))
        try:
            return super().query(state, optional_embedding, key=key)
        finally:
            self.k = original_k


def feature_snapshot_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenEmbeddingSupportEstimator(SupportEstimator):
    """A raw-state ledger plus an immutable feature-extractor snapshot."""

    estimator_type = "frozen_embedding"
    representation_type = "frozen_dqn_penultimate"

    def __init__(self, delegate: _VectorSupportEstimator) -> None:
        super().__init__(support_tau=delegate.support_tau)
        self.delegate = delegate
        self.index_type = delegate.index_type
        self.raw_observations: list[tuple[np.ndarray, Hashable]] = []
        self.feature_extractor: nn.Module | None = None
        self.embedding_snapshot_hash: str | None = None
        self.embedding_freeze_step: int | None = None

    @property
    def is_frozen(self) -> bool:
        return self.feature_extractor is not None

    def _embed(self, state: np.ndarray) -> np.ndarray:
        if self.feature_extractor is None:
            raise RuntimeError("Embedding feature extractor is not frozen")
        tensor = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            features = self.feature_extractor.extract_features(tensor)
        return features.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def freeze(self, feature_extractor: nn.Module, freeze_step: int) -> str:
        if self.is_frozen:
            if int(freeze_step) != self.embedding_freeze_step:
                raise RuntimeError("Embedding snapshot is already frozen")
            return str(self.embedding_snapshot_hash)
        snapshot = copy.deepcopy(feature_extractor).cpu().eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        self.feature_extractor = snapshot
        self.embedding_freeze_step = int(freeze_step)
        self.embedding_snapshot_hash = feature_snapshot_sha256(snapshot)
        for raw_state, key in self.raw_observations:
            self.delegate.observe(raw_state, key, self._embed(raw_state))
        return self.embedding_snapshot_hash

    def observe(self, state, key, optional_embedding=None) -> None:
        raw = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        self.raw_observations.append((raw, key))
        if self.is_frozen:
            self.delegate.observe(raw, key, self._embed(raw))

    def query(self, state, optional_embedding=None, *, key=None):
        started = time.perf_counter_ns()
        if not self.is_frozen:
            estimate = _empty_estimate()
            self._record(estimate, started)
            return estimate
        raw = np.asarray(state, dtype=np.float32).reshape(-1)
        estimate = self.delegate.query(raw, self._embed(raw), key=key)
        self._last_diagnostics = self.delegate.diagnostics()
        self._last_diagnostics.update(
            {
                "estimator_type": self.delegate.estimator_type,
                "representation_type": self.representation_type,
                "embedding_snapshot_hash": self.embedding_snapshot_hash,
                "embedding_freeze_step": self.embedding_freeze_step,
                "memory_bytes": self.memory_bytes(),
            }
        )
        return estimate

    def memory_bytes(self) -> int:
        raw_bytes = sum(state.nbytes for state, _ in self.raw_observations)
        parameter_bytes = 0
        if self.feature_extractor is not None:
            parameter_bytes = sum(
                parameter.nelement() * parameter.element_size()
                for parameter in self.feature_extractor.parameters()
            )
        return int(raw_bytes + parameter_bytes + self.delegate.memory_bytes())

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        diagnostics.update(
            {
                "embedding_snapshot_hash": self.embedding_snapshot_hash,
                "embedding_freeze_step": self.embedding_freeze_step,
                "embedding_frozen": self.is_frozen,
            }
        )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "estimator_type": self.estimator_type,
            "embedding_snapshot_hash": self.embedding_snapshot_hash,
            "embedding_freeze_step": self.embedding_freeze_step,
            "raw_observations": [
                (state.copy(), key) for state, key in self.raw_observations
            ],
            "delegate": self.delegate.state_dict(),
            "feature_extractor_state": (
                None
                if self.feature_extractor is None
                else copy.deepcopy(self.feature_extractor.state_dict())
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.raw_observations = [
            (np.asarray(raw, dtype=np.float32).copy(), key)
            for raw, key in state.get("raw_observations", [])
        ]
        self.embedding_snapshot_hash = state.get("embedding_snapshot_hash")
        freeze_step = state.get("embedding_freeze_step")
        self.embedding_freeze_step = None if freeze_step is None else int(freeze_step)
        self.delegate.load_state_dict(state.get("delegate", {}))
        model_state = state.get("feature_extractor_state")
        if model_state is not None:
            if self.feature_extractor is None:
                raise ValueError("Attach a compatible frozen feature extractor before loading")
            self.feature_extractor.load_state_dict(model_state)


def create_support_estimator(
    estimator_type: str,
    *,
    k: int = 5,
    bandwidth: float = 0.25,
    support_tau: float = 2.0,
    index_type: str = "brute_force",
    representation_type: str = "raw_state",
    scale_epsilon: float = 1e-8,
    covariance_regularization: float = 1e-3,
) -> SupportEstimator:
    """Construct an estimator from backward-compatible scalar config fields."""

    aliases = {
        "knn_support": "tolerance_knn",
        "feature_distance_support": "gaussian_affinity",
        "gaussian_feature_distance_affinity": "gaussian_affinity",
    }
    estimator_type = aliases.get(estimator_type, estimator_type)
    common = dict(
        k=k,
        bandwidth=bandwidth,
        support_tau=support_tau,
        index_type=index_type,
        representation_type="raw_state",
    )
    if estimator_type == "exact_count":
        estimator: SupportEstimator = ExactCountSupportEstimator(
            support_tau=support_tau
        )
    elif estimator_type == "tolerance_knn":
        estimator = ToleranceKNNSupportEstimator(**common)
    elif estimator_type == "gaussian_affinity":
        estimator = GaussianAffinitySupportEstimator(**common)
    elif estimator_type == "zscore_knn":
        estimator = ZScoreKNNSupportEstimator(
            **common, scale_epsilon=scale_epsilon
        )
    elif estimator_type == "regularized_mahalanobis":
        estimator = MahalanobisSupportEstimator(
            **common, covariance_regularization=covariance_regularization
        )
    elif estimator_type == "regularized_gaussian_density":
        estimator = RegularizedGaussianDensitySupportEstimator(
            **common, covariance_regularization=covariance_regularization
        )
    else:
        raise ValueError(f"Unknown support estimator type: {estimator_type}")
    if representation_type == "frozen_dqn_penultimate":
        if not isinstance(estimator, _VectorSupportEstimator):
            raise ValueError("Frozen embeddings require a distance-based estimator")
        return FrozenEmbeddingSupportEstimator(estimator)
    if representation_type != "raw_state":
        raise ValueError(f"Unknown support representation type: {representation_type}")
    return estimator


__all__ = [
    "ExactCountSupportEstimator",
    "FrozenEmbeddingSupportEstimator",
    "GaussianAffinitySupportEstimator",
    "MahalanobisSupportEstimator",
    "RegularizedGaussianDensitySupportEstimator",
    "SupportEstimate",
    "SupportEstimator",
    "ToleranceKNNSupportEstimator",
    "ZScoreKNNSupportEstimator",
    "create_support_estimator",
    "feature_snapshot_sha256",
]
