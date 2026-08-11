from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Callable, Iterable


# These variables must be set before importing NumPy/SciPy/PyTorch in the
# standalone benchmark process. The benchmark is deliberately CPU-only.
CPU_THREADS = 1
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = str(CPU_THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.agents import QNetwork  # noqa: E402
from hybrid_q.support_estimators import feature_snapshot_sha256  # noqa: E402


DEFAULT_MEMORY_SIZES = (100, 1_000, 10_000, 100_000)
DEFAULT_DIMENSIONS = (4, 22, 64)
DEFAULT_SEED = 20260805
DEFAULT_OUTPUT = ROOT / "results/reviewer1_remaining/support_scaling/scaling.csv"
DEFAULT_TABLE = ROOT / "tables/table_support_scaling.csv"
DEFAULT_FIGURE = ROOT / "figures/fig_support_scaling.pdf"
DEFAULT_REPORT = (
    ROOT / "project_admin/reviewer1_remaining/STEP12_SUPPORT_SCALING_REPORT.md"
)

ESTIMATOR_ORDER = (
    "exact_hash_lookup",
    "brute_force_euclidean_knn",
    "ckdtree_indexed_knn",
    "regularized_mahalanobis",
    "frozen_embedding_ckdtree_knn",
    "regularized_gaussian_density",
)

EXPECTED_COMPLEXITY = {
    "exact_hash_lookup": (
        "build O(n*d), expected query O(d) for state-byte hashing; "
        "worst-case hash lookup O(n)"
    ),
    "brute_force_euclidean_knn": "build O(n*d), query O(n*d)",
    "ckdtree_indexed_knn": (
        "build O(n*log(n)*d), average query O(log(n)*d + k); "
        "worst-case O(n*d)"
    ),
    "regularized_mahalanobis": (
        "build O(n*d^2 + d^3), query O(n*d + d^2) after whitening"
    ),
    "frozen_embedding_ckdtree_knn": (
        "build O(n*network_cost + n*log(n)*e), query O(network_cost + "
        "log(n)*e + k), worst-case indexed search O(n*e)"
    ),
    "regularized_gaussian_density": (
        "build O(n*d^2 + d^3), query O(n*d + d^2)"
    ),
}

SCALING_COLUMNS = [
    "estimator",
    "memory_size",
    "dimension",
    "k",
    "status",
    "reason_if_unavailable",
    "build_time_seconds",
    "median_query_latency_seconds",
    "p95_query_latency_seconds",
    "memory_bytes",
    "serialized_size_bytes",
    "warmup_calls",
    "measured_queries_per_repeat",
    "timing_repeats",
    "query_sample_count",
    "build_repeats",
    "cpu_threads",
    "logical_cpu_count",
    "cpu_model",
    "platform",
    "python_version",
    "numpy_version",
    "scipy_version",
    "torch_version",
    "device",
    "gpu_used",
    "synthetic_seed",
    "data_dtype",
    "query_hit_fraction",
    "embedding_dimension",
    "embedding_snapshot_hash",
    "expected_asymptotic_time",
    "observed_query_scaling_exponent",
    "observed_memory_scaling_exponent",
    "memory_definition",
    "serialized_size_definition",
    "timing_scope",
    "query_checksum",
]

MEMORY_DEFINITION = (
    "Estimator-owned arrays/model tensors and exposed cKDTree arrays plus "
    "Python container sizes where measurable; allocator fragmentation and "
    "temporary query workspace excluded."
)
SERIALIZED_DEFINITION = (
    "Length in bytes of pickle protocol 5 for the immutable estimator state "
    "used by this microbenchmark."
)
TIMING_SCOPE = (
    "Single-thread warmed CPU wall-clock; per-query timings include query "
    "representation and search/density evaluation but exclude synthetic data "
    "generation and serialization."
)


class ScalingBenchmarkError(ValueError):
    pass


def _seed_sequence(seed: int, memory_size: int, dimension: int, stream: int):
    return np.random.SeedSequence([seed, memory_size, dimension, stream])


def generate_memory_bank(
    memory_size: int,
    dimension: int,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Create the locked deterministic float32 synthetic memory bank."""

    if memory_size < 1 or dimension < 1:
        raise ValueError("memory_size and dimension must be positive")
    rng = np.random.default_rng(_seed_sequence(seed, memory_size, dimension, 0))
    bank = rng.standard_normal((memory_size, dimension), dtype=np.float32)
    # A deterministic low-amplitude structured component avoids an exactly
    # isotropic toy bank without creating singular covariance.
    row = np.arange(memory_size, dtype=np.float32)[:, None]
    column = np.arange(dimension, dtype=np.float32)[None, :]
    bank += np.float32(0.01) * np.sin(
        row * np.float32(0.017) + column * np.float32(0.113)
    )
    return np.ascontiguousarray(bank, dtype=np.float32)


def generate_queries(
    bank: np.ndarray,
    count: int,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Create a reproducible half-hit/half-novel query schedule."""

    if count < 1:
        raise ValueError("query count must be positive")
    memory_size, dimension = bank.shape
    rng = np.random.default_rng(_seed_sequence(seed, memory_size, dimension, 1))
    queries = rng.standard_normal((count, dimension), dtype=np.float32)
    hit_rows = np.arange(0, count, 2)
    hit_indices = rng.integers(0, memory_size, size=len(hit_rows))
    queries[hit_rows] = bank[hit_indices]
    novel_rows = np.arange(1, count, 2)
    if novel_rows.size:
        queries[novel_rows, 0] += np.float32(0.1234567)
    return np.ascontiguousarray(queries, dtype=np.float32)


def _array_bytes(*arrays: np.ndarray) -> int:
    return int(sum(array.nbytes for array in arrays))


def _tree_bytes(tree: cKDTree) -> int:
    arrays = [tree.data, tree.indices, tree.maxes, tree.mins]
    return int(tree.__sizeof__() + _array_bytes(*arrays))


class _BenchmarkEstimator:
    embedding_dimension = 0
    embedding_snapshot_hash = ""

    def query(self, state: np.ndarray) -> float:
        raise NotImplementedError

    def memory_bytes(self) -> int:
        raise NotImplementedError

    def serializable_state(self) -> Any:
        raise NotImplementedError

    def serialized_size_bytes(self) -> int:
        return len(pickle.dumps(self.serializable_state(), protocol=5))


class _ExactHashLookup(_BenchmarkEstimator):
    def __init__(self, bank: np.ndarray) -> None:
        self.counts = {row.tobytes(): 1 for row in bank}

    def query(self, state: np.ndarray) -> float:
        return float(self.counts.get(np.asarray(state, dtype=np.float32).tobytes(), 0))

    def memory_bytes(self) -> int:
        return int(
            sys.getsizeof(self.counts)
            + sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in self.counts.items())
        )

    def serializable_state(self) -> Any:
        return self.counts


class _BruteForceEuclideanKNN(_BenchmarkEstimator):
    def __init__(self, bank: np.ndarray, *, k: int) -> None:
        self.bank = np.array(bank, dtype=np.float32, order="C", copy=True)
        self.k = min(int(k), len(self.bank))

    def query(self, state: np.ndarray) -> float:
        delta = self.bank - np.asarray(state, dtype=np.float32)[None, :]
        squared = np.einsum("ij,ij->i", delta, delta, optimize=False)
        nearest = np.argpartition(squared, self.k - 1)[: self.k]
        return float(np.sqrt(np.maximum(squared[nearest], 0.0)).sum())

    def memory_bytes(self) -> int:
        return int(self.bank.nbytes)

    def serializable_state(self) -> Any:
        return {"bank": self.bank, "k": self.k}


class _IndexedEuclideanKNN(_BenchmarkEstimator):
    def __init__(self, bank: np.ndarray, *, k: int) -> None:
        self.k = min(int(k), len(bank))
        self.tree = cKDTree(bank, compact_nodes=True, balanced_tree=True, copy_data=True)

    def query(self, state: np.ndarray) -> float:
        distances, _ = self.tree.query(np.asarray(state, dtype=np.float32), k=self.k)
        return float(np.atleast_1d(distances).sum())

    def memory_bytes(self) -> int:
        return _tree_bytes(self.tree)

    def serializable_state(self) -> Any:
        return {"tree": self.tree, "k": self.k}


def _whitening(bank: np.ndarray, regularization: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    matrix = np.asarray(bank, dtype=np.float64)
    mean = matrix.mean(axis=0)
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    covariance += float(regularization) * np.eye(matrix.shape[1], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ScalingBenchmarkError("regularized covariance is not positive definite")
    cholesky = np.linalg.cholesky(covariance)
    transform = np.linalg.inv(cholesky)
    whitened = np.ascontiguousarray((matrix - mean) @ transform.T)
    return mean, transform, whitened, float(logdet)


class _RegularizedMahalanobis(_BenchmarkEstimator):
    def __init__(self, bank: np.ndarray, *, k: int, regularization: float) -> None:
        self.mean, self.transform, self.bank, _ = _whitening(bank, regularization)
        self.k = min(int(k), len(self.bank))
        self.regularization = float(regularization)

    def query(self, state: np.ndarray) -> float:
        query = (np.asarray(state, dtype=np.float64) - self.mean) @ self.transform.T
        delta = self.bank - query[None, :]
        squared = np.einsum("ij,ij->i", delta, delta, optimize=False)
        nearest = np.argpartition(squared, self.k - 1)[: self.k]
        return float(np.sqrt(np.maximum(squared[nearest], 0.0)).sum())

    def memory_bytes(self) -> int:
        return _array_bytes(self.mean, self.transform, self.bank)

    def serializable_state(self) -> Any:
        return {
            "mean": self.mean,
            "transform": self.transform,
            "bank": self.bank,
            "k": self.k,
            "regularization": self.regularization,
        }


class _RegularizedGaussianDensity(_BenchmarkEstimator):
    def __init__(self, bank: np.ndarray, *, bandwidth: float, regularization: float) -> None:
        self.mean, self.transform, self.bank, self.logdet = _whitening(bank, regularization)
        self.bandwidth = float(bandwidth)
        self.regularization = float(regularization)

    def query(self, state: np.ndarray) -> float:
        query = (np.asarray(state, dtype=np.float64) - self.mean) @ self.transform.T
        delta = self.bank - query[None, :]
        squared = np.einsum("ij,ij->i", delta, delta, optimize=False)
        log_weights = -0.5 * squared / (self.bandwidth**2)
        maximum = float(np.max(log_weights))
        log_kernel_sum = maximum + math.log(float(np.exp(log_weights - maximum).sum()))
        dimension = self.bank.shape[1]
        normalization = (
            math.log(len(self.bank))
            + dimension * math.log(self.bandwidth)
            + 0.5 * dimension * math.log(2.0 * math.pi)
            + 0.5 * self.logdet
        )
        return float(log_kernel_sum - normalization)

    def memory_bytes(self) -> int:
        return _array_bytes(self.mean, self.transform, self.bank)

    def serializable_state(self) -> Any:
        return {
            "mean": self.mean,
            "transform": self.transform,
            "bank": self.bank,
            "logdet": self.logdet,
            "bandwidth": self.bandwidth,
            "regularization": self.regularization,
        }


class _FrozenEmbeddingIndexedKNN(_BenchmarkEstimator):
    def __init__(
        self,
        bank: np.ndarray,
        *,
        k: int,
        hidden_size: int,
        seed: int,
        batch_size: int,
    ) -> None:
        torch.manual_seed(int(seed))
        self.network = QNetwork(bank.shape[1], 2, hidden_size).cpu().eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.embedding_snapshot_hash = feature_snapshot_sha256(self.network)
        self.embedding_dimension = int(hidden_size)
        self.k = min(int(k), len(bank))
        self.raw_bank = np.array(bank, dtype=np.float32, order="C", copy=True)
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(bank), batch_size):
                tensor = torch.from_numpy(bank[start : start + batch_size])
                batches.append(self.network.extract_features(tensor).cpu().numpy())
        embeddings = np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)
        self.tree = cKDTree(
            embeddings,
            compact_nodes=True,
            balanced_tree=True,
            copy_data=True,
        )

    def query(self, state: np.ndarray) -> float:
        tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            embedding = self.network.extract_features(tensor).squeeze(0).cpu().numpy()
        distances, _ = self.tree.query(embedding, k=self.k)
        return float(np.atleast_1d(distances).sum())

    def _parameter_bytes(self) -> int:
        return int(
            sum(parameter.nelement() * parameter.element_size() for parameter in self.network.parameters())
        )

    def memory_bytes(self) -> int:
        return int(self.raw_bank.nbytes + _tree_bytes(self.tree) + self._parameter_bytes())

    def serializable_state(self) -> Any:
        parameters = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self.network.state_dict().items()
        }
        return {
            "raw_bank": self.raw_bank,
            "tree": self.tree,
            "k": self.k,
            "parameters": parameters,
            "embedding_snapshot_hash": self.embedding_snapshot_hash,
        }


@dataclass(frozen=True)
class BenchmarkSettings:
    k: int = 5
    bandwidth: float = 1.0
    covariance_regularization: float = 1e-3
    hidden_size: int = 64
    embedding_batch_size: int = 4096
    warmup_calls: int = 5
    measured_queries_per_repeat: int = 20
    timing_repeats: int = 5
    build_repeats: int = 3
    seed: int = DEFAULT_SEED
    max_working_set_mib: float = 2048.0


def _factory(
    estimator: str,
    bank: np.ndarray,
    settings: BenchmarkSettings,
) -> _BenchmarkEstimator:
    if estimator == "exact_hash_lookup":
        return _ExactHashLookup(bank)
    if estimator == "brute_force_euclidean_knn":
        return _BruteForceEuclideanKNN(bank, k=settings.k)
    if estimator == "ckdtree_indexed_knn":
        return _IndexedEuclideanKNN(bank, k=settings.k)
    if estimator == "regularized_mahalanobis":
        return _RegularizedMahalanobis(
            bank,
            k=settings.k,
            regularization=settings.covariance_regularization,
        )
    if estimator == "frozen_embedding_ckdtree_knn":
        return _FrozenEmbeddingIndexedKNN(
            bank,
            k=settings.k,
            hidden_size=settings.hidden_size,
            seed=settings.seed,
            batch_size=settings.embedding_batch_size,
        )
    if estimator == "regularized_gaussian_density":
        return _RegularizedGaussianDensity(
            bank,
            bandwidth=settings.bandwidth,
            regularization=settings.covariance_regularization,
        )
    raise ValueError(f"unknown estimator: {estimator}")


def _estimated_peak_bytes(estimator: str, memory_size: int, dimension: int, settings: BenchmarkSettings) -> int:
    raw = memory_size * dimension * np.dtype(np.float32).itemsize
    if estimator == "exact_hash_lookup":
        return int(raw + memory_size * 128)
    if estimator == "brute_force_euclidean_knn":
        return int(raw * 3)
    if estimator == "ckdtree_indexed_knn":
        return int(raw * 5)
    if estimator in {"regularized_mahalanobis", "regularized_gaussian_density"}:
        return int(raw + memory_size * dimension * 8 * 3 + dimension * dimension * 8 * 4)
    if estimator == "frozen_embedding_ckdtree_knn":
        embedding = memory_size * settings.hidden_size * 8
        return int(raw * 2 + embedding * 3)
    return raw


def _runtime_metadata() -> dict[str, Any]:
    cpu_model = (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.uname().processor
        or "unknown"
    )
    return {
        "cpu_threads": CPU_THREADS,
        "logical_cpu_count": int(os.cpu_count() or 0),
        "cpu_model": cpu_model,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "gpu_used": False,
    }


def _benchmark_one(
    estimator_name: str,
    bank: np.ndarray,
    queries: np.ndarray,
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    metadata = _runtime_metadata()
    memory_size, dimension = bank.shape
    estimated_peak = _estimated_peak_bytes(estimator_name, memory_size, dimension, settings)
    limit = int(settings.max_working_set_mib * 1024 * 1024)
    base = {
        "estimator": estimator_name,
        "memory_size": memory_size,
        "dimension": dimension,
        "k": settings.k if "density" not in estimator_name and "exact" not in estimator_name else 0,
        "warmup_calls": settings.warmup_calls,
        "measured_queries_per_repeat": settings.measured_queries_per_repeat,
        "timing_repeats": settings.timing_repeats,
        "query_sample_count": settings.measured_queries_per_repeat * settings.timing_repeats,
        "build_repeats": settings.build_repeats,
        "synthetic_seed": settings.seed,
        "data_dtype": "float32",
        "query_hit_fraction": 0.5,
        "expected_asymptotic_time": EXPECTED_COMPLEXITY[estimator_name],
        "memory_definition": MEMORY_DEFINITION,
        "serialized_size_definition": SERIALIZED_DEFINITION,
        "timing_scope": TIMING_SCOPE,
        **metadata,
    }
    if estimated_peak > limit:
        return {
            **base,
            "status": "not_feasible_memory_guard",
            "reason_if_unavailable": (
                f"estimated peak {estimated_peak / 2**20:.1f} MiB exceeds "
                f"guard {settings.max_working_set_mib:.1f} MiB"
            ),
            "build_time_seconds": np.nan,
            "median_query_latency_seconds": np.nan,
            "p95_query_latency_seconds": np.nan,
            "memory_bytes": np.nan,
            "serialized_size_bytes": np.nan,
            "embedding_dimension": settings.hidden_size if "embedding" in estimator_name else 0,
            "embedding_snapshot_hash": "",
            "observed_query_scaling_exponent": np.nan,
            "observed_memory_scaling_exponent": np.nan,
            "query_checksum": np.nan,
        }

    build_samples: list[float] = []
    instance: _BenchmarkEstimator | None = None
    for _ in range(settings.build_repeats):
        started = time.perf_counter_ns()
        instance = _factory(estimator_name, bank, settings)
        build_samples.append((time.perf_counter_ns() - started) / 1e9)
    assert instance is not None

    for index in range(settings.warmup_calls):
        instance.query(queries[index % len(queries)])
    timings: list[float] = []
    checksum = 0.0
    measured = settings.measured_queries_per_repeat
    for repeat in range(settings.timing_repeats):
        for query_index in range(measured):
            index = (repeat * measured + query_index) % len(queries)
            started = time.perf_counter_ns()
            checksum += instance.query(queries[index])
            timings.append((time.perf_counter_ns() - started) / 1e9)

    return {
        **base,
        "status": "completed",
        "reason_if_unavailable": "",
        "build_time_seconds": float(np.median(build_samples)),
        "median_query_latency_seconds": float(np.median(timings)),
        "p95_query_latency_seconds": float(np.quantile(timings, 0.95)),
        "memory_bytes": int(instance.memory_bytes()),
        "serialized_size_bytes": int(instance.serialized_size_bytes()),
        "embedding_dimension": int(instance.embedding_dimension),
        "embedding_snapshot_hash": str(instance.embedding_snapshot_hash),
        "observed_query_scaling_exponent": np.nan,
        "observed_memory_scaling_exponent": np.nan,
        "query_checksum": float(checksum),
    }


def _add_observed_scaling(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for (_, _), indices in result.groupby(["estimator", "dimension"]).groups.items():
        group = result.loc[indices]
        completed = group[group["status"].eq("completed")]
        if len(completed) < 2:
            continue
        log_size = np.log(completed["memory_size"].to_numpy(dtype=float))
        query_slope = float(
            np.polyfit(
                log_size,
                np.log(completed["median_query_latency_seconds"].to_numpy(dtype=float)),
                1,
            )[0]
        )
        memory_slope = float(
            np.polyfit(
                log_size,
                np.log(completed["memory_bytes"].to_numpy(dtype=float)),
                1,
            )[0]
        )
        result.loc[indices, "observed_query_scaling_exponent"] = query_slope
        result.loc[indices, "observed_memory_scaling_exponent"] = memory_slope
    return result


def benchmark_grid(
    memory_sizes: Iterable[int],
    dimensions: Iterable[int],
    *,
    settings: BenchmarkSettings | None = None,
    estimators: Iterable[str] = ESTIMATOR_ORDER,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    settings = settings or BenchmarkSettings()
    sizes = tuple(int(value) for value in memory_sizes)
    dims = tuple(int(value) for value in dimensions)
    names = tuple(str(value) for value in estimators)
    if not sizes or not dims or not names:
        raise ValueError("memory sizes, dimensions, and estimators must be non-empty")
    unknown = sorted(set(names) - set(ESTIMATOR_ORDER))
    if unknown:
        raise ValueError(f"unknown estimators: {unknown}")
    required_queries = max(
        settings.warmup_calls,
        settings.measured_queries_per_repeat * settings.timing_repeats,
    )
    rows: list[dict[str, Any]] = []
    for dimension in dims:
        for memory_size in sizes:
            bank = generate_memory_bank(memory_size, dimension, seed=settings.seed)
            queries = generate_queries(bank, required_queries, seed=settings.seed)
            for estimator_name in names:
                if progress is not None:
                    progress(
                        f"BENCHMARK estimator={estimator_name} n={memory_size} d={dimension}"
                    )
                rows.append(_benchmark_one(estimator_name, bank, queries, settings))
    frame = _add_observed_scaling(pd.DataFrame(rows, columns=SCALING_COLUMNS))
    validate_scaling_schema(frame, sizes, dims, names)
    return frame.sort_values(["dimension", "memory_size", "estimator"]).reset_index(drop=True)


def validate_scaling_schema(
    frame: pd.DataFrame,
    memory_sizes: Iterable[int],
    dimensions: Iterable[int],
    estimators: Iterable[str] = ESTIMATOR_ORDER,
) -> None:
    missing = sorted(set(SCALING_COLUMNS) - set(frame.columns))
    if missing:
        raise ScalingBenchmarkError(f"scaling output missing columns: {missing}")
    expected = {
        (str(estimator), int(size), int(dimension))
        for estimator in estimators
        for size in memory_sizes
        for dimension in dimensions
    }
    observed = set(
        frame[["estimator", "memory_size", "dimension"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(frame) != len(expected):
        raise ScalingBenchmarkError("scaling output has missing or duplicate benchmark cells")
    if frame.duplicated(["estimator", "memory_size", "dimension"]).any():
        raise ScalingBenchmarkError("scaling output contains duplicate benchmark cells")
    completed = frame[frame["status"].eq("completed")]
    numeric = completed[
        [
            "build_time_seconds",
            "median_query_latency_seconds",
            "p95_query_latency_seconds",
            "memory_bytes",
            "serialized_size_bytes",
            "observed_query_scaling_exponent",
            "observed_memory_scaling_exponent",
            "query_checksum",
        ]
    ].to_numpy(dtype=float)
    if numeric.size and not np.isfinite(numeric).all():
        raise ScalingBenchmarkError("completed benchmark rows contain non-finite metrics")
    if not (
        completed["p95_query_latency_seconds"]
        >= completed["median_query_latency_seconds"]
    ).all():
        raise ScalingBenchmarkError("p95 query latency is below median")
    if not completed["cpu_threads"].eq(CPU_THREADS).all():
        raise ScalingBenchmarkError("benchmark CPU thread count is not fixed")
    if completed["gpu_used"].any() or not completed["device"].eq("cpu").all():
        raise ScalingBenchmarkError("GPU use is prohibited in the primary benchmark")
    if completed["cpu_model"].astype(str).str.strip().eq("").any():
        raise ScalingBenchmarkError("CPU metadata is missing")
    for (_, _), group in completed.groupby(["estimator", "dimension"]):
        ordered = group.sort_values("memory_size")
        if (np.diff(ordered["memory_bytes"].to_numpy(dtype=float)) < 0).any():
            raise ScalingBenchmarkError("accounted memory is not monotonic with bank size")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "estimator",
        "memory_size",
        "dimension",
        "status",
        "build_time_seconds",
        "median_query_latency_seconds",
        "p95_query_latency_seconds",
        "memory_bytes",
        "serialized_size_bytes",
        "observed_query_scaling_exponent",
        "observed_memory_scaling_exponent",
        "expected_asymptotic_time",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(path, index=False)


def _write_figure(frame: pd.DataFrame, path: Path) -> None:
    completed = frame[frame["status"].eq("completed")].copy()
    dimensions = sorted(completed["dimension"].unique())
    figure, axes = plt.subplots(2, len(dimensions), figsize=(15.0, 8.2), sharex="col")
    colors = {
        "exact_hash_lookup": "#243B53",
        "brute_force_euclidean_knn": "#D97706",
        "ckdtree_indexed_knn": "#2563EB",
        "regularized_mahalanobis": "#9A3412",
        "frozen_embedding_ckdtree_knn": "#60A5FA",
        "regularized_gaussian_density": "#6B7280",
    }
    markers = {
        "exact_hash_lookup": "o",
        "brute_force_euclidean_knn": "s",
        "ckdtree_indexed_knn": "^",
        "regularized_mahalanobis": "D",
        "frozen_embedding_ckdtree_knn": "P",
        "regularized_gaussian_density": "X",
    }
    linestyles = {
        "exact_hash_lookup": "-",
        "brute_force_euclidean_knn": "-",
        "ckdtree_indexed_knn": "--",
        "regularized_mahalanobis": "--",
        "frozen_embedding_ckdtree_knn": "-.",
        "regularized_gaussian_density": ":",
    }
    for column, dimension in enumerate(dimensions):
        subset = completed[completed["dimension"].eq(dimension)]
        for estimator in ESTIMATOR_ORDER:
            group = subset[subset["estimator"].eq(estimator)].sort_values("memory_size")
            if group.empty:
                continue
            common = dict(
                color=colors[estimator],
                marker=markers[estimator],
                linestyle=linestyles[estimator],
                linewidth=1.5,
                markersize=5,
                label=estimator.replace("_", " "),
            )
            axes[0, column].plot(
                group["memory_size"],
                group["median_query_latency_seconds"] * 1e6,
                **common,
            )
            axes[1, column].plot(
                group["memory_size"],
                group["memory_bytes"] / (1024 * 1024),
                **common,
            )
        axes[0, column].set_title(f"dimension = {dimension}")
        for row in range(2):
            axes[row, column].set_xscale("log")
            axes[row, column].set_yscale("log")
            axes[row, column].grid(True, which="both", color="#D1D5DB", alpha=0.55)
        axes[1, column].set_xlabel("Memory-bank size")
    axes[0, 0].set_ylabel("Median query latency (microseconds)")
    axes[1, 0].set_ylabel("Accounted estimator memory (MiB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    figure.suptitle(
        "CPU-only support-estimator scaling microbenchmark\n"
        "Synthetic deterministic banks; 1 CPU thread; warmed repeated queries",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _write_report(frame: pd.DataFrame, path: Path) -> None:
    completed = frame[frame["status"].eq("completed")]
    unavailable = frame[~frame["status"].eq("completed")]
    metadata = completed.iloc[0]
    dimension = 22 if 22 in set(completed["dimension"]) else int(completed["dimension"].iloc[0])
    slopes = (
        completed[completed["dimension"].eq(dimension)]
        .drop_duplicates("estimator")
        .set_index("estimator")
    )
    largest_size = int(completed["memory_size"].max())
    largest = completed[completed["memory_size"].eq(largest_size)]
    indexed_comparisons: list[str] = []
    for current_dimension in sorted(largest["dimension"].unique()):
        group = largest[largest["dimension"].eq(current_dimension)].set_index(
            "estimator"
        )
        brute = float(
            group.loc[
                "brute_force_euclidean_knn", "median_query_latency_seconds"
            ]
        )
        indexed = float(
            group.loc["ckdtree_indexed_knn", "median_query_latency_seconds"]
        )
        if indexed < brute:
            comparison = f"cKDTree was {brute / indexed:.1f}x faster"
        else:
            comparison = f"cKDTree was {indexed / brute:.2f}x slower"
        indexed_comparisons.append(
            f"- At n={largest_size:,}, d={int(current_dimension)}, {comparison} "
            f"than brute force ({indexed * 1e6:.1f} versus {brute * 1e6:.1f} microseconds)."
        )
    lines = [
        "# Step 12 - Support-Estimator Computational Scaling",
        "",
        "## Scope and outcome",
        "",
        (
            f"The CPU-only microbenchmark completed {len(completed)}/{len(frame)} requested estimator-size-dimension cells. "
            f"{len(unavailable)} cells were unavailable under the predeclared memory guard."
        ),
        "This is a controlled kernel microbenchmark, not a deployment-readiness or end-to-end control-loop claim.",
        "",
        "## Locked measurement design",
        "",
        f"- Memory sizes: {sorted(frame['memory_size'].unique().tolist())}",
        f"- Dimensions: {sorted(frame['dimension'].unique().tolist())}",
        f"- Fixed CPU threads: {int(metadata['cpu_threads'])}; GPU used: {bool(metadata['gpu_used'])}",
        f"- Warm-up calls: {int(metadata['warmup_calls'])}; measured queries per repeat: {int(metadata['measured_queries_per_repeat'])}; repeats: {int(metadata['timing_repeats'])}",
        f"- Build repetitions: {int(metadata['build_repeats'])}; synthetic seed: {int(metadata['synthetic_seed'])}",
        "- Every method receives the same float32 bank and half-hit/half-novel deterministic query schedule for a given size and dimension.",
        "- Brute-force Euclidean and cKDTree use the same k=5 query definition and are directly comparable.",
        "",
        "## Hardware and software",
        "",
        f"- CPU: {metadata['cpu_model']}",
        f"- Platform: {metadata['platform']}",
        f"- Python {metadata['python_version']}; NumPy {metadata['numpy_version']}; SciPy {metadata['scipy_version']}; PyTorch {metadata['torch_version']}",
        "",
        f"## Observed log-log scaling at dimension {dimension}",
        "",
        "| Estimator | Query exponent | Memory exponent | Expected time |",
        "|---|---:|---:|---|",
    ]
    for estimator in ESTIMATOR_ORDER:
        if estimator not in slopes.index:
            continue
        row = slopes.loc[estimator]
        lines.append(
            f"| `{estimator}` | {float(row['observed_query_scaling_exponent']):.3f} | "
            f"{float(row['observed_memory_scaling_exponent']):.3f} | {EXPECTED_COMPLEXITY[estimator]} |"
        )
    lines.extend(
        [
            "",
            "Observed exponents summarize only four memory-bank sizes and include fixed Python/timing overhead, so they should not be read as proofs of asymptotic complexity.",
            "",
            f"## Direct brute-force versus indexed comparison at n={largest_size:,}",
            "",
            *indexed_comparisons,
            "",
            "The indexed method therefore helped strongly in four dimensions but did not outperform the vectorized brute-force kernel at 22 or 64 dimensions on this bank geometry and hardware.",
            "",
            "## Method definitions",
            "",
            "- Exact lookup hashes the complete float32 state bytes; its 50% hit schedule is reported separately from metric-neighbour semantics.",
            "- Brute-force and cKDTree operate on the identical raw matrix and query vectors with k=5.",
            "- Mahalanobis and Gaussian density use a regularized covariance and immutable-bank whitening built once during the measured build phase.",
            "- Frozen embedding includes CPU feature extraction from a deterministic 64-unit Q-network snapshot plus cKDTree search. The snapshot is randomly initialized and measures representation cost, not representation quality.",
            "",
            "## Limitations",
            "",
            "- The benchmark uses immutable vectorized adapters for the declared estimator kernels; it does not time mutable Python-dictionary ledger materialization in the production agent API. That API can add state-management overhead beyond these measurements.",
            "- Timings exclude environment simulation, agent inference beyond the frozen feature extractor, Python orchestration outside each query, and concurrent workload contention.",
            "- The immutable-bank benchmark does not measure online insertions or index/covariance rebuild frequency. Production latency can differ when support memory changes during training.",
            "- Accounted memory excludes allocator fragmentation, temporary query workspace, and opaque native-node overhead not exposed by SciPy; serialized size is separately measured from the actual immutable state payload.",
            "- cKDTree average-case behavior can degrade with dimension and data geometry; the observed 4/22/64-dimensional results must not be generalized to arbitrary state distributions.",
            "- Wall-clock measurements are machine-specific and naturally non-deterministic even though all data and query schedules are deterministic.",
            "",
            "## Generated evidence",
            "",
            "- `results/reviewer1_remaining/support_scaling/scaling.csv` contains the full benchmark grain and hardware/software metadata.",
            "- `tables/table_support_scaling.csv` is the publication-facing numerical table.",
            "- `figures/fig_support_scaling.pdf` shows query-latency and accounted-memory scaling on log-log axes.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-sizes", type=_parse_int_list, default=DEFAULT_MEMORY_SIZES)
    parser.add_argument("--dimensions", type=_parse_int_list, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--measured-queries", type=int, default=20)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--build-repeats", type=int, default=3)
    parser.add_argument("--max-working-set-mib", type=float, default=2048.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="preserve recorded timings and regenerate only table, figure, and report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.warmup_calls,
        args.measured_queries,
        args.timing_repeats,
        args.build_repeats,
    ) < 1:
        raise ScalingBenchmarkError("warm-up, query, timing, and build counts must be positive")
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(CPU_THREADS)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    settings = BenchmarkSettings(
        warmup_calls=args.warmup_calls,
        measured_queries_per_repeat=args.measured_queries,
        timing_repeats=args.timing_repeats,
        build_repeats=args.build_repeats,
        max_working_set_mib=args.max_working_set_mib,
    )
    if args.render_existing:
        if not args.output.is_file():
            raise ScalingBenchmarkError(
                f"recorded scaling evidence does not exist: {args.output}"
            )
        frame = pd.read_csv(args.output, float_precision="round_trip")
        validate_scaling_schema(
            frame,
            args.memory_sizes,
            args.dimensions,
            ESTIMATOR_ORDER,
        )
    else:
        frame = benchmark_grid(
            args.memory_sizes,
            args.dimensions,
            settings=settings,
            progress=lambda message: print(message, flush=True),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    _write_table(frame, args.table)
    _write_figure(frame, args.figure)
    _write_report(frame, args.report)
    print(
        "SUPPORT_SCALING_PASS "
        f"rows={len(frame)} completed={int(frame['status'].eq('completed').sum())} "
        f"unavailable={int((~frame['status'].eq('completed')).sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
