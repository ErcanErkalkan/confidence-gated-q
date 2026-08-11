from __future__ import annotations

import numpy as np

from scripts.benchmark_support_scaling import (
    BenchmarkSettings,
    _BruteForceEuclideanKNN,
    _IndexedEuclideanKNN,
    benchmark_grid,
    generate_memory_bank,
    generate_queries,
    validate_scaling_schema,
)


def test_synthetic_memory_bank_and_queries_are_deterministic() -> None:
    first = generate_memory_bank(32, 4, seed=17)
    second = generate_memory_bank(32, 4, seed=17)
    different = generate_memory_bank(32, 4, seed=18)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert not np.array_equal(first, different)

    queries_first = generate_queries(first, 12, seed=17)
    queries_second = generate_queries(second, 12, seed=17)
    np.testing.assert_array_equal(queries_first, queries_second)
    bank_rows = {row.tobytes() for row in first}
    assert sum(row.tobytes() in bank_rows for row in queries_first) == 6


def test_scaling_schema_for_small_deterministic_grid() -> None:
    settings = BenchmarkSettings(
        warmup_calls=1,
        measured_queries_per_repeat=2,
        timing_repeats=2,
        build_repeats=1,
        seed=23,
    )
    estimators = (
        "exact_hash_lookup",
        "brute_force_euclidean_knn",
        "ckdtree_indexed_knn",
    )
    frame = benchmark_grid((16, 32), (4,), settings=settings, estimators=estimators)
    validate_scaling_schema(frame, (16, 32), (4,), estimators)
    assert len(frame) == 6
    assert frame["status"].eq("completed").all()
    assert frame["query_sample_count"].eq(4).all()
    assert frame["gpu_used"].eq(False).all()  # noqa: E712


def test_brute_force_and_ckdtree_use_comparable_knn_distance() -> None:
    bank = generate_memory_bank(128, 4, seed=29)
    query = generate_queries(bank, 1, seed=29)[0]
    brute = _BruteForceEuclideanKNN(bank, k=5)
    indexed = _IndexedEuclideanKNN(bank, k=5)
    np.testing.assert_allclose(
        brute.query(query), indexed.query(query), rtol=1e-6, atol=1e-7
    )
