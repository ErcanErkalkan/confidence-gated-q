from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.trace_evidence import (  # noqa: E402
    SAFETY_TRACE_SCHEMA_VERSION,
    SENSOR_COLUMNS,
    TRACE_SCHEMA_VERSION,
    SUPPORTED_TRACE_SCHEMA_VERSIONS,
    trace_fieldnames,
)


DEFAULT_TRACE_DIR = (
    ROOT / "results/diagnostic_extensions/sensor_aliasing/smoke/trace_shards"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/diagnostic_extensions/sensor_aliasing"
DEFAULT_LATENT_RADIUS = 0.10
DEFAULT_OBSERVATION_RADIUS = 0.15
DEFAULT_MATERIAL_LATENT_DISTANCE = 0.30
DEFAULT_MAX_PAIRS = 1_000_000
GROUP_COLUMNS = ["environment", "agent", "seed", "phase"]
LATENT_COLUMNS = [
    "latent_position_x",
    "latent_position_y",
    "latent_position_z",
    "latent_velocity_x",
    "latent_velocity_y",
    "latent_velocity_z",
    "target_x",
    "target_y",
    "target_z",
]
LATENT_SCALE = np.asarray(
    [1.0, 1.0, 1.2, 0.8, 0.8, 0.8, 1.0, 1.0, 1.2],
    dtype=np.float64,
)


class SensorAliasingError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uncompressed_bytes(path: Path) -> int:
    total = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
    return total


def validate_trace_schema(frame: pd.DataFrame) -> None:
    if "trace_schema_version" not in frame:
        raise SensorAliasingError("trace missing trace_schema_version")
    versions = set(frame["trace_schema_version"].astype(str))
    if len(versions) != 1 or not versions.issubset(
        SUPPORTED_TRACE_SCHEMA_VERSIONS
    ):
        raise SensorAliasingError("trace schema version mismatch")
    schema_version = next(iter(versions))
    missing = set(trace_fieldnames(schema_version)) - set(frame.columns)
    if missing:
        raise SensorAliasingError(f"trace missing columns: {sorted(missing)}")
    if frame.empty:
        raise SensorAliasingError("trace contains no decision rows")
    grain = [*GROUP_COLUMNS, "checkpoint", "episode", "step"]
    if frame.duplicated(grain).any():
        raise SensorAliasingError("duplicate decision rows at the declared grain")
    numeric = frame[[*LATENT_COLUMNS, *SENSOR_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise SensorAliasingError("latent or sensor columns contain non-finite values")
    sensors = numeric[SENSOR_COLUMNS].to_numpy(dtype=float)
    if np.any(sensors < -1.000001) or np.any(sensors > 1.000001):
        raise SensorAliasingError("sensor values fall outside [-1, 1]")
    for column in ("encoded_vector_hash", "exact_key_hash"):
        values = frame[column].astype(str)
        if not values.str.fullmatch(r"[0-9a-f]{64}").all():
            raise SensorAliasingError(f"invalid SHA-256 values in {column}")
    timestamps = frame[
        ["observation_timestamp", "command_timestamp", "effective_latency"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(timestamps.to_numpy(dtype=float)).all():
        raise SensorAliasingError("timestamps contain non-finite values")
    observed_latency = (
        timestamps["command_timestamp"] - timestamps["observation_timestamp"]
    )
    if not np.allclose(
        observed_latency, timestamps["effective_latency"], atol=1e-9
    ):
        raise SensorAliasingError("effective latency contradicts timestamps")
    if (timestamps["effective_latency"] < -1e-12).any():
        raise SensorAliasingError("effective latency cannot be negative")
    if schema_version in {TRACE_SCHEMA_VERSION, SAFETY_TRACE_SCHEMA_VERSION}:
        outcome_numeric = frame[
            [
                "post_action_trajectory_error",
                "post_action_motor_saturation",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(outcome_numeric.to_numpy(dtype=float)).all():
            raise SensorAliasingError("v2 trace outcomes contain non-finite values")
        if (outcome_numeric["post_action_trajectory_error"] < 0).any():
            raise SensorAliasingError("trajectory error cannot be negative")
        saturation = outcome_numeric["post_action_motor_saturation"]
        if ((saturation < 0) | (saturation > 1)).any():
            raise SensorAliasingError("motor saturation must be in [0, 1]")
        binary_columns = [
            "post_action_constraint_active",
            "post_action_risk_zone",
            "post_action_collision",
            "post_action_saturation_active",
            "post_action_target_visibility",
            "post_action_recovery_event",
            "post_action_terminated",
            "post_action_truncated",
            "post_action_success",
        ]
        binary = frame[binary_columns].apply(pd.to_numeric, errors="coerce")
        if not binary.isin([0, 1]).all().all():
            raise SensorAliasingError("v2 outcome flags must be binary")
        if frame["post_action_failure_stage"].astype(str).str.len().eq(0).any():
            raise SensorAliasingError("v2 failure-stage labels must be present")
    if schema_version == SAFETY_TRACE_SCHEMA_VERSION:
        required_finite = [
            "control_timestep_seconds",
            "post_action_timestamp",
            "post_action_latent_position_x",
            "post_action_latent_position_y",
            "post_action_latent_position_z",
            "post_action_reference_x",
            "post_action_reference_y",
            "post_action_reference_z",
            "post_action_trajectory_deviation",
            "post_action_minimum_obstacle_clearance",
        ]
        safety_numeric = frame[required_finite].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(safety_numeric.to_numpy(dtype=float)).all():
            raise SensorAliasingError("v3 safety outcomes contain non-finite values")
        if (safety_numeric["post_action_trajectory_deviation"] < 0).any():
            raise SensorAliasingError("trajectory deviation cannot be negative")
        if (safety_numeric["control_timestep_seconds"] <= 0).any():
            raise SensorAliasingError("control timestep must be positive")
        flags = frame[
            ["post_action_perturbation_active", "post_action_near_miss"]
        ].apply(pd.to_numeric, errors="coerce")
        if not flags.isin([0, 1]).all().all():
            raise SensorAliasingError("v3 safety flags must be binary")
        if frame["nominal_reference_path_id"].astype(str).str.len().eq(0).any():
            raise SensorAliasingError("v3 reference-path labels must be present")


def _pairs_within(
    matrix: np.ndarray, radius: float, max_pairs: int
) -> np.ndarray:
    if radius < 0:
        raise SensorAliasingError("similarity radius must be non-negative")
    if matrix.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    pairs = cKDTree(matrix).query_pairs(radius, output_type="ndarray")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) > max_pairs:
        raise SensorAliasingError(
            f"eligible pair count {len(pairs)} exceeds max_pairs={max_pairs}"
        )
    if len(pairs):
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
    return pairs


def _latent_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[LATENT_COLUMNS].to_numpy(dtype=np.float64) / LATENT_SCALE


def fragmentation_rows(
    frame: pd.DataFrame,
    *,
    latent_radius: float = DEFAULT_LATENT_RADIUS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        group = group.reset_index(drop=True)
        pairs = _pairs_within(_latent_matrix(group), latent_radius, max_pairs)
        hashes = group["exact_key_hash"].astype(str).to_numpy()
        fragmented = int(
            np.count_nonzero(hashes[pairs[:, 0]] != hashes[pairs[:, 1]])
        ) if len(pairs) else 0
        denominator = int(len(pairs))
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, identity)),
                "similar_latent_pair_count": denominator,
                "different_exact_key_pair_count": fragmented,
                "fragmentation_rate": (
                    fragmented / denominator if denominator else np.nan
                ),
                "latent_similarity_radius": latent_radius,
                "status": "available" if denominator else "not_available",
                "reason_if_unavailable": (
                    "" if denominator else "no similar-latent unordered pairs"
                ),
            }
        )
    return pd.DataFrame(rows)


def aliasing_rows(
    frame: pd.DataFrame,
    *,
    observation_radius: float = DEFAULT_OBSERVATION_RADIUS,
    material_latent_distance: float = DEFAULT_MATERIAL_LATENT_DISTANCE,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        group = group.reset_index(drop=True)
        observations = group[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        pairs = _pairs_within(observations, observation_radius, max_pairs)
        latent = _latent_matrix(group)
        if len(pairs):
            latent_distance = np.linalg.norm(
                latent[pairs[:, 0]] - latent[pairs[:, 1]], axis=1
            )
            materially_different = latent_distance > material_latent_distance
            labels = pd.to_numeric(
                group["optimal_action"], errors="coerce"
            ).to_numpy(dtype=float)
            left = labels[pairs[:, 0]]
            right = labels[pairs[:, 1]]
            label_mismatch = np.isfinite(left) & np.isfinite(right) & (left != right)
            aliased = materially_different | label_mismatch
        else:
            materially_different = np.asarray([], dtype=bool)
            label_mismatch = np.asarray([], dtype=bool)
            aliased = np.asarray([], dtype=bool)
        denominator = int(len(pairs))
        numerator = int(np.count_nonzero(aliased))
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, identity)),
                "similar_observation_pair_count": denominator,
                "materially_different_latent_pair_count": int(
                    np.count_nonzero(materially_different)
                ),
                "different_optimal_label_pair_count": int(
                    np.count_nonzero(label_mismatch)
                ),
                "aliased_pair_count": numerator,
                "aliasing_rate": numerator / denominator if denominator else np.nan,
                "observation_similarity_radius": observation_radius,
                "material_latent_distance": material_latent_distance,
                "status": "available" if denominator else "not_available",
                "reason_if_unavailable": (
                    "" if denominator else "no similar-observation unordered pairs"
                ),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbor_disagreement_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        current = pd.to_numeric(
            group["tabular_greedy_action"], errors="coerce"
        ).to_numpy(dtype=float)
        neighbor = pd.to_numeric(
            group["nearest_neighbor_greedy_action"], errors="coerce"
        ).to_numpy(dtype=float)
        eligible = np.isfinite(current) & np.isfinite(neighbor)
        denominator = int(np.count_nonzero(eligible))
        numerator = int(np.count_nonzero(current[eligible] != neighbor[eligible]))
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, identity)),
                "eligible_decision_count": denominator,
                "disagreement_count": numerator,
                "nearest_neighbor_action_disagreement_rate": (
                    numerator / denominator if denominator else np.nan
                ),
                "status": "available" if denominator else "not_available",
                "reason_if_unavailable": (
                    "" if denominator else "nearest/current tabular actions unavailable"
                ),
            }
        )
    return pd.DataFrame(rows)


def support_correctness_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, 1.0, 6)
    for identity, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        support = pd.to_numeric(group["support_score"], errors="coerce")
        exact = pd.to_numeric(group["tabular_action_correct"], errors="coerce")
        proxy = pd.to_numeric(
            group["tabular_reference_action_agreement"], errors="coerce"
        )
        if (np.isfinite(support) & np.isfinite(exact)).any():
            correctness = exact
            target = "analytic_optimal_action"
            inferential_status = "diagnostic_correctness"
        else:
            correctness = proxy
            target = "nominal_free_space_reference_action"
            inferential_status = "reference_agreement_proxy_not_optimality"
        eligible = np.isfinite(support) & np.isfinite(correctness)
        if not eligible.any():
            rows.append(
                {
                    **dict(zip(GROUP_COLUMNS, identity)),
                    "correctness_target": "unavailable",
                    "inferential_status": "not_available",
                    "support_bin_low": np.nan,
                    "support_bin_high": np.nan,
                    "eligible_decision_count": 0,
                    "mean_support": np.nan,
                    "correctness_rate": np.nan,
                    "status": "not_available",
                    "reason_if_unavailable": "support/correctness pair unavailable",
                }
            )
            continue
        eligible_support = support[eligible].to_numpy(dtype=float)
        eligible_correctness = correctness[eligible].to_numpy(dtype=float)
        indices = np.minimum(np.digitize(eligible_support, bins[1:-1]), 4)
        for index in range(5):
            selected = indices == index
            if not selected.any():
                continue
            rows.append(
                {
                    **dict(zip(GROUP_COLUMNS, identity)),
                    "correctness_target": target,
                    "inferential_status": inferential_status,
                    "support_bin_low": bins[index],
                    "support_bin_high": bins[index + 1],
                    "eligible_decision_count": int(np.count_nonzero(selected)),
                    "mean_support": float(eligible_support[selected].mean()),
                    "correctness_rate": float(
                        eligible_correctness[selected].mean()
                    ),
                    "status": "available",
                    "reason_if_unavailable": "",
                }
            )
    return pd.DataFrame(rows)


def _weighted_summary(
    frame: pd.DataFrame, numerator: str, denominator: str, metric: str
) -> tuple[int, int, float]:
    valid = frame[frame["status"] == "available"]
    n = int(valid[numerator].sum()) if not valid.empty else 0
    d = int(valid[denominator].sum()) if not valid.empty else 0
    return n, d, n / d if d else np.nan


def _load_traces(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest = []
    for path in sorted(paths):
        frame = pd.read_csv(path, compression="gzip")
        validate_trace_schema(frame)
        frames.append(frame)
        compressed = path.stat().st_size
        uncompressed = _uncompressed_bytes(path)
        manifest.append(
            {
                "source_file": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "row_count": len(frame),
                "compressed_bytes": compressed,
                "uncompressed_bytes": uncompressed,
                "compression_ratio": compressed / uncompressed,
                "sha256": sha256(path),
                "schema_status": "PASS",
            }
        )
    if not frames:
        raise SensorAliasingError("no compressed trace shards discovered")
    combined = pd.concat(frames, ignore_index=True)
    validate_trace_schema(combined)
    return combined, pd.DataFrame(manifest)


def aggregate(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    latent_radius: float = DEFAULT_LATENT_RADIUS,
    observation_radius: float = DEFAULT_OBSERVATION_RADIUS,
    material_latent_distance: float = DEFAULT_MATERIAL_LATENT_DISTANCE,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> dict[str, int | float]:
    traces, trace_manifest = _load_traces(trace_dir.glob("*.csv.gz"))
    fragmentation = fragmentation_rows(
        traces, latent_radius=latent_radius, max_pairs=max_pairs
    )
    aliasing = aliasing_rows(
        traces,
        observation_radius=observation_radius,
        material_latent_distance=material_latent_distance,
        max_pairs=max_pairs,
    )
    disagreement = nearest_neighbor_disagreement_rows(traces)
    support_correctness = support_correctness_rows(traces)

    fragmented, similar_latent, fragmentation_rate = _weighted_summary(
        fragmentation,
        "different_exact_key_pair_count",
        "similar_latent_pair_count",
        "fragmentation_rate",
    )
    aliased, similar_observation, aliasing_rate = _weighted_summary(
        aliasing,
        "aliased_pair_count",
        "similar_observation_pair_count",
        "aliasing_rate",
    )
    disagreements, eligible_neighbors, disagreement_rate = _weighted_summary(
        disagreement,
        "disagreement_count",
        "eligible_decision_count",
        "nearest_neighbor_action_disagreement_rate",
    )
    summary = pd.DataFrame(
        [
            {
                "trace_rows": len(traces),
                "trace_shards": len(trace_manifest),
                "similar_latent_pair_count": similar_latent,
                "different_exact_key_pair_count": fragmented,
                "fragmentation_rate": fragmentation_rate,
                "similar_observation_pair_count": similar_observation,
                "aliased_pair_count": aliased,
                "aliasing_rate": aliasing_rate,
                "eligible_neighbor_decision_count": eligible_neighbors,
                "neighbor_action_disagreement_count": disagreements,
                "nearest_neighbor_action_disagreement_rate": disagreement_rate,
                "final_seed_rows": int(traces["seed"].between(16000, 16099).sum()),
            }
        ]
    )
    source = "results/diagnostic_extensions/sensor_aliasing/smoke/trace_shards/*.csv.gz"
    definitions = pd.DataFrame(
        [
            {
                "metric_name": "fragmentation_rate",
                "definition": (
                    "similar normalized latent-state unordered pairs with different "
                    "exact-key hashes / all similar latent-state unordered pairs"
                ),
                "denominator": "similar_latent_pair_count",
                "threshold": latent_radius,
                "direction": "lower_is_less_fragmented",
                "availability": "available" if similar_latent else "not_available",
                "reason_if_unavailable": "" if similar_latent else "zero denominator",
                "source_file": source,
            },
            {
                "metric_name": "aliasing_rate",
                "definition": (
                    "similar 22-D observation unordered pairs with materially "
                    "different normalized latent states or different analytic "
                    "optimal labels / all similar-observation unordered pairs"
                ),
                "denominator": "similar_observation_pair_count",
                "threshold": observation_radius,
                "direction": "lower_is_less_aliased",
                "availability": "available" if similar_observation else "not_available",
                "reason_if_unavailable": (
                    "" if similar_observation else "zero denominator"
                ),
                "source_file": source,
            },
            {
                "metric_name": "nearest_neighbor_action_disagreement_rate",
                "definition": (
                    "current tabular greedy action differs from stored nearest-neighbor "
                    "tabular greedy action / decisions with both actions available"
                ),
                "denominator": "eligible_neighbor_decision_count",
                "threshold": np.nan,
                "direction": "lower_is_more_action_consistent",
                "availability": "available" if eligible_neighbors else "not_available",
                "reason_if_unavailable": "" if eligible_neighbors else "zero denominator",
                "source_file": source,
            },
            {
                "metric_name": "support_correctness_relationship",
                "definition": (
                    "correctness rate within five pre-declared support-score bins; "
                    "nominal reference agreement is labeled as a proxy when analytic "
                    "optimal actions are unavailable"
                ),
                "denominator": "eligible decisions per support bin",
                "threshold": 0.2,
                "direction": "descriptive",
                "availability": (
                    "available"
                    if (support_correctness["status"] == "available").any()
                    else "not_available"
                ),
                "reason_if_unavailable": (
                    ""
                    if (support_correctness["status"] == "available").any()
                    else "no paired support/correctness evidence"
                ),
                "source_file": source,
            },
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "trace_manifest.csv": trace_manifest,
        "fragmentation.csv": fragmentation,
        "aliasing.csv": aliasing,
        "nearest_neighbor_action_disagreement.csv": disagreement,
        "support_correctness.csv": support_correctness,
        "diagnostic_manifest.csv": definitions,
        "summary.csv": summary,
    }
    for name, artifact in artifacts.items():
        artifact.to_csv(output_dir / name, index=False)
    return {
        "trace_rows": len(traces),
        "trace_shards": len(trace_manifest),
        "fragmentation_rate": fragmentation_rate,
        "aliasing_rate": aliasing_rate,
        "neighbor_disagreement_rate": disagreement_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-radius", type=float, default=DEFAULT_LATENT_RADIUS)
    parser.add_argument(
        "--observation-radius", type=float, default=DEFAULT_OBSERVATION_RADIUS
    )
    parser.add_argument(
        "--material-latent-distance",
        type=float,
        default=DEFAULT_MATERIAL_LATENT_DISTANCE,
    )
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    args = parser.parse_args()
    result = aggregate(
        args.trace_dir,
        args.output_dir,
        latent_radius=args.latent_radius,
        observation_radius=args.observation_radius,
        material_latent_distance=args.material_latent_distance,
        max_pairs=args.max_pairs,
    )
    print(
        "SENSOR_ALIASING_AGGREGATION_PASS "
        f"rows={result['trace_rows']} shards={result['trace_shards']}"
    )


if __name__ == "__main__":
    main()
