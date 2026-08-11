from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.experiment import run_config  # noqa: E402


DEFAULT_REGISTRY = ROOT / (
    "configs/diagnostic_extensions/support_development/candidate_registry.yaml"
)
DEFAULT_REGISTRY_DIGEST = DEFAULT_REGISTRY.with_suffix(".sha256")
DEFAULT_SEED_REGISTRY = ROOT / (
    "configs/diagnostic_extensions/seed_registry.yaml"
)
DEFAULT_CONFIG = ROOT / (
    "configs/diagnostic_extensions/support_development/matched_candidates.yaml"
)
DEFAULT_RAW = ROOT / (
    "results/diagnostic_extensions/support_development/execution/raw.csv"
)
EXPECTED_FAMILIES = {
    "raw_normalized",
    "zscore",
    "mahalanobis",
    "frozen_embedding",
    "gaussian_density",
}


class SupportDevelopmentError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportDevelopmentError(f"YAML root must be a mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_registry_digest(
    registry_path: Path = DEFAULT_REGISTRY,
    digest_path: Path = DEFAULT_REGISTRY_DIGEST,
) -> str:
    if not digest_path.exists():
        raise SupportDevelopmentError(
            f"missing candidate registry digest: {digest_path}"
        )
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = file_sha256(registry_path)
    if observed != expected:
        raise SupportDevelopmentError("support candidate registry digest mismatch")
    return observed


def flatten_candidates(registry: dict[str, Any]) -> list[dict[str, Any]]:
    slots = {
        item["slot_id"]: item for item in registry["fractional_design"]["slots"]
    }
    rows: list[dict[str, Any]] = []
    for family in registry["families"]:
        for candidate in family["candidates"]:
            slot = slots.get(candidate.get("slot_id"))
            if slot is None:
                raise SupportDevelopmentError(
                    f"unknown slot for {candidate.get('candidate_id')}"
                )
            rows.append(
                {
                    **slot,
                    **candidate,
                    "family_id": family["family_id"],
                    "estimator_type": family["estimator_type"],
                    "representation": family["representation"],
                    "scale_epsilon": family.get("scale_epsilon", 1e-8),
                    "effective_index": family.get(
                        "effective_index", slot["requested_index"]
                    ),
                    "k_semantics": family.get(
                        "k_semantics", "nearest_k_retrieval"
                    ),
                }
            )
    return rows


def validate_registry(
    registry: dict[str, Any], seed_registry: dict[str, Any]
) -> None:
    if registry.get("status") != "locked_before_development_outcomes":
        raise SupportDevelopmentError("candidate registry is not outcome-locked")
    if registry.get("evidence_class") != "development":
        raise SupportDevelopmentError("candidate registry is not development-only")
    families = registry.get("families", [])
    if {item.get("family_id") for item in families} != EXPECTED_FAMILIES:
        raise SupportDevelopmentError("support estimator family set is invalid")
    budget = registry["matched_budget"]
    expected_per_family = int(budget["candidate_count_per_family"])
    counts = [len(item.get("candidates", [])) for item in families]
    if len(set(counts)) != 1 or counts[0] != expected_per_family:
        raise SupportDevelopmentError("candidate counts differ across families")
    candidates = flatten_candidates(registry)
    if len(candidates) != int(budget["total_candidate_count"]):
        raise SupportDevelopmentError("total candidate count differs from budget")
    ids = [item.get("candidate_id") for item in candidates]
    if None in ids or len(ids) != len(set(ids)):
        raise SupportDevelopmentError("candidate IDs are missing or duplicated")
    required = {"k", "h", "tau_approx", "regularization", "requested_index"}
    for candidate in candidates:
        if not required <= candidate.keys():
            raise SupportDevelopmentError(
                f"candidate dimensions missing for {candidate['candidate_id']}"
            )
        if int(candidate["k"]) < 1 or float(candidate["h"]) <= 0:
            raise SupportDevelopmentError("invalid k or h")
        if float(candidate["tau_approx"]) <= 0:
            raise SupportDevelopmentError("invalid tau_approx")

    seed_lock = registry["seed_lock"]
    reservation = seed_registry["reserved_ranges"][seed_lock["registry_key"]]
    seeds = [int(seed) for seed in seed_lock["development_seeds"]]
    if not seeds or any(
        seed < int(reservation["start"]) or seed > int(reservation["end"])
        for seed in seeds
    ):
        raise SupportDevelopmentError("development seed outside registered range")
    if len(seeds) != len(set(seeds)):
        raise SupportDevelopmentError("duplicate development seed")
    final_reservation = seed_registry["reserved_ranges"][
        seed_lock["final_registry_key"]
    ]
    if any(
        int(final_reservation["start"]) <= seed <= int(final_reservation["end"])
        for seed in seeds
    ):
        raise SupportDevelopmentError("reserved support final seed detected")
    for environment in registry["environments"]:
        if [int(seed) for seed in environment["development_seeds"]] != seeds:
            raise SupportDevelopmentError("environment seed sets are not matched")
    if {item["environment_key"] for item in registry["environments"]} != {
        "application_goal_shift",
        "locked_transition_dynamics_shift",
    }:
        raise SupportDevelopmentError("required development environments missing")
    locked = next(
        item
        for item in registry["environments"]
        if item["environment_key"] == "locked_transition_dynamics_shift"
    )
    if (
        locked["environment_id"] != "TransitionDynamicsShift-v0"
        or locked["severity_id"] != "transition_slip_035"
        or float(locked["kwargs"]["post_shift_slip_probability"]) != 0.35
    ):
        raise SupportDevelopmentError("locked transition mechanism differs")
    expected_order = [
        "lowest_mean_environment_rank",
        "lowest_worst_environment_rank",
        "lower_failure_probability",
        "lower_query_latency",
        "lexical_candidate_id",
    ]
    if registry["selection_rule"]["ordered_criteria"] != expected_order:
        raise SupportDevelopmentError("selection criteria differ from lock")


def expected_candidate_params(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in flatten_candidates(registry):
        representation = (
            "frozen_dqn_penultimate"
            if candidate["family_id"] == "frozen_embedding"
            else "raw_state"
        )
        params: dict[str, Any] = {
            "support_estimator_type": candidate["estimator_type"],
            "support_representation_type": representation,
            "support_index_type": candidate["requested_index"],
            "support_scale_epsilon": float(candidate["scale_epsilon"]),
            "support_covariance_regularization": float(
                candidate["regularization"]
            ),
            "approximate_support_k": int(candidate["k"]),
            "approximate_support_bandwidth": float(candidate["h"]),
            "approximate_support_tau": float(candidate["tau_approx"]),
        }
        if candidate["family_id"] == "frozen_embedding":
            params["support_embedding_freeze_step"] = int(
                candidate["embedding_freeze_step"]
            )
        result[candidate["candidate_id"]] = params
    return result


def validate_config(config: dict[str, Any], registry: dict[str, Any]) -> None:
    if config.get("analysis", {}).get("evidence_class") != "development":
        raise SupportDevelopmentError("config is not development-only")
    if config.get("output_dir") != (
        "results/diagnostic_extensions/support_development/execution"
    ):
        raise SupportDevelopmentError("unexpected output directory")
    budget = registry["matched_budget"]
    runtime = config.get("runtime", {})
    for field in ("torch_threads", "torch_interop_threads", "workers"):
        if int(runtime.get(field, -1)) != int(budget[field]):
            raise SupportDevelopmentError(f"runtime budget differs: {field}")
    evaluation = config.get("evaluation", {})
    if int(evaluation.get("interval_steps", -1)) != int(
        budget["checkpoint_interval"]
    ) or int(evaluation.get("episodes", -1)) != int(
        budget["evaluation_episodes_per_checkpoint"]
    ):
        raise SupportDevelopmentError("evaluation budget differs from registry")
    expected_envs = {
        item["environment_name"]: item for item in registry["environments"]
    }
    if {item.get("name") for item in config.get("envs", [])} != set(expected_envs):
        raise SupportDevelopmentError("environment set differs from registry")
    for environment in config["envs"]:
        expected = expected_envs[environment["name"]]
        if environment.get("id") != expected["environment_id"]:
            raise SupportDevelopmentError("environment ID differs from registry")
        if environment.get("kwargs", {}) != expected["kwargs"]:
            raise SupportDevelopmentError("environment kwargs differ from registry")
        if environment.get("eval_kwargs") != expected.get("eval_kwargs"):
            raise SupportDevelopmentError("environment eval kwargs differ")
        if [int(seed) for seed in environment.get("seeds", [])] != [
            int(seed) for seed in expected["development_seeds"]
        ]:
            raise SupportDevelopmentError("environment seed budget differs")
        if int(environment.get("training_steps", -1)) != int(
            budget["training_steps_per_agent_seed"]
        ):
            raise SupportDevelopmentError("training budget differs")
    expected_params = expected_candidate_params(registry)
    agents = config.get("agents", [])
    if {item.get("name") for item in agents} != set(expected_params):
        raise SupportDevelopmentError("candidate set differs from registry")
    for agent in agents:
        if agent.get("kind") != "support_estimator_gate":
            raise SupportDevelopmentError("candidate agent kind differs")
        params = agent.get("params", {})
        for key, expected in expected_params[agent["name"]].items():
            observed = params.get(key)
            if isinstance(expected, float):
                observed = float(observed)
            elif isinstance(expected, int):
                observed = int(observed)
            if observed != expected:
                raise SupportDevelopmentError(
                    f"{agent['name']} parameter {key} differs from registry"
                )


def audit_execution(
    raw_path: Path, config: dict[str, Any], registry: dict[str, Any]
) -> None:
    raw = pd.read_csv(raw_path, low_memory=False)
    candidates = {item["candidate_id"] for item in flatten_candidates(registry)}
    allowed_seeds = set(int(seed) for seed in registry["seed_lock"]["development_seeds"])
    observed_seeds = set(pd.to_numeric(raw["seed"], errors="raise").astype(int))
    if observed_seeds != allowed_seeds:
        raise SupportDevelopmentError("executed seed set differs from lock")
    if set(raw["agent"]) != candidates:
        raise SupportDevelopmentError("executed candidate coverage mismatch")
    if not raw["execution_inputs_clean"].astype(str).str.lower().eq("true").all():
        raise SupportDevelopmentError("execution inputs were not clean")
    evaluation = raw[raw["phase"] == "eval"].copy()
    numeric = evaluation[["checkpoint", "eval_return", "success"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise SupportDevelopmentError("non-finite evaluation outcome")
    budget = registry["matched_budget"]
    expected_checkpoints = set(
        range(
            int(budget["checkpoint_first"]),
            int(budget["checkpoint_last"]) + 1,
            int(budget["checkpoint_interval"]),
        )
    )
    keys = ["environment", "agent", "seed"]
    if evaluation.groupby(keys).ngroups != (
        int(budget["total_candidate_count"])
        * int(budget["environments_per_candidate"])
        * int(budget["seeds_per_environment"])
    ):
        raise SupportDevelopmentError("completed run count mismatch")
    for key, frame in evaluation.groupby(keys, sort=False):
        if set(frame["checkpoint"].astype(int)) != expected_checkpoints:
            raise SupportDevelopmentError(f"checkpoint coverage mismatch for {key}")
        counts = frame.groupby("checkpoint").size()
        if not counts.eq(int(budget["evaluation_episodes_per_checkpoint"])).all():
            raise SupportDevelopmentError(f"episode budget mismatch for {key}")
        if frame.duplicated(["checkpoint", "episode"]).any():
            raise SupportDevelopmentError(f"duplicate evaluation row for {key}")


def deterministic_gzip_copy(source: Path) -> Path:
    destination = source.with_suffix(source.suffix + ".gz")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output_handle, mtime=0
        ) as compressed:
            while chunk := input_handle.read(1024 * 1024):
                compressed.write(chunk)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked matched support-estimator development screen."
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    registry_digest = verify_registry_digest()
    registry = load_yaml(DEFAULT_REGISTRY)
    seed_registry = load_yaml(DEFAULT_SEED_REGISTRY)
    config = load_yaml(DEFAULT_CONFIG)
    validate_registry(registry, seed_registry)
    validate_config(config, registry)
    if args.validate_only:
        print(
            "SUPPORT_DEVELOPMENT_LOCK_PASS "
            f"registry_sha256={registry_digest} candidates=20 final_seed_rows=0"
        )
        return
    raw_path = run_config(DEFAULT_CONFIG)
    audit_execution(raw_path, config, registry)
    compressed = deterministic_gzip_copy(raw_path)
    print(
        "SUPPORT_DEVELOPMENT_PASS "
        "runs=200 seeds=13000-13004 final_seed_rows=0 "
        f"raw={compressed}"
    )


if __name__ == "__main__":
    main()
