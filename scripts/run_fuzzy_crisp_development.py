from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.experiment import run_config  # noqa: E402


PROTOCOL = ROOT / "configs/diagnostic_extensions/INDEPENDENT_SHIFT_PROTOCOL.yaml"
PROTOCOL_DIGEST = PROTOCOL.with_suffix(".sha256")
DEFAULT_REGISTRY = ROOT / "configs/fuzzy_crisp_candidate_registry.yaml"
DEFAULT_REGISTRY_DIGEST = DEFAULT_REGISTRY.with_suffix(".sha256")
DEFAULT_CONFIG = ROOT / "configs/diagnostic_extensions/fuzzy_crisp_development.yaml"
FINAL_SEED_MIN = 12000
FINAL_SEED_MAX = 12099


class FuzzyCrispDevelopmentError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FuzzyCrispDevelopmentError(f"YAML root must be a mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_registry_digest(
    registry_path: Path = DEFAULT_REGISTRY,
    digest_path: Path = DEFAULT_REGISTRY_DIGEST,
) -> str:
    if not digest_path.exists():
        raise FuzzyCrispDevelopmentError(
            f"missing candidate registry digest: {digest_path}"
        )
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = file_sha256(registry_path)
    if expected != observed:
        raise FuzzyCrispDevelopmentError("candidate registry digest mismatch")
    return observed


def candidate_family(candidate_id: str, registry: dict[str, Any]) -> str:
    crisp = {item["candidate_id"] for item in registry["crisp_candidates"]}
    fuzzy = {item["candidate_id"] for item in registry["fuzzy_candidates"]}
    if candidate_id in crisp:
        return "crisp"
    if candidate_id in fuzzy:
        return "fuzzy"
    raise FuzzyCrispDevelopmentError(f"unknown candidate: {candidate_id}")


def validate_registry(registry: dict[str, Any], protocol: dict[str, Any]) -> None:
    if registry.get("status") != "locked_before_development_outcomes":
        raise FuzzyCrispDevelopmentError("candidate registry is not outcome-locked")
    if registry.get("evidence_class") != "development":
        raise FuzzyCrispDevelopmentError("registry must be development evidence")
    boundary = registry.get("protocol_boundary", {})
    if boundary.get("final_seed_access") != "prohibited":
        raise FuzzyCrispDevelopmentError("final seed access must be prohibited")

    crisp = registry.get("crisp_candidates", [])
    fuzzy = registry.get("fuzzy_candidates", [])
    expected = int(registry["matched_budget"]["candidate_count_per_family"])
    if len(crisp) != len(fuzzy) or len(crisp) != expected:
        raise FuzzyCrispDevelopmentError("fuzzy/crisp candidate counts are unequal")
    ids = [item.get("candidate_id") for item in [*crisp, *fuzzy]]
    if None in ids or len(ids) != len(set(ids)):
        raise FuzzyCrispDevelopmentError("candidate IDs are missing or duplicated")
    support_thresholds = [float(item["support_threshold"]) for item in crisp]
    reliability_thresholds = [float(item["reliability_threshold"]) for item in crisp]
    if min(support_thresholds) > 0.30 or max(support_thresholds) < 0.70:
        raise FuzzyCrispDevelopmentError(
            "crisp support thresholds do not span [0.30, 0.70]"
        )
    if min(reliability_thresholds) > 0.30 or max(reliability_thresholds) < 0.70:
        raise FuzzyCrispDevelopmentError(
            "crisp reliability thresholds do not span [0.30, 0.70]"
        )
    shapes = {item.get("membership_family") for item in fuzzy}
    if shapes != {"triangular", "shoulder"}:
        raise FuzzyCrispDevelopmentError(
            "fuzzy candidates must include triangular and shoulder families"
        )
    common = registry["common_mapping_inputs"]
    expected_bounds = (float(common["gate_min"]), float(common["gate_max"]))
    for item in [*crisp, *fuzzy]:
        if (float(item["gate_min"]), float(item["gate_max"])) != expected_bounds:
            raise FuzzyCrispDevelopmentError("candidate gate bounds are not matched")
    for item in fuzzy:
        for key in ("support_breakpoints", "reliability_breakpoints"):
            points = [float(value) for value in item.get(key, [])]
            if len(points) != 3 or not (
                0.0 <= points[0] < points[1] < points[2] <= 1.0
            ):
                raise FuzzyCrispDevelopmentError(
                    f"invalid {key} for {item['candidate_id']}"
                )
        if len(item.get("consequents", [])) != 5:
            raise FuzzyCrispDevelopmentError(
                f"invalid consequents for {item['candidate_id']}"
            )

    locked_by_id = {item["mechanism_id"]: item for item in protocol["mechanisms"]}
    registry_envs = registry.get("environments", [])
    if set(item.get("mechanism_id") for item in registry_envs) != set(locked_by_id):
        raise FuzzyCrispDevelopmentError(
            "registry environments differ from the locked mechanisms"
        )
    for item in registry_envs:
        locked = locked_by_id[item["mechanism_id"]]
        development = locked["development_seeds"]
        expected_seeds = list(
            range(int(development["start"]), int(development["end"]) + 1)
        )
        seeds = [int(seed) for seed in item.get("development_seeds", [])]
        if seeds != expected_seeds:
            raise FuzzyCrispDevelopmentError(
                f"development seed mismatch for {item['mechanism_id']}"
            )
        if any(FINAL_SEED_MIN <= seed <= FINAL_SEED_MAX for seed in seeds):
            raise FuzzyCrispDevelopmentError("reserved final seed detected")
        first_severity = locked["development_severity_candidates"][0]
        if item.get("severity_id") != first_severity["severity_id"]:
            raise FuzzyCrispDevelopmentError(
                "development severity is not the predeclared first candidate"
            )


def _registry_candidate_params(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in registry["crisp_candidates"]:
        result[item["candidate_id"]] = {
            "fuzzy_risk_ablation_mode": "crisp_threshold",
            "fuzzy_crisp_support_threshold": float(item["support_threshold"]),
            "fuzzy_crisp_reliability_threshold": float(item["reliability_threshold"]),
            "gate_min": float(item["gate_min"]),
            "gate_max": float(item["gate_max"]),
        }
    for item in registry["fuzzy_candidates"]:
        result[item["candidate_id"]] = {
            "fuzzy_risk_ablation_mode": "full",
            "fuzzy_membership_shape": item["membership_family"],
            "fuzzy_reliability_membership_shape": item["membership_family"],
            "fuzzy_support_breakpoints": [
                float(value) for value in item["support_breakpoints"]
            ],
            "fuzzy_reliability_breakpoints": [
                float(value) for value in item["reliability_breakpoints"]
            ],
            "fuzzy_reliability_consequents": [
                float(value) for value in item["consequents"]
            ],
            "gate_min": float(item["gate_min"]),
            "gate_max": float(item["gate_max"]),
        }
    return result


def validate_config(config: dict[str, Any], registry: dict[str, Any]) -> None:
    if config.get("analysis", {}).get("evidence_class") != "development":
        raise FuzzyCrispDevelopmentError("config is not development-only")
    if (
        config.get("output_dir")
        != "results/diagnostic_extensions/fuzzy_crisp_development/execution_corrected"
    ):
        raise FuzzyCrispDevelopmentError("unexpected development output directory")
    budget = registry["matched_budget"]
    evaluation = config.get("evaluation", {})
    if int(evaluation.get("interval_steps", -1)) != int(budget["checkpoint_interval"]):
        raise FuzzyCrispDevelopmentError("checkpoint interval differs from registry")
    if int(evaluation.get("episodes", -1)) != int(
        budget["evaluation_episodes_per_checkpoint"]
    ):
        raise FuzzyCrispDevelopmentError(
            "evaluation episode budget differs from registry"
        )
    expected_envs = {
        item["environment_name"]: item for item in registry["environments"]
    }
    if {item.get("name") for item in config.get("envs", [])} != set(expected_envs):
        raise FuzzyCrispDevelopmentError("config environment set differs from registry")
    for env in config["envs"]:
        declared = expected_envs[env["name"]]
        if env.get("id") != declared["environment_id"]:
            raise FuzzyCrispDevelopmentError("environment ID differs from registry")
        if [int(seed) for seed in env.get("seeds", [])] != declared[
            "development_seeds"
        ]:
            raise FuzzyCrispDevelopmentError(
                "environment seed set differs from registry"
            )
        if int(env.get("training_steps", -1)) != int(
            budget["training_steps_per_agent_seed"]
        ):
            raise FuzzyCrispDevelopmentError("training budget differs from registry")
        for key, value in declared["severity_kwargs"].items():
            if env.get("kwargs", {}).get(key) != value:
                raise FuzzyCrispDevelopmentError(
                    f"severity field {key} differs from registry"
                )
        if int(env.get("kwargs", {}).get("shift_after", -1)) != int(
            budget["shift_onset"]
        ):
            raise FuzzyCrispDevelopmentError("shift onset differs from registry")

    expected_params = _registry_candidate_params(registry)
    agents = config.get("agents", [])
    if {item.get("name") for item in agents} != set(expected_params):
        raise FuzzyCrispDevelopmentError("config candidate set differs from registry")
    for agent in agents:
        if agent.get("kind") != "fuzzy_reliability_gate":
            raise FuzzyCrispDevelopmentError(
                "candidate kind must be fuzzy_reliability_gate"
            )
        params = agent.get("params", {})
        if params.get("fuzzy_fallback_risk_mode") != "disabled":
            raise FuzzyCrispDevelopmentError(
                "fallback-risk mapping must be disabled for mapping isolation"
            )
        for key, expected in expected_params[agent["name"]].items():
            observed = params.get(key)
            if isinstance(expected, list):
                observed = [float(value) for value in observed]
            elif isinstance(expected, float):
                observed = float(observed)
            if observed != expected:
                raise FuzzyCrispDevelopmentError(
                    f"{agent['name']} parameter {key} differs from registry"
                )


def audit_execution(
    raw_path: Path, config: dict[str, Any], registry: dict[str, Any]
) -> None:
    raw = pd.read_csv(raw_path)
    observed_seeds = set(raw["seed"].astype(int))
    if any(FINAL_SEED_MIN <= seed <= FINAL_SEED_MAX for seed in observed_seeds):
        raise FuzzyCrispDevelopmentError("reserved final result row detected")
    expected_candidates = {
        item["candidate_id"]
        for item in [*registry["crisp_candidates"], *registry["fuzzy_candidates"]]
    }
    if set(raw["agent"]) != expected_candidates:
        raise FuzzyCrispDevelopmentError("executed candidate coverage mismatch")
    budget = registry["matched_budget"]
    expected_checkpoints = set(
        range(
            int(budget["checkpoint_first"]),
            int(budget["checkpoint_last"]) + 1,
            int(budget["checkpoint_interval"]),
        )
    )
    evaluation = raw[raw["phase"] == "eval"].copy()
    env_by_name = {item["environment_name"]: item for item in registry["environments"]}
    for environment, frame in raw.groupby("environment", sort=False):
        expected_seeds = set(env_by_name[environment]["development_seeds"])
        if set(frame["seed"].astype(int)) != expected_seeds:
            raise FuzzyCrispDevelopmentError(
                f"seed coverage mismatch for {environment}"
            )
    keys = ["environment", "agent", "seed"]
    for key, frame in evaluation.groupby(keys, sort=False):
        if set(frame["checkpoint"].astype(int)) != expected_checkpoints:
            raise FuzzyCrispDevelopmentError(f"checkpoint coverage mismatch for {key}")
        counts = frame.groupby("checkpoint").size()
        if not counts.eq(int(budget["evaluation_episodes_per_checkpoint"])).all():
            raise FuzzyCrispDevelopmentError(f"evaluation episode mismatch for {key}")
    expected_runs = len(expected_candidates) * sum(
        len(item["development_seeds"]) for item in registry["environments"]
    )
    if evaluation.groupby(keys).ngroups != expected_runs:
        raise FuzzyCrispDevelopmentError("completed run count mismatch")


def deterministic_gzip_copy(source: Path) -> Path:
    """Write a byte-reproducible gzip copy without changing the source CSV."""

    destination = source.with_suffix(source.suffix + ".gz")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=output_handle,
            mtime=0,
        ) as compressed:
            while chunk := input_handle.read(1024 * 1024):
                compressed.write(chunk)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked matched fuzzy/crisp development screen."
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol = _load_yaml(PROTOCOL)
    verify_registry_digest(PROTOCOL, PROTOCOL_DIGEST)
    digest = verify_registry_digest()
    registry = _load_yaml(DEFAULT_REGISTRY)
    config = _load_yaml(DEFAULT_CONFIG)
    validate_registry(registry, protocol)
    validate_config(config, registry)
    if args.validate_only:
        print(f"FUZZY_CRISP_DEVELOPMENT_LOCK_PASS registry_sha256={digest}")
        return
    raw_path = run_config(DEFAULT_CONFIG)
    audit_execution(raw_path, config, registry)
    compressed_path = deterministic_gzip_copy(raw_path)
    print(
        "FUZZY_CRISP_DEVELOPMENT_PASS "
        f"runs={8 * 30} final_seed_rows=0 raw={compressed_path}"
    )


if __name__ == "__main__":
    main()
