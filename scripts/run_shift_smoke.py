from __future__ import annotations

import json
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
from scripts.lock_protocol import (  # noqa: E402
    DEFAULT_COMPANION,
    DEFAULT_DIGEST,
    DEFAULT_PROTOCOL,
    DEFAULT_SEED_REGISTRY,
    load_and_validate,
    verify_or_write_digest,
)


SMOKE_CONFIGS = (
    ROOT
    / "configs/diagnostic_extensions/smoke/transition_dynamics_shift.yaml",
    ROOT / "configs/diagnostic_extensions/smoke/observation_shift.yaml",
    ROOT / "configs/diagnostic_extensions/smoke/localized_reward_shift.yaml",
)
EXPECTED_IDS = {
    "transition_dynamics_shift": "TransitionDynamicsShift-v0",
    "observation_shift": "ObservationShift-v0",
    "localized_multistep_reward_or_policy_shift": (
        "LocalizedRewardShift-v0"
    ),
}
EXPECTED_AGENTS = {
    "relative_reliability_fuzzy",
    "count_gated_tau_20",
    "same_input_crisp",
}


class ShiftSmokeError(ValueError):
    pass


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ShiftSmokeError(f"invalid smoke config: {path}")
    return config


def validate_smoke_config(
    config: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    analysis = config.get("analysis", {})
    mechanism_id = analysis.get("mechanism_id")
    locked = {
        mechanism["mechanism_id"]: mechanism
        for mechanism in protocol["mechanisms"]
    }.get(mechanism_id)
    if locked is None:
        raise ShiftSmokeError(f"unknown mechanism_id: {mechanism_id}")
    if analysis.get("analysis_status") != (
        "development_smoke_only_not_severity_selection"
    ):
        raise ShiftSmokeError("smoke analysis status is not development-only")
    if analysis.get("primary_metric") != "normalized_return_auc":
        raise ShiftSmokeError("smoke primary metric differs from lock")

    dev = locked["development_seeds"]
    expected_seeds = [int(dev["start"]), int(dev["start"]) + 1]
    seeds = [int(seed) for seed in config.get("seeds", [])]
    final = protocol["seed_registry"]["final_registered_range"]
    final_start, final_end = (int(item) for item in final.split("-"))
    if any(final_start <= seed <= final_end for seed in seeds):
        raise ShiftSmokeError("reserved final seed detected")
    if seeds != expected_seeds:
        raise ShiftSmokeError(
            f"{mechanism_id} smoke seeds must be {expected_seeds}"
        )

    output_dir = Path(config.get("output_dir", "")).as_posix()
    if "/smoke/" not in f"/{output_dir}/" or "/final/" in f"/{output_dir}/":
        raise ShiftSmokeError("smoke output must use the smoke-only root")
    if config.get("runtime", {}).get("allow_dirty_execution_inputs"):
        raise ShiftSmokeError("smoke execution must use committed clean inputs")

    envs = config.get("envs", [])
    if len(envs) != 1:
        raise ShiftSmokeError("each smoke config must contain one mechanism")
    env = envs[0]
    if env.get("id") != EXPECTED_IDS[mechanism_id]:
        raise ShiftSmokeError(f"{mechanism_id} uses an unexpected environment ID")
    onset = int(locked["shift_onset"]["training_interaction"])
    if int(env.get("kwargs", {}).get("shift_after", -1)) != onset:
        raise ShiftSmokeError(f"{mechanism_id} shift onset differs from lock")
    training_steps = int(env.get("training_steps", 0))
    if not onset < training_steps <= int(
        locked["training_interaction_budget"]["per_agent_seed"]
    ):
        raise ShiftSmokeError("smoke must cross onset without exceeding budget")

    severity = locked["development_severity_candidates"][0]
    severity_fields = {
        key: value for key, value in severity.items() if key != "severity_id"
    }
    for key, value in severity_fields.items():
        if env.get("kwargs", {}).get(key) != value:
            raise ShiftSmokeError(
                f"{mechanism_id} smoke severity {key} differs from lock"
            )
    agents = {agent.get("name") for agent in config.get("agents", [])}
    if agents != EXPECTED_AGENTS:
        raise ShiftSmokeError(f"{mechanism_id} agent set differs from lock")
    return locked


def audit_smoke_output(
    raw_path: Path, config: dict[str, Any], locked: dict[str, Any]
) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    expected_seeds = {int(seed) for seed in config["seeds"]}
    if set(raw["seed"].astype(int)) != expected_seeds:
        raise ShiftSmokeError(f"seed mismatch in {raw_path}")
    final_start, final_end = 12000, 12099
    if raw["seed"].astype(int).between(final_start, final_end).any():
        raise ShiftSmokeError(f"reserved final row detected in {raw_path}")
    if raw[raw["phase"] == "eval"].empty:
        raise ShiftSmokeError(f"missing evaluation rows in {raw_path}")
    if int(raw["environment_steps"].max()) != int(
        config["envs"][0]["training_steps"]
    ):
        raise ShiftSmokeError(f"incomplete interaction budget in {raw_path}")
    evaluation = raw[raw["phase"] == "eval"].copy()
    observed_checkpoints = set(evaluation["checkpoint"].astype(int))
    expected_checkpoints = {6000, 12000, 12020}
    if observed_checkpoints != expected_checkpoints:
        raise ShiftSmokeError(
            f"checkpoint mismatch in {raw_path}: {observed_checkpoints}"
        )
    pre = evaluation[evaluation["checkpoint"] == 6000]["post_shift"]
    post = evaluation[evaluation["checkpoint"] >= 12000]["post_shift"]
    if not pre.eq(0.0).all() or not post.eq(1.0).all():
        raise ShiftSmokeError(f"shift regime mismatch in {raw_path}")
    expected_runs = len(expected_seeds) * len(config["agents"])
    observed_runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    if len(observed_runs) != expected_runs:
        raise ShiftSmokeError(f"run coverage mismatch in {raw_path}")
    return {
        "mechanism_id": locked["mechanism_id"],
        "config": Path(config["_config_path"]).relative_to(ROOT).as_posix(),
        "raw_file": raw_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "seeds": sorted(expected_seeds),
        "agents": sorted(EXPECTED_AGENTS),
        "completed_runs": expected_runs,
        "training_steps_per_run": int(config["envs"][0]["training_steps"]),
        "evaluation_checkpoints": sorted(expected_checkpoints),
        "final_seed_rows": 0,
        "status": "PASS",
    }


def main() -> None:
    protocol = load_and_validate(DEFAULT_PROTOCOL, DEFAULT_SEED_REGISTRY)
    verify_or_write_digest(DEFAULT_PROTOCOL, DEFAULT_COMPANION, DEFAULT_DIGEST)
    summaries = []
    for config_path in SMOKE_CONFIGS:
        config = _load_config(config_path)
        config["_config_path"] = str(config_path)
        locked = validate_smoke_config(config, protocol)
        public_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        # run_config reloads the checked-in file; private validation fields are
        # never injected into execution metadata.
        raw_path = run_config(config_path)
        summaries.append(audit_smoke_output(raw_path, config, locked))
        assert public_config == _load_config(config_path)

    manifest_path = (
        ROOT
        / "results/diagnostic_extensions/new_shift/smoke/smoke_run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_composite_sha256": (
                    DEFAULT_DIGEST.read_text(encoding="utf-8").split()[0]
                ),
                "development_smoke_only": True,
                "final_results_inspected": False,
                "mechanisms": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "SHIFT_SMOKE_PASS "
        f"mechanisms={len(summaries)} runs="
        f"{sum(item['completed_runs'] for item in summaries)}"
    )


if __name__ == "__main__":
    main()
