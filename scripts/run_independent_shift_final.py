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

from hybrid_q.envs import make_env  # noqa: E402
from hybrid_q.experiment import run_config  # noqa: E402


PROTOCOL = ROOT / "configs/diagnostic_extensions/INDEPENDENT_SHIFT_PROTOCOL.yaml"
CONFIG_ROOT = ROOT / "configs/diagnostic_extensions/final_shifts"
CONFIGS = {
    "transition_dynamics_shift": CONFIG_ROOT / "transition_dynamics_shift.yaml",
    "observation_shift": CONFIG_ROOT / "observation_shift.yaml",
    "localized_multistep_reward_or_policy_shift": CONFIG_ROOT
    / "localized_multistep_reward_or_policy_shift.yaml",
}
OUTPUT_ROOT = ROOT / "results/diagnostic_extensions/final_shifts/execution"
EXPECTED = {
    "transition_dynamics_shift": {
        "severity_id": "transition_slip_035",
        "environment_id": "TransitionDynamicsShift-v0",
        "seeds": list(range(12000, 12030)),
        "output": "transition_dynamics_shift",
        "max_steps": 160,
        "kwargs": {
            "size": 9,
            "goal_split": "all",
            "max_steps": 160,
            "shift_after": 12000,
            "pre_shift_slip_probability": 0.05,
            "post_shift_slip_probability": 0.35,
        },
    },
    "observation_shift": {
        "severity_id": "observation_gain_085",
        "environment_id": "ObservationShift-v0",
        "seeds": list(range(12030, 12060)),
        "output": "observation_shift",
        "max_steps": 160,
        "kwargs": {
            "size": 9,
            "goal_split": "all",
            "max_steps": 160,
            "shift_after": 12000,
            "slip_probability": 0.10,
            "pre_shift_sensor_gain": 1.0,
            "post_shift_sensor_gain": 0.85,
        },
    },
    "localized_multistep_reward_or_policy_shift": {
        "severity_id": "localized_risk_penalty_064",
        "environment_id": "LocalizedRewardShift-v0",
        "seeds": list(range(12060, 12090)),
        "output": "localized_multistep_reward_or_policy_shift",
        "max_steps": 120,
        "kwargs": {
            "size": 9,
            "goal_split": "deployment",
            "max_steps": 120,
            "shift_after": 12000,
            "slip_probability": 0.05,
            "pre_shift_risk_penalty": 0.08,
            "post_shift_risk_penalty": 0.64,
        },
    },
}
EXPECTED_AGENTS = {
    "tabular",
    "dqn",
    "count_gated_tau_20",
    "relative_reliability_fuzzy",
    "same_input_crisp",
}
FINAL_MIN = 12000
FINAL_MAX = 12089


class FinalShiftError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalShiftError(f"YAML root must be a mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise FinalShiftError(f"missing SHA-256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = file_sha256(path)
    if expected != observed:
        raise FinalShiftError(f"SHA-256 mismatch: {path}")
    return observed


def validate_prerequisites(protocol: dict[str, Any]) -> None:
    verify_sidecar(PROTOCOL)
    if protocol.get("lock_status") != "finalized_locked_design":
        raise FinalShiftError("independent-shift protocol is not finalized")
    finalized = protocol.get("finalized_selection", {})
    expected_selected = {
        mechanism: values["severity_id"] for mechanism, values in EXPECTED.items()
    }
    if finalized.get("selected_severities") != expected_selected:
        raise FinalShiftError("finalized severity selection mismatch")
    if finalized.get("relative_reliability_fuzzy", {}).get("candidate_id") != "fuzzy_triangular_balanced":
        raise FinalShiftError("finalized fuzzy mapping mismatch")
    if finalized.get("same_input_crisp", {}).get("candidate_id") != "crisp_s070_r030":
        raise FinalShiftError("finalized crisp mapping mismatch")
    if finalized.get("final_seed_results_used_for_selection") is not False:
        raise FinalShiftError("final outcomes must not enter development selection")
    locked_mechanisms = {item["mechanism_id"] for item in protocol.get("mechanisms", [])}
    if locked_mechanisms != set(EXPECTED):
        raise FinalShiftError("final mechanism set differs from protocol")


def _assert_param(params: dict[str, Any], key: str, expected: Any, agent: str) -> None:
    observed = params.get(key)
    if isinstance(expected, float):
        try:
            equal = float(observed) == expected
        except (TypeError, ValueError):
            equal = False
    else:
        equal = observed == expected
    if not equal:
        raise FinalShiftError(f"{agent} parameter {key} differs from final lock")


def validate_final_config(config: dict[str, Any], mechanism: str) -> None:
    expected = EXPECTED[mechanism]
    expected_output = (
        f"results/diagnostic_extensions/final_shifts/execution/{expected['output']}"
    )
    if config.get("output_dir") != expected_output:
        raise FinalShiftError(f"unexpected output directory for {mechanism}")
    analysis = config.get("analysis", {})
    required_analysis = {
        "evidence_class": "replication",
        "mechanism_id": mechanism,
        "severity_id": expected["severity_id"],
        "primary_metric": "normalized_return_auc",
        "report_level_holm_family": "independent_shift_primary",
        "generator_pooling": "prohibited",
    }
    for key, value in required_analysis.items():
        if analysis.get(key) != value:
            raise FinalShiftError(f"analysis field {key} differs for {mechanism}")
    contrasts = analysis.get("planned_contrasts", [])
    observed_contrasts = [
        (row.get("name"), row.get("left"), row.get("right"), row.get("metric"))
        for row in contrasts
    ]
    expected_contrasts = [
        (
            "relative_reliability_fuzzy_vs_count_gated_tau_20",
            "relative_reliability_fuzzy",
            "count_gated_tau_20",
            "normalized_return_auc",
        ),
        (
            "same_input_crisp_vs_relative_reliability_fuzzy",
            "same_input_crisp",
            "relative_reliability_fuzzy",
            "normalized_return_auc",
        ),
    ]
    if observed_contrasts != expected_contrasts:
        raise FinalShiftError(f"planned contrasts differ for {mechanism}")
    evaluation = config.get("evaluation", {})
    if int(evaluation.get("interval_steps", -1)) != 1000:
        raise FinalShiftError("checkpoint interval differs from lock")
    if int(evaluation.get("episodes", -1)) != 200:
        raise FinalShiftError("evaluation episodes differ from lock")
    envs = config.get("envs", [])
    if len(envs) != 1:
        raise FinalShiftError(f"exactly one environment required for {mechanism}")
    env = envs[0]
    if env.get("mechanism_id") != mechanism:
        raise FinalShiftError("environment mechanism mismatch")
    if env.get("severity_id") != expected["severity_id"]:
        raise FinalShiftError("environment severity mismatch")
    if env.get("id") != expected["environment_id"]:
        raise FinalShiftError("environment ID mismatch")
    seeds = [int(seed) for seed in env.get("seeds", [])]
    if seeds != expected["seeds"] or len(set(seeds)) != 30:
        raise FinalShiftError(f"final seed set mismatch for {mechanism}")
    if int(env.get("training_steps", -1)) != 24000:
        raise FinalShiftError("training interaction budget differs from lock")
    if int(env.get("max_steps", -1)) != expected["max_steps"]:
        raise FinalShiftError("episode horizon differs from lock")
    if env.get("kwargs") != expected["kwargs"]:
        raise FinalShiftError(f"environment parameters differ for {mechanism}")
    instance = make_env(env)
    instance.close()

    agents = {item.get("name"): item for item in config.get("agents", [])}
    if set(agents) != EXPECTED_AGENTS or len(agents) != 5:
        raise FinalShiftError("required final agent set mismatch")
    expected_kinds = {
        "tabular": "tabular",
        "dqn": "dqn",
        "count_gated_tau_20": "count_gated",
        "relative_reliability_fuzzy": "fuzzy_reliability_gate",
        "same_input_crisp": "fuzzy_reliability_gate",
    }
    for name, kind in expected_kinds.items():
        if agents[name].get("kind") != kind:
            raise FinalShiftError(f"agent kind mismatch: {name}")
    common = {
        "double_dqn": False,
        "learning_rate": 0.0003,
        "tabular_learning_rate": 0.2,
        "batch_size": 32,
        "replay_capacity": 100000,
        "replay_warmup": 512,
        "train_frequency": 16,
        "target_update_interval": 500,
        "hidden_size": 128,
        "tau": 20,
        "fuzzy_tau_support": 20,
        "reliability_beta": 0.05,
        "reliability_prior_strength": 5,
        "reliability_epsilon": 1.0e-8,
        "epsilon_decay_steps": 20000,
    }
    for name in (
        "count_gated_tau_20",
        "relative_reliability_fuzzy",
        "same_input_crisp",
    ):
        for key, value in common.items():
            _assert_param(agents[name]["params"], key, value, name)
    fuzzy_expected = {
        "gate_min": 0.05,
        "gate_max": 0.95,
        "fuzzy_fallback_risk_mode": "disabled",
        "fuzzy_risk_ablation_mode": "full",
        "fuzzy_membership_shape": "triangular",
        "fuzzy_reliability_membership_shape": "triangular",
        "fuzzy_support_breakpoints": [0.0, 0.5, 1.0],
        "fuzzy_reliability_breakpoints": [0.0, 0.5, 1.0],
        "fuzzy_reliability_consequents": [0.0, 0.2, 0.7, 0.1, 0.95],
    }
    crisp_expected = {
        "gate_min": 0.05,
        "gate_max": 0.95,
        "fuzzy_fallback_risk_mode": "disabled",
        "fuzzy_risk_ablation_mode": "crisp_threshold",
        "fuzzy_crisp_support_threshold": 0.70,
        "fuzzy_crisp_reliability_threshold": 0.30,
    }
    for key, value in fuzzy_expected.items():
        _assert_param(agents["relative_reliability_fuzzy"]["params"], key, value, "relative_reliability_fuzzy")
    for key, value in crisp_expected.items():
        _assert_param(agents["same_input_crisp"]["params"], key, value, "same_input_crisp")


def assert_output_empty(mechanisms: list[str]) -> None:
    for mechanism in mechanisms:
        directory = OUTPUT_ROOT / EXPECTED[mechanism]["output"]
        if directory.exists() and any(directory.iterdir()):
            raise FinalShiftError(f"final output directory is not empty: {directory}")


def audit_execution(raw_path: Path, config: dict[str, Any], mechanism: str) -> None:
    expected = EXPECTED[mechanism]
    env = config["envs"][0]
    runs_dir = raw_path.parent / "runs"
    shards = sorted(runs_dir.glob("*.csv"))
    if len(shards) != 150:
        raise FinalShiftError(f"{mechanism}: expected 150 completed shards")
    if any(runs_dir.glob("*.tmp")):
        raise FinalShiftError(f"{mechanism}: incomplete temporary shards remain")
    observed_runs: set[tuple[str, int]] = set()
    expected_checkpoints = set(range(1000, 24001, 1000))
    commits: set[str] = set()
    snapshots: set[str] = set()
    for shard in shards:
        columns = [
            "environment",
            "agent",
            "seed",
            "phase",
            "checkpoint",
            "episode",
            "return",
            "success",
            "environment_steps",
            "nonfinite_loss_count",
            "completed_checkpoint_count",
            "expected_checkpoint_count",
            "git_commit_hash",
            "source_snapshot_sha256",
        ]
        frame = pd.read_csv(shard, usecols=columns)
        agents = set(frame["agent"].astype(str))
        seeds = set(frame["seed"].astype(int))
        environments = set(frame["environment"].astype(str))
        if len(agents) != 1 or len(seeds) != 1 or len(environments) != 1:
            raise FinalShiftError(f"mixed identity in shard: {shard}")
        agent = next(iter(agents))
        seed = next(iter(seeds))
        if agent not in EXPECTED_AGENTS or seed not in expected["seeds"]:
            raise FinalShiftError(f"unexpected agent/seed in shard: {shard}")
        run = (agent, seed)
        if run in observed_runs:
            raise FinalShiftError(f"duplicate agent-seed shard: {run}")
        observed_runs.add(run)
        evaluation = frame[frame["phase"] == "eval"].copy()
        if evaluation.duplicated(["checkpoint", "episode"]).any():
            raise FinalShiftError(f"duplicate evaluation rows: {shard}")
        if set(evaluation["checkpoint"].astype(int)) != expected_checkpoints:
            raise FinalShiftError(f"checkpoint coverage mismatch: {shard}")
        if not evaluation.groupby("checkpoint").size().eq(200).all():
            raise FinalShiftError(f"evaluation episode coverage mismatch: {shard}")
        for column in ("return", "success", "environment_steps"):
            values = pd.to_numeric(evaluation[column], errors="coerce").to_numpy()
            if not np.isfinite(values).all():
                raise FinalShiftError(f"non-finite {column} in {shard}")
        nonfinite_losses = pd.to_numeric(
            frame["nonfinite_loss_count"], errors="coerce"
        ).dropna()
        if len(nonfinite_losses) and (nonfinite_losses != 0).any():
            raise FinalShiftError(f"non-finite logged loss in {shard}")
        final_rows = evaluation[evaluation["checkpoint"] == 24000]
        if not pd.to_numeric(final_rows["completed_checkpoint_count"], errors="coerce").eq(24).all():
            raise FinalShiftError(f"completed checkpoint count mismatch: {shard}")
        if not pd.to_numeric(final_rows["expected_checkpoint_count"], errors="coerce").eq(24).all():
            raise FinalShiftError(f"expected checkpoint count mismatch: {shard}")
        commits.update(frame["git_commit_hash"].dropna().astype(str))
        snapshots.update(frame["source_snapshot_sha256"].dropna().astype(str))
    expected_runs = {(agent, seed) for agent in EXPECTED_AGENTS for seed in expected["seeds"]}
    if observed_runs != expected_runs:
        raise FinalShiftError(f"agent-seed coverage mismatch for {mechanism}")
    if len(commits) != 1 or len(snapshots) != 1:
        raise FinalShiftError(f"mixed source provenance for {mechanism}")


def deterministic_gzip_copy(source: Path) -> Path:
    destination = source.with_suffix(source.suffix + ".gz")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_handle, mtime=0) as compressed:
            while chunk := input_handle.read(1024 * 1024):
                compressed.write(chunk)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the finalized independent-shift experiment family.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-empty", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--mechanism", choices=[*CONFIGS, "all"], default="all")
    args = parser.parse_args()
    protocol = load_yaml(PROTOCOL)
    validate_prerequisites(protocol)
    mechanisms = list(CONFIGS) if args.mechanism == "all" else [args.mechanism]
    configs = {mechanism: load_yaml(CONFIGS[mechanism]) for mechanism in mechanisms}
    for mechanism, config in configs.items():
        validate_final_config(config, mechanism)
    if args.require_empty:
        assert_output_empty(mechanisms)
    if args.validate_only:
        print(
            "INDEPENDENT_SHIFT_FINAL_LOCK_PASS "
            f"mechanisms={len(mechanisms)} final_seeds={sum(len(EXPECTED[m]['seeds']) for m in mechanisms)}"
        )
        return
    for mechanism in mechanisms:
        config = configs[mechanism]
        raw_path = Path(config["output_dir"]) / "raw.csv"
        if not args.audit_only:
            raw_path = run_config(CONFIGS[mechanism])
        if not raw_path.exists():
            raise FinalShiftError(f"missing final raw output: {raw_path}")
        audit_execution(raw_path, config, mechanism)
        deterministic_gzip_copy(raw_path)
        print(f"INDEPENDENT_SHIFT_FINAL_MECHANISM_PASS mechanism={mechanism} runs=150")
    print(f"INDEPENDENT_SHIFT_FINAL_PASS mechanisms={len(mechanisms)}")


if __name__ == "__main__":
    main()
