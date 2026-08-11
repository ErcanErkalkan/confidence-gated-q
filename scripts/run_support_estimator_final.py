from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for path in (ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_q.envs import make_env  # noqa: E402
from hybrid_q.experiment import run_config  # noqa: E402
from scripts.generate_support_final_configs import (  # noqa: E402
    CONFIG_ROOT,
    PROTOCOL,
    build_config,
    file_sha256,
    load_yaml,
    verify_protocol_digest,
)
from scripts.lock_protocol import (  # noqa: E402
    DEFAULT_COMPANION,
    DEFAULT_DIGEST,
    DEFAULT_PROTOCOL,
    verify_or_write_digest,
)
from scripts.run_fuzzy_crisp_development import (  # noqa: E402
    deterministic_gzip_copy,
)


SEED_REGISTRY = ROOT / "configs/diagnostic_extensions/seed_registry.yaml"
SELECTED = ROOT / "results/diagnostic_extensions/support_development/selected_estimators.yaml"
SELECTION_REPORT = ROOT / (
    "results/diagnostic_extensions/support_development/selected_estimators.yaml"
)
SEVERITY_SELECTION = ROOT / (
    "configs/diagnostic_extensions/selected_shift_severities.yaml"
)
OUTPUT_ROOT = ROOT / "results/diagnostic_extensions/support_final/execution"
SUPPORT_AGENT_IDS = {
    "exact_count_gate",
    "tolerance_knn_executed",
    "gaussian_affinity_executed",
    "raw_normalized_s1",
    "mahalanobis_s1",
    "frozen_embedding_s1",
    "gaussian_density_s3",
}
REFERENCE_AGENT_IDS = {"tabular", "dqn"}


class SupportFinalError(ValueError):
    pass


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise SupportFinalError(f"missing SHA-256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = file_sha256(path)
    if observed != expected:
        raise SupportFinalError(f"SHA-256 mismatch: {path}")
    return observed


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "locked_before_final_outcomes":
        raise SupportFinalError("support-final protocol is not outcome-locked")
    if protocol.get("evidence_class") != "replication":
        raise SupportFinalError("support-final evidence class must be replication")
    environments = protocol.get("environments", [])
    if len(environments) != 3:
        raise SupportFinalError("exactly three support-final environments required")
    all_seeds: list[int] = []
    for environment in environments:
        seeds = [int(seed) for seed in environment.get("seeds", [])]
        if len(seeds) != 30 or len(set(seeds)) != 30:
            raise SupportFinalError("each environment requires 30 unique seeds")
        if any(seed < 14000 or seed > 14099 for seed in seeds):
            raise SupportFinalError("support-final seed outside reserved range")
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != 90:
        raise SupportFinalError("environment final-seed blocks overlap")
    agent_ids = {str(item.get("agent_id")) for item in protocol.get("agents", [])}
    if agent_ids != SUPPORT_AGENT_IDS | REFERENCE_AGENT_IDS:
        raise SupportFinalError("support-final agent set differs from lock")
    contrasts = protocol.get("planned_contrasts", [])
    if len(contrasts) != 12 or len({item["name"] for item in contrasts}) != 12:
        raise SupportFinalError("support-final planned contrasts are invalid")
    allowed = agent_ids
    if any(item["left"] not in allowed or item["right"] not in allowed for item in contrasts):
        raise SupportFinalError("planned contrast references an unknown agent")
    if sum(item["status"] == "primary_replication" for item in contrasts) != 1:
        raise SupportFinalError("exactly one primary contrast per environment required")


def validate_prerequisites(protocol: dict[str, Any]) -> None:
    verify_protocol_digest()
    verify_or_write_digest(DEFAULT_PROTOCOL, DEFAULT_COMPANION, DEFAULT_DIGEST)
    selected_digest = _verify_sidecar(SELECTED)
    if selected_digest != protocol["prerequisites"]["selected_estimators_sha256"]:
        raise SupportFinalError("selected-estimator digest differs from protocol")
    if not SELECTION_REPORT.is_file():
        raise SupportFinalError("Step 10 selection report is missing")
    selected = load_yaml(SELECTED)
    if selected.get("final_seed_results_used") is not False:
        raise SupportFinalError("development selection accessed final seeds")
    if selected["overall_selected_estimator"]["candidate_id"] != "raw_normalized_s1":
        raise SupportFinalError("overall development selection differs")
    family = selected["within_family_descriptive_selections"]
    expected = {
        "mahalanobis": "mahalanobis_s1",
        "frozen_embedding": "frozen_embedding_s1",
        "gaussian_density": "gaussian_density_s3",
    }
    if {key: family[key]["candidate_id"] for key in expected} != expected:
        raise SupportFinalError("within-family development selection differs")
    severity_digest = _verify_sidecar(SEVERITY_SELECTION)
    if severity_digest != protocol["prerequisites"][
        "shift_severity_selection_sha256"
    ]:
        raise SupportFinalError("shift severity selection digest differs")
    seed_registry = load_yaml(SEED_REGISTRY)
    reservation = seed_registry["reserved_ranges"]["support_estimator_final"]
    if int(reservation["start"]) != 14000 or int(reservation["end"]) != 14099:
        raise SupportFinalError("support-final seed reservation differs")
    allocated = set(reservation.get("allocated_ranges", []))
    if allocated != {"14000-14029", "14030-14059", "14060-14089"}:
        raise SupportFinalError("support-final allocation differs from protocol")


def config_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {
        item["environment_key"]: CONFIG_ROOT / f"{item['environment_key']}.yaml"
        for item in protocol["environments"]
    }


def validate_configs(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configs = {}
    for environment in protocol["environments"]:
        key = str(environment["environment_key"])
        path = CONFIG_ROOT / f"{key}.yaml"
        if not path.exists():
            raise SupportFinalError(f"missing generated final config: {path}")
        config = load_yaml(path)
        if config != build_config(protocol, environment):
            raise SupportFinalError(f"generated config differs from protocol: {key}")
        instance = make_env(config["envs"][0])
        instance.close()
        configs[key] = config
    return configs


def assert_final_seeds_unused(root: Path = ROOT) -> None:
    hits: list[str] = []
    results = root / "results"
    output = results / "diagnostic_extensions/support_final"
    for metadata_path in results.rglob("metadata.json"):
        try:
            if metadata_path.is_relative_to(output):
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        config = metadata.get("config", metadata)
        declared = list(config.get("seeds", []))
        for environment in config.get("envs", []):
            declared.extend(environment.get("seeds", []))
        used = sorted({int(seed) for seed in declared if 14000 <= int(seed) <= 14099})
        if used:
            hits.append(f"{metadata_path}: {used}")
    seed_pattern = re.compile(r"(?:^|__)(14\d{3})(?:\.csv|__|$)")
    for path in results.rglob("*.csv"):
        try:
            if path.is_relative_to(output):
                continue
        except ValueError:
            pass
        match = seed_pattern.search(path.name)
        if match and 14000 <= int(match.group(1)) <= 14099:
            hits.append(str(path))
    if hits:
        raise SupportFinalError(
            "support-final seeds already appear in prior outputs: " + "; ".join(hits[:5])
        )


def assert_output_empty(keys: list[str]) -> None:
    for key in keys:
        directory = OUTPUT_ROOT / key
        if directory.exists() and any(directory.iterdir()):
            raise SupportFinalError(f"support-final output is not empty: {directory}")


def audit_execution(
    raw_path: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    agents = {item["agent_id"] for item in load_yaml(PROTOCOL)["agents"]}
    expected_seeds = {int(seed) for seed in environment["seeds"]}
    budget = load_yaml(PROTOCOL)["matched_budget"]
    runs_dir = raw_path.parent / "runs"
    shards = sorted(runs_dir.glob("*.csv"))
    expected_runs = len(agents) * len(expected_seeds)
    if len(shards) != expected_runs:
        raise SupportFinalError(
            f"{environment['environment_key']}: {len(shards)} shards != {expected_runs}"
        )
    if any(runs_dir.glob("*.tmp")):
        raise SupportFinalError("incomplete support-final temporary shard remains")
    checkpoints = set(
        range(
            int(budget["checkpoint_first"]),
            int(budget["checkpoint_last"]) + 1,
            int(budget["checkpoint_interval"]),
        )
    )
    observed: set[tuple[str, int]] = set()
    commits: set[str] = set()
    snapshots: set[str] = set()
    columns = [
        "environment",
        "agent",
        "seed",
        "phase",
        "checkpoint",
        "episode",
        "return",
        "success",
        "support_score_mean",
        "tabular_action_correct",
        "neural_action_correct",
        "nonfinite_loss_count",
        "completed_checkpoint_count",
        "expected_checkpoint_count",
        "execution_inputs_clean",
        "git_commit_hash",
        "source_snapshot_sha256",
    ]
    for shard in shards:
        frame = pd.read_csv(shard, usecols=columns, low_memory=False)
        identities = (
            set(frame["agent"].astype(str)),
            set(frame["seed"].astype(int)),
            set(frame["environment"].astype(str)),
        )
        if any(len(values) != 1 for values in identities):
            raise SupportFinalError(f"mixed run identity: {shard}")
        agent = next(iter(identities[0]))
        seed = next(iter(identities[1]))
        if agent not in agents or seed not in expected_seeds:
            raise SupportFinalError(f"unexpected agent or seed: {shard}")
        if (agent, seed) in observed:
            raise SupportFinalError(f"duplicate support-final run: {(agent, seed)}")
        observed.add((agent, seed))
        evaluation = frame[frame["phase"].eq("eval")].copy()
        if evaluation.duplicated(["checkpoint", "episode"]).any():
            raise SupportFinalError(f"duplicate evaluation row: {shard}")
        if set(evaluation["checkpoint"].astype(int)) != checkpoints:
            raise SupportFinalError(f"checkpoint coverage mismatch: {shard}")
        counts = evaluation.groupby("checkpoint").size()
        if not counts.eq(int(budget["evaluation_episodes_per_checkpoint"])).all():
            raise SupportFinalError(f"evaluation episode coverage mismatch: {shard}")
        outcomes = evaluation[["return", "success"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(outcomes.to_numpy(dtype=float)).all():
            raise SupportFinalError(f"non-finite evaluation outcome: {shard}")
        if agent in SUPPORT_AGENT_IDS:
            support = pd.to_numeric(
                evaluation["support_score_mean"], errors="coerce"
            )
            if support.isna().any() or ((support < 0) | (support > 1)).any():
                raise SupportFinalError(f"invalid support score: {shard}")
        nonfinite = pd.to_numeric(
            frame["nonfinite_loss_count"], errors="coerce"
        ).dropna()
        if len(nonfinite) and (nonfinite != 0).any():
            raise SupportFinalError(f"non-finite logged loss: {shard}")
        final = evaluation[evaluation["checkpoint"] == int(budget["checkpoint_last"])]
        if not pd.to_numeric(
            final["completed_checkpoint_count"], errors="coerce"
        ).eq(int(budget["expected_checkpoint_count"])).all():
            raise SupportFinalError(f"completed checkpoint mismatch: {shard}")
        if not pd.to_numeric(
            final["expected_checkpoint_count"], errors="coerce"
        ).eq(int(budget["expected_checkpoint_count"])).all():
            raise SupportFinalError(f"expected checkpoint mismatch: {shard}")
        if not frame["execution_inputs_clean"].astype(str).str.lower().eq("true").all():
            raise SupportFinalError(f"dirty execution provenance: {shard}")
        commits.update(frame["git_commit_hash"].dropna().astype(str))
        snapshots.update(frame["source_snapshot_sha256"].dropna().astype(str))
    expected = {(agent, seed) for agent in agents for seed in expected_seeds}
    if observed != expected:
        raise SupportFinalError("support-final run coverage differs from lock")
    if len(commits) != 1 or len(snapshots) != 1:
        raise SupportFinalError("mixed support-final source provenance")
    return {
        "environment_key": environment["environment_key"],
        "expected_runs": expected_runs,
        "observed_runs": len(observed),
        "excluded_seeds": 0,
        "execution_commit": next(iter(commits)),
        "source_snapshot_sha256": next(iter(snapshots)),
        "audit_status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked support-estimator final replication."
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-empty", action="store_true")
    parser.add_argument("--require-unused", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--environment",
        choices=[
            "application_goal_shift",
            "independent_observation_shift",
            "independent_transition_dynamics_shift",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    protocol = load_yaml(PROTOCOL)
    validate_protocol(protocol)
    validate_prerequisites(protocol)
    configs = validate_configs(protocol)
    keys = list(configs) if args.environment == "all" else [args.environment]
    if args.require_unused:
        assert_final_seeds_unused()
    if args.require_empty:
        assert_output_empty(keys)
    if args.validate_only:
        print(
            "SUPPORT_FINAL_LOCK_PASS "
            f"environments={len(keys)} agents=9 seeds={30 * len(keys)} "
            f"protocol_sha256={file_sha256(PROTOCOL)}"
        )
        return
    environment_map = {
        item["environment_key"]: item for item in protocol["environments"]
    }
    audits = []
    paths = config_paths(protocol)
    for key in keys:
        raw_path = Path(configs[key]["output_dir"]) / "raw.csv"
        if not args.audit_only:
            raw_path = run_config(paths[key])
        if not raw_path.exists():
            raise SupportFinalError(f"missing support-final raw output: {raw_path}")
        audits.append(audit_execution(raw_path, configs[key], environment_map[key]))
        deterministic_gzip_copy(raw_path)
        print(f"SUPPORT_FINAL_ENVIRONMENT_PASS environment={key} runs=270")
    print(
        "SUPPORT_FINAL_EXECUTION_PASS "
        f"environments={len(audits)} runs={sum(item['observed_runs'] for item in audits)}"
    )


if __name__ == "__main__":
    main()
