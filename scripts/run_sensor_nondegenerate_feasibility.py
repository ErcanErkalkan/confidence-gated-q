from __future__ import annotations

import argparse
import hashlib
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
from hybrid_q.statistics import aggregate as aggregate_standard  # noqa: E402
from scripts.audit_results import audit  # noqa: E402


CONFIG_PATH = ROOT / "configs/diagnostic_extensions/sensor_nondegenerate_development/feasibility_grid.yaml"
LOCK_YAML = ROOT / "configs/diagnostic_extensions/sensor_nondegenerate_development/closure_lock.yaml"
LOCK_MD = LOCK_YAML.with_suffix(".md")
LOCK_DIGEST = LOCK_YAML.with_suffix(".sha256")
EXPECTED_SEEDS = [15006, 15007, 15008]
LEARNED_AGENTS = {"feed_forward_dqn", "selected_temporal_drqn"}
REFERENCE_BY_INTERFACE = {
    "low_level": "sensorized_motor_controller",
    "high_level": "velocity_setpoint_controller",
}


class SensorFeasibilityError(ValueError):
    pass


def composite_sha256() -> str:
    digest = hashlib.sha256()
    for path in (LOCK_YAML, LOCK_MD):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_and_validate(config_path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = yaml.safe_load(LOCK_YAML.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    digest = composite_sha256()
    expected_digest = LOCK_DIGEST.read_text(encoding="utf-8").split()[0]
    if digest != expected_digest:
        raise SensorFeasibilityError("sensor closure protocol digest mismatch")
    if protocol.get("status") != "immutable_before_feasibility_execution":
        raise SensorFeasibilityError("sensor closure protocol is not immutable")
    if config.get("analysis", {}).get("protocol_sha256") != digest:
        raise SensorFeasibilityError("feasibility config protocol digest mismatch")
    if [int(seed) for seed in config.get("seeds", [])] != EXPECTED_SEEDS:
        raise SensorFeasibilityError("feasibility development seed mismatch")
    if any(16000 <= int(seed) <= 16099 for seed in config["seeds"]):
        raise SensorFeasibilityError("sensorized final seed in feasibility config")

    locked_candidates = {
        item["candidate_id"]: (
            int(item["episode_horizon"]),
            int(item["training_interactions"]),
        )
        for item in protocol["feasibility_phase"]["candidates"]
    }
    envs = config.get("envs", [])
    if len(envs) != 2 * len(locked_candidates):
        raise SensorFeasibilityError("feasibility environment count mismatch")
    observed: dict[str, set[tuple[str, int, int]]] = {}
    for env in envs:
        candidate = env.get("feasibility_candidate_id")
        interface = env.get("control_interface")
        if candidate not in locked_candidates or interface not in REFERENCE_BY_INTERFACE:
            raise SensorFeasibilityError("unknown feasibility candidate/interface")
        horizon, budget = locked_candidates[candidate]
        if int(env["max_steps"]) != horizon or int(env["kwargs"]["max_steps"]) != horizon:
            raise SensorFeasibilityError(f"horizon mismatch for {env['name']}")
        if int(env["training_steps"]) != budget:
            raise SensorFeasibilityError(f"budget mismatch for {env['name']}")
        if env["kwargs"]["control_interface_mode"] != interface:
            raise SensorFeasibilityError(f"interface mismatch for {env['name']}")
        observed.setdefault(candidate, set()).add((interface, horizon, budget))
    for candidate, values in observed.items():
        if {item[0] for item in values} != set(REFERENCE_BY_INTERFACE):
            raise SensorFeasibilityError(f"interface coverage mismatch: {candidate}")
    return protocol, config, digest


def build_final_checkpoint_summary(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    evaluation = raw.loc[raw["phase"].eq("eval")].copy()
    episode_seed = (
        evaluation.groupby(
            ["environment", "agent", "seed", "checkpoint"], sort=True
        )["success"]
        .mean()
        .rename("seed_success")
        .reset_index()
    )
    final = (
        episode_seed.sort_values("checkpoint")
        .groupby(["environment", "agent", "seed"], sort=True)
        .tail(1)
    )
    env_meta = {
        env["name"]: {
            "candidate_id": env["feasibility_candidate_id"],
            "control_interface": env["control_interface"],
            "training_steps": int(env["training_steps"]),
            "episode_horizon": int(env["max_steps"]),
        }
        for env in config["envs"]
    }
    rows = []
    for (environment, agent), group in final.groupby(
        ["environment", "agent"], sort=True
    ):
        values = group["seed_success"]
        rows.append(
            {
                "candidate_id": env_meta[environment]["candidate_id"],
                "control_interface": env_meta[environment]["control_interface"],
                "environment": environment,
                "agent": agent,
                "training_steps": env_meta[environment]["training_steps"],
                "episode_horizon": env_meta[environment]["episode_horizon"],
                "n_seeds": int(group["seed"].nunique()),
                "success_mean": float(values.mean()),
                "success_min_seed": float(values.min()),
                "success_max_seed": float(values.max()),
                "source_file": "results/diagnostic_extensions/sensor_nondegenerate_development/feasibility/raw.csv",
            }
        )
    return pd.DataFrame(rows)


def select_candidate(summary: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    diagnostics = []
    selected: str | None = None
    for candidate in protocol["feasibility_phase"]["candidates"]:
        candidate_id = candidate["candidate_id"]
        group = summary.loc[summary["candidate_id"].eq(candidate_id)]
        learned = group.loc[group["agent"].isin(LEARNED_AGENTS)]
        references = group.loc[
            group.apply(
                lambda row: REFERENCE_BY_INTERFACE.get(row["control_interface"])
                == row["agent"],
                axis=1,
            )
        ]
        learned_values = learned["success_mean"].astype(float)
        reference_feasible = bool(
            len(references) and references["success_mean"].max() > 0.10
        )
        learned_nondegenerate = bool(
            len(learned_values)
            and learned_values.between(0.05, 0.95, inclusive="both").any()
        )
        learned_range = (
            float(learned_values.max() - learned_values.min())
            if len(learned_values)
            else float("nan")
        )
        range_sufficient = bool(pd.notna(learned_range) and learned_range >= 0.10)
        eligible = reference_feasible and learned_nondegenerate and range_sufficient
        diagnostics.append(
            {
                "candidate_id": candidate_id,
                "training_interactions": int(candidate["training_interactions"]),
                "episode_horizon": int(candidate["episode_horizon"]),
                "reference_success_max": (
                    float(references["success_mean"].max())
                    if len(references)
                    else float("nan")
                ),
                "learned_success_min": (
                    float(learned_values.min()) if len(learned_values) else float("nan")
                ),
                "learned_success_max": (
                    float(learned_values.max()) if len(learned_values) else float("nan")
                ),
                "learned_success_range": learned_range,
                "reference_feasible": reference_feasible,
                "learned_nondegenerate": learned_nondegenerate,
                "range_sufficient": range_sufficient,
                "eligible": eligible,
            }
        )
        if selected is None and eligible:
            selected = candidate_id

    result: dict[str, Any] = {
        "selection_status": "selected" if selected else "no_eligible_candidate",
        "selected_candidate_id": selected,
        "p_values_used": False,
        "candidate_diagnostics": diagnostics,
    }
    if selected:
        chosen = summary.loc[
            summary["candidate_id"].eq(selected)
            & summary["agent"].isin(LEARNED_AGENTS)
        ]
        by_interface = chosen.groupby("control_interface")["success_mean"].mean()
        high_cells = chosen.loc[chosen["control_interface"].eq("high_level")]
        high_nondegenerate = high_cells["success_mean"].between(
            0.05, 0.95, inclusive="both"
        ).any()
        high_advantage = float(
            by_interface.get("high_level", float("nan"))
            - by_interface.get("low_level", float("nan"))
        )
        selected_interface = (
            "high_level"
            if high_nondegenerate and high_advantage >= 0.10
            else "low_level"
        )
        candidate = next(
            item
            for item in protocol["feasibility_phase"]["candidates"]
            if item["candidate_id"] == selected
        )
        result.update(
            {
                "selected_training_interactions": int(
                    candidate["training_interactions"]
                ),
                "selected_episode_horizon": int(candidate["episode_horizon"]),
                "selected_factorial_interface": selected_interface,
                "mean_learned_high_minus_low_success": high_advantage,
            }
        )
    return result


def audit_execution(raw: pd.DataFrame, config: dict[str, Any], digest: str) -> dict[str, Any]:
    runs = raw[["environment", "agent", "seed"]].drop_duplicates()
    evaluation = raw.loc[raw["phase"].eq("eval")]
    expected_runs = 54
    expected_eval_rows = (6 * 3 * 2 + 6 * 3 * 4 + 6 * 3 * 8) * 10
    violations: list[str] = []
    if len(runs) != expected_runs:
        violations.append(f"run count {len(runs)} != {expected_runs}")
    if set(raw["seed"].astype(int)) != set(EXPECTED_SEEDS):
        violations.append("development seed coverage mismatch")
    if raw["seed"].astype(int).between(16000, 16099).any():
        violations.append("final sensorized seed row detected")
    if len(evaluation) != expected_eval_rows:
        violations.append(
            f"evaluation rows {len(evaluation)} != {expected_eval_rows}"
        )
    return {
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "development_feasibility",
        "protocol_sha256": digest,
        "expected_agent_seed_environment_runs": expected_runs,
        "observed_agent_seed_environment_runs": len(runs),
        "evaluation_rows": len(evaluation),
        "development_seeds": EXPECTED_SEEDS,
        "final_seed_rows": int(raw["seed"].astype(int).between(16000, 16099).sum()),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol, config, digest = load_and_validate(config_path)
    output_dir = ROOT / config["output_dir"]
    raw_path = run_config(config_path)
    aggregate_standard(raw_path, output_dir)
    standard = audit(config_path, output_dir)
    raw = pd.read_csv(raw_path)
    execution = audit_execution(raw, config, digest)
    summary = build_final_checkpoint_summary(raw, config)
    selection = select_candidate(summary, protocol)
    summary.to_csv(output_dir / "feasibility_summary.csv", index=False)
    selection_path = output_dir / "feasibility_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump(selection, sort_keys=False), encoding="utf-8"
    )
    selection_digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    (output_dir / "feasibility_selection.sha256").write_text(
        f"{selection_digest}  feasibility_selection.yaml\n", encoding="utf-8"
    )
    execution["standard_result_audit"] = standard["status"]
    execution["standard_provenance_audit"] = standard["provenance_status"]
    execution["selection_status"] = selection["selection_status"]
    (output_dir / "execution_audit.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )
    if execution["status"] != "PASS" or standard["status"] != "PASS":
        raise SystemExit(json.dumps(execution, indent=2))
    print(
        "SENSOR_NONDEGENERATE_FEASIBILITY_PASS "
        f"runs={execution['observed_agent_seed_environment_runs']} "
        f"selection={selection['selection_status']}"
    )
    if selection["selection_status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
