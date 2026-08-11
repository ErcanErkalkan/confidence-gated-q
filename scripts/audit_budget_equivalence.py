from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_q.agents import AgentConfig, DuelingQNetwork, QNetwork  # noqa: E402
from hybrid_q.complexity import (  # noqa: E402
    approximate_inference_flops,
    approximate_inference_macs,
    mapping_operation_estimate,
    trainable_parameter_count,
)
from scripts.audit_multiplicity import (  # noqa: E402
    build_multiplicity_manifest,
)
from scripts.audit_results import raw_result_paths  # noqa: E402


AUDIT_COLUMNS = [
    "family",
    "contrast",
    "left_agent",
    "right_agent",
    "left_seed_set",
    "right_seed_set",
    "seed_sets_equal",
    "training_steps",
    "training_steps_equal",
    "checkpoint_schedule",
    "checkpoint_schedule_equal",
    "evaluation_episodes",
    "evaluation_episodes_equal",
    "gradient_updates_left",
    "gradient_updates_right",
    "interaction_budget_matched",
    "compute_budget_identical",
    "audit_status",
    "reason",
    "environment",
    "metric",
    "evidence_class",
    "historical_audit_scope",
    "left_source_files",
    "right_source_files",
    "planned_contrast_source_file",
]

SPECIAL_SOURCE_BUNDLES = {
    "results/approx_support/planned_contrasts.csv": [
        "results/approx_support/knn_support",
        "results/approx_support/feature_distance_support",
        "results/application_navigation_case_study",
    ],
    "results/strong_baselines/planned_contrasts.csv": [
        "results/strong_baselines/double_dqn",
        "results/strong_baselines/dueling_double_dqn",
        "results/application_navigation_case_study",
    ],
    "results/fuzzy_ablation/combined/planned_contrasts.csv": [
        "results/fuzzy_ablation",
        "results/application_navigation_case_study",
    ],
    "results/reviewer1_remaining/final_shifts/planned_contrasts.csv": [
        "results/reviewer1_remaining/final_shifts/execution/transition_dynamics_shift",
        "results/reviewer1_remaining/final_shifts/execution/observation_shift",
        "results/reviewer1_remaining/final_shifts/execution/localized_multistep_reward_or_policy_shift",
    ],
    "results/reviewer1_remaining/support_final/planned_contrasts.csv": [
        "results/reviewer1_remaining/support_final/execution/application_goal_shift",
        "results/reviewer1_remaining/support_final/execution/independent_observation_shift",
        "results/reviewer1_remaining/support_final/execution/independent_transition_dynamics_shift",
    ],
}


class BudgetAuditError(ValueError):
    pass


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _checkpoint_schedule(config: dict[str, Any], env: dict[str, Any]) -> list[int]:
    if env.get("training_steps") is not None:
        total = int(env["training_steps"])
        interval = int(config["evaluation"]["interval_steps"])
    else:
        total = int(env["episodes"])
        interval = int(config["evaluation"]["interval"])
    schedule = list(range(interval, total + 1, interval))
    if not schedule or schedule[-1] != total:
        schedule.append(total)
    return schedule


def _canonical_result_dir(source_file: str) -> str:
    special = SPECIAL_SOURCE_BUNDLES.get(source_file)
    if special:
        raise BudgetAuditError("special source has more than one result directory")
    parent = Path(source_file).parent
    if parent.name in {"combined", "reported"}:
        parent = parent.parent
    return parent.as_posix()


def _source_bundle(source_file: str) -> list[str]:
    if source_file in SPECIAL_SOURCE_BUNDLES:
        return SPECIAL_SOURCE_BUNDLES[source_file]
    return [_canonical_result_dir(source_file)]


def _agent_config_signature(spec: dict[str, Any]) -> str:
    valid = {item.name for item in fields(AgentConfig)}
    params = dict(spec.get("params", {}))
    resolved = asdict(
        AgentConfig(**{key: value for key, value in params.items() if key in valid})
    )
    extras = {key: value for key, value in params.items() if key not in valid}
    return _compact_json(
        {"kind": spec["kind"], "resolved_agent_config": resolved, "extras": extras}
    )


class ResultSource:
    def __init__(self, root: Path, relative_dir: str):
        self.root = root
        self.relative_dir = relative_dir
        self.path = root / relative_dir
        metadata_path = self.path / "metadata.json"
        if not metadata_path.exists():
            raise BudgetAuditError(f"missing metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.config = self.metadata["config"]
        keep = {
            "environment",
            "agent",
            "seed",
            "phase",
            "checkpoint",
            "episode",
            "environment_steps",
            "gradient_updates",
            "training_loss_mean",
            "training_loss_max",
            "nonfinite_loss_count",
            "completed_checkpoint_count",
            "expected_checkpoint_count",
        }
        self.raw_paths = raw_result_paths(self.path)
        self.raw = pd.concat(
            [
                pd.read_csv(path, usecols=lambda column: column in keep)
                for path in self.raw_paths
            ],
            ignore_index=True,
            sort=False,
        )
        seed_metrics_path = self.path / "seed_metrics.csv"
        self.seed_metrics = (
            pd.read_csv(seed_metrics_path)
            if seed_metrics_path.exists()
            else pd.DataFrame()
        )

    def has(self, environment: str, agent: str) -> bool:
        rows = self.raw[
            (self.raw["environment"].astype(str) == environment)
            & (self.raw["agent"].astype(str) == agent)
        ]
        return not rows.empty

    def summarize(self, environment: str, agent: str) -> dict[str, Any]:
        env_specs = {
            str(spec.get("name", spec["id"])): spec for spec in self.config["envs"]
        }
        agent_specs = {str(spec["name"]): spec for spec in self.config["agents"]}
        if environment not in env_specs or agent not in agent_specs:
            raise BudgetAuditError(
                f"{self.relative_dir}: config does not declare {environment}/{agent}"
            )
        env_spec = env_specs[environment]
        agent_spec = agent_specs[agent]
        rows = self.raw[
            (self.raw["environment"].astype(str) == environment)
            & (self.raw["agent"].astype(str) == agent)
        ].copy()
        evaluation = rows[rows["phase"] == "eval"].copy()
        actual_seeds = sorted(int(value) for value in evaluation["seed"].unique())
        declared_seeds = sorted(
            int(value)
            for value in env_spec.get("seeds", self.config.get("seeds", []))
        )
        declared_steps = int(
            env_spec.get("training_steps", env_spec.get("episodes"))
        )
        observed_steps_by_seed = {
            str(int(seed)): int(group["environment_steps"].max())
            for seed, group in rows.groupby("seed")
        }
        declared_schedule = _checkpoint_schedule(self.config, env_spec)
        observed_schedules = {
            tuple(sorted(int(value) for value in group["checkpoint"].unique()))
            for _, group in evaluation.groupby("seed")
        }
        declared_eval_episodes = int(self.config["evaluation"]["episodes"])
        episode_counts = sorted(
            int(value)
            for value in evaluation.groupby(["seed", "checkpoint"])["episode"]
            .nunique()
            .unique()
        )
        final_rows = evaluation.sort_values("checkpoint").groupby("seed").tail(1)
        gradient_updates = {
            str(int(row.seed)): int(row.gradient_updates)
            for row in final_rows.itertuples()
        }
        raw_complete = (
            actual_seeds == declared_seeds
            and set(observed_steps_by_seed.values()) == {declared_steps}
            and observed_schedules == {tuple(declared_schedule)}
            and episode_counts == [declared_eval_episodes]
        )

        seed_metrics_consistent = True
        if not self.seed_metrics.empty:
            metrics = self.seed_metrics[
                (self.seed_metrics["environment"].astype(str) == environment)
                & (self.seed_metrics["agent"].astype(str) == agent)
            ]
            if metrics.empty:
                seed_metrics_consistent = False
            else:
                metric_updates = {
                    str(int(row.seed)): int(row.gradient_updates)
                    for row in metrics.itertuples()
                }
                seed_metrics_consistent = metric_updates == gradient_updates

        stability_columns = {
            "training_loss_mean",
            "training_loss_max",
            "nonfinite_loss_count",
            "completed_checkpoint_count",
            "expected_checkpoint_count",
        }
        loss_logged = stability_columns.issubset(rows.columns)
        source_files = [
            (path.relative_to(self.root)).as_posix() for path in self.raw_paths
        ]
        source_files.extend(
            [
                f"{self.relative_dir}/metadata.json",
                f"{self.relative_dir}/seed_metrics.csv",
            ]
        )
        return {
            "actual_seeds": actual_seeds,
            "declared_seeds": declared_seeds,
            "declared_steps": declared_steps,
            "observed_steps": observed_steps_by_seed,
            "declared_schedule": declared_schedule,
            "observed_schedules": [list(item) for item in sorted(observed_schedules)],
            "declared_eval_episodes": declared_eval_episodes,
            "observed_eval_episode_counts": episode_counts,
            "gradient_updates": gradient_updates,
            "raw_complete": raw_complete,
            "seed_metrics_consistent": seed_metrics_consistent,
            "loss_logged": loss_logged,
            "agent_signature": _agent_config_signature(agent_spec),
            "runtime_signature": _compact_json(self.config.get("runtime", {})),
            "source_files": source_files,
        }


def _select_source(
    sources: list[ResultSource], environment: str, agent: str
) -> ResultSource:
    matches = [source for source in sources if source.has(environment, agent)]
    if len(matches) != 1:
        names = [source.relative_dir for source in matches]
        raise BudgetAuditError(
            f"expected exactly one source for {environment}/{agent}; found {names}"
        )
    return matches[0]


def _comparison_row(
    record: pd.Series,
    sources: list[ResultSource],
) -> dict[str, Any]:
    environment = str(record["environment_or_severity"])
    left_agent = str(record["left"])
    right_agent = str(record["right"])
    left = _select_source(sources, environment, left_agent).summarize(
        environment, left_agent
    )
    right = _select_source(sources, environment, right_agent).summarize(
        environment, right_agent
    )

    seed_equal = left["actual_seeds"] == right["actual_seeds"]
    steps_equal = left["declared_steps"] == right["declared_steps"]
    schedule_equal = left["declared_schedule"] == right["declared_schedule"]
    episodes_equal = (
        left["declared_eval_episodes"] == right["declared_eval_episodes"]
    )
    sources_complete = all(
        (
            left["raw_complete"],
            right["raw_complete"],
            left["seed_metrics_consistent"],
            right["seed_metrics_consistent"],
        )
    )
    interaction_matched = all(
        (seed_equal, steps_equal, schedule_equal, episodes_equal, sources_complete)
    )
    signature_equal = left["agent_signature"] == right["agent_signature"]
    updates_equal = left["gradient_updates"] == right["gradient_updates"]
    runtime_equal = left["runtime_signature"] == right["runtime_signature"]
    compute_identical = all(
        (interaction_matched, signature_equal, updates_equal, runtime_equal)
    )

    reasons: list[str] = []
    if not sources_complete:
        reasons.append("declared/observed raw or seed-metric coverage mismatch")
    if not all((seed_equal, steps_equal, schedule_equal, episodes_equal)):
        reasons.append("one or more interaction-budget dimensions differ")
    if interaction_matched and not compute_identical:
        differences = []
        if not signature_equal:
            differences.append("algorithm/agent configuration")
        if not updates_equal:
            differences.append("gradient-update counts")
        if not runtime_equal:
            differences.append("runtime configuration")
        reasons.append(
            "interaction budget matched; compute identity not established because "
            + ", ".join(differences)
            + "; wall-clock equality is not treated as operation-count identity"
        )
    if compute_identical:
        reasons.append(
            "interaction schedule, resolved agent configuration, runtime settings, "
            "and per-seed gradient-update counts match"
        )
    scope = (
        "metric/checkpoint/loss audit"
        if left["loss_logged"] and right["loss_logged"]
        else "metric/checkpoint audit only"
    )
    return {
        "family": record["report_family"],
        "contrast": record["contrast_name"],
        "left_agent": left_agent,
        "right_agent": right_agent,
        "left_seed_set": _compact_json(left["actual_seeds"]),
        "right_seed_set": _compact_json(right["actual_seeds"]),
        "seed_sets_equal": seed_equal,
        "training_steps": _compact_json(
            {
                "left_declared": left["declared_steps"],
                "left_observed_by_seed": left["observed_steps"],
                "right_declared": right["declared_steps"],
                "right_observed_by_seed": right["observed_steps"],
            }
        ),
        "training_steps_equal": steps_equal,
        "checkpoint_schedule": _compact_json(
            {
                "left_declared": left["declared_schedule"],
                "left_observed": left["observed_schedules"],
                "right_declared": right["declared_schedule"],
                "right_observed": right["observed_schedules"],
            }
        ),
        "checkpoint_schedule_equal": schedule_equal,
        "evaluation_episodes": _compact_json(
            {
                "left_declared": left["declared_eval_episodes"],
                "left_observed_counts": left["observed_eval_episode_counts"],
                "right_declared": right["declared_eval_episodes"],
                "right_observed_counts": right["observed_eval_episode_counts"],
            }
        ),
        "evaluation_episodes_equal": episodes_equal,
        "gradient_updates_left": _compact_json(left["gradient_updates"]),
        "gradient_updates_right": _compact_json(right["gradient_updates"]),
        "interaction_budget_matched": interaction_matched,
        "compute_budget_identical": compute_identical,
        "audit_status": "PASS" if interaction_matched else "FAIL",
        "reason": "; ".join(reasons),
        "environment": environment,
        "metric": record["metric"],
        "evidence_class": record["evidence_class"],
        "historical_audit_scope": scope,
        "left_source_files": _compact_json(left["source_files"]),
        "right_source_files": _compact_json(right["source_files"]),
        "planned_contrast_source_file": record["source_file"],
    }


def build_budget_audit(root: Path, registry: dict[str, Any]) -> pd.DataFrame:
    manifest = build_multiplicity_manifest(root, registry)
    planned = manifest[
        (manifest["source_kind"] == "planned")
        & manifest["principal_planned"].astype(bool)
    ].copy()
    cache: dict[str, ResultSource] = {}
    rows: list[dict[str, Any]] = []
    for _, record in planned.iterrows():
        bundle = _source_bundle(str(record["source_file"]))
        for path in bundle:
            if path not in cache:
                cache[path] = ResultSource(root, path)
        sources = [cache[path] for path in bundle]
        rows.append(_comparison_row(record, sources))
    result = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    duplicate_key = ["family", "environment", "contrast", "metric"]
    if result.duplicated(duplicate_key).any():
        raise BudgetAuditError("duplicate planned comparison/environment rows")
    if len(result) != len(planned):
        raise BudgetAuditError("not every principal planned row was audited")
    return result.sort_values(duplicate_key).reset_index(drop=True)


def build_dqn_candidate_grid(config_path: Path, root: Path) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    param_columns = sorted(
        {key for agent in config["agents"] for key in agent.get("params", {})}
    )
    rows = []
    for agent in config["agents"]:
        row = {"candidate": agent["name"], "algorithm": agent["kind"]}
        row.update({key: agent.get("params", {}).get(key, "") for key in param_columns})
        row.update(
            {
                "development_environment_count": len(config["envs"]),
                "seed_set": _compact_json(config["seeds"]),
                "evaluation_interval_steps": config["evaluation"]["interval_steps"],
                "evaluation_episodes": config["evaluation"]["episodes"],
                "selection_rule": config["analysis"]["selection_rule"],
                "source_config": config_path.relative_to(root).as_posix(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_complexity_table() -> pd.DataFrame:
    input_dim, action_dim, hidden_size = 4, 5, 128
    dqn = QNetwork(input_dim, action_dim, hidden_size)
    dueling = DuelingQNetwork(input_dim, action_dim, hidden_size)
    rows = []
    for name, model, note in (
        ("DQN", dqn, "Vanilla DQN online Q-network."),
        (
            "Double DQN",
            dqn,
            "Same inference network as DQN; Double DQN changes training targets.",
        ),
        (
            "Dueling Double DQN",
            dueling,
            "Shared trunk with value and advantage heads; output combination excluded.",
        ),
    ):
        macs = approximate_inference_macs(model)
        rows.append(
            {
                "component": name,
                "scope": "one online-network inference, batch size 1",
                "input_dimension": input_dim,
                "action_count": action_dim,
                "hidden_size": hidden_size,
                "exact_trainable_parameters": trainable_parameter_count(model),
                "approximate_inference_macs": macs,
                "approximate_inference_flops": approximate_inference_flops(model),
                "comparison_ops_approx": 0,
                "special_functions_approx": 0,
                "definition_and_exclusions": (
                    note
                    + " MACs sum Linear in_features*out_features; approximate "
                    "FLOPs=2*MACs. Bias, ReLU, reductions, comparisons, memory "
                    "traffic, target network, replay, and training are excluded."
                ),
                "source": (
                    "configs/application_navigation_case_study.json; "
                    "src/hybrid_q/agents.py"
                ),
            }
        )
    for name, mapping in (
        ("Fuzzy mapping overhead", "fuzzy_triangular_five_rule"),
        ("Same-input crisp mapping overhead", "same_input_crisp_threshold"),
    ):
        estimate = mapping_operation_estimate(mapping)
        rows.append(
            {
                "component": name,
                "scope": "mapping only after the same two normalized scalar inputs",
                "input_dimension": 2,
                "action_count": "",
                "hidden_size": "",
                "exact_trainable_parameters": 0,
                "approximate_inference_macs": 0,
                "approximate_inference_flops": estimate.arithmetic_flops,
                "comparison_ops_approx": estimate.comparisons,
                "special_functions_approx": estimate.special_functions,
                "definition_and_exclusions": estimate.definition,
                "source": (
                    "configs/application_navigation_case_study.json; "
                    "src/hybrid_q/agents.py; src/hybrid_q/complexity.py"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_tex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.write_text(
        frame.to_latex(
            index=False,
            escape=True,
            caption=caption,
            label=label,
            na_rep="--",
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit planned-comparison budgets without retraining."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    registry_path = root / "project_admin/reviewer1_remaining/EVIDENCE_CLASS_REGISTRY.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    audit = build_budget_audit(root, registry)
    output_dir = root / "results/reviewer1_remaining/budgets"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "budget_equivalence_audit.csv"
    audit.to_csv(audit_path, index=False)

    tables_dir = root / "tables"
    candidate = build_dqn_candidate_grid(
        root / "configs/dqn_tuning_development.json", root
    )
    candidate.to_csv(tables_dir / "table_dqn_candidate_grid.csv", index=False)
    _write_tex(
        candidate,
        tables_dir / "table_dqn_candidate_grid.tex",
        "Complete DQN development candidate grid.",
        "tab:dqn-candidate-grid",
    )
    complexity = build_model_complexity_table()
    complexity.to_csv(tables_dir / "table_model_complexity.csv", index=False)

    failed = int((audit["audit_status"] != "PASS").sum())
    print(
        f"BUDGET_AUDIT_ROWS={len(audit)} FAILURES={failed} "
        f"COMPUTE_IDENTICAL={int(audit['compute_budget_identical'].sum())}"
    )
    print(f"DQN_CANDIDATES={len(candidate)} COMPLEXITY_ROWS={len(complexity)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
