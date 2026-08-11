from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / (
    "configs/diagnostic_extensions/support_final/protocol_lock.yaml"
)
PROTOCOL_DIGEST = PROTOCOL.with_suffix(".sha256")
CONFIG_ROOT = ROOT / "configs/diagnostic_extensions/support_final"


class SupportFinalConfigError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportFinalConfigError(f"YAML root must be a mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_digest() -> str:
    if not PROTOCOL_DIGEST.exists():
        raise SupportFinalConfigError(
            f"missing support-final protocol digest: {PROTOCOL_DIGEST}"
        )
    expected = PROTOCOL_DIGEST.read_text(encoding="utf-8").split()[0]
    observed = file_sha256(PROTOCOL)
    if expected != observed:
        raise SupportFinalConfigError("support-final protocol digest mismatch")
    return observed


def build_config(protocol: dict[str, Any], environment: dict[str, Any]) -> dict:
    budget = protocol["matched_budget"]
    environment_key = str(environment["environment_key"])
    agents = [
        {
            "name": item["agent_id"],
            "kind": item["kind"],
            "params": item["params"],
        }
        for item in protocol["agents"]
    ]
    contrasts = [
        {
            "name": item["name"],
            "left": item["left"],
            "right": item["right"],
            "metric": protocol["primary_metric"]["name"],
            "status": item["status"],
        }
        for item in protocol["planned_contrasts"]
    ]
    env_spec = {
        "name": environment["environment_name"],
        "id": environment["environment_id"],
        "mechanism_class": environment["mechanism_class"],
        "seeds": environment["seeds"],
        "kwargs": environment["kwargs"],
        "training_steps": budget["training_interactions_per_agent_seed"],
        "max_steps": environment["max_steps"],
        "success_mode": environment["success_mode"],
    }
    if environment.get("eval_kwargs") is not None:
        env_spec["eval_kwargs"] = environment["eval_kwargs"]
    if environment.get("severity_id") is not None:
        env_spec["severity_id"] = environment["severity_id"]
    return {
        "experiment_name": f"support_final_{environment_key}",
        "output_dir": (
            "results/diagnostic_extensions/support_final/execution/"
            f"{environment_key}"
        ),
        "runtime": {
            "torch_threads": budget["torch_threads"],
            "torch_interop_threads": budget["torch_interop_threads"],
            "workers": budget["workers"],
        },
        "seeds": [],
        "evaluation": {
            "interval_steps": budget["checkpoint_interval"],
            "episodes": budget["evaluation_episodes_per_checkpoint"],
        },
        "analysis": {
            "evidence_class": protocol["evidence_class"],
            "analysis_status": "locked_support_estimator_final_replication",
            "environment_key": environment_key,
            "primary_metric": protocol["primary_metric"]["name"],
            "primary_window_first": environment["primary_window_first"],
            "primary_window_last": environment["primary_window_last"],
            "report_level_holm_family": protocol["multiplicity"][
                "report_family_template"
            ].format(environment_key=environment_key),
            "global_sensitivity_scope": protocol["multiplicity"][
                "global_sensitivity_scope"
            ],
            "no_early_stopping": budget["no_early_stopping"],
            "planned_contrasts": contrasts,
            "protocol_file": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256": file_sha256(PROTOCOL),
        },
        "envs": [env_spec],
        "agents": agents,
    }


def generate_configs(protocol: dict[str, Any]) -> list[Path]:
    if protocol.get("status") != "locked_before_final_outcomes":
        raise SupportFinalConfigError("support-final protocol is not locked")
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for environment in protocol["environments"]:
        path = CONFIG_ROOT / f"{environment['environment_key']}.yaml"
        payload = build_config(protocol, environment)
        path.write_bytes(
            yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
        )
        paths.append(path)
    return paths


def main() -> None:
    digest = verify_protocol_digest()
    paths = generate_configs(load_yaml(PROTOCOL))
    print(
        "SUPPORT_FINAL_CONFIG_GENERATION_PASS "
        f"configs={len(paths)} protocol_sha256={digest}"
    )


if __name__ == "__main__":
    main()
