from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hybrid_q.envs import has_uav_backend


LEGACY_CONFIGS = [
    "dqn_tuning_development.json",
    "dqn_strong_validation.json",
    "confirmatory_extended_compact.json",
    "support_abstention_replication.json",
    "minigrid_extended_diagnostic.json",
    "application_navigation_case_study.json",
    "adaptive_gate_compact_validation.json",
    "cost_support_metrics.json",
]

# E17-E28 are executed by ten script families because E18-E20 share one
# finalized independent-shift runner. The ordering preserves development ->
# final/replication dependencies encoded by the frozen configs.
DIAGNOSTIC_EXTENSION_PIPELINE = [
    ("scripts/run_fuzzy_crisp_development.py", "scripts/aggregate_fuzzy_crisp_development.py"),       # E17
    ("scripts/run_independent_shift_final.py", "scripts/aggregate_independent_shifts.py"),             # E18-E20
    ("scripts/run_reliability_calibration_independent.py", "scripts/aggregate_reliability_calibration_independent.py"),  # E21
    ("scripts/run_support_estimator_development.py", "scripts/aggregate_support_estimator_development.py"),            # E22
    ("scripts/run_support_estimator_final.py", "scripts/aggregate_support_estimator_final.py"),                        # E23
    ("scripts/run_sensor_factorial_development.py", "scripts/aggregate_sensor_factorial.py"),                          # E24
    ("scripts/run_temporal_interface_development.py", "scripts/aggregate_temporal_interface_development.py"),          # E25
    ("scripts/run_sensorized_final.py", "scripts/aggregate_sensorized_final.py"),                                      # E26
    ("scripts/run_safety_trace_reruns.py", "scripts/aggregate_safety_traces.py"),                                      # E27
    ("scripts/run_sensor_nondegenerate_feasibility.py", None),                                                         # E28
]


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")) if part
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def result_dir(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return ROOT / config["output_dir"]


def raw_result(output: Path) -> Path:
    for name in ("raw.csv", "raw.csv.gz", "raw.csv.xz"):
        candidate = output / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing raw experiment data in {output}. Run the corresponding full experiment first."
    )


def run_legacy_e01_e16() -> None:
    """Execute the original E01-E16 families and their public aggregations."""
    for name in LEGACY_CONFIGS:
        run("scripts/run_benchmark.py", "--config", f"configs/{name}")

    run("scripts/run_strong_baselines.py")
    run("scripts/aggregate_strong_baselines.py")
    run("scripts/run_approx_support_experiments.py")
    run("scripts/aggregate_approx_support.py")
    run("scripts/run_fuzzy_ablation.py")
    run("scripts/aggregate_fuzzy_ablation.py")
    run("scripts/run_application_risk_variants.py")
    run("scripts/aggregate_application_risk.py")

    if not has_uav_backend():
        raise RuntimeError(
            "Full E01-E28 reproduction requires the recorded UAV environment. "
            "Use the Python/UAV stack documented in requirements-tested-uav.txt."
        )
    run("scripts/run_uav_validation.py")
    run("scripts/aggregate_uav_validation.py")
    run("scripts/run_uav_validation.py", "--config", "configs/uav_sensorized_motor_30seed.yaml")
    run("scripts/aggregate_uav_sensorized_validation.py")

    for config_name, result_name in (
        ("configs/fuzzy_reliability_confirmatory_30seed.yaml", "results/fuzzy_reliability_confirmatory"),
        ("configs/fuzzy_reliability_shift_confirmatory_30seed.yaml", "results/fuzzy_reliability_shift_confirmatory"),
    ):
        run("scripts/run_benchmark.py", "--config", config_name)
        run(
            "scripts/aggregate_results.py",
            "--input", str(raw_result(ROOT / result_name).relative_to(ROOT)),
            "--output", result_name,
        )
        run(
            "scripts/audit_results.py",
            "--config", config_name,
            "--result-dir", result_name,
            "--output", f"{result_name}/audit.json",
        )
    run("scripts/aggregate_fuzzy_reliability.py")


def aggregate_and_audit_legacy_configs() -> None:
    for name in LEGACY_CONFIGS:
        config_path = ROOT / "configs" / name
        output = result_dir(config_path)
        raw = raw_result(output)
        run(
            "scripts/aggregate_results.py",
            "--input", str(raw.relative_to(ROOT)),
            "--output", str(output.relative_to(ROOT)),
        )
        run(
            "scripts/audit_results.py",
            "--config", str(config_path.relative_to(ROOT)),
            "--result-dir", str(output.relative_to(ROOT)),
            "--output", str((output / "audit.json").relative_to(ROOT)),
        )


def run_diagnostic_extensions_e17_e28() -> None:
    for runner, aggregator in DIAGNOSTIC_EXTENSION_PIPELINE:
        if not (ROOT / runner).is_file():
            raise FileNotFoundError(f"Missing registered diagnostic runner: {runner}")
        run(runner)
        if aggregator is not None:
            if not (ROOT / aggregator).is_file():
                raise FileNotFoundError(f"Missing registered diagnostic aggregator: {aggregator}")
            run(aggregator)
    # E28's canonical public CSV and the public fuzzy-rule table are generated here.
    run("scripts/generate_tables.py")
    run("scripts/generate_figures.py")


def run_full_e01_e28() -> None:
    run_legacy_e01_e16()
    aggregate_and_audit_legacy_configs()
    run_diagnostic_extensions_e17_e28()


def run_continuous_control_grid() -> None:
    """Execute the frozen supplemental continuous-control grid in Environment B."""
    protocol_path = ROOT / "configs" / "continuous_control" / "CONTINUOUS_CONTROL_PROTOCOL.yaml"
    manifest_path = ROOT / "configs" / "continuous_control" / "CONTINUOUS_CONTROL_PROTOCOL_SHA256.txt"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    algorithms = [entry["id"] for entry in protocol["algorithms"]]
    environments = [entry["id"] for entry in protocol["environments"]]
    seeds = protocol["seeds"]["supplemental_training"]
    expected = int(protocol["budget"]["planned_trained_agents"])
    observed = len(algorithms) * len(environments) * len(seeds)
    if observed != expected:
        raise RuntimeError(f"continuous-control grid size mismatch: {observed} != {expected}")

    for algorithm in algorithms:
        for environment in environments:
            for seed in seeds:
                run(
                    "scripts/run_continuous_control.py",
                    "--lock-yaml", str(protocol_path.relative_to(ROOT)),
                    "--lock-manifest", str(manifest_path.relative_to(ROOT)),
                    "--mode", "supplemental",
                    "--algorithm", algorithm,
                    "--environment", environment,
                    "--seed", str(seed),
                    "--output", "results/continuous_control",
                )
    run("scripts/aggregate_continuous_control.py")
    run_public_audit(require_manifest=False)


def run_public_audit(*, require_manifest: bool) -> None:
    arguments = [
        "scripts/audit_artifact.py",
        "--root", ".",
        "--output", "artifact_audit.json",
    ]
    if require_manifest:
        arguments.append("--require-manifest")
    run(*arguments)


def quick_check() -> None:
    """Cheap smoke + full registered-output integrity audit; no expensive retraining."""
    quick_dir = ROOT / ".quick_repro"
    if quick_dir.exists():
        shutil.rmtree(quick_dir)
    quick_dir.mkdir()
    pytest_base = Path(tempfile.mkdtemp(prefix="hybrid_q_quick_pytest_"))
    try:
        run("-m", "pytest", "-q", "--basetemp", str(pytest_base))
    finally:
        shutil.rmtree(pytest_base, ignore_errors=True)

    smoke_config = ROOT / "configs" / "quick_reproduction_smoke.yaml"
    run("scripts/run_benchmark.py", "--config", str(smoke_config.relative_to(ROOT)))
    smoke_output = quick_dir / "results"
    run(
        "scripts/aggregate_results.py",
        "--input", str((smoke_output / "raw.csv").relative_to(ROOT)),
        "--output", str(smoke_output.relative_to(ROOT)),
    )
    run(
        "scripts/audit_results.py",
        "--config", str(smoke_config.relative_to(ROOT)),
        "--result-dir", str(smoke_output.relative_to(ROOT)),
        "--output", str((smoke_output / "audit.json").relative_to(ROOT)),
    )

    # Deterministic public-only derived assets; these never write into the private paper tree.
    run("scripts/generate_tables.py")
    run("scripts/generate_figures.py")
    run_public_audit(require_manifest=False)
    print("QUICK_REPRO_PASS coverage=E01-E28 continuous_control=S1+S2", flush=True)


def run_release_preflight(*, require_manifest: bool) -> None:
    arguments = ["scripts/preflight_release.py"]
    if require_manifest:
        arguments.append("--require-manifest")
    run(*arguments)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce and audit the confidence-gated-q public artifact. "
            "Environment A covers E01-E28; Environment B covers the supplemental continuous-control grid."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="Run tests, smoke reproduction, and full public-artifact integrity checks without expensive retraining.")
    mode.add_argument("--full", action="store_true", help="Rerun E01-E28 in the compact/UAV environment, regenerate public outputs, then audit the registered artifact.")
    mode.add_argument("--continuous-control", action="store_true", help="Rerun the frozen 30-agent supplemental continuous-control grid in its separate tested environment.")
    mode.add_argument("--audit-only", action="store_true", help="Audit existing registered outputs without running experiments or smoke tests.")
    mode.add_argument("--preflight", action="store_true", help="Run compile, pytest, protocol/hash, and public-artifact audits without training or regenerating scientific results.")
    parser.add_argument("--require-manifest", action="store_true", help="With --audit-only or --preflight, require complete final MANIFEST.sha256 frozen-release coverage.")
    args = parser.parse_args()

    if args.require_manifest and not (args.audit_only or args.preflight):
        parser.error("--require-manifest is only valid with --audit-only or --preflight")

    if args.full:
        run_full_e01_e28()
        quick_check()
    elif args.continuous_control:
        run_continuous_control_grid()
    elif args.audit_only:
        run_public_audit(require_manifest=args.require_manifest)
    elif args.preflight:
        run_release_preflight(require_manifest=args.require_manifest)
    else:
        quick_check()


if __name__ == "__main__":
    main()
