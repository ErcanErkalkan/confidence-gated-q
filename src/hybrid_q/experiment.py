from __future__ import annotations

import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

from .gym_compat import gym as gymnasium
import numpy as np
import torch

from . import __version__ as PACKAGE_VERSION
from .agents import BaseAgent, create_agent
from .config import load_config
from .encoding import ObservationEncoder
from .envs import episode_succeeded, make_env, resolve_env_id
from .provenance import (
    execution_input_manifest,
    execution_inputs_clean,
    execution_snapshot_sha256,
    git_commit_hash,
    repository_root,
)


FIELDNAMES = [
    "experiment_name",
    "experiment",
    "config_file",
    "environment",
    "environment_id",
    "resolved_environment_id",
    "observation_representation",
    "agent",
    "seed",
    "phase",
    "train_steps",
    "eval_checkpoint",
    "eval_return",
    "checkpoint",
    "episode",
    "return",
    "length",
    "success",
    "environment_steps",
    "gradient_updates",
    "elapsed_seconds",
    "training_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "wall_clock_training_time",
    "wall_clock_evaluation_time",
    "git_commit_hash",
    "package_version",
    "source_snapshot_sha256",
    "execution_inputs_clean",
    "python_version",
    "torch_version",
    "numpy_version",
    "gymnasium_version",
    "minigrid_version",
    "gym_pybullet_drones_version",
    "pybullet_version",
    "mean_gate",
    "support_abstention_rate",
    "global_tabular_error",
    "global_neural_error",
    "visited_states",
    "failure_rate",
    "collision_rate",
    "risk_zone_rate",
    "idle_rate",
    "risk_adjusted_score",
    "unsupported_state_ratio",
    "memory_branch_usage_ratio",
    "neural_branch_usage_ratio",
    "abstention_ratio",
    "adaptive_alpha_mean",
    "adaptive_alpha_iqr",
    "support_score_mean",
    "uncertainty_score_mean",
    "inference_time_us_per_decision_mean",
    "inference_time_us_per_decision_median",
    "selected_branch",
    "memory_cost_states",
    "memory_cost_entries",
    "memory_cost_bytes_estimated",
    "localization_error_mean",
    "sensor_dropout_rate",
    "camera_visible_rate",
    "motor_saturation_rate",
]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def epsilon_at(step: int, params: dict[str, Any]) -> float:
    start = float(params.get("epsilon_start", 1.0))
    end = float(params.get("epsilon_end", 0.05))
    decay_steps = max(1, int(params.get("epsilon_decay_steps", 20_000)))
    fraction = min(1.0, step / decay_steps)
    return start + fraction * (end - start)


def run_episode(
    env,
    encoder: ObservationEncoder,
    agent: BaseAgent,
    seed: int,
    max_steps: int,
    training: bool,
    epsilon_params: dict[str, Any],
    success_mode: str,
) -> dict[str, float]:
    observation, _ = env.reset(seed=seed)
    encoded = encoder.encode(observation)
    total_return = 0.0
    final_reward = 0.0
    terminated = False
    truncated = False
    trace = _new_decision_trace()

    for length in range(1, max_steps + 1):
        epsilon = (
            epsilon_at(agent.environment_steps, epsilon_params) if training else 0.0
        )
        decision_started = time.perf_counter_ns()
        action = agent.act(encoded.vector, encoded.key, epsilon)
        decision_ns = time.perf_counter_ns() - decision_started
        next_observation, reward, terminated, truncated, info = env.step(
            action
        )
        _append_decision_trace(
            trace, agent.decision_diagnostics(), decision_ns, info
        )
        next_encoded = encoder.encode(next_observation)
        done = bool(terminated or truncated)
        if training:
            agent.observe(
                encoded.vector,
                encoded.key,
                action,
                float(reward),
                next_encoded.vector,
                next_encoded.key,
                done,
            )
        total_return += float(reward)
        final_reward = float(reward)
        encoded = next_encoded
        if done:
            break

    success = episode_succeeded(
        success_mode, terminated, truncated, final_reward
    )
    return {
        "return": total_return,
        "length": length,
        "success": float(success),
        **_summarize_decision_trace(trace, success),
    }


def _new_decision_trace() -> dict[str, list]:
    return {
        "unsupported_state": [],
        "memory_branch_weight": [],
        "neural_branch_weight": [],
        "abstention": [],
        "adaptive_alpha": [],
        "support_score": [],
        "uncertainty_score": [],
        "selected_branch": [],
        "inference_ns": [],
        "collision": [],
        "risk_zone": [],
        "idle": [],
        "lambda_collision": [],
        "lambda_idle": [],
        "localization_error": [],
        "sensor_dropout": [],
        "camera_visible": [],
        "motor_saturation": [],
    }


def _append_decision_trace(
    trace: dict[str, list],
    decision: dict[str, float | str],
    inference_ns: int,
    info: dict[str, Any],
) -> None:
    for key in (
        "unsupported_state",
        "memory_branch_weight",
        "neural_branch_weight",
        "abstention",
        "adaptive_alpha",
        "support_score",
        "uncertainty_score",
        "selected_branch",
    ):
        trace[key].append(decision[key])
    trace["inference_ns"].append(float(inference_ns))
    trace["collision"].append(float(bool(info.get("collision", False))))
    trace["risk_zone"].append(float(bool(info.get("risk_zone", False))))
    trace["idle"].append(float(bool(info.get("idle", False))))
    trace["lambda_collision"].append(float(info.get("lambda_collision", 1.0)))
    trace["lambda_idle"].append(float(info.get("lambda_idle", 0.1)))
    trace["localization_error"].append(
        float(info.get("localization_error", float("nan")))
    )
    trace["sensor_dropout"].append(
        float(bool(info.get("sensor_dropout", False)))
    )
    trace["camera_visible"].append(
        float(bool(info.get("camera_visible", False)))
    )
    trace["motor_saturation"].append(
        float(info.get("motor_saturation", float("nan")))
    )


def _finite_mean(values: list) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _summarize_decision_trace(
    trace: dict[str, list], success: bool
) -> dict[str, float | str]:
    branches = trace["selected_branch"]
    selected_branch = "none"
    if branches:
        counts = {
            branch: branches.count(branch) for branch in set(branches)
        }
        selected_branch = max(sorted(counts), key=counts.get)
    alpha = np.asarray(trace["adaptive_alpha"], dtype=float)
    alpha = alpha[np.isfinite(alpha)]
    inference_us = np.asarray(trace["inference_ns"], dtype=float) / 1_000.0
    collision_rate = _finite_mean(trace["collision"])
    idle_rate = _finite_mean(trace["idle"])
    lambda_collision = _finite_mean(trace["lambda_collision"])
    lambda_idle = _finite_mean(trace["lambda_idle"])
    return {
        "failure_rate": float(not success),
        "collision_rate": collision_rate,
        "risk_zone_rate": _finite_mean(trace["risk_zone"]),
        "idle_rate": idle_rate,
        "risk_adjusted_score": (
            float(success)
            - lambda_collision * collision_rate
            - lambda_idle * idle_rate
        ),
        "localization_error_mean": _finite_mean(
            trace["localization_error"]
        ),
        "sensor_dropout_rate": _finite_mean(trace["sensor_dropout"]),
        "camera_visible_rate": _finite_mean(trace["camera_visible"]),
        "motor_saturation_rate": _finite_mean(trace["motor_saturation"]),
        "unsupported_state_ratio": _finite_mean(
            trace["unsupported_state"]
        ),
        "memory_branch_usage_ratio": _finite_mean(
            trace["memory_branch_weight"]
        ),
        "neural_branch_usage_ratio": _finite_mean(
            trace["neural_branch_weight"]
        ),
        "abstention_ratio": _finite_mean(trace["abstention"]),
        "adaptive_alpha_mean": (
            float(alpha.mean()) if alpha.size else float("nan")
        ),
        "adaptive_alpha_iqr": (
            float(np.quantile(alpha, 0.75) - np.quantile(alpha, 0.25))
            if alpha.size
            else float("nan")
        ),
        "support_score_mean": _finite_mean(trace["support_score"]),
        "uncertainty_score_mean": _finite_mean(
            trace["uncertainty_score"]
        ),
        "inference_time_us_per_decision_mean": float(inference_us.mean()),
        "inference_time_us_per_decision_median": float(
            np.median(inference_us)
        ),
        "selected_branch": selected_branch,
    }


def evaluate(
    env_spec: dict[str, Any],
    encoder: ObservationEncoder,
    agent: BaseAgent,
    base_seed: int,
    checkpoint: int,
    episodes: int,
) -> list[dict[str, float]]:
    env = make_env(env_spec, evaluation=True)
    rows = []
    rng_state = copy.deepcopy(agent.rng.bit_generator.state)
    gate_sum = getattr(agent, "gate_sum", None)
    gate_queries = getattr(agent, "gate_queries", None)
    support_queries = getattr(agent, "support_queries", None)
    support_abstentions = getattr(agent, "support_abstentions", None)
    last_decision = copy.deepcopy(agent._last_decision)
    try:
        for episode in range(episodes):
            eval_seed = (
                1_000_000_000 + base_seed * 1_000_000 + checkpoint * 1_000 + episode
            )
            rows.append(
                run_episode(
                    env=env,
                    encoder=encoder,
                    agent=agent,
                    seed=eval_seed,
                    max_steps=int(env_spec["max_steps"]),
                    training=False,
                    epsilon_params={},
                    success_mode=env_spec.get(
                        "success_mode", "positive_terminal"
                    ),
                )
            )
    finally:
        agent.rng.bit_generator.state = rng_state
        if gate_sum is not None:
            agent.gate_sum = gate_sum
        if gate_queries is not None:
            agent.gate_queries = gate_queries
        if support_queries is not None:
            agent.support_queries = support_queries
        if support_abstentions is not None:
            agent.support_abstentions = support_abstentions
        agent._last_decision = last_decision
        env.close()
    return rows


def write_metadata(
    config: dict[str, Any],
    output_dir: Path,
    config_file: str,
    provenance: dict[str, Any],
) -> None:
    config_text = json.dumps(config, sort_keys=True).encode("utf-8")
    packages = {}
    for name in (
        "gymnasium",
        "minigrid",
        "gym-pybullet-drones",
        "pybullet",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "config_sha256": hashlib.sha256(config_text).hexdigest(),
        "config_file": config_file,
        **provenance,
        "environment_observations": [
            {
                "name": spec.get("name", spec["id"]),
                "requested_id": spec["id"],
                "resolved_id": resolve_env_id(spec["id"]),
                "representation": spec.get(
                    "observation", "native_gymnasium_observation"
                ),
            }
            for spec in config["envs"]
        ],
        "packages": packages,
        "config": config,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_row(
    writer: csv.DictWriter,
    config: dict[str, Any],
    env_spec: dict[str, Any],
    agent_spec: dict[str, Any],
    seed: int,
    phase: str,
    checkpoint: int,
    episode: int,
    metrics: dict[str, float],
    agent: BaseAgent,
    started: float,
    evaluation_elapsed_seconds: float = 0.0,
) -> None:
    diagnostics = agent.diagnostics()
    elapsed_seconds = time.perf_counter() - started
    training_elapsed_seconds = elapsed_seconds - evaluation_elapsed_seconds
    is_evaluation = phase == "eval"
    provenance = config["_provenance"]
    writer.writerow(
        {
            "experiment_name": config["experiment_name"],
            "experiment": config["experiment_name"],
            "config_file": config["_config_file"],
            "environment": env_spec.get("name", env_spec["id"]),
            "environment_id": env_spec["id"],
            "resolved_environment_id": resolve_env_id(env_spec["id"]),
            "observation_representation": env_spec.get(
                "observation", "native_gymnasium_observation"
            ),
            "agent": agent_spec["name"],
            "seed": seed,
            "phase": phase,
            "train_steps": agent.environment_steps,
            "eval_checkpoint": checkpoint if is_evaluation else "",
            "eval_return": metrics["return"] if is_evaluation else "",
            "checkpoint": checkpoint,
            "episode": episode,
            **metrics,
            "environment_steps": agent.environment_steps,
            "gradient_updates": agent.gradient_updates,
            "elapsed_seconds": elapsed_seconds,
            "training_elapsed_seconds": training_elapsed_seconds,
            "evaluation_elapsed_seconds": evaluation_elapsed_seconds,
            "wall_clock_training_time": training_elapsed_seconds,
            "wall_clock_evaluation_time": evaluation_elapsed_seconds,
            "git_commit_hash": provenance["git_commit_hash"],
            "package_version": provenance["package_version"],
            "source_snapshot_sha256": provenance[
                "source_snapshot_sha256"
            ],
            "execution_inputs_clean": provenance[
                "execution_inputs_clean"
            ],
            "python_version": provenance["python_version"],
            "torch_version": provenance["torch_version"],
            "numpy_version": provenance["numpy_version"],
            "gymnasium_version": provenance["gymnasium_version"],
            "minigrid_version": provenance["minigrid_version"],
            "gym_pybullet_drones_version": provenance[
                "gym_pybullet_drones_version"
            ],
            "pybullet_version": provenance["pybullet_version"],
            "mean_gate": diagnostics.get("mean_gate", ""),
            "support_abstention_rate": diagnostics.get(
                "support_abstention_rate", ""
            ),
            "global_tabular_error": diagnostics.get(
                "global_tabular_error", ""
            ),
            "global_neural_error": diagnostics.get("global_neural_error", ""),
            "visited_states": diagnostics.get("visited_states", ""),
            "memory_cost_states": diagnostics.get(
                "memory_cost_states", ""
            ),
            "memory_cost_entries": diagnostics.get(
                "memory_cost_entries", ""
            ),
            "memory_cost_bytes_estimated": diagnostics.get(
                "memory_cost_bytes_estimated", ""
            ),
        }
    )


def _run_single(
    config: dict[str, Any],
    env_spec: dict[str, Any],
    agent_spec: dict[str, Any],
    seed: int,
    shard_path: Path,
) -> None:
    runtime = config.get("runtime", {})
    torch.set_num_threads(int(runtime.get("torch_threads", 1)))
    try:
        torch.set_num_interop_threads(
            int(runtime.get("torch_interop_threads", 1))
        )
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    temporary_path = shard_path.with_suffix(".tmp")
    env = make_env(env_spec)
    encoder = ObservationEncoder(env.observation_space)
    agent = create_agent(
        kind=agent_spec["kind"],
        input_dim=encoder.input_dim,
        action_dim=env.action_space.n,
        seed=seed,
        params=agent_spec.get("params", {}),
    )
    started = time.perf_counter()
    evaluation_elapsed_seconds = 0.0
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        episode = 0
        training_steps = env_spec.get("training_steps")
        if training_steps is not None:
            training_steps = int(training_steps)
            interval = int(config["evaluation"]["interval_steps"])
            checkpoints = list(range(interval, training_steps + 1, interval))
            if not checkpoints or checkpoints[-1] != training_steps:
                checkpoints.append(training_steps)
            next_checkpoint_index = 0
            while agent.environment_steps < training_steps:
                episode += 1
                observation, _ = env.reset(
                    seed=seed * 1_000_000 + episode
                )
                encoded = encoder.encode(observation)
                total_return = 0.0
                final_reward = 0.0
                terminated = False
                truncated = False
                trace = _new_decision_trace()
                max_steps = min(
                    int(env_spec["max_steps"]),
                    training_steps - agent.environment_steps,
                )
                for length in range(1, max_steps + 1):
                    epsilon = epsilon_at(
                        agent.environment_steps,
                        agent_spec.get("params", {}),
                    )
                    decision_started = time.perf_counter_ns()
                    action = agent.act(
                        encoded.vector, encoded.key, epsilon
                    )
                    decision_ns = (
                        time.perf_counter_ns() - decision_started
                    )
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)
                    _append_decision_trace(
                        trace,
                        agent.decision_diagnostics(),
                        decision_ns,
                        info,
                    )
                    next_encoded = encoder.encode(next_observation)
                    done = bool(terminated or truncated)
                    agent.observe(
                        encoded.vector,
                        encoded.key,
                        action,
                        float(reward),
                        next_encoded.vector,
                        next_encoded.key,
                        done,
                    )
                    total_return += float(reward)
                    final_reward = float(reward)
                    encoded = next_encoded

                    while (
                        next_checkpoint_index < len(checkpoints)
                        and agent.environment_steps
                        == checkpoints[next_checkpoint_index]
                    ):
                        checkpoint = checkpoints[next_checkpoint_index]
                        evaluation_started = time.perf_counter()
                        eval_rows = evaluate(
                            env_spec=env_spec,
                            encoder=encoder,
                            agent=agent,
                            base_seed=seed,
                            checkpoint=checkpoint,
                            episodes=int(
                                config["evaluation"]["episodes"]
                            ),
                        )
                        evaluation_elapsed_seconds += (
                            time.perf_counter() - evaluation_started
                        )
                        for eval_episode, eval_metrics in enumerate(
                            eval_rows, start=1
                        ):
                            _write_row(
                                writer,
                                config,
                                env_spec,
                                agent_spec,
                                seed,
                                "eval",
                                checkpoint,
                                eval_episode,
                                eval_metrics,
                                agent,
                                started,
                                evaluation_elapsed_seconds,
                            )
                        next_checkpoint_index += 1
                        handle.flush()

                    if done:
                        break

                metrics = {
                    "return": total_return,
                    "length": length,
                    "success": float(success := episode_succeeded(
                        env_spec.get(
                            "success_mode", "positive_terminal"
                        ),
                        terminated,
                        truncated,
                        final_reward,
                    )),
                    **_summarize_decision_trace(
                        trace,
                        success,
                    ),
                }
                _write_row(
                    writer,
                    config,
                    env_spec,
                    agent_spec,
                    seed,
                    "train",
                    agent.environment_steps,
                    episode,
                    metrics,
                    agent,
                    started,
                    evaluation_elapsed_seconds,
                )
        else:
            episodes = int(env_spec["episodes"])
            eval_interval = int(config["evaluation"]["interval"])
            for episode in range(1, episodes + 1):
                metrics = run_episode(
                    env=env,
                    encoder=encoder,
                    agent=agent,
                    seed=seed * 1_000_000 + episode,
                    max_steps=int(env_spec["max_steps"]),
                    training=True,
                    epsilon_params=agent_spec.get("params", {}),
                    success_mode=env_spec.get(
                        "success_mode", "positive_terminal"
                    ),
                )
                _write_row(
                    writer,
                    config,
                    env_spec,
                    agent_spec,
                    seed,
                    "train",
                    episode,
                    episode,
                    metrics,
                    agent,
                    started,
                    evaluation_elapsed_seconds,
                )
                if episode % eval_interval == 0 or episode == episodes:
                    evaluation_started = time.perf_counter()
                    eval_rows = evaluate(
                        env_spec=env_spec,
                        encoder=encoder,
                        agent=agent,
                        base_seed=seed,
                        checkpoint=episode,
                        episodes=int(config["evaluation"]["episodes"]),
                    )
                    evaluation_elapsed_seconds += (
                        time.perf_counter() - evaluation_started
                    )
                    for eval_episode, eval_metrics in enumerate(
                        eval_rows, start=1
                    ):
                        _write_row(
                            writer,
                            config,
                            env_spec,
                            agent_spec,
                            seed,
                            "eval",
                            episode,
                            eval_episode,
                            eval_metrics,
                            agent,
                            started,
                            evaluation_elapsed_seconds,
                        )
                    handle.flush()
    env.close()
    temporary_path.replace(shard_path)


def _combine_shards(shards: list[Path], raw_path: Path) -> None:
    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        for shard in sorted(shards):
            with shard.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                writer.writerows(reader)


def run_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = repository_root(config_path)
    try:
        config_file = config_path.relative_to(root).as_posix()
    except ValueError:
        config_file = f"external-config/{config_path.name}"
    input_manifest = execution_input_manifest(config_path, root)
    input_snapshot = execution_snapshot_sha256(input_manifest)
    inputs_clean = execution_inputs_clean(config_path, root)
    allow_dirty = bool(
        config.get("runtime", {}).get("allow_dirty_execution_inputs", False)
    ) or os.environ.get("HYBRID_Q_ALLOW_DIRTY_INPUTS") == "1"
    if inputs_clean is False and not allow_dirty:
        raise RuntimeError(
            "Execution inputs differ from the recorded git commit. Commit "
            "src/config/dependency changes first, or set "
            "HYBRID_Q_ALLOW_DIRTY_INPUTS=1 for a non-public smoke run."
        )
    commit_hash = git_commit_hash(root)
    config["_config_file"] = config_file
    config["_provenance"] = {
        "git_commit_hash": commit_hash,
        "package_version": PACKAGE_VERSION,
        "source_snapshot_sha256": input_snapshot,
        "execution_inputs_clean": inputs_clean,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gymnasium_version": gymnasium.__version__,
        "minigrid_version": package_version("minigrid"),
        "gym_pybullet_drones_version": package_version(
            "gym-pybullet-drones"
        ),
        "pybullet_version": package_version("pybullet"),
    }
    runtime = config.get("runtime", {})
    torch.set_num_threads(int(runtime.get("torch_threads", 1)))
    try:
        torch.set_num_interop_threads(int(runtime.get("torch_interop_threads", 1)))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    output_dir = Path(config["output_dir"])
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    config_sha256 = hashlib.sha256(
        json.dumps(public_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if metadata_path.exists() and any(runs_dir.glob("*.csv")):
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_sha256:
            raise RuntimeError(
                "Configuration changed for an output directory containing "
                "completed runs. Use a new output_dir."
            )
    write_metadata(
        public_config,
        output_dir,
        config_file,
        {
            **config["_provenance"],
            "execution_input_manifest": input_manifest,
        },
    )

    shards = []
    pending = []
    for env_spec in config["envs"]:
        env_seeds = env_spec.get("seeds", config["seeds"])
        for agent_spec in config["agents"]:
            for seed_value in env_seeds:
                seed = int(seed_value)
                shard = runs_dir / (
                    f"{_slug(env_spec.get('name', env_spec['id']))}__"
                    f"{_slug(agent_spec['name'])}__seed_{seed:03d}.csv"
                )
                shards.append(shard)
                if shard.exists():
                    continue
                pending.append(
                    (config, env_spec, agent_spec, seed, shard)
                )

    workers = max(1, int(runtime.get("workers", 1)))
    if workers == 1:
        for task in pending:
            _run_single(*task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_single, *task) for task in pending
            ]
            for future in as_completed(futures):
                future.result()

    raw_path = output_dir / "raw.csv"
    _combine_shards(shards, raw_path)
    return raw_path
