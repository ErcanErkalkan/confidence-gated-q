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

import gymnasium
import numpy as np
import torch

from .agents import BaseAgent, create_agent
from .encoding import ObservationEncoder
from .envs import episode_succeeded, make_env


FIELDNAMES = [
    "experiment",
    "environment",
    "agent",
    "seed",
    "phase",
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
    "mean_gate",
    "support_abstention_rate",
    "global_tabular_error",
    "global_neural_error",
    "visited_states",
]


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

    for length in range(1, max_steps + 1):
        epsilon = (
            epsilon_at(agent.environment_steps, epsilon_params) if training else 0.0
        )
        action = agent.act(encoded.vector, encoded.key, epsilon)
        next_observation, reward, terminated, truncated, _ = env.step(action)
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
        env.close()
    return rows


def write_metadata(config: dict[str, Any], output_dir: Path) -> None:
    config_text = json.dumps(config, sort_keys=True).encode("utf-8")
    packages = {}
    for name in (
        "gymnasium",
        "minigrid",
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
    writer.writerow(
        {
            "experiment": config["experiment_name"],
            "environment": env_spec.get("name", env_spec["id"]),
            "agent": agent_spec["name"],
            "seed": seed,
            "phase": phase,
            "checkpoint": checkpoint,
            "episode": episode,
            **metrics,
            "environment_steps": agent.environment_steps,
            "gradient_updates": agent.gradient_updates,
            "elapsed_seconds": elapsed_seconds,
            "training_elapsed_seconds": (
                elapsed_seconds - evaluation_elapsed_seconds
            ),
            "evaluation_elapsed_seconds": evaluation_elapsed_seconds,
            "mean_gate": diagnostics.get("mean_gate", ""),
            "support_abstention_rate": diagnostics.get(
                "support_abstention_rate", ""
            ),
            "global_tabular_error": diagnostics.get(
                "global_tabular_error", ""
            ),
            "global_neural_error": diagnostics.get("global_neural_error", ""),
            "visited_states": diagnostics.get("visited_states", ""),
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
                max_steps = min(
                    int(env_spec["max_steps"]),
                    training_steps - agent.environment_steps,
                )
                for length in range(1, max_steps + 1):
                    epsilon = epsilon_at(
                        agent.environment_steps,
                        agent_spec.get("params", {}),
                    )
                    action = agent.act(
                        encoded.vector, encoded.key, epsilon
                    )
                    next_observation, reward, terminated, truncated, _ = (
                        env.step(action)
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
                    "success": float(
                        episode_succeeded(
                            env_spec.get(
                                "success_mode", "positive_terminal"
                            ),
                            terminated,
                            truncated,
                            final_reward,
                        )
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
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if metadata_path.exists() and any(runs_dir.glob("*.csv")):
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_sha256:
            raise RuntimeError(
                "Configuration changed for an output directory containing "
                "completed runs. Use a new output_dir."
            )
    write_metadata(config, output_dir)

    shards = []
    pending = []
    for env_spec in config["envs"]:
        for agent_spec in config["agents"]:
            for seed_value in config["seeds"]:
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
