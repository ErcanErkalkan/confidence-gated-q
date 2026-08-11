from __future__ import annotations

"""Resource-constrained supplemental continuous-control runner.

This runner intentionally supports one algorithm/environment/training-seed per
invocation for auditable distributed execution. It reads the immutable YAML lock
and refuses unregistered algorithm/environment/seed combinations.
"""

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            digest, filename = line.split(maxsplit=1)
            out[Path(filename).name] = digest
    return out


def require_exact_runtime(cfg: dict[str, Any]) -> dict[str, str]:
    required = {
        "gymnasium": cfg["runtime"]["gymnasium"],
        "stable_baselines3": cfg["runtime"]["stable_baselines3"],
        "sb3_contrib": cfg["runtime"]["sb3_contrib"],
        "mujoco": cfg["runtime"]["mujoco"],
    }
    versions = {}
    for name, req in required.items():
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", None)
        if ver != req:
            raise RuntimeError(f"Locked runtime mismatch: {name}={ver!r}, required {req!r}")
        versions[name] = ver
    return versions


def import_rl():
    import gymnasium as gym
    import torch
    from stable_baselines3 import SAC
    from sb3_contrib import CrossQ, TQC
    return gym, torch, {"SAC": SAC, "CrossQ": CrossQ, "TQC": TQC}


def make_env(env_id: str, condition: str):
    gym, _, _ = import_rl()

    class ObsDelayOne(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._prev_obs = None

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self._prev_obs = np.array(obs, copy=True)
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self._prev_obs is None:
                raise RuntimeError("ObsDelayOne.step called before reset")
            presented = np.array(self._prev_obs, copy=True)
            self._prev_obs = np.array(obs, copy=True)
            info = dict(info)
            info["continuous_control_underlying_current_observation"] = np.array(obs, copy=True)
            return presented, reward, terminated, truncated, info

    class ActGain075(gym.ActionWrapper):
        def action(self, action):
            return np.clip(0.75 * np.asarray(action), self.action_space.low, self.action_space.high)

    env = gym.make(env_id)
    if condition == "NOMINAL":
        return env
    if condition == "OBS_DELAY_1":
        return ObsDelayOne(env)
    if condition == "ACT_GAIN_075":
        return ActGain075(env)
    if condition == "COMBINED_DELAY_GAIN":
        return ObsDelayOne(ActGain075(env))
    raise ValueError(condition)


def replay_observations(model) -> np.ndarray:
    rb = model.replay_buffer
    n = rb.buffer_size if rb.full else rb.pos
    if n <= 0:
        raise RuntimeError("Replay buffer is empty")
    obs = np.asarray(rb.observations[:n])
    if obs.ndim >= 3 and obs.shape[1] == 1:
        obs = obs[:, 0]
    obs = obs.reshape(obs.shape[0], -1)
    return np.asarray(obs)


def build_support(replay_obs: np.ndarray, seed: int, cfg: dict[str, Any]):
    support_cfg = cfg["support"]
    rng = np.random.default_rng(41000 + seed)
    replay_raw = np.ascontiguousarray(replay_obs.reshape(len(replay_obs), -1))
    replay64 = np.asarray(replay_raw, dtype=np.float64)
    n = len(replay64)
    m = min(int(support_cfg["bank_max_n"]), n)
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    bank = replay64[idx]
    mean = bank.mean(axis=0)
    std = np.maximum(bank.std(axis=0), float(support_cfg["standardization"]["std_floor"]))
    z = (bank - mean) / std
    tree = cKDTree(z)

    sub_n = min(int(support_cfg["calibration_subset_max_n"]), m)
    sub_idx = rng.choice(m, size=sub_n, replace=False) if sub_n < m else np.arange(m)
    if m < 6:
        raise RuntimeError("At least six replay observations are required for k=5 self-excluded calibration")
    dist, _ = tree.query(z[sub_idx], k=6)
    r95 = float(np.percentile(dist[:, 5], 95))

    row_dtype = np.dtype((np.void, replay_raw.dtype.itemsize * replay_raw.shape[1]))
    exact_keys = np.unique(replay_raw.view(row_dtype).reshape(-1))
    return {
        "bank": bank, "mean": mean, "std": std, "tree": tree, "r95": r95,
        "exact_keys": exact_keys, "exact_row_dtype": row_dtype,
        "exact_scalar_dtype": replay_raw.dtype,
    }


def support_metrics(obs_rows: np.ndarray, support: dict[str, Any]) -> dict[str, Any]:
    obs64 = np.asarray(obs_rows, dtype=np.float64).reshape(len(obs_rows), -1)
    z = (obs64 - support["mean"]) / support["std"]
    dist, _ = support["tree"].query(z, k=5)
    d1 = dist[:, 0]
    d5 = dist[:, 4]
    rho = d5 / max(float(support["r95"]), 1e-12)
    supported = rho <= 1.0
    coverage = float(np.mean(supported))

    qraw = np.asarray(obs_rows, dtype=support["exact_scalar_dtype"]).reshape(len(obs_rows), -1)
    qkeys = np.ascontiguousarray(qraw).view(support["exact_row_dtype"]).reshape(-1)
    exact = float(np.mean(np.isin(qkeys, support["exact_keys"])))
    return {
        "support_coverage": coverage,
        "low_support_exposure": 1.0 - coverage,
        "support_decisions": int(len(rho)),
        "supported_decisions": int(np.sum(supported)),
        "mean_d1": float(np.mean(d1)),
        "median_d1": float(np.median(d1)),
        "mean_d5": float(np.mean(d5)),
        "median_d5": float(np.median(d5)),
        "mean_rho": float(np.mean(rho)),
        "median_rho": float(np.median(rho)),
        "exact_recurrence_rate": exact,
    }


def critic_diagnostics(model, algorithm: str, obs_rows: np.ndarray) -> dict[str, float | None]:
    _, torch, _ = import_rl()
    gaps = []
    iqrs = []
    was_training = bool(getattr(model.policy, "training", False))
    try:
        model.policy.set_training_mode(False)
    except Exception:
        pass
    for obs in obs_rows:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        action_np, _ = model.predict(obs, deterministic=True)
        action_tensor = torch.as_tensor(np.asarray(action_np), dtype=torch.float32, device=model.device).reshape(1, -1)
        with torch.no_grad():
            out = model.critic(obs_tensor, action_tensor)
        if isinstance(out, tuple):
            vals = [x.detach().cpu().numpy().reshape(-1) for x in out]
            if len(vals) >= 2:
                gaps.append(abs(float(vals[0].mean()) - float(vals[1].mean())))
        else:
            arr = out.detach().cpu().numpy()
            # TQC is expected to contain batch x critics x quantiles, but this
            # branch is dimension-robust to implementation orientation.
            arr = np.asarray(arr)
            if arr.ndim >= 3:
                # choose the dimension of size 1 as batch when possible
                b_axis = 0 if arr.shape[0] == 1 else None
                if b_axis == 0:
                    sample = arr[0]
                else:
                    sample = arr.reshape(-1, arr.shape[-1])
                if sample.ndim == 2 and sample.shape[0] >= 2:
                    means = sample.mean(axis=-1)
                    gaps.append(abs(float(means[0]) - float(means[1])))
                    flat = sample.reshape(-1)
                    iqrs.append(float(np.quantile(flat, 0.75) - np.quantile(flat, 0.25)))
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                gaps.append(abs(float(arr[0, 0]) - float(arr[0, 1])))
    try:
        model.policy.set_training_mode(was_training)
    except Exception:
        pass
    return {
        "critic_gap_mean": float(np.mean(gaps)) if gaps else None,
        "tqc_quantile_iqr_mean": float(np.mean(iqrs)) if iqrs else None,
    }


def evaluate_returns_only(model, env_id: str, reset_seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    was_training = bool(getattr(model.policy, "training", False))
    try:
        model.policy.set_training_mode(False)
    except Exception:
        pass
    try:
        for reset_seed in reset_seeds:
            env = make_env(env_id, "NOMINAL")
            obs, _ = env.reset(seed=reset_seed)
            total = 0.0
            steps = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            env.close()
            rows.append({"eval_reset_seed": int(reset_seed), "episode_return": total, "episode_steps": steps})
    finally:
        try:
            model.policy.set_training_mode(was_training)
        except Exception:
            pass
    return rows


def evaluate(model, env_id: str, condition: str, reset_seeds: list[int], support: dict[str, Any], algorithm: str):
    rows = []
    episode_summaries = []
    for reset_seed in reset_seeds:
        env = make_env(env_id, condition)
        obs, info = env.reset(seed=reset_seed)
        done = False
        total = 0.0
        observations = []
        nonfinite = 0
        episode_steps = 0
        terminated_final = False
        truncated_final = False
        while not done:
            observations.append(np.asarray(obs, dtype=np.float64).reshape(-1))
            action, _ = model.predict(obs, deterministic=True)
            if not np.all(np.isfinite(action)) or not np.all(np.isfinite(obs)):
                nonfinite += 1
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(reward):
                nonfinite += 1
            total += float(reward)
            episode_steps += 1
            done = bool(terminated or truncated)
            terminated_final = bool(terminated)
            truncated_final = bool(truncated)
        env.close()
        obs_arr = np.asarray(observations, dtype=np.float64)
        sm = support_metrics(obs_arr, support)
        cd = critic_diagnostics(model, algorithm, obs_arr[:: max(1, len(obs_arr)//50)])
        summary = {
            "eval_reset_seed": reset_seed,
            "episode_return": total,
            "episode_steps": episode_steps,
            "terminated": terminated_final,
            "truncated": truncated_final,
            "unhealthy_termination": (terminated_final if env_id == "Walker2d-v5" else None),
            "nonfinite_count": nonfinite,
            **sm,
            **cd,
        }
        episode_summaries.append(summary)
    return episode_summaries


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def resolved_configuration(model, cls) -> dict[str, Any]:
    attrs = [
        "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau", "gamma",
        "train_freq", "gradient_steps", "action_noise", "replay_buffer_class",
        "replay_buffer_kwargs", "optimize_memory_usage", "policy_kwargs", "ent_coef",
        "target_update_interval", "target_entropy", "top_quantiles_to_drop_per_net",
    ]
    values = {}
    for name in attrs:
        if hasattr(model, name):
            values[name] = _jsonable(getattr(model, name))
    trainable = int(sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.policy.parameters()))
    return {
        "constructor_signature": str(inspect.signature(cls.__init__)),
        "resolved_attributes": values,
        "policy_repr": repr(model.policy),
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
    }


def pip_freeze_text() -> str:
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")


def make_checkpoint_callback(env_id: str, eval_seeds: list[int], every: int, out_csv: Path):
    from stable_baselines3.common.callbacks import BaseCallback
    try:
        import psutil
    except Exception:
        psutil = None

    class LockedCheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.next_checkpoint = int(every)
            self.peak_rss_bytes = 0
            self._header_written = False

        def _sample_rss(self):
            if psutil is not None:
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(psutil.Process(os.getpid()).memory_info().rss))

        def _append_rows(self, timestep: int, rows: list[dict[str, Any]]):
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self._header_written else "w"
            with out_csv.open(mode, newline="", encoding="utf-8") as f:
                fieldnames = ["timestep", "eval_reset_seed", "episode_return", "episode_steps"]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not self._header_written:
                    w.writeheader()
                for row in rows:
                    w.writerow({"timestep": int(timestep), **row})
            self._header_written = True

        def _on_step(self) -> bool:
            if self.n_calls % 1000 == 0:
                self._sample_rss()
            if int(self.num_timesteps) >= self.next_checkpoint:
                self._sample_rss()
                rows = evaluate_returns_only(self.model, env_id, eval_seeds)
                self._append_rows(self.next_checkpoint, rows)
                self.next_checkpoint += int(every)
            return True

        def _on_training_end(self) -> None:
            self._sample_rss()

    return LockedCheckpointCallback()


def resolved_constructor(cls, policy: str, env, seed: int):
    # Instantiate only with the explicitly common fixed fields; all other
    # constructor values come from the pinned implementation defaults.
    model = cls(policy, env, seed=seed, device="cpu", verbose=0)
    return model, resolved_configuration(model, cls)


def runtime_manifest(cfg: dict[str, Any], versions: dict[str, str], lock_yaml: Path, lock_manifest: Path) -> dict[str, Any]:
    import torch
    return {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {**versions, "numpy": np.__version__, "scipy": importlib.import_module("scipy").__version__, "torch": torch.__version__},
        "lock_yaml_sha256": sha256(lock_yaml),
        "lock_manifest_sha256": sha256(lock_manifest),
    }


def train_one(args, cfg):
    versions = require_exact_runtime(cfg)
    gym, torch, algos = import_rl()
    torch.set_num_threads(int(args.torch_threads))
    torch.set_num_interop_threads(int(args.torch_interop_threads))
    allowed_envs = [e["id"] for e in cfg["environments"]]
    allowed_algos = [a["id"] for a in cfg["algorithms"]]
    if args.environment not in allowed_envs:
        raise ValueError(f"Unregistered environment: {args.environment}")
    if args.algorithm not in allowed_algos:
        raise ValueError(f"Unregistered algorithm: {args.algorithm}")

    if args.mode == "smoke":
        allowed_seeds = cfg["seeds"]["smoke"]
        total_steps = int(args.smoke_steps)
        if total_steps > int(cfg["budget"]["smoke_train_steps_max"]):
            raise ValueError("Smoke steps exceed locked maximum")
        eval_seeds = cfg["seeds"]["final_eval_reset"][:2]
        performance_evidence = False
        inferential_status = "NONPERFORMANCE_SMOKE"
    else:
        allowed_seeds = cfg["seeds"]["supplemental_training"]
        total_steps = int(cfg["budget"]["supplemental_train_steps"])
        eval_seeds = cfg["seeds"]["final_eval_reset"]
        performance_evidence = True
        inferential_status = "SUPPLEMENTAL_NONCONFIRMATORY"
    if args.seed not in allowed_seeds:
        raise ValueError(f"Seed {args.seed} is not registered for mode={args.mode}")

    outdir = args.output / args.mode / args.algorithm / args.environment / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(cfg, versions, args.lock_yaml, args.lock_manifest)
    manifest["evidence_class"] = "RESOURCE_CONSTRAINED_SUPPLEMENTAL_NONCONFIRMATORY" if args.mode == "supplemental" else "NONPERFORMANCE_SMOKE"
    manifest["torch_threads"] = int(args.torch_threads)
    manifest["torch_interop_threads"] = int(args.torch_interop_threads)
    freeze = pip_freeze_text()
    (outdir / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    manifest["pip_freeze_sha256"] = hashlib.sha256(freeze.encode("utf-8")).hexdigest()
    manifest["expected_lock_hashes"] = parse_manifest(args.lock_manifest)
    manifest["runner_sha256"] = sha256(Path(__file__))
    (outdir / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = make_env(args.environment, "NOMINAL")
    model, ctor = resolved_constructor(algos[args.algorithm], "MlpPolicy", env, args.seed)
    (outdir / "resolved_configuration.json").write_text(json.dumps(ctor, indent=2), encoding="utf-8")

    checkpoint_csv = outdir / "checkpoint_results.csv"
    checkpoint_cb = None
    if args.mode == "supplemental":
        checkpoint_cb = make_checkpoint_callback(
            args.environment,
            [int(x) for x in cfg["seeds"]["checkpoint_eval_reset"]],
            int(cfg["budget"]["eval_checkpoint_every_steps"]),
            checkpoint_csv,
        )

    started = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False, callback=checkpoint_cb)
    elapsed = time.time() - started
    model.save(outdir / "model")
    replay_obs = replay_observations(model)
    support = build_support(replay_obs, args.seed, cfg)
    np.savez_compressed(outdir / "support_stats.npz", mean=support["mean"], std=support["std"], r95=np.array([support["r95"]]))

    conditions = [c["id"] for c in cfg["conditions"]]
    all_rows = []
    for condition in conditions:
        eps = evaluate(model, args.environment, condition, eval_seeds, support, args.algorithm)
        for ep in eps:
            all_rows.append({
                "mode": args.mode,
                "algorithm": args.algorithm,
                "environment": args.environment,
                "training_seed": args.seed,
                "condition": condition,
                "performance_evidence": performance_evidence,
                "inferential_status": inferential_status,
                "protocol_lock_sha256": manifest["lock_yaml_sha256"],
                "lock_yaml_sha256": manifest["lock_yaml_sha256"],
                **ep,
            })
    env.close()
    with (outdir / "episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    summary = {
        "status": "COMPLETE",
        "mode": args.mode,
        "algorithm": args.algorithm,
        "environment": args.environment,
        "seed": args.seed,
        "train_steps": total_steps,
        "train_elapsed_seconds": elapsed,
        "replay_observations": int(len(replay_obs)),
        "support_r95": float(support["r95"]),
        "episodes": len(all_rows),
        "checkpoint_rows": (sum(1 for _ in checkpoint_csv.open("r", encoding="utf-8")) - 1) if checkpoint_csv.exists() else 0,
        "peak_rss_bytes": int(getattr(checkpoint_cb, "peak_rss_bytes", 0) or 0),
        "trainable_parameter_count": int(ctor["trainable_parameter_count"]),
        "performance_evidence": performance_evidence,
        "inferential_status": inferential_status,
        "torch_threads": int(args.torch_threads),
        "torch_interop_threads": int(args.torch_interop_threads),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-yaml", type=Path, required=True)
    ap.add_argument("--lock-manifest", type=Path, required=True)
    ap.add_argument("--mode", choices=["smoke", "supplemental"], required=True)
    ap.add_argument("--algorithm", choices=["SAC", "CrossQ", "TQC"], required=True)
    ap.add_argument("--environment", choices=["HalfCheetah-v5", "Walker2d-v5"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke-steps", type=int, default=2000)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--torch-interop-threads", type=int, default=1)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.lock_yaml.read_text(encoding="utf-8"))

    expected = parse_manifest(args.lock_manifest).get(args.lock_yaml.name)
    actual = sha256(args.lock_yaml)
    if expected != actual:
        raise RuntimeError(f"Lock hash mismatch: expected={expected} actual={actual}")
    train_one(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
