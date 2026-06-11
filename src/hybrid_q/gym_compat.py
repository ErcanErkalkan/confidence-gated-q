from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised in fully provisioned environments
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    HAS_GYMNASIUM = True
except ModuleNotFoundError:  # lightweight fallback for artifact tests/smoke runs
    HAS_GYMNASIUM = False

    class Space:
        pass

    class Discrete(Space):
        def __init__(self, n: int):
            self.n = int(n)
            self.shape = ()
            self.dtype = np.int64

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class Box(Space):
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

    def _flatdim(space: Space) -> int:
        if isinstance(space, Discrete):
            return int(space.n)
        if isinstance(space, Box):
            return int(np.prod(space.shape))
        raise TypeError(f"unsupported fallback space: {type(space)!r}")

    def _flatten(space: Space, observation: Any) -> np.ndarray:
        if isinstance(space, Discrete):
            one_hot = np.zeros(space.n, dtype=np.float32)
            one_hot[int(observation)] = 1.0
            return one_hot
        if isinstance(space, Box):
            return np.asarray(observation, dtype=np.float32).reshape(-1)
        raise TypeError(f"unsupported fallback space: {type(space)!r}")

    class _Utils:
        flatdim = staticmethod(_flatdim)
        flatten = staticmethod(_flatten)

    spaces = SimpleNamespace(Discrete=Discrete, Box=Box, utils=_Utils)

    class Env:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: Any = None):
            self.np_random = np.random.default_rng(seed)
            return None, {}

        def close(self) -> None:
            return None

    registry = {
        "FrozenLake-v1",
        "CliffWalking-v1",
        "Taxi-v4",
    }

    def make(env_id: str, **kwargs):
        # Imported lazily to avoid a circular import during module initialization.
        from .envs import (
            FallbackCliffWalkingEnv,
            FallbackFrozenLakeEnv,
            FallbackTaxiEnv,
        )

        if env_id == "FrozenLake-v1":
            return FallbackFrozenLakeEnv(**kwargs)
        if env_id == "CliffWalking-v1":
            return FallbackCliffWalkingEnv(**kwargs)
        if env_id == "Taxi-v4":
            return FallbackTaxiEnv(**kwargs)
        raise KeyError(f"fallback gym registry does not contain {env_id!r}")

    gym = SimpleNamespace(
        Env=Env,
        Space=Space,
        spaces=spaces,
        registry=registry,
        make=make,
        __version__="fallback-local",
    )
