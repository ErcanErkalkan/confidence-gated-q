from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

from .gym_compat import gym, spaces, HAS_GYMNASIUM
import numpy as np


@dataclass(frozen=True)
class EncodedObservation:
    vector: np.ndarray
    key: Hashable


class ObservationEncoder:
    def __init__(self, observation_space: gym.Space):
        self.space = observation_space
        self.input_dim = int(gym.spaces.utils.flatdim(observation_space))

    def encode(self, observation: Any) -> EncodedObservation:
        vector = gym.spaces.utils.flatten(self.space, observation).astype(
            np.float32, copy=False
        )
        if isinstance(self.space, gym.spaces.Box):
            high = gym.spaces.utils.flatten(self.space, self.space.high).astype(
                np.float32, copy=False
            )
            scale = np.where(np.isfinite(high) & (np.abs(high) > 1), np.abs(high), 1)
            vector = vector / scale

        if isinstance(observation, np.ndarray):
            key: Hashable = (
                observation.shape,
                observation.dtype.str,
                observation.tobytes(),
            )
        elif np.isscalar(observation):
            key = int(observation)
        else:
            key = tuple(np.asarray(vector, dtype=np.float32).round(6))

        return EncodedObservation(vector=vector, key=key)
