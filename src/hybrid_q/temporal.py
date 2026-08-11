from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn

from .gym_compat import gym, spaces


Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class _ObservationWrapper(gym.Env):
    """Small dependency-compatible observation wrapper base."""

    metadata: dict[str, Any] = {}

    def __init__(self, env: gym.Env):
        self.env = env
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def close(self) -> None:
        self.env.close()


class FrameStackObservationWrapper(_ObservationWrapper):
    """Concatenate a fixed number of observations with reset-local history."""

    def __init__(self, env: gym.Env, stack_size: int):
        super().__init__(env)
        if stack_size < 1:
            raise ValueError("stack_size must be positive")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("frame stacking requires a Box observation space")
        self.stack_size = int(stack_size)
        self._frames: deque[np.ndarray] = deque(maxlen=self.stack_size)
        low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = spaces.Box(
            low=float(np.min(low)),
            high=float(np.max(high)),
            shape=(low.size * self.stack_size,),
            dtype=np.float32,
        )

    def _stack(self) -> np.ndarray:
        return np.concatenate(tuple(self._frames)).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        frame = np.asarray(observation, dtype=np.float32).reshape(-1)
        self._frames.clear()
        for _ in range(self.stack_size):
            self._frames.append(frame.copy())
        return self._stack(), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(
            np.asarray(observation, dtype=np.float32).reshape(-1).copy()
        )
        return self._stack(), reward, terminated, truncated, info


class FilteredBeliefObservationWrapper(_ObservationWrapper):
    """Expose an explicit exponential-filter belief state."""

    def __init__(self, env: gym.Env, alpha: float):
        super().__init__(env)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("filter alpha must lie in (0, 1]")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("belief filtering requires a Box observation space")
        self.alpha = float(alpha)
        self.observation_space = env.observation_space
        self._belief: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._belief = np.asarray(observation, dtype=np.float32).copy()
        return self._belief.copy(), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current = np.asarray(observation, dtype=np.float32)
        if self._belief is None:
            self._belief = current.copy()
        else:
            self._belief = (
                self.alpha * current + (1.0 - self.alpha) * self._belief
            ).astype(np.float32)
        info = dict(info)
        info["temporal_filter_alpha"] = self.alpha
        return self._belief.copy(), reward, terminated, truncated, info


class SequenceReplayBuffer:
    """Episode-bounded replay with deterministic contiguous sequence sampling."""

    def __init__(self, capacity: int, sequence_length: int, seed: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.episodes: deque[list[Transition]] = deque()
        self.current_episode: list[Transition] = []
        self.rng = np.random.default_rng(seed)
        self.transition_count = 0

    def add(self, transition: Transition) -> None:
        state, action, reward, next_state, done = transition
        item: Transition = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
        )
        self.current_episode.append(item)
        self.transition_count += 1
        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
        self._trim()

    def end_episode(self) -> None:
        """Close a truncated collector episode without fabricating a done flag."""

        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            self._trim()

    def _trim(self) -> None:
        while self.transition_count > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.transition_count -= len(removed)

    def eligible_sequence_count(self) -> int:
        return sum(
            max(0, len(episode) - self.sequence_length + 1)
            for episode in self.episodes
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        windows = [
            (episode_index, start)
            for episode_index, episode in enumerate(self.episodes)
            for start in range(len(episode) - self.sequence_length + 1)
        ]
        if batch_size > len(windows):
            raise ValueError("insufficient complete replay sequences")
        chosen = self.rng.choice(len(windows), size=batch_size, replace=False)
        sequences = [
            list(self.episodes[windows[int(index)][0]])[
                windows[int(index)][1] : windows[int(index)][1]
                + self.sequence_length
            ]
            for index in chosen
        ]
        states, actions, rewards, next_states, dones = zip(
            *(tuple(zip(*sequence)) for sequence in sequences)
        )
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return self.transition_count


class RecurrentQNetwork(nn.Module):
    """GRU action-value network with an unchanged action-value output contract."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, action_dim)

    def forward(
        self,
        states: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim == 2:
            states = states.unsqueeze(1)
        features = self.input_projection(states)
        recurrent, next_hidden = self.gru(features, hidden)
        return self.output(recurrent), next_hidden
