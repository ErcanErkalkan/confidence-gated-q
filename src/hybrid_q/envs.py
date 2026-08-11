from __future__ import annotations

import contextlib
from collections import deque
import importlib.util
import io
from typing import Any

from .gym_compat import gym, spaces, HAS_GYMNASIUM
from .temporal import (
    FilteredBeliefObservationWrapper,
    FrameStackObservationWrapper,
)
import numpy as np


ENV_ID_COMPATIBILITY = {
    # Gymnasium 1.3 renamed Taxi-v3 to Taxi-v4. The dynamics remain exposed
    # through TaxiEnv; metadata records both the requested and resolved IDs.
    "Taxi-v3": "Taxi-v4",
}


class StructuredFourRoomsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        size: int = 9,
        goal_split: str = "train",
        slip_probability: float = 0.0,
        max_steps: int = 200,
    ):
        if size < 7 or size % 2 == 0:
            raise ValueError("size must be an odd integer >= 7")
        if goal_split not in {"train", "test", "all"}:
            raise ValueError("goal_split must be train, test, or all")
        self.size = size
        self.goal_split = goal_split
        self.slip_probability = float(slip_probability)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.walls = self._build_walls()
        self.valid_cells = [
            (x, y)
            for x in range(size)
            for y in range(size)
            if (x, y) not in self.walls
        ]
        train_goals = [
            cell
            for cell in self.valid_cells
            if self._goal_partition(cell) == "train"
        ]
        test_goals = [
            cell
            for cell in self.valid_cells
            if self._goal_partition(cell) == "test"
        ]
        self.goals = {
            "train": train_goals,
            "test": test_goals,
            "all": self.valid_cells,
        }[goal_split]
        self.agent_position = (0, 0)
        self.goal_position = (0, 0)
        self.steps = 0

    def _build_walls(self) -> set[tuple[int, int]]:
        middle = self.size // 2
        doors = {
            (middle, 1),
            (middle, self.size - 2),
            (1, middle),
            (self.size - 2, middle),
        }
        walls = {
            (middle, coordinate) for coordinate in range(self.size)
        } | {
            (coordinate, middle) for coordinate in range(self.size)
        }
        return walls - doors

    def _goal_partition(self, cell: tuple[int, int]) -> str:
        x, y = cell
        return "train" if (x * 3 + y * 5) % 4 else "test"

    def _observation(self) -> np.ndarray:
        scale = float(self.size - 1)
        return np.asarray(
            [
                self.agent_position[0] / scale,
                self.agent_position[1] / scale,
                self.goal_position[0] / scale,
                self.goal_position[1] / scale,
            ],
            dtype=np.float32,
        )

    def _latent_state_id(self) -> str:
        return (
            f"agent={self.agent_position[0]},{self.agent_position[1]}|"
            f"goal={self.goal_position[0]},{self.goal_position[1]}"
        )

    def _destination(
        self, position: tuple[int, int], action: int
    ) -> tuple[int, int]:
        moves = ((0, 1), (1, 0), (0, -1), (-1, 0))
        dx, dy = moves[int(action)]
        candidate = (position[0] + dx, position[1] + dy)
        if (
            candidate in self.walls
            or not 0 <= candidate[0] < self.size
            or not 0 <= candidate[1] < self.size
        ):
            return position
        return candidate

    def _distance_to_goal(self, start: tuple[int, int]) -> int:
        if start == self.goal_position:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            position, distance = queue.popleft()
            for action in range(self.action_space.n):
                candidate = self._destination(position, action)
                if candidate in visited:
                    continue
                if candidate == self.goal_position:
                    return distance + 1
                visited.add(candidate)
                queue.append((candidate, distance + 1))
        return self.size * self.size + 1

    def _optimal_action(self) -> int | None:
        distances = [
            self._distance_to_goal(
                self._destination(self.agent_position, action)
            )
            for action in range(self.action_space.n)
        ]
        minimum = min(distances)
        actions = [
            action
            for action, distance in enumerate(distances)
            if distance == minimum
        ]
        return actions[0] if len(actions) == 1 else None

    def _apply_action(
        self, action: int, slip_probability: float
    ) -> tuple[int, bool]:
        executed_action = int(action)
        if self.np_random.random() < float(slip_probability):
            executed_action = int(
                self.np_random.integers(self.action_space.n)
            )
        destination = self._destination(
            self.agent_position, executed_action
        )
        hit_wall = destination == self.agent_position
        self.agent_position = destination
        return executed_action, hit_wall

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        goal_index = int(self.np_random.integers(len(self.goals)))
        self.goal_position = self.goals[goal_index]
        starts = [
            cell for cell in self.valid_cells if cell != self.goal_position
        ]
        start_index = int(self.np_random.integers(len(starts)))
        self.agent_position = starts[start_index]
        self.steps = 0
        return self._observation(), {}

    def step(self, action: int):
        _, hit_wall = self._apply_action(action, self.slip_probability)
        self.steps += 1
        terminated = self.agent_position == self.goal_position
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else (-0.02 if hit_wall else -0.01)
        return self._observation(), reward, terminated, truncated, {}


class TransitionDynamicsShiftFourRoomsEnv(StructuredFourRoomsEnv):
    """FourRooms with a locked change to the action-outcome kernel only."""

    def __init__(
        self,
        size: int = 9,
        goal_split: str = "all",
        max_steps: int = 160,
        shift_after: int = 12000,
        pre_shift_slip_probability: float = 0.05,
        post_shift_slip_probability: float = 0.15,
    ):
        if shift_after < 0:
            raise ValueError("shift_after must be non-negative")
        for name, value in (
            ("pre_shift_slip_probability", pre_shift_slip_probability),
            ("post_shift_slip_probability", post_shift_slip_probability),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        super().__init__(
            size=size,
            goal_split=goal_split,
            slip_probability=pre_shift_slip_probability,
            max_steps=max_steps,
        )
        self.shift_after = int(shift_after)
        self.pre_shift_slip_probability = float(
            pre_shift_slip_probability
        )
        self.post_shift_slip_probability = float(
            post_shift_slip_probability
        )
        self.total_steps = 0

    def _post_shift(self) -> bool:
        return self.total_steps >= self.shift_after

    def _shift_info(
        self,
        *,
        post_shift: bool | None = None,
        latent_state_id: str | None = None,
        requested_action: int | None = None,
        executed_action: int | None = None,
    ) -> dict[str, Any]:
        post = self._post_shift() if post_shift is None else post_shift
        return {
            "post_shift": post,
            "shift_type": "transition_dynamics_shift",
            "shift_severity": self.post_shift_slip_probability,
            "shift_region": "global_action_outcome" if post else "none",
            "latent_state_id": (
                self._latent_state_id()
                if latent_state_id is None
                else latent_state_id
            ),
            "next_latent_state_id": self._latent_state_id(),
            "optimal_action": self._optimal_action(),
            "steps_since_shift": (
                max(0, self.total_steps - self.shift_after)
                if post else None
            ),
            "pre_shift_slip_probability": self.pre_shift_slip_probability,
            "post_shift_slip_probability": self.post_shift_slip_probability,
            "effective_slip_probability": (
                self.post_shift_slip_probability
                if post else self.pre_shift_slip_probability
            ),
            "requested_action": requested_action,
            "executed_action": executed_action,
        }

    def reset(self, *, seed: int | None = None, options=None):
        observation, _ = super().reset(seed=seed, options=options)
        return observation, self._shift_info()

    def step(self, action: int):
        post_shift = self._post_shift()
        origin = self._latent_state_id()
        optimal_action = self._optimal_action()
        slip_probability = (
            self.post_shift_slip_probability
            if post_shift
            else self.pre_shift_slip_probability
        )
        executed_action, hit_wall = self._apply_action(
            action, slip_probability
        )
        self.steps += 1
        terminated = self.agent_position == self.goal_position
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else (-0.02 if hit_wall else -0.01)
        info = self._shift_info(
            post_shift=post_shift,
            latent_state_id=origin,
            requested_action=int(action),
            executed_action=executed_action,
        )
        info["optimal_action"] = optimal_action
        self.total_steps += 1
        return self._observation(), reward, terminated, truncated, info


class ObservationShiftFourRoomsEnv(StructuredFourRoomsEnv):
    """FourRooms with fixed latent dynamics and a shifted sensor gain."""

    def __init__(
        self,
        size: int = 9,
        goal_split: str = "all",
        max_steps: int = 160,
        shift_after: int = 12000,
        slip_probability: float = 0.10,
        pre_shift_sensor_gain: float = 1.0,
        post_shift_sensor_gain: float = 0.85,
    ):
        if shift_after < 0:
            raise ValueError("shift_after must be non-negative")
        if not 0.0 < float(pre_shift_sensor_gain) <= 1.0:
            raise ValueError("pre_shift_sensor_gain must be in (0, 1]")
        if not 0.0 < float(post_shift_sensor_gain) <= 1.0:
            raise ValueError("post_shift_sensor_gain must be in (0, 1]")
        self.shift_after = int(shift_after)
        self.pre_shift_sensor_gain = float(pre_shift_sensor_gain)
        self.post_shift_sensor_gain = float(post_shift_sensor_gain)
        self.total_steps = 0
        super().__init__(
            size=size,
            goal_split=goal_split,
            slip_probability=slip_probability,
            max_steps=max_steps,
        )

    def _post_shift(self) -> bool:
        return self.total_steps >= self.shift_after

    def _latent_observation(self) -> np.ndarray:
        return StructuredFourRoomsEnv._observation(self)

    def _sensor_gain(self, post_shift: bool | None = None) -> float:
        post = self._post_shift() if post_shift is None else post_shift
        return (
            self.post_shift_sensor_gain
            if post else self.pre_shift_sensor_gain
        )

    def _observation_for_regime(self, post_shift: bool) -> np.ndarray:
        latent = self._latent_observation()
        gain = self._sensor_gain(post_shift)
        return np.asarray(
            0.5 + gain * (latent - 0.5), dtype=np.float32
        )

    def _observation(self) -> np.ndarray:
        return self._observation_for_regime(self._post_shift())

    def _shift_info(
        self,
        *,
        post_shift: bool | None = None,
        latent_state_id: str | None = None,
        observation: np.ndarray | None = None,
    ) -> dict[str, Any]:
        post = self._post_shift() if post_shift is None else post_shift
        latent_observation = self._latent_observation()
        observed = (
            self._observation_for_regime(post)
            if observation is None
            else observation
        )
        return {
            "post_shift": post,
            "shift_type": "observation_shift",
            "shift_severity": self.post_shift_sensor_gain,
            "shift_region": (
                "all_observation_channels" if post else "none"
            ),
            "latent_state_id": (
                self._latent_state_id()
                if latent_state_id is None
                else latent_state_id
            ),
            "next_latent_state_id": self._latent_state_id(),
            "optimal_action": self._optimal_action(),
            "steps_since_shift": (
                max(0, self.total_steps - self.shift_after)
                if post else None
            ),
            "latent_observation": latent_observation.tolist(),
            "observed_observation": observed.tolist(),
            "observation_perturbation": (
                observed - latent_observation
            ).tolist(),
            "pre_shift_sensor_gain": self.pre_shift_sensor_gain,
            "post_shift_sensor_gain": self.post_shift_sensor_gain,
            "effective_sensor_gain": self._sensor_gain(post),
            "transition_slip_probability": self.slip_probability,
        }

    def reset(self, *, seed: int | None = None, options=None):
        observation, _ = super().reset(seed=seed, options=options)
        return observation, self._shift_info(observation=observation)

    def step(self, action: int):
        post_shift = self._post_shift()
        origin = self._latent_state_id()
        optimal_action = self._optimal_action()
        _, hit_wall = self._apply_action(action, self.slip_probability)
        self.steps += 1
        terminated = self.agent_position == self.goal_position
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else (-0.02 if hit_wall else -0.01)
        observation = self._observation_for_regime(post_shift)
        info = self._shift_info(
            post_shift=post_shift,
            latent_state_id=origin,
            observation=observation,
        )
        info["optimal_action"] = optimal_action
        self.total_steps += 1
        return observation, reward, terminated, truncated, info


class ApplicationNavigationSupportShiftEnv(gym.Env):
    """Small warehouse-style navigation task with deployment goal shift."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        size: int = 9,
        goal_split: str = "train",
        slip_probability: float = 0.05,
        max_steps: int = 120,
        hold_penalty: float = 0.0,
        lambda_collision: float = 1.0,
        lambda_idle: float = 0.1,
        risk_penalty: float = 0.08,
    ):
        if size != 9:
            raise ValueError("application navigation currently requires size=9")
        if goal_split not in {"train", "test", "deployment", "all"}:
            raise ValueError(
                "goal_split must be train, test, deployment, or all"
            )
        self.size = int(size)
        self.goal_split = goal_split
        self.slip_probability = float(slip_probability)
        self.max_steps = int(max_steps)
        self.hold_penalty = float(hold_penalty)
        self.risk_penalty = float(risk_penalty)
        self.lambda_collision = float(lambda_collision)
        self.lambda_idle = float(lambda_idle)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.walls = self._build_walls()
        self.valid_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.walls
        ]
        self.train_goals = [(1, 1), (7, 1), (1, 7)]
        self.test_goals = [(7, 7), (5, 5)]
        self.goals = {
            "train": self.train_goals,
            "test": self.test_goals,
            "deployment": self.train_goals + self.test_goals,
            "all": self.valid_cells,
        }[goal_split]
        self.start_cells = [(1, 3), (3, 1), (5, 7), (7, 5)]
        self.risk_cells = {(2, 2), (2, 6), (6, 2), (6, 6)}
        self.agent_position = self.start_cells[0]
        self.goal_position = self.goals[0]
        self.steps = 0

    def _build_walls(self) -> set[tuple[int, int]]:
        walls = {(4, y) for y in range(self.size)}
        walls |= {(x, 4) for x in range(self.size)}
        doors = {(4, 1), (4, 7), (1, 4), (7, 4)}
        return walls - doors

    def _observation(self) -> np.ndarray:
        scale = float(self.size - 1)
        return np.asarray(
            [
                self.agent_position[0] / scale,
                self.agent_position[1] / scale,
                self.goal_position[0] / scale,
                self.goal_position[1] / scale,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.goal_position = self.goals[
            int(self.np_random.integers(len(self.goals)))
        ]
        starts = [
            cell for cell in self.start_cells if cell != self.goal_position
        ]
        self.agent_position = starts[
            int(self.np_random.integers(len(starts)))
        ]
        self.steps = 0
        return self._observation(), {
            "goal_split": self.goal_split,
            "goal_is_shifted": self.goal_position in self.test_goals,
        }

    def _step_with_risk_penalty(
        self, action: int, risk_penalty: float
    ):
        if self.np_random.random() < self.slip_probability:
            action = int(self.np_random.integers(self.action_space.n))
        moves = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0))
        dx, dy = moves[int(action)]
        candidate = (
            self.agent_position[0] + dx,
            self.agent_position[1] + dy,
        )
        collision = int(action) != 4 and (
            candidate in self.walls
            or not 0 <= candidate[0] < self.size
            or not 0 <= candidate[1] < self.size
        )
        if not collision:
            self.agent_position = candidate
        self.steps += 1
        terminated = self.agent_position == self.goal_position
        truncated = self.steps >= self.max_steps and not terminated
        in_risk_zone = self.agent_position in self.risk_cells
        idle = int(action) == 4
        if terminated:
            reward = 5.0
        else:
            reward = -0.02
            if collision:
                reward -= 0.23
            if in_risk_zone:
                reward -= float(risk_penalty)
            if idle:
                reward -= self.hold_penalty
        info = {
            "collision": collision,
            "risk_zone": in_risk_zone,
            "idle": idle,
            "lambda_collision": self.lambda_collision,
            "lambda_idle": self.lambda_idle,
            "goal_is_shifted": self.goal_position in self.test_goals,
        }
        return self._observation(), reward, terminated, truncated, info

    def step(self, action: int):
        return self._step_with_risk_penalty(action, self.risk_penalty)


class LocalizedRewardShiftNavigationEnv(
    ApplicationNavigationSupportShiftEnv
):
    """Navigation task whose reward changes only in locked risk cells."""

    def __init__(
        self,
        size: int = 9,
        goal_split: str = "deployment",
        slip_probability: float = 0.05,
        max_steps: int = 120,
        hold_penalty: float = 0.0,
        lambda_collision: float = 1.0,
        lambda_idle: float = 0.1,
        shift_after: int = 12000,
        pre_shift_risk_penalty: float = 0.08,
        post_shift_risk_penalty: float = 0.16,
    ):
        if shift_after < 0:
            raise ValueError("shift_after must be non-negative")
        if pre_shift_risk_penalty < 0.0 or post_shift_risk_penalty < 0.0:
            raise ValueError("risk penalties must be non-negative")
        super().__init__(
            size=size,
            goal_split=goal_split,
            slip_probability=slip_probability,
            max_steps=max_steps,
            hold_penalty=hold_penalty,
            risk_penalty=pre_shift_risk_penalty,
            lambda_collision=lambda_collision,
            lambda_idle=lambda_idle,
        )
        self.shift_after = int(shift_after)
        self.pre_shift_risk_penalty = float(pre_shift_risk_penalty)
        self.post_shift_risk_penalty = float(post_shift_risk_penalty)
        self.total_steps = 0

    def _post_shift(self) -> bool:
        return self.total_steps >= self.shift_after

    def _latent_state_id(self) -> str:
        return (
            f"agent={self.agent_position[0]},{self.agent_position[1]}|"
            f"goal={self.goal_position[0]},{self.goal_position[1]}"
        )

    def _shift_info(
        self,
        *,
        post_shift: bool | None = None,
        latent_state_id: str | None = None,
        transition_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        post = self._post_shift() if post_shift is None else post_shift
        info = {} if transition_info is None else dict(transition_info)
        localized = bool(
            post
            and info.get("risk_zone", self.agent_position in self.risk_cells)
        )
        info.update(
            {
                "post_shift": post,
                "shift_type": "localized_multistep_reward_or_policy_shift",
                "shift_severity": self.post_shift_risk_penalty,
                "shift_region": (
                    "localized_risk_cells"
                    if localized
                    else ("unaffected" if post else "none")
                ),
                "latent_state_id": (
                    self._latent_state_id()
                    if latent_state_id is None
                    else latent_state_id
                ),
                "next_latent_state_id": self._latent_state_id(),
                "optimal_action": None,
                "steps_since_shift": (
                    max(0, self.total_steps - self.shift_after)
                    if post else None
                ),
                "pre_shift_risk_penalty": self.pre_shift_risk_penalty,
                "post_shift_risk_penalty": self.post_shift_risk_penalty,
                "effective_risk_penalty": (
                    self.post_shift_risk_penalty
                    if post else self.pre_shift_risk_penalty
                ),
                "localized_shift_applied": localized,
                "localized_risk_cells": sorted(self.risk_cells),
            }
        )
        return info

    def reset(self, *, seed: int | None = None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return observation, self._shift_info(transition_info=info)

    def step(self, action: int):
        post_shift = self._post_shift()
        origin = self._latent_state_id()
        penalty = (
            self.post_shift_risk_penalty
            if post_shift else self.pre_shift_risk_penalty
        )
        observation, reward, terminated, truncated, info = (
            self._step_with_risk_penalty(action, penalty)
        )
        info = self._shift_info(
            post_shift=post_shift,
            latent_state_id=origin,
            transition_info=info,
        )
        self.total_steps += 1
        return observation, reward, terminated, truncated, info


class ReliabilityShiftBanditEnv(gym.Env):
    """Deliberate mechanism diagnostic for stale recurring-state support.

    This one-step contextual bandit changes its optimal-action threshold after
    repeated exposure to the same finite context set. It isolates whether a
    relative-reliability signal can detect stale high-count memory. It is not a
    general reinforcement-learning benchmark or an application model.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        context_count: int = 41,
        regime: str = "switch",
        shift_after: int = 6000,
        pre_boundary: float = 0.5,
        post_boundary: float = 0.3,
    ):
        if context_count < 5:
            raise ValueError("context_count must be at least 5")
        if regime not in {"pre", "post", "switch"}:
            raise ValueError("regime must be pre, post, or switch")
        if not 0.0 < pre_boundary < 1.0:
            raise ValueError("pre_boundary must be in (0, 1)")
        if not 0.0 < post_boundary < 1.0:
            raise ValueError("post_boundary must be in (0, 1)")
        self.context_count = int(context_count)
        self.regime = regime
        self.shift_after = int(shift_after)
        self.pre_boundary = float(pre_boundary)
        self.post_boundary = float(post_boundary)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.total_steps = 0
        self.context_index = 0

    def _post_shift(self) -> bool:
        return self.regime == "post" or (
            self.regime == "switch" and self.total_steps >= self.shift_after
        )

    def _observation(self) -> np.ndarray:
        return np.asarray(
            [self.context_index / (self.context_count - 1)],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.context_index = int(
            self.np_random.integers(self.context_count)
        )
        return self._observation(), self._oracle_info()

    def _oracle_info(self) -> dict[str, Any]:
        post_shift = self._post_shift()
        boundary = self.post_boundary if post_shift else self.pre_boundary
        context = self.context_index / (self.context_count - 1)
        optimal_action = int(context >= boundary)
        lower = min(self.pre_boundary, self.post_boundary)
        upper = max(self.pre_boundary, self.post_boundary)
        if lower <= context < upper:
            shift_region = "changed_optimal_action"
        else:
            shift_region = f"stable_action_{optimal_action}"
        steps_since_shift = (
            max(0, self.total_steps - self.shift_after)
            if post_shift and self.regime == "switch"
            else (0 if post_shift else None)
        )
        true_action_values = [
            1.0 if action == optimal_action else -1.0
            for action in range(self.action_space.n)
        ]
        return {
            "post_shift": post_shift,
            "optimal_action": optimal_action,
            "true_action_values": true_action_values,
            "shift_region": shift_region,
            "steps_since_shift": steps_since_shift,
        }

    def step(self, action: int):
        post_shift = self._post_shift()
        oracle = self._oracle_info()
        optimal_action = int(oracle["optimal_action"])
        success = int(action) == optimal_action
        reward = 1.0 if success else -1.0
        self.total_steps += 1
        return self._observation(), reward, True, False, oracle


def has_uav_backend() -> bool:
    return (
        importlib.util.find_spec("gym_pybullet_drones") is not None
        and importlib.util.find_spec("pybullet") is not None
    )


class PyBulletUAVWaypointSupportShiftEnv(gym.Env):
    """Crazyflie waypoint task using the gym-pybullet-drones physics backend."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_split: str = "train",
        physics: str = "pyb",
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        action_repeat: int = 4,
        max_steps: int = 60,
        speed_fraction: float = 0.8,
        goal_tolerance: float = 0.13,
        state_quantization: float = 0.05,
        initial_position_jitter: float = 0.02,
        initial_attitude_jitter: float = 0.01,
        wind_force_std: float = 0.0,
        lambda_collision: float = 1.0,
        lambda_idle: float = 0.05,
    ):
        if not has_uav_backend():
            raise ImportError(
                "PyBullet UAV validation requires the optional 'uav' "
                "dependencies. Install with: python -m pip install -e .[uav]"
            )
        if target_split not in {"train", "deployment", "all"}:
            raise ValueError(
                "target_split must be train, deployment, or all"
            )
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")

        import pybullet as pybullet
        from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
        from gym_pybullet_drones.utils.enums import Physics

        self.target_split = target_split
        self.action_repeat = int(action_repeat)
        self.max_steps = int(max_steps)
        self.speed_fraction = float(speed_fraction)
        self.goal_tolerance = float(goal_tolerance)
        self.state_quantization = float(state_quantization)
        self.initial_position_jitter = float(initial_position_jitter)
        self.initial_attitude_jitter = float(initial_attitude_jitter)
        self.wind_force_std = float(wind_force_std)
        self.lambda_collision = float(lambda_collision)
        self.lambda_idle = float(lambda_idle)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )
        self.train_targets = np.asarray(
            [
                [0.60, 0.00, 0.55],
                [-0.60, 0.00, 0.55],
                [0.00, 0.60, 0.65],
                [0.00, -0.60, 0.65],
                [0.45, 0.45, 0.75],
                [-0.45, 0.45, 0.75],
            ],
            dtype=np.float32,
        )
        self.deployment_targets = np.asarray(
            [
                [0.58, -0.58, 0.70],
                [-0.58, -0.58, 0.70],
                [0.72, 0.30, 0.82],
                [-0.72, 0.30, 0.82],
            ],
            dtype=np.float32,
        )
        self.targets = {
            "train": self.train_targets,
            "deployment": self.deployment_targets,
            "all": np.vstack(
                (self.train_targets, self.deployment_targets)
            ),
        }[target_split]
        self.obstacle_specs = (
            (np.asarray([0.30, -0.28, 0.36]), np.asarray([0.12, 0.12, 0.36])),
            (np.asarray([-0.30, -0.28, 0.36]), np.asarray([0.12, 0.12, 0.36])),
        )
        self.initial_position = np.asarray(
            [0.0, 0.0, 0.50], dtype=np.float32
        )
        self._pybullet = pybullet
        self._obstacle_ids: list[int] = []
        self.target = self.targets[0].copy()
        self.steps = 0
        self.previous_distance = 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            self.simulator = VelocityAviary(
                num_drones=1,
                initial_xyzs=self.initial_position[None, :].copy(),
                initial_rpys=np.zeros((1, 3), dtype=np.float32),
                physics=Physics(physics),
                pyb_freq=int(pyb_freq),
                ctrl_freq=int(ctrl_freq),
                gui=False,
                record=False,
                obstacles=False,
                user_debug_gui=False,
            )

    def _add_obstacles(self) -> None:
        p = self._pybullet
        self._obstacle_ids = []
        for center, half_extents in self.obstacle_specs:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents.tolist(),
                physicsClientId=self.simulator.CLIENT,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents.tolist(),
                rgbaColor=[0.75, 0.18, 0.16, 1.0],
                physicsClientId=self.simulator.CLIENT,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=center.tolist(),
                physicsClientId=self.simulator.CLIENT,
            )
            self._obstacle_ids.append(int(body))

    def _state(self, raw_observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_observation, dtype=np.float32).reshape(1, -1)[0]
        position = raw[0:3]
        rpy = raw[7:10]
        velocity = raw[10:13]
        relative_target = self.target - position
        obstacle_centers = np.stack(
            [center for center, _ in self.obstacle_specs]
        )
        nearest = obstacle_centers[
            np.argmin(np.linalg.norm(obstacle_centers - position, axis=1))
        ]
        relative_obstacle = nearest - position
        state = np.concatenate(
            (
                position / np.asarray([1.0, 1.0, 1.2]),
                relative_target / np.asarray([1.5, 1.5, 1.2]),
                np.clip(velocity / 0.8, -1.0, 1.0),
                np.clip(rpy / np.pi, -1.0, 1.0),
                relative_obstacle / np.asarray([1.5, 1.5, 1.2]),
            )
        )
        state = np.clip(state, -1.0, 1.0)
        quantum = max(self.state_quantization, 1e-6)
        return (
            np.round(state / quantum) * quantum
        ).astype(np.float32)

    def _distance(self, raw_observation: np.ndarray) -> float:
        position = np.asarray(raw_observation).reshape(1, -1)[0, 0:3]
        return float(np.linalg.norm(self.target - position))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        target_index = int(self.np_random.integers(len(self.targets)))
        self.target = self.targets[target_index].copy()
        position_noise = self.np_random.uniform(
            -self.initial_position_jitter,
            self.initial_position_jitter,
            size=3,
        )
        position_noise[2] *= 0.5
        attitude_noise = self.np_random.uniform(
            -self.initial_attitude_jitter,
            self.initial_attitude_jitter,
            size=3,
        )
        attitude_noise[2] = 0.0
        self.simulator.INIT_XYZS[0] = self.initial_position + position_noise
        self.simulator.INIT_RPYS[0] = attitude_noise
        with contextlib.redirect_stdout(io.StringIO()):
            raw_observation, _ = self.simulator.reset(seed=seed)
        self._add_obstacles()
        self.steps = 0
        self.previous_distance = self._distance(raw_observation)
        return self._state(raw_observation), {
            "target_split": self.target_split,
            "target": self.target.tolist(),
            "physics_backend": "gym-pybullet-drones",
        }

    def step(self, action: int):
        directions = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        action = int(action)
        direction = directions[action]
        speed = 0.0 if action == 6 else self.speed_fraction
        command = np.asarray(
            [[direction[0], direction[1], direction[2], speed]],
            dtype=np.float32,
        )
        raw_observation = None
        for _ in range(self.action_repeat):
            if self.wind_force_std > 0:
                force = self.np_random.normal(
                    0.0, self.wind_force_std, size=3
                )
                self._pybullet.applyExternalForce(
                    int(self.simulator.DRONE_IDS[0]),
                    -1,
                    force.tolist(),
                    [0.0, 0.0, 0.0],
                    self._pybullet.LINK_FRAME,
                    physicsClientId=self.simulator.CLIENT,
                )
            raw_observation, _, _, _, _ = self.simulator.step(command)

        self.steps += 1
        raw = np.asarray(raw_observation).reshape(1, -1)[0]
        position = raw[0:3]
        rpy = raw[7:10]
        velocity = raw[10:13]
        distance = self._distance(raw_observation)
        progress = self.previous_distance - distance
        self.previous_distance = distance
        contacts = self._pybullet.getContactPoints(
            bodyA=int(self.simulator.DRONE_IDS[0]),
            physicsClientId=self.simulator.CLIENT,
        )
        out_of_bounds = (
            abs(position[0]) > 0.95
            or abs(position[1]) > 0.95
            or position[2] < 0.08
            or position[2] > 1.15
        )
        unstable = abs(rpy[0]) > 0.85 or abs(rpy[1]) > 0.85
        collision = bool(contacts) or bool(out_of_bounds) or bool(unstable)
        risk_zone = any(
            self._pybullet.getClosestPoints(
                int(self.simulator.DRONE_IDS[0]),
                obstacle_id,
                distance=0.18,
                physicsClientId=self.simulator.CLIENT,
            )
            for obstacle_id in self._obstacle_ids
        )
        success = (
            distance <= self.goal_tolerance
            and float(np.linalg.norm(velocity)) <= 0.35
        )
        idle = action == 6
        terminated = bool(success or collision)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        reward = 4.0 * progress - 0.01
        reward -= 0.03 * float(np.linalg.norm(rpy[0:2]))
        if risk_zone:
            reward -= 0.04
        if idle:
            reward -= 0.01
        if collision:
            reward -= 4.0
        if success:
            reward += 5.0
        info = {
            "collision": collision,
            "risk_zone": risk_zone,
            "idle": idle,
            "lambda_collision": self.lambda_collision,
            "lambda_idle": self.lambda_idle,
            "distance_to_goal": distance,
            "target_split": self.target_split,
            "physics_backend": "gym-pybullet-drones",
        }
        return (
            self._state(raw_observation),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.simulator.close()


class SensorizedPyBulletUAVWaypointEnv(gym.Env):
    """Crazyflie SIL task with delayed sensing and low-level flight commands."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_split: str = "train",
        physics: str = "pyb",
        pyb_freq: int = 240,
        ctrl_freq: int = 60,
        action_repeat: int = 2,
        max_steps: int = 100,
        collective_delta_rpm: float = 180.0,
        tilt_radians: float = 0.08,
        goal_tolerance: float = 0.16,
        state_quantization: float = 0.05,
        lidar_range: float = 1.2,
        lidar_noise_std: float = 0.01,
        vio_position_noise_std: float = 0.015,
        vio_velocity_noise_std: float = 0.025,
        imu_attitude_noise_std: float = 0.008,
        localization_latency_steps: int = 2,
        localization_dropout_probability: float = 0.02,
        range_dropout_probability: float = 0.01,
        sensor_bias_walk_std: float = 0.0005,
        camera_fov_degrees: float = 100.0,
        camera_dropout_probability: float = 0.02,
        initial_position_jitter: float = 0.02,
        initial_attitude_jitter: float = 0.01,
        wind_force_std: float = 0.0,
        lambda_collision: float = 1.0,
        lambda_idle: float = 0.05,
        sensor_noise_enabled: bool = True,
        sensor_latency_enabled: bool = True,
        localization_dropout_enabled: bool = True,
        range_dropout_enabled: bool = True,
        camera_dropout_enabled: bool = True,
        visibility_occlusion_enabled: bool = True,
        observation_mode: str = "sensorized",
        control_interface_mode: str = "low_level",
        high_level_speed: float = 0.18,
        perturbation_onset_step: int = 0,
        motor_saturation_threshold_fraction: float = 0.99,
        near_miss_clearance: float = 0.10,
        constraint_clearance: float = 0.18,
        trace_schema_version: str = "sensorized_sil_trace_v2",
    ):
        if not has_uav_backend():
            raise ImportError(
                "Sensorized PyBullet UAV validation requires the optional "
                "'uav' dependencies. Install with: "
                "python -m pip install -e .[uav]"
            )
        if target_split not in {"train", "deployment", "all"}:
            raise ValueError(
                "target_split must be train, deployment, or all"
            )
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        if localization_latency_steps < 0:
            raise ValueError("localization_latency_steps must be non-negative")
        if perturbation_onset_step < 0:
            raise ValueError("perturbation_onset_step must be non-negative")
        if not 0.0 < motor_saturation_threshold_fraction <= 1.0:
            raise ValueError(
                "motor_saturation_threshold_fraction must be in (0, 1]"
            )
        if not 0.0 <= near_miss_clearance <= constraint_clearance:
            raise ValueError(
                "near_miss_clearance must be between zero and constraint_clearance"
            )
        if trace_schema_version not in {
            "sensorized_sil_trace_v1",
            "sensorized_sil_trace_v2",
            "sensorized_sil_trace_v3",
        }:
            raise ValueError("unsupported trace_schema_version")
        if observation_mode not in {"sensorized", "state_accessible"}:
            raise ValueError(
                "observation_mode must be sensorized or state_accessible"
            )
        if control_interface_mode not in {"low_level", "high_level"}:
            raise ValueError(
                "control_interface_mode must be low_level or high_level"
            )

        import pybullet as pybullet
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics

        self.target_split = target_split
        self.observation_mode = observation_mode
        self.control_interface_mode = control_interface_mode
        self.high_level_speed = float(high_level_speed)
        self.perturbation_onset_step = int(perturbation_onset_step)
        self.motor_saturation_threshold_fraction = float(
            motor_saturation_threshold_fraction
        )
        self.near_miss_clearance = float(near_miss_clearance)
        self.constraint_clearance = float(constraint_clearance)
        self.trace_schema_version = str(trace_schema_version)
        self.action_repeat = int(action_repeat)
        self.max_steps = int(max_steps)
        self.collective_delta_rpm = float(collective_delta_rpm)
        self.tilt_radians = float(tilt_radians)
        self.goal_tolerance = float(goal_tolerance)
        self.state_quantization = float(state_quantization)
        self.lidar_range = float(lidar_range)
        self.sensor_noise_enabled = bool(sensor_noise_enabled)
        self.sensor_latency_enabled = bool(sensor_latency_enabled)
        self.localization_dropout_enabled = bool(
            localization_dropout_enabled
        )
        self.range_dropout_enabled = bool(range_dropout_enabled)
        self.camera_dropout_enabled = bool(camera_dropout_enabled)
        self.visibility_occlusion_enabled = bool(
            visibility_occlusion_enabled
        )
        noise_multiplier = float(self.sensor_noise_enabled)
        self.lidar_noise_std = float(lidar_noise_std) * noise_multiplier
        self.vio_position_noise_std = (
            float(vio_position_noise_std) * noise_multiplier
        )
        self.vio_velocity_noise_std = (
            float(vio_velocity_noise_std) * noise_multiplier
        )
        self.imu_attitude_noise_std = (
            float(imu_attitude_noise_std) * noise_multiplier
        )
        self.localization_latency_steps = (
            int(localization_latency_steps)
            if self.sensor_latency_enabled
            else 0
        )
        self.localization_dropout_probability = (
            float(localization_dropout_probability)
            if self.localization_dropout_enabled
            else 0.0
        )
        self.range_dropout_probability = (
            float(range_dropout_probability)
            if self.range_dropout_enabled
            else 0.0
        )
        self.sensor_bias_walk_std = (
            float(sensor_bias_walk_std) * noise_multiplier
        )
        self.camera_fov_cosine = float(
            np.cos(np.deg2rad(camera_fov_degrees / 2.0))
        )
        self.camera_dropout_probability = (
            float(camera_dropout_probability)
            if self.camera_dropout_enabled
            else 0.0
        )
        self.initial_position_jitter = float(initial_position_jitter)
        self.initial_attitude_jitter = float(initial_attitude_jitter)
        self.wind_force_std = float(wind_force_std)
        self.lambda_collision = float(lambda_collision)
        self.lambda_idle = float(lambda_idle)
        self.ctrl_freq = int(ctrl_freq)
        self.action_space = spaces.Discrete(
            27 if self.control_interface_mode == "low_level" else 7
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(22 if self.observation_mode == "sensorized" else 15,),
            dtype=np.float32,
        )
        self.train_targets = np.asarray(
            [
                [0.60, 0.00, 0.55],
                [-0.60, 0.00, 0.55],
                [0.00, 0.60, 0.65],
                [0.00, -0.60, 0.65],
                [0.45, 0.45, 0.75],
                [-0.45, 0.45, 0.75],
            ],
            dtype=np.float32,
        )
        self.deployment_targets = np.asarray(
            [
                [0.58, -0.58, 0.70],
                [-0.58, -0.58, 0.70],
                [0.72, 0.30, 0.82],
                [-0.72, 0.30, 0.82],
            ],
            dtype=np.float32,
        )
        self.targets = {
            "train": self.train_targets,
            "deployment": self.deployment_targets,
            "all": np.vstack(
                (self.train_targets, self.deployment_targets)
            ),
        }[target_split]
        self.obstacle_specs = (
            (
                np.asarray([0.30, -0.28, 0.36]),
                np.asarray([0.12, 0.12, 0.36]),
            ),
            (
                np.asarray([-0.30, -0.28, 0.36]),
                np.asarray([0.12, 0.12, 0.36]),
            ),
        )
        self.initial_position = np.asarray(
            [0.0, 0.0, 0.50], dtype=np.float32
        )
        self._pybullet = pybullet
        self._obstacle_ids: list[int] = []
        self.target = self.targets[0].copy()
        self.steps = 0
        self.previous_distance = 0.0
        self.localization_age = 0
        self.localization_valid = True
        self.camera_visible = False
        self.sensor_dropout = False
        self.localization_dropout = False
        self.range_dropout = False
        self.camera_dropout = False
        self.initial_distance = 0.0
        self.minimum_distance = 0.0
        self.previous_constraint_active = False
        self.control_ticks = 0
        self.position_bias = np.zeros(3, dtype=np.float32)
        self.estimate = {
            "position": self.initial_position.copy(),
            "velocity": np.zeros(3, dtype=np.float32),
            "rpy": np.zeros(3, dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
        }
        self._localization_buffer: deque[dict[str, np.ndarray] | None] = deque(
            maxlen=self.localization_latency_steps + 1
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.simulator = CtrlAviary(
                num_drones=1,
                initial_xyzs=self.initial_position[None, :].copy(),
                initial_rpys=np.zeros((1, 3), dtype=np.float32),
                physics=Physics(physics),
                pyb_freq=int(pyb_freq),
                ctrl_freq=int(ctrl_freq),
                gui=False,
                record=False,
                obstacles=False,
                user_debug_gui=False,
            )
        self.stabilizer = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.hover_pwm = (
            self.simulator.HOVER_RPM - self.stabilizer.PWM2RPM_CONST
        ) / self.stabilizer.PWM2RPM_SCALE
        self.altitude_setpoint = float(self.initial_position[2])
        self._last_raw_observation: np.ndarray | None = None
        self.reference_start_position = self.initial_position.copy()

    def _perturbation_active(self, step: int | None = None) -> bool:
        effective_step = self.steps if step is None else int(step)
        return effective_step >= self.perturbation_onset_step

    def _nominal_reference_position(self, step: int | None = None) -> np.ndarray:
        effective_step = self.steps if step is None else int(step)
        fraction = min(max(effective_step, 0) / max(self.max_steps, 1), 1.0)
        return (
            self.reference_start_position
            + fraction * (self.target - self.reference_start_position)
        ).astype(np.float32)

    def _add_obstacles(self) -> None:
        p = self._pybullet
        self._obstacle_ids = []
        for center, half_extents in self.obstacle_specs:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents.tolist(),
                physicsClientId=self.simulator.CLIENT,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents.tolist(),
                rgbaColor=[0.75, 0.18, 0.16, 1.0],
                physicsClientId=self.simulator.CLIENT,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=center.tolist(),
                physicsClientId=self.simulator.CLIENT,
            )
            self._obstacle_ids.append(int(body))

    @staticmethod
    def _raw(raw_observation: np.ndarray) -> np.ndarray:
        return np.asarray(raw_observation, dtype=np.float32).reshape(1, -1)[0]

    def _new_localization_measurement(
        self, raw_observation: np.ndarray, *, perturbation_active: bool | None = None
    ) -> dict[str, np.ndarray] | None:
        active = (
            self._perturbation_active()
            if perturbation_active is None
            else bool(perturbation_active)
        )
        dropout_probability = self.localization_dropout_probability if active else 0.0
        if dropout_probability > 0.0 and self.np_random.random() < dropout_probability:
            self.localization_dropout = True
            return None
        self.localization_dropout = False
        raw = self._raw(raw_observation)
        if active and self.sensor_bias_walk_std > 0.0:
            self.position_bias += self.np_random.normal(
                0.0, self.sensor_bias_walk_std, size=3
            ).astype(np.float32)
        position_noise = (
            self.np_random.normal(0.0, self.vio_position_noise_std, size=3)
            if active and self.vio_position_noise_std > 0.0
            else np.zeros(3)
        )
        velocity_noise = (
            self.np_random.normal(0.0, self.vio_velocity_noise_std, size=3)
            if active and self.vio_velocity_noise_std > 0.0
            else np.zeros(3)
        )
        attitude_noise = (
            self.np_random.normal(0.0, self.imu_attitude_noise_std, size=3)
            if active and self.imu_attitude_noise_std > 0.0
            else np.zeros(3)
        )
        return {
            "position": (
                raw[0:3]
                + self.position_bias
                + position_noise
            ).astype(np.float32),
            "velocity": (
                raw[10:13] + velocity_noise
            ).astype(np.float32),
            "rpy": (raw[7:10] + attitude_noise).astype(np.float32),
            "angular_velocity": (raw[13:16] + attitude_noise).astype(np.float32),
        }

    def _update_localization(self, raw_observation: np.ndarray) -> None:
        active = self._perturbation_active(self.steps + 1)
        measurement = self._new_localization_measurement(
            raw_observation, perturbation_active=active
        )
        self._localization_buffer.append(measurement)
        delayed = (
            self._localization_buffer[0]
            if active and self.sensor_latency_enabled
            else self._localization_buffer[-1]
        )
        self.localization_valid = delayed is not None
        if delayed is None:
            self.localization_age += 1
        else:
            self.estimate = {
                key: value.copy() for key, value in delayed.items()
            }
            self.localization_age = 0
        raw = self._raw(raw_observation)
        imu_noise = (
            self.np_random.normal(0.0, self.imu_attitude_noise_std, size=3)
            if active and self.imu_attitude_noise_std > 0.0
            else np.zeros(3)
        )
        self.estimate["rpy"] = (raw[7:10] + imu_noise).astype(np.float32)
        self.estimate["angular_velocity"] = (
            raw[13:16] + imu_noise
        ).astype(np.float32)

    def _lidar_ranges(
        self, raw_observation: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        raw = self._raw(raw_observation)
        position = raw[0:3]
        rotation = np.asarray(
            self._pybullet.getMatrixFromQuaternion(raw[3:7])
        ).reshape(3, 3)
        body_directions = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        body_directions[6:10] /= np.sqrt(2.0)
        world_directions = body_directions @ rotation.T
        starts = position[None, :] + 0.08 * world_directions
        ends = position[None, :] + self.lidar_range * world_directions
        hits = self._pybullet.rayTestBatch(
            starts.tolist(),
            ends.tolist(),
            physicsClientId=self.simulator.CLIENT,
        )
        ranges = np.asarray(
            [
                self.lidar_range
                if int(hit[0]) == int(self.simulator.DRONE_IDS[0])
                else max(0.0, float(hit[2]) * self.lidar_range - 0.08)
                for hit in hits
            ],
            dtype=np.float32,
        )
        active = self._perturbation_active()
        if active and self.lidar_noise_std > 0.0:
            ranges += self.np_random.normal(
                0.0, self.lidar_noise_std, size=ranges.shape
            ).astype(np.float32)
        dropped = (
            self.np_random.random(ranges.shape) < self.range_dropout_probability
            if active and self.range_dropout_probability > 0.0
            else np.zeros(ranges.shape, dtype=bool)
        )
        ranges[dropped] = self.lidar_range
        self.range_dropout = bool(dropped.any())
        return np.clip(ranges, 0.0, self.lidar_range), bool(dropped.any())

    def _camera_target_visible(self, raw_observation: np.ndarray) -> bool:
        active = self._perturbation_active()
        if (
            active
            and self.camera_dropout_probability > 0.0
            and self.np_random.random() < self.camera_dropout_probability
        ):
            self.camera_dropout = True
            return False
        self.camera_dropout = False
        if not active or not self.visibility_occlusion_enabled:
            return True
        raw = self._raw(raw_observation)
        position = raw[0:3]
        target_vector = self.target - position
        distance = float(np.linalg.norm(target_vector))
        if distance <= 1e-8:
            return True
        rotation = np.asarray(
            self._pybullet.getMatrixFromQuaternion(raw[3:7])
        ).reshape(3, 3)
        forward = rotation[:, 0]
        in_view = (
            float(np.dot(forward, target_vector / distance))
            >= self.camera_fov_cosine
        )
        if not in_view:
            return False
        hit = self._pybullet.rayTest(
            position.tolist(),
            self.target.tolist(),
            physicsClientId=self.simulator.CLIENT,
        )[0]
        return int(hit[0]) in {-1, int(self.simulator.DRONE_IDS[0])}

    def _sensorized_state(self, raw_observation: np.ndarray) -> np.ndarray:
        lidar_ranges, range_dropout = self._lidar_ranges(raw_observation)
        self.camera_visible = self._camera_target_visible(raw_observation)
        self.sensor_dropout = bool(
            range_dropout or not self.localization_valid
        )
        relative_target = self.target - self.estimate["position"]
        state = np.concatenate(
            (
                relative_target / np.asarray([1.5, 1.5, 1.2]),
                np.clip(self.estimate["velocity"] / 0.8, -1.0, 1.0),
                np.clip(self.estimate["rpy"] / np.pi, -1.0, 1.0),
                2.0 * lidar_ranges / self.lidar_range - 1.0,
                np.asarray(
                    [
                        min(self.localization_age, 10) / 5.0 - 1.0,
                        1.0 if self.localization_valid else -1.0,
                        1.0 if self.camera_visible else -1.0,
                    ]
                ),
            )
        )
        quantum = max(self.state_quantization, 1e-6)
        return (
            np.round(np.clip(state, -1.0, 1.0) / quantum) * quantum
        ).astype(np.float32)

    def _state_accessible_observation(
        self, raw_observation: np.ndarray
    ) -> np.ndarray:
        raw = self._raw(raw_observation)
        position = raw[0:3]
        relative_target = self.target - position
        obstacle_centers = np.stack(
            [center for center, _ in self.obstacle_specs]
        )
        nearest = obstacle_centers[
            np.argmin(np.linalg.norm(obstacle_centers - position, axis=1))
        ]
        state = np.concatenate(
            (
                position / np.asarray([1.0, 1.0, 1.2]),
                relative_target / np.asarray([1.5, 1.5, 1.2]),
                np.clip(raw[10:13] / 0.8, -1.0, 1.0),
                np.clip(raw[7:10] / np.pi, -1.0, 1.0),
                (nearest - position) / np.asarray([1.5, 1.5, 1.2]),
            )
        )
        quantum = max(self.state_quantization, 1e-6)
        return (
            np.round(np.clip(state, -1.0, 1.0) / quantum) * quantum
        ).astype(np.float32)

    def _learning_observation(
        self,
        raw_observation: np.ndarray,
        sensorized_state: np.ndarray,
    ) -> np.ndarray:
        if self.observation_mode == "sensorized":
            return sensorized_state
        return self._state_accessible_observation(raw_observation)

    def _nominal_reference_action(self, raw_observation: np.ndarray) -> int:
        """Return a declared free-space control label, not an optimal action."""

        raw = self._raw(raw_observation)
        relative_target = self.target - raw[0:3]
        desired_velocity = np.clip(0.8 * relative_target, -0.18, 0.18)
        velocity_error = desired_velocity - raw[10:13]
        if self.control_interface_mode == "high_level":
            axis = int(np.argmax(np.abs(relative_target)))
            sign = float(relative_target[axis])
            if abs(sign) <= 0.02:
                return 6
            return 2 * axis + (0 if sign > 0.0 else 1)
        command = np.zeros(3, dtype=int)
        active = np.abs(velocity_error) >= 0.06
        command[active] = np.sign(velocity_error[active]).astype(int)
        return int(
            (command[0] + 1) * 9
            + (command[1] + 1) * 3
            + command[2]
            + 1
        )

    def _trace_info(
        self,
        raw_observation: np.ndarray,
        observation: np.ndarray,
        *,
        selected_action: int | None,
    ) -> dict[str, Any]:
        raw = self._raw(raw_observation)
        command_timestamp = self.control_ticks / max(self.ctrl_freq, 1)
        perturbation_active = self._perturbation_active()
        latency_ticks = (
            self.localization_latency_steps if perturbation_active else 0
        ) + self.localization_age
        observation_timestamp = max(
            0.0,
            (self.control_ticks - latency_ticks) / max(self.ctrl_freq, 1),
        )
        perturbation_flags = {
            "localization_dropout": bool(self.localization_dropout),
            "range_dropout": bool(self.range_dropout),
            "camera_dropout": bool(self.camera_dropout),
        }
        return {
            "trace_schema_version": self.trace_schema_version,
            "latent_position": tuple(float(value) for value in raw[0:3]),
            "latent_velocity": tuple(float(value) for value in raw[10:13]),
            "target_state": tuple(float(value) for value in self.target),
            "raw_sensor_observation": tuple(
                float(value)
                for value in np.asarray(observation, dtype=np.float32)
            ),
            "selected_action": selected_action,
            "reference_control_action": self._nominal_reference_action(
                raw_observation
            ),
            "reference_control_label": "nominal_free_space_latent_controller",
            "perturbation_flags": perturbation_flags,
            "observation_perturbed": bool(any(perturbation_flags.values())),
            "observation_timestamp": float(observation_timestamp),
            "command_timestamp": float(command_timestamp),
            "effective_latency": float(
                command_timestamp - observation_timestamp
            ),
            "target_visibility": bool(self.camera_visible),
            "perturbation_onset_step": self.perturbation_onset_step,
            "perturbation_active": perturbation_active,
            "learning_observation_mode": self.observation_mode,
            "control_interface_mode": self.control_interface_mode,
            "sensor_factor_noise": self.sensor_noise_enabled,
            "sensor_factor_latency": self.sensor_latency_enabled,
            "sensor_factor_localization_dropout": (
                self.localization_dropout_enabled
            ),
            "sensor_factor_range_dropout": self.range_dropout_enabled,
            "sensor_factor_camera_dropout": self.camera_dropout_enabled,
            "sensor_factor_visibility_occlusion": (
                self.visibility_occlusion_enabled
            ),
        }

    def _distance(self, raw_observation: np.ndarray) -> float:
        position = self._raw(raw_observation)[0:3]
        return float(np.linalg.norm(self.target - position))

    def _motor_command(self, action: int) -> np.ndarray:
        action = int(action)
        x_command = action // 9 - 1
        y_command = (action % 9) // 3 - 1
        z_command = action % 3 - 1
        estimated_position = self.estimate["position"]
        estimated_velocity = self.estimate["velocity"]
        estimated_rpy = self.estimate["rpy"]
        if z_command > 0:
            self.altitude_setpoint = min(1.0, self.altitude_setpoint + 0.025)
        elif z_command < 0:
            self.altitude_setpoint = max(0.18, self.altitude_setpoint - 0.025)
        altitude_error = self.altitude_setpoint - float(estimated_position[2])
        pwm = (
            self.hover_pwm
            + 5000.0 * altitude_error
            - 900.0 * float(estimated_velocity[2])
        )
        estimated_quaternion = np.asarray(
            self._pybullet.getQuaternionFromEuler(estimated_rpy.tolist())
        )
        target_rpy = np.asarray(
            [
                -y_command * self.tilt_radians,
                x_command * self.tilt_radians,
                0.0,
            ],
            dtype=np.float32,
        )
        stabilizing_rpm = self.stabilizer._dslPIDAttitudeControl(
            1.0 / self.simulator.CTRL_FREQ,
            pwm,
            estimated_quaternion,
            target_rpy,
            np.zeros(3),
        )
        collective_delta = (
            self.collective_delta_rpm
            * z_command
            * np.ones(4, dtype=np.float32)
        )
        rpm = stabilizing_rpm + collective_delta
        return np.clip(rpm, 0.0, self.simulator.MAX_RPM).astype(np.float32)

    def _high_level_motor_command(
        self, action: int, raw_observation: np.ndarray
    ) -> np.ndarray:
        directions = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        raw = self._raw(raw_observation)
        target_velocity = directions[int(action)] * self.high_level_speed
        target_position = raw[0:3] + target_velocity / max(self.ctrl_freq, 1)
        rpm, _, _ = self.stabilizer.computeControl(
            control_timestep=1.0 / self.simulator.CTRL_FREQ,
            cur_pos=raw[0:3],
            cur_quat=raw[3:7],
            cur_vel=raw[10:13],
            cur_ang_vel=raw[13:16],
            target_pos=target_position,
            target_vel=target_velocity,
        )
        return np.clip(rpm, 0.0, self.simulator.MAX_RPM).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        target_index = int(self.np_random.integers(len(self.targets)))
        self.target = self.targets[target_index].copy()
        position_noise = self.np_random.uniform(
            -self.initial_position_jitter,
            self.initial_position_jitter,
            size=3,
        )
        position_noise[2] *= 0.5
        attitude_noise = self.np_random.uniform(
            -self.initial_attitude_jitter,
            self.initial_attitude_jitter,
            size=3,
        )
        attitude_noise[2] = 0.0
        self.simulator.INIT_XYZS[0] = self.initial_position + position_noise
        self.simulator.INIT_RPYS[0] = attitude_noise
        with contextlib.redirect_stdout(io.StringIO()):
            raw_observation, _ = self.simulator.reset(seed=seed)
        self._add_obstacles()
        self.stabilizer.reset()
        self.steps = 0
        self.control_ticks = 0
        self.position_bias = np.zeros(3, dtype=np.float32)
        self.localization_age = 0
        self.localization_valid = True
        self._localization_buffer.clear()
        first_measurement = self._new_localization_measurement(raw_observation)
        if first_measurement is None:
            raw = self._raw(raw_observation)
            first_measurement = {
                "position": raw[0:3].copy(),
                "velocity": raw[10:13].copy(),
                "rpy": raw[7:10].copy(),
                "angular_velocity": raw[13:16].copy(),
            }
        self.estimate = {
            key: value.copy() for key, value in first_measurement.items()
        }
        for _ in range(self.localization_latency_steps + 1):
            self._localization_buffer.append(
                {
                    key: value.copy()
                    for key, value in first_measurement.items()
                }
            )
        self.altitude_setpoint = float(self.estimate["position"][2])
        self.previous_distance = self._distance(raw_observation)
        self.initial_distance = self.previous_distance
        self.minimum_distance = self.previous_distance
        self.previous_constraint_active = False
        self._last_raw_observation = np.asarray(raw_observation).copy()
        self.reference_start_position = self._raw(raw_observation)[0:3].copy()
        sensorized_state = self._sensorized_state(raw_observation)
        observation = self._learning_observation(
            raw_observation, sensorized_state
        )
        observation_source = (
            "latent_state_accessible"
            if self.observation_mode == "state_accessible"
            else "delayed_vio_imu_lidar_pinhole_target_detector"
        )
        control_interface = (
            "velocity_setpoint_to_internal_controller"
            if self.control_interface_mode == "high_level"
            else "attitude_collective_to_motor_rpm"
        )
        info = {
            "target_split": self.target_split,
            "physics_backend": "gym-pybullet-drones",
            "observation_source": observation_source,
            "control_interface": control_interface,
        }
        info.update(
            self._trace_info(
                raw_observation, sensorized_state, selected_action=None
            )
        )
        return observation, info

    def step(self, action: int):
        action = int(action)
        raw_observation = None
        rpm = None
        for _ in range(self.action_repeat):
            if self.wind_force_std > 0:
                force = self.np_random.normal(
                    0.0, self.wind_force_std, size=3
                )
                self._pybullet.applyExternalForce(
                    int(self.simulator.DRONE_IDS[0]),
                    -1,
                    force.tolist(),
                    [0.0, 0.0, 0.0],
                    self._pybullet.LINK_FRAME,
                    physicsClientId=self.simulator.CLIENT,
                )
            if self.control_interface_mode == "high_level":
                if self._last_raw_observation is None:
                    raise RuntimeError("high-level controller has no plant state")
                rpm = self._high_level_motor_command(
                    action, self._last_raw_observation
                )
            else:
                rpm = self._motor_command(action)
            raw_observation, _, _, _, _ = self.simulator.step(rpm[None, :])
            self._last_raw_observation = np.asarray(raw_observation).copy()
            self.control_ticks += 1
            self._update_localization(raw_observation)

        self.steps += 1
        raw = self._raw(raw_observation)
        position = raw[0:3]
        rpy = raw[7:10]
        velocity = raw[10:13]
        distance = self._distance(raw_observation)
        previous_distance = self.previous_distance
        progress = previous_distance - distance
        self.previous_distance = distance
        self.minimum_distance = min(self.minimum_distance, distance)
        contacts = self._pybullet.getContactPoints(
            bodyA=int(self.simulator.DRONE_IDS[0]),
            physicsClientId=self.simulator.CLIENT,
        )
        out_of_bounds = (
            abs(position[0]) > 0.95
            or abs(position[1]) > 0.95
            or position[2] < 0.08
            or position[2] > 1.15
        )
        unstable = abs(rpy[0]) > 0.85 or abs(rpy[1]) > 0.85
        collision = bool(contacts) or bool(out_of_bounds) or bool(unstable)
        obstacle_points = [
            point
            for obstacle_id in self._obstacle_ids
            for point in self._pybullet.getClosestPoints(
                int(self.simulator.DRONE_IDS[0]),
                obstacle_id,
                distance=10.0,
                physicsClientId=self.simulator.CLIENT,
            )
        ]
        minimum_obstacle_clearance = min(
            (float(point[8]) for point in obstacle_points), default=float("nan")
        )
        risk_zone = bool(
            np.isfinite(minimum_obstacle_clearance)
            and minimum_obstacle_clearance <= self.constraint_clearance
        )
        success = (
            distance <= self.goal_tolerance
            and float(np.linalg.norm(velocity)) <= 0.35
        )
        idle = action == (
            13 if self.control_interface_mode == "low_level" else 6
        )
        terminated = bool(success or collision)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        reward = 4.0 * progress - 0.01
        reward -= 0.03 * float(np.linalg.norm(rpy[0:2]))
        if risk_zone:
            reward -= 0.04
        if idle:
            reward -= 0.01
        if collision:
            reward -= 4.0
        if success:
            reward += 5.0
        localization_error = float(
            np.linalg.norm(position - self.estimate["position"])
        )
        motor_saturation = float(
            np.mean(
                (rpm <= 1.0)
                | (
                    rpm
                    >= self.motor_saturation_threshold_fraction
                    * float(self.simulator.MAX_RPM)
                )
            )
        )
        near_miss = bool(
            not collision
            and np.isfinite(minimum_obstacle_clearance)
            and minimum_obstacle_clearance <= self.near_miss_clearance
        )
        reference_position = self._nominal_reference_position()
        trajectory_deviation = float(np.linalg.norm(position - reference_position))
        constraint_active = bool(collision or risk_zone)
        recovery_event = bool(
            self.previous_constraint_active
            and not constraint_active
            and distance < previous_distance
        )
        self.previous_constraint_active = constraint_active
        if success:
            failure_stage = "success"
        elif collision:
            failure_stage = (
                "collision_out_of_bounds"
                if out_of_bounds
                else (
                    "collision_unstable" if unstable else "collision_contact"
                )
            )
        elif truncated:
            if distance >= self.initial_distance - 0.05:
                failure_stage = "timeout_no_progress"
            elif self.minimum_distance <= self.goal_tolerance:
                failure_stage = "timeout_near_target"
            else:
                failure_stage = "timeout_partial_progress"
        else:
            failure_stage = "ongoing"
        sensorized_state = self._sensorized_state(raw_observation)
        observation = self._learning_observation(
            raw_observation, sensorized_state
        )
        observation_source = (
            "latent_state_accessible"
            if self.observation_mode == "state_accessible"
            else "delayed_vio_imu_lidar_pinhole_target_detector"
        )
        control_interface = (
            "velocity_setpoint_to_internal_controller"
            if self.control_interface_mode == "high_level"
            else "attitude_collective_to_motor_rpm"
        )
        info = {
            "collision": collision,
            "risk_zone": risk_zone,
            "idle": idle,
            "lambda_collision": self.lambda_collision,
            "lambda_idle": self.lambda_idle,
            "distance_to_goal": distance,
            "target_split": self.target_split,
            "physics_backend": "gym-pybullet-drones",
            "observation_source": observation_source,
            "control_interface": control_interface,
            "localization_error": localization_error,
            "sensor_dropout": self.sensor_dropout,
            "camera_visible": self.camera_visible,
            "motor_saturation": motor_saturation,
            "trajectory_error": distance,
            "trajectory_deviation": trajectory_deviation,
            "latent_position": tuple(float(value) for value in position),
            "nominal_reference_position": tuple(
                float(value) for value in reference_position
            ),
            "nominal_reference_path_id": "linear_reset_position_to_episode_target_v1",
            "minimum_obstacle_clearance": minimum_obstacle_clearance,
            "near_miss": near_miss,
            "perturbation_onset_step": self.perturbation_onset_step,
            "perturbation_active": self._perturbation_active(),
            "control_timestep_seconds": 1.0 / max(self.ctrl_freq, 1),
            "post_action_timestamp": self.steps / max(self.ctrl_freq, 1),
            "constraint_active": constraint_active,
            "saturation_active": bool(motor_saturation > 0.0),
            "recovery_event": recovery_event,
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
            "failure_stage": failure_stage,
        }
        info.update(
            self._trace_info(
                raw_observation, sensorized_state, selected_action=action
            )
        )
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.simulator.close()


class FallbackFrozenLakeEnv(gym.Env):
    """Minimal deterministic/slippery grid task used only when gymnasium is absent."""

    metadata = {"render_modes": []}

    def __init__(self, map_name: str = "4x4", is_slippery: bool = True, **_: Any):
        self.size = 8 if "8" in str(map_name) else 4
        self.is_slippery = bool(is_slippery)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.position = (0, 0)
        self.steps = 0
        self.max_steps = self.size * self.size * 4
        self.np_random = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.position = (0, 0)
        self.steps = 0
        return self._state(), {}

    def _state(self) -> int:
        return self.position[1] * self.size + self.position[0]

    def step(self, action: int):
        if self.is_slippery and self.np_random.random() < 0.2:
            action = int(self.np_random.integers(self.action_space.n))
        moves = ((0, -1), (1, 0), (0, 1), (-1, 0))
        dx, dy = moves[int(action)]
        x = int(np.clip(self.position[0] + dx, 0, self.size - 1))
        y = int(np.clip(self.position[1] + dy, 0, self.size - 1))
        self.position = (x, y)
        self.steps += 1
        terminated = self.position == (self.size - 1, self.size - 1)
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else 0.0
        return self._state(), reward, terminated, truncated, {}


class FallbackCliffWalkingEnv(gym.Env):
    """Small cliff-walking stand-in for dependency-free tests."""

    metadata = {"render_modes": []}

    def __init__(self, **_: Any):
        self.width = 12
        self.height = 4
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.width * self.height)
        self.start = (0, self.height - 1)
        self.goal = (self.width - 1, self.height - 1)
        self.position = self.start
        self.np_random = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.position = self.start
        return self._state(), {}

    def _state(self) -> int:
        return self.position[1] * self.width + self.position[0]

    def step(self, action: int):
        moves = ((0, -1), (1, 0), (0, 1), (-1, 0))
        dx, dy = moves[int(action)]
        x = int(np.clip(self.position[0] + dx, 0, self.width - 1))
        y = int(np.clip(self.position[1] + dy, 0, self.height - 1))
        reward = -1.0
        terminated = False
        if y == self.height - 1 and 1 <= x <= self.width - 2:
            self.position = self.start
            reward = -100.0
        else:
            self.position = (x, y)
            terminated = self.position == self.goal
        return self._state(), reward, terminated, False, {}


class FallbackTaxiEnv(gym.Env):
    """Compact Taxi-like discrete stand-in when gymnasium is unavailable."""

    metadata = {"render_modes": []}

    def __init__(self, **_: Any):
        self.size = 5
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.position = (0, 0)
        self.goal = (4, 4)
        self.np_random = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.position = (0, 0)
        return self._state(), {}

    def _state(self) -> int:
        return self.position[1] * self.size + self.position[0]

    def step(self, action: int):
        moves = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0), (0, 0))
        dx, dy = moves[int(action)]
        x = int(np.clip(self.position[0] + dx, 0, self.size - 1))
        y = int(np.clip(self.position[1] + dy, 0, self.size - 1))
        self.position = (x, y)
        terminated = self.position == self.goal and int(action) in {4, 5}
        reward = 20.0 if terminated else -1.0
        return self._state(), reward, terminated, False, {}


class FallbackMiniGridImageEnv(gym.Env):
    """Fully observable image-style stand-in for MiniGrid dependency tests."""

    metadata = {"render_modes": []}

    def __init__(self, env_id: str, **_: Any):
        self.env_id = env_id
        self.size = 8 if "8x8" in env_id else 6 if "6x6" in env_id else 5
        if "FourRooms" in env_id:
            self.size = 7
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(self.size, self.size, 3), dtype=np.float32
        )
        self.position = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.steps = 0
        self.max_steps = self.size * self.size * 4
        self.np_random = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.position = (0, 0)
        self.steps = 0
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[self.goal[1], self.goal[0], 1] = 2.0
        obs[self.position[1], self.position[0], 0] = 5.0
        return obs

    def step(self, action: int):
        moves = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0), (0, 0), (0, 0))
        dx, dy = moves[int(action) % len(moves)]
        x = int(np.clip(self.position[0] + dx, 0, self.size - 1))
        y = int(np.clip(self.position[1] + dy, 0, self.size - 1))
        self.position = (x, y)
        self.steps += 1
        terminated = self.position == self.goal
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.01
        return self._observation(), reward, terminated, truncated, {}


def resolve_env_id(env_id: str) -> str:
    if env_id in gym.registry:
        return env_id
    compatible = ENV_ID_COMPATIBILITY.get(env_id)
    if compatible and compatible in gym.registry:
        return compatible
    return env_id


def _apply_temporal_observation_wrappers(
    env: gym.Env, spec: dict[str, Any]
) -> gym.Env:
    stack_size = int(spec.get("frame_stack", 1))
    filter_alpha = spec.get("temporal_filter_alpha")
    if stack_size > 1 and filter_alpha is not None:
        raise ValueError(
            "frame_stack and temporal_filter_alpha cannot be combined"
        )
    if stack_size > 1:
        env = FrameStackObservationWrapper(env, stack_size)
    if filter_alpha is not None:
        env = FilteredBeliefObservationWrapper(env, float(filter_alpha))
    return env


def make_env(spec: dict[str, Any], evaluation: bool = False) -> gym.Env:
    env_id = spec["id"]
    kwargs = dict(spec.get("kwargs", {}))
    if evaluation:
        kwargs.update(spec.get("eval_kwargs", {}))

    if env_id == "StructuredFourRooms-v0":
        return _apply_temporal_observation_wrappers(
            StructuredFourRoomsEnv(**kwargs), spec
        )
    if env_id == "TransitionDynamicsShift-v0":
        return _apply_temporal_observation_wrappers(
            TransitionDynamicsShiftFourRoomsEnv(**kwargs), spec
        )
    if env_id == "ObservationShift-v0":
        return _apply_temporal_observation_wrappers(
            ObservationShiftFourRoomsEnv(**kwargs), spec
        )
    if env_id == "ApplicationNavigationSupportShift-v0":
        return _apply_temporal_observation_wrappers(
            ApplicationNavigationSupportShiftEnv(**kwargs), spec
        )
    if env_id == "LocalizedRewardShift-v0":
        return _apply_temporal_observation_wrappers(
            LocalizedRewardShiftNavigationEnv(**kwargs), spec
        )
    if env_id == "ReliabilityShiftBandit-v0":
        return _apply_temporal_observation_wrappers(
            ReliabilityShiftBanditEnv(**kwargs), spec
        )
    if env_id == "PyBulletUAVWaypointSupportShift-v0":
        return _apply_temporal_observation_wrappers(
            PyBulletUAVWaypointSupportShiftEnv(**kwargs), spec
        )
    if env_id == "SensorizedPyBulletUAVWaypoint-v0":
        return _apply_temporal_observation_wrappers(
            SensorizedPyBulletUAVWaypointEnv(**kwargs), spec
        )

    if env_id.startswith("MiniGrid-"):
        if not HAS_GYMNASIUM:
            return FallbackMiniGridImageEnv(env_id, **kwargs)
        import minigrid  # noqa: F401 - registers MiniGrid environments
        from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    env = gym.make(resolve_env_id(env_id), **kwargs)

    if spec.get("observation") == "fully_observable_image":
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)

    return _apply_temporal_observation_wrappers(env, spec)


def episode_succeeded(
    mode: str,
    terminated: bool,
    truncated: bool,
    final_reward: float,
) -> bool:
    if mode == "terminated":
        return bool(terminated and not truncated)
    if mode == "positive_terminal":
        return bool(terminated and final_reward > 0)
    raise ValueError(f"Unknown success mode: {mode}")
