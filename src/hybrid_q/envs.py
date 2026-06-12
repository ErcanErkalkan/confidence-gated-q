from __future__ import annotations

import contextlib
import importlib.util
import io
from typing import Any

from .gym_compat import gym, spaces, HAS_GYMNASIUM
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
        if self.np_random.random() < self.slip_probability:
            action = int(self.np_random.integers(self.action_space.n))
        moves = ((0, 1), (1, 0), (0, -1), (-1, 0))
        dx, dy = moves[int(action)]
        candidate = (
            self.agent_position[0] + dx,
            self.agent_position[1] + dy,
        )
        hit_wall = (
            candidate in self.walls
            or not 0 <= candidate[0] < self.size
            or not 0 <= candidate[1] < self.size
        )
        if not hit_wall:
            self.agent_position = candidate
        self.steps += 1
        terminated = self.agent_position == self.goal_position
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else (-0.02 if hit_wall else -0.01)
        return self._observation(), reward, terminated, truncated, {}


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

    def step(self, action: int):
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
                reward -= 0.08
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


class ReliabilityShiftBanditEnv(gym.Env):
    """Contextual bandit with recurring states and a delayed reward shift."""

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
        return self._observation(), {
            "post_shift": self._post_shift(),
        }

    def step(self, action: int):
        post_shift = self._post_shift()
        boundary = (
            self.post_boundary if post_shift else self.pre_boundary
        )
        context = self.context_index / (self.context_count - 1)
        optimal_action = int(context >= boundary)
        success = int(action) == optimal_action
        reward = 1.0 if success else -1.0
        self.total_steps += 1
        return self._observation(), reward, True, False, {
            "post_shift": post_shift,
            "optimal_action": optimal_action,
        }


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


def make_env(spec: dict[str, Any], evaluation: bool = False) -> gym.Env:
    env_id = spec["id"]
    kwargs = dict(spec.get("kwargs", {}))
    if evaluation:
        kwargs.update(spec.get("eval_kwargs", {}))

    if env_id == "StructuredFourRooms-v0":
        return StructuredFourRoomsEnv(**kwargs)
    if env_id == "ApplicationNavigationSupportShift-v0":
        return ApplicationNavigationSupportShiftEnv(**kwargs)
    if env_id == "ReliabilityShiftBandit-v0":
        return ReliabilityShiftBanditEnv(**kwargs)
    if env_id == "PyBulletUAVWaypointSupportShift-v0":
        return PyBulletUAVWaypointSupportShiftEnv(**kwargs)

    if env_id.startswith("MiniGrid-"):
        if not HAS_GYMNASIUM:
            return FallbackMiniGridImageEnv(env_id, **kwargs)
        import minigrid  # noqa: F401 - registers MiniGrid environments
        from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    env = gym.make(resolve_env_id(env_id), **kwargs)

    if spec.get("observation") == "fully_observable_image":
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)

    return env


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
