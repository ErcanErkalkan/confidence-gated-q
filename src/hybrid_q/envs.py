from __future__ import annotations

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
        if terminated:
            reward = 5.0
        else:
            reward = -0.02
            if collision:
                reward -= 0.23
            if in_risk_zone:
                reward -= 0.08
        info = {
            "collision": collision,
            "risk_zone": in_risk_zone,
            "goal_is_shifted": self.goal_position in self.test_goals,
        }
        return self._observation(), reward, terminated, truncated, info

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
