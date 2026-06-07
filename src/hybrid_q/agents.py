from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, fields
from typing import Hashable

import numpy as np
import torch
from torch import nn


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    tabular_learning_rate: float = 0.2
    batch_size: int = 64
    replay_capacity: int = 50_000
    replay_warmup: int = 1_000
    target_update_interval: int = 250
    train_frequency: int = 1
    hidden_size: int = 128
    double_dqn: bool = True
    tau: float = 20.0
    fixed_gate: float = 0.5
    reliability_beta: float = 0.05
    reliability_prior: float = 1.0
    reliability_prior_strength: float = 10.0
    gate_min: float = 0.05
    gate_max: float = 0.95
    fuzzy_tau_support: float = 20.0
    fuzzy_uncertainty_scale: float = 1.0
    fuzzy_abstain_threshold: float = 0.5
    fuzzy_abstain_zero_support: bool = True
    abstain_action: int = -1


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.items: deque[tuple] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, transition: tuple) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        batch = [self.items[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.items)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class BaseAgent:
    def __init__(self, action_dim: int, seed: int):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.environment_steps = 0
        self.gradient_updates = 0
        self.training_support: set[Hashable] = set()
        self._last_decision = {
            "unsupported_state": 1.0,
            "memory_branch_weight": 0.0,
            "neural_branch_weight": 0.0,
            "abstention": 0.0,
            "adaptive_alpha": np.nan,
            "support_score": 0.0,
            "uncertainty_score": np.nan,
            "selected_branch": "none",
        }

    def q_values(self, state: np.ndarray, key: Hashable) -> np.ndarray:
        raise NotImplementedError

    def act(self, state: np.ndarray, key: Hashable, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))
        values = self.q_values(state, key)
        best = np.flatnonzero(np.isclose(values, values.max()))
        return int(self.rng.choice(best))

    def observe(
        self,
        state: np.ndarray,
        key: Hashable,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_key: Hashable,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        return {
            "memory_cost_states": 0.0,
            "memory_cost_entries": 0.0,
            "memory_cost_bytes_estimated": 0.0,
        }

    def decision_diagnostics(self) -> dict[str, float | str]:
        return dict(self._last_decision)

    def _record_decision(
        self,
        *,
        key: Hashable,
        memory_weight: float,
        neural_weight: float,
        abstention: float = 0.0,
        adaptive_alpha: float = np.nan,
        support_score: float = 0.0,
        uncertainty_score: float = np.nan,
    ) -> None:
        if abstention > 0:
            selected_branch = "abstention"
        elif memory_weight > neural_weight:
            selected_branch = "memory"
        elif neural_weight > memory_weight:
            selected_branch = "neural"
        else:
            selected_branch = "mixed"
        self._last_decision = {
            "unsupported_state": float(key not in self.training_support),
            "memory_branch_weight": float(memory_weight),
            "neural_branch_weight": float(neural_weight),
            "abstention": float(abstention),
            "adaptive_alpha": float(adaptive_alpha),
            "support_score": float(support_score),
            "uncertainty_score": float(uncertainty_score),
            "selected_branch": selected_branch,
        }


class RandomAgent(BaseAgent):
    def q_values(self, state: np.ndarray, key: Hashable) -> np.ndarray:
        self._record_decision(
            key=key,
            memory_weight=0.0,
            neural_weight=0.0,
        )
        return np.zeros(self.action_dim, dtype=np.float32)

    def observe(
        self,
        state: np.ndarray,
        key: Hashable,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_key: Hashable,
        done: bool,
    ) -> None:
        self.training_support.add(key)
        self.environment_steps += 1


class TabularQAgent(BaseAgent):
    def __init__(
        self,
        action_dim: int,
        seed: int,
        config: AgentConfig,
    ):
        super().__init__(action_dim, seed)
        self.config = config
        self.table: defaultdict[Hashable, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_dim, dtype=np.float32)
        )
        self.counts: defaultdict[Hashable, int] = defaultdict(int)

    def q_values(self, state: np.ndarray, key: Hashable) -> np.ndarray:
        supported = key in self.training_support
        self._record_decision(
            key=key,
            memory_weight=float(supported),
            neural_weight=0.0,
            abstention=float(not supported),
            support_score=float(supported),
        )
        values = self.table.get(key)
        if values is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        return values.copy()

    def observe(
        self,
        state: np.ndarray,
        key: Hashable,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_key: Hashable,
        done: bool,
    ) -> None:
        self.training_support.add(key)
        current = self.table[key][action]
        bootstrap = 0.0 if done else float(self.table[next_key].max())
        target = reward + self.config.gamma * bootstrap
        self.table[key][action] += (
            self.config.tabular_learning_rate * (target - current)
        )
        self.counts[key] += 1
        self.environment_steps += 1

    def diagnostics(self) -> dict[str, float]:
        states = len(self.table)
        entries = states * self.action_dim
        return {
            "memory_cost_states": float(states),
            "memory_cost_entries": float(entries),
            "memory_cost_bytes_estimated": float(
                entries * np.dtype(np.float32).itemsize
                + states * np.dtype(np.int64).itemsize
            ),
        }


class DQNAgent(BaseAgent):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        seed: int,
        config: AgentConfig,
    ):
        super().__init__(action_dim, seed)
        self.config = config
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.online = QNetwork(input_dim, action_dim, config.hidden_size).to(
            self.device
        )
        self.target = QNetwork(input_dim, action_dim, config.hidden_size).to(
            self.device
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(config.replay_capacity, seed)

    def neural_td_residual(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            next_state_t = torch.as_tensor(
                next_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            prediction = float(self.online(state_t)[0, action])
            if done:
                bootstrap = 0.0
            else:
                bootstrap = float(
                    self._next_values(next_state_t).squeeze(0)
                )
        target = reward + self.config.gamma * bootstrap
        return target - prediction

    def neural_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.online(tensor).squeeze(0).cpu().numpy()

    def q_values(self, state: np.ndarray, key: Hashable) -> np.ndarray:
        self._record_decision(
            key=key,
            memory_weight=0.0,
            neural_weight=1.0,
        )
        return self.neural_q_values(state)

    def _next_values(self, next_states: torch.Tensor) -> torch.Tensor:
        if self.config.double_dqn:
            next_actions = self.online(next_states).argmax(
                dim=1, keepdim=True
            )
            return self.target(next_states).gather(
                1, next_actions
            ).squeeze(1)
        return self.target(next_states).max(dim=1).values

    def observe(
        self,
        state: np.ndarray,
        key: Hashable,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_key: Hashable,
        done: bool,
    ) -> None:
        self.training_support.add(key)
        self.replay.add((state, action, reward, next_state, done))
        self.environment_steps += 1

        ready = len(self.replay) >= max(
            self.config.replay_warmup, self.config.batch_size
        )
        if ready and self.environment_steps % self.config.train_frequency == 0:
            self._gradient_step()

        if self.environment_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _gradient_step(self) -> None:
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        predicted = self.online(states_t).gather(1, actions_t[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = self._next_values(next_states_t)
            targets = rewards_t + (1.0 - dones_t) * self.config.gamma * next_values

        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.gradient_updates += 1


class HybridQAgent(DQNAgent):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        seed: int,
        config: AgentConfig,
        gate_kind: str,
    ):
        super().__init__(input_dim, action_dim, seed, config)
        self.gate_kind = gate_kind
        self.table: defaultdict[Hashable, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_dim, dtype=np.float32)
        )
        self.counts: defaultdict[Hashable, int] = defaultdict(int)
        self.tabular_error: defaultdict[Hashable, float] = defaultdict(
            lambda: self.config.reliability_prior
        )
        self.neural_error: defaultdict[Hashable, float] = defaultdict(
            lambda: self.config.reliability_prior
        )
        self.error_counts: defaultdict[Hashable, int] = defaultdict(int)
        self.global_tabular_error = self.config.reliability_prior
        self.global_neural_error = self.config.reliability_prior
        self.gate_sum = 0.0
        self.gate_queries = 0
        self.support_queries = 0
        self.support_abstentions = 0

    def _error_estimates(self, key: Hashable) -> tuple[float, float]:
        count = self.error_counts.get(key, 0)
        shrinkage = count / (
            count + self.config.reliability_prior_strength
        )
        tabular_error = (
            shrinkage
            * self.tabular_error.get(key, self.config.reliability_prior)
            + (1.0 - shrinkage) * self.global_tabular_error
        )
        neural_error = (
            shrinkage
            * self.neural_error.get(key, self.config.reliability_prior)
            + (1.0 - shrinkage) * self.global_neural_error
        )
        return float(tabular_error), float(neural_error)

    def fuzzy_gate_components(
        self, key: Hashable
    ) -> tuple[float, float, float]:
        count = self.counts.get(key, 0)
        tau = max(float(self.config.fuzzy_tau_support), 1e-8)
        support = float(1.0 - np.exp(-count / tau))
        _, neural_error = self._error_estimates(key)
        scale = max(float(self.config.fuzzy_uncertainty_scale), 1e-8)
        uncertainty = float(
            np.clip(neural_error / (neural_error + scale), 0.0, 1.0)
        )

        low_support = float(np.clip(1.0 - 2.0 * support, 0.0, 1.0))
        medium_support = float(
            np.clip(1.0 - abs(2.0 * support - 1.0), 0.0, 1.0)
        )
        high_support = float(np.clip(2.0 * support - 1.0, 0.0, 1.0))
        low_uncertainty = 1.0 - uncertainty
        high_uncertainty = uncertainty
        rules = (
            (low_support, 0.0),
            (medium_support * low_uncertainty, 0.35),
            (medium_support * high_uncertainty, 0.65),
            (high_support * low_uncertainty, 0.75),
            (high_support * high_uncertainty, 0.95),
        )
        total_membership = sum(weight for weight, _ in rules)
        alpha = (
            sum(weight * consequence for weight, consequence in rules)
            / total_membership
            if total_membership > 1e-12
            else 0.0
        )
        alpha = float(
            np.clip(alpha, self.config.gate_min, self.config.gate_max)
        )
        if count == 0:
            alpha = 0.0
        return alpha, support, uncertainty

    def gate(self, key: Hashable) -> float:
        if self.gate_kind == "fixed":
            return float(self.config.fixed_gate)
        if self.gate_kind in {"count", "support_abstain"}:
            count = self.counts.get(key, 0)
            if self.gate_kind == "support_abstain" and count == 0:
                return 1.0
            return float(count / (count + self.config.tau))
        if self.gate_kind == "reliability":
            tabular_error, neural_error = self._error_estimates(key)
            denominator = tabular_error + neural_error + 1e-8
            tabular_weight = neural_error / denominator
            return float(
                np.clip(
                    tabular_weight,
                    self.config.gate_min,
                    self.config.gate_max,
                )
            )
        if self.gate_kind == "fuzzy_support_adaptive":
            alpha, _, _ = self.fuzzy_gate_components(key)
            return alpha
        raise ValueError(f"Unknown gate kind: {self.gate_kind}")

    def q_values(self, state: np.ndarray, key: Hashable) -> np.ndarray:
        abstention = False
        adaptive_alpha = np.nan
        support_score = float(
            1.0 - np.exp(
                -self.counts.get(key, 0) / max(self.config.tau, 1e-8)
            )
        )
        uncertainty_score = np.nan
        if self.gate_kind == "support_abstain":
            self.support_queries += 1
            if self.counts.get(key, 0) == 0:
                self.support_abstentions += 1
                abstention = True
        if self.gate_kind == "fuzzy_support_adaptive":
            adaptive_alpha, support_score, uncertainty_score = (
                self.fuzzy_gate_components(key)
            )
            abstention = bool(
                self.config.fuzzy_abstain_zero_support
                and self.counts.get(key, 0) == 0
                and uncertainty_score
                >= self.config.fuzzy_abstain_threshold
            )
        gate = self.gate(key)
        self.gate_sum += gate
        self.gate_queries += 1
        tabular_values = self.table.get(key)
        if tabular_values is None:
            tabular_values = np.zeros(
                self.action_dim, dtype=np.float32
            )
        if abstention:
            self._record_decision(
                key=key,
                memory_weight=0.0,
                neural_weight=0.0,
                abstention=1.0,
                adaptive_alpha=adaptive_alpha,
                support_score=support_score,
                uncertainty_score=uncertainty_score,
            )
            values = np.zeros(self.action_dim, dtype=np.float32)
            if 0 <= self.config.abstain_action < self.action_dim:
                values.fill(-1.0)
                values[self.config.abstain_action] = 0.0
            return values
        self._record_decision(
            key=key,
            memory_weight=gate if self.counts.get(key, 0) > 0 else 0.0,
            neural_weight=1.0 - gate,
            adaptive_alpha=adaptive_alpha,
            support_score=support_score,
            uncertainty_score=uncertainty_score,
        )
        return (
            gate * tabular_values
            + (1.0 - gate) * self.neural_q_values(state)
        )

    def observe(
        self,
        state: np.ndarray,
        key: Hashable,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_key: Hashable,
        done: bool,
    ) -> None:
        current = self.table[key][action]
        bootstrap = 0.0 if done else float(self.table[next_key].max())
        target = reward + self.config.gamma * bootstrap
        tabular_residual = target - current
        neural_residual = self.neural_td_residual(
            state, action, reward, next_state, done
        )
        if self.gate_kind in {"reliability", "fuzzy_support_adaptive"}:
            beta = self.config.reliability_beta
            tabular_squared = tabular_residual**2
            neural_squared = neural_residual**2
            self.tabular_error[key] = (
                (1.0 - beta) * self.tabular_error[key]
                + beta * tabular_squared
            )
            self.neural_error[key] = (
                (1.0 - beta) * self.neural_error[key]
                + beta * neural_squared
            )
            self.global_tabular_error = (
                (1.0 - beta) * self.global_tabular_error
                + beta * tabular_squared
            )
            self.global_neural_error = (
                (1.0 - beta) * self.global_neural_error
                + beta * neural_squared
            )
            self.error_counts[key] += 1
        self.table[key][action] += (
            self.config.tabular_learning_rate * tabular_residual
        )
        self.counts[key] += 1
        super().observe(
            state, key, action, reward, next_state, next_key, done
        )

    def diagnostics(self) -> dict[str, float]:
        mean_gate = self.gate_sum / self.gate_queries if self.gate_queries else 0.0
        support_abstention_rate = (
            self.support_abstentions / self.support_queries
            if self.support_queries
            else 0.0
        )
        states = len(self.table)
        entries = states * self.action_dim
        return {
            "mean_gate": mean_gate,
            "support_abstention_rate": support_abstention_rate,
            "global_tabular_error": self.global_tabular_error,
            "global_neural_error": self.global_neural_error,
            "visited_states": float(len(self.counts)),
            "memory_cost_states": float(states),
            "memory_cost_entries": float(entries),
            "memory_cost_bytes_estimated": float(
                entries * np.dtype(np.float32).itemsize
                + states * np.dtype(np.int64).itemsize
            ),
        }


def create_agent(
    kind: str,
    input_dim: int,
    action_dim: int,
    seed: int,
    params: dict,
) -> BaseAgent:
    config_fields = {field.name for field in fields(AgentConfig)}
    config = AgentConfig(
        **{key: value for key, value in params.items() if key in config_fields}
    )
    if kind == "random":
        return RandomAgent(action_dim, seed)
    if kind == "tabular":
        return TabularQAgent(action_dim, seed, config)
    if kind == "dqn":
        return DQNAgent(input_dim, action_dim, seed, config)
    if kind == "double_dqn":
        config.double_dqn = True
        return DQNAgent(input_dim, action_dim, seed, config)
    if kind == "fixed_hybrid":
        return HybridQAgent(input_dim, action_dim, seed, config, gate_kind="fixed")
    if kind == "count_gated":
        return HybridQAgent(input_dim, action_dim, seed, config, gate_kind="count")
    if kind == "support_abstain_gate":
        return HybridQAgent(
            input_dim,
            action_dim,
            seed,
            config,
            gate_kind="support_abstain",
        )
    if kind == "reliability_gated":
        return HybridQAgent(
            input_dim, action_dim, seed, config, gate_kind="reliability"
        )
    if kind == "fuzzy_support_adaptive_gate":
        return HybridQAgent(
            input_dim,
            action_dim,
            seed,
            config,
            gate_kind="fuzzy_support_adaptive",
        )
    raise ValueError(f"Unknown agent kind: {kind}")
