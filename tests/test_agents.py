import numpy as np
import torch
from torch import nn

from src.hybrid_q.agents import A2CAgent, AgentConfig, DQNAgent, HybridQAgent, TabularQAgent


def test_tabular_update_moves_toward_reward():
    agent = TabularQAgent(2, seed=0, config=AgentConfig(gamma=0.9))
    state = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(state, "s", 1, 2.0, next_state, "next", True)
    assert agent.q_values(state, "s")[1] > 0


def test_optimizer_contains_each_online_parameter_once():
    agent = DQNAgent(
        input_dim=2,
        action_dim=3,
        seed=0,
        config=AgentConfig(),
    )
    optimized = [
        parameter
        for group in agent.optimizer.param_groups
        for parameter in group["params"]
    ]
    online = list(agent.online.parameters())
    target_ids = {id(parameter) for parameter in agent.target.parameters()}

    assert len(optimized) == len({id(parameter) for parameter in optimized})
    assert {id(parameter) for parameter in optimized} == {
        id(parameter) for parameter in online
    }
    assert not ({id(parameter) for parameter in optimized} & target_ids)


def test_count_gate_increases_with_visits():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(
            batch_size=2,
            replay_warmup=100,
            tau=5,
        ),
        gate_kind="count",
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    before = agent.gate("s")
    for _ in range(5):
        agent.observe(state, "s", 0, 1.0, next_state, "next", True)
    after = agent.gate("s")
    assert before == 0.0
    assert after == 0.5


def test_reliability_gate_favors_lower_error_estimator():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(gate_min=0.0, gate_max=1.0),
        gate_kind="reliability",
    )
    agent.global_tabular_error = 0.1
    agent.global_neural_error = 1.0
    assert agent.gate("unseen") > 0.8


def test_support_abstain_uses_zero_tabular_values_outside_support():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=3,
        seed=0,
        config=AgentConfig(replay_warmup=100, tau=5),
        gate_kind="support_abstain",
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    values = agent.q_values(state, "unseen")
    assert np.array_equal(values, np.zeros(3, dtype=np.float32))
    assert agent.gate("unseen") == 1.0
    assert agent.diagnostics()["support_abstention_rate"] == 1.0


def test_support_abstain_reverts_to_count_gate_after_visit():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(replay_warmup=100, tau=5),
        gate_kind="support_abstain",
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(state, "seen", 0, 1.0, next_state, "next", True)
    assert np.isclose(agent.gate("seen"), 1.0 / 6.0)


def test_fuzzy_gate_is_bounded_and_increases_with_support():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(
            gate_min=0.0,
            gate_max=1.0,
            fuzzy_tau_support=5.0,
        ),
        gate_kind="fuzzy_support_adaptive",
    )
    unseen_alpha, unseen_support, _ = agent.fuzzy_gate_components("s")
    agent.counts["s"] = 50
    seen_alpha, seen_support, _ = agent.fuzzy_gate_components("s")
    assert unseen_alpha == 0.0
    assert unseen_support == 0.0
    assert 0.0 <= seen_alpha <= 1.0
    assert seen_support > 0.99
    assert seen_alpha > unseen_alpha


def test_fuzzy_gate_favors_memory_when_neural_uncertainty_increases():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(
            gate_min=0.0,
            gate_max=1.0,
            fuzzy_tau_support=5.0,
            reliability_prior_strength=0.0,
        ),
        gate_kind="fuzzy_support_adaptive",
    )
    agent.counts["s"] = 20
    agent.error_counts["s"] = 20
    agent.neural_error["s"] = 0.01
    low_uncertainty_alpha, _, _ = agent.fuzzy_gate_components("s")
    agent.neural_error["s"] = 100.0
    high_uncertainty_alpha, _, _ = agent.fuzzy_gate_components("s")
    assert high_uncertainty_alpha > low_uncertainty_alpha


def test_fuzzy_gate_can_use_neural_fallback_outside_exact_support():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(fuzzy_abstain_zero_support=False),
        gate_kind="fuzzy_support_adaptive",
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    agent.q_values(state, "unseen")
    decision = agent.decision_diagnostics()
    assert decision["unsupported_state"] == 1.0
    assert decision["neural_branch_weight"] == 1.0
    assert decision["abstention"] == 0.0


def test_support_abstention_can_select_a_declared_safe_action():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=5,
        seed=0,
        config=AgentConfig(abstain_action=4),
        gate_kind="support_abstain",
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    assert agent.act(state, "unseen", epsilon=0.0) == 4


class FixedQNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer(
            "values", torch.tensor(values, dtype=torch.float32)
        )

    def forward(self, states):
        return self.values.repeat(states.shape[0], 1)


def test_double_dqn_separates_online_selection_from_target_evaluation():
    agent = DQNAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(double_dqn=True),
    )
    agent.online = FixedQNetwork([1.0, 2.0])
    agent.target = FixedQNetwork([10.0, 0.0])
    values = agent._next_values(torch.zeros((1, 2)))
    assert values.item() == 0.0


def test_vanilla_dqn_uses_target_network_maximum():
    agent = DQNAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(double_dqn=False),
    )
    agent.online = FixedQNetwork([1.0, 2.0])
    agent.target = FixedQNetwork([10.0, 0.0])
    values = agent._next_values(torch.zeros((1, 2)))
    assert values.item() == 10.0


def test_approximate_support_gate_retrieves_nearby_memory():
    agent = HybridQAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(
            replay_warmup=100,
            approximate_support_k=1,
            approximate_support_bandwidth=10.0,
            approximate_support_tau=1.0,
        ),
        gate_kind="approx_count",
    )
    seen = np.array([1.0, 0.0], dtype=np.float32)
    nearby = np.array([0.9, 0.1], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(seen, "seen", 1, 2.0, next_state, "next", True)
    values = agent.q_values(nearby, "unseen_nearby")
    decision = agent.decision_diagnostics()
    assert decision["unsupported_state"] == 1.0
    assert decision["memory_branch_weight"] > 0.0
    assert values[1] > values[0]


def test_dueling_dqn_network_outputs_action_values():
    agent = DQNAgent(
        input_dim=3,
        action_dim=4,
        seed=0,
        config=AgentConfig(network_architecture="dueling_mlp"),
    )
    values = agent.q_values(np.zeros(3, dtype=np.float32), "s")
    assert values.shape == (4,)


def test_a2c_agent_updates_online():
    agent = A2CAgent(
        input_dim=2,
        action_dim=2,
        seed=0,
        config=AgentConfig(learning_rate=0.001),
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(state, "s", 0, 1.0, next_state, "next", True)
    assert agent.gradient_updates == 1
    assert agent.environment_steps == 1
