import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.dqn_agent import DQNAgent
from config.dqn_hyperparameters import (
    DQNHyperparameters,
)


def make_agent(input_dim=6):
    return DQNAgent(
        input_dim=input_dim,
        action_dim=22,
        hidden_units=(64, 64),
        hyperparameters=DQNHyperparameters(
            replay_capacity=100,
            batch_size=2,
            min_replay_size=2,
            target_update_interval=2,
        ),
        seed=7,
    )


def test_agent_preserves_normalized_state():
    agent = make_agent()

    state = np.array(
        [0.2, 0.4, 0.6, 0.3, 0.5, 0.7],
        dtype=np.float32,
    )

    prepared = agent._prepare_state(state)

    assert np.allclose(
        prepared,
        state,
    )


def test_wrong_state_shape_raises():
    agent = make_agent()

    with pytest.raises(ValueError):
        agent._prepare_state(
            np.zeros(3)
        )


def test_remember_transition():
    agent = make_agent()

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    next_state = np.ones(
        6,
        dtype=np.float32,
    )

    agent.remember(
        state,
        3,
        1.0,
        next_state,
        False,
    )

    assert len(agent.replay_buffer) == 1


def test_invalid_action_raises():
    agent = make_agent()

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        agent.remember(
            state,
            22,
            0.0,
            state,
            False,
        )


def test_training_waits_for_minimum_replay():
    agent = make_agent()

    assert agent.can_train() is False
    assert agent.train_step() is None


def test_epsilon_decay_respects_minimum():
    agent = make_agent()

    agent.epsilon = 0.051
    agent.epsilon_decay = 0.5

    value = agent.decay_epsilon()

    assert value == agent.epsilon_min


def test_action_in_valid_range():
    agent = make_agent()

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    action = agent.select_action(
        state,
        explore=True,
    )

    assert 0 <= action < 22
def test_target_network_starts_synchronized():
    agent = make_agent()

    online = agent.online_model.get_weights()
    target = agent.target_model.get_weights()

    assert len(online) == len(target)

    for online_w, target_w in zip(online, target):
        assert np.allclose(online_w, target_w)


def test_training_step_updates_online_network():
    agent = make_agent()

    state_a = np.zeros(6, dtype=np.float32)
    state_b = np.ones(6, dtype=np.float32)

    before = [
        weight.copy()
        for weight in agent.online_model.get_weights()
    ]

    agent.remember(
        state_a,
        0,
        1.0,
        state_b,
        False,
    )

    agent.remember(
        state_b,
        1,
        -1.0,
        state_a,
        True,
    )

    loss = agent.train_step()

    after = agent.online_model.get_weights()

    changed = any(
        not np.allclose(a, b)
        for a, b in zip(before, after)
    )

    assert loss is not None
    assert changed


def test_seeded_exploration_is_reproducible():
    agent_a = make_agent()
    agent_b = make_agent()

    state = np.zeros(6, dtype=np.float32)

    actions_a = [
        agent_a.select_action(state, explore=True)
        for _ in range(10)
    ]

    actions_b = [
        agent_b.select_action(state, explore=True)
        for _ in range(10)
    ]

    assert actions_a == actions_b
