import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from agent.dqn_agent import DQNAgent
from common.action_space import ACTION_SPACE_SIZE
from config.dqn_hyperparameters import (
    DQNHyperparameters,
)

def make_agent():
    return DQNAgent(
        input_dim=6,
        action_dim=ACTION_SPACE_SIZE,
        hidden_units=(64, 64),
        hyperparameters=DQNHyperparameters(
            batch_size=1,
            min_replay_size=1,
        ),
        seed=7,
    )


def test_checkpoint_round_trip(tmp_path):
    agent = make_agent()

    state = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        dtype=np.float32,
    )

    q_before = agent.model.predict(
        state.reshape(1, -1),
        verbose=0,
    )

    agent.save_checkpoint(
        tmp_path,
        metadata={
            "model_variant": "M1",
            "seed": 7,
        },
    )

    restored = make_agent()

    metadata = (
        restored.load_checkpoint(
            tmp_path
        )
    )

    q_after = restored.model.predict(
        state.reshape(1, -1),
        verbose=0,
    )

    assert np.allclose(
        q_before,
        q_after,
    )

    assert (
        metadata["model_variant"]
        == "M1"
    )

    assert metadata["seed"] == 7


def test_evaluation_mode_sets_zero_epsilon():
    agent = make_agent()

    agent.epsilon = 0.8

    agent.set_evaluation_mode(
        True
    )

    assert agent.epsilon == 0.0


def test_evaluation_mode_blocks_replay():
    agent = make_agent()

    agent.set_evaluation_mode(
        True
    )

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    before = len(
        agent.replay_buffer
    )

    agent.remember(
        state,
        0,
        1.0,
        state,
        False,
    )

    after = len(
        agent.replay_buffer
    )

    assert before == after


def test_evaluation_mode_blocks_training():
    agent = make_agent()

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    agent.remember(
        state,
        0,
        1.0,
        state,
        False,
    )

    weights_before = [
        w.copy()
        for w in agent.model.get_weights()
    ]

    agent.set_evaluation_mode(
        True
    )

    result = agent.train_step()

    weights_after = (
        agent.model.get_weights()
    )

    assert result is None

    for before, after in zip(
        weights_before,
        weights_after,
    ):
        assert np.array_equal(
            before,
            after,
        )
