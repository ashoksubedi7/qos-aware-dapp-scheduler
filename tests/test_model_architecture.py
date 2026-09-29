import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from agent.model_definitions import get_model_definition
from agent.q_network import build_q_network
from common.action_space import (
    ACTION_SPACE_SIZE,
    ACTION_WEIGHTS,
    get_action_weights,
)


def test_action_space_size():
    assert ACTION_SPACE_SIZE == 22
    assert ACTION_WEIGHTS.shape == (22, 3)


def test_action_weights_sum():
    for action, weights in enumerate(ACTION_WEIGHTS):
        if action == 0:
            assert np.sum(weights) == 0
        else:
            assert np.sum(weights) == 100


def test_known_actions():
    assert np.array_equal(
        get_action_weights(0),
        np.array([0, 0, 0]),
    )

    assert np.array_equal(
        get_action_weights(6),
        np.array([0, 100, 0]),
    )

    assert np.array_equal(
        get_action_weights(11),
        np.array([100, 0, 0]),
    )


def test_m1_architecture():
    definition = get_model_definition("M1")

    model = build_q_network(
        definition["input_dim"],
        definition["hidden_units"],
    )

    assert model.input_shape == (None, 6)
    assert model.output_shape == (None, 22)


def test_m2_architecture():
    definition = get_model_definition("M2")

    model = build_q_network(
        definition["input_dim"],
        definition["hidden_units"],
    )

    assert model.input_shape == (None, 7)
    assert model.output_shape == (None, 22)


def test_m3_matches_m2_architecture():
    m2 = get_model_definition("M2")
    m3 = get_model_definition("M3")

    assert m2["input_dim"] == m3["input_dim"]
    assert m2["hidden_units"] == m3["hidden_units"]
    assert m2["output_dim"] == m3["output_dim"]
