import pytest

from config.dqn_hyperparameters import (
    DQNHyperparameters,
)


def test_canonical_dqn_defaults():
    config = DQNHyperparameters()

    assert config.learning_rate == pytest.approx(
        1e-3
    )

    assert config.gamma == pytest.approx(
        0.99
    )

    assert config.replay_capacity == 100_000
    assert config.batch_size == 64
    assert config.min_replay_size == 1_000
    assert config.target_update_interval == 250

    assert config.epsilon_start == pytest.approx(
        1.0
    )

    assert config.epsilon_min == pytest.approx(
        0.05
    )

    assert config.epsilon_decay == pytest.approx(
        0.9995
    )


def test_min_replay_must_cover_batch():
    with pytest.raises(
        ValueError
    ):
        DQNHyperparameters(
            batch_size=128,
            min_replay_size=64,
        )


def test_replay_capacity_must_cover_minimum():
    with pytest.raises(
        ValueError
    ):
        DQNHyperparameters(
            replay_capacity=500,
            min_replay_size=1_000,
        )


def test_invalid_epsilon_order_fails():
    with pytest.raises(
        ValueError
    ):
        DQNHyperparameters(
            epsilon_start=0.1,
            epsilon_min=0.2,
        )


def test_to_dict_is_reproducible():
    config = DQNHyperparameters()

    manifest = config.to_dict()

    assert manifest[
        "learning_rate"
    ] == pytest.approx(
        1e-3
    )

    assert manifest[
        "replay_capacity"
    ] == 100_000
