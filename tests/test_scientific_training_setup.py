from dataclasses import replace

import pytest

from config.dqn_hyperparameters import (
    DQNHyperparameters,
)
from config.experiment_config import (
    ExperimentConfig,
)
from config.training_protocol import (
    TrainingProtocol,
)
from experiments.scientific_training_setup import (
    build_scientific_training_setup,
)


def _protocol():
    # Fixture values only. They are not study authority.
    return TrainingProtocol(
        successful_gradient_update_budget=5,
        epsilon_floor_update=4,
        checkpoint_interval_updates=2,
    )


def test_setup_derives_epsilon_decay_from_protocol():
    base_config = ExperimentConfig(
        model_variant="M1",
        seed=101,
        training_mode=True,
        traffic_scenario="CONGESTED",
        radio_scenario="STATIC_GOOD",
    )

    setup = build_scientific_training_setup(
        protocol=_protocol(),
        config=base_config,
    )

    expected_decay = (
        base_config
        .dqn_hyperparameters
        .epsilon_min
        /
        base_config
        .dqn_hyperparameters
        .epsilon_start
    ) ** (1.0 / 4.0)

    assert (
        setup.config
        .dqn_hyperparameters
        .epsilon_decay
        == pytest.approx(expected_decay)
    )

    assert (
        setup.config
        .dqn_hyperparameters
        .epsilon_decay
        != base_config
        .dqn_hyperparameters
        .epsilon_decay
    )


def test_setup_preserves_nonderived_run_configuration():
    base_config = ExperimentConfig(
        model_variant="M2",
        seed=103,
        training_mode=True,
        traffic_scenario="URLLC_HIGH",
        radio_scenario="STATIC_MODERATE",
    )

    setup = build_scientific_training_setup(
        protocol=_protocol(),
        config=base_config,
    )

    assert setup.config.model_variant == "M2"
    assert setup.config.seed == 103

    assert (
        setup.config.traffic_scenario
        == "URLLC_HIGH"
    )

    assert (
        setup.config.radio_scenario
        == "STATIC_MODERATE"
    )

    assert (
        setup.config.control_interval_ms
        == base_config.control_interval_ms
    )

    assert (
        setup.config.drain_duration_ms
        == base_config.drain_duration_ms
    )


def test_setup_derives_active_duration_from_replay_authority():
    hyperparameters = replace(
        DQNHyperparameters(),
        batch_size=64,
        min_replay_size=64,
    )

    base_config = ExperimentConfig(
        seed=107,
        training_mode=True,
        dqn_hyperparameters=hyperparameters,
    )

    setup = build_scientific_training_setup(
        protocol=_protocol(),
        config=base_config,
    )

    # 64 warm-up transitions, then 5 successful
    # updates total -> transition count = 64 + 5 - 1.
    assert (
        setup.horizon.active_duration_ms
        == pytest.approx(68.0)
    )

    assert (
        setup.horizon.drain_duration_ms
        == pytest.approx(
            base_config.drain_duration_ms
        )
    )

    assert (
        setup.horizon.total_duration_ms
        == pytest.approx(
            68.0
            + base_config.drain_duration_ms
        )
    )


def test_setup_does_not_mutate_input_config():
    base_config = ExperimentConfig(
        seed=109,
        training_mode=True,
    )

    original_hyperparameters = (
        base_config.dqn_hyperparameters
    )

    setup = build_scientific_training_setup(
        protocol=_protocol(),
        config=base_config,
    )

    assert setup.config is not base_config

    assert (
        base_config.dqn_hyperparameters
        is original_hyperparameters
    )

    assert (
        base_config
        .dqn_hyperparameters
        .epsilon_decay
        == DQNHyperparameters().epsilon_decay
    )


def test_setup_requires_training_mode():
    config = ExperimentConfig(
        seed=113,
        training_mode=False,
        checkpoint_path="checkpoint",
    )

    with pytest.raises(ValueError):
        build_scientific_training_setup(
            protocol=_protocol(),
            config=config,
        )


def test_setup_rejects_training_checkpoint_path():
    config = ExperimentConfig(
        seed=127,
        training_mode=True,
        checkpoint_path="old_checkpoint",
    )

    with pytest.raises(ValueError):
        build_scientific_training_setup(
            protocol=_protocol(),
            config=config,
        )


def test_setup_requires_frozen_training_protocol():
    config = ExperimentConfig(
        seed=131,
        training_mode=True,
    )

    protocol = TrainingProtocol()

    with pytest.raises(RuntimeError):
        build_scientific_training_setup(
            protocol=protocol,
            config=config,
        )
