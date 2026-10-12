import copy

import pytest

from config.dqn_hyperparameters import DQNHyperparameters
from config.experiment_config import ExperimentConfig


def make_config():
    return ExperimentConfig(
        model_variant="M2",
        seed=101,
        training_mode=False,
        checkpoint_path="/tmp/synthetic-checkpoint",
        traffic_scenario="CONGESTED",
        radio_scenario="FLUCTUATING",
        dqn_hyperparameters=DQNHyperparameters(
            learning_rate=0.0005,
            target_update_interval=125,
        ),
    )


def deserialize(data):
    from experiments.scientific_evaluation_worker import (
        deserialize_evaluation_config,
    )

    return deserialize_evaluation_config(data)


def test_configuration_round_trip():
    original = make_config()
    serialized = original.to_dict()
    preserved = copy.deepcopy(serialized)

    restored = deserialize(serialized)

    assert isinstance(restored, ExperimentConfig)
    assert isinstance(
        restored.dqn_hyperparameters,
        DQNHyperparameters,
    )
    assert restored == original
    assert restored.to_dict() == preserved
    assert serialized == preserved


def test_rejects_training_mode():
    data = make_config().to_dict()
    data["training_mode"] = True

    with pytest.raises(ValueError, match="training_mode"):
        deserialize(data)


def test_rejects_missing_top_level_field():
    data = make_config().to_dict()
    del data["seed"]

    with pytest.raises(ValueError, match="missing"):
        deserialize(data)


def test_rejects_unknown_top_level_field():
    data = make_config().to_dict()
    data["unrecognized_parameter"] = 123

    with pytest.raises(ValueError, match="unknown"):
        deserialize(data)


def test_rejects_missing_nested_hyperparameter():
    data = make_config().to_dict()
    del data["dqn_hyperparameters"]["target_update_interval"]

    with pytest.raises(ValueError, match="missing"):
        deserialize(data)


def test_rejects_unknown_nested_hyperparameter():
    data = make_config().to_dict()
    data["dqn_hyperparameters"]["unexpected"] = 5

    with pytest.raises(ValueError, match="unknown"):
        deserialize(data)


def test_rejects_invalid_nested_hyperparameter_type():
    data = make_config().to_dict()
    data["dqn_hyperparameters"] = []

    with pytest.raises(
        TypeError,
        match="dqn_hyperparameters",
    ):
        deserialize(data)


def test_rejects_missing_evaluation_checkpoint():
    data = make_config().to_dict()
    data["checkpoint_path"] = None

    with pytest.raises(ValueError, match="checkpoint_path"):
        deserialize(data)
