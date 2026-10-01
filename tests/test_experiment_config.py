import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.experiment_config import (
    ExperimentConfig,
)


def test_default_config_is_valid():
    config = ExperimentConfig()

    assert config.model_variant == "M1"
    assert config.seed == 7
    assert config.control_interval_ms == 1.0


def test_config_serializes_to_dict():
    config = ExperimentConfig(
        model_variant="M2",
        seed=21,
    )

    result = config.to_dict()

    assert result["model_variant"] == "M2"
    assert result["seed"] == 21


def test_invalid_model_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(
            model_variant="M4"
        )


def test_invalid_control_interval_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(
            control_interval_ms=0
        )


def test_reward_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ExperimentConfig(
            reward_urllc_service=0.50
        )
