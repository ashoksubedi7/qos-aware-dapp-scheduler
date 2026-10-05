import pytest

from config.experiment_config import (
    ExperimentConfig,
)


def test_zero_drain_duration_is_valid():
    config = ExperimentConfig(
        drain_duration_ms=0.0,
    )

    assert (
        config.drain_duration_ms
        == 0.0
    )


def test_positive_drain_duration_is_valid():
    config = ExperimentConfig(
        drain_duration_ms=100.0,
    )

    assert (
        config.drain_duration_ms
        == 100.0
    )


def test_negative_drain_duration_is_rejected():
    with pytest.raises(
        ValueError,
        match="drain_duration_ms",
    ):
        ExperimentConfig(
            drain_duration_ms=-1.0,
        )
