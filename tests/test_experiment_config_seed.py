import pytest

from config.experiment_config import (
    ExperimentConfig,
)


@pytest.mark.parametrize(
    "seed",
    (
        0,
        11,
    ),
)
def test_integer_seed_is_valid(
    seed,
):
    config = ExperimentConfig(
        seed=seed,
    )

    assert config.seed == seed
    assert type(config.seed) is int


@pytest.mark.parametrize(
    "seed",
    (
        True,
        False,
        11.0,
        11.5,
    ),
)
def test_noninteger_or_boolean_seed_fails(
    seed,
):
    with pytest.raises(
        TypeError,
        match="seed must be a non-Boolean integer",
    ):
        ExperimentConfig(
            seed=seed,
        )


def test_negative_integer_seed_fails():
    with pytest.raises(
        ValueError,
        match="seed must be non-negative",
    ):
        ExperimentConfig(
            seed=-1,
        )
