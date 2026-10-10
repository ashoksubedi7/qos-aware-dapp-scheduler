import pytest

from config.study_seed_partitions import (
    CALIBRATION_SEEDS,
    StudySeedPartitions,
)


def test_calibration_namespace_is_reserved():
    assert CALIBRATION_SEEDS == (
        7,
        17,
        27,
    )


def test_default_partition_plan_fails_closed():
    partitions = StudySeedPartitions()

    assert partitions.unresolved_items() == (
        "pilot_seeds",
        "training_seeds",
        "checkpoint_validation_seeds",
        "final_test_seeds",
    )

    with pytest.raises(
        RuntimeError
    ):
        partitions.assert_ready()


def test_calibration_seed_reuse_fails():
    with pytest.raises(
        ValueError
    ):
        StudySeedPartitions(
            training_seeds=(
                7,
            )
        )


def test_overlap_between_scientific_namespaces_fails():
    with pytest.raises(
        ValueError
    ):
        StudySeedPartitions(
            pilot_seeds=(
                101,
            ),
            training_seeds=(
                101,
            ),
        )


def test_duplicate_seed_within_namespace_fails():
    with pytest.raises(
        ValueError
    ):
        StudySeedPartitions(
            training_seeds=(
                101,
                101,
            )
        )


def test_negative_seed_fails():
    with pytest.raises(
        ValueError
    ):
        StudySeedPartitions(
            training_seeds=(
                -1,
            )
        )


def test_noninteger_seed_fails():
    with pytest.raises(
        TypeError
    ):
        StudySeedPartitions(
            training_seeds=(
                1.5,
            )
        )


@pytest.mark.parametrize(
    "seed",
    (
        True,
        False,
    ),
)
def test_boolean_seed_fails(
    seed,
):
    with pytest.raises(
        TypeError,
        match="training_seeds must contain integer seeds",
    ):
        StudySeedPartitions(
            training_seeds=(
                seed,
            )
        )


def test_non_tuple_namespace_fails():
    with pytest.raises(
        TypeError
    ):
        StudySeedPartitions(
            training_seeds=[
                101,
            ]
        )


def test_disjoint_fully_specified_fixture_is_ready():
    # These values are unit-test fixtures only.
    # They do NOT select the scientific-study seeds.
    partitions = StudySeedPartitions(
        pilot_seeds=(
            101,
            102,
        ),
        training_seeds=(
            201,
            202,
        ),
        checkpoint_validation_seeds=(
            301,
            302,
        ),
        final_test_seeds=(
            401,
            402,
        ),
    )

    assert (
        partitions.unresolved_items()
        ==
        ()
    )

    assert (
        partitions.assert_ready()
        is True
    )
