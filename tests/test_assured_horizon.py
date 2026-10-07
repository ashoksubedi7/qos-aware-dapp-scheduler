import pytest

from experiments.assured_horizon import (
    build_experiment_horizon,
)


def test_total_duration_is_active_plus_drain():
    horizon = build_experiment_horizon(
        active_duration_ms=10_000,
        drain_duration_ms=250,
    )

    assert (
        horizon.active_duration_ms
        == pytest.approx(
            10_000.0
        )
    )

    assert (
        horizon.drain_duration_ms
        == pytest.approx(
            250.0
        )
    )

    assert (
        horizon.total_duration_ms
        == pytest.approx(
            10_250.0
        )
    )


def test_active_duration_stays_constant_across_drain_candidates():
    active = 10_000

    drain_candidates = [
        0,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
    ]

    horizons = [
        build_experiment_horizon(
            active_duration_ms=active,
            drain_duration_ms=drain,
        )
        for drain in drain_candidates
    ]

    assert all(
        horizon.active_duration_ms
        == active
        for horizon in horizons
    )

    assert [
        horizon.total_duration_ms
        for horizon in horizons
    ] == [
        10_000,
        10_010,
        10_025,
        10_050,
        10_100,
        10_250,
        10_500,
        11_000,
    ]


@pytest.mark.parametrize(
    "active_duration_ms",
    [
        0,
        -1,
    ],
)
def test_invalid_active_duration_rejected(
    active_duration_ms,
):
    with pytest.raises(
        ValueError,
        match="active_duration_ms",
    ):
        build_experiment_horizon(
            active_duration_ms=(
                active_duration_ms
            ),
            drain_duration_ms=0,
        )


def test_negative_drain_rejected():
    with pytest.raises(
        ValueError,
        match="drain_duration_ms",
    ):
        build_experiment_horizon(
            active_duration_ms=1000,
            drain_duration_ms=-1,
        )
