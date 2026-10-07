from experiments.drain_sweep import (
    ACTIVE_DURATION_MS,
    CALIBRATION_SEEDS,
    DRAIN_CANDIDATES_MS,
    TRAFFIC_SCENARIOS,
)


def test_drain_sweep_matrix_size():
    total = (
        len(
            TRAFFIC_SCENARIOS
        )
        * len(
            CALIBRATION_SEEDS
        )
        * len(
            DRAIN_CANDIDATES_MS
        )
    )

    assert total == 96


def test_active_duration_is_fixed():
    assert (
        ACTIVE_DURATION_MS
        == 10_000.0
    )


def test_drain_candidates_are_strictly_increasing():
    assert (
        tuple(
            sorted(
                DRAIN_CANDIDATES_MS
            )
        )
        == DRAIN_CANDIDATES_MS
    )

    assert (
        len(
            set(
                DRAIN_CANDIDATES_MS
            )
        )
        == len(
            DRAIN_CANDIDATES_MS
        )
    )

import json

from experiments.drain_sweep import (
    load_drain_sweep,
    save_drain_sweep,
)


def test_save_and_load_drain_sweep(
    tmp_path,
):
    output = (
        tmp_path
        / "drain_sweep.json"
    )

    results = [
        {
            "traffic_scenario": "BASELINE",
            "seed": 7,
            "drain_duration_ms": 0.0,
        }
    ]

    save_drain_sweep(
        results,
        output,
    )

    assert output.exists()

    assert not (
        tmp_path
        / "drain_sweep.json.tmp"
    ).exists()

    loaded = load_drain_sweep(
        output
    )

    assert loaded == results

    with output.open(
        "r",
        encoding="utf-8",
    ) as handle:
        assert (
            json.load(handle)
            == results
        )
