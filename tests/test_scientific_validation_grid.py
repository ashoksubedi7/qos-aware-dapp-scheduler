from types import SimpleNamespace

import pytest

from experiments.scientific_validation_grid import (
    validate_declared_validation_grid,
)


TRAFFIC_SCENARIOS = (
    "BASELINE",
    "CONGESTED",
)

RADIO_SCENARIOS = (
    "STATIC_GOOD",
)

VALIDATION_SEEDS = (
    201,
    202,
)


def make_summary(traffic, radio, seed):
    return SimpleNamespace(
        model_variant="M1",
        traffic_scenario=traffic,
        radio_scenario=radio,
        seed=seed,
    )


def complete_grid():
    return [
        make_summary(traffic, radio, seed)
        for traffic in TRAFFIC_SCENARIOS
        for radio in RADIO_SCENARIOS
        for seed in VALIDATION_SEEDS
    ]


def validate(summaries, **overrides):
    options = {
        "traffic_scenarios": TRAFFIC_SCENARIOS,
        "radio_scenarios": RADIO_SCENARIOS,
        "seeds": VALIDATION_SEEDS,
    }

    options.update(overrides)

    return validate_declared_validation_grid(
        summaries,
        **options,
    )


def test_accepts_complete_declared_grid():
    result = validate(complete_grid())

    assert result["run_count"] == 4
    assert result["seeds"] == (201, 202)
    assert result["conditions"] == (
        ("BASELINE", "STATIC_GOOD"),
        ("CONGESTED", "STATIC_GOOD"),
    )


def test_rejects_entire_missing_condition():
    summaries = [
        item
        for item in complete_grid()
        if item.traffic_scenario == "BASELINE"
    ]

    with pytest.raises(ValueError, match="missing"):
        validate(summaries)


def test_rejects_entire_missing_seed():
    summaries = [
        item
        for item in complete_grid()
        if item.seed == 201
    ]

    with pytest.raises(ValueError, match="missing"):
        validate(summaries)


def test_rejects_one_missing_cell():
    summaries = complete_grid()[:-1]

    with pytest.raises(ValueError, match="missing"):
        validate(summaries)


def test_rejects_duplicate_condition_seed():
    summaries = complete_grid()
    summaries.append(summaries[0])

    with pytest.raises(ValueError, match="duplicate"):
        validate(summaries)


def test_rejects_undeclared_condition():
    summaries = complete_grid()

    summaries.append(
        make_summary(
            "URLLC_HIGH",
            "STATIC_GOOD",
            201,
        )
    )

    with pytest.raises(ValueError, match="unexpected"):
        validate(summaries)


@pytest.mark.parametrize(
    "overrides",
    [
        {"traffic_scenarios": ("BASELINE", "BASELINE")},
        {"radio_scenarios": ("STATIC_GOOD", "STATIC_GOOD")},
        {"seeds": (201, 201)},
    ],
)
def test_rejects_duplicate_declared_axis(overrides):
    with pytest.raises(ValueError, match="duplicate"):
        validate(complete_grid(), **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"traffic_scenarios": ()},
        {"radio_scenarios": ()},
        {"seeds": ()},
    ],
)
def test_rejects_empty_declared_axis(overrides):
    with pytest.raises(ValueError, match="empty"):
        validate(complete_grid(), **overrides)
