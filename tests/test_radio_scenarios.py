import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from environment.radio_scenarios import (
    RadioScenario,
    STATIC_GOOD_DB,
    STATIC_MODERATE_DB,
    STATIC_POOR_DB,
)


def make_radio(
    scenario,
    seed=7,
    slice_name="URLLC",
    ue_id="ue1",
):
    return RadioScenario(
        scenario=scenario,
        experiment_seed=seed,
        slice_name=slice_name,
        ue_id=ue_id,
        simulation_time_ms=1000,
        update_interval_ms=50,
    )


def test_static_good_is_constant():
    radio = make_radio(
        "STATIC_GOOD"
    )

    assert radio.sinr_at(0) == STATIC_GOOD_DB
    assert radio.sinr_at(500) == STATIC_GOOD_DB
    assert radio.sinr_at(999) == STATIC_GOOD_DB


def test_static_moderate_is_constant():
    radio = make_radio(
        "STATIC_MODERATE"
    )

    assert (
        radio.sinr_at(100)
        == STATIC_MODERATE_DB
    )


def test_static_poor_is_constant():
    radio = make_radio(
        "STATIC_POOR"
    )

    assert (
        radio.sinr_at(100)
        == STATIC_POOR_DB
    )


def test_step_degradation_changes_at_midpoint():
    radio = make_radio(
        "STEP_DEGRADATION"
    )

    assert (
        radio.sinr_at(499)
        == STATIC_GOOD_DB
    )

    assert (
        radio.sinr_at(500)
        == STATIC_POOR_DB
    )


def test_step_recovery_changes_at_midpoint():
    radio = make_radio(
        "STEP_RECOVERY"
    )

    assert (
        radio.sinr_at(499)
        == STATIC_POOR_DB
    )

    assert (
        radio.sinr_at(500)
        == STATIC_GOOD_DB
    )


def test_fluctuating_trace_is_reproducible():
    a = make_radio(
        "FLUCTUATING",
        seed=17,
        slice_name="URLLC",
        ue_id="ue3",
    )

    b = make_radio(
        "FLUCTUATING",
        seed=17,
        slice_name="URLLC",
        ue_id="ue3",
    )

    trace_a = [
        a.sinr_at(t)
        for t in range(
            0,
            500,
            50,
        )
    ]

    trace_b = [
        b.sinr_at(t)
        for t in range(
            0,
            500,
            50,
        )
    ]

    assert trace_a == trace_b


def test_different_ue_gets_different_trace():
    a = make_radio(
        "FLUCTUATING",
        ue_id="ue1",
    )

    b = make_radio(
        "FLUCTUATING",
        ue_id="ue2",
    )

    trace_a = [
        a.sinr_at(t)
        for t in range(
            0,
            500,
            50,
        )
    ]

    trace_b = [
        b.sinr_at(t)
        for t in range(
            0,
            500,
            50,
        )
    ]

    assert trace_a != trace_b


def test_invalid_scenario_rejected():
    with pytest.raises(
        ValueError,
        match="unsupported radio scenario",
    ):
        make_radio(
            "UNKNOWN"
        )
def test_same_time_returns_same_fluctuating_value():
    radio = make_radio(
        "FLUCTUATING",
        seed=17,
        slice_name="URLLC",
        ue_id="ue3",
    )

    first = radio.sinr_at(200)
    second = radio.sinr_at(200)

    assert first == second
