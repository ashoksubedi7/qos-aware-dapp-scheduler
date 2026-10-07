import pytest

from environment.traffic_scenarios import (
    get_traffic_scenario,
)


def test_baseline_traffic_profile():
    scenario = get_traffic_scenario(
        "BASELINE"
    )

    assert scenario.name == "BASELINE"

    assert (
        scenario.embb.users_dl
        == 4
    )

    assert (
        scenario.urllc.users_dl
        == 8
    )

    assert (
        scenario.mmtc.users_dl
        == 30
    )


def test_scenario_lookup_is_case_insensitive():
    scenario = get_traffic_scenario(
        "congested"
    )

    assert (
        scenario.name
        == "CONGESTED"
    )


def test_unknown_scenario_fails_closed():
    with pytest.raises(
        ValueError
    ):
        get_traffic_scenario(
            "NOT_REAL"
        )


def test_stress_scenarios_are_distinct():
    baseline = get_traffic_scenario(
        "BASELINE"
    )

    congested = get_traffic_scenario(
        "CONGESTED"
    )

    urllc_high = get_traffic_scenario(
        "URLLC_HIGH"
    )

    simultaneous = get_traffic_scenario(
        "SIMULTANEOUS_HIGH"
    )

    assert (
        congested
        != baseline
    )

    assert (
        urllc_high
        != baseline
    )

    assert (
        simultaneous
        != baseline
    )
