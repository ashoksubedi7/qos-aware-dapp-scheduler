from dataclasses import asdict

from environment.traffic_scenarios import (
    get_traffic_scenario,
)


def test_light_load_registered():
    scenario = get_traffic_scenario("LIGHT_LOAD")

    assert scenario.name == "LIGHT_LOAD"
    assert get_traffic_scenario("light_load") == scenario


def test_light_load_preserves_population_and_packet_profiles():
    baseline = get_traffic_scenario("BASELINE")
    light = get_traffic_scenario("LIGHT_LOAD")

    for name in ("embb", "urllc", "mmtc"):
        original = asdict(getattr(baseline, name))
        reduced = asdict(getattr(light, name))

        for key in (
            "users_dl",
            "packet_size_bytes",
            "size_distribution",
            "arrival_distribution",
        ):
            assert reduced[key] == original[key]


def test_interarrival_parameters_are_twenty_times_larger():
    baseline = get_traffic_scenario("BASELINE")
    light = get_traffic_scenario("LIGHT_LOAD")

    for name in ("embb", "urllc", "mmtc"):
        original = getattr(baseline, name)
        reduced = getattr(light, name)

        assert reduced.arrival_rate == (
            original.arrival_rate * 20
        )
