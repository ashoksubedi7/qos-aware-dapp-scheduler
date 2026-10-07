from dataclasses import dataclass


@dataclass(frozen=True)
class SliceTrafficProfile:
    users_dl: int
    packet_size_bytes: int
    arrival_rate: float
    size_distribution: str
    arrival_distribution: str


@dataclass(frozen=True)
class TrafficScenario:
    name: str
    embb: SliceTrafficProfile
    urllc: SliceTrafficProfile
    mmtc: SliceTrafficProfile


BASELINE = TrafficScenario(
    name="BASELINE",
    embb=SliceTrafficProfile(
        users_dl=4,
        packet_size_bytes=1200,
        arrival_rate=1.2,
        size_distribution="Pareto",
        arrival_distribution="Uniform",
    ),
    urllc=SliceTrafficProfile(
        users_dl=8,
        packet_size_bytes=300,
        arrival_rate=0.6,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
    mmtc=SliceTrafficProfile(
        users_dl=30,
        packet_size_bytes=100,
        arrival_rate=10.0,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
)


CONGESTED = TrafficScenario(
    name="CONGESTED",
    embb=SliceTrafficProfile(
        users_dl=8,
        packet_size_bytes=1200,
        arrival_rate=0.8,
        size_distribution="Pareto",
        arrival_distribution="Uniform",
    ),
    urllc=SliceTrafficProfile(
        users_dl=16,
        packet_size_bytes=300,
        arrival_rate=0.4,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
    mmtc=SliceTrafficProfile(
        users_dl=60,
        packet_size_bytes=100,
        arrival_rate=5.0,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
)


URLLC_HIGH = TrafficScenario(
    name="URLLC_HIGH",
    embb=SliceTrafficProfile(
        users_dl=4,
        packet_size_bytes=1200,
        arrival_rate=1.2,
        size_distribution="Pareto",
        arrival_distribution="Uniform",
    ),
    urllc=SliceTrafficProfile(
        users_dl=24,
        packet_size_bytes=300,
        arrival_rate=0.2,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
    mmtc=SliceTrafficProfile(
        users_dl=30,
        packet_size_bytes=100,
        arrival_rate=10.0,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
)


SIMULTANEOUS_HIGH = TrafficScenario(
    name="SIMULTANEOUS_HIGH",
    embb=SliceTrafficProfile(
        users_dl=10,
        packet_size_bytes=1200,
        arrival_rate=0.6,
        size_distribution="Pareto",
        arrival_distribution="Uniform",
    ),
    urllc=SliceTrafficProfile(
        users_dl=20,
        packet_size_bytes=300,
        arrival_rate=0.3,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
    mmtc=SliceTrafficProfile(
        users_dl=80,
        packet_size_bytes=100,
        arrival_rate=4.0,
        size_distribution="Constant",
        arrival_distribution="Uniform",
    ),
)


_SCENARIOS = {
    BASELINE.name: BASELINE,
    CONGESTED.name: CONGESTED,
    URLLC_HIGH.name: URLLC_HIGH,
    SIMULTANEOUS_HIGH.name: SIMULTANEOUS_HIGH,
}


def get_traffic_scenario(
    name,
):
    key = str(
        name
    ).upper()

    if key not in _SCENARIOS:
        raise ValueError(
            "unknown traffic scenario: "
            f"{name}"
        )

    return _SCENARIOS[
        key
    ]
