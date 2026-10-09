from types import SimpleNamespace

import simpy

from backlog_calibration_runner import (
    _sample_backlog_process,
    run_backlog_calibration_case,
)


def _dummy_slice(
    packet_sizes,
):
    packets = [
        SimpleNamespace(
            size=float(size)
        )
        for size in packet_sizes
    ]

    ue = SimpleNamespace(
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=packets
                )
            )
        ]
    )

    return (
        SimpleNamespace(
            schedulerDL=SimpleNamespace(
                ues={
                    "ue1": ue
                }
            )
        ),
        packets,
    )


def test_sampler_precedes_controller_same_boundary():
    env = simpy.Environment()

    embb, embb_packets = (
        _dummy_slice(
            [100]
        )
    )

    urllc, _ = _dummy_slice([])
    mmtc, _ = _dummy_slice([])

    inter_slice = SimpleNamespace(
        slices={
            "eMBB": embb,
            "URLLC": urllc,
            "mMTC": mmtc,
        }
    )

    times = []

    samples = {
        "eMBB": [],
        "URLLC": [],
        "mMTC": [],
    }

    def controller():
        while True:
            # If the sampler runs first at the
            # boundary, it sees the old value.
            embb_packets[0].size += 100

            yield env.timeout(
                1.0
            )

    # This is the registration order used by the
    # real backlog-calibration runner.
    env.process(
        _sample_backlog_process(
            env=env,
            inter_slice=inter_slice,
            active_duration_ms=2.0,
            sample_interval_ms=1.0,
            times_ms=times,
            samples=samples,
        )
    )

    env.process(
        controller()
    )

    env.run(
        until=2.0
    )

    assert times == [
        0.0,
        1.0,
    ]

    assert samples["eMBB"] == [
        100.0,
        200.0,
    ]


def test_backlog_calibration_smoke():
    result = (
        run_backlog_calibration_case(
            active_duration_ms=3.0,
            seed=7,
            traffic_scenario="BASELINE",
        )
    )

    assert result.seed == 7

    assert (
        result.traffic_scenario
        == "BASELINE"
    )

    assert result.times_ms == (
        0.0,
        1.0,
        2.0,
    )

    assert result.sample_count == 3

    assert len(
        result.embb_bytes
    ) == 3

    assert len(
        result.urllc_bytes
    ) == 3

    assert len(
        result.mmtc_bytes
    ) == 3

    assert all(
        value >= 0
        for value
        in result.embb_bytes
    )

    assert all(
        value >= 0
        for value
        in result.urllc_bytes
    )

    assert all(
        value >= 0
        for value
        in result.mmtc_bytes
    )
