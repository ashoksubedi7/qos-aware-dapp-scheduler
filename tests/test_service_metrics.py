import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.service_metrics import (
    packet_service_ratio,
    throughput_mbps,
    normalized_throughput_service,
    prb_utilization,
    starvation_penalty,
)


def test_packet_service_ratio():
    value = packet_service_ratio(
        backlog_before=20,
        generated_packets=10,
        delivered_packets=15,
    )

    assert np.isclose(value, 0.5)


def test_packet_service_ratio_no_demand():
    assert packet_service_ratio(
        0,
        0,
        0,
    ) == 1.0


def test_packet_service_ratio_clips():
    assert packet_service_ratio(
        5,
        0,
        10,
    ) == 1.0


def test_throughput_mbps():
    value = throughput_mbps(
        delivered_bytes=1024 * 1024,
        interval_ms=1000,
    )

    assert np.isclose(value, 8.0)


def test_normalized_throughput_service():
    value = normalized_throughput_service(
        delivered_bytes=1024 * 1024,
        interval_ms=1000,
        target_mbps=16.0,
    )

    assert np.isclose(value, 0.5)


def test_prb_utilization():
    assert np.isclose(
        prb_utilization(
            used_prbs=25,
            available_prbs=100,
        ),
        0.25,
    )


def test_prb_utilization_clips():
    assert prb_utilization(
        used_prbs=120,
        available_prbs=100,
    ) == 1.0


def test_starvation_accumulates():
    duration, penalty = starvation_penalty(
        backlog_before=10,
        delivered_bytes=0,
        previous_starvation_ms=5,
        interval_ms=5,
        threshold_ms=20,
    )

    assert np.isclose(duration, 10.0)
    assert np.isclose(penalty, 0.5)


def test_starvation_resets_after_service():
    duration, penalty = starvation_penalty(
        backlog_before=10,
        delivered_bytes=500,
        previous_starvation_ms=15,
        interval_ms=5,
        threshold_ms=20,
    )

    assert duration == 0.0
    assert penalty == 0.0
