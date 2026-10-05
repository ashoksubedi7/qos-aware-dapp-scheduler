import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.transition_metrics import (
    SliceCounters,
    RadioCounters,
    counter_delta,
    derive_transition_metrics,
)


def test_counter_delta():
    assert counter_delta(10, 15) == 5
    assert counter_delta(15, 10) == 0


def test_transition_metrics():
    embb_before = SliceCounters(
        generated_packets=100,
        delivered_packets=80,
        delivered_bytes=0,
        backlog=10,
    )

    embb_after = SliceCounters(
        generated_packets=110,
        delivered_packets=90,
        delivered_bytes=1_000_000,
        backlog=8,
    )

    urllc_before = SliceCounters(
        generated_packets=100,
        delivered_packets=50,
        delivered_bytes=0,
        backlog=10,
    )

    urllc_after = SliceCounters(
        generated_packets=110,
        delivered_packets=65,
        delivered_bytes=1000,
        backlog=5,
    )

    mmtc_before = SliceCounters(
        generated_packets=100,
        delivered_packets=80,
        delivered_bytes=0,
        backlog=10,
    )

    mmtc_after = SliceCounters(
        generated_packets=110,
        delivered_packets=90,
        delivered_bytes=500,
        backlog=10,
    )

    radio_before = RadioCounters(
        used_prbs=100,
        available_prbs=200,
    )

    radio_after = RadioCounters(
        used_prbs=150,
        available_prbs=300,
    )

    metrics = derive_transition_metrics(
        embb_before,
        embb_after,
        urllc_before,
        urllc_after,
        mmtc_before,
        mmtc_after,
        radio_before,
        radio_after,
        deadline_before=(100, 10),
        deadline_after=(120, 14),
        interval_ms=1000,
        embb_target_mbps=16.0,
    )

    assert np.isclose(metrics.embb_service, 0.5)
    assert np.isclose(metrics.urllc_service, 0.75)
    assert np.isclose(metrics.mmtc_service, 0.5)
    assert np.isclose(metrics.utilization, 0.5)
    assert np.isclose(metrics.deadline_miss_ratio, 0.2)
