import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.interval_metrics import (
    SliceSnapshot,
    interval_from_snapshots,
)


def test_interval_deltas():
    before = SliceSnapshot(
        sent_packets=100,
        lost_packets=5,
        received_bytes=10000,
        backlog=30,
    )

    after = SliceSnapshot(
        sent_packets=112,
        lost_packets=7,
        received_bytes=13000,
        backlog=25,
    )

    interval = interval_from_snapshots(
        before,
        after,
    )

    assert interval.generated_packets == 12
    assert interval.lost_packets == 2
    assert interval.received_bytes == 3000
    assert interval.backlog_before == 30
    assert interval.backlog_after == 25


def test_served_backlog_estimate():
    before = SliceSnapshot(
        sent_packets=100,
        lost_packets=0,
        received_bytes=10000,
        backlog=20,
    )

    after = SliceSnapshot(
        sent_packets=105,
        lost_packets=0,
        received_bytes=13000,
        backlog=15,
    )

    interval = interval_from_snapshots(
        before,
        after,
    )

    assert interval.served_backlog == 10
