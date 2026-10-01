import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.deadline_metrics import (
    aggregate_deadline_counts,
    collect_scheduling_delays,
    deadline_miss_ratio,
    delay_statistics,
)


def make_flow(evaluated, misses, delays):
    return SimpleNamespace(
        deadlineEvaluated=evaluated,
        deadlineMisses=misses,
        schedulingDelays=delays,
    )


def test_deadline_counts():
    flows = [
        make_flow(10, 2, [0.5, 1.5]),
        make_flow(20, 3, [0.8, 2.0]),
    ]

    evaluated, misses = aggregate_deadline_counts(flows)

    assert evaluated == 30
    assert misses == 5


def test_deadline_miss_ratio():
    flows = [
        make_flow(10, 2, []),
        make_flow(10, 3, []),
    ]

    assert np.isclose(
        deadline_miss_ratio(flows),
        0.25,
    )


def test_empty_deadline_ratio():
    flows = [
        make_flow(0, 0, []),
    ]

    assert deadline_miss_ratio(flows) == 0.0


def test_collect_delays():
    flows = [
        make_flow(2, 1, [0.5, 1.5]),
        make_flow(2, 1, [0.8, 2.0]),
    ]

    assert collect_scheduling_delays(flows) == [
        0.5,
        1.5,
        0.8,
        2.0,
    ]


def test_delay_statistics():
    flows = [
        make_flow(
            5,
            2,
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
    ]

    stats = delay_statistics(flows)

    assert stats["count"] == 5
    assert np.isclose(stats["mean"], 3.0)
    assert np.isclose(stats["max"], 5.0)
    assert stats["p95"] > 4.0
    assert stats["p99"] > stats["p95"]


from metrics.deadline_metrics import (
    snapshot_deadline_counts,
    interval_deadline_metrics,
)


def test_snapshot_deadline_counts():
    flows = [
        make_flow(10, 2, []),
        make_flow(20, 3, []),
    ]

    assert snapshot_deadline_counts(flows) == (30, 5)


def test_interval_deadline_metrics():
    before = (100, 20)
    after = (140, 30)

    interval = interval_deadline_metrics(
        before,
        after,
    )

    assert interval["evaluated"] == 40
    assert interval["misses"] == 10
    assert np.isclose(
        interval["miss_ratio"],
        0.25,
    )


def test_interval_deadline_metrics_no_packets():
    before = (100, 20)
    after = (100, 20)

    interval = interval_deadline_metrics(
        before,
        after,
    )

    assert interval["evaluated"] == 0
    assert interval["misses"] == 0
    assert interval["miss_ratio"] == 0.0
