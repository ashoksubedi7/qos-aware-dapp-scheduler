import pytest

from metrics.starvation_metrics import (
    StarvationTracker,
)


def test_starvation_accumulates_without_service():
    tracker = StarvationTracker(
        threshold_ms=10.0
    )

    tracker.update(
        backlog_before=5,
        scheduled_prbs=0,
        interval_ms=2.0,
    )

    tracker.update(
        backlog_before=5,
        scheduled_prbs=0,
        interval_ms=2.0,
    )

    assert tracker.current_ms == pytest.approx(
        4.0
    )

    assert tracker.total_ms == pytest.approx(
        4.0
    )

    assert tracker.event_count == 0


def test_starvation_event_counted_once_when_threshold_crossed():
    tracker = StarvationTracker(
        threshold_ms=10.0
    )

    for _ in range(10):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    assert tracker.current_ms == pytest.approx(
        10.0
    )

    assert tracker.event_count == 1


def test_long_starvation_is_not_counted_multiple_times():
    tracker = StarvationTracker(
        threshold_ms=10.0
    )

    for _ in range(25):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    assert tracker.event_count == 1

    assert tracker.current_ms == pytest.approx(
        25.0
    )

    assert tracker.max_ms == pytest.approx(
        25.0
    )

    assert tracker.total_ms == pytest.approx(
        25.0
    )


def test_service_ends_starvation_event():
    tracker = StarvationTracker(
        threshold_ms=5.0
    )

    for _ in range(5):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    assert tracker.event_count == 1

    tracker.update(
        backlog_before=5,
        scheduled_prbs=4,
        interval_ms=1.0,
    )

    assert tracker.current_ms == pytest.approx(
        0.0
    )


def test_second_starvation_period_creates_second_event():
    tracker = StarvationTracker(
        threshold_ms=3.0
    )

    for _ in range(3):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    assert tracker.event_count == 1

    tracker.update(
        backlog_before=5,
        scheduled_prbs=4,
        interval_ms=1.0,
    )

    for _ in range(3):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    assert tracker.event_count == 2


def test_backlog_zero_is_not_starvation():
    tracker = StarvationTracker(
        threshold_ms=5.0
    )

    starving = tracker.update(
        backlog_before=0,
        scheduled_prbs=0,
        interval_ms=1.0,
    )

    assert not starving
    assert tracker.current_ms == 0.0
    assert tracker.total_ms == 0.0
    assert tracker.event_count == 0


def test_prb_service_prevents_starvation():
    tracker = StarvationTracker(
        threshold_ms=5.0
    )

    starving = tracker.update(
        backlog_before=10,
        scheduled_prbs=4,
        interval_ms=1.0,
    )

    assert not starving
    assert tracker.current_ms == 0.0


def test_starvation_tracks_maximum_event_duration():
    tracker = StarvationTracker(
        threshold_ms=2.0
    )

    for _ in range(3):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    tracker.update(
        backlog_before=5,
        scheduled_prbs=4,
        interval_ms=1.0,
    )

    for _ in range(7):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=1.0,
        )

    assert tracker.max_ms == pytest.approx(
        7.0
    )

    assert tracker.total_ms == pytest.approx(
        10.0
    )

    assert tracker.event_count == 2


def test_starvation_rejects_zero_threshold():
    with pytest.raises(
        ValueError,
        match="threshold_ms",
    ):
        StarvationTracker(
            threshold_ms=0.0
        )


def test_starvation_rejects_nonpositive_interval():
    tracker = StarvationTracker(
        threshold_ms=5.0
    )

    with pytest.raises(
        ValueError,
        match="interval_ms",
    ):
        tracker.update(
            backlog_before=5,
            scheduled_prbs=0,
            interval_ms=0.0,
        )
