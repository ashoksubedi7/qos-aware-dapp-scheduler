import pytest

from assurance.known_deadline_demand import KnownDeadlinePacket
from assurance.scheduling_opportunities import (
    SchedulingClock,
    assess_packet_opportunity,
    assess_known_deadline_demand,
)


URLLC_CLOCK = SchedulingClock(
    first_event_ms=0.0,
    period_ms=0.25,
)


def waiting_packet(
    arrival=0.0,
    deadline=1.0,
    locations=("bearer",),
    evaluated=False,
):
    return KnownDeadlinePacket(
        ue_id="ue1",
        packet_id=1,
        arrival_ms=arrival,
        deadline_ms=deadline,
        absolute_deadline_ms=arrival + deadline,
        slack_ms=deadline,
        deadline_already_evaluated=evaluated,
        locations=locations,
    )


def test_urllc_has_four_future_events_in_one_ms():
    result = assess_packet_opportunity(
        waiting_packet(),
        0.0,
        URLLC_CLOCK,
    )

    assert result.first_future_event_ms == 0.25
    assert result.future_events_before_deadline == 4
    assert result.on_current_event_boundary is True
    assert result.status == "future_event_candidate"


def test_one_ms_tti_has_one_future_event():
    result = assess_packet_opportunity(
        waiting_packet(),
        0.0,
        SchedulingClock(0.0, 1.0),
    )

    assert result.first_future_event_ms == 1.0
    assert result.future_events_before_deadline == 1


def test_event_at_exact_deadline_is_included():
    result = assess_packet_opportunity(
        waiting_packet(arrival=0.75, deadline=0.25),
        0.75,
        URLLC_CLOCK,
    )

    assert result.first_future_event_ms == 1.0
    assert result.future_events_before_deadline == 1


def test_current_event_is_excluded():
    result = assess_packet_opportunity(
        waiting_packet(arrival=0.0, deadline=0.75),
        0.75,
        URLLC_CLOCK,
    )

    assert result.first_future_event_ms == 1.0
    assert result.future_events_before_deadline == 0
    assert result.on_current_event_boundary


def test_no_future_event_before_deadline():
    result = assess_packet_opportunity(
        waiting_packet(arrival=0.10, deadline=0.10),
        0.10,
        URLLC_CLOCK,
    )

    assert result.future_events_before_deadline == 0
    assert result.status == "no_future_clock_event"


def test_same_time_boundary_is_not_assumed_available():
    result = assess_packet_opportunity(
        waiting_packet(arrival=0.0, deadline=0.25),
        0.25,
        URLLC_CLOCK,
    )

    assert result.future_events_before_deadline == 0
    assert result.status == "current_event_order_unknown"


def test_application_only_packet_is_not_assumed_mac_ready():
    result = assess_packet_opportunity(
        waiting_packet(locations=("application",)),
        0.0,
        URLLC_CLOCK,
    )

    assert result.future_events_before_deadline == 4
    assert result.bearer_visible_now is False
    assert result.status == "bearer_transfer_unknown"


def test_already_overdue_packet_cannot_be_rescued():
    packet = waiting_packet(
        arrival=0.0,
        deadline=0.5,
    )

    result = assess_packet_opportunity(
        packet,
        0.75,
        URLLC_CLOCK,
    )

    assert result.status == "already_overdue"


def test_previous_evaluation_is_flagged():
    result = assess_packet_opportunity(
        waiting_packet(evaluated=True),
        0.0,
        URLLC_CLOCK,
    )

    assert result.status == "previously_evaluated"


def test_nonzero_clock_phase():
    clock = SchedulingClock(0.125, 0.25)

    result = assess_packet_opportunity(
        waiting_packet(deadline=0.875),
        0.0,
        clock,
    )

    assert result.first_future_event_ms == 0.125
    assert result.future_events_before_deadline == 4


def test_batch_preserves_packet_count():
    packets = (
        waiting_packet(),
        waiting_packet(arrival=0.1),
    )

    result = assess_known_deadline_demand(
        packets,
        0.25,
        URLLC_CLOCK,
    )

    assert len(result) == len(packets)


@pytest.mark.parametrize(
    "phase, period",
    [
        (-1.0, 0.25),
        (0.0, 0.0),
        (0.0, -0.25),
        (float("nan"), 0.25),
        (0.0, float("inf")),
    ],
)
def test_invalid_clocks_are_rejected(phase, period):
    with pytest.raises(ValueError):
        SchedulingClock(phase, period)


def test_invalid_observation_time_is_rejected():
    with pytest.raises(ValueError):
        assess_packet_opportunity(
            waiting_packet(),
            float("nan"),
            URLLC_CLOCK,
        )
