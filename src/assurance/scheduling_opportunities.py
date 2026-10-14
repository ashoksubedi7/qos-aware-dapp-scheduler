"""Known-packet scheduling opportunities under a measured MAC clock."""

import math
from dataclasses import dataclass

from assurance.known_deadline_demand import KnownDeadlinePacket


@dataclass(frozen=True)
class SchedulingClock:
    first_event_ms: float
    period_ms: float

    def __post_init__(self):
        if not math.isfinite(self.first_event_ms):
            raise ValueError("clock phase must be finite")
        if not math.isfinite(self.period_ms):
            raise ValueError("clock period must be finite")
        if self.first_event_ms < 0.0:
            raise ValueError("clock phase cannot be negative")
        if self.period_ms <= 0.0:
            raise ValueError("clock period must be positive")

    def event_time(self, index):
        if index < 0:
            raise ValueError("event index cannot be negative")
        return self.first_event_ms + index * self.period_ms


@dataclass(frozen=True)
class PacketOpportunity:
    ue_id: str
    packet_id: int
    first_future_event_ms: float
    future_events_before_deadline: int
    on_current_event_boundary: bool
    bearer_visible_now: bool
    status: str


def assess_packet_opportunity(packet, now_ms, clock):
    """
    Examine strictly future scheduling clock events.

    The event at now_ms is excluded because its execution
    order relative to this observation is not established.

    A future event is only a scheduling opportunity.
    It does not guarantee UE selection or TB insertion.
    """
    if not isinstance(packet, KnownDeadlinePacket):
        raise TypeError("expected KnownDeadlinePacket")

    if not isinstance(clock, SchedulingClock):
        raise TypeError("expected SchedulingClock")

    now = float(now_ms)
    deadline = float(packet.absolute_deadline_ms)

    if not math.isfinite(now) or now < 0.0:
        raise ValueError("observation time must be finite and nonnegative")

    if not math.isfinite(deadline):
        raise ValueError("absolute deadline must be finite")

    phase = clock.first_event_ms
    period = clock.period_ms

    # Tolerance is only used to conservatively exclude
    # events indistinguishable from the current instant.
    tolerance = min(1e-10, period * 1e-7)

    relative = (now - phase) / period
    nearest_index = max(0, round(relative))

    on_boundary = (
        abs(clock.event_time(nearest_index) - now)
        <= tolerance
    )

    first_index = max(
        0,
        math.floor(relative) + 1,
    )

    while clock.event_time(first_index) <= now + tolerance:
        first_index += 1

    first_future = clock.event_time(first_index)

    last_index = math.floor(
        (deadline - phase) / period
    )

    # Correct possible floating-point overshoot at
    # the computed final event.
    if (
        last_index >= first_index
        and clock.event_time(last_index) > deadline
    ):
        last_index -= 1

    count = max(
        0,
        last_index - first_index + 1,
    )

    bearer_visible = "bearer" in packet.locations

    # Evaluate expiration at the requested observation time.
    # The packet's stored slack may refer to an older snapshot.
    # Scheduling exactly at the deadline is not late.
    if now > deadline:
        status = "already_overdue"
    elif packet.deadline_already_evaluated:
        status = "previously_evaluated"
    elif not bearer_visible:
        status = "bearer_transfer_unknown"
    elif count > 0:
        status = "future_event_candidate"
    elif on_boundary:
        status = "current_event_order_unknown"
    else:
        status = "no_future_clock_event"

    return PacketOpportunity(
        ue_id=packet.ue_id,
        packet_id=packet.packet_id,
        first_future_event_ms=first_future,
        future_events_before_deadline=count,
        on_current_event_boundary=on_boundary,
        bearer_visible_now=bearer_visible,
        status=status,
    )


def assess_known_deadline_demand(packets, now_ms, clock):
    """Assess a deadline-demand snapshot without modifying it."""
    return tuple(
        assess_packet_opportunity(packet, now_ms, clock)
        for packet in packets
    )
