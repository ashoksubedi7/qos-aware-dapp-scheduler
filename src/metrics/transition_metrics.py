from dataclasses import dataclass

from metrics.deadline_metrics import interval_deadline_metrics
from metrics.service_metrics import (
    normalized_throughput_service,
    packet_service_ratio,
    prb_utilization,
)


@dataclass
class SliceCounters:
    generated_packets: int
    delivered_packets: int
    delivered_bytes: int
    backlog: int
    scheduled_prbs: int = 0


@dataclass
class RadioCounters:
    used_prbs: int
    available_prbs: int


@dataclass
class TransitionMetrics:
    urllc_service: float
    embb_service: float
    mmtc_service: float
    utilization: float
    deadline_miss_ratio: float


def counter_delta(before, after):
    return max(0, int(after) - int(before))


def derive_transition_metrics(
    embb_before,
    embb_after,
    urllc_before,
    urllc_after,
    mmtc_before,
    mmtc_after,
    radio_before,
    radio_after,
    deadline_before,
    deadline_after,
    interval_ms,
    embb_target_mbps,
):
    embb_delivered_bytes = counter_delta(
        embb_before.delivered_bytes,
        embb_after.delivered_bytes,
    )

    urllc_generated = counter_delta(
        urllc_before.generated_packets,
        urllc_after.generated_packets,
    )
    urllc_delivered = counter_delta(
        urllc_before.delivered_packets,
        urllc_after.delivered_packets,
    )

    mmtc_generated = counter_delta(
        mmtc_before.generated_packets,
        mmtc_after.generated_packets,
    )
    mmtc_delivered = counter_delta(
        mmtc_before.delivered_packets,
        mmtc_after.delivered_packets,
    )

    used_prbs = counter_delta(
        radio_before.used_prbs,
        radio_after.used_prbs,
    )
    available_prbs = counter_delta(
        radio_before.available_prbs,
        radio_after.available_prbs,
    )

    deadline = interval_deadline_metrics(
        deadline_before,
        deadline_after,
    )

    embb_service = normalized_throughput_service(
        embb_delivered_bytes,
        interval_ms,
        embb_target_mbps,
    )

    urllc_service = packet_service_ratio(
        urllc_before.backlog,
        urllc_generated,
        urllc_delivered,
    )

    mmtc_service = packet_service_ratio(
        mmtc_before.backlog,
        mmtc_generated,
        mmtc_delivered,
    )

    utilization = prb_utilization(
        used_prbs,
        available_prbs,
    )

    return TransitionMetrics(
        urllc_service=urllc_service,
        embb_service=embb_service,
        mmtc_service=mmtc_service,
        utilization=utilization,
        deadline_miss_ratio=deadline["miss_ratio"],
    )
