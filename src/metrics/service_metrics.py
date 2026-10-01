def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def packet_service_ratio(
    backlog_before,
    generated_packets,
    delivered_packets,
):
    """
    Fraction of packet demand served during an interval.

    Demand includes packets already waiting plus packets generated
    during the interval.
    """
    demand = int(backlog_before) + int(generated_packets)

    if demand <= 0:
        return 1.0

    return clamp01(
        float(delivered_packets) / float(demand)
    )


def throughput_mbps(
    delivered_bytes,
    interval_ms,
):
    """Convert delivered bytes during an interval to Mbps."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be greater than zero")

    return (
        float(delivered_bytes)
        * 8.0
        * 1000.0
        / (float(interval_ms) * 1024.0 * 1024.0)
    )


def normalized_throughput_service(
    delivered_bytes,
    interval_ms,
    target_mbps,
):
    """
    Normalize achieved throughput against a configured target.
    """
    if target_mbps <= 0:
        raise ValueError("target_mbps must be greater than zero")

    achieved = throughput_mbps(
        delivered_bytes,
        interval_ms,
    )

    return clamp01(
        achieved / float(target_mbps)
    )


def prb_utilization(
    used_prbs,
    available_prbs,
):
    """Fraction of available PRB opportunities actually consumed."""
    if available_prbs <= 0:
        return 0.0

    return clamp01(
        float(used_prbs) / float(available_prbs)
    )


def starvation_penalty(
    backlog_before,
    delivered_bytes,
    previous_starvation_ms,
    interval_ms,
    threshold_ms,
):
    """
    Update starvation duration and return a normalized penalty.

    Starvation continues only when demand exists but no useful
    application data is delivered during the interval.
    """
    if interval_ms <= 0:
        raise ValueError("interval_ms must be greater than zero")

    if threshold_ms <= 0:
        raise ValueError("threshold_ms must be greater than zero")

    if backlog_before > 0 and delivered_bytes <= 0:
        starvation_ms = (
            float(previous_starvation_ms)
            + float(interval_ms)
        )
    else:
        starvation_ms = 0.0

    penalty = clamp01(
        starvation_ms / float(threshold_ms)
    )

    return starvation_ms, penalty
