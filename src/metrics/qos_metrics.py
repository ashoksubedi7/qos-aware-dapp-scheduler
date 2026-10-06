from dataclasses import dataclass
from math import isfinite

import numpy as np

from metrics.service_metrics import (
    prb_utilization,
    throughput_mbps,
)


@dataclass(frozen=True)
class LifecycleRatios:
    generated: int
    delivered: int
    dropped: int
    residual: int

    completion_ratio: float
    true_drop_ratio: float
    residual_ratio: float

    conservation_ok: bool


@dataclass(frozen=True)
class DelaySummary:
    count: int
    mean_ms: float
    p95_ms: float
    p99_ms: float
    p99_9_ms: float
    max_ms: float


@dataclass(frozen=True)
class DeadlineSummary:
    evaluated: int
    misses: int
    miss_ratio: float


def lifecycle_ratios(
    generated,
    delivered,
    dropped,
    residual,
):
    generated = int(generated)
    delivered = int(delivered)
    dropped = int(dropped)
    residual = int(residual)

    values = (
        generated,
        delivered,
        dropped,
        residual,
    )

    if any(value < 0 for value in values):
        raise ValueError(
            "packet lifecycle counts must "
            "be non-negative"
        )

    conservation_ok = (
        generated
        == delivered
        + dropped
        + residual
    )

    if generated == 0:
        completion_ratio = 0.0
        true_drop_ratio = 0.0
        residual_ratio = 0.0
    else:
        completion_ratio = (
            float(delivered)
            / float(generated)
        )

        true_drop_ratio = (
            float(dropped)
            / float(generated)
        )

        residual_ratio = (
            float(residual)
            / float(generated)
        )

    return LifecycleRatios(
        generated=generated,
        delivered=delivered,
        dropped=dropped,
        residual=residual,
        completion_ratio=completion_ratio,
        true_drop_ratio=true_drop_ratio,
        residual_ratio=residual_ratio,
        conservation_ok=conservation_ok,
    )


def delay_summary(
    delays_ms,
):
    values = np.asarray(
        list(delays_ms),
        dtype=np.float64,
    )

    if values.size == 0:
        return DelaySummary(
            count=0,
            mean_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            p99_9_ms=0.0,
            max_ms=0.0,
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "delay values must be finite"
        )

    if np.any(values < 0):
        raise ValueError(
            "delay values must be "
            "non-negative"
        )

    return DelaySummary(
        count=int(values.size),
        mean_ms=float(
            np.mean(values)
        ),
        p95_ms=float(
            np.percentile(
                values,
                95,
            )
        ),
        p99_ms=float(
            np.percentile(
                values,
                99,
            )
        ),
        p99_9_ms=float(
            np.percentile(
                values,
                99.9,
            )
        ),
        max_ms=float(
            np.max(values)
        ),
    )


def deadline_summary(
    evaluated,
    misses,
):
    evaluated = int(evaluated)
    misses = int(misses)

    if evaluated < 0:
        raise ValueError(
            "evaluated must be "
            "non-negative"
        )

    if misses < 0:
        raise ValueError(
            "misses must be "
            "non-negative"
        )

    if misses > evaluated:
        raise ValueError(
            "misses cannot exceed "
            "evaluated packets"
        )

    if evaluated == 0:
        ratio = 0.0
    else:
        ratio = (
            float(misses)
            / float(evaluated)
        )

    return DeadlineSummary(
        evaluated=evaluated,
        misses=misses,
        miss_ratio=ratio,
    )


def active_throughput_mbps(
    delivered_bytes,
    active_duration_ms,
):
    if delivered_bytes < 0:
        raise ValueError(
            "delivered_bytes must be "
            "non-negative"
        )

    return throughput_mbps(
        delivered_bytes,
        active_duration_ms,
    )


def radio_prb_utilization(
    used_prbs,
    available_prbs,
):
    if used_prbs < 0:
        raise ValueError(
            "used_prbs must be "
            "non-negative"
        )

    if available_prbs < 0:
        raise ValueError(
            "available_prbs must be "
            "non-negative"
        )

    if used_prbs > available_prbs:
        raise ValueError(
            "used_prbs cannot exceed "
            "available_prbs"
        )

    return prb_utilization(
        used_prbs,
        available_prbs,
    )


def jains_fairness_index(
    values,
):
    values = [
        float(value)
        for value in values
    ]

    if not values:
        return None

    if any(
        not isfinite(value)
        for value in values
    ):
        raise ValueError(
            "fairness values must "
            "be finite"
        )

    if any(
        value < 0
        for value in values
    ):
        raise ValueError(
            "fairness values must "
            "be non-negative"
        )

    denominator = (
        len(values)
        * sum(
            value * value
            for value in values
        )
    )

    if denominator == 0:
        return None

    numerator = (
        sum(values)
        ** 2
    )

    return (
        numerator
        / denominator
    )
