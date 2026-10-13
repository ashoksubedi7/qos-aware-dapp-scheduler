"""Validate canonical QoS metric relationships."""

import math

from metrics.qos_metrics import (
    deadline_summary,
    lifecycle_ratios,
)
from metrics.run_summary import RunSummary


def _require_ratio(actual, expected, name):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{name} inconsistent with underlying counts"
        )


def validate_semantic_run_summary(summary):
    """
    Check metric invariants using the simulator's
    existing definitions.

    Structural validation is performed separately.
    """
    if not isinstance(summary, RunSummary):
        raise TypeError(
            "summary must be a RunSummary"
        )

    for name in ("embb", "urllc", "mmtc"):
        lifecycle = getattr(
            summary,
            f"{name}_lifecycle",
        )

        expected = lifecycle_ratios(
            generated=lifecycle.generated,
            delivered=lifecycle.delivered,
            dropped=lifecycle.dropped,
            residual=lifecycle.residual,
        )

        if not expected.conservation_ok:
            raise ValueError(
                f"{name} packet lifecycle conservation failed"
            )

        for field in (
            "completion_ratio",
            "true_drop_ratio",
            "residual_ratio",
        ):
            _require_ratio(
                getattr(lifecycle, field),
                getattr(expected, field),
                f"{name}.{field}",
            )

    qos = summary.urllc_qos

    expected_deadline = deadline_summary(
        evaluated=qos.deadline_evaluated,
        misses=qos.deadline_misses,
    )

    _require_ratio(
        qos.scheduling_deadline_miss_ratio,
        expected_deadline.miss_ratio,
        "urllc.scheduling_deadline_miss_ratio",
    )

    utilization = summary.system.prb_utilization

    if not 0.0 <= utilization <= 1.0:
        raise ValueError(
            "system PRB utilization outside [0, 1]"
        )

    for name in ("embb", "mmtc"):
        performance = getattr(
            summary,
            f"{name}_performance",
        )

        if performance.active_throughput_mbps < 0:
            raise ValueError(
                f"{name} throughput cannot be negative"
            )

        fairness = performance.jain_fairness

        if fairness is not None and (
            fairness < -1e-12
            or fairness > 1.0 + 1e-12
        ):
            raise ValueError(
                f"{name} fairness outside [0, 1]"
            )

        starvation = performance.starvation

        if starvation is not None:
            if (
                starvation.event_count < 0
                or starvation.total_ms < 0
                or starvation.max_ms < 0
            ):
                raise ValueError(
                    f"{name} starvation metrics "
                    "cannot be negative"
                )

    return summary
