import numpy as np


def aggregate_deadline_counts(flows):
    evaluated = sum(
        int(flow.deadlineEvaluated)
        for flow in flows
    )

    misses = sum(
        int(flow.deadlineMisses)
        for flow in flows
    )

    return evaluated, misses


def deadline_miss_ratio(flows):
    evaluated, misses = aggregate_deadline_counts(flows)

    if evaluated == 0:
        return 0.0

    return float(misses) / float(evaluated)


def collect_scheduling_delays(flows):
    return [
        float(delay)
        for flow in flows
        for delay in flow.schedulingDelays
    ]


def delay_statistics(flows):
    delays = np.asarray(
        collect_scheduling_delays(flows),
        dtype=np.float64,
    )

    if delays.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(delays.size),
        "mean": float(np.mean(delays)),
        "p95": float(np.percentile(delays, 95)),
        "p99": float(np.percentile(delays, 99)),
        "p99_9": float(np.percentile(delays, 99.9)),
        "max": float(np.max(delays)),
    }


def snapshot_deadline_counts(flows):
    """Capture cumulative deadline counters before or after an action interval."""
    return aggregate_deadline_counts(flows)


def interval_deadline_metrics(before, after):
    before_evaluated, before_misses = before
    after_evaluated, after_misses = after

    evaluated = max(
        0,
        after_evaluated - before_evaluated,
    )

    misses = max(
        0,
        after_misses - before_misses,
    )

    if evaluated == 0:
        ratio = 0.0
    else:
        ratio = float(misses) / float(evaluated)

    return {
        "evaluated": evaluated,
        "misses": misses,
        "miss_ratio": ratio,
    }
