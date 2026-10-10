from dataclasses import replace

import pytest

from experiments.checkpoint_selection import (
    CHECKPOINT_SELECTION_METRICS,
    score_run_summary,
)
from metrics.run_summary import (
    RunSummary,
    SliceLifecycleSummary,
    SlicePerformanceSummary,
    StarvationSummary,
    SystemSummary,
    URLLCQoSSummary,
)


def lifecycle(
    completion=0.95,
    drop=0.05,
):
    return SliceLifecycleSummary(
        generated=100,
        delivered=int(
            100 * completion
        ),
        dropped=int(
            100 * drop
        ),
        residual=0,
        completion_ratio=completion,
        true_drop_ratio=drop,
        residual_ratio=0.0,
    )


def starvation(
    total_ms=2.0,
    max_ms=1.0,
):
    return StarvationSummary(
        event_count=0,
        total_ms=total_ms,
        max_ms=max_ms,
    )


def performance(
    throughput=10.0,
    fairness=0.95,
    starvation_value=None,
):
    if starvation_value is None:
        starvation_value = starvation()

    return SlicePerformanceSummary(
        active_throughput_mbps=throughput,
        jain_fairness=fairness,
        starvation=starvation_value,
    )


def urllc_qos(
    miss_ratio=0.01,
    evaluated=100,
):
    return URLLCQoSSummary(
        deadline_evaluated=evaluated,
        deadline_misses=int(
            evaluated * miss_ratio
        ),
        scheduling_deadline_miss_ratio=(
            miss_ratio
        ),
        scheduling_delay_count=100,
        scheduling_delay_mean_ms=0.5,
        scheduling_delay_p95_ms=0.8,
        scheduling_delay_p99_ms=0.9,
        scheduling_delay_p99_9_ms=1.0,
        scheduling_delay_max_ms=1.1,
        completion_delay_count=100,
        completion_delay_mean_ms=1.0,
        completion_delay_p95_ms=1.5,
        completion_delay_p99_ms=2.0,
        completion_delay_p99_9_ms=2.5,
        completion_delay_max_ms=3.0,
    )


def make_summary():
    return RunSummary(
        model_variant="M1",
        seed=7,
        traffic_scenario="VALIDATION",
        radio_scenario="STATIC_GOOD",
        control_interval_ms=1.0,
        active_duration_ms=10_000.0,
        drain_duration_ms=1_000.0,
        embb_lifecycle=lifecycle(),
        urllc_lifecycle=lifecycle(
            completion=0.99,
            drop=0.01,
        ),
        mmtc_lifecycle=lifecycle(),
        embb_performance=performance(
            throughput=20.0,
        ),
        mmtc_performance=performance(
            throughput=5.0,
        ),
        urllc_qos=urllc_qos(),
        system=SystemSummary(
            prb_utilization=0.8
        ),
        conservation_ok=True,
    )


def test_metric_order_is_explicit():
    assert CHECKPOINT_SELECTION_METRICS == (
        "min:urllc_scheduling_deadline_miss_ratio",
        "min:urllc_true_drop_ratio",
        "min:worst_non_urllc_starvation_max_ms",
        "min:combined_non_urllc_starvation_total_ms",
        "max:embb_active_throughput_mbps",
        "max:mmtc_completion_ratio",
        "max:minimum_non_urllc_jain_fairness",
        "max:prb_utilization",
    )


def test_lower_deadline_miss_wins_before_efficiency():
    base = make_summary()

    safer = replace(
        base,
        urllc_qos=urllc_qos(
            miss_ratio=0.01
        ),
        system=SystemSummary(
            prb_utilization=0.5
        ),
    )

    efficient_but_less_safe = replace(
        base,
        urllc_qos=urllc_qos(
            miss_ratio=0.02
        ),
        system=SystemSummary(
            prb_utilization=0.99
        ),
    )

    assert (
        score_run_summary(
            safer
        ).lexicographic_key()
        <
        score_run_summary(
            efficient_but_less_safe
        ).lexicographic_key()
    )


def test_starvation_precedes_throughput():
    base = make_summary()

    protected = replace(
        base,
        embb_performance=performance(
            throughput=10.0,
            starvation_value=starvation(
                total_ms=2.0,
                max_ms=1.0,
            ),
        ),
    )

    throughput_but_starving = replace(
        base,
        embb_performance=performance(
            throughput=100.0,
            starvation_value=starvation(
                total_ms=20.0,
                max_ms=10.0,
            ),
        ),
    )

    assert (
        score_run_summary(
            protected
        ).lexicographic_key()
        <
        score_run_summary(
            throughput_but_starving
        ).lexicographic_key()
    )


def test_zero_deadline_denominator_fails_closed():
    summary = replace(
        make_summary(),
        urllc_qos=urllc_qos(
            miss_ratio=0.0,
            evaluated=0,
        ),
    )

    with pytest.raises(
        ValueError
    ):
        score_run_summary(summary)


def test_undefined_fairness_fails_closed():
    summary = make_summary()

    summary = replace(
        summary,
        embb_performance=replace(
            summary.embb_performance,
            jain_fairness=None,
        ),
    )

    with pytest.raises(
        ValueError
    ):
        score_run_summary(summary)


def test_conservation_failure_fails_closed():
    summary = replace(
        make_summary(),
        conservation_ok=False,
    )

    with pytest.raises(
        ValueError
    ):
        score_run_summary(summary)
