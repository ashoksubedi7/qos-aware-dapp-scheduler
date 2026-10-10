from dataclasses import replace

import pytest

from experiments.checkpoint_aggregation import (
    aggregate_validation_suite,
    validate_validation_grid,
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
    generated=100,
    delivered=95,
    dropped=5,
):
    residual = (
        generated
        - delivered
        - dropped
    )

    return SliceLifecycleSummary(
        generated=generated,
        delivered=delivered,
        dropped=dropped,
        residual=residual,
        completion_ratio=(
            float(delivered)
            / float(generated)
        ),
        true_drop_ratio=(
            float(dropped)
            / float(generated)
        ),
        residual_ratio=(
            float(residual)
            / float(generated)
        ),
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


def qos(
    evaluated=100,
    misses=1,
):
    return URLLCQoSSummary(
        deadline_evaluated=evaluated,
        deadline_misses=misses,
        scheduling_deadline_miss_ratio=(
            float(misses)
            / float(evaluated)
            if evaluated
            else 0.0
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


def summary(
    *,
    seed,
    traffic,
    radio="STATIC_GOOD",
    evaluated=100,
    misses=1,
    urllc_generated=100,
    urllc_dropped=1,
    mmtc_generated=100,
    mmtc_delivered=95,
    embb_throughput=20.0,
    fairness=0.95,
    starvation_total=2.0,
    starvation_max=1.0,
    utilization=0.8,
    active_duration=10_000.0,
):
    return RunSummary(
        model_variant="M1",
        seed=seed,
        traffic_scenario=traffic,
        radio_scenario=radio,
        control_interval_ms=1.0,
        active_duration_ms=active_duration,
        drain_duration_ms=1_000.0,
        embb_lifecycle=lifecycle(),
        urllc_lifecycle=lifecycle(
            generated=urllc_generated,
            delivered=(
                urllc_generated
                - urllc_dropped
            ),
            dropped=urllc_dropped,
        ),
        mmtc_lifecycle=lifecycle(
            generated=mmtc_generated,
            delivered=mmtc_delivered,
            dropped=(
                mmtc_generated
                - mmtc_delivered
            ),
        ),
        embb_performance=performance(
            throughput=embb_throughput,
            fairness=fairness,
            starvation_value=starvation(
                total_ms=starvation_total,
                max_ms=starvation_max,
            ),
        ),
        mmtc_performance=performance(
            throughput=5.0,
            fairness=fairness,
            starvation_value=starvation(
                total_ms=starvation_total,
                max_ms=starvation_max,
            ),
        ),
        urllc_qos=qos(
            evaluated=evaluated,
            misses=misses,
        ),
        system=SystemSummary(
            prb_utilization=utilization
        ),
        conservation_ok=True,
    )


def complete_grid():
    return [
        summary(
            seed=7,
            traffic="A",
        ),
        summary(
            seed=17,
            traffic="A",
        ),
        summary(
            seed=7,
            traffic="B",
        ),
        summary(
            seed=17,
            traffic="B",
        ),
    ]


def test_complete_rectangular_grid_passes():
    metadata = validate_validation_grid(
        complete_grid()
    )

    assert metadata["run_count"] == 4
    assert metadata["seeds"] == (7, 17)

    assert metadata["conditions"] == (
        ("A", "STATIC_GOOD"),
        ("B", "STATIC_GOOD"),
    )


def test_duplicate_condition_seed_fails():
    runs = complete_grid()

    runs.append(
        runs[0]
    )

    with pytest.raises(
        ValueError
    ):
        validate_validation_grid(runs)


def test_incomplete_grid_fails():
    runs = complete_grid()

    runs.pop()

    with pytest.raises(
        ValueError
    ):
        validate_validation_grid(runs)


def test_mixed_active_duration_fails():
    runs = complete_grid()

    runs[-1] = replace(
        runs[-1],
        active_duration_ms=5_000.0,
    )

    with pytest.raises(
        ValueError
    ):
        validate_validation_grid(runs)


def test_deadline_ratio_is_pooled_within_condition():
    runs = [
        summary(
            seed=7,
            traffic="A",
            evaluated=100,
            misses=0,
        ),
        summary(
            seed=17,
            traffic="A",
            evaluated=900,
            misses=90,
        ),
    ]

    result = aggregate_validation_suite(
        runs
    )

    # Pooled ratio = 90 / 1000 = 0.09.
    # Mean of per-seed ratios would be 0.05,
    # which is intentionally not used.
    assert (
        result.urllc_deadline_miss_ratio
        == pytest.approx(0.09)
    )


def test_worst_condition_safety_is_not_averaged_away():
    runs = [
        summary(
            seed=7,
            traffic="A",
            evaluated=100,
            misses=0,
        ),
        summary(
            seed=17,
            traffic="A",
            evaluated=100,
            misses=0,
        ),
        summary(
            seed=7,
            traffic="B",
            evaluated=100,
            misses=20,
        ),
        summary(
            seed=17,
            traffic="B",
            evaluated=100,
            misses=20,
        ),
    ]

    result = aggregate_validation_suite(
        runs
    )

    assert (
        result.urllc_deadline_miss_ratio
        == pytest.approx(0.20)
    )


def test_service_is_macro_averaged_across_conditions():
    runs = [
        summary(
            seed=7,
            traffic="A",
            embb_throughput=10.0,
        ),
        summary(
            seed=17,
            traffic="A",
            embb_throughput=10.0,
        ),
        summary(
            seed=7,
            traffic="B",
            embb_throughput=30.0,
        ),
        summary(
            seed=17,
            traffic="B",
            embb_throughput=30.0,
        ),
    ]

    result = aggregate_validation_suite(
        runs
    )

    assert (
        result.embb_active_throughput_mbps
        == pytest.approx(20.0)
    )


def test_worst_starvation_condition_wins():
    runs = [
        summary(
            seed=7,
            traffic="A",
            starvation_total=1.0,
            starvation_max=1.0,
        ),
        summary(
            seed=17,
            traffic="A",
            starvation_total=1.0,
            starvation_max=1.0,
        ),
        summary(
            seed=7,
            traffic="B",
            starvation_total=10.0,
            starvation_max=8.0,
        ),
        summary(
            seed=17,
            traffic="B",
            starvation_total=12.0,
            starvation_max=9.0,
        ),
    ]

    result = aggregate_validation_suite(
        runs
    )

    # Each run has eMBB + mMTC starvation total,
    # so condition B values are 20 and 24 ms,
    # whose seed mean is 22 ms.
    assert (
        result.combined_starvation_total_ms
        == pytest.approx(22.0)
    )

    assert (
        result.worst_starvation_max_ms
        == pytest.approx(9.0)
    )


def test_input_order_does_not_change_result():
    runs = complete_grid()

    forward = aggregate_validation_suite(
        runs
    )

    reverse = aggregate_validation_suite(
        list(reversed(runs))
    )

    assert forward == reverse
