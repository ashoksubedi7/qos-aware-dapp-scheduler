from collections import defaultdict

from experiments.checkpoint_selection import (
    CheckpointScore,
    score_run_summary,
)


VALIDATION_GRID_POLICY = (
    "complete_rectangular_condition_seed_grid"
)

VALIDATION_AGGREGATION_POLICY = (
    "pooled_count_ratios_within_condition_"
    "worst_qos_macro_service_across_conditions"
)


def _mean(values):
    values = tuple(
        float(value)
        for value in values
    )

    if not values:
        raise ValueError(
            "cannot average an empty sequence"
        )

    return sum(values) / float(len(values))


def _condition_key(summary):
    return (
        str(summary.traffic_scenario),
        str(summary.radio_scenario),
    )


def validate_validation_grid(
    summaries,
):
    """
    Validate that one checkpoint was evaluated on
    a complete traffic x radio condition / seed grid.
    """

    summaries = tuple(summaries)

    if not summaries:
        raise ValueError(
            "validation suite cannot be empty"
        )

    # Also validates conservation, deadline
    # evaluability, starvation summaries, and
    # fairness availability.
    for summary in summaries:
        score_run_summary(summary)

    model_variants = {
        summary.model_variant
        for summary in summaries
    }

    if len(model_variants) != 1:
        raise ValueError(
            "validation summaries must belong "
            "to one model variant"
        )

    control_intervals = {
        float(summary.control_interval_ms)
        for summary in summaries
    }

    active_durations = {
        float(summary.active_duration_ms)
        for summary in summaries
    }

    drain_durations = {
        float(summary.drain_duration_ms)
        for summary in summaries
    }

    if len(control_intervals) != 1:
        raise ValueError(
            "validation runs must use one "
            "control interval"
        )

    if len(active_durations) != 1:
        raise ValueError(
            "validation runs must use one "
            "active duration"
        )

    if len(drain_durations) != 1:
        raise ValueError(
            "validation runs must use one "
            "drain duration"
        )

    seeds = {
        int(summary.seed)
        for summary in summaries
    }

    conditions = {
        _condition_key(summary)
        for summary in summaries
    }

    observed = set()

    for summary in summaries:
        key = (
            _condition_key(summary),
            int(summary.seed),
        )

        if key in observed:
            raise ValueError(
                "duplicate validation "
                "condition/seed result"
            )

        observed.add(key)

    expected = {
        (
            condition,
            seed,
        )
        for condition in conditions
        for seed in seeds
    }

    if observed != expected:
        missing = sorted(
            expected - observed
        )

        extra = sorted(
            observed - expected
        )

        raise ValueError(
            "incomplete validation grid; "
            f"missing={missing}, "
            f"extra={extra}"
        )

    return {
        "run_count": len(summaries),
        "conditions": tuple(
            sorted(conditions)
        ),
        "seeds": tuple(
            sorted(seeds)
        ),
        "model_variant": next(
            iter(model_variants)
        ),
        "control_interval_ms": next(
            iter(control_intervals)
        ),
        "active_duration_ms": next(
            iter(active_durations)
        ),
        "drain_duration_ms": next(
            iter(drain_durations)
        ),
    }


def aggregate_validation_suite(
    summaries,
):
    """
    Aggregate one checkpoint's complete validation
    suite into a QoS-first CheckpointScore.

    Within a traffic/radio condition:
      * count-based QoS ratios are pooled across seeds;
      * starvation maxima are worst-seed;
      * continuous service metrics are seed means.

    Across conditions:
      * protected QoS/starvation metrics use the
        worst condition;
      * service/efficiency metrics use an equal-weight
        macro-average across conditions.
    """

    summaries = tuple(summaries)

    validate_validation_grid(
        summaries
    )

    grouped = defaultdict(list)

    for summary in summaries:
        grouped[
            _condition_key(summary)
        ].append(summary)

    condition_scores = []

    for condition in sorted(grouped):
        runs = grouped[condition]

        deadline_evaluated = sum(
            int(
                run
                .urllc_qos
                .deadline_evaluated
            )
            for run in runs
        )

        deadline_misses = sum(
            int(
                run
                .urllc_qos
                .deadline_misses
            )
            for run in runs
        )

        if deadline_evaluated <= 0:
            raise ValueError(
                "validation condition has no "
                "evaluated URLLC deadlines"
            )

        urllc_generated = sum(
            int(
                run
                .urllc_lifecycle
                .generated
            )
            for run in runs
        )

        urllc_dropped = sum(
            int(
                run
                .urllc_lifecycle
                .dropped
            )
            for run in runs
        )

        if urllc_generated <= 0:
            raise ValueError(
                "validation condition has no "
                "generated URLLC packets"
            )

        mmtc_generated = sum(
            int(
                run
                .mmtc_lifecycle
                .generated
            )
            for run in runs
        )

        mmtc_delivered = sum(
            int(
                run
                .mmtc_lifecycle
                .delivered
            )
            for run in runs
        )

        if mmtc_generated <= 0:
            raise ValueError(
                "validation condition has no "
                "generated mMTC packets"
            )

        starvation_max_values = []

        starvation_total_values = []

        fairness_values = []

        for run in runs:
            embb_starvation = (
                run
                .embb_performance
                .starvation
            )

            mmtc_starvation = (
                run
                .mmtc_performance
                .starvation
            )

            starvation_max_values.append(
                max(
                    float(
                        embb_starvation.max_ms
                    ),
                    float(
                        mmtc_starvation.max_ms
                    ),
                )
            )

            starvation_total_values.append(
                float(
                    embb_starvation.total_ms
                )
                +
                float(
                    mmtc_starvation.total_ms
                )
            )

            fairness_values.append(
                min(
                    float(
                        run
                        .embb_performance
                        .jain_fairness
                    ),
                    float(
                        run
                        .mmtc_performance
                        .jain_fairness
                    ),
                )
            )

        condition_scores.append(
            CheckpointScore(
                urllc_deadline_miss_ratio=(
                    float(deadline_misses)
                    /
                    float(deadline_evaluated)
                ),
                urllc_true_drop_ratio=(
                    float(urllc_dropped)
                    /
                    float(urllc_generated)
                ),
                worst_starvation_max_ms=max(
                    starvation_max_values
                ),
                combined_starvation_total_ms=(
                    _mean(
                        starvation_total_values
                    )
                ),
                embb_active_throughput_mbps=(
                    _mean(
                        run
                        .embb_performance
                        .active_throughput_mbps
                        for run in runs
                    )
                ),
                mmtc_completion_ratio=(
                    float(mmtc_delivered)
                    /
                    float(mmtc_generated)
                ),
                minimum_non_urllc_fairness=(
                    _mean(fairness_values)
                ),
                prb_utilization=(
                    _mean(
                        run.system.prb_utilization
                        for run in runs
                    )
                ),
            )
        )

    return CheckpointScore(
        urllc_deadline_miss_ratio=max(
            score.urllc_deadline_miss_ratio
            for score in condition_scores
        ),
        urllc_true_drop_ratio=max(
            score.urllc_true_drop_ratio
            for score in condition_scores
        ),
        worst_starvation_max_ms=max(
            score.worst_starvation_max_ms
            for score in condition_scores
        ),
        combined_starvation_total_ms=max(
            score.combined_starvation_total_ms
            for score in condition_scores
        ),
        embb_active_throughput_mbps=_mean(
            score.embb_active_throughput_mbps
            for score in condition_scores
        ),
        mmtc_completion_ratio=_mean(
            score.mmtc_completion_ratio
            for score in condition_scores
        ),
        minimum_non_urllc_fairness=_mean(
            score.minimum_non_urllc_fairness
            for score in condition_scores
        ),
        prb_utilization=_mean(
            score.prb_utilization
            for score in condition_scores
        ),
    )
