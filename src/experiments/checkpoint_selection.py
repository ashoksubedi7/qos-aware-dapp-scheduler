from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointScore:
    urllc_deadline_miss_ratio: float
    urllc_true_drop_ratio: float

    worst_starvation_max_ms: float
    combined_starvation_total_ms: float

    embb_active_throughput_mbps: float
    mmtc_completion_ratio: float

    minimum_non_urllc_fairness: float

    prb_utilization: float

    def lexicographic_key(self):
        """
        Lower tuple is always better.

        Safety/QoS metrics are minimized directly.
        Service/efficiency metrics are negated so
        maximizing them becomes minimization.
        """
        return (
            self.urllc_deadline_miss_ratio,
            self.urllc_true_drop_ratio,
            self.worst_starvation_max_ms,
            self.combined_starvation_total_ms,
            -self.embb_active_throughput_mbps,
            -self.mmtc_completion_ratio,
            -self.minimum_non_urllc_fairness,
            -self.prb_utilization,
        )


def score_run_summary(summary):
    """
    Convert one canonical RunSummary into the
    QoS-first checkpoint-selection vector.

    Fail closed when a protected metric cannot be
    meaningfully evaluated.
    """

    if not summary.conservation_ok:
        raise ValueError(
            "checkpoint validation requires "
            "packet conservation"
        )

    if (
        summary.urllc_qos.deadline_evaluated
        <= 0
    ):
        raise ValueError(
            "checkpoint validation requires at "
            "least one evaluated URLLC deadline"
        )

    if (
        summary.urllc_qos.scheduling_delay_count
        <= 0
    ):
        raise ValueError(
            "checkpoint validation requires "
            "URLLC scheduling-delay observations"
        )

    embb_starvation = (
        summary.embb_performance.starvation
    )
    mmtc_starvation = (
        summary.mmtc_performance.starvation
    )

    if (
        embb_starvation is None
        or mmtc_starvation is None
    ):
        raise ValueError(
            "checkpoint validation requires "
            "starvation summaries"
        )

    embb_fairness = (
        summary.embb_performance.jain_fairness
    )
    mmtc_fairness = (
        summary.mmtc_performance.jain_fairness
    )

    if (
        embb_fairness is None
        or mmtc_fairness is None
    ):
        raise ValueError(
            "checkpoint validation requires "
            "defined non-URLLC fairness"
        )

    return CheckpointScore(
        urllc_deadline_miss_ratio=float(
            summary
            .urllc_qos
            .scheduling_deadline_miss_ratio
        ),
        urllc_true_drop_ratio=float(
            summary
            .urllc_lifecycle
            .true_drop_ratio
        ),
        worst_starvation_max_ms=max(
            float(
                embb_starvation.max_ms
            ),
            float(
                mmtc_starvation.max_ms
            ),
        ),
        combined_starvation_total_ms=(
            float(
                embb_starvation.total_ms
            )
            +
            float(
                mmtc_starvation.total_ms
            )
        ),
        embb_active_throughput_mbps=float(
            summary
            .embb_performance
            .active_throughput_mbps
        ),
        mmtc_completion_ratio=float(
            summary
            .mmtc_lifecycle
            .completion_ratio
        ),
        minimum_non_urllc_fairness=min(
            float(embb_fairness),
            float(mmtc_fairness),
        ),
        prb_utilization=float(
            summary.system.prb_utilization
        ),
    )


CHECKPOINT_SELECTION_METRICS = (
    "min:urllc_scheduling_deadline_miss_ratio",
    "min:urllc_true_drop_ratio",
    "min:worst_non_urllc_starvation_max_ms",
    "min:combined_non_urllc_starvation_total_ms",
    "max:embb_active_throughput_mbps",
    "max:mmtc_completion_ratio",
    "max:minimum_non_urllc_jain_fairness",
    "max:prb_utilization",
)
