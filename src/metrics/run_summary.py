from dataclasses import (
    asdict,
    dataclass,
)

from metrics.deadline_metrics import (
    aggregate_deadline_counts,
    collect_completion_delays,
    collect_scheduling_delays,
)

from assured_interval import (
    snapshot_radio_counters,
)

from metrics.packet_accounting import (
    collect_slice_packet_accounting,
)

from metrics.qos_metrics import (
    active_throughput_mbps,
    deadline_summary,
    delay_summary,
    jains_fairness_index,
    lifecycle_ratios,
    radio_prb_utilization,
)


@dataclass(frozen=True)
class StarvationSummary:
    event_count: int
    total_ms: float
    max_ms: float


@dataclass(frozen=True)
class SliceLifecycleSummary:
    generated: int
    delivered: int
    dropped: int
    residual: int

    completion_ratio: float
    true_drop_ratio: float
    residual_ratio: float


@dataclass(frozen=True)
class URLLCQoSSummary:
    deadline_evaluated: int
    deadline_misses: int
    scheduling_deadline_miss_ratio: float

    scheduling_delay_count: int
    scheduling_delay_mean_ms: float
    scheduling_delay_p95_ms: float
    scheduling_delay_p99_ms: float
    scheduling_delay_p99_9_ms: float
    scheduling_delay_max_ms: float

    completion_delay_count: int
    completion_delay_mean_ms: float
    completion_delay_p95_ms: float
    completion_delay_p99_ms: float
    completion_delay_p99_9_ms: float
    completion_delay_max_ms: float


@dataclass(frozen=True)
class SlicePerformanceSummary:
    active_throughput_mbps: float
    jain_fairness: float | None
    starvation: StarvationSummary | None


@dataclass(frozen=True)
class SystemSummary:
    prb_utilization: float


@dataclass(frozen=True)
class RunSummary:
    model_variant: str
    seed: int
    traffic_scenario: str
    radio_scenario: str

    control_interval_ms: float
    active_duration_ms: float
    drain_duration_ms: float

    embb_lifecycle: SliceLifecycleSummary
    urllc_lifecycle: SliceLifecycleSummary
    mmtc_lifecycle: SliceLifecycleSummary

    embb_performance: SlicePerformanceSummary
    mmtc_performance: SlicePerformanceSummary

    urllc_qos: URLLCQoSSummary

    system: SystemSummary

    conservation_ok: bool

    def to_dict(self):
        return asdict(self)

def make_lifecycle_summary(
    accounting,
):
    ratios = lifecycle_ratios(
        generated=accounting.generated,
        delivered=accounting.delivered,
        dropped=accounting.dropped,
        residual=accounting.residual,
    )

    if not ratios.conservation_ok:
        raise RuntimeError(
            "cannot create run summary "
            "from non-conserving packet state"
        )

    return SliceLifecycleSummary(
        generated=ratios.generated,
        delivered=ratios.delivered,
        dropped=ratios.dropped,
        residual=ratios.residual,
        completion_ratio=(
            ratios.completion_ratio
        ),
        true_drop_ratio=(
            ratios.true_drop_ratio
        ),
        residual_ratio=(
            ratios.residual_ratio
        ),
    )

def make_starvation_summary(
    tracker,
):
    return StarvationSummary(
        event_count=int(
            tracker.event_count
        ),
        total_ms=float(
            tracker.total_ms
        ),
        max_ms=float(
            tracker.max_ms
        ),
    )

def _slice_flows(
    slice_obj,
):
    """
    Return the primary downlink data flow for every UE
    that has at least one packet flow.
    """

    return [
        ue.packetFlows[0]
        for ue in (
            slice_obj
            .schedulerDL
            .ues
            .values()
        )
        if ue.packetFlows
    ]


def _active_slice_bytes(
    slice_obj,
):
    """
    Return data bytes successfully delivered during the
    active measurement window.
    """

    return sum(
        int(
            ue.packetFlows[
                0
            ].activeDeliveredBytes
        )
        for ue in (
            slice_obj
            .schedulerDL
            .ues
            .values()
        )
        if ue.packetFlows
    )


def _per_ue_active_throughputs(
    slice_obj,
    active_duration_ms,
):
    """
    Return one SI-Mbps active-window throughput value
    per UE for within-slice fairness calculation.
    """

    values = []

    for ue in (
        slice_obj
        .schedulerDL
        .ues
        .values()
    ):
        if not ue.packetFlows:
            continue

        flow = ue.packetFlows[0]

        values.append(
            active_throughput_mbps(
                delivered_bytes=(
                    flow.activeDeliveredBytes
                ),
                active_duration_ms=(
                    active_duration_ms
                ),
            )
        )

    return values


def make_urllc_qos_summary(
    slice_obj,
):
    """
    Build scheduler-level URLLC deadline and delay
    statistics.

    Scheduling delay and completion delay remain
    intentionally distinct.
    """

    flows = _slice_flows(
        slice_obj
    )

    evaluated, misses = (
        aggregate_deadline_counts(
            flows
        )
    )

    deadline = deadline_summary(
        evaluated=evaluated,
        misses=misses,
    )

    scheduling = delay_summary(
        collect_scheduling_delays(
            flows
        )
    )

    completion = delay_summary(
        collect_completion_delays(
            flows
        )
    )

    return URLLCQoSSummary(
        deadline_evaluated=(
            deadline.evaluated
        ),
        deadline_misses=(
            deadline.misses
        ),
        scheduling_deadline_miss_ratio=(
            deadline.miss_ratio
        ),

        scheduling_delay_count=(
            scheduling.count
        ),
        scheduling_delay_mean_ms=(
            scheduling.mean_ms
        ),
        scheduling_delay_p95_ms=(
            scheduling.p95_ms
        ),
        scheduling_delay_p99_ms=(
            scheduling.p99_ms
        ),
        scheduling_delay_p99_9_ms=(
            scheduling.p99_9_ms
        ),
        scheduling_delay_max_ms=(
            scheduling.max_ms
        ),

        completion_delay_count=(
            completion.count
        ),
        completion_delay_mean_ms=(
            completion.mean_ms
        ),
        completion_delay_p95_ms=(
            completion.p95_ms
        ),
        completion_delay_p99_ms=(
            completion.p99_ms
        ),
        completion_delay_p99_9_ms=(
            completion.p99_9_ms
        ),
        completion_delay_max_ms=(
            completion.max_ms
        ),
    )
def build_run_summary(
    assured_scheduler,
    t_sim_ms,
):
    """
    Build the canonical end-of-run AssuredQoS summary.

    Downstream experiment export, drain calibration,
    statistical analysis, plots, and paper tables should
    consume this object instead of recomputing KPIs.
    """

    config = assured_scheduler.config

    t_sim_ms = float(
        t_sim_ms
    )

    if t_sim_ms <= 0:
        raise ValueError(
            "t_sim_ms must be "
            "greater than zero"
        )

    drain_duration_ms = float(
        config.drain_duration_ms
    )

    if drain_duration_ms < 0:
        raise ValueError(
            "drain duration cannot "
            "be negative"
        )

    active_duration_ms = (
        t_sim_ms
        - drain_duration_ms
    )

    if active_duration_ms <= 0:
        raise ValueError(
            "active duration must be "
            "greater than zero"
        )

    slices = (
        assured_scheduler
        ._canonical_slices()
    )

    embb = slices["eMBB"]
    urllc = slices["URLLC"]
    mmtc = slices["mMTC"]

    embb_accounting = (
        collect_slice_packet_accounting(
            embb
        )
    )

    urllc_accounting = (
        collect_slice_packet_accounting(
            urllc
        )
    )

    mmtc_accounting = (
        collect_slice_packet_accounting(
            mmtc
        )
    )

    embb_lifecycle = (
        make_lifecycle_summary(
            embb_accounting
        )
    )

    urllc_lifecycle = (
        make_lifecycle_summary(
            urllc_accounting
        )
    )

    mmtc_lifecycle = (
        make_lifecycle_summary(
            mmtc_accounting
        )
    )

    embb_throughput = (
        active_throughput_mbps(
            delivered_bytes=(
                _active_slice_bytes(
                    embb
                )
            ),
            active_duration_ms=(
                active_duration_ms
            ),
        )
    )

    mmtc_throughput = (
        active_throughput_mbps(
            delivered_bytes=(
                _active_slice_bytes(
                    mmtc
                )
            ),
            active_duration_ms=(
                active_duration_ms
            ),
        )
    )

    embb_fairness = (
        jains_fairness_index(
            _per_ue_active_throughputs(
                embb,
                active_duration_ms,
            )
        )
    )

    mmtc_fairness = (
        jains_fairness_index(
            _per_ue_active_throughputs(
                mmtc,
                active_duration_ms,
            )
        )
    )

    radio = snapshot_radio_counters(
        slices
    )

    utilization = (
        radio_prb_utilization(
            used_prbs=(
                radio.used_prbs
            ),
            available_prbs=(
                radio.available_prbs
            ),
        )
    )

    conservation_ok = all(
        (
            embb_accounting.conservation_ok,
            urllc_accounting.conservation_ok,
            mmtc_accounting.conservation_ok,
        )
    )

    if not conservation_ok:
        raise RuntimeError(
            "run-level packet "
            "conservation failed"
        )

    return RunSummary(
        model_variant=(
            config.model_variant
        ),
        seed=int(
            config.seed
        ),
        traffic_scenario=(
            config.traffic_scenario
        ),
        radio_scenario=(
            config.radio_scenario
        ),

        control_interval_ms=float(
            config.control_interval_ms
        ),
        active_duration_ms=(
            active_duration_ms
        ),
        drain_duration_ms=(
            drain_duration_ms
        ),

        embb_lifecycle=(
            embb_lifecycle
        ),
        urllc_lifecycle=(
            urllc_lifecycle
        ),
        mmtc_lifecycle=(
            mmtc_lifecycle
        ),

        embb_performance=(
            SlicePerformanceSummary(
                active_throughput_mbps=(
                    embb_throughput
                ),
                jain_fairness=(
                    embb_fairness
                ),
                starvation=(
                    make_starvation_summary(
                        assured_scheduler
                        .embb_starvation_tracker
                    )
                ),
            )
        ),

        mmtc_performance=(
            SlicePerformanceSummary(
                active_throughput_mbps=(
                    mmtc_throughput
                ),
                jain_fairness=(
                    mmtc_fairness
                ),
                starvation=(
                    make_starvation_summary(
                        assured_scheduler
                        .mmtc_starvation_tracker
                    )
                ),
            )
        ),

        urllc_qos=(
            make_urllc_qos_summary(
                urllc
            )
        ),

        system=SystemSummary(
            prb_utilization=(
                utilization
            )
        ),

        conservation_ok=True,
    )
