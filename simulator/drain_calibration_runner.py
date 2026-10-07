from dataclasses import dataclass

import simpy

from Cell import Cell
from Results import UEgroup

from config.experiment_config import (
    ExperimentConfig,
)

from experiments.assured_horizon import (
    build_experiment_horizon,
)

from metrics.deadline_metrics import (
    collect_completion_delays,
)

from metrics.packet_accounting import (
    collect_slice_packet_accounting,
)

from metrics.qos_metrics import (
    delay_summary,
    lifecycle_ratios,
)

from environment.traffic_scenarios import (
    get_traffic_scenario,
)

@dataclass(frozen=True)
class DrainSliceSummary:
    generated: int
    delivered: int
    dropped: int
    residual: int

    completion_ratio: float
    true_drop_ratio: float
    residual_ratio: float

    completion_delay_count: int
    completion_delay_mean_ms: float
    completion_delay_p95_ms: float
    completion_delay_p99_ms: float
    completion_delay_p99_9_ms: float
    completion_delay_max_ms: float


@dataclass(frozen=True)
class DrainCalibrationSummary:
    seed: int
    traffic_scenario: str
    
    active_duration_ms: float
    drain_duration_ms: float
    total_duration_ms: float

    embb: DrainSliceSummary
    urllc: DrainSliceSummary
    mmtc: DrainSliceSummary


def _slice_flows(
    slice_obj,
):
    return [
        ue.packetFlows[0]
        for ue
        in slice_obj.schedulerDL.ues.values()
        if ue.packetFlows
    ]


def _summarize_slice(
    slice_obj,
):
    accounting = (
        collect_slice_packet_accounting(
            slice_obj
        )
    )

    lifecycle = lifecycle_ratios(
        generated=accounting.generated,
        delivered=accounting.delivered,
        dropped=accounting.dropped,
        residual=accounting.residual,
    )

    if not lifecycle.conservation_ok:
        raise RuntimeError(
            "drain calibration packet "
            "conservation failed"
        )

    completion = delay_summary(
        collect_completion_delays(
            _slice_flows(
                slice_obj
            )
        )
    )

    return DrainSliceSummary(
        generated=lifecycle.generated,
        delivered=lifecycle.delivered,
        dropped=lifecycle.dropped,
        residual=lifecycle.residual,

        completion_ratio=(
            lifecycle.completion_ratio
        ),
        true_drop_ratio=(
            lifecycle.true_drop_ratio
        ),
        residual_ratio=(
            lifecycle.residual_ratio
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
def run_drain_calibration_case(
    active_duration_ms,
    drain_duration_ms,
    seed,
    traffic_scenario="BASELINE",
):
    horizon = build_experiment_horizon(
        active_duration_ms=(
            active_duration_ms
        ),
        drain_duration_ms=(
            drain_duration_ms
        ),
    )
    traffic = get_traffic_scenario(
        traffic_scenario
    )

    config = ExperimentConfig(
        model_variant="M1",
        seed=int(seed),
        control_interval_ms=1.0,
        radio_update_interval_ms=10.0,
        drain_duration_ms=(
            horizon.drain_duration_ms
        ),
        urllc_scheduling_deadline_ms=1.0,
        training_mode=True,
        radio_scenario="STATIC_GOOD",
        traffic_scenario=traffic.name,
    )

    env = simpy.Environment()

    bandwidth = [
        10
    ]

    cell = Cell(
        "c1",
        bandwidth,
        "FR1",
        False,
        81920,
        False,
        1.0,

        # Empty scheduler name selects the base
        # deterministic InterSliceScheduler.
        "",

        experiment_config=config,
    )

    inter_slice = (
        cell.interSliceSched
    )

    t_sim = (
        horizon.total_duration_ms
    )

    meas_interval = 50.0
    embb_group = UEgroup(
        traffic.embb.users_dl,
        0,
        traffic.embb.packet_size_bytes,
        0,
        traffic.embb.arrival_rate,
        0,
        0,
        1,
        traffic.embb.size_distribution,
        traffic.embb.arrival_distribution,
        "eMBB",
        20,
        "",
        "Rr",
        "SU",
        4,
        cell,
        t_sim,
        meas_interval,
        env,
        "S37",
        None,
        experiment_config=config,
    )

    urllc_group = UEgroup(
        traffic.urllc.users_dl,
        0,
        traffic.urllc.packet_size_bytes,
        0,
        traffic.urllc.arrival_rate,
        0,
        0,
        1,
        traffic.urllc.size_distribution,
        traffic.urllc.arrival_distribution,
        "URLLC",
        5,
        "",
        "Rr",
        "SU",
        4,
        cell,
        t_sim,
        meas_interval,
        env,
        "S37",
        config.urllc_scheduling_deadline_ms,
        experiment_config=config,
    )

    mmtc_group = UEgroup(
        traffic.mmtc.users_dl,
        0,
        traffic.mmtc.packet_size_bytes,
        0,
        traffic.mmtc.arrival_rate,
        0,
        0,
        1,
        traffic.mmtc.size_distribution,
        traffic.mmtc.arrival_distribution,
        "mMTC",
        20,
        "",
        "Rr",
        "SU",
        4,
        cell,
        t_sim,
        meas_interval,
        env,
        "S37",
        None,
        experiment_config=config,
    )

    groups = [
        embb_group,
        urllc_group,
        mmtc_group,
    ]
    for group in groups:
        inter_slice.createSlice(
            group.req[
                "reqDelay"
            ],
            group.req[
                "reqThroughputDL"
            ],
            group.req[
                "reqThroughputUL"
            ],
            group.req[
                "reqAvailability"
            ],
            group.num_usersDL,
            group.num_usersUL,
            "B1",
            False,
            group.mmMd,
            group.lyrs,
            group.label,
            group.sch,
        )

    env.process(
        inter_slice.resAlloc(
            env
        )
    )

    for group in groups:
        group.activateSliceScheds(
            inter_slice,
            env,
        )

    env.run(
        until=t_sim
    )
    for slice_obj in (
        inter_slice
        .slices
        .values()
    ):
        slice_obj.schedulerDL.dbFile.close()

        if (
            slice_obj.label
            != "LTE"
        ):
            slice_obj.schedulerUL.dbFile.close()

    slices = (
        inter_slice.slices
    )

    return DrainCalibrationSummary(
        seed=int(
            seed
        ),
        traffic_scenario=traffic.name,
        active_duration_ms=(
            horizon.active_duration_ms
        ),
        drain_duration_ms=(
            horizon.drain_duration_ms
        ),
        total_duration_ms=(
            horizon.total_duration_ms
        ),

        embb=_summarize_slice(
            slices["eMBB"]
        ),
        urllc=_summarize_slice(
            slices["URLLC"]
        ),
        mmtc=_summarize_slice(
            slices["mMTC"]
        ),
    )


