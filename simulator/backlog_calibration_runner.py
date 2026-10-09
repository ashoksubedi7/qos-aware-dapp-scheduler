from dataclasses import dataclass
import math

import csv
from pathlib import Path

import simpy

from Cell import Cell
from Results import UEgroup

from assured_metrics import (
    slice_backlog_bytes,
)

from config.experiment_config import (
    ExperimentConfig,
)

from environment.traffic_scenarios import (
    get_traffic_scenario,
)


CALIBRATION_SCENARIOS = (
    "BASELINE",
    "CONGESTED",
    "URLLC_HIGH",
    "SIMULTANEOUS_HIGH",
)

SAMPLE_INTERVAL_MS = 1.0


@dataclass(frozen=True)
class BacklogCalibrationRun:
    seed: int
    traffic_scenario: str
    active_duration_ms: float
    sample_interval_ms: float

    times_ms: tuple
    embb_bytes: tuple
    urllc_bytes: tuple
    mmtc_bytes: tuple

    @property
    def sample_count(self):
        return len(self.times_ms)


def _expected_sample_count(
    active_duration_ms,
    sample_interval_ms,
):
    active_duration_ms = float(
        active_duration_ms
    )

    sample_interval_ms = float(
        sample_interval_ms
    )

    if active_duration_ms <= 0:
        raise ValueError(
            "active_duration_ms must be positive"
        )

    if sample_interval_ms <= 0:
        raise ValueError(
            "sample_interval_ms must be positive"
        )

    ratio = (
        active_duration_ms
        / sample_interval_ms
    )

    rounded = round(ratio)

    if not math.isclose(
        ratio,
        rounded,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "active duration must be an integer "
            "multiple of the sampling interval"
        )

    return int(rounded)


def _sample_backlog_process(
    env,
    inter_slice,
    active_duration_ms,
    sample_interval_ms,
    times_ms,
    samples,
):
    """
    Record controller-equivalent pre-action backlog.

    This process must be registered before the
    inter-slice resAlloc process.
    """

    while (
        float(env.now)
        < float(active_duration_ms)
    ):
        times_ms.append(
            float(env.now)
        )

        samples["eMBB"].append(
            slice_backlog_bytes(
                inter_slice.slices["eMBB"]
            )
        )

        samples["URLLC"].append(
            slice_backlog_bytes(
                inter_slice.slices["URLLC"]
            )
        )

        samples["mMTC"].append(
            slice_backlog_bytes(
                inter_slice.slices["mMTC"]
            )
        )

        yield env.timeout(
            float(sample_interval_ms)
        )


def _make_group(
    profile,
    label,
    required_delay_ms,
    scheduling_deadline_ms,
    cell,
    t_sim,
    meas_interval,
    env,
    config,
):
    return UEgroup(
        profile.users_dl,
        0,
        profile.packet_size_bytes,
        0,
        profile.arrival_rate,
        0,

        0,
        1,

        profile.size_distribution,
        profile.arrival_distribution,

        label,
        required_delay_ms,
        "",
        "Rr",
        "SU",
        4,

        cell,
        t_sim,
        meas_interval,
        env,

        "S37",
        scheduling_deadline_ms,

        experiment_config=config,
    )


def _close_simulator_files(
    inter_slice,
):
    for slice_obj in (
        inter_slice.slices.values()
    ):
        if not (
            slice_obj.schedulerDL.dbFile.closed
        ):
            slice_obj.schedulerDL.dbFile.close()

        if (
            slice_obj.label != "LTE"
            and not (
                slice_obj
                .schedulerUL
                .dbFile
                .closed
            )
        ):
            slice_obj.schedulerUL.dbFile.close()

    if (
        hasattr(inter_slice, "dbFile")
        and not inter_slice.dbFile.closed
    ):
        inter_slice.dbFile.close()


def run_backlog_calibration_case(
    active_duration_ms,
    seed,
    traffic_scenario="BASELINE",
    sample_interval_ms=SAMPLE_INTERVAL_MS,
):
    """
    Run one raw backlog-byte calibration case.

    Samples are collected during the active period
    only. No drain-period observations are included.
    """

    if (
        traffic_scenario
        not in CALIBRATION_SCENARIOS
    ):
        raise ValueError(
            "unsupported backlog calibration "
            f"scenario: {traffic_scenario}"
        )

    expected_samples = (
        _expected_sample_count(
            active_duration_ms,
            sample_interval_ms,
        )
    )

    # The preregistered protocol requires 1-ms
    # controller-equivalent sampling.
    if not math.isclose(
        float(sample_interval_ms),
        SAMPLE_INTERVAL_MS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "backlog calibration requires a "
            "1-ms sampling interval"
        )

    traffic = get_traffic_scenario(
        traffic_scenario
    )

    config = ExperimentConfig(
        model_variant="M1",
        seed=int(seed),
        control_interval_ms=1.0,
        radio_update_interval_ms=10.0,
        drain_duration_ms=0.0,
        urllc_scheduling_deadline_ms=1.0,
        training_mode=True,
        radio_scenario="STATIC_GOOD",
        traffic_scenario=traffic.name,
    )

    env = simpy.Environment()

    cell = Cell(
        "c1",
        [10],
        "FR1",
        False,
        81920,
        False,
        1.0,

        # Empty scheduler name selects the
        # deterministic base InterSliceScheduler.
        "",

        experiment_config=config,
    )

    inter_slice = (
        cell.interSliceSched
    )

    t_sim = float(
        active_duration_ms
    )

    meas_interval = 50.0

    embb_group = _make_group(
        profile=traffic.embb,
        label="eMBB",
        required_delay_ms=20,
        scheduling_deadline_ms=None,
        cell=cell,
        t_sim=t_sim,
        meas_interval=meas_interval,
        env=env,
        config=config,
    )

    urllc_group = _make_group(
        profile=traffic.urllc,
        label="URLLC",
        required_delay_ms=5,
        scheduling_deadline_ms=(
            config
            .urllc_scheduling_deadline_ms
        ),
        cell=cell,
        t_sim=t_sim,
        meas_interval=meas_interval,
        env=env,
        config=config,
    )

    mmtc_group = _make_group(
        profile=traffic.mmtc,
        label="mMTC",
        required_delay_ms=20,
        scheduling_deadline_ms=None,
        cell=cell,
        t_sim=t_sim,
        meas_interval=meas_interval,
        env=env,
        config=config,
    )

    groups = [
        embb_group,
        urllc_group,
        mmtc_group,
    ]

    for group in groups:
        inter_slice.createSlice(
            group.req["reqDelay"],
            group.req["reqThroughputDL"],
            group.req["reqThroughputUL"],
            group.req["reqAvailability"],
            group.num_usersDL,
            group.num_usersUL,
            "B1",
            False,
            group.mmMd,
            group.lyrs,
            group.label,
            group.sch,
        )

    times_ms = []

    samples = {
        "eMBB": [],
        "URLLC": [],
        "mMTC": [],
    }

    # IMPORTANT EVENT ORDER
    #
    # UE traffic/radio processes were registered during
    # UEgroup construction.
    #
    # Register the sampler BEFORE resAlloc so that at
    # every 1-ms boundary it observes the state before
    # the deterministic inter-slice action.
    env.process(
        _sample_backlog_process(
            env=env,
            inter_slice=inter_slice,
            active_duration_ms=t_sim,
            sample_interval_ms=(
                sample_interval_ms
            ),
            times_ms=times_ms,
            samples=samples,
        )
    )

    env.process(
        inter_slice.resAlloc(
            env
        )
    )

    # Intra-slice schedulers are intentionally
    # registered after the inter-slice controller,
    # matching the existing simulation runner.
    for group in groups:
        group.activateSliceScheds(
            inter_slice,
            env,
        )

    env.run(
        until=t_sim
    )

    _close_simulator_files(
        inter_slice
    )

    lengths = {
        "time": len(times_ms),
        "eMBB": len(samples["eMBB"]),
        "URLLC": len(samples["URLLC"]),
        "mMTC": len(samples["mMTC"]),
    }

    if any(
        value != expected_samples
        for value in lengths.values()
    ):
        raise RuntimeError(
            "unexpected backlog sample count: "
            f"{lengths}; expected "
            f"{expected_samples}"
        )

    return BacklogCalibrationRun(
        seed=int(seed),
        traffic_scenario=traffic.name,
        active_duration_ms=t_sim,
        sample_interval_ms=float(
            sample_interval_ms
        ),

        times_ms=tuple(times_ms),

        embb_bytes=tuple(
            samples["eMBB"]
        ),

        urllc_bytes=tuple(
            samples["URLLC"]
        ),

        mmtc_bytes=tuple(
            samples["mMTC"]
        ),
    )
def write_backlog_run_csv(
    run,
    output_path,
):
    """
    Write one calibration run without normalization
    or aggregation.

    One row corresponds to one controller-equivalent
    pre-action observation.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lengths = {
        len(run.times_ms),
        len(run.embb_bytes),
        len(run.urllc_bytes),
        len(run.mmtc_bytes),
    }

    if len(lengths) != 1:
        raise ValueError(
            "backlog run arrays have "
            "inconsistent lengths"
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "time_ms",
                "embb_bytes",
                "urllc_bytes",
                "mmtc_bytes",
            ]
        )

        for values in zip(
            run.times_ms,
            run.embb_bytes,
            run.urllc_bytes,
            run.mmtc_bytes,
        ):
            writer.writerow(
                values
            )

    return output_path
