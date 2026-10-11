from pathlib import Path

import simpy

from Cell import Cell
from Results import UEgroup

from environment.traffic_scenarios import (
    get_traffic_scenario,
)
from experiments.scientific_run_provenance import (
    write_scientific_run_manifest,
)
from experiments.scientific_training_setup import (
    build_scientific_training_setup,
)
from experiments.training_update_controller import (
    TrainingUpdateController,
)
from metrics.run_summary import (
    build_run_summary,
)


_MODEL_VARIANT_TO_SCHEDULER = {
    "M1": "AQM1",
    "M2": "AQM2",
    "M3": "AQM3",
}


def _make_group(
    profile,
    label,
    delay_requirement,
    scheduling_deadline,
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
        delay_requirement,
        "",
        "Rr",
        "SU",
        4,
        cell,
        t_sim,
        meas_interval,
        env,
        "S37",
        scheduling_deadline,
        experiment_config=config,
    )


def _close_scheduler_files(
    inter_slice,
):
    for slice_obj in (
        inter_slice.slices.values()
    ):
        slice_obj.schedulerDL.dbFile.close()

        if slice_obj.label != "LTE":
            slice_obj.schedulerUL.dbFile.close()


def _create_slice(
    inter_slice,
    group,
):
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


def run_scientific_training(
    protocol,
    config,
    run_directory,
    source_commit,
):
    """
    Execute one scientific AssuredQoS training run.

    Training and drain execute as one continuous SimPy
    run. Learning stops internally on the successful-
    gradient-update budget through TrainingUpdateController.
    """
    setup = build_scientific_training_setup(
        protocol=protocol,
        config=config,
    )

    derived_config = setup.config
    horizon = setup.horizon

    run_directory = Path(
        run_directory
    )

    _, provenance = (
        write_scientific_run_manifest(
            run_directory=run_directory,
            protocol=protocol,
            config=derived_config,
            source_commit=source_commit,
        )
    )

    checkpoint_metadata = {
        "source_commit": (
            provenance["source_commit"]
        ),
        "training_protocol_sha256": (
            provenance[
                "training_protocol_sha256"
            ]
        ),
        "experiment_config_sha256": (
            provenance[
                "experiment_config_sha256"
            ]
        ),
        "run_definition_sha256": (
            provenance[
                "run_definition_sha256"
            ]
        ),
    }

    controller = TrainingUpdateController(
        protocol=protocol,
        checkpoint_root=(
            run_directory
            / "checkpoints"
        ),
        base_checkpoint_metadata=(
            checkpoint_metadata
        ),
    )

    scheduler_name = (
        _MODEL_VARIANT_TO_SCHEDULER[
            derived_config.model_variant
        ]
    )

    env = simpy.Environment()

    cell = Cell(
        "c1",
        [10],
        "FR1",
        False,
        81920,
        False,
        derived_config.control_interval_ms,
        scheduler_name,
        experiment_config=derived_config,
    )

    inter_slice = (
        cell.interSliceSched
    )

    inter_slice.set_successful_update_callback(
        controller
    )

    traffic = get_traffic_scenario(
        derived_config.traffic_scenario
    )

    t_sim = (
        horizon.total_duration_ms
    )

    meas_interval = 50.0

    groups = [
        _make_group(
            traffic.embb,
            "eMBB",
            20,
            None,
            cell,
            t_sim,
            meas_interval,
            env,
            derived_config,
        ),
        _make_group(
            traffic.urllc,
            "URLLC",
            5,
            derived_config
            .urllc_scheduling_deadline_ms,
            cell,
            t_sim,
            meas_interval,
            env,
            derived_config,
        ),
        _make_group(
            traffic.mmtc,
            "mMTC",
            20,
            None,
            cell,
            t_sim,
            meas_interval,
            env,
            derived_config,
        ),
    ]

    for group in groups:
        _create_slice(
            inter_slice,
            group,
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

    try:
        env.run(
            until=t_sim
        )
    finally:
        _close_scheduler_files(
            inter_slice
        )

    return build_run_summary(
        inter_slice,
        t_sim,
    )
