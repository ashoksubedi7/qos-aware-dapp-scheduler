import simpy

from Cell import Cell
from Results import UEgroup

from config.experiment_config import (
    ExperimentConfig,
)
from environment.traffic_scenarios import (
    get_traffic_scenario,
)
from experiments.assured_horizon import (
    build_experiment_horizon,
)


ACTIVE_DURATION_MS = 100.0
DRAIN_DURATION_MS = 1000.0
SEED = 7
TRAFFIC_SCENARIO = "BASELINE"


def make_group(
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


def close_scheduler_files(
    inter_slice,
):
    for slice_obj in (
        inter_slice.slices.values()
    ):
        slice_obj.schedulerDL.dbFile.close()

        if slice_obj.label != "LTE":
            slice_obj.schedulerUL.dbFile.close()


def packet_ids(packets):
    return {
        int(packet.secNum)
        for packet in packets
    }


def check_urllc_flow(
    ue,
    flow,
):
    generated = {
        int(value)
        for value in flow.generatedPacketIds
    }

    evaluated = {
        int(value)
        for value
        in flow.deadlineEvaluatedPacketIds
    }

    missed = {
        int(value)
        for value
        in flow.deadlineMissPacketIds
    }

    completed = {
        int(value)
        for value in flow.completedPacketIds
    }

    dropped = {
        int(value)
        for value in flow.droppedPacketIds
    }

    residual = (
        generated
        - completed
        - dropped
    )

    generation_time_ids = {
        int(value)
        for value
        in flow.packetGenerationTimes.keys()
    }

    app_ids = packet_ids(
        flow.appBuff.pckts
    )

    bearer_ids = set()

    if ue.bearers:
        bearer_ids = packet_ids(
            ue.bearers[
                0
            ].buffer.pckts
        )

    physical_waiting_ids = (
        app_ids
        | bearer_ids
    )

    assert (
        completed
        <= generated
    ), (
        f"{ue.id}: completed packet "
        "outside generated set"
    )

    assert (
        dropped
        <= generated
    ), (
        f"{ue.id}: dropped packet "
        "outside generated set"
    )

    assert completed.isdisjoint(
        dropped
    ), (
        f"{ue.id}: packet is both "
        "completed and dropped"
    )

    assert (
        generated
        == completed
        | dropped
        | residual
    ), (
        f"{ue.id}: lifecycle "
        "conservation failed"
    )

    assert (
        generation_time_ids
        == residual
    ), (
        f"{ue.id}: packetGenerationTimes "
        "does not equal lifecycle residual"
    )

    assert (
        physical_waiting_ids
        <= residual
    ), (
        f"{ue.id}: physically queued packet "
        "is not lifecycle residual"
    )

    assert (
        int(flow.deadlineEvaluated)
        == len(evaluated)
    ), (
        f"{ue.id}: evaluated counter/set "
        "mismatch"
    )

    assert (
        int(flow.deadlineMisses)
        == len(missed)
    ), (
        f"{ue.id}: miss counter/set "
        "mismatch"
    )

    assert (
        missed
        <= evaluated
    ), (
        f"{ue.id}: missed set is not "
        "a subset of evaluated set"
    )

    assert (
        evaluated
        <= generated
    ), (
        f"{ue.id}: evaluated packet "
        "outside generated set"
    )

    # Critical deadline-accounting invariant.
    #
    # Traffic generation has stopped and the run has
    # continued for 1000 ms. Every 1-ms scheduling
    # deadline must therefore have been resolved,
    # regardless of whether the packet was delivered,
    # dropped, scheduled-but-in-flight, or remains
    # unscheduled residual traffic.
    assert (
        evaluated
        == generated
    ), (
        f"{ue.id}: not every generated "
        "deadline-bearing packet was evaluated; "
        f"generated={len(generated)}, "
        f"evaluated={len(evaluated)}"
    )

    return {
        "generated": len(generated),
        "evaluated": len(evaluated),
        "missed": len(missed),
        "completed": len(completed),
        "dropped": len(dropped),
        "residual": len(residual),
        "physically_waiting": (
            len(physical_waiting_ids)
        ),
    }


def run():
    horizon = (
        build_experiment_horizon(
            active_duration_ms=(
                ACTIVE_DURATION_MS
            ),
            drain_duration_ms=(
                DRAIN_DURATION_MS
            ),
        )
    )

    traffic = get_traffic_scenario(
        TRAFFIC_SCENARIO
    )

    config = ExperimentConfig(
        model_variant="M1",
        seed=SEED,
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

    cell = Cell(
        "c1",
        [10],
        "FR1",
        False,
        81920,
        False,
        1.0,
        "AQM1",
        experiment_config=config,
    )

    inter_slice = (
        cell.interSliceSched
    )

    # This is a correctness/invariant run, not a
    # learning experiment. Keep the production
    # AssuredScheduler control loop and action
    # selection, but prevent replay/training state
    # from being mutated during the test.
    inter_slice.agent.remember = (
        lambda *args, **kwargs: None
    )

    inter_slice.agent.train_step = (
        lambda *args, **kwargs: None
    )

    inter_slice.agent.decay_epsilon = (
        lambda *args, **kwargs: None
    )

    t_sim = (
        horizon.total_duration_ms
    )

    meas_interval = 50.0

    embb_group = make_group(
        traffic.embb,
        "eMBB",
        20,
        None,
        cell,
        t_sim,
        meas_interval,
        env,
        config,
    )

    urllc_group = make_group(
        traffic.urllc,
        "URLLC",
        5,
        config.urllc_scheduling_deadline_ms,
        cell,
        t_sim,
        meas_interval,
        env,
        config,
    )

    mmtc_group = make_group(
        traffic.mmtc,
        "mMTC",
        20,
        None,
        cell,
        t_sim,
        meas_interval,
        env,
        config,
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

    close_scheduler_files(
        inter_slice
    )

    totals = {
        "generated": 0,
        "evaluated": 0,
        "missed": 0,
        "completed": 0,
        "dropped": 0,
        "residual": 0,
        "physically_waiting": 0,
    }

    print()
    print(
        "AssuredQoS real-simulator "
        "deadline invariant"
    )
    print("=" * 72)

    print(
        "active_duration_ms:",
        horizon.active_duration_ms,
    )

    print(
        "drain_duration_ms:",
        horizon.drain_duration_ms,
    )

    print(
        "total_duration_ms:",
        horizon.total_duration_ms,
    )

    print(
        "traffic_scenario:",
        traffic.name,
    )

    print(
        "seed:",
        SEED,
    )

    print()

    for ue, flow in zip(
        urllc_group.usersDL,
        urllc_group.flowsDL,
    ):
        result = check_urllc_flow(
            ue,
            flow,
        )

        print(
            f"{ue.id}: "
            f"G={result['generated']} "
            f"E={result['evaluated']} "
            f"M={result['missed']} "
            f"C={result['completed']} "
            f"D={result['dropped']} "
            f"R={result['residual']} "
            f"Q={result['physically_waiting']}"
        )

        for key in totals:
            totals[key] += result[
                key
            ]

    print()
    print("-" * 72)

    print(
        "TOTAL:",
        f"G={totals['generated']}",
        f"E={totals['evaluated']}",
        f"M={totals['missed']}",
        f"C={totals['completed']}",
        f"D={totals['dropped']}",
        f"R={totals['residual']}",
        f"Q={totals['physically_waiting']}",
    )

    assert (
        totals["evaluated"]
        == totals["generated"]
    )

    assert (
        totals["missed"]
        <= totals["evaluated"]
    )

    assert (
        totals["generated"]
        == (
            totals["completed"]
            + totals["dropped"]
            + totals["residual"]
        )
    )

    print()
    print(
        "REAL-SIM DEADLINE INVARIANT: PASS"
    )


if __name__ == "__main__":
    run()
