import math

import simpy

from config.experiment_config import ExperimentConfig
from experiments import scientific_evaluation_executor as evaluation


def test_control_loop_handles_no_connected_embb_users(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    config = ExperimentConfig(
        model_variant="M1",
        seed=202,
        training_mode=True,
        traffic_scenario="BASELINE",
        radio_scenario="STATIC_GOOD",
    )

    env = simpy.Environment()

    cell = evaluation.Cell(
        "c1",
        [10],
        "FR1",
        False,
        81920,
        False,
        config.control_interval_ms,
        evaluation._MODEL_VARIANT_TO_SCHEDULER["M1"],
        experiment_config=config,
    )

    scheduler = cell.interSliceSched
    traffic = evaluation.get_traffic_scenario("BASELINE")

    total_duration_ms = (
        300.0 + config.drain_duration_ms
    )

    specifications = (
        (traffic.embb, "eMBB", 20, None),
        (
            traffic.urllc,
            "URLLC",
            5,
            config.urllc_scheduling_deadline_ms,
        ),
        (traffic.mmtc, "mMTC", 20, None),
    )

    groups = []

    for profile, label, delay, deadline in specifications:
        group = evaluation._make_group(
            profile,
            label,
            delay,
            deadline,
            cell,
            total_duration_ms,
            50.0,
            env,
            config,
        )

        # Delay traffic generation beyond the first
        # control interval to exercise a valid idle state.
        for flow in group.flowsDL:
            flow.activationTime = 0.02

        groups.append(group)

    for group in groups:
        evaluation._create_slice(scheduler, group)

    embb = scheduler.slices["eMBB"]

    assert embb.reqThroughputDL > 0
    assert len(embb.schedulerDL.ues) == 0

    observed_targets = []
    original_target = scheduler._embb_target_mbps

    def record_target():
        target = original_target()
        observed_targets.append((env.now, target))
        return target

    scheduler._embb_target_mbps = record_target

    env.process(scheduler.resAlloc(env))

    for group in groups:
        group.activateSliceScheds(scheduler, env)

    try:
        env.run(until=1.01)
    finally:
        evaluation._close_scheduler_files(scheduler)

    assert observed_targets == [(1.0, 0.0)]
    assert len(embb.schedulerDL.ues) == 0
    assert math.isfinite(scheduler.reward)
