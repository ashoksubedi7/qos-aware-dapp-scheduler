from collections import deque
from types import SimpleNamespace

import pytest

from metrics.run_summary import (
    _active_slice_bytes,
    _per_ue_active_throughputs,
    make_lifecycle_summary,
    make_starvation_summary,
    make_urllc_qos_summary,
    build_run_summary,
)

def make_flow(
    active_bytes=0,
    scheduling_delays=None,
    completion_delays=None,
    deadline_evaluated=0,
    deadline_misses=0,
):
    return SimpleNamespace(
        activeDeliveredBytes=active_bytes,
        schedulingDelays=list(
            scheduling_delays or []
        ),
        completionDelays=list(
            completion_delays or []
        ),
        deadlineEvaluated=deadline_evaluated,
        deadlineMisses=deadline_misses,
    )


def make_slice(
    flows,
):
    ues = {}

    for index, flow in enumerate(
        flows,
        start=1,
    ):
        ues[f"ue{index}"] = (
            SimpleNamespace(
                packetFlows=[
                    flow
                ]
            )
        )

    return SimpleNamespace(
        schedulerDL=SimpleNamespace(
            ues=ues
        )
    )

@pytest.fixture
def fake_assured_scheduler():
    def make_packet(
        packet_id,
    ):
        return SimpleNamespace(
            secNum=packet_id
        )

    def make_flow(
        generated,
        delivered,
        dropped,
        residual,
        active_bytes,
        drain_bytes=0,
        scheduling_delays=None,
        completion_delays=None,
        deadline_evaluated=0,
        deadline_misses=0,
    ):
        generated_ids = set(
            range(
                1,
                generated + 1,
            )
        )

        delivered_ids = set(
            range(
                1,
                delivered + 1,
            )
        )

        dropped_ids = set(
            range(
                delivered + 1,
                delivered + dropped + 1,
            )
        )

        residual_ids = (
            generated_ids
            - delivered_ids
            - dropped_ids
        )

        return SimpleNamespace(
            generatedPacketIds=(
                generated_ids
            ),
            completedPacketIds=(
                delivered_ids
            ),
            droppedPacketIds=(
                dropped_ids
            ),

            appBuff=SimpleNamespace(
                pckts=[
                    make_packet(
                        packet_id
                    )
                    for packet_id
                    in residual_ids
                ]
            ),

            activeDeliveredBytes=(
                active_bytes
            ),
            drainDeliveredBytes=(
                drain_bytes
            ),

            schedulingDelays=list(
                scheduling_delays or []
            ),
            completionDelays=list(
                completion_delays or []
            ),

            deadlineEvaluated=(
                deadline_evaluated
            ),
            deadlineMisses=(
                deadline_misses
            ),
        )

    def make_ue(
        ue_id,
        flow,
    ):
        return SimpleNamespace(
            id=ue_id,

            packetFlows=[
                flow
            ],

            bearers=[
                SimpleNamespace(
                    buffer=SimpleNamespace(
                        pckts=[]
                    )
                )
            ],

            pendingPckts={},
            pendingTB=[],
        )

    def make_slice(
        ues,
        used_prbs,
        available_prbs,
    ):
        scheduler = SimpleNamespace(
            ues={
                ue.id: ue
                for ue in ues
            },

            queue=SimpleNamespace(
                res=[]
            ),

            assuredPrbsUsed=(
                used_prbs
            ),
            assuredPrbsAvailable=(
                available_prbs
            ),
        )

        return SimpleNamespace(
            schedulerDL=scheduler
        )

    embb_flow_1 = make_flow(
        generated=10,
        delivered=8,
        dropped=1,
        residual=1,
        active_bytes=62_500,
    )

    embb_flow_2 = make_flow(
        generated=10,
        delivered=8,
        dropped=1,
        residual=1,
        active_bytes=62_500,
    )

    urllc_flow = make_flow(
        generated=10,
        delivered=9,
        dropped=0,
        residual=1,
        active_bytes=50_000,
        scheduling_delays=[
            0.2,
            0.8,
        ],
        completion_delays=[
            1.2,
            2.0,
        ],
        deadline_evaluated=2,
        deadline_misses=1,
    )

    mmtc_flow = make_flow(
        generated=10,
        delivered=7,
        dropped=1,
        residual=2,
        active_bytes=25_000,
    )

    slices = {
        "eMBB": make_slice(
            [
                make_ue(
                    "embb-1",
                    embb_flow_1,
                ),
                make_ue(
                    "embb-2",
                    embb_flow_2,
                ),
            ],
            used_prbs=30,
            available_prbs=50,
        ),

        "URLLC": make_slice(
            [
                make_ue(
                    "urllc-1",
                    urllc_flow,
                ),
            ],
            used_prbs=10,
            available_prbs=20,
        ),

        "mMTC": make_slice(
            [
                make_ue(
                    "mmtc-1",
                    mmtc_flow,
                ),
            ],
            used_prbs=5,
            available_prbs=20,
        ),
    }

    config = SimpleNamespace(
        model_variant="M2",
        seed=7,
        traffic_scenario="moderate",
        radio_scenario="stable",
        control_interval_ms=1.0,
        drain_duration_ms=100.0,
    )

    scheduler = SimpleNamespace(
        config=config,

        embb_starvation_tracker=(
            SimpleNamespace(
                event_count=1,
                total_ms=12.0,
                max_ms=8.0,
            )
        ),

        mmtc_starvation_tracker=(
            SimpleNamespace(
                event_count=2,
                total_ms=20.0,
                max_ms=11.0,
            )
        ),
    )

    scheduler._canonical_slices = (
        lambda: slices
    )

    return scheduler

def test_active_slice_bytes_sum_ues():
    slice_obj = make_slice(
        [
            make_flow(
                active_bytes=1000
            ),
            make_flow(
                active_bytes=2500
            ),
        ]
    )

    assert (
        _active_slice_bytes(
            slice_obj
        )
        == 3500
    )

def test_per_ue_active_throughputs_use_si_mbps():
    slice_obj = make_slice(
        [
            make_flow(
                active_bytes=125_000
            ),
            make_flow(
                active_bytes=250_000
            ),
        ]
    )

    values = (
        _per_ue_active_throughputs(
            slice_obj,
            active_duration_ms=1000.0,
        )
    )

    assert values == pytest.approx(
        [
            1.0,
            2.0,
        ]
    )

def test_urllc_summary_keeps_scheduling_and_completion_delay_separate():
    slice_obj = make_slice(
        [
            make_flow(
                scheduling_delays=[
                    0.2,
                    0.8,
                ],
                completion_delays=[
                    1.2,
                    2.0,
                ],
                deadline_evaluated=2,
                deadline_misses=1,
            )
        ]
    )

    result = (
        make_urllc_qos_summary(
            slice_obj
        )
    )

    assert (
        result.deadline_evaluated
        == 2
    )

    assert (
        result.deadline_misses
        == 1
    )

    assert (
        result.scheduling_deadline_miss_ratio
        == pytest.approx(
            0.5
        )
    )

    assert (
        result.scheduling_delay_mean_ms
        == pytest.approx(
            0.5
        )
    )

    assert (
        result.completion_delay_mean_ms
        == pytest.approx(
            1.6
        )
    )

def test_starvation_summary_preserves_tracker_values():
    tracker = SimpleNamespace(
        event_count=3,
        total_ms=42.0,
        max_ms=20.0,
    )

    result = (
        make_starvation_summary(
            tracker
        )
    )

    assert result.event_count == 3
    assert result.total_ms == 42.0
    assert result.max_ms == 20.0

def test_lifecycle_summary_uses_canonical_ratios():
    accounting = SimpleNamespace(
        generated=100,
        delivered=80,
        dropped=5,
        residual=15,
    )

    result = (
        make_lifecycle_summary(
            accounting
        )
    )

    assert result.completion_ratio == pytest.approx(
        0.80
    )

    assert result.true_drop_ratio == pytest.approx(
        0.05
    )

    assert result.residual_ratio == pytest.approx(
        0.15
    )


def test_build_run_summary_core_invariants(
    fake_assured_scheduler,
):
    summary = build_run_summary(
        fake_assured_scheduler,
        t_sim_ms=1100.0,
    )

    assert (
        summary.active_duration_ms
        == pytest.approx(
            1000.0
        )
    )

    assert summary.conservation_ok

    assert (
        0.0
        <= summary.system.prb_utilization
        <= 1.0
    )

    assert (
        summary.embb_lifecycle.generated
        ==
        summary.embb_lifecycle.delivered
        + summary.embb_lifecycle.dropped
        + summary.embb_lifecycle.residual
    )
def test_drain_bytes_do_not_inflate_active_throughput(
    fake_assured_scheduler,
):
    embb = (
        fake_assured_scheduler
        ._canonical_slices()["eMBB"]
    )

    flow = next(
        iter(
            embb.schedulerDL.ues.values()
        )
    ).packetFlows[0]

    flow.activeDeliveredBytes = 125_000
    flow.drainDeliveredBytes = 0

    first = build_run_summary(
        fake_assured_scheduler,
        t_sim_ms=1100.0,
    )

    throughput_before = (
        first
        .embb_performance
        .active_throughput_mbps
    )

    flow.drainDeliveredBytes = (
        999_999
    )

    second = build_run_summary(
        fake_assured_scheduler,
        t_sim_ms=1100.0,
    )

    throughput_after = (
        second
        .embb_performance
        .active_throughput_mbps
    )

    assert (
        throughput_after
        == pytest.approx(
            throughput_before
        )
    )
def test_run_summary_to_dict_is_nested_plain_dict(
    fake_assured_scheduler,
):
    summary = build_run_summary(
        fake_assured_scheduler,
        t_sim_ms=1100.0,
    )

    data = summary.to_dict()

    assert isinstance(
        data,
        dict,
    )

    assert isinstance(
        data["embb_lifecycle"],
        dict,
    )

    assert isinstance(
        data["system"],
        dict,
    )
