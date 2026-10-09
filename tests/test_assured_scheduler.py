import sys
from pathlib import Path
from types import SimpleNamespace
from UE import Packet, PacketFlow
import numpy as np
import pytest
from metrics.starvation_metrics import (
    StarvationTracker,
)

from metrics.transition_metrics import (
    SliceCounters,
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator" / "lib"),
)


from assured_scheduler import AssuredScheduler

from config.experiment_config import (
    ExperimentConfig,
)


def test_valid_model_variants():
    assert AssuredScheduler.VALID_VARIANTS == (
        "M1",
        "M2",
        "M3",
    )


def test_m1_definition():
    definition = {
        "input_dim": 6,
    }

    assert definition["input_dim"] == 6


def test_m2_definition():
    definition = {
        "input_dim": 7,
    }

    assert definition["input_dim"] == 7


def make_scheduler_without_init():
    scheduler = object.__new__(
        AssuredScheduler
    )

    return scheduler


def test_find_slice_key_exact_names():
    scheduler = (
        make_scheduler_without_init()
    )

    scheduler.slices = {
        "eMBB": SimpleNamespace(),
        "URLLC": SimpleNamespace(),
        "mMTC": SimpleNamespace(),
    }

    assert (
        scheduler._find_slice_key(
            "URLLC"
        )
        == "URLLC"
    )


def test_find_slice_key_extended_labels():
    scheduler = (
        make_scheduler_without_init()
    )

    scheduler.slices = {
        "eMBB-1": SimpleNamespace(),
        "URLLC-critical": SimpleNamespace(),
        "mMTC-30ue": SimpleNamespace(),
    }

    assert (
        scheduler._find_slice_key(
            "eMBB"
        )
        == "eMBB-1"
    )

    assert (
        scheduler._find_slice_key(
            "URLLC"
        )
        == "URLLC-critical"
    )


def test_duplicate_service_slice_is_rejected():
    scheduler = (
        make_scheduler_without_init()
    )

    scheduler.slices = {
        "URLLC-1": SimpleNamespace(),
        "URLLC-2": SimpleNamespace(),
        "eMBB": SimpleNamespace(),
        "mMTC": SimpleNamespace(),
    }

    with pytest.raises(
        ValueError,
        match="exactly one URLLC",
    ):
        scheduler._find_slice_key(
            "URLLC"
        )


def test_model_variants_are_distinct():
    assert (
        "M1"
        in AssuredScheduler.VALID_VARIANTS
    )

    assert (
        "M2"
        in AssuredScheduler.VALID_VARIANTS
    )

    assert (
        "M3"
        in AssuredScheduler.VALID_VARIANTS
    )


class FakeSlice:
    def __init__(
        self,
        factor,
    ):
        self.numRefFactor = factor
        self.last_prbs = None

    def updateConfig(
        self,
        prbs,
    ):
        self.last_prbs = prbs


def make_action_scheduler():
    scheduler = object.__new__(
        AssuredScheduler
    )

    scheduler.PRBs = 52

    scheduler.slices = {
        "eMBB": FakeSlice(1),
        "URLLC": FakeSlice(4),
        "mMTC": FakeSlice(1),
    }

    scheduler.last_requested_weights = None
    scheduler.last_slice_prbs = None
    scheduler.last_reference_prbs = None

    return scheduler


def test_apply_action_preserves_reference_budget():
    scheduler = make_action_scheduler()

    result = scheduler.apply_action(
        16
    )

    assert result["weights"] == (
        60,
        20,
        20,
    )

    assert result["slice_prbs"] == (
        30,
        3,
        10,
    )

    assert result[
        "reference_prbs"
    ] == (
        30,
        12,
        10,
    )

    assert (
        result[
            "reference_prbs_used"
        ]
        == 52
    )


def test_apply_action_updates_slice_configuration():
    scheduler = make_action_scheduler()

    scheduler.apply_action(
        16
    )

    assert (
        scheduler.slices[
            "eMBB"
        ].last_prbs
        == 30
    )

    assert (
        scheduler.slices[
            "URLLC"
        ].last_prbs
        == 3
    )

    assert (
        scheduler.slices[
            "mMTC"
        ].last_prbs
        == 10
    )


def test_idle_action_really_allocates_zero():
    scheduler = make_action_scheduler()

    result = scheduler.apply_action(
        0
    )

    assert result["slice_prbs"] == (
        0,
        0,
        0,
    )

    assert (
        result[
            "reference_prbs_used"
        ]
        == 0
    )


def test_embb_target_uses_number_of_ues():
    scheduler = object.__new__(
        AssuredScheduler
    )

    fake_embb = SimpleNamespace(
        reqThroughputDL=2.5,
        schedulerDL=SimpleNamespace(
            ues={
                "ue1": object(),
                "ue2": object(),
                "ue3": object(),
                "ue4": object(),
            }
        ),
    )

    scheduler.slices = {
        "eMBB": fake_embb,
        "URLLC": SimpleNamespace(),
        "mMTC": SimpleNamespace(),
    }

    assert (
        scheduler._embb_target_mbps()
        == 10.0
    )


def test_agent_action_space_matches_model_output():
    scheduler = object.__new__(
        AssuredScheduler
    )

    scheduler.model_definition = {
        "output_dim": 22,
    }

    scheduler.action_space_size = 22

    assert (
        scheduler.model_definition[
            "output_dim"
        ]
        == scheduler.action_space_size
    )


def test_scheduler_uses_config_values():
    config = ExperimentConfig(
        model_variant="M2",
        seed=21,
        control_interval_ms=2.0,
        urllc_scheduling_deadline_ms=1.5,
        starvation_threshold_ms=15.0,
    )

    scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        100,
        config=config,
    )

    assert (
        scheduler.model_variant
        == "M2"
    )

    assert (
        scheduler.granularity
        == 2.0
    )

    assert (
        scheduler.urllc_deadline_ms
        == 1.5
    )

    assert (
        scheduler.starvation_threshold_ms
        == 15.0
    )


def test_scheduler_rejects_non_config():
    with pytest.raises(
        TypeError,
    ):
        AssuredScheduler(
            [10],
            "FR1",
            False,
            False,
            1.0,
            config="bad-config",
        )


def test_agent_uses_config_seed():
    config_a = ExperimentConfig(
        seed=33,
    )

    config_b = ExperimentConfig(
        seed=33,
    )

    scheduler_a = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=config_a,
    )

    scheduler_b = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=config_b,
    )

    assert (
        scheduler_a.agent.rng.random()
        ==
        scheduler_b.agent.rng.random()
    )


def test_training_mode_enables_exploration():
    config = ExperimentConfig(
        training_mode=True,
        seed=7,
    )

    scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=config,
    )

    scheduler.agent.epsilon = 1.0

    state = np.zeros(
        scheduler.model_definition[
            "input_dim"
        ],
        dtype=np.float32,
    )

    actions = {
        scheduler.select_policy_action(
            state
        )
        for _ in range(20)
    }

    assert len(actions) > 1


def test_evaluation_mode_is_greedy(
    tmp_path,
):
    train_config = ExperimentConfig(
        model_variant="M1",
        training_mode=True,
        seed=7,
    )

    train_scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=train_config,
    )

    checkpoint_dir = (
        tmp_path
        / "m1_checkpoint"
    )

    train_scheduler.agent.save_checkpoint(
        checkpoint_dir,
        metadata={
            "model_variant": "M1",
            "seed": 7,
        },
    )

    eval_config = ExperimentConfig(
        model_variant="M1",
        training_mode=False,
        seed=7,
        checkpoint_path=str(
            checkpoint_dir
        ),
    )

    eval_scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=eval_config,
    )

    assert (
        eval_scheduler.agent.evaluation_mode
        is True
    )

    assert (
        eval_scheduler.agent.epsilon
        == 0.0
    )

    assert (
        eval_scheduler.loaded_checkpoint_metadata[
            "model_variant"
        ]
        == "M1"
    )

    state = np.zeros(
        eval_scheduler.model_definition[
            "input_dim"
        ],
        dtype=np.float32,
    )

    actions = [
        eval_scheduler.select_policy_action(
            state
        )
        for _ in range(10)
    ]

    assert (
        len(
            set(actions)
        )
        == 1
    )


def test_evaluation_mode_does_not_require_exploration(
    tmp_path,
):
    train_config = ExperimentConfig(
        model_variant="M1",
        training_mode=True,
        seed=7,
    )

    train_scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=train_config,
    )

    checkpoint_dir = (
        tmp_path
        / "m1_checkpoint"
    )

    train_scheduler.agent.save_checkpoint(
        checkpoint_dir
    )

    eval_config = ExperimentConfig(
        model_variant="M1",
        training_mode=False,
        seed=7,
        checkpoint_path=str(
            checkpoint_dir
        ),
    )

    eval_scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=eval_config,
    )

    state = np.zeros(
        eval_scheduler.model_definition[
            "input_dim"
        ],
        dtype=np.float32,
    )

    action_with_explore_true = (
        eval_scheduler.agent.select_action(
            state,
            explore=True,
        )
    )

    action_with_explore_false = (
        eval_scheduler.agent.select_action(
            state,
            explore=False,
        )
    )

    assert (
        action_with_explore_true
        == action_with_explore_false
    )

    assert (
        eval_scheduler.agent.evaluation_mode
        is True
    )

    assert (
        eval_scheduler.agent.epsilon
        == 0.0
    )


def test_evaluation_mode_blocks_replay_and_training(
    tmp_path,
):
    train_config = ExperimentConfig(
        model_variant="M1",
        training_mode=True,
        seed=7,
    )

    train_scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=train_config,
    )

    checkpoint_dir = (
        tmp_path
        / "m1_checkpoint"
    )

    train_scheduler.agent.save_checkpoint(
        checkpoint_dir
    )

    eval_config = ExperimentConfig(
        model_variant="M1",
        training_mode=False,
        seed=7,
        checkpoint_path=str(
            checkpoint_dir
        ),
    )

    eval_scheduler = AssuredScheduler(
        [10],
        "FR1",
        False,
        False,
        1.0,
        config=eval_config,
    )

    state = np.zeros(
        eval_scheduler.model_definition[
            "input_dim"
        ],
        dtype=np.float32,
    )

    replay_size_before = len(
        eval_scheduler.agent.replay_buffer
    )

    training_steps_before = (
        eval_scheduler.agent.training_steps
    )

    weights_before = [
        weight.copy()
        for weight
        in eval_scheduler.agent.online_model.get_weights()
    ]

    eval_scheduler.agent.remember(
        state,
        0,
        1.0,
        state,
        False,
    )

    result = (
        eval_scheduler.agent.train_step()
    )

    replay_size_after = len(
        eval_scheduler.agent.replay_buffer
    )

    training_steps_after = (
        eval_scheduler.agent.training_steps
    )

    weights_after = (
        eval_scheduler.agent.online_model.get_weights()
    )

    assert result is None

    assert (
        replay_size_after
        == replay_size_before
    )

    assert (
        training_steps_after
        == training_steps_before
    )

    for before, after in zip(
        weights_before,
        weights_after,
    ):
        assert np.array_equal(
            before,
            after,
        )
def make_deadline_scanner_fixture(
    app_packets=None,
    bearer_packets=None,
):
    if app_packets is None:
        app_packets = []

    if bearer_packets is None:
        bearer_packets = []

    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    for packet in app_packets:
        flow.appBuff.insertPckt(
            packet
        )

    bearer = SimpleNamespace(
        buffer=SimpleNamespace(
            pckts=list(
                bearer_packets
            )
        )
    )

    ue = SimpleNamespace(
        packetFlows=[
            flow
        ],
        bearers=[
            bearer
        ],
    )

    urllc_slice = SimpleNamespace(
        schedulerDL=SimpleNamespace(
            ues={
                "ue1": ue
            }
        )
    )

    scheduler = (
        AssuredScheduler.__new__(
            AssuredScheduler
        )
    )

    scheduler._canonical_slices = (
        lambda: {
            "URLLC": urllc_slice
        }
    )

    return (
        scheduler,
        flow,
        bearer,
    )
def test_deadline_scanner_detects_app_buffer_packet():
    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    scheduler, flow, _ = (
        make_deadline_scanner_fixture(
            app_packets=[
                packet
            ]
        )
    )

    flow.generatedPacketIds.add(
        1
    )

    flow.packetGenerationTimes[
        1
    ] = 10.0

    newly_evaluated = (
        scheduler.evaluate_deadline_crossings(
            11.1
        )
    )

    assert newly_evaluated == 1
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1
    assert packet.deadline_missed is True


def test_deadline_scanner_detects_bearer_buffer_packet():
    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    scheduler, flow, _ = (
        make_deadline_scanner_fixture(
            bearer_packets=[
                packet
            ]
        )
    )

    flow.generatedPacketIds.add(
        1
    )

    flow.packetGenerationTimes[
        1
    ] = 10.0

    newly_evaluated = (
        scheduler.evaluate_deadline_crossings(
            11.1
        )
    )

    assert newly_evaluated == 1
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1

def make_starvation_test_scheduler():
    scheduler = AssuredScheduler.__new__(
        AssuredScheduler
    )

    scheduler.granularity = 1.0
    scheduler.starvation_threshold_ms = 10.0

    scheduler.embb_starvation_ms = 0.0
    scheduler.mmtc_starvation_ms = 0.0

    scheduler.embb_starvation_tracker = (
        StarvationTracker(
            threshold_ms=10.0
        )
    )

    scheduler.mmtc_starvation_tracker = (
        StarvationTracker(
            threshold_ms=10.0
        )
    )

    return scheduler

def test_starvation_uses_prb_delta_not_delivery():
    scheduler = (
        make_starvation_test_scheduler()
    )

    before = {
        "eMBB": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            delivered_bytes=100,
            backlog=5,
            scheduled_prbs=10,
        ),
        "mMTC": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            delivered_bytes=0,
            backlog=0,
            scheduled_prbs=20,
        ),
    }

    after = {
        "eMBB": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            # No successful delivery.
            delivered_bytes=100,
            backlog=5,
            # Four PRBs were nevertheless used.
            scheduled_prbs=14,
        ),
        "mMTC": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            delivered_bytes=0,
            backlog=0,
            scheduled_prbs=20,
        ),
    }

    penalty = scheduler.update_starvation(
        before,
        after,
    )

    assert scheduler.embb_starvation_ms == 0.0
    assert scheduler.embb_starvation_tracker.total_ms == 0.0
    assert penalty == 0.0

def test_equal_cumulative_prbs_means_zero_interval_service():
    scheduler = (
        make_starvation_test_scheduler()
    )

    before = {
        "eMBB": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            delivered_bytes=100,
            backlog=5,
            scheduled_prbs=10,
        ),
        "mMTC": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            delivered_bytes=0,
            backlog=0,
            scheduled_prbs=20,
        ),
    }

    after = {
        "eMBB": SliceCounters(
            generated_packets=0,
            delivered_packets=1,
            # Delivery may complete from an earlier
            # scheduling decision.
            delivered_bytes=500,
            backlog=5,
            # No new PRB use in this interval.
            scheduled_prbs=10,
        ),
        "mMTC": SliceCounters(
            generated_packets=0,
            delivered_packets=0,
            delivered_bytes=0,
            backlog=0,
            scheduled_prbs=20,
        ),
    }

    penalty = scheduler.update_starvation(
        before,
        after,
    )

    assert scheduler.embb_starvation_ms == 1.0
    assert scheduler.embb_starvation_tracker.total_ms == 1.0
    assert penalty == 0.1

def test_deadline_scanner_deduplicates_same_packet():
    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    scheduler, flow, _ = (
        make_deadline_scanner_fixture(
            app_packets=[
                packet
            ],
            bearer_packets=[
                packet
            ],
        )
    )

    flow.generatedPacketIds.add(
        1
    )

    flow.packetGenerationTimes[
        1
    ] = 10.0

    newly_evaluated = (
        scheduler.evaluate_deadline_crossings(
            11.1
        )
    )

    assert newly_evaluated == 1
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1

    # Repeated scans must also be idempotent.
    second = (
        scheduler.evaluate_deadline_crossings(
            12.0
        )
    )

    assert second == 0
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1


def test_deadline_scanner_does_not_reclassify_scheduled_packet():
    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    scheduler, flow, bearer = (
        make_deadline_scanner_fixture()
    )

    flow.generatedPacketIds.add(
        1
    )

    flow.packetGenerationTimes[
        1
    ] = 10.0

    flow.recordSchedulingOutcome(
        packet,
        10.5,
    )

    bearer.buffer.pckts.append(
        packet
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 0

    newly_evaluated = (
        scheduler.evaluate_deadline_crossings(
            12.0
        )
    )

    assert newly_evaluated == 0
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 0
    assert packet.deadline_missed is False
