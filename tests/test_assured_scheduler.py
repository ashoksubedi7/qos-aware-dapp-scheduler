import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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
