import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from config.training_protocol import (
    TrainingProtocol,
)
from experiments.checkpoint_aggregation import (
    VALIDATION_AGGREGATION_POLICY,
    VALIDATION_GRID_POLICY,
)
from experiments.checkpoint_schedule import (
    CHECKPOINT_STEP_BASIS,
    EXACT_TIE_BREAK_POLICY,
    FINAL_CHECKPOINT_POLICY,
)
from experiments.checkpoint_selection import (
    CHECKPOINT_SELECTION_METRICS,
)


def test_default_protocol_is_fail_closed():
    protocol = TrainingProtocol()

    assert protocol.unresolved_items() == (
        "successful_gradient_update_budget",
        "epsilon_floor_update",
        "checkpoint_interval_updates",
    )

    with pytest.raises(
        RuntimeError
    ):
        protocol.assert_ready_for_scientific_training()


def test_replay_warmup_has_single_authority():
    protocol = TrainingProtocol()

    assert (
        protocol.replay_warmup_authority
        ==
        "dqn_hyperparameters.min_replay_size"
    )


def test_scientific_runs_are_fresh():
    protocol = TrainingProtocol()

    assert (
        protocol.reuse_replay_between_runs
        is False
    )

    assert (
        protocol.restore_optimizer_state
        is False
    )

    assert (
        protocol.resume_training_from_checkpoint
        is False
    )


def test_deterministic_cpu_execution_is_required():
    protocol = TrainingProtocol()

    assert protocol.execution_device == "cpu"

    assert (
        protocol.require_process_pythonhashseed
        is True
    )

    assert (
        protocol.require_deterministic_tensorflow_ops
        is True
    )


def test_checkpoint_validation_is_greedy_and_isolated():
    protocol = TrainingProtocol()

    assert (
        protocol.checkpoint_validation_epsilon
        == 0.0
    )

    assert (
        protocol.checkpoint_validation_execution
        ==
        "fresh_process_frozen_checkpoint"
    )

    assert (
        protocol.checkpoint_validation_allows_training
        is False
    )


def test_final_evaluation_is_greedy():
    protocol = TrainingProtocol()

    assert (
        protocol.final_evaluation_epsilon
        == 0.0
    )


def test_exploratory_checkpoint_validation_fails():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            checkpoint_validation_epsilon=0.1
        )


def test_training_during_checkpoint_validation_fails():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            checkpoint_validation_allows_training=True
        )


def test_nonisolated_checkpoint_validation_fails():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            checkpoint_validation_execution=(
                "same_training_process"
            )
        )


def test_checkpoint_usage_is_validation_and_evaluation():
    protocol = TrainingProtocol()

    assert protocol.checkpoint_usage == (
        "validation_selection_and_"
        "frozen_policy_evaluation"
    )


def test_checkpoint_metric_order_is_frozen():
    protocol = TrainingProtocol()

    assert (
        protocol.checkpoint_selection_metrics
        ==
        CHECKPOINT_SELECTION_METRICS
    )


def test_checkpoint_metric_order_cannot_drift():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            checkpoint_selection_metrics=(
                "different_metric",
            )
        )


def test_validation_grid_policy_is_frozen():
    protocol = TrainingProtocol()

    assert (
        protocol.validation_grid_policy
        ==
        VALIDATION_GRID_POLICY
    )


def test_validation_aggregation_policy_is_frozen():
    protocol = TrainingProtocol()

    assert (
        protocol.validation_aggregation_policy
        ==
        VALIDATION_AGGREGATION_POLICY
    )


def test_checkpoint_clock_is_frozen():
    protocol = TrainingProtocol()

    assert (
        protocol.checkpoint_step_basis
        ==
        CHECKPOINT_STEP_BASIS
    )


def test_final_checkpoint_policy_is_frozen():
    protocol = TrainingProtocol()

    assert (
        protocol.final_checkpoint_policy
        ==
        FINAL_CHECKPOINT_POLICY
    )


def test_tie_break_policy_is_frozen():
    protocol = TrainingProtocol()

    assert (
        protocol.exact_tie_break_policy
        ==
        EXACT_TIE_BREAK_POLICY
    )


def test_final_test_cannot_select_checkpoint():
    protocol = TrainingProtocol()

    assert (
        protocol
        .final_test_used_for_checkpoint_selection
        is False
    )

    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            final_test_used_for_checkpoint_selection=(
                True
            )
        )


def test_positive_budget_required():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            successful_gradient_update_budget=0
        )


def test_positive_epsilon_floor_required():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            epsilon_floor_update=0
        )


def test_positive_checkpoint_interval_required():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            checkpoint_interval_updates=0
        )


def test_epsilon_floor_cannot_exceed_budget():
    with pytest.raises(
        ValueError
    ):
        TrainingProtocol(
            successful_gradient_update_budget=10_000,
            epsilon_floor_update=10_001,
        )


def test_epsilon_decay_is_derived_from_horizon():
    protocol = TrainingProtocol(
        successful_gradient_update_budget=50_000,
        epsilon_floor_update=50_000,
        checkpoint_interval_updates=5_000,
    )

    decay = protocol.derive_epsilon_decay(
        epsilon_start=1.0,
        epsilon_min=0.05,
    )

    assert decay == pytest.approx(
        0.999940087,
        rel=1e-8,
    )

    reached = (
        1.0
        * decay ** 50_000
    )

    assert reached == pytest.approx(
        0.05,
        rel=1e-10,
    )


def test_derivation_requires_frozen_horizon():
    protocol = TrainingProtocol()

    with pytest.raises(
        RuntimeError
    ):
        protocol.derive_epsilon_decay(
            epsilon_start=1.0,
            epsilon_min=0.05,
        )


def test_fully_specified_protocol_can_be_ready():
    # Numerical values here are test fixtures only.
    # They do not freeze the scientific budget.
    protocol = TrainingProtocol(
        successful_gradient_update_budget=25_000,
        epsilon_floor_update=20_000,
        checkpoint_interval_updates=5_000,
    )

    assert (
        protocol.assert_ready_for_scientific_training()
        is True
    )


def test_invalid_epsilon_derivation_range_fails():
    protocol = TrainingProtocol(
        successful_gradient_update_budget=100,
        epsilon_floor_update=100,
        checkpoint_interval_updates=25,
    )

    with pytest.raises(
        ValueError
    ):
        protocol.derive_epsilon_decay(
            epsilon_start=0.05,
            epsilon_min=0.05,
        )
