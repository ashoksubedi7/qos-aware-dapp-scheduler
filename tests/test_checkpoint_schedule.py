import pytest

from experiments.checkpoint_schedule import (
    CHECKPOINT_STEP_BASIS,
    EXACT_TIE_BREAK_POLICY,
    CheckpointCandidate,
    candidate_checkpoint_steps,
    select_best_checkpoint,
)
from experiments.checkpoint_selection import (
    CheckpointScore,
)


def score(
    deadline=0.01,
    drop=0.01,
    starvation_max=1.0,
    starvation_total=2.0,
    embb=20.0,
    mmtc_completion=0.95,
    fairness=0.95,
    utilization=0.8,
):
    return CheckpointScore(
        urllc_deadline_miss_ratio=deadline,
        urllc_true_drop_ratio=drop,
        worst_starvation_max_ms=(
            starvation_max
        ),
        combined_starvation_total_ms=(
            starvation_total
        ),
        embb_active_throughput_mbps=embb,
        mmtc_completion_ratio=(
            mmtc_completion
        ),
        minimum_non_urllc_fairness=(
            fairness
        ),
        prb_utilization=utilization,
    )


def test_checkpoint_clock_is_successful_updates():
    assert CHECKPOINT_STEP_BASIS == (
        "successful_gradient_update"
    )


def test_exact_tie_uses_earliest_step():
    assert EXACT_TIE_BREAK_POLICY == (
        "earliest_training_step"
    )

    shared = score()

    later = CheckpointCandidate(
        training_step=20_000,
        score=shared,
    )

    earlier = CheckpointCandidate(
        training_step=10_000,
        score=shared,
    )

    selected = select_best_checkpoint(
        [later, earlier]
    )

    assert selected.training_step == 10_000


def test_better_qos_beats_earlier_checkpoint():
    earlier = CheckpointCandidate(
        training_step=1_000,
        score=score(
            deadline=0.02
        ),
    )

    later = CheckpointCandidate(
        training_step=2_000,
        score=score(
            deadline=0.01
        ),
    )

    selected = select_best_checkpoint(
        [earlier, later]
    )

    assert selected.training_step == 2_000


def test_regular_checkpoint_schedule():
    assert candidate_checkpoint_steps(
        update_budget=10_000,
        interval_updates=2_000,
    ) == (
        2_000,
        4_000,
        6_000,
        8_000,
        10_000,
    )


def test_final_budget_is_always_included():
    assert candidate_checkpoint_steps(
        update_budget=10_500,
        interval_updates=2_000,
    ) == (
        2_000,
        4_000,
        6_000,
        8_000,
        10_000,
        10_500,
    )


def test_interval_larger_than_budget_still_checks_final():
    assert candidate_checkpoint_steps(
        update_budget=1_000,
        interval_updates=2_000,
    ) == (
        1_000,
    )


def test_invalid_budget_fails():
    with pytest.raises(
        ValueError
    ):
        candidate_checkpoint_steps(
            update_budget=0,
            interval_updates=100,
        )


def test_invalid_interval_fails():
    with pytest.raises(
        ValueError
    ):
        candidate_checkpoint_steps(
            update_budget=1_000,
            interval_updates=0,
        )


def test_duplicate_checkpoint_step_fails():
    shared = score()

    with pytest.raises(
        ValueError
    ):
        select_best_checkpoint(
            [
                CheckpointCandidate(
                    training_step=1_000,
                    score=shared,
                ),
                CheckpointCandidate(
                    training_step=1_000,
                    score=shared,
                ),
            ]
        )


def test_nonpositive_checkpoint_step_fails():
    with pytest.raises(
        ValueError
    ):
        select_best_checkpoint(
            [
                CheckpointCandidate(
                    training_step=0,
                    score=score(),
                )
            ]
        )


def test_input_order_does_not_change_selection():
    a = CheckpointCandidate(
        training_step=1_000,
        score=score(
            deadline=0.02
        ),
    )

    b = CheckpointCandidate(
        training_step=2_000,
        score=score(
            deadline=0.01
        ),
    )

    first = select_best_checkpoint(
        [a, b]
    )

    second = select_best_checkpoint(
        [b, a]
    )

    assert first == second
    assert first == b
