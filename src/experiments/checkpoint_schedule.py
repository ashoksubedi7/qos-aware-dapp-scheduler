from dataclasses import dataclass

from experiments.checkpoint_selection import (
    CheckpointScore,
)


CHECKPOINT_STEP_BASIS = (
    "successful_gradient_update"
)

EXACT_TIE_BREAK_POLICY = (
    "earliest_training_step"
)

FINAL_CHECKPOINT_POLICY = (
    "always_include_final_update_budget"
)


@dataclass(frozen=True)
class CheckpointCandidate:
    training_step: int
    score: CheckpointScore

    def selection_key(self):
        """
        Lower key is better.

        The QoS-first lexicographic score dominates.
        Training step is consulted only when every
        validation metric is exactly tied.
        """
        if (
            not isinstance(self.training_step, int)
            or isinstance(self.training_step, bool)
        ):
            raise TypeError(
                "checkpoint training step must be a "
                "non-Boolean integer"
            )

        return (
            self.score.lexicographic_key()
            +
            (self.training_step,)
        )


def candidate_checkpoint_steps(
    update_budget,
    interval_updates,
):
    """
    Return deterministic checkpoint-validation steps.

    The clock is successful gradient updates.

    Regular checkpoints occur at positive multiples
    of interval_updates. The final training budget is
    always included even when it is not an exact
    interval multiple.
    """

    if (
        not isinstance(update_budget, int)
        or isinstance(update_budget, bool)
    ):
        raise TypeError(
            "update budget must be a "
            "non-Boolean integer"
        )

    if (
        not isinstance(interval_updates, int)
        or isinstance(interval_updates, bool)
    ):
        raise TypeError(
            "checkpoint interval must be a "
            "non-Boolean integer"
        )

    if update_budget <= 0:
        raise ValueError(
            "update budget must be positive"
        )

    if interval_updates <= 0:
        raise ValueError(
            "checkpoint interval must be positive"
        )

    steps = list(
        range(
            interval_updates,
            update_budget + 1,
            interval_updates,
        )
    )

    if (
        not steps
        or steps[-1] != update_budget
    ):
        steps.append(
            update_budget
        )

    return tuple(steps)


def select_best_checkpoint(
    candidates,
):
    """
    Select the best candidate using the frozen
    QoS-first score and earliest-step exact tie break.
    """

    candidates = tuple(
        candidates
    )

    if not candidates:
        raise ValueError(
            "checkpoint candidate set cannot be empty"
        )

    observed_steps = set()

    for candidate in candidates:
        if not isinstance(
            candidate,
            CheckpointCandidate,
        ):
            raise TypeError(
                "all candidates must be "
                "CheckpointCandidate instances"
            )

        step = candidate.training_step

        if (
            not isinstance(step, int)
            or isinstance(step, bool)
        ):
            raise TypeError(
                "checkpoint training step must be a "
                "non-Boolean integer"
            )

        if step <= 0:
            raise ValueError(
                "checkpoint training step "
                "must be positive"
            )

        if step in observed_steps:
            raise ValueError(
                "duplicate checkpoint "
                "training step"
            )

        observed_steps.add(
            step
        )

    return min(
        candidates,
        key=lambda candidate:
            candidate.selection_key(),
    )
