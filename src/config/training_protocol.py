from dataclasses import dataclass

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


@dataclass(frozen=True)
class TrainingProtocol:
    """
    Methodological controls for scientific
    AssuredQoS DQN training.

    The checkpoint-selection methodology is frozen
    structurally. Numerical training-budget decisions
    remain unset until selected prospectively.
    """

    # Replay warm-up has one numerical authority:
    # DQNHyperparameters.min_replay_size.
    replay_warmup_authority: str = (
        "dqn_hyperparameters.min_replay_size"
    )

    # Exploration changes only after an actual
    # optimizer update succeeds.
    epsilon_decay_basis: str = (
        "successful_gradient_update"
    )

    # Candidate-checkpoint validation and final
    # held-out evaluation both use frozen greedy
    # policies.
    checkpoint_validation_epsilon: float = 0.0
    final_evaluation_epsilon: float = 0.0

    # Validation is deliberately isolated from the
    # ongoing training process so it cannot mutate
    # replay memory, optimizer state, RNG state, or
    # network weights.
    checkpoint_validation_execution: str = (
        "fresh_process_frozen_checkpoint"
    )

    checkpoint_validation_allows_training: bool = False

    # Independent scientific training runs must not
    # share learning state.
    reuse_replay_between_runs: bool = False
    restore_optimizer_state: bool = False
    resume_training_from_checkpoint: bool = False

    # Checkpoints are used for validation-based model
    # selection and subsequent frozen-policy evaluation,
    # not as complete resumable-training snapshots.
    checkpoint_usage: str = (
        "validation_selection_and_"
        "frozen_policy_evaluation"
    )

    # Validated reproducibility execution path.
    execution_device: str = "cpu"
    require_process_pythonhashseed: bool = True
    require_deterministic_tensorflow_ops: bool = True

    # -------------------------------
    # Frozen checkpoint methodology.
    # -------------------------------

    checkpoint_selection_policy: str = (
        "qos_first_lexicographic"
    )

    checkpoint_selection_metrics: tuple[str, ...] = (
        CHECKPOINT_SELECTION_METRICS
    )

    validation_grid_policy: str = (
        VALIDATION_GRID_POLICY
    )

    validation_aggregation_policy: str = (
        VALIDATION_AGGREGATION_POLICY
    )

    checkpoint_step_basis: str = (
        CHECKPOINT_STEP_BASIS
    )

    final_checkpoint_policy: str = (
        FINAL_CHECKPOINT_POLICY
    )

    exact_tie_break_policy: str = (
        EXACT_TIE_BREAK_POLICY
    )

    # Final held-out test results must never determine
    # checkpoint selection.
    final_test_used_for_checkpoint_selection: bool = (
        False
    )

    # ---------------------------------------
    # Prospectively unresolved numeric items.
    # ---------------------------------------

    # Total successful optimizer updates in one
    # scientific training run.
    successful_gradient_update_budget: int | None = (
        None
    )

    # Successful optimizer update at which epsilon
    # first reaches epsilon_min.
    epsilon_floor_update: int | None = None

    # Number of successful optimizer updates between
    # validation-checkpoint candidates. The final
    # training-budget checkpoint is always included.
    checkpoint_interval_updates: int | None = None

    def __post_init__(self):
        if (
            self.replay_warmup_authority
            !=
            "dqn_hyperparameters.min_replay_size"
        ):
            raise ValueError(
                "replay warm-up must use "
                "DQNHyperparameters.min_replay_size "
                "as its sole numerical authority"
            )

        if self.epsilon_decay_basis != (
            "successful_gradient_update"
        ):
            raise ValueError(
                "epsilon decay must be based on "
                "successful gradient updates"
            )

        if self.checkpoint_validation_epsilon != 0.0:
            raise ValueError(
                "checkpoint validation must use "
                "greedy epsilon=0"
            )

        if self.final_evaluation_epsilon != 0.0:
            raise ValueError(
                "final scientific evaluation must "
                "use greedy epsilon=0"
            )

        if self.checkpoint_validation_execution != (
            "fresh_process_frozen_checkpoint"
        ):
            raise ValueError(
                "checkpoint validation must use a "
                "fresh frozen-policy process"
            )

        if self.checkpoint_validation_allows_training:
            raise ValueError(
                "checkpoint validation must not "
                "perform training updates"
            )

        if self.reuse_replay_between_runs:
            raise ValueError(
                "scientific runs must not reuse "
                "replay buffers"
            )

        if self.restore_optimizer_state:
            raise ValueError(
                "scientific runs must not restore "
                "optimizer state"
            )

        if self.resume_training_from_checkpoint:
            raise ValueError(
                "scientific training must begin "
                "from a fresh agent rather than "
                "resume from a checkpoint"
            )

        if self.checkpoint_usage != (
            "validation_selection_and_"
            "frozen_policy_evaluation"
        ):
            raise ValueError(
                "unsupported checkpoint usage"
            )

        if self.execution_device != "cpu":
            raise ValueError(
                "the currently validated scientific "
                "training path is CPU-only"
            )

        if not self.require_process_pythonhashseed:
            raise ValueError(
                "PYTHONHASHSEED must be fixed before "
                "the Python process starts"
            )

        if (
            not
            self.require_deterministic_tensorflow_ops
        ):
            raise ValueError(
                "TensorFlow deterministic operations "
                "must be required"
            )

        if self.checkpoint_selection_policy != (
            "qos_first_lexicographic"
        ):
            raise ValueError(
                "checkpoint selection must use the "
                "QoS-first lexicographic policy"
            )

        if (
            self.checkpoint_selection_metrics
            !=
            CHECKPOINT_SELECTION_METRICS
        ):
            raise ValueError(
                "checkpoint-selection metric order "
                "must match the frozen policy"
            )

        if self.validation_grid_policy != (
            VALIDATION_GRID_POLICY
        ):
            raise ValueError(
                "validation-grid policy must match "
                "the frozen policy"
            )

        if self.validation_aggregation_policy != (
            VALIDATION_AGGREGATION_POLICY
        ):
            raise ValueError(
                "validation aggregation must match "
                "the frozen policy"
            )

        if self.checkpoint_step_basis != (
            CHECKPOINT_STEP_BASIS
        ):
            raise ValueError(
                "checkpoint clock must use "
                "successful gradient updates"
            )

        if self.final_checkpoint_policy != (
            FINAL_CHECKPOINT_POLICY
        ):
            raise ValueError(
                "final-checkpoint policy must match "
                "the frozen policy"
            )

        if self.exact_tie_break_policy != (
            EXACT_TIE_BREAK_POLICY
        ):
            raise ValueError(
                "checkpoint tie-break policy must "
                "match the frozen policy"
            )

        if (
            self.final_test_used_for_checkpoint_selection
        ):
            raise ValueError(
                "final held-out test results must "
                "not influence checkpoint selection"
            )

        if (
            self.successful_gradient_update_budget
            is not None
            and
            self.successful_gradient_update_budget
            <= 0
        ):
            raise ValueError(
                "successful gradient-update budget "
                "must be positive"
            )

        if (
            self.epsilon_floor_update
            is not None
            and
            self.epsilon_floor_update <= 0
        ):
            raise ValueError(
                "epsilon-floor update must be "
                "positive"
            )

        if (
            self.checkpoint_interval_updates
            is not None
            and
            self.checkpoint_interval_updates <= 0
        ):
            raise ValueError(
                "checkpoint interval must be "
                "positive"
            )

        if (
            self.successful_gradient_update_budget
            is not None
            and
            self.epsilon_floor_update
            is not None
            and
            self.epsilon_floor_update
            >
            self.successful_gradient_update_budget
        ):
            raise ValueError(
                "epsilon-floor update cannot exceed "
                "the total training-update budget"
            )

    def derive_epsilon_decay(
        self,
        epsilon_start,
        epsilon_min,
    ):
        """
        Derive multiplicative epsilon decay so epsilon
        reaches epsilon_min exactly at the prospectively
        frozen epsilon-floor update.
        """

        if self.epsilon_floor_update is None:
            raise RuntimeError(
                "epsilon-floor update has not been "
                "frozen"
            )

        epsilon_start = float(
            epsilon_start
        )
        epsilon_min = float(
            epsilon_min
        )

        if not (
            0.0
            <
            epsilon_min
            <
            epsilon_start
            <=
            1.0
        ):
            raise ValueError(
                "epsilon values must satisfy "
                "0 < min < start <= 1"
            )

        return (
            epsilon_min
            / epsilon_start
        ) ** (
            1.0
            / float(
                self.epsilon_floor_update
            )
        )

    def unresolved_items(self):
        unresolved = []

        if (
            self.successful_gradient_update_budget
            is None
        ):
            unresolved.append(
                "successful_gradient_update_budget"
            )

        if self.epsilon_floor_update is None:
            unresolved.append(
                "epsilon_floor_update"
            )

        if (
            self.checkpoint_interval_updates
            is None
        ):
            unresolved.append(
                "checkpoint_interval_updates"
            )

        return tuple(
            unresolved
        )

    def assert_ready_for_scientific_training(self):
        unresolved = self.unresolved_items()

        if unresolved:
            raise RuntimeError(
                "training protocol is not frozen; "
                "unresolved: "
                + ", ".join(unresolved)
            )

        return True
