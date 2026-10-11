from pathlib import Path

from config.training_protocol import TrainingProtocol
from experiments.checkpoint_schedule import (
    candidate_checkpoint_steps,
)


class TrainingUpdateController:
    """
    Handle successful optimizer-update boundaries for
    one scientific training run.

    Checkpoints are captured on the successful-gradient
    update clock. Learning is frozen exactly at the
    prospectively frozen final update budget.
    """

    def __init__(
        self,
        protocol,
        checkpoint_root,
        base_checkpoint_metadata=None,
    ):
        if not isinstance(
            protocol,
            TrainingProtocol,
        ):
            raise TypeError(
                "protocol must be a TrainingProtocol"
            )

        protocol.assert_ready_for_scientific_training()

        self.protocol = protocol
        self.checkpoint_root = Path(
            checkpoint_root
        )

        if base_checkpoint_metadata is None:
            base_checkpoint_metadata = {}

        self.base_checkpoint_metadata = dict(
            base_checkpoint_metadata
        )

        controller_owned_metadata_keys = {
            "training_step",
            "training_loss",
            "model_variant",
            "seed",
            "step_basis",
        }

        conflicting_keys = (
            controller_owned_metadata_keys
            .intersection(
                self.base_checkpoint_metadata
            )
        )

        if conflicting_keys:
            raise ValueError(
                "base checkpoint metadata cannot "
                "override controller-owned fields: "
                + ", ".join(
                    sorted(conflicting_keys)
                )
            )

        self._candidate_steps = set(
            candidate_checkpoint_steps(
                protocol.successful_gradient_update_budget,
                protocol.checkpoint_interval_updates,
            )
        )

        self._observed_steps = []
        self._saved_checkpoint_steps = []

    @property
    def observed_steps(self):
        return tuple(
            self._observed_steps
        )

    @property
    def saved_checkpoint_steps(self):
        return tuple(
            self._saved_checkpoint_steps
        )

    def __call__(
        self,
        scheduler,
        training_step,
        training_loss,
    ):
        if (
            not isinstance(training_step, int)
            or isinstance(training_step, bool)
        ):
            raise TypeError(
                "training_step must be a "
                "non-Boolean integer"
            )

        budget = (
            self.protocol
            .successful_gradient_update_budget
        )

        if training_step > budget:
            raise RuntimeError(
                "successful update exceeded "
                "the frozen training budget"
            )

        expected_step = (
            len(self._observed_steps) + 1
        )

        if training_step != expected_step:
            raise RuntimeError(
                "successful updates must be "
                "observed sequentially"
            )

        agent_step = int(
            scheduler.agent.training_steps
        )

        if agent_step != training_step:
            raise RuntimeError(
                "callback training step does not "
                "match the agent update clock"
            )

        self._observed_steps.append(
            training_step
        )

        if training_step in self._candidate_steps:
            checkpoint_directory = (
                self.checkpoint_root
                / (
                    "update_"
                    f"{training_step:08d}"
                )
            )

            checkpoint_metadata = dict(
                self.base_checkpoint_metadata
            )

            checkpoint_metadata.update(
                {
                    "training_step": training_step,
                    "training_loss": float(
                        training_loss
                    ),
                    "model_variant": (
                        scheduler.config.model_variant
                    ),
                    "seed": int(
                        scheduler.config.seed
                    ),
                    "step_basis": (
                        "successful_gradient_update"
                    ),
                }
            )

            scheduler.agent.save_checkpoint(
                checkpoint_directory,
                metadata=checkpoint_metadata,
            )

            self._saved_checkpoint_steps.append(
                training_step
            )

        if training_step == budget:
            scheduler.freeze_learning()
