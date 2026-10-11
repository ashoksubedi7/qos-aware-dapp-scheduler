from types import SimpleNamespace

import pytest

from config.training_protocol import TrainingProtocol
from experiments.training_update_controller import (
    TrainingUpdateController,
)


class _ProbeAgent:
    def __init__(self):
        self.training_steps = 0
        self.saved = []

    def save_checkpoint(
        self,
        directory,
        metadata=None,
    ):
        self.saved.append(
            (
                str(directory),
                dict(metadata or {}),
            )
        )

        return {
            "online_weights": (
                str(directory)
                + "/online.weights.h5"
            ),
            "target_weights": (
                str(directory)
                + "/target.weights.h5"
            ),
            "metadata": (
                str(directory)
                + "/metadata.json"
            ),
        }


class _ProbeScheduler:
    def __init__(self):
        self.agent = _ProbeAgent()
        self.learning_enabled = True
        self.freeze_calls = 0
        self.config = SimpleNamespace(
            model_variant="M1",
            seed=101,
        )

    def freeze_learning(self):
        self.freeze_calls += 1
        self.learning_enabled = False


def _protocol():
    return TrainingProtocol(
        successful_gradient_update_budget=5,
        epsilon_floor_update=4,
        checkpoint_interval_updates=2,
    )


def test_controller_saves_only_candidate_checkpoints(
    tmp_path,
):
    controller = TrainingUpdateController(
        protocol=_protocol(),
        checkpoint_root=tmp_path,
    )

    scheduler = _ProbeScheduler()

    for step in range(1, 5):
        scheduler.agent.training_steps = step

        controller(
            scheduler,
            step,
            0.1 * step,
        )

    assert [
        item[1]["training_step"]
        for item in scheduler.agent.saved
    ] == [2, 4]

    assert scheduler.freeze_calls == 0
    assert scheduler.learning_enabled is True


def test_controller_saves_final_checkpoint_and_freezes(
    tmp_path,
):
    controller = TrainingUpdateController(
        protocol=_protocol(),
        checkpoint_root=tmp_path,
    )

    scheduler = _ProbeScheduler()

    for step in range(1, 6):
        scheduler.agent.training_steps = step

        controller(
            scheduler,
            step,
            0.1 * step,
        )

    assert [
        item[1]["training_step"]
        for item in scheduler.agent.saved
    ] == [2, 4, 5]

    assert scheduler.freeze_calls == 1
    assert scheduler.learning_enabled is False

    assert controller.observed_steps == (
        1,
        2,
        3,
        4,
        5,
    )

    assert controller.saved_checkpoint_steps == (
        2,
        4,
        5,
    )


def test_controller_requires_sequential_successful_updates(
    tmp_path,
):
    controller = TrainingUpdateController(
        protocol=_protocol(),
        checkpoint_root=tmp_path,
    )

    scheduler = _ProbeScheduler()

    scheduler.agent.training_steps = 1

    controller(
        scheduler,
        1,
        0.1,
    )

    scheduler.agent.training_steps = 3

    with pytest.raises(RuntimeError):
        controller(
            scheduler,
            3,
            0.3,
        )


def test_controller_requires_agent_clock_to_match_callback(
    tmp_path,
):
    controller = TrainingUpdateController(
        protocol=_protocol(),
        checkpoint_root=tmp_path,
    )

    scheduler = _ProbeScheduler()
    scheduler.agent.training_steps = 2

    with pytest.raises(RuntimeError):
        controller(
            scheduler,
            1,
            0.1,
        )


def test_controller_rejects_updates_after_budget(
    tmp_path,
):
    controller = TrainingUpdateController(
        protocol=_protocol(),
        checkpoint_root=tmp_path,
    )

    scheduler = _ProbeScheduler()

    for step in range(1, 6):
        scheduler.agent.training_steps = step

        controller(
            scheduler,
            step,
            0.1,
        )

    scheduler.agent.training_steps = 6

    with pytest.raises(RuntimeError):
        controller(
            scheduler,
            6,
            0.1,
        )


def test_controller_carries_base_checkpoint_metadata(
    tmp_path,
):
    base_metadata = {
        "source_commit": (
            "0123456789abcdef"
            "0123456789abcdef"
            "01234567"
        ),
        "training_protocol_sha256": "a" * 64,
        "experiment_config_sha256": "b" * 64,
        "run_definition_sha256": "c" * 64,
    }

    controller = TrainingUpdateController(
        protocol=_protocol(),
        checkpoint_root=tmp_path,
        base_checkpoint_metadata=base_metadata,
    )

    # The controller must own a snapshot rather than
    # retaining a mutable caller dictionary.
    base_metadata[
        "run_definition_sha256"
    ] = "d" * 64

    scheduler = _ProbeScheduler()

    for step in range(1, 3):
        scheduler.agent.training_steps = step

        controller(
            scheduler,
            step,
            0.1 * step,
        )

    assert len(
        scheduler.agent.saved
    ) == 1

    saved_metadata = (
        scheduler.agent.saved[0][1]
    )

    assert saved_metadata[
        "source_commit"
    ] == (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    assert saved_metadata[
        "training_protocol_sha256"
    ] == "a" * 64

    assert saved_metadata[
        "experiment_config_sha256"
    ] == "b" * 64

    assert saved_metadata[
        "run_definition_sha256"
    ] == "c" * 64

    assert saved_metadata[
        "training_step"
    ] == 2

    assert saved_metadata[
        "step_basis"
    ] == "successful_gradient_update"


@pytest.mark.parametrize(
    "reserved_key",
    (
        "training_step",
        "training_loss",
        "model_variant",
        "seed",
        "step_basis",
    ),
)
def test_controller_rejects_base_metadata_collisions(
    tmp_path,
    reserved_key,
):
    with pytest.raises(ValueError):
        TrainingUpdateController(
            protocol=_protocol(),
            checkpoint_root=tmp_path,
            base_checkpoint_metadata={
                reserved_key: "invalid",
            },
        )
