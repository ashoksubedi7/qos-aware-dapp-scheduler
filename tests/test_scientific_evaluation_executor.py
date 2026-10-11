from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from config.experiment_config import ExperimentConfig
from experiments import scientific_evaluation_executor as evaluation


class Value:
    def __init__(self, data):
        self.data = np.array(data)

    def numpy(self):
        return self.data.copy()


class FakeOptimizer:
    def __init__(self):
        self.iterations = Value(2)
        self.slot = Value([0.25, 0.5])
        self.variables = [
            self.iterations,
            self.slot,
        ]


class FakeModel:
    def __init__(self):
        self.weights = [
            np.array([1.0, 2.0], dtype=np.float32)
        ]
        self.optimizer = FakeOptimizer()

    def get_weights(self):
        return [
            weight.copy()
            for weight in self.weights
        ]


def make_agent():
    return SimpleNamespace(
        training_steps=2,
        epsilon=0.05,
        replay_buffer=[
            (1, 2, 3, 4, False)
        ],
        online_model=FakeModel(),
        target_model=FakeModel(),
    )


def make_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    (checkpoint / "metadata.json").write_text(
        '{"model_variant": "M1"}',
        encoding="utf-8",
    )

    return ExperimentConfig(
        training_mode=False,
        checkpoint_path=str(checkpoint),
    )


def test_rejects_non_experiment_config(tmp_path):
    with pytest.raises(TypeError):
        evaluation.run_checkpoint_evaluation(
            object(),
            10.0,
            tmp_path / "run",
        )


def test_rejects_training_mode(tmp_path):
    with pytest.raises(ValueError, match="training_mode"):
        evaluation.run_checkpoint_evaluation(
            ExperimentConfig(),
            10.0,
            tmp_path / "run",
        )

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "duration",
    [0, -1, float("nan"), float("inf")],
)
def test_rejects_invalid_duration(tmp_path, duration):
    config = make_config(tmp_path)
    output = tmp_path / "run"

    with pytest.raises(ValueError):
        evaluation.run_checkpoint_evaluation(
            config,
            duration,
            output,
        )

    assert not output.exists()


def test_rejects_missing_checkpoint(tmp_path):
    config = ExperimentConfig(
        training_mode=False,
        checkpoint_path=str(tmp_path / "missing"),
    )
    output = tmp_path / "run"

    with pytest.raises(FileNotFoundError):
        evaluation.run_checkpoint_evaluation(
            config,
            10.0,
            output,
        )

    assert not output.exists()


def test_rejects_existing_output_directory(tmp_path):
    config = make_config(tmp_path)
    output = tmp_path / "run"
    output.mkdir()

    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        evaluation.run_checkpoint_evaluation(
            config,
            10.0,
            output,
        )

    assert sentinel.read_text(
        encoding="utf-8"
    ) == "preserve"


def test_restores_working_directory_on_startup_error(
    tmp_path,
    monkeypatch,
):
    config = make_config(tmp_path)
    output = tmp_path / "run"
    original = Path.cwd()

    def fail_cell(*args, **kwargs):
        raise RuntimeError("simulator startup failed")

    monkeypatch.setattr(
        evaluation,
        "Cell",
        fail_cell,
    )

    with pytest.raises(
        RuntimeError,
        match="simulator startup failed",
    ):
        evaluation.run_checkpoint_evaluation(
            config,
            10.0,
            output,
        )

    assert Path.cwd() == original
    assert (output / "runtime").is_dir()


def test_unchanged_learning_state_is_accepted():
    agent = make_agent()

    before = evaluation._learning_snapshot(agent)
    after = evaluation._learning_snapshot(agent)

    evaluation._assert_learning_unchanged(
        before,
        after,
    )


@pytest.mark.parametrize(
    "field",
    [
        "training_steps",
        "epsilon",
        "replay_count",
        "optimizer_steps",
    ],
)
def test_detects_scalar_state_changes(field):
    agent = make_agent()

    before = evaluation._learning_snapshot(agent)
    after = dict(before)

    after[field] = after[field] + 1

    with pytest.raises(RuntimeError):
        evaluation._assert_learning_unchanged(
            before,
            after,
        )


@pytest.mark.parametrize(
    "network",
    ["online_model", "target_model"],
)
def test_detects_network_weight_changes(network):
    agent = make_agent()

    before = evaluation._learning_snapshot(agent)

    model = getattr(agent, network)
    model.weights[0][0] += 1.0

    after = evaluation._learning_snapshot(agent)

    with pytest.raises(RuntimeError):
        evaluation._assert_learning_unchanged(
            before,
            after,
        )


def test_detects_same_length_replay_mutation():
    agent = make_agent()

    before = evaluation._learning_snapshot(agent)

    agent.replay_buffer[0] = (
        9, 8, 7, 6, False
    )

    after = evaluation._learning_snapshot(agent)

    with pytest.raises(RuntimeError, match="replay"):
        evaluation._assert_learning_unchanged(
            before,
            after,
        )


def test_detects_optimizer_slot_mutation():
    agent = make_agent()

    before = evaluation._learning_snapshot(agent)

    agent.online_model.optimizer.slot.data[0] += 1.0

    after = evaluation._learning_snapshot(agent)

    with pytest.raises(RuntimeError, match="optimizer"):
        evaluation._assert_learning_unchanged(
            before,
            after,
        )


def test_rejects_checkpoint_model_variant_mismatch(
    tmp_path,
    monkeypatch,
):
    import json

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    metadata = {
        "model_variant": "M2",
        "training_step": 500,
        "training_steps": 500,
    }

    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    config = ExperimentConfig(
        model_variant="M3",
        training_mode=False,
        checkpoint_path=str(checkpoint),
    )

    def unexpected_simulator_start(*args, **kwargs):
        raise AssertionError(
            "Simulator started with a mismatched checkpoint"
        )

    monkeypatch.setattr(
        evaluation,
        "Cell",
        unexpected_simulator_start,
    )

    output = tmp_path / "evaluation"

    with pytest.raises(
        ValueError,
        match="model.variant",
    ):
        evaluation.run_checkpoint_evaluation(
            config=config,
            active_duration_ms=10.0,
            run_directory=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "metadata_contents",
    [None, "not valid json", "[]", "{}"],
)
def test_rejects_missing_or_invalid_checkpoint_metadata(
    tmp_path,
    metadata_contents,
):
    config = make_config(tmp_path)
    metadata_path = (
        Path(config.checkpoint_path) / "metadata.json"
    )

    if metadata_contents is None:
        metadata_path.unlink()
    else:
        metadata_path.write_text(
            metadata_contents,
            encoding="utf-8",
        )

    output = tmp_path / "evaluation"

    with pytest.raises(
        ValueError,
        match="metadata|model.variant",
    ):
        evaluation.run_checkpoint_evaluation(
            config=config,
            active_duration_ms=10.0,
            run_directory=output,
        )

    assert not output.exists()
