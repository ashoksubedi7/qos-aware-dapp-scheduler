import csv
import hashlib
import json

import numpy as np
import pytest

from experiments.scientific_training_recorder import (
    ScientificTrainingRecorder,
)


def make_recorder(tmp_path):
    return ScientificTrainingRecorder(
        run_directory=tmp_path / "run",
        model_variant="M1",
        seed=901,
        flush_every=2,
    )


def add_transition(recorder, index):
    recorder.record_transition(
        state=np.asarray(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
        action=index % 22,
        reward=0.5,
        next_state=np.asarray(
            [0.2, 0.3, 0.4],
            dtype=np.float32,
        ),
        done=False,
        training_step=0,
        epsilon=1.0,
    )


def test_persists_every_transition(tmp_path):
    recorder = make_recorder(tmp_path)

    for index in range(5):
        add_transition(recorder, index)

    recorder.finish(success=True)

    path = (
        tmp_path
        / "run"
        / "training_data"
        / "transitions.jsonl"
    )

    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
    ]

    assert len(records) == 5

    assert [
        row["index"] for row in records
    ] == [1, 2, 3, 4, 5]

    assert records[0]["action"] == 0
    assert records[4]["action"] == 4
    assert len(records[0]["state"]) == 3
    assert records[0]["done"] is False


def test_persists_training_loss_and_epsilon(tmp_path):
    recorder = make_recorder(tmp_path)

    recorder.record_update(
        training_step=1,
        training_loss=0.25,
        epsilon=0.9,
    )

    recorder.record_update(
        training_step=2,
        training_loss=0.125,
        epsilon=0.8,
    )

    recorder.finish(success=True)

    path = (
        tmp_path
        / "run"
        / "training_data"
        / "training_metrics.csv"
    )

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2

    assert [
        int(row["training_step"])
        for row in rows
    ] == [1, 2]

    assert float(rows[0]["training_loss"]) == 0.25
    assert float(rows[1]["epsilon"]) == 0.8


def test_completed_manifest_contains_real_hashes(tmp_path):
    recorder = make_recorder(tmp_path)

    add_transition(recorder, 0)

    recorder.record_update(
        training_step=1,
        training_loss=0.25,
        epsilon=0.9,
    )

    recorder.finish(success=True)

    directory = (
        tmp_path / "run" / "training_data"
    )

    manifest = json.loads(
        (directory / "recording_manifest.json").read_text()
    )

    assert manifest["status"] == "complete"
    assert manifest["model_variant"] == "M1"
    assert manifest["seed"] == 901
    assert manifest["transition_count"] == 1
    assert manifest["update_count"] == 1

    for filename, key in (
        ("transitions.jsonl", "transitions_sha256"),
        ("training_metrics.csv", "training_metrics_sha256"),
    ):
        digest = hashlib.sha256(
            (directory / filename).read_bytes()
        ).hexdigest()

        assert manifest[key] == digest


def test_interrupted_recording_is_not_complete(tmp_path):
    recorder = make_recorder(tmp_path)

    add_transition(recorder, 0)
    recorder.finish(success=False)

    directory = (
        tmp_path / "run" / "training_data"
    )

    manifest = json.loads(
        (directory / "recording_manifest.json").read_text()
    )

    assert manifest["status"] == "interrupted"
    assert manifest["transition_count"] == 1


def test_existing_recording_directory_is_not_overwritten(
    tmp_path,
):
    recorder = make_recorder(tmp_path)
    recorder.finish(success=True)

    with pytest.raises(FileExistsError):
        make_recorder(tmp_path)


def test_nonfinite_reward_is_rejected(tmp_path):
    recorder = make_recorder(tmp_path)

    with pytest.raises(ValueError):
        recorder.record_transition(
            state=[0.1, 0.2],
            action=0,
            reward=float("nan"),
            next_state=[0.2, 0.3],
            done=False,
            training_step=0,
            epsilon=1.0,
        )

    recorder.finish(success=False)
