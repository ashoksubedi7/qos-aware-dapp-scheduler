import json

import pytest

from config.experiment_config import ExperimentConfig
from config.training_protocol import TrainingProtocol
from experiments.scientific_run_provenance import (
    build_scientific_run_provenance,
)

SOURCE_COMMIT = (
    "841f978577f226bb630382ac632980331ca6a5c1"
)


@pytest.fixture
def checkpoint(tmp_path):
    protocol = TrainingProtocol(
        successful_gradient_update_budget=2,
        epsilon_floor_update=2,
        checkpoint_interval_updates=1,
    )

    config = ExperimentConfig(
        model_variant="M1",
        seed=7,
        training_mode=True,
    )

    manifest = build_scientific_run_provenance(
        protocol=protocol,
        config=config,
        source_commit=SOURCE_COMMIT,
    )

    run = tmp_path / "training"
    run.mkdir()

    (run / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    folder = run / "checkpoints" / "update_00000002"
    folder.mkdir(parents=True)

    metadata = {
        "model_variant": "M1",
        "seed": 7,
        "training_step": 2,
        "training_steps": 2,
        "source_commit": manifest["source_commit"],
        "training_protocol_sha256": (
            manifest["training_protocol_sha256"]
        ),
        "experiment_config_sha256": (
            manifest["experiment_config_sha256"]
        ),
        "run_definition_sha256": (
            manifest["run_definition_sha256"]
        ),
    }

    (folder / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    # Synthetic files for provenance tests only.
    # These are not intended for model loading.
    (folder / "online.weights.h5").write_bytes(b"online")
    (folder / "target.weights.h5").write_bytes(b"target")

    return folder


def validate(checkpoint):
    from experiments.scientific_evaluation_provenance import (
        validate_checkpoint_provenance,
    )

    return validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=SOURCE_COMMIT,
    )


def test_accepts_matching_training_provenance(checkpoint):
    result = validate(checkpoint)
    assert result["source_commit"] == SOURCE_COMMIT


def test_rejects_different_training_source(checkpoint):
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    metadata["source_commit"] = "a" * 40
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_commit"):
        validate(checkpoint)


def test_rejects_model_identity_mismatch(checkpoint):
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    metadata["model_variant"] = "M3"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_variant"):
        validate(checkpoint)


def test_rejects_tampered_training_manifest(checkpoint):
    manifest_path = (
        checkpoint.parent.parent / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())

    manifest["experiment_config"]["seed"] = 99

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash|digest"):
        validate(checkpoint)


def test_rejects_different_active_source(checkpoint):
    from experiments.scientific_evaluation_provenance import (
        validate_checkpoint_provenance,
    )

    with pytest.raises(ValueError, match="source_commit"):
        validate_checkpoint_provenance(
            checkpoint_directory=checkpoint,
            active_source_commit="b" * 40,
        )

def test_rejects_checkpoint_seed_mismatch(checkpoint):
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    metadata["seed"] = 99

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seed"):
        validate(checkpoint)


def test_rejects_inconsistent_training_step(checkpoint):
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    metadata["training_steps"] = 3

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training_step"):
        validate(checkpoint)


def test_rejects_wrong_checkpoint_directory_step(checkpoint):
    renamed = checkpoint.with_name("update_00000003")
    checkpoint.rename(renamed)

    with pytest.raises(
        ValueError,
        match="checkpoint directory step",
    ):
        validate(renamed)


@pytest.mark.parametrize(
    "weight_file",
    [
        "online.weights.h5",
        "target.weights.h5",
    ],
)
def test_rejects_missing_weight_file(
    checkpoint,
    weight_file,
):
    (checkpoint / weight_file).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="checkpoint file missing",
    ):
        validate(checkpoint)


def test_rejects_missing_training_manifest(checkpoint):
    manifest_path = (
        checkpoint.parent.parent / "manifest.json"
    )

    manifest_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="required provenance file missing",
    ):
        validate(checkpoint)


def test_rejects_checkpoint_digest_mismatch(checkpoint):
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    metadata["run_definition_sha256"] = "0" * 64

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="run_definition_sha256",
    ):
        validate(checkpoint)
