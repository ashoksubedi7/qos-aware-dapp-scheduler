import hashlib
import json
import re
from pathlib import Path

from experiments.scientific_run_provenance import (
    SCIENTIFIC_RUN_PROVENANCE_SCHEMA_VERSION,
    _sha256_json,
)


_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _read_json_object(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"required provenance file missing: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            f"invalid provenance JSON: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"provenance must be a JSON object: {path}"
        )

    return data


def _file_sha256(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint file missing: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _require_equal(name, actual, expected):
    if actual != expected:
        raise ValueError(
            f"{name} mismatch: "
            f"expected {expected!r}, found {actual!r}"
        )


def validate_checkpoint_provenance(
    checkpoint_directory,
    active_source_commit,
):
    """
    Validate checkpoint identity against its original
    scientific training manifest.

    Weight hashes establish the bytes observed during this
    check. They do not independently authenticate the
    original training-time checkpoint contents.
    """
    if (
        not isinstance(active_source_commit, str)
        or _GIT_SHA_PATTERN.fullmatch(
            active_source_commit
        ) is None
    ):
        raise ValueError(
            "active_source_commit must be a full "
            "40-character lowercase Git SHA"
        )

    checkpoint = Path(
        checkpoint_directory
    ).expanduser().resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory not found: {checkpoint}"
        )

    manifest_path = (
        checkpoint.parent.parent / "manifest.json"
    )
    metadata_path = checkpoint / "metadata.json"

    manifest = _read_json_object(manifest_path)
    metadata = _read_json_object(metadata_path)

    schema_version = manifest.get("schema_version")

    _require_equal(
        "provenance schema_version",
        schema_version,
        SCIENTIFIC_RUN_PROVENANCE_SCHEMA_VERSION,
    )

    training_protocol = manifest.get(
        "training_protocol"
    )
    experiment_config = manifest.get(
        "experiment_config"
    )

    if not isinstance(training_protocol, dict):
        raise ValueError(
            "training_protocol must be a JSON object"
        )

    if not isinstance(experiment_config, dict):
        raise ValueError(
            "experiment_config must be a JSON object"
        )

    protocol_hash = _sha256_json(
        training_protocol
    )
    config_hash = _sha256_json(
        experiment_config
    )

    _require_equal(
        "training_protocol_sha256 hash",
        manifest.get("training_protocol_sha256"),
        protocol_hash,
    )

    _require_equal(
        "experiment_config_sha256 hash",
        manifest.get("experiment_config_sha256"),
        config_hash,
    )

    source_commit = manifest.get("source_commit")

    if (
        not isinstance(source_commit, str)
        or _GIT_SHA_PATTERN.fullmatch(
            source_commit
        ) is None
    ):
        raise ValueError(
            "invalid training source_commit"
        )

    run_definition = {
        "schema_version": schema_version,
        "source_commit": source_commit,
        "training_protocol_sha256": protocol_hash,
        "experiment_config_sha256": config_hash,
    }

    run_definition_hash = _sha256_json(
        run_definition
    )

    _require_equal(
        "run_definition_sha256 hash",
        manifest.get("run_definition_sha256"),
        run_definition_hash,
    )

    _require_equal(
        "source_commit",
        active_source_commit,
        source_commit,
    )

    for name in (
        "source_commit",
        "training_protocol_sha256",
        "experiment_config_sha256",
        "run_definition_sha256",
    ):
        _require_equal(
            f"checkpoint {name}",
            metadata.get(name),
            manifest.get(name),
        )

    model_variant = experiment_config.get(
        "model_variant"
    )
    seed = experiment_config.get("seed")

    if (
        not isinstance(model_variant, str)
        or not model_variant
    ):
        raise ValueError(
            "invalid manifest model_variant"
        )

    if type(seed) is not int:
        raise ValueError(
            "invalid manifest seed"
        )

    _require_equal(
        "checkpoint model_variant",
        metadata.get("model_variant"),
        model_variant,
    )

    _require_equal(
        "checkpoint seed",
        metadata.get("seed"),
        seed,
    )

    step = metadata.get("training_step")
    training_steps = metadata.get("training_steps")

    if (
        type(step) is not int
        or step <= 0
        or type(training_steps) is not int
        or training_steps != step
    ):
        raise ValueError(
            "checkpoint training_step mismatch"
        )

    _require_equal(
        "checkpoint directory step",
        checkpoint.name,
        f"update_{step:08d}",
    )

    return {
        "schema_version": schema_version,
        "checkpoint_directory": str(checkpoint),
        "source_commit": source_commit,
        "model_variant": model_variant,
        "seed": seed,
        "training_step": step,
        "training_protocol_sha256": protocol_hash,
        "experiment_config_sha256": config_hash,
        "run_definition_sha256": run_definition_hash,
        "manifest_sha256": _file_sha256(
            manifest_path
        ),
        "metadata_sha256": _file_sha256(
            metadata_path
        ),
        "online_weights_sha256": _file_sha256(
            checkpoint / "online.weights.h5"
        ),
        "target_weights_sha256": _file_sha256(
            checkpoint / "target.weights.h5"
        ),
    }
