import hashlib
import json
import re
from dataclasses import asdict

from config.experiment_config import (
    ExperimentConfig,
)
from config.training_protocol import (
    TrainingProtocol,
)


SCIENTIFIC_RUN_PROVENANCE_SCHEMA_VERSION = 1

_FULL_LOWERCASE_GIT_SHA1 = re.compile(
    r"^[0-9a-f]{40}$"
)


def _canonical_json_bytes(value):
    """
    Serialize scientific provenance deterministically.

    Whitespace and dictionary insertion order must not
    influence any provenance digest.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_json(value):
    return hashlib.sha256(
        _canonical_json_bytes(value)
    ).hexdigest()


def build_scientific_run_provenance(
    protocol,
    config,
    source_commit,
):
    """
    Build immutable provenance for one scientific run.

    The run-definition digest binds the exact source
    commit, frozen training protocol, and derived
    experiment configuration.
    """
    if not isinstance(
        protocol,
        TrainingProtocol,
    ):
        raise TypeError(
            "protocol must be a TrainingProtocol"
        )

    if not isinstance(
        config,
        ExperimentConfig,
    ):
        raise TypeError(
            "config must be an ExperimentConfig"
        )

    protocol.assert_ready_for_scientific_training()

    if (
        not isinstance(source_commit, str)
        or _FULL_LOWERCASE_GIT_SHA1.fullmatch(
            source_commit
        )
        is None
    ):
        raise ValueError(
            "source_commit must be a full "
            "40-character lowercase Git SHA-1"
        )

    training_protocol = asdict(
        protocol
    )

    experiment_config = config.to_dict()

    training_protocol_sha256 = (
        _sha256_json(
            training_protocol
        )
    )

    experiment_config_sha256 = (
        _sha256_json(
            experiment_config
        )
    )

    run_definition = {
        "schema_version": (
            SCIENTIFIC_RUN_PROVENANCE_SCHEMA_VERSION
        ),
        "source_commit": source_commit,
        "training_protocol_sha256": (
            training_protocol_sha256
        ),
        "experiment_config_sha256": (
            experiment_config_sha256
        ),
    }

    run_definition_sha256 = (
        _sha256_json(
            run_definition
        )
    )

    return {
        "schema_version": (
            SCIENTIFIC_RUN_PROVENANCE_SCHEMA_VERSION
        ),
        "source_commit": source_commit,
        "training_protocol": training_protocol,
        "training_protocol_sha256": (
            training_protocol_sha256
        ),
        "experiment_config": experiment_config,
        "experiment_config_sha256": (
            experiment_config_sha256
        ),
        "run_definition_sha256": (
            run_definition_sha256
        ),
    }
