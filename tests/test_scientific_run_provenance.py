from dataclasses import replace

import pytest

from config.experiment_config import (
    ExperimentConfig,
)
from config.training_protocol import (
    TrainingProtocol,
)
from experiments.scientific_run_provenance import (
    build_scientific_run_provenance,
)
from experiments.scientific_training_setup import (
    build_scientific_training_setup,
)


SOURCE_COMMIT = (
    "0123456789abcdef0123456789abcdef01234567"
)


def _protocol():
    # Fixture values only. They are not study authority.
    return TrainingProtocol(
        successful_gradient_update_budget=5,
        epsilon_floor_update=4,
        checkpoint_interval_updates=2,
    )


def _derived_config():
    setup = build_scientific_training_setup(
        protocol=_protocol(),
        config=ExperimentConfig(
            model_variant="M1",
            seed=101,
            training_mode=True,
            traffic_scenario="CONGESTED",
            radio_scenario="STATIC_GOOD",
        ),
    )

    return setup.config


def test_provenance_is_deterministic():
    first = build_scientific_run_provenance(
        protocol=_protocol(),
        config=_derived_config(),
        source_commit=SOURCE_COMMIT,
    )

    second = build_scientific_run_provenance(
        protocol=_protocol(),
        config=_derived_config(),
        source_commit=SOURCE_COMMIT,
    )

    assert first == second

    assert first["schema_version"] == 1
    assert first["source_commit"] == SOURCE_COMMIT

    for key in (
        "training_protocol_sha256",
        "experiment_config_sha256",
        "run_definition_sha256",
    ):
        digest = first[key]

        assert len(digest) == 64
        assert set(digest) <= set(
            "0123456789abcdef"
        )


def test_provenance_records_complete_protocol_and_derived_config():
    protocol = _protocol()
    config = _derived_config()

    provenance = build_scientific_run_provenance(
        protocol=protocol,
        config=config,
        source_commit=SOURCE_COMMIT,
    )

    assert (
        provenance["training_protocol"][
            "successful_gradient_update_budget"
        ]
        == 5
    )

    assert (
        provenance["training_protocol"][
            "epsilon_floor_update"
        ]
        == 4
    )

    assert (
        provenance["training_protocol"][
            "checkpoint_interval_updates"
        ]
        == 2
    )

    assert (
        provenance["experiment_config"][
            "dqn_hyperparameters"
        ]["epsilon_decay"]
        == config.dqn_hyperparameters.epsilon_decay
    )


def test_config_change_changes_only_config_and_run_hashes():
    protocol = _protocol()
    config = _derived_config()

    first = build_scientific_run_provenance(
        protocol=protocol,
        config=config,
        source_commit=SOURCE_COMMIT,
    )

    changed_config = replace(
        config,
        seed=103,
    )

    second = build_scientific_run_provenance(
        protocol=protocol,
        config=changed_config,
        source_commit=SOURCE_COMMIT,
    )

    assert (
        first["training_protocol_sha256"]
        == second["training_protocol_sha256"]
    )

    assert (
        first["experiment_config_sha256"]
        != second["experiment_config_sha256"]
    )

    assert (
        first["run_definition_sha256"]
        != second["run_definition_sha256"]
    )


def test_protocol_change_changes_only_protocol_and_run_hashes():
    config = _derived_config()

    first_protocol = _protocol()

    second_protocol = TrainingProtocol(
        successful_gradient_update_budget=6,
        epsilon_floor_update=4,
        checkpoint_interval_updates=2,
    )

    first = build_scientific_run_provenance(
        protocol=first_protocol,
        config=config,
        source_commit=SOURCE_COMMIT,
    )

    second = build_scientific_run_provenance(
        protocol=second_protocol,
        config=config,
        source_commit=SOURCE_COMMIT,
    )

    assert (
        first["training_protocol_sha256"]
        != second["training_protocol_sha256"]
    )

    assert (
        first["experiment_config_sha256"]
        == second["experiment_config_sha256"]
    )

    assert (
        first["run_definition_sha256"]
        != second["run_definition_sha256"]
    )


def test_source_commit_changes_only_run_definition_hash():
    protocol = _protocol()
    config = _derived_config()

    first = build_scientific_run_provenance(
        protocol=protocol,
        config=config,
        source_commit=SOURCE_COMMIT,
    )

    second_commit = (
        "abcdef0123456789abcdef0123456789abcdef01"
    )

    second = build_scientific_run_provenance(
        protocol=protocol,
        config=config,
        source_commit=second_commit,
    )

    assert (
        first["training_protocol_sha256"]
        == second["training_protocol_sha256"]
    )

    assert (
        first["experiment_config_sha256"]
        == second["experiment_config_sha256"]
    )

    assert (
        first["run_definition_sha256"]
        != second["run_definition_sha256"]
    )


@pytest.mark.parametrize(
    "source_commit",
    (
        "",
        "abc123",
        "g" * 40,
        "0" * 39,
        "0" * 41,
    ),
)
def test_provenance_requires_full_lowercase_git_sha1(
    source_commit,
):
    with pytest.raises(ValueError):
        build_scientific_run_provenance(
            protocol=_protocol(),
            config=_derived_config(),
            source_commit=source_commit,
        )


def test_provenance_requires_frozen_protocol():
    with pytest.raises(RuntimeError):
        build_scientific_run_provenance(
            protocol=TrainingProtocol(),
            config=_derived_config(),
            source_commit=SOURCE_COMMIT,
        )
