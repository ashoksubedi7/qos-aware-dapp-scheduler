"""Load and verify accepted scientific evaluation artifacts."""

import hashlib
import json
from dataclasses import fields
from pathlib import Path

from config.dqn_hyperparameters import DQNHyperparameters
from config.experiment_config import ExperimentConfig

from experiments.scientific_evaluation_launcher import (
    _verify_evaluation_summary,
)
from experiments.scientific_evaluation_provenance import (
    validate_checkpoint_provenance,
)

from experiments.scientific_validation_semantics import (
    validate_semantic_run_summary,
)

from metrics.run_summary import (
    RunSummary,
    SliceLifecycleSummary,
    SlicePerformanceSummary,
    StarvationSummary,
    SystemSummary,
    URLLCQoSSummary,
)


def _read_json_object(path):
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            f"invalid JSON artifact: {path.name}"
        ) from exc

    if type(value) is not dict:
        raise ValueError(
            f"{path.name} must contain a JSON object"
        )

    return value


def _require_fields(value, expected, name):
    if type(value) is not dict:
        raise ValueError(
            f"{name} must be a JSON object"
        )

    if set(value) != set(expected):
        raise ValueError(
            f"{name} has missing or unknown fields"
        )


def _require_equal(name, actual, expected):
    if actual != expected:
        raise ValueError(
            f"{name} mismatch"
        )


def _verified_artifact_bytes(
    directory,
    filename,
    expected_hash,
):
    if (
        type(expected_hash) is not str
        or len(expected_hash) != 64
    ):
        raise ValueError(
            f"invalid {filename} hash"
        )

    contents = (
        directory / filename
    ).read_bytes()

    actual_hash = hashlib.sha256(
        contents
    ).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            f"{filename} hash mismatch"
        )

    return contents


def _reconstruct_config(value):
    expected_fields = {
        item.name
        for item in fields(ExperimentConfig)
    }

    _require_fields(
        value,
        expected_fields,
        "experiment_config",
    )

    hyperparameters = value["dqn_hyperparameters"]

    _require_fields(
        hyperparameters,
        {
            item.name
            for item in fields(DQNHyperparameters)
        },
        "dqn_hyperparameters",
    )

    options = dict(value)

    options["dqn_hyperparameters"] = (
        DQNHyperparameters(**hyperparameters)
    )

    config = ExperimentConfig(**options)

    if config.training_mode is not False:
        raise ValueError(
            "accepted evaluation must disable training"
        )

    return config


def _reconstruct_performance(value):
    starvation = value["starvation"]

    if starvation is not None:
        starvation = StarvationSummary(
            **starvation
        )

    return SlicePerformanceSummary(
        active_throughput_mbps=(
            value["active_throughput_mbps"]
        ),
        jain_fairness=value["jain_fairness"],
        starvation=starvation,
    )


def _reconstruct_summary(value):
    return RunSummary(
        model_variant=value["model_variant"],
        seed=value["seed"],
        traffic_scenario=value["traffic_scenario"],
        radio_scenario=value["radio_scenario"],
        control_interval_ms=value["control_interval_ms"],
        active_duration_ms=value["active_duration_ms"],
        drain_duration_ms=value["drain_duration_ms"],
        embb_lifecycle=SliceLifecycleSummary(
            **value["embb_lifecycle"]
        ),
        urllc_lifecycle=SliceLifecycleSummary(
            **value["urllc_lifecycle"]
        ),
        mmtc_lifecycle=SliceLifecycleSummary(
            **value["mmtc_lifecycle"]
        ),
        embb_performance=_reconstruct_performance(
            value["embb_performance"]
        ),
        mmtc_performance=_reconstruct_performance(
            value["mmtc_performance"]
        ),
        urllc_qos=URLLCQoSSummary(
            **value["urllc_qos"]
        ),
        system=SystemSummary(
            **value["system"]
        ),
        conservation_ok=value["conservation_ok"],
    )


def load_accepted_evaluation(
    evaluation_directory,
    *,
    checkpoint_directory,
    active_source_commit,
):
    """
    Verify an accepted evaluation and return RunSummary.

    Artifact hashes establish consistency of the
    observed files, not independent authentication
    of their original producer.
    """
    evaluation = Path(
        evaluation_directory
    ).expanduser().resolve()

    checkpoint = Path(
        checkpoint_directory
    ).expanduser().resolve()

    manifest = _read_json_object(
        evaluation / "evaluation_manifest.json"
    )

    _require_fields(
        manifest,
        {
            "schema_version",
            "status",
            "active_source_commit",
            "model_variant",
            "evaluation_seed",
            "active_duration_ms",
            "checkpoint_provenance",
            "request_sha256",
            "summary_sha256",
            "console_sha256",
        },
        "evaluation_manifest",
    )

    if manifest["schema_version"] != 1:
        raise ValueError(
            "unsupported evaluation manifest schema"
        )

    if manifest["status"] != "accepted":
        raise ValueError(
            "evaluation manifest is not accepted"
        )

    _require_equal(
        "source commit",
        manifest["active_source_commit"],
        active_source_commit,
    )

    provenance = manifest["checkpoint_provenance"]

    if type(provenance) is not dict:
        raise ValueError(
            "invalid checkpoint provenance"
        )

    _require_equal(
        "checkpoint directory",
        provenance.get("checkpoint_directory"),
        str(checkpoint),
    )

    _require_equal(
        "checkpoint source",
        provenance.get("source_commit"),
        active_source_commit,
    )

    # Verify the original training manifest and
    # checkpoint files against the current disk state.
    verified_provenance = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=active_source_commit,
    )

    _require_equal(
        "checkpoint provenance",
        provenance,
        verified_provenance,
    )

    request_bytes = _verified_artifact_bytes(
        evaluation,
        "request.json",
        manifest["request_sha256"],
    )

    summary_bytes = _verified_artifact_bytes(
        evaluation,
        "summary.json",
        manifest["summary_sha256"],
    )

    _verified_artifact_bytes(
        evaluation,
        "console.log",
        manifest["console_sha256"],
    )

    try:
        request = json.loads(request_bytes)
        summary = json.loads(summary_bytes)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "invalid evaluation artifact JSON"
        ) from exc

    _require_fields(
        request,
        {
            "schema_version",
            "active_source_commit",
            "active_duration_ms",
            "experiment_config",
            "checkpoint_provenance",
        },
        "evaluation_request",
    )

    if request["schema_version"] != 1:
        raise ValueError(
            "unsupported evaluation request schema"
        )

    _require_equal(
        "request source",
        request["active_source_commit"],
        active_source_commit,
    )

    _require_equal(
        "request checkpoint provenance",
        request["checkpoint_provenance"],
        verified_provenance,
    )

    config = _reconstruct_config(
        request["experiment_config"]
    )

    if (
        type(config.checkpoint_path) is not str
        or Path(
            config.checkpoint_path
        ).expanduser().resolve() != checkpoint
    ):
        raise ValueError(
            "request checkpoint path mismatch"
        )

    _require_equal(
        "manifest model variant",
        manifest["model_variant"],
        config.model_variant,
    )

    _require_equal(
        "checkpoint model variant",
        verified_provenance["model_variant"],
        config.model_variant,
    )

    _require_equal(
        "manifest evaluation seed",
        manifest["evaluation_seed"],
        config.seed,
    )

    _require_equal(
        "manifest active duration",
        manifest["active_duration_ms"],
        request["active_duration_ms"],
    )

    # Enforce the exact canonical RunSummary structure,
    # its metric types and its evaluation identity.
    _verify_evaluation_summary(
        summary,
        config,
        request["active_duration_ms"],
    )

    result = _reconstruct_summary(summary)

    validate_semantic_run_summary(result)

    if not result.conservation_ok:
        raise ValueError(
            "evaluation packet conservation failed"
        )

    # Recheck the checkpoint after reading artifacts.
    final_provenance = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=active_source_commit,
    )

    _require_equal(
        "final checkpoint provenance",
        final_provenance,
        verified_provenance,
    )

    return result
