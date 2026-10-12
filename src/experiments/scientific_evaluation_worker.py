import json
import math
import os
import subprocess
from dataclasses import fields
from pathlib import Path

from experiments.scientific_evaluation_provenance import (
    validate_checkpoint_provenance,
)

from config.dqn_hyperparameters import DQNHyperparameters
from config.experiment_config import ExperimentConfig


def _require_exact_fields(data, data_type, label):
    if not isinstance(data, dict):
        raise TypeError(
            f"{label} must be a JSON object"
        )

    expected = {
        field.name
        for field in fields(data_type)
    }

    provided = set(data)

    missing = expected - provided
    unknown = provided - expected

    if missing:
        raise ValueError(
            f"{label} has missing fields: "
            + ", ".join(sorted(missing))
        )

    if unknown:
        raise ValueError(
            f"{label} has unknown fields: "
            + ", ".join(sorted(unknown))
        )


def deserialize_evaluation_config(data):
    """
    Reconstruct a frozen evaluation configuration
    from its complete serialized representation.

    Missing and unexpected fields are rejected to
    prevent silent configuration changes.
    """
    _require_exact_fields(
        data,
        ExperimentConfig,
        "experiment_config",
    )

    if data["training_mode"] is not False:
        raise ValueError(
            "evaluation requires training_mode=False"
        )

    hyperparameters = data["dqn_hyperparameters"]

    _require_exact_fields(
        hyperparameters,
        DQNHyperparameters,
        "dqn_hyperparameters",
    )

    reconstructed_hyperparameters = (
        DQNHyperparameters(**hyperparameters)
    )

    config_values = dict(data)

    config_values["dqn_hyperparameters"] = (
        reconstructed_hyperparameters
    )

    config = ExperimentConfig(**config_values)

    if config.to_dict() != data:
        raise ValueError(
            "evaluation configuration round-trip mismatch"
        )

    return config

def validate_evaluation_process_environment(config):
    """
    Validate the deterministic CPU execution environment.

    The parent launcher must establish these variables
    before starting the child Python interpreter.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            "config must be an ExperimentConfig"
        )

    if config.training_mode:
        raise ValueError(
            "scientific evaluation requires "
            "training_mode=False"
        )

    expected_seed = str(config.seed)

    actual_seed = os.environ.get(
        "PYTHONHASHSEED"
    )

    if actual_seed != expected_seed:
        raise RuntimeError(
            "PYTHONHASHSEED must match the "
            "evaluation seed; expected "
            f"{expected_seed!r}, found "
            f"{actual_seed!r}"
        )

    deterministic_ops = os.environ.get(
        "TF_DETERMINISTIC_OPS"
    )

    if deterministic_ops != "1":
        raise RuntimeError(
            "TF_DETERMINISTIC_OPS must be '1'; "
            f"found {deterministic_ops!r}"
        )

    cuda_devices = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )

    if cuda_devices != "":
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be "
            "the empty string for CPU-only "
            "scientific evaluation; "
            f"found {cuda_devices!r}"
        )

    return True

def _current_source_commit():
    repository = Path(__file__).resolve().parents[2]

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()


def _load_evaluation_runtime():
    from common.reproducibility import set_global_seed
    from experiments.scientific_evaluation_executor import (
        run_checkpoint_evaluation,
    )

    return set_global_seed, run_checkpoint_evaluation


def _read_evaluation_request(path):
    try:
        request = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            "invalid evaluation request JSON"
        ) from exc

    if not isinstance(request, dict):
        raise ValueError(
            "evaluation request must be a JSON object"
        )

    expected = {
        "schema_version",
        "active_source_commit",
        "active_duration_ms",
        "experiment_config",
        "checkpoint_provenance",
    }

    missing = expected - set(request)
    unknown = set(request) - expected

    if missing:
        raise ValueError(
            "evaluation request has missing fields: "
            + ", ".join(sorted(missing))
        )

    if unknown:
        raise ValueError(
            "evaluation request has unknown fields: "
            + ", ".join(sorted(unknown))
        )

    if (
        type(request["schema_version"]) is not int
        or request["schema_version"] != 1
    ):
        raise ValueError(
            "unsupported evaluation request schema_version"
        )

    duration = request["active_duration_ms"]

    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(
            "active_duration_ms must be finite and positive"
        )

    if not isinstance(
        request["checkpoint_provenance"], dict
    ):
        raise ValueError(
            "checkpoint_provenance must be a JSON object"
        )

    return request


def _publish_summary(path, summary):
    data = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    temporary_path = path.with_name(
        ".summary.json.tmp"
    )

    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        # A hard link publishes complete bytes without
        # overwriting an existing summary.
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def execute_evaluation_request(request_path):
    request_path = Path(
        request_path
    ).expanduser().resolve()

    request = _read_evaluation_request(
        request_path
    )

    root = request_path.parent
    summary_path = root / "summary.json"
    simulation_directory = root / "simulation"

    if summary_path.exists():
        raise FileExistsError(
            f"evaluation summary already exists: {summary_path}"
        )

    if simulation_directory.exists():
        raise FileExistsError(
            "evaluation simulation directory already exists"
        )

    config = deserialize_evaluation_config(
        request["experiment_config"]
    )

    requested_source = request["active_source_commit"]
    actual_source = _current_source_commit()

    if requested_source != actual_source:
        raise ValueError(
            "active source_commit mismatch"
        )

    validate_evaluation_process_environment(
        config
    )

    expected_provenance = request[
        "checkpoint_provenance"
    ]

    checkpoint = Path(
        config.checkpoint_path
    ).expanduser().resolve()

    if (
        expected_provenance.get("checkpoint_directory")
        != str(checkpoint)
    ):
        raise ValueError(
            "checkpoint provenance directory mismatch"
        )

    if (
        expected_provenance.get("source_commit")
        != requested_source
    ):
        raise ValueError(
            "checkpoint provenance source_commit mismatch"
        )

    if (
        expected_provenance.get("model_variant")
        != config.model_variant
    ):
        raise ValueError(
            "checkpoint provenance model_variant mismatch"
        )

    before = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=requested_source,
    )

    if before != expected_provenance:
        raise ValueError(
            "checkpoint provenance mismatch before evaluation"
        )

    set_global_seed, evaluate = (
        _load_evaluation_runtime()
    )

    set_global_seed(config.seed)

    result = evaluate(
        config=config,
        active_duration_ms=request["active_duration_ms"],
        run_directory=simulation_directory,
    )

    after = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=requested_source,
    )

    if after != before:
        raise ValueError(
            "checkpoint provenance changed during evaluation"
        )

    summary = result.to_dict()

    if not isinstance(summary, dict):
        raise ValueError(
            "evaluation summary must be a dictionary"
        )

    expected_fields = {
        "model_variant": config.model_variant,
        "seed": config.seed,
        "traffic_scenario": config.traffic_scenario,
        "radio_scenario": config.radio_scenario,
        "active_duration_ms": float(
            request["active_duration_ms"]
        ),
        "drain_duration_ms": config.drain_duration_ms,
        "conservation_ok": True,
    }

    for field, expected_value in expected_fields.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"evaluation summary {field} mismatch"
            )

    _publish_summary(
        summary_path,
        summary,
    )

    return summary_path


def main():
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m "
            "experiments.scientific_evaluation_worker "
            "REQUEST_JSON"
        )

    execute_evaluation_request(
        sys.argv[1]
    )


if __name__ == "__main__":
    main()
