import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass, replace
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints
from pathlib import Path

from config.experiment_config import ExperimentConfig
from experiments.scientific_evaluation_provenance import (
    validate_checkpoint_provenance,
)
from experiments.scientific_run_provenance import (
    _canonical_json_bytes,
)
from experiments.scientific_training_launcher import (
    _git_head,
    _repository_root,
    _require_clean_tracked_worktree,
)


def _validate_run_summary_structure(
    value,
    expected_type,
    location,
):
    origin = get_origin(expected_type)

    if origin in (Union, UnionType):
        options = get_args(expected_type)

        if value is None:
            if type(None) in options:
                return
            raise ValueError(
                f"RunSummary {location} cannot be null"
            )

        alternatives = [
            option
            for option in options
            if option is not type(None)
        ]

        if len(alternatives) != 1:
            raise ValueError(
                f"unsupported RunSummary type at {location}"
            )

        return _validate_run_summary_structure(
            value,
            alternatives[0],
            location,
        )

    if is_dataclass(expected_type):
        if type(value) is not dict:
            raise ValueError(
                f"RunSummary {location} must be an object"
            )

        expected = {
            item.name
            for item in fields(expected_type)
        }

        missing = expected - set(value)
        unknown = set(value) - expected

        if missing:
            raise ValueError(
                f"RunSummary {location} missing fields: "
                + ", ".join(sorted(missing))
            )

        if unknown:
            raise ValueError(
                f"RunSummary {location} unknown fields: "
                + ", ".join(sorted(unknown))
            )

        annotations = get_type_hints(expected_type)

        for name in sorted(expected):
            _validate_run_summary_structure(
                value[name],
                annotations[name],
                f"{location}.{name}",
            )

        return

    if expected_type is float:
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"RunSummary {location} must be finite numeric"
            )
        return

    if type(value) is not expected_type:
        raise ValueError(
            f"RunSummary {location} has invalid type"
        )


def _verify_evaluation_summary(summary, config, duration):
    if not isinstance(summary, dict):
        raise ValueError(
            "evaluation summary must be a JSON object"
        )

    # Import only on the result-verification path.
    from metrics.run_summary import RunSummary

    _validate_run_summary_structure(
        summary,
        RunSummary,
        "root",
    )

    expected = {
        "control_interval_ms": config.control_interval_ms,
        "model_variant": config.model_variant,
        "seed": config.seed,
        "traffic_scenario": config.traffic_scenario,
        "radio_scenario": config.radio_scenario,
        "active_duration_ms": duration,
        "drain_duration_ms": config.drain_duration_ms,
        "conservation_ok": True,
    }

    for name, value in expected.items():
        actual = summary.get(name)

        if name == "conservation_ok":
            valid = actual is True
        elif name == "seed":
            valid = type(actual) is int and actual == value
        else:
            valid = (
                type(actual) is type(value)
                and actual == value
            )

        if not valid:
            raise ValueError(
                f"evaluation summary {name} mismatch"
            )

    # Reject nonfinite numerical results, including nested values.
    _canonical_json_bytes(summary)
    json.dumps(summary, allow_nan=False)

    return True



def _sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _publish_acceptance_manifest(path, manifest):
    """
    Publish a complete acceptance manifest without
    replacing any existing file.
    """
    data = _canonical_json_bytes(manifest)

    temporary = path.with_name(
        ".evaluation_manifest.json.tmp"
    )

    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary, path)

    finally:
        temporary.unlink(missing_ok=True)


def launch_scientific_evaluation(
    config,
    active_duration_ms,
    run_directory,
):
    """
    Launch one frozen-checkpoint evaluation in a new
    deterministic CPU-only Python process.

    The output directory is never reused or overwritten.
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

    if (
        isinstance(active_duration_ms, bool)
        or not isinstance(active_duration_ms, (int, float))
        or not math.isfinite(active_duration_ms)
        or active_duration_ms <= 0
    ):
        raise ValueError(
            "active_duration_ms must be finite and positive"
        )

    duration = float(active_duration_ms)

    output = Path(
        run_directory
    ).expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"evaluation output already exists: {output}"
        )

    repository = _repository_root()

    _require_clean_tracked_worktree(
        repository
    )

    source_commit = _git_head(
        repository
    )

    checkpoint = Path(
        config.checkpoint_path
    ).expanduser().resolve()

    provenance = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=source_commit,
    )

    if provenance.get("source_commit") != source_commit:
        raise ValueError(
            "checkpoint source_commit mismatch"
        )

    if provenance.get("checkpoint_directory") != str(
        checkpoint
    ):
        raise ValueError(
            "checkpoint provenance directory mismatch"
        )

    if provenance.get("model_variant") != config.model_variant:
        raise ValueError(
            "checkpoint model_variant mismatch"
        )

    config = replace(
        config,
        checkpoint_path=str(checkpoint),
    )

    request = {
        "schema_version": 1,
        "active_source_commit": source_commit,
        "active_duration_ms": duration,
        "experiment_config": config.to_dict(),
        "checkpoint_provenance": provenance,
    }

    request_bytes = _canonical_json_bytes(
        request
    )

    expected_request_sha256 = hashlib.sha256(
        request_bytes
    ).hexdigest()

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    request_path = output / "request.json"
    summary_path = output / "summary.json"
    log_path = output / "console.log"

    with request_path.open("xb") as handle:
        handle.write(request_bytes)

    environment = os.environ.copy()

    environment.update({
        "PYTHONHASHSEED": str(config.seed),
        "TF_DETERMINISTIC_OPS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONUNBUFFERED": "1",
    })

    module_paths = [
        str(repository / "simulator"),
        str(repository / "simulator" / "lib"),
        str(repository / "src"),
    ]

    previous_pythonpath = environment.get(
        "PYTHONPATH"
    )

    if previous_pythonpath:
        module_paths.append(
            previous_pythonpath
        )

    environment["PYTHONPATH"] = os.pathsep.join(
        module_paths
    )

    command = [
        sys.executable,
        "-m",
        "experiments.scientific_evaluation_worker",
        str(request_path),
    ]

    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            "evaluation child process failed with "
            f"exit code {completed.returncode}; "
            f"see {log_path}"
        )

    if _sha256_file(request_path) != expected_request_sha256:
        raise ValueError(
            "evaluation request changed during child execution"
        )

    if not summary_path.is_file():
        raise RuntimeError(
            "evaluation child process completed "
            "without producing summary.json"
        )

    after = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=source_commit,
    )

    if after != provenance:
        raise ValueError(
            "checkpoint provenance changed "
            "during evaluation"
        )

    try:
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            "invalid evaluation summary JSON"
        ) from exc

    _verify_evaluation_summary(
        summary,
        config,
        duration,
    )

    if _sha256_file(request_path) != expected_request_sha256:
        raise ValueError(
            "evaluation request changed before acceptance"
        )

    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "active_source_commit": source_commit,
        "model_variant": config.model_variant,
        "evaluation_seed": config.seed,
        "active_duration_ms": duration,
        "checkpoint_provenance": provenance,
        "request_sha256": expected_request_sha256,
        "summary_sha256": _sha256_file(
            summary_path
        ),
        "console_sha256": _sha256_file(
            log_path
        ),
    }

    _publish_acceptance_manifest(
        output / "evaluation_manifest.json",
        manifest,
    )

    return summary_path
