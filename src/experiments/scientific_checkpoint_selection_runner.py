"""Validate and select scientific training checkpoints."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from config.experiment_config import ExperimentConfig

from experiments.checkpoint_schedule import (
    CheckpointCandidate,
    candidate_checkpoint_steps,
    select_best_checkpoint,
)
from experiments.scientific_evaluation_provenance import (
    validate_checkpoint_provenance,
)
from experiments.scientific_validation_grid import (
    _validate_axis,
)
from experiments.scientific_validation_orchestrator import (
    run_checkpoint_validation_grid,
)


@dataclass(frozen=True)
class ScientificCheckpointSelection:
    checkpoint_directory: Path
    candidate: CheckpointCandidate


_RUN_IDENTITY_FIELDS = (
    "source_commit",
    "model_variant",
    "seed",
    "run_definition_sha256",
    "training_protocol_sha256",
    "experiment_config_sha256",
    "manifest_sha256",
)


def _validate_steps(steps):
    if isinstance(steps, (str, bytes)):
        raise TypeError(
            "expected_training_steps must be a sequence"
        )

    steps = tuple(steps)

    if not steps:
        raise ValueError(
            "expected checkpoint schedule is empty"
        )

    for step in steps:
        if type(step) is not int or step <= 0:
            raise ValueError(
                "checkpoint steps must be positive integers"
            )

    if len(set(steps)) != len(steps):
        raise ValueError(
            "duplicate expected checkpoint step"
        )

    return steps


def _validate_checkpoint_provenance_identity(
    provenance,
    checkpoint,
    source_commit,
    model_variant,
):
    if type(provenance) is not dict:
        raise ValueError(
            "invalid checkpoint provenance"
        )

    if provenance.get("source_commit") != source_commit:
        raise ValueError(
            "checkpoint source mismatch"
        )

    if provenance.get("checkpoint_directory") != str(
        checkpoint
    ):
        raise ValueError(
            "checkpoint directory mismatch"
        )

    if provenance.get("model_variant") != model_variant:
        raise ValueError(
            "checkpoint model variant mismatch"
        )

    step = provenance.get("training_step")

    if type(step) is not int or step <= 0:
        raise ValueError(
            "invalid checkpoint training step"
        )

    training_seed = provenance.get("seed")

    if type(training_seed) is not int:
        raise ValueError(
            "invalid checkpoint training seed"
        )

    for field in _RUN_IDENTITY_FIELDS:
        if provenance.get(field) is None:
            raise ValueError(
                f"missing training run identity: {field}"
            )

    return step


def run_scientific_checkpoint_selection(
    *,
    checkpoint_directories,
    expected_training_steps,
    base_config,
    traffic_scenarios,
    radio_scenarios,
    seeds,
    active_duration_ms,
    output_directory,
    active_source_commit,
):
    """
    Evaluate every declared candidate checkpoint.

    Checkpoint selection occurs only after all
    required candidates successfully complete
    their validation grids.
    """
    if not isinstance(base_config, ExperimentConfig):
        raise TypeError(
            "base_config must be an ExperimentConfig"
        )

    if base_config.training_mode:
        raise ValueError(
            "checkpoint selection requires training_mode=False"
        )

    if (
        type(active_duration_ms) not in (int, float)
        or not math.isfinite(active_duration_ms)
        or active_duration_ms <= 0
    ):
        raise ValueError(
            "active_duration_ms must be finite and positive"
        )

    duration = float(active_duration_ms)

    expected_steps = _validate_steps(
        expected_training_steps
    )

    traffic_scenarios = _validate_axis(
        traffic_scenarios,
        "traffic_scenarios",
        str,
    )

    radio_scenarios = _validate_axis(
        radio_scenarios,
        "radio_scenarios",
        str,
    )

    seeds = _validate_axis(
        seeds,
        "seeds",
        int,
    )

    checkpoints = tuple(
        Path(path).expanduser().resolve()
        for path in checkpoint_directories
    )

    output = Path(
        output_directory
    ).expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"selection output already exists: {output}"
        )

    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError(
            "duplicate checkpoint directory"
        )

    # Every selected checkpoint must be a sibling
    # within the same training run's checkpoint root.
    if len({
        checkpoint.parent
        for checkpoint in checkpoints
    }) > 1:
        raise ValueError(
            "checkpoints belong to different training runs"
        )

    provenance_by_step = {}
    common_identity = None
    training_seed = None

    # Preflight all checkpoints before launching
    # any simulations.
    for checkpoint in checkpoints:
        provenance = validate_checkpoint_provenance(
            checkpoint_directory=checkpoint,
            active_source_commit=active_source_commit,
        )

        step = _validate_checkpoint_provenance_identity(
            provenance,
            checkpoint,
            active_source_commit,
            base_config.model_variant,
        )

        if step in provenance_by_step:
            raise ValueError(
                "duplicate checkpoint training step"
            )

        identity = tuple(
            provenance[field]
            for field in _RUN_IDENTITY_FIELDS
        )

        if common_identity is None:
            common_identity = identity
            training_seed = provenance["seed"]

        elif identity != common_identity:
            raise ValueError(
                "checkpoints belong to different training runs"
            )

        provenance_by_step[step] = (
            checkpoint,
            provenance,
        )

    observed_steps = set(provenance_by_step)
    required_steps = set(expected_steps)

    missing = sorted(required_steps - observed_steps)
    unexpected = sorted(observed_steps - required_steps)

    if missing or unexpected:
        raise ValueError(
            "checkpoint schedule mismatch; "
            f"missing={missing}; unexpected={unexpected}"
        )

    # Derive the authoritative candidate schedule
    # from the original training manifest.
    manifest_path = (
        checkpoints[0].parent.parent / "manifest.json"
    )

    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            "original training manifest unavailable"
        ) from exc

    manifest_hash = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    for _, provenance in provenance_by_step.values():
        if provenance["manifest_sha256"] != manifest_hash:
            raise ValueError(
                "original training manifest hash mismatch"
            )

    try:
        manifest = json.loads(manifest_bytes)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "invalid original training manifest"
        ) from exc

    if type(manifest) is not dict:
        raise ValueError(
            "invalid original training manifest"
        )

    protocol = manifest.get("training_protocol")

    if type(protocol) is not dict:
        raise ValueError(
            "missing original training protocol"
        )

    try:
        authoritative_steps = candidate_checkpoint_steps(
            protocol["successful_gradient_update_budget"],
            protocol["checkpoint_interval_updates"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "invalid original training checkpoint schedule"
        ) from exc

    if expected_steps != authoritative_steps:
        missing = sorted(
            set(authoritative_steps) - set(expected_steps)
        )
        unexpected = sorted(
            set(expected_steps) - set(authoritative_steps)
        )

        raise ValueError(
            "declared checkpoint schedule differs "
            "from original training manifest; "
            f"missing={missing}; unexpected={unexpected}"
        )

    if training_seed in seeds:
        raise ValueError(
            "training seed cannot appear in validation grid"
        )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    candidates = []
    checkpoint_for_step = {}

    # Evaluate every required checkpoint.
    for step in expected_steps:
        checkpoint, original_provenance = (
            provenance_by_step[step]
        )

        checkpoint_output = (
            output / f"update_{step:08d}"
        )

        candidate = run_checkpoint_validation_grid(
            checkpoint_directory=checkpoint,
            base_config=base_config,
            traffic_scenarios=traffic_scenarios,
            radio_scenarios=radio_scenarios,
            seeds=seeds,
            active_duration_ms=duration,
            output_directory=checkpoint_output,
            active_source_commit=active_source_commit,
        )

        if not isinstance(candidate, CheckpointCandidate):
            raise TypeError(
                "validation must return a CheckpointCandidate"
            )

        if candidate.training_step != step:
            raise ValueError(
                "validation returned wrong checkpoint step"
            )

        # Detect any change after the validation run.
        current_provenance = validate_checkpoint_provenance(
            checkpoint_directory=checkpoint,
            active_source_commit=active_source_commit,
        )

        if current_provenance != original_provenance:
            raise ValueError(
                "checkpoint provenance changed during selection"
            )

        candidates.append(candidate)
        checkpoint_for_step[step] = checkpoint

    # The existing frozen QoS-first selector handles
    # ranking and earliest-step exact tie breaking.
    selected = select_best_checkpoint(
        candidates
    )

    return ScientificCheckpointSelection(
        checkpoint_directory=(
            checkpoint_for_step[selected.training_step]
        ),
        candidate=selected,
    )
