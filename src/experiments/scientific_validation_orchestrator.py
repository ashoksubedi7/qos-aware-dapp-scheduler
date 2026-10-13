"""Execute complete frozen-checkpoint validation grids."""

import math
from dataclasses import replace
from pathlib import Path

from config.experiment_config import ExperimentConfig

from experiments.checkpoint_aggregation import (
    aggregate_validation_suite,
)
from experiments.checkpoint_schedule import (
    CheckpointCandidate,
)
from experiments.checkpoint_selection import (
    CheckpointScore,
)
from experiments.scientific_evaluation_launcher import (
    launch_scientific_evaluation,
)
from experiments.scientific_evaluation_provenance import (
    validate_checkpoint_provenance,
)
from experiments.scientific_validation_artifacts import (
    load_accepted_evaluation,
)
from experiments.scientific_validation_grid import (
    _validate_axis,
    validate_declared_validation_grid,
)


def _require_checkpoint_identity(
    provenance,
    checkpoint,
    source_commit,
    model_variant,
):
    if not isinstance(provenance, dict):
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

    return step


def run_checkpoint_validation_grid(
    *,
    checkpoint_directory,
    base_config,
    traffic_scenarios,
    radio_scenarios,
    seeds,
    active_duration_ms,
    output_directory,
    active_source_commit,
):
    """
    Evaluate every declared condition and seed.

    No checkpoint score is produced until all
    evaluations are accepted, verified, and complete.

    Scientific checkpoint selection is handled
    separately by select_best_checkpoint().
    """
    if not isinstance(base_config, ExperimentConfig):
        raise TypeError(
            "base_config must be an ExperimentConfig"
        )

    if base_config.training_mode:
        raise ValueError(
            "checkpoint validation requires "
            "training_mode=False"
        )

    if (
        type(active_duration_ms) not in (int, float)
        or not math.isfinite(active_duration_ms)
        or active_duration_ms <= 0
    ):
        raise ValueError(
            "active_duration_ms must be finite "
            "and positive"
        )

    duration = float(active_duration_ms)

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

    checkpoint = Path(
        checkpoint_directory
    ).expanduser().resolve()

    output = Path(
        output_directory
    ).expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"validation output already exists: {output}"
        )

    provenance = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=active_source_commit,
    )

    training_step = _require_checkpoint_identity(
        provenance,
        checkpoint,
        active_source_commit,
        base_config.model_variant,
    )

    # Construct every configuration before executing
    # any simulation. Invalid scenario declarations
    # therefore cannot cause partial execution.
    planned = []

    for traffic in traffic_scenarios:
        for radio in radio_scenarios:
            for seed in seeds:
                config = replace(
                    base_config,
                    checkpoint_path=str(checkpoint),
                    training_mode=False,
                    traffic_scenario=traffic,
                    radio_scenario=radio,
                    seed=seed,
                )

                planned.append(config)

    # Names use deterministic indices rather than
    # user-provided scenario strings as path elements.
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    summaries = []

    for index, config in enumerate(planned, start=1):
        evaluation_directory = (
            output / f"cell_{index:06d}"
        )

        launch_scientific_evaluation(
            config=config,
            active_duration_ms=duration,
            run_directory=evaluation_directory,
        )

        summary = load_accepted_evaluation(
            evaluation_directory,
            checkpoint_directory=checkpoint,
            active_source_commit=active_source_commit,
        )

        summaries.append(summary)

    # Detect missing cells, extra cells, duplicate
    # results, and missing entire scenario/seed axes.
    validate_declared_validation_grid(
        summaries,
        traffic_scenarios=traffic_scenarios,
        radio_scenarios=radio_scenarios,
        seeds=seeds,
    )

    # Rectangular completeness is insufficient if
    # two valid results were assigned to the wrong
    # execution directories. Check each requested
    # cell against its actual returned identity.
    for config, summary in zip(
        planned,
        summaries,
        strict=True,
    ):
        expected = (
            config.model_variant,
            config.traffic_scenario,
            config.radio_scenario,
            config.seed,
            config.control_interval_ms,
            duration,
            config.drain_duration_ms,
        )

        actual = (
            summary.model_variant,
            summary.traffic_scenario,
            summary.radio_scenario,
            summary.seed,
            summary.control_interval_ms,
            summary.active_duration_ms,
            summary.drain_duration_ms,
        )

        if actual != expected:
            raise ValueError(
                "evaluation result identity does "
                "not match its requested grid cell"
            )

    # Existing frozen scientific methodology:
    # pooled ratios, worst-condition QoS, and
    # macro-averaged service metrics.
    score = aggregate_validation_suite(
        summaries
    )

    if not isinstance(score, CheckpointScore):
        raise TypeError(
            "aggregation must produce a "
            "CheckpointScore"
        )

    # Reconfirm checkpoint identity before
    # returning an eligible candidate.
    final_provenance = validate_checkpoint_provenance(
        checkpoint_directory=checkpoint,
        active_source_commit=active_source_commit,
    )

    if final_provenance != provenance:
        raise ValueError(
            "checkpoint provenance changed "
            "during validation"
        )

    return CheckpointCandidate(
        training_step=training_step,
        score=score,
    )
