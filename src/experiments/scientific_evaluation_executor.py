import hashlib
import json
import math
import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import simpy

from Cell import Cell
from config.experiment_config import ExperimentConfig
from environment.traffic_scenarios import get_traffic_scenario
from experiments.scientific_training_executor import (
    _MODEL_VARIANT_TO_SCHEDULER,
    _close_scheduler_files,
    _create_slice,
    _make_group,
)
from metrics.run_summary import build_run_summary


def _learning_snapshot(agent):
    optimizer = agent.online_model.optimizer
    variables = optimizer.variables
    if callable(variables):
        variables = variables()

    return {
        "training_steps": int(agent.training_steps),
        "epsilon": float(agent.epsilon),
        "replay_count": len(agent.replay_buffer),
        "replay_sha256": hashlib.sha256(
            pickle.dumps(
                tuple(agent.replay_buffer),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        ).hexdigest(),
        "optimizer_steps": int(
            optimizer.iterations.numpy()
        ),
        "optimizer_variables": [
            np.asarray(variable.numpy()).copy()
            for variable in variables
        ],
        "online_weights": [
            weight.copy()
            for weight in agent.online_model.get_weights()
        ],
        "target_weights": [
            weight.copy()
            for weight in agent.target_model.get_weights()
        ],
    }


def _assert_learning_unchanged(before, after):
    for key in (
        "training_steps",
        "epsilon",
        "replay_count",
        "replay_sha256",
        "optimizer_steps",
    ):
        if before[key] != after[key]:
            raise RuntimeError(
                f"evaluation modified learning state: {key}"
            )

    for key in (
        "online_weights",
        "target_weights",
        "optimizer_variables",
    ):
        first = before[key]
        second = after[key]

        if len(first) != len(second):
            raise RuntimeError(
                f"evaluation changed {key} tensor count"
            )

        for original, current in zip(first, second):
            if not np.array_equal(original, current):
                raise RuntimeError(
                    f"evaluation modified {key}"
                )


def run_checkpoint_evaluation(
    config,
    active_duration_ms,
    run_directory,
):
    """
    Evaluate one checkpoint without training.

    This function executes one simulator run. A separate
    launcher must provide fresh-process isolation and
    scientific provenance.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            "config must be an ExperimentConfig"
        )

    if config.training_mode:
        raise ValueError(
            "checkpoint evaluation requires "
            "training_mode=False"
        )

    active_duration_ms = float(active_duration_ms)

    if (
        not math.isfinite(active_duration_ms)
        or active_duration_ms <= 0
    ):
        raise ValueError(
            "active_duration_ms must be finite "
            "and greater than zero"
        )

    checkpoint = Path(
        config.checkpoint_path
    ).expanduser().resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory not found: {checkpoint}"
        )

    metadata_path = checkpoint / "metadata.json"

    if not metadata_path.is_file():
        raise ValueError(
            "evaluation checkpoint requires metadata.json"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            "invalid checkpoint metadata.json"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(
            "checkpoint metadata must be a JSON object"
        )

    if metadata.get("model_variant") != config.model_variant:
        raise ValueError(
            "checkpoint model_variant does not match "
            "evaluation model_variant"
        )

    config = replace(
        config,
        checkpoint_path=str(checkpoint),
    )

    run_directory = Path(run_directory).resolve()
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    runtime_directory = run_directory / "runtime"
    runtime_directory.mkdir()

    total_duration_ms = (
        active_duration_ms
        + config.drain_duration_ms
    )

    caller_directory = Path.cwd()

    try:
        os.chdir(runtime_directory)

        env = simpy.Environment()

        cell = Cell(
            "c1",
            [10],
            "FR1",
            False,
            81920,
            False,
            config.control_interval_ms,
            _MODEL_VARIANT_TO_SCHEDULER[
                config.model_variant
            ],
            experiment_config=config,
        )

        scheduler = cell.interSliceSched
        agent = scheduler.agent

        if not agent.evaluation_mode:
            raise RuntimeError(
                "checkpoint evaluation mode is disabled"
            )

        if agent.epsilon != 0.0:
            raise RuntimeError(
                "checkpoint evaluation must use epsilon=0"
            )

        before = _learning_snapshot(agent)

        if before["replay_count"] != 0:
            raise RuntimeError(
                "evaluation must start with empty replay"
            )

        traffic = get_traffic_scenario(
            config.traffic_scenario
        )

        groups = [
            _make_group(
                traffic.embb,
                "eMBB",
                20,
                None,
                cell,
                total_duration_ms,
                50.0,
                env,
                config,
            ),
            _make_group(
                traffic.urllc,
                "URLLC",
                5,
                config.urllc_scheduling_deadline_ms,
                cell,
                total_duration_ms,
                50.0,
                env,
                config,
            ),
            _make_group(
                traffic.mmtc,
                "mMTC",
                20,
                None,
                cell,
                total_duration_ms,
                50.0,
                env,
                config,
            ),
        ]

        for group in groups:
            _create_slice(scheduler, group)

        env.process(scheduler.resAlloc(env))

        for group in groups:
            group.activateSliceScheds(
                scheduler,
                env,
            )

        try:
            env.run(until=total_duration_ms)
        finally:
            _close_scheduler_files(scheduler)

        after = _learning_snapshot(agent)
        _assert_learning_unchanged(before, after)

        if agent.epsilon != 0.0:
            raise RuntimeError(
                "evaluation epsilon changed"
            )

        result = build_run_summary(
            scheduler,
            total_duration_ms,
        )

        if not result.conservation_ok:
            raise RuntimeError(
                "evaluation packet conservation failed"
            )

        return result

    finally:
        os.chdir(caller_directory)
