"""Persistent training records for scientific DQN runs."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _valid_number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite"
        )

    return result


def _valid_integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    result = int(value)

    if result < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}"
        )

    return result


def _state_values(state):
    array = np.asarray(
        state,
        dtype=np.float32,
    )

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "state must be a nonempty vector"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "state must contain finite values"
        )

    return array.tolist()


class ScientificTrainingRecorder:
    """Write append-only records outside the replay buffer."""

    def __init__(
        self,
        run_directory,
        model_variant,
        seed,
        flush_every=100,
    ):
        if (
            not isinstance(model_variant, str)
            or not model_variant
        ):
            raise ValueError(
                "model_variant must be nonempty"
            )

        self.model_variant = model_variant
        self.seed = _valid_integer(
            seed,
            "seed",
        )

        self.flush_every = _valid_integer(
            flush_every,
            "flush_every",
            minimum=1,
        )

        self.directory = (
            Path(run_directory).expanduser().resolve()
            / "training_data"
        )

        # Never reuse or overwrite existing training data.
        self.directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.transitions_path = (
            self.directory / "transitions.jsonl"
        )

        self.metrics_path = (
            self.directory / "training_metrics.csv"
        )

        self.manifest_path = (
            self.directory / "recording_manifest.json"
        )

        self.transition_count = 0
        self.update_count = 0
        self._closed = False

        # An unexpected process termination must not
        # leave behind a manifest claiming completion.
        self._write_manifest("in_progress")

        self._transitions = (
            self.transitions_path.open(
                "x",
                encoding="utf-8",
                newline="",
            )
        )

        self._metrics = (
            self.metrics_path.open(
                "x",
                encoding="utf-8",
                newline="",
            )
        )

        self._metrics_writer = csv.writer(
            self._metrics
        )

        self._metrics_writer.writerow(
            (
                "training_step",
                "training_loss",
                "epsilon",
            )
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError(
                "training recorder is closed"
            )

    def _write_manifest(self, status, hashes=None):
        document = {
            "schema_version": 1,
            "status": status,
            "model_variant": self.model_variant,
            "seed": self.seed,
            "transition_count": self.transition_count,
            "update_count": self.update_count,
        }

        if hashes is not None:
            document.update(hashes)

        temporary = self.manifest_path.with_suffix(
            ".json.tmp"
        )

        temporary.write_text(
            json.dumps(
                document,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            self.manifest_path
        )

    def _flush_if_needed(self, handle, count):
        if count % self.flush_every == 0:
            handle.flush()

    def record_transition(
        self,
        *,
        state,
        action,
        reward,
        next_state,
        done,
        training_step,
        epsilon,
    ):
        self._require_open()

        current = _state_values(state)
        following = _state_values(next_state)

        if len(current) != len(following):
            raise ValueError(
                "state dimensions must match"
            )

        action = _valid_integer(
            action,
            "action",
        )

        reward = _valid_number(
            reward,
            "reward",
        )

        training_step = _valid_integer(
            training_step,
            "training_step",
        )

        epsilon = _valid_number(
            epsilon,
            "epsilon",
        )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be in [0, 1]"
            )

        if not isinstance(
            done,
            (bool, np.bool_),
        ):
            raise TypeError(
                "done must be Boolean"
            )

        record = {
            "index": self.transition_count + 1,
            "state": current,
            "action": action,
            "reward": reward,
            "next_state": following,
            "done": bool(done),
            "training_step": training_step,
            "epsilon": epsilon,
        }

        line = json.dumps(
            record,
            separators=(",", ":"),
            allow_nan=False,
        )

        self._transitions.write(
            line + "\n"
        )

        self.transition_count += 1

        self._flush_if_needed(
            self._transitions,
            self.transition_count,
        )

    def record_update(
        self,
        *,
        training_step,
        training_loss,
        epsilon,
    ):
        self._require_open()

        training_step = _valid_integer(
            training_step,
            "training_step",
            minimum=1,
        )

        if training_step != self.update_count + 1:
            raise ValueError(
                "successful update numbers "
                "must be sequential"
            )

        loss = _valid_number(
            training_loss,
            "training_loss",
        )

        epsilon = _valid_number(
            epsilon,
            "epsilon",
        )

        if loss < 0:
            raise ValueError(
                "training loss must be nonnegative"
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be in [0, 1]"
            )

        self._metrics_writer.writerow(
            (
                training_step,
                loss,
                epsilon,
            )
        )

        self.update_count += 1

        self._flush_if_needed(
            self._metrics,
            self.update_count,
        )

    def finish(self, *, success):
        self._require_open()

        if not isinstance(success, bool):
            raise TypeError(
                "success must be Boolean"
            )

        try:
            self._transitions.flush()
            self._metrics.flush()
        finally:
            self._transitions.close()
            self._metrics.close()
            self._closed = True

        hashes = {
            "transitions_sha256": _sha256(
                self.transitions_path
            ),
            "training_metrics_sha256": _sha256(
                self.metrics_path
            ),
        }

        self._write_manifest(
            "complete" if success else "interrupted",
            hashes=hashes,
        )
