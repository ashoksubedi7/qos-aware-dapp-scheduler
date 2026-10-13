import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.experiment_config import ExperimentConfig
from experiments.checkpoint_schedule import (
    CheckpointCandidate,
)
from experiments.checkpoint_selection import (
    CheckpointScore,
)
from experiments.scientific_checkpoint_selection_runner import (
    run_scientific_checkpoint_selection,
)


SOURCE_COMMIT = "a" * 40


def make_score(deadline_miss_ratio):
    return CheckpointScore(
        urllc_deadline_miss_ratio=deadline_miss_ratio,
        urllc_true_drop_ratio=0.01,
        worst_starvation_max_ms=1.0,
        combined_starvation_total_ms=2.0,
        embb_active_throughput_mbps=10.0,
        mmtc_completion_ratio=0.9,
        minimum_non_urllc_fairness=0.95,
        prb_utilization=0.8,
    )


@pytest.fixture
def selection_case(tmp_path, monkeypatch):
    import experiments.scientific_checkpoint_selection_runner as module

    root = tmp_path / "training"

    checkpoints = tuple(
        root / "checkpoints" / f"update_{step:08d}"
        for step in (100, 200)
    )

    for checkpoint in checkpoints:
        checkpoint.mkdir(parents=True)

    # Synthetic original training manifest.
    # The numerical values are for tests only.
    manifest = {
        "training_protocol": {
            "successful_gradient_update_budget": 200,
            "epsilon_floor_update": 200,
            "checkpoint_interval_updates": 100,
        },
    }

    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    manifest_sha256 = hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()

    provenance = {
        checkpoint: {
            "source_commit": SOURCE_COMMIT,
            "checkpoint_directory": str(checkpoint),
            "model_variant": "M1",
            "seed": 7,
            "training_step": step,
            "run_definition_sha256": "same-training-run",
            "training_protocol_sha256": "same-protocol",
            "experiment_config_sha256": "same-config",
            "manifest_sha256": manifest_sha256,
        }
        for checkpoint, step in zip(
            checkpoints,
            (100, 200),
        )
    }

    calls = []

    def fake_provenance(
        *,
        checkpoint_directory,
        active_source_commit,
    ):
        return provenance[
            Path(checkpoint_directory)
        ]

    def fake_validation(**kwargs):
        checkpoint = Path(
            kwargs["checkpoint_directory"]
        )

        step = provenance[checkpoint]["training_step"]

        calls.append(kwargs)

        miss_ratio = {
            100: 0.2,
            200: 0.1,
        }[step]

        return CheckpointCandidate(
            training_step=step,
            score=make_score(miss_ratio),
        )

    monkeypatch.setattr(
        module,
        "validate_checkpoint_provenance",
        fake_provenance,
    )

    monkeypatch.setattr(
        module,
        "run_checkpoint_validation_grid",
        fake_validation,
    )

    config = ExperimentConfig(
        model_variant="M1",
        seed=201,
        training_mode=False,
        checkpoint_path=str(checkpoints[0]),
    )

    return SimpleNamespace(
        module=module,
        checkpoints=checkpoints,
        provenance=provenance,
        calls=calls,
        config=config,
        output=tmp_path / "selection",
    )


def run(case, **overrides):
    options = {
        "checkpoint_directories": case.checkpoints,
        "expected_training_steps": (100, 200),
        "base_config": case.config,
        "traffic_scenarios": (
            "BASELINE",
            "CONGESTED",
        ),
        "radio_scenarios": (
            "STATIC_GOOD",
        ),
        "seeds": (201, 202),
        "active_duration_ms": 100.0,
        "output_directory": case.output,
        "active_source_commit": SOURCE_COMMIT,
    }

    options.update(overrides)

    return run_scientific_checkpoint_selection(
        **options
    )


def test_selects_best_qos_checkpoint(selection_case):
    case = selection_case

    selected = run(case)

    assert selected.checkpoint_directory == (
        case.checkpoints[1]
    )
    assert selected.candidate.training_step == 200
    assert len(case.calls) == 2


def test_rejects_missing_checkpoint_before_evaluation(
    selection_case,
):
    case = selection_case

    with pytest.raises(ValueError, match="missing"):
        run(
            case,
            checkpoint_directories=(
                case.checkpoints[0],
            ),
        )

    assert case.calls == []


def test_rejects_unexpected_checkpoint_before_evaluation(
    selection_case,
):
    case = selection_case

    with pytest.raises(ValueError, match="unexpected"):
        run(
            case,
            expected_training_steps=(100,),
        )

    assert case.calls == []


def test_rejects_mixed_training_runs(
    selection_case,
):
    case = selection_case

    case.provenance[
        case.checkpoints[1]
    ]["run_definition_sha256"] = "different-run"

    with pytest.raises(ValueError, match="training run"):
        run(case)

    assert case.calls == []


def test_rejects_checkpoint_source_mismatch(
    selection_case,
):
    case = selection_case

    case.provenance[
        case.checkpoints[1]
    ]["source_commit"] = "b" * 40

    with pytest.raises(ValueError, match="source"):
        run(case)

    assert case.calls == []


def test_rejects_training_seed_in_validation_grid(
    selection_case,
):
    case = selection_case

    with pytest.raises(ValueError, match="training seed"):
        run(
            case,
            seeds=(7, 201),
        )

    assert case.calls == []


def test_rejects_existing_selection_directory(
    selection_case,
):
    case = selection_case

    case.output.mkdir()

    with pytest.raises(FileExistsError):
        run(case)

    assert case.calls == []


def test_failed_checkpoint_validation_blocks_selection(
    selection_case,
    monkeypatch,
):
    case = selection_case

    original = (
        case.module.run_checkpoint_validation_grid
    )

    def failing_validation(**kwargs):
        checkpoint = Path(
            kwargs["checkpoint_directory"]
        )

        if checkpoint == case.checkpoints[1]:
            raise RuntimeError(
                "checkpoint validation failed"
            )

        return original(**kwargs)

    monkeypatch.setattr(
        case.module,
        "run_checkpoint_validation_grid",
        failing_validation,
    )

    with pytest.raises(
        RuntimeError,
        match="validation failed",
    ):
        run(case)

    assert len(case.calls) == 1

def test_rejects_subset_of_original_checkpoint_schedule(
    selection_case,
):
    case = selection_case

    # The manifest requires checkpoints 100 and 200.
    # Supplying only 100 must not permit selection.
    with pytest.raises(
        ValueError,
        match="schedule|missing",
    ):
        run(
            case,
            checkpoint_directories=(
                case.checkpoints[0],
            ),
            expected_training_steps=(100,),
        )

    # Reject before starting any evaluation.
    assert case.calls == []

def test_rejects_modified_training_manifest(
    selection_case,
):
    case = selection_case

    manifest_path = (
        case.checkpoints[0].parent.parent / "manifest.json"
    )

    # The JSON remains valid, but its bytes change.
    manifest_path.write_bytes(
        manifest_path.read_bytes() + b"\\n"
    )

    with pytest.raises(
        ValueError,
        match="manifest hash",
    ):
        run(case)

    assert case.calls == []


def test_rejects_inconsistent_manifest_schedule(
    selection_case,
):
    case = selection_case

    manifest_path = (
        case.checkpoints[0].parent.parent / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    # Original schedule: 100, 200.
    # Modified schedule: 200 only.
    manifest["training_protocol"][
        "checkpoint_interval_updates"
    ] = 200

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    # Simulate internally consistent artifact hashes.
    # The original declared steps must still disagree
    # with this changed manifest schedule.
    digest = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    for provenance in case.provenance.values():
        provenance["manifest_sha256"] = digest

    with pytest.raises(
        ValueError,
        match="schedule",
    ):
        run(case)

    assert case.calls == []


def test_exact_qos_tie_selects_earlier_checkpoint(
    selection_case,
    monkeypatch,
):
    case = selection_case

    original = (
        case.module.run_checkpoint_validation_grid
    )

    def equal_score_validation(**kwargs):
        candidate = original(**kwargs)

        return CheckpointCandidate(
            training_step=candidate.training_step,
            score=make_score(0.1),
        )

    monkeypatch.setattr(
        case.module,
        "run_checkpoint_validation_grid",
        equal_score_validation,
    )

    selected = run(case)

    assert selected.candidate.training_step == 100
    assert selected.checkpoint_directory == (
        case.checkpoints[0]
    )
    assert len(case.calls) == 2
