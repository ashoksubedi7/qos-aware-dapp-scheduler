import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from config.experiment_config import ExperimentConfig
from metrics.run_summary import (
    RunSummary,
    SliceLifecycleSummary,
    SlicePerformanceSummary,
    StarvationSummary,
    SystemSummary,
    URLLCQoSSummary,
)
from experiments.scientific_validation_artifacts import (
    load_accepted_evaluation,
)


SOURCE_COMMIT = "a" * 40


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_summary():
    lifecycle = SliceLifecycleSummary(
        generated=10,
        delivered=9,
        dropped=1,
        residual=0,
        completion_ratio=0.9,
        true_drop_ratio=0.1,
        residual_ratio=0.0,
    )

    performance = SlicePerformanceSummary(
        active_throughput_mbps=10.0,
        jain_fairness=0.95,
        starvation=StarvationSummary(
            event_count=0,
            total_ms=0.0,
            max_ms=0.0,
        ),
    )

    qos = URLLCQoSSummary(
        deadline_evaluated=10,
        deadline_misses=1,
        scheduling_deadline_miss_ratio=0.1,
        scheduling_delay_count=10,
        scheduling_delay_mean_ms=0.5,
        scheduling_delay_p95_ms=0.8,
        scheduling_delay_p99_ms=0.9,
        scheduling_delay_p99_9_ms=1.0,
        scheduling_delay_max_ms=1.1,
        completion_delay_count=10,
        completion_delay_mean_ms=1.0,
        completion_delay_p95_ms=1.5,
        completion_delay_p99_ms=2.0,
        completion_delay_p99_9_ms=2.5,
        completion_delay_max_ms=3.0,
    )

    return RunSummary(
        model_variant="M1",
        seed=201,
        traffic_scenario="CONGESTED",
        radio_scenario="STATIC_GOOD",
        control_interval_ms=1.0,
        active_duration_ms=100.0,
        drain_duration_ms=1000.0,
        embb_lifecycle=lifecycle,
        urllc_lifecycle=lifecycle,
        mmtc_lifecycle=lifecycle,
        embb_performance=performance,
        mmtc_performance=performance,
        urllc_qos=qos,
        system=SystemSummary(
            prb_utilization=0.5,
        ),
        conservation_ok=True,
    )


@pytest.fixture
def accepted_case(tmp_path, monkeypatch):
    import experiments.scientific_validation_artifacts as module

    checkpoint = (
        tmp_path
        / "training"
        / "checkpoints"
        / "update_00000002"
    )
    checkpoint.mkdir(parents=True)

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()

    config = ExperimentConfig(
        model_variant="M1",
        seed=201,
        training_mode=False,
        checkpoint_path=str(checkpoint),
        traffic_scenario="CONGESTED",
        radio_scenario="STATIC_GOOD",
    )

    provenance = {
        "source_commit": SOURCE_COMMIT,
        "checkpoint_directory": str(checkpoint),
        "model_variant": "M1",
        "seed": 7,
        "training_step": 2,
    }

    request = {
        "schema_version": 1,
        "active_source_commit": SOURCE_COMMIT,
        "active_duration_ms": 100.0,
        "experiment_config": config.to_dict(),
        "checkpoint_provenance": provenance,
    }

    (evaluation / "request.json").write_text(
        json.dumps(request),
        encoding="utf-8",
    )

    (evaluation / "summary.json").write_text(
        json.dumps(asdict(make_summary())),
        encoding="utf-8",
    )

    (evaluation / "console.log").write_text(
        "synthetic evaluation log",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "active_source_commit": SOURCE_COMMIT,
        "model_variant": "M1",
        "evaluation_seed": 201,
        "active_duration_ms": 100.0,
        "checkpoint_provenance": provenance,
        "request_sha256": digest(
            evaluation / "request.json"
        ),
        "summary_sha256": digest(
            evaluation / "summary.json"
        ),
        "console_sha256": digest(
            evaluation / "console.log"
        ),
    }

    (evaluation / "evaluation_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "validate_checkpoint_provenance",
        lambda **kwargs: provenance,
    )

    return evaluation, checkpoint


def load_case(evaluation, checkpoint, **overrides):
    options = {
        "checkpoint_directory": checkpoint,
        "active_source_commit": SOURCE_COMMIT,
    }
    options.update(overrides)

    return load_accepted_evaluation(
        evaluation,
        **options,
    )


def test_loads_verified_accepted_summary(accepted_case):
    evaluation, checkpoint = accepted_case

    summary = load_case(evaluation, checkpoint)

    assert isinstance(summary, RunSummary)
    assert summary.model_variant == "M1"
    assert summary.seed == 201
    assert summary.conservation_ok is True
    assert summary.urllc_qos.deadline_misses == 1


def test_rejects_modified_summary(accepted_case):
    evaluation, checkpoint = accepted_case

    path = evaluation / "summary.json"
    summary = json.loads(path.read_text())

    summary["system"]["prb_utilization"] = 0.9

    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="hash"):
        load_case(evaluation, checkpoint)


def test_rejects_modified_request(accepted_case):
    evaluation, checkpoint = accepted_case

    path = evaluation / "request.json"
    request = json.loads(path.read_text())

    request["active_duration_ms"] = 200.0

    path.write_text(json.dumps(request))

    with pytest.raises(ValueError, match="hash"):
        load_case(evaluation, checkpoint)


def test_rejects_wrong_source_commit(accepted_case):
    evaluation, checkpoint = accepted_case

    with pytest.raises(ValueError, match="source"):
        load_case(
            evaluation,
            checkpoint,
            active_source_commit="b" * 40,
        )


def test_rejects_wrong_checkpoint(accepted_case, tmp_path):
    evaluation, _ = accepted_case

    other_checkpoint = tmp_path / "other_checkpoint"
    other_checkpoint.mkdir()

    with pytest.raises(ValueError, match="checkpoint"):
        load_case(evaluation, other_checkpoint)


def test_rejects_missing_acceptance_manifest(accepted_case):
    evaluation, checkpoint = accepted_case

    (evaluation / "evaluation_manifest.json").unlink()

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="evaluation_manifest.json",
    ):
        load_case(evaluation, checkpoint)


def test_rejects_nonaccepted_manifest(accepted_case):
    evaluation, checkpoint = accepted_case

    path = evaluation / "evaluation_manifest.json"
    manifest = json.loads(path.read_text())

    manifest["status"] = "rejected"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="accepted"):
        load_case(evaluation, checkpoint)

@pytest.mark.parametrize(
    "invalid_case",
    [
        "packet_count_balance",
        "deadline_miss_ratio",
        "prb_utilization_range",
        "negative_starvation_duration",
    ],
)
def test_rejects_semantically_inconsistent_summary(
    accepted_case,
    invalid_case,
):
    evaluation, checkpoint = accepted_case

    summary_path = evaluation / "summary.json"
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if invalid_case == "packet_count_balance":
        summary["urllc_lifecycle"]["delivered"] = 15

    elif invalid_case == "deadline_miss_ratio":
        summary["urllc_qos"][
            "scheduling_deadline_miss_ratio"
        ] = 0.8

    elif invalid_case == "prb_utilization_range":
        summary["system"]["prb_utilization"] = 1.5

    elif invalid_case == "negative_starvation_duration":
        summary["embb_performance"]["starvation"][
            "total_ms"
        ] = -1.0

    summary_path.write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    # Recompute the hash so this test isolates
    # semantic consistency, not hash verification.
    manifest_path = (
        evaluation / "evaluation_manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    manifest["summary_sha256"] = digest(
        summary_path
    )

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_case(evaluation, checkpoint)
