import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.experiment_config import ExperimentConfig


SOURCE = "dd379a961aaf360a38db2771cfe6d532b2c51dcc"


def get_launcher():
    return importlib.import_module(
        "experiments.scientific_evaluation_launcher"
    )


@pytest.fixture
def evaluation_case(tmp_path, monkeypatch):
    launcher = get_launcher()

    checkpoint = tmp_path / "training" / "checkpoints" / "update_00000002"
    checkpoint.mkdir(parents=True)

    config = ExperimentConfig(
        model_variant="M1",
        seed=101,
        training_mode=False,
        checkpoint_path=str(checkpoint),
        traffic_scenario="CONGESTED",
        radio_scenario="STATIC_GOOD",
    )

    provenance = {
        "checkpoint_directory": str(checkpoint.resolve()),
        "source_commit": SOURCE,
        "model_variant": "M1",
        "seed": 7,
        "training_step": 2,
        "manifest_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "online_weights_sha256": "3" * 64,
        "target_weights_sha256": "4" * 64,
    }

    output = tmp_path / "evaluation"

    monkeypatch.setattr(
        launcher,
        "_git_head",
        lambda root: SOURCE,
    )

    monkeypatch.setattr(
        launcher,
        "_require_clean_tracked_worktree",
        lambda root: None,
    )

    monkeypatch.setattr(
        launcher,
        "validate_checkpoint_provenance",
        lambda checkpoint_directory, active_source_commit:
            copy.deepcopy(provenance),
    )

    return SimpleNamespace(
        launcher=launcher,
        config=config,
        provenance=provenance,
        output=output,
    )


def launch(case):
    return case.launcher.launch_scientific_evaluation(
        config=case.config,
        active_duration_ms=10.0,
        run_directory=case.output,
    )


def test_launcher_import_does_not_load_tensorflow():
    code = (
        "import sys\n"
        "import experiments.scientific_evaluation_launcher\n"
        "assert 'tensorflow' not in sys.modules\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_launches_new_python_process(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case
    invocations = []

    monkeypatch.setenv("PYTHONHASHSEED", "unrelated")

    def fake_run(argv, **kwargs):
        invocations.append((argv, kwargs))

        assert argv == [
            sys.executable,
            "-m",
            "experiments.scientific_evaluation_worker",
            str(case.output / "request.json"),
        ]

        environment = kwargs["env"]

        assert environment["PYTHONHASHSEED"] == "101"
        assert environment["TF_DETERMINISTIC_OPS"] == "1"
        assert environment["CUDA_VISIBLE_DEVICES"] == ""

        request = json.loads(
            (case.output / "request.json").read_text(
                encoding="utf-8"
            )
        )

        assert request["schema_version"] == 1
        assert request["active_source_commit"] == SOURCE
        assert request["active_duration_ms"] == 10.0
        assert request["experiment_config"] == case.config.to_dict()
        assert request["checkpoint_provenance"] == case.provenance

        summary = _valid_child_summary()

        (case.output / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )

        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        case.launcher.subprocess,
        "run",
        fake_run,
    )

    result = launch(case)

    assert Path(result) == case.output / "summary.json"
    assert len(invocations) == 1
    assert os.environ["PYTHONHASHSEED"] == "unrelated"


def test_rejects_incompatible_source_before_output(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    monkeypatch.setattr(
        case.launcher,
        "_git_head",
        lambda root: "b" * 40,
    )

    with pytest.raises(ValueError, match="source_commit"):
        launch(case)

    assert not case.output.exists()


def test_preserves_existing_output_directory(
    evaluation_case,
):
    case = evaluation_case

    case.output.mkdir()
    marker = case.output / "original.txt"

    marker.write_text(
        "preserve",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        launch(case)

    assert marker.read_text(
        encoding="utf-8"
    ) == "preserve"


def test_rejects_dirty_repository(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def reject_dirty(root):
        raise RuntimeError("repository is dirty")

    monkeypatch.setattr(
        case.launcher,
        "_require_clean_tracked_worktree",
        reject_dirty,
    )

    with pytest.raises(RuntimeError, match="dirty"):
        launch(case)

    assert not case.output.exists()


def test_rejects_failed_child_process(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def fake_failure(argv, **kwargs):
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(
        case.launcher.subprocess,
        "run",
        fake_failure,
    )

    with pytest.raises(RuntimeError, match="child|exit|process"):
        launch(case)

    assert not (
        case.output / "summary.json"
    ).exists()

def _valid_child_summary():
    lifecycle = {
        "generated": 0,
        "delivered": 0,
        "dropped": 0,
        "residual": 0,
        "completion_ratio": 0.0,
        "true_drop_ratio": 0.0,
        "residual_ratio": 0.0,
    }

    performance = {
        "active_throughput_mbps": 0.0,
        "jain_fairness": None,
        "starvation": None,
    }

    urllc_qos = {
        "deadline_evaluated": 0,
        "deadline_misses": 0,
        "scheduling_deadline_miss_ratio": 0.0,
        "scheduling_delay_count": 0,
        "scheduling_delay_mean_ms": 0.0,
        "scheduling_delay_p95_ms": 0.0,
        "scheduling_delay_p99_ms": 0.0,
        "scheduling_delay_p99_9_ms": 0.0,
        "scheduling_delay_max_ms": 0.0,
        "completion_delay_count": 0,
        "completion_delay_mean_ms": 0.0,
        "completion_delay_p95_ms": 0.0,
        "completion_delay_p99_ms": 0.0,
        "completion_delay_p99_9_ms": 0.0,
        "completion_delay_max_ms": 0.0,
    }

    return {
        "model_variant": "M1",
        "seed": 101,
        "traffic_scenario": "CONGESTED",
        "radio_scenario": "STATIC_GOOD",
        "control_interval_ms": 1.0,
        "active_duration_ms": 10.0,
        "drain_duration_ms": 1000.0,
        "embb_lifecycle": copy.deepcopy(lifecycle),
        "urllc_lifecycle": copy.deepcopy(lifecycle),
        "mmtc_lifecycle": copy.deepcopy(lifecycle),
        "embb_performance": copy.deepcopy(performance),
        "mmtc_performance": copy.deepcopy(performance),
        "urllc_qos": urllc_qos,
        "system": {
            "prb_utilization": 0.0,
        },
        "conservation_ok": True,
    }


def _mock_child_result(
    case,
    monkeypatch,
    summary=None,
    returncode=0,
):
    if summary is None:
        summary = _valid_child_summary()

    def fake_run(argv, **kwargs):
        summary_path = case.output / "summary.json"

        summary_path.write_text(
            json.dumps(summary),
            encoding="utf-8",
        )

        return SimpleNamespace(
            returncode=returncode,
        )

    monkeypatch.setattr(
        case.launcher.subprocess,
        "run",
        fake_run,
    )


def test_publishes_verified_evaluation_manifest(
    evaluation_case,
    monkeypatch,
):
    import hashlib

    case = evaluation_case

    _mock_child_result(
        case,
        monkeypatch,
    )

    result = launch(case)

    assert result == case.output / "summary.json"

    manifest_path = (
        case.output / "evaluation_manifest.json"
    )

    assert manifest_path.is_file()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["status"] == "accepted"

    assert manifest["active_source_commit"] == SOURCE
    assert manifest["model_variant"] == "M1"
    assert manifest["evaluation_seed"] == 101
    assert manifest["active_duration_ms"] == 10.0

    assert (
        manifest["checkpoint_provenance"]
        == case.provenance
    )

    for filename, digest_field in (
        ("request.json", "request_sha256"),
        ("summary.json", "summary_sha256"),
        ("console.log", "console_sha256"),
    ):
        file_path = case.output / filename

        expected_hash = hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()

        assert manifest[digest_field] == expected_hash


def test_failed_child_has_no_acceptance_manifest(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    _mock_child_result(
        case,
        monkeypatch,
        returncode=7,
    )

    with pytest.raises(RuntimeError):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_changed_checkpoint_has_no_acceptance_manifest(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    _mock_child_result(
        case,
        monkeypatch,
    )

    validation_count = 0

    def changing_provenance(
        checkpoint_directory,
        active_source_commit,
    ):
        nonlocal validation_count

        validation_count += 1

        result = copy.deepcopy(case.provenance)

        if validation_count == 2:
            result["online_weights_sha256"] = "9" * 64

        return result

    monkeypatch.setattr(
        case.launcher,
        "validate_checkpoint_provenance",
        changing_provenance,
    )

    with pytest.raises(
        ValueError,
        match="provenance changed",
    ):
        launch(case)

    assert validation_count == 2

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_nonfinite_summary_has_no_acceptance_manifest(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    summary = _valid_child_summary()

    summary["urllc_qos"] = {
        "diagnostic_metric": float("nan")
    }

    _mock_child_result(
        case,
        monkeypatch,
        summary=summary,
    )

    with pytest.raises(ValueError):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()

def test_rejects_request_modified_by_child(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def fake_run(argv, **kwargs):
        request_path = case.output / "request.json"

        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )

        request["active_duration_ms"] = 99.0

        request_path.write_text(
            json.dumps(request),
            encoding="utf-8",
        )

        (case.output / "summary.json").write_text(
            json.dumps(_valid_child_summary()),
            encoding="utf-8",
        )

        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        case.launcher.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        ValueError,
        match="request.*changed|request.*hash",
    ):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_rejects_missing_child_summary(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    monkeypatch.setattr(
        case.launcher.subprocess,
        "run",
        lambda argv, **kwargs:
            SimpleNamespace(returncode=0),
    )

    with pytest.raises(
        RuntimeError,
        match="summary.json",
    ):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_rejects_malformed_child_summary(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def fake_run(argv, **kwargs):
        (case.output / "summary.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )

        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        case.launcher.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        ValueError,
        match="invalid evaluation summary JSON",
    ):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_rejects_wrong_summary_seed(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case
    summary = _valid_child_summary()
    summary["seed"] = 999

    _mock_child_result(
        case,
        monkeypatch,
        summary=summary,
    )

    with pytest.raises(
        ValueError,
        match="evaluation summary seed",
    ):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_rejects_failed_packet_conservation(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case
    summary = _valid_child_summary()
    summary["conservation_ok"] = False

    _mock_child_result(
        case,
        monkeypatch,
        summary=summary,
    )

    with pytest.raises(
        ValueError,
        match="conservation_ok",
    ):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()

def test_rejects_incomplete_run_summary(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    # Contains valid identity fields but omits the
    # scientific metrics required by RunSummary.
    incomplete = _valid_child_summary()
    del incomplete["urllc_qos"]

    _mock_child_result(
        case,
        monkeypatch,
        summary=incomplete,
    )

    with pytest.raises(ValueError, match="RunSummary"):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_urllc_percentile",
        "unknown_urllc_metric",
        "fractional_deadline_count",
        "boolean_deadline_count",
        "boolean_prb_utilization",
        "null_lifecycle_count",
        "invalid_performance_object",
        "infinite_scheduling_delay",
    ],
)
def test_rejects_invalid_nested_run_summary(
    evaluation_case,
    monkeypatch,
    invalid_case,
):
    case = evaluation_case
    summary = _valid_child_summary()

    if invalid_case == "missing_urllc_percentile":
        del summary["urllc_qos"]["scheduling_delay_p99_9_ms"]

    elif invalid_case == "unknown_urllc_metric":
        summary["urllc_qos"]["unknown_metric"] = 123

    elif invalid_case == "fractional_deadline_count":
        summary["urllc_qos"]["deadline_misses"] = 1.5

    elif invalid_case == "boolean_deadline_count":
        summary["urllc_qos"]["deadline_misses"] = True

    elif invalid_case == "boolean_prb_utilization":
        summary["system"]["prb_utilization"] = True

    elif invalid_case == "null_lifecycle_count":
        summary["embb_lifecycle"]["generated"] = None

    elif invalid_case == "invalid_performance_object":
        summary["mmtc_performance"] = []

    elif invalid_case == "infinite_scheduling_delay":
        summary["urllc_qos"]["scheduling_delay_p99_ms"] = (
            float("inf")
        )

    _mock_child_result(
        case,
        monkeypatch,
        summary=summary,
    )

    with pytest.raises(ValueError, match="RunSummary"):
        launch(case)

    assert not (
        case.output / "evaluation_manifest.json"
    ).exists()


def test_accepts_valid_optional_starvation_metrics(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case
    summary = _valid_child_summary()

    summary["embb_performance"]["jain_fairness"] = 1.0
    summary["embb_performance"]["starvation"] = {
        "event_count": 0,
        "total_ms": 0.0,
        "max_ms": 0.0,
    }

    _mock_child_result(
        case,
        monkeypatch,
        summary=summary,
    )

    result = launch(case)

    assert result == case.output / "summary.json"

    manifest = json.loads(
        (
            case.output / "evaluation_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "accepted"
