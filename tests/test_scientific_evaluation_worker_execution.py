import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.experiment_config import ExperimentConfig
from experiments import scientific_evaluation_worker as worker


SOURCE_COMMIT = "a" * 40


@pytest.fixture
def evaluation_case(tmp_path, monkeypatch):
    root = tmp_path / "evaluation"
    root.mkdir()

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

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
        "source_commit": SOURCE_COMMIT,
        "model_variant": "M1",
        "seed": 101,
        "training_step": 2,
        "manifest_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "online_weights_sha256": "3" * 64,
        "target_weights_sha256": "4" * 64,
    }

    request = {
        "schema_version": 1,
        "active_source_commit": SOURCE_COMMIT,
        "active_duration_ms": 10.0,
        "experiment_config": config.to_dict(),
        "checkpoint_provenance": copy.deepcopy(provenance),
    }

    request_path = root / "request.json"

    def write_request():
        request_path.write_text(
            json.dumps(request),
            encoding="utf-8",
        )

    write_request()

    monkeypatch.setenv("PYTHONHASHSEED", "101")
    monkeypatch.setenv("TF_DETERMINISTIC_OPS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    events = []

    def fake_provenance_validation(
        checkpoint_directory,
        active_source_commit,
    ):
        assert Path(checkpoint_directory) == checkpoint
        assert active_source_commit == SOURCE_COMMIT
        events.append("validate")
        return copy.deepcopy(provenance)

    def fake_seed(seed):
        events.append(f"seed:{seed}")

    summary_data = {
        "model_variant": "M1",
        "seed": 101,
        "traffic_scenario": "CONGESTED",
        "radio_scenario": "STATIC_GOOD",
        "active_duration_ms": 10.0,
        "drain_duration_ms": 1000.0,
        "conservation_ok": True,
    }

    def fake_evaluate(
        config,
        active_duration_ms,
        run_directory,
    ):
        assert config == deserialize_expected()
        assert active_duration_ms == 10.0
        assert Path(run_directory) == root / "simulation"
        events.append("evaluate")

        return SimpleNamespace(
            to_dict=lambda: copy.deepcopy(summary_data),
        )

    def deserialize_expected():
        return worker.deserialize_evaluation_config(
            request["experiment_config"]
        )

    monkeypatch.setattr(
        worker,
        "_current_source_commit",
        lambda: SOURCE_COMMIT,
        raising=False,
    )

    monkeypatch.setattr(
        worker,
        "validate_checkpoint_provenance",
        fake_provenance_validation,
        raising=False,
    )

    monkeypatch.setattr(
        worker,
        "_load_evaluation_runtime",
        lambda: (fake_seed, fake_evaluate),
        raising=False,
    )

    return SimpleNamespace(
        root=root,
        request=request,
        request_path=request_path,
        write_request=write_request,
        provenance=provenance,
        summary_data=summary_data,
        events=events,
    )


def execute(case):
    return worker.execute_evaluation_request(
        case.request_path
    )


def test_executes_and_saves_canonical_summary(evaluation_case):
    case = evaluation_case

    execute(case)

    assert case.events == [
        "validate",
        "seed:101",
        "evaluate",
        "validate",
    ]

    summary_path = case.root / "summary.json"
    assert summary_path.is_file()

    expected_bytes = json.dumps(
        case.summary_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    assert summary_path.read_bytes() == expected_bytes


def test_rejects_different_active_source(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    monkeypatch.setattr(
        worker,
        "_current_source_commit",
        lambda: "b" * 40,
        raising=False,
    )

    with pytest.raises(ValueError, match="source_commit"):
        execute(case)

    assert case.events == []
    assert not (case.root / "summary.json").exists()


def test_rejects_request_provenance_mismatch(
    evaluation_case,
):
    case = evaluation_case
    case.request["checkpoint_provenance"]["seed"] = 99
    case.write_request()

    with pytest.raises(ValueError, match="provenance"):
        execute(case)

    assert "evaluate" not in case.events
    assert not (case.root / "summary.json").exists()


def test_rejects_checkpoint_changes_during_execution(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case
    count = 0

    def changing_provenance(
        checkpoint_directory,
        active_source_commit,
    ):
        nonlocal count
        count += 1

        result = copy.deepcopy(case.provenance)

        if count == 2:
            result["online_weights_sha256"] = "9" * 64

        case.events.append("validate")
        return result

    monkeypatch.setattr(
        worker,
        "validate_checkpoint_provenance",
        changing_provenance,
    )

    with pytest.raises(ValueError, match="provenance"):
        execute(case)

    assert count == 2
    assert "evaluate" in case.events
    assert not (case.root / "summary.json").exists()


@pytest.mark.parametrize(
    "change",
    ["missing_field", "unknown_field"],
)
def test_rejects_invalid_request_schema(
    evaluation_case,
    change,
):
    case = evaluation_case

    if change == "missing_field":
        del case.request["active_duration_ms"]
    else:
        case.request["unknown_parameter"] = True

    case.write_request()

    with pytest.raises(ValueError, match="missing|unknown"):
        execute(case)

    assert case.events == []
    assert not (case.root / "summary.json").exists()

def _mock_execution_result(
    monkeypatch,
    case,
    summary=None,
    error=None,
):
    def fake_seed(seed):
        case.events.append(f"seed:{seed}")

    def fake_evaluate(**kwargs):
        case.events.append("evaluate")

        if error is not None:
            raise error

        return SimpleNamespace(
            to_dict=lambda: copy.deepcopy(summary)
        )

    monkeypatch.setattr(
        worker,
        "_load_evaluation_runtime",
        lambda: (fake_seed, fake_evaluate),
    )


def test_failed_simulation_does_not_publish(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    _mock_execution_result(
        monkeypatch,
        case,
        error=RuntimeError("simulation interrupted"),
    )

    with pytest.raises(
        RuntimeError,
        match="simulation interrupted",
    ):
        execute(case)

    assert "evaluate" in case.events
    assert not (case.root / "summary.json").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("seed", 999),
        ("conservation_ok", False),
    ],
)
def test_rejects_invalid_result_identity_or_conservation(
    evaluation_case,
    monkeypatch,
    field,
    bad_value,
):
    case = evaluation_case

    summary = copy.deepcopy(case.summary_data)
    summary[field] = bad_value

    _mock_execution_result(
        monkeypatch,
        case,
        summary=summary,
    )

    with pytest.raises(
        ValueError,
        match="evaluation summary",
    ):
        execute(case)

    assert not (case.root / "summary.json").exists()


def test_rejects_nonfinite_summary_value(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    summary = copy.deepcopy(case.summary_data)
    summary["diagnostic"] = float("nan")

    _mock_execution_result(
        monkeypatch,
        case,
        summary=summary,
    )

    with pytest.raises(ValueError):
        execute(case)

    assert not (case.root / "summary.json").exists()


def test_existing_summary_is_not_overwritten(
    evaluation_case,
):
    case = evaluation_case

    summary_path = case.root / "summary.json"
    summary_path.write_text(
        "existing scientific result",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        execute(case)

    assert summary_path.read_text(
        encoding="utf-8"
    ) == "existing scientific result"

    assert case.events == []


def test_existing_simulation_directory_is_preserved(
    evaluation_case,
):
    case = evaluation_case

    simulation = case.root / "simulation"
    simulation.mkdir()

    marker = simulation / "existing.txt"
    marker.write_text(
        "preserve",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        execute(case)

    assert marker.read_text(
        encoding="utf-8"
    ) == "preserve"

    assert case.events == []
    assert not (case.root / "summary.json").exists()


def test_invalid_request_json_is_rejected(
    evaluation_case,
):
    case = evaluation_case

    case.request_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid evaluation request JSON",
    ):
        execute(case)

    assert case.events == []
    assert not (case.root / "summary.json").exists()
