from pathlib import Path
from types import SimpleNamespace

import pytest

from config.experiment_config import ExperimentConfig
from experiments.checkpoint_schedule import CheckpointCandidate
from experiments.checkpoint_selection import CheckpointScore
from experiments.scientific_validation_orchestrator import (
    run_checkpoint_validation_grid,
)


SOURCE_COMMIT = "a" * 40


@pytest.fixture
def evaluation_case(tmp_path, monkeypatch):
    import experiments.scientific_validation_orchestrator as module

    checkpoint = (
        tmp_path / "training" / "checkpoints" / "update_00000002"
    )
    output = tmp_path / "validation"

    base_config = ExperimentConfig(
        model_variant="M1",
        seed=7,
        training_mode=False,
        checkpoint_path=str(checkpoint),
    )

    provenance = {
        "source_commit": SOURCE_COMMIT,
        "checkpoint_directory": str(checkpoint),
        "model_variant": "M1",
        "training_step": 2,
    }

    launched = []
    loaded = []
    aggregated = []

    score = CheckpointScore(
        urllc_deadline_miss_ratio=0.1,
        urllc_true_drop_ratio=0.01,
        worst_starvation_max_ms=1.0,
        combined_starvation_total_ms=2.0,
        embb_active_throughput_mbps=10.0,
        mmtc_completion_ratio=0.9,
        minimum_non_urllc_fairness=0.95,
        prb_utilization=0.8,
    )

    def fake_provenance(**kwargs):
        assert kwargs["checkpoint_directory"] == checkpoint
        assert kwargs["active_source_commit"] == SOURCE_COMMIT
        return provenance

    def fake_launch(config, active_duration_ms, run_directory):
        directory = Path(run_directory)

        assert config.training_mode is False
        assert config.checkpoint_path == str(checkpoint)
        assert active_duration_ms == 100.0
        assert not directory.exists()

        directory.mkdir(parents=True)
        launched.append((config, directory))

        return directory / "summary.json"

    def fake_load(
        evaluation_directory,
        *,
        checkpoint_directory,
        active_source_commit,
    ):
        assert checkpoint_directory == checkpoint
        assert active_source_commit == SOURCE_COMMIT

        config = next(
            config
            for config, directory in launched
            if directory == Path(evaluation_directory)
        )

        loaded.append(Path(evaluation_directory))

        return SimpleNamespace(
            model_variant=config.model_variant,
            traffic_scenario=config.traffic_scenario,
            radio_scenario=config.radio_scenario,
            seed=config.seed,
            control_interval_ms=config.control_interval_ms,
            active_duration_ms=100.0,
            drain_duration_ms=config.drain_duration_ms,
        )

    def fake_aggregate(summaries):
        aggregated.append(tuple(summaries))
        return score

    monkeypatch.setattr(
        module,
        "validate_checkpoint_provenance",
        fake_provenance,
    )
    monkeypatch.setattr(
        module,
        "launch_scientific_evaluation",
        fake_launch,
    )
    monkeypatch.setattr(
        module,
        "load_accepted_evaluation",
        fake_load,
    )
    monkeypatch.setattr(
        module,
        "aggregate_validation_suite",
        fake_aggregate,
    )

    return SimpleNamespace(
        module=module,
        checkpoint=checkpoint,
        output=output,
        base_config=base_config,
        launched=launched,
        loaded=loaded,
        aggregated=aggregated,
        score=score,
    )


def run(case, **overrides):
    options = {
        "checkpoint_directory": case.checkpoint,
        "base_config": case.base_config,
        "traffic_scenarios": ("BASELINE", "CONGESTED"),
        "radio_scenarios": ("STATIC_GOOD",),
        "seeds": (201, 202),
        "active_duration_ms": 100.0,
        "output_directory": case.output,
        "active_source_commit": SOURCE_COMMIT,
    }

    options.update(overrides)

    return run_checkpoint_validation_grid(**options)


def test_executes_and_verifies_every_declared_cell(evaluation_case):
    case = evaluation_case

    candidate = run(case)

    assert isinstance(candidate, CheckpointCandidate)
    assert candidate.training_step == 2
    assert candidate.score == case.score

    assert len(case.launched) == 4
    assert len(case.loaded) == 4
    assert len(case.aggregated) == 1
    assert len(case.aggregated[0]) == 4

    observed = {
        (
            config.traffic_scenario,
            config.radio_scenario,
            config.seed,
        )
        for config, _ in case.launched
    }

    assert observed == {
        ("BASELINE", "STATIC_GOOD", 201),
        ("BASELINE", "STATIC_GOOD", 202),
        ("CONGESTED", "STATIC_GOOD", 201),
        ("CONGESTED", "STATIC_GOOD", 202),
    }


def test_rejects_duplicate_declared_seed_before_launch(
    evaluation_case,
):
    case = evaluation_case

    with pytest.raises(ValueError, match="duplicate"):
        run(case, seeds=(201, 201))

    assert case.launched == []
    assert case.aggregated == []


def test_rejects_duplicate_observed_result(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def duplicate_loader(*args, **kwargs):
        return SimpleNamespace(
            model_variant="M1",
            traffic_scenario="BASELINE",
            radio_scenario="STATIC_GOOD",
            seed=201,
        )

    monkeypatch.setattr(
        case.module,
        "load_accepted_evaluation",
        duplicate_loader,
    )

    with pytest.raises(ValueError, match="duplicate"):
        run(case)

    assert case.aggregated == []


def test_failed_evaluation_never_produces_a_score(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def failed_launch(*args, **kwargs):
        raise RuntimeError("evaluation child failed")

    monkeypatch.setattr(
        case.module,
        "launch_scientific_evaluation",
        failed_launch,
    )

    with pytest.raises(RuntimeError, match="child failed"):
        run(case)

    assert case.aggregated == []


def test_rejects_checkpoint_source_mismatch(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    monkeypatch.setattr(
        case.module,
        "validate_checkpoint_provenance",
        lambda **kwargs: {
            "source_commit": "b" * 40,
            "checkpoint_directory": str(case.checkpoint),
            "model_variant": "M1",
            "training_step": 2,
        },
    )

    with pytest.raises(ValueError, match="source"):
        run(case)

    assert case.launched == []
    assert case.aggregated == []

def test_rejects_swapped_scenario_results(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    original_loader = (
        case.module.load_accepted_evaluation
    )

    def swapped_loader(*args, **kwargs):
        result = original_loader(*args, **kwargs)

        if result.traffic_scenario == "BASELINE":
            result.traffic_scenario = "CONGESTED"
        else:
            result.traffic_scenario = "BASELINE"

        return result

    monkeypatch.setattr(
        case.module,
        "load_accepted_evaluation",
        swapped_loader,
    )

    with pytest.raises(
        ValueError,
        match="requested grid cell",
    ):
        run(case)

    assert case.aggregated == []


def test_rejects_wrong_evaluation_duration(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    original_loader = (
        case.module.load_accepted_evaluation
    )

    def wrong_duration_loader(*args, **kwargs):
        result = original_loader(*args, **kwargs)
        result.active_duration_ms = 200.0
        return result

    monkeypatch.setattr(
        case.module,
        "load_accepted_evaluation",
        wrong_duration_loader,
    )

    with pytest.raises(
        ValueError,
        match="requested grid cell",
    ):
        run(case)

    assert case.aggregated == []


def test_rejects_existing_output_before_launch(
    evaluation_case,
):
    case = evaluation_case
    case.output.mkdir()

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        run(case)

    assert case.launched == []
    assert case.aggregated == []


def test_qos_ineligible_grid_never_returns_candidate(
    evaluation_case,
    monkeypatch,
):
    case = evaluation_case

    def reject_ineligible(summaries):
        raise ValueError(
            "URLLC scheduling-delay observations missing"
        )

    monkeypatch.setattr(
        case.module,
        "aggregate_validation_suite",
        reject_ineligible,
    )

    with pytest.raises(
        ValueError,
        match="scheduling-delay observations",
    ):
        run(case)

    assert len(case.launched) == 4
    assert len(case.loaded) == 4
