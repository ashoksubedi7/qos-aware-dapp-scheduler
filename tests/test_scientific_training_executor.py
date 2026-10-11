from pathlib import Path
from types import SimpleNamespace

import pytest

from config.experiment_config import ExperimentConfig
from config.training_protocol import TrainingProtocol
from experiments.scientific_training_setup import (
    build_scientific_training_setup,
)

import experiments.scientific_training_executor as executor_module


SOURCE_COMMIT = (
    "0123456789abcdef0123456789abcdef01234567"
)


def _protocol():
    # Fixture values only. They are not study authority.
    return TrainingProtocol(
        successful_gradient_update_budget=5,
        epsilon_floor_update=4,
        checkpoint_interval_updates=2,
    )


def _config():
    return ExperimentConfig(
        model_variant="M1",
        seed=101,
        training_mode=True,
        traffic_scenario="CONGESTED",
        radio_scenario="STATIC_GOOD",
    )


class _ProbeAgent:
    def __init__(self):
        self.training_steps = 0


class _ProbeScheduler:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.agent = _ProbeAgent()

        self.learning_enabled = True
        self.callback = None
        self.slice_calls = []

    def set_successful_update_callback(
        self,
        callback,
    ):
        self.callback = callback

    def createSlice(
        self,
        *args,
    ):
        self.slice_calls.append(
            args
        )

    def resAlloc(
        self,
        env,
    ):
        if False:
            yield env


class _ProbeCell:
    def __init__(
        self,
        scheduler,
    ):
        self.interSliceSched = scheduler


class _ProbeGroup:
    def __init__(
        self,
        label,
    ):
        self.label = label

        self.req = {
            "reqDelay": 1,
            "reqThroughputDL": 1,
            "reqThroughputUL": 0,
            "reqAvailability": 1,
        }

        self.num_usersDL = 1
        self.num_usersUL = 0
        self.mmMd = False
        self.lyrs = 1
        self.sch = "Rr"

        self.activation_calls = []

    def activateSliceScheds(
        self,
        inter_slice,
        env,
    ):
        self.activation_calls.append(
            (
                inter_slice,
                env,
            )
        )


class _ProbeController:
    def __init__(
        self,
        protocol,
        checkpoint_root,
        base_checkpoint_metadata=None,
    ):
        self.protocol = protocol
        self.checkpoint_root = Path(
            checkpoint_root
        )

        self.base_checkpoint_metadata = dict(
            base_checkpoint_metadata or {}
        )

        self._observed_steps = []
        self._saved_checkpoint_steps = []

    @property
    def observed_steps(self):
        return tuple(
            self._observed_steps
        )

    @property
    def saved_checkpoint_steps(self):
        return tuple(
            self._saved_checkpoint_steps
        )

    def __call__(
        self,
        scheduler,
        training_step,
        training_loss,
    ):
        return None


class _ProbeEnvironment:
    def __init__(
        self,
        state,
        protocol,
        fail_run=False,
    ):
        self.state = state
        self.protocol = protocol
        self.fail_run = fail_run

        self.process_calls = []
        self.run_calls = []

    def process(
        self,
        process,
    ):
        self.process_calls.append(
            process
        )

        return process

    def run(
        self,
        until,
    ):
        self.run_calls.append(
            float(until)
        )

        if self.fail_run:
            raise RuntimeError(
                "simulation failure"
            )

        scheduler = self.state[
            "scheduler"
        ]

        controller = self.state[
            "controller"
        ]

        budget = (
            self.protocol
            .successful_gradient_update_budget
        )

        scheduler.agent.training_steps = (
            budget
        )

        scheduler.learning_enabled = False

        controller._observed_steps = list(
            range(
                1,
                budget + 1,
            )
        )

        controller._saved_checkpoint_steps = [
            2,
            4,
            5,
        ]


def _install_probe_runtime(
    monkeypatch,
    tmp_path,
    fail_run=False,
):
    state = {
        "manifest_calls": [],
        "cell_calls": [],
        "groups": [],
        "close_calls": [],
        "summary_calls": [],
    }

    protocol = _protocol()

    environment = _ProbeEnvironment(
        state=state,
        protocol=protocol,
        fail_run=fail_run,
    )

    monkeypatch.setattr(
        executor_module.simpy,
        "Environment",
        lambda: environment,
    )

    provenance = {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "training_protocol_sha256": (
            "a" * 64
        ),
        "experiment_config_sha256": (
            "b" * 64
        ),
        "run_definition_sha256": (
            "c" * 64
        ),
    }

    def write_manifest(
        run_directory,
        protocol,
        config,
        source_commit,
    ):
        run_directory = Path(
            run_directory
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            run_directory
            / "manifest.json"
        )

        manifest_path.write_text(
            "{}",
            encoding="utf-8",
        )

        state["manifest_calls"].append(
            {
                "run_directory": (
                    run_directory
                ),
                "protocol": protocol,
                "config": config,
                "source_commit": (
                    source_commit
                ),
            }
        )

        return (
            manifest_path,
            provenance,
        )

    monkeypatch.setattr(
        executor_module,
        "write_scientific_run_manifest",
        write_manifest,
    )

    def controller_factory(
        protocol,
        checkpoint_root,
        base_checkpoint_metadata=None,
    ):
        controller = _ProbeController(
            protocol=protocol,
            checkpoint_root=(
                checkpoint_root
            ),
            base_checkpoint_metadata=(
                base_checkpoint_metadata
            ),
        )

        state["controller"] = controller

        return controller

    monkeypatch.setattr(
        executor_module,
        "TrainingUpdateController",
        controller_factory,
    )

    def cell_factory(
        *args,
        experiment_config=None,
        **kwargs,
    ):
        scheduler = _ProbeScheduler(
            experiment_config
        )

        state["scheduler"] = scheduler

        state["cell_calls"].append(
            {
                "args": args,
                "kwargs": kwargs,
                "config": experiment_config,
            }
        )

        return _ProbeCell(
            scheduler
        )

    monkeypatch.setattr(
        executor_module,
        "Cell",
        cell_factory,
    )

    traffic = SimpleNamespace(
        name="CONGESTED",
        embb=object(),
        urllc=object(),
        mmtc=object(),
    )

    monkeypatch.setattr(
        executor_module,
        "get_traffic_scenario",
        lambda name: traffic,
    )

    def make_group(
        profile,
        label,
        delay_requirement,
        scheduling_deadline,
        cell,
        t_sim,
        meas_interval,
        env,
        config,
    ):
        group = _ProbeGroup(
            label
        )

        state["groups"].append(
            {
                "group": group,
                "profile": profile,
                "label": label,
                "deadline": (
                    scheduling_deadline
                ),
                "t_sim": float(
                    t_sim
                ),
                "config": config,
            }
        )

        return group

    monkeypatch.setattr(
        executor_module,
        "_make_group",
        make_group,
    )

    def close_scheduler_files(
        scheduler,
    ):
        state["close_calls"].append(
            scheduler
        )

    monkeypatch.setattr(
        executor_module,
        "_close_scheduler_files",
        close_scheduler_files,
    )

    summary = object()
    state["summary"] = summary

    def build_summary(
        scheduler,
        t_sim_ms,
    ):
        state["summary_calls"].append(
            (
                scheduler,
                float(t_sim_ms),
            )
        )

        return summary

    monkeypatch.setattr(
        executor_module,
        "build_run_summary",
        build_summary,
    )

    state["environment"] = environment
    state["provenance"] = provenance
    state["protocol"] = protocol

    return state


def test_executor_runs_once_through_total_horizon_and_returns_summary(
    monkeypatch,
    tmp_path,
):
    state = _install_probe_runtime(
        monkeypatch,
        tmp_path,
    )

    protocol = state["protocol"]
    config = _config()

    expected_setup = (
        build_scientific_training_setup(
            protocol=protocol,
            config=config,
        )
    )

    result = (
        executor_module.run_scientific_training(
            protocol=protocol,
            config=config,
            run_directory=tmp_path / "run",
            source_commit=SOURCE_COMMIT,
        )
    )

    assert result is state["summary"]

    assert state[
        "environment"
    ].run_calls == [
        expected_setup
        .horizon
        .total_duration_ms
    ]

    assert (
        state["scheduler"].callback
        is state["controller"]
    )

    assert state["summary_calls"] == [
        (
            state["scheduler"],
            expected_setup
            .horizon
            .total_duration_ms,
        )
    ]

    assert state["close_calls"] == [
        state["scheduler"]
    ]

    assert len(
        state["scheduler"].slice_calls
    ) == 3

    assert [
        entry["label"]
        for entry in state["groups"]
    ] == [
        "eMBB",
        "URLLC",
        "mMTC",
    ]

    for entry in state["groups"]:
        assert (
            entry["group"]
            .activation_calls
        ) == [
            (
                state["scheduler"],
                state["environment"],
            )
        ]


def test_executor_links_manifest_provenance_to_checkpoints(
    monkeypatch,
    tmp_path,
):
    state = _install_probe_runtime(
        monkeypatch,
        tmp_path,
    )

    protocol = state["protocol"]
    config = _config()

    expected_setup = (
        build_scientific_training_setup(
            protocol=protocol,
            config=config,
        )
    )

    executor_module.run_scientific_training(
        protocol=protocol,
        config=config,
        run_directory=tmp_path / "run",
        source_commit=SOURCE_COMMIT,
    )

    assert len(
        state["manifest_calls"]
    ) == 1

    manifest_call = (
        state["manifest_calls"][0]
    )

    assert (
        manifest_call["config"]
        == expected_setup.config
    )

    assert (
        manifest_call["source_commit"]
        == SOURCE_COMMIT
    )

    controller = state[
        "controller"
    ]

    assert controller.checkpoint_root == (
        tmp_path
        / "run"
        / "checkpoints"
    )

    assert (
        controller.base_checkpoint_metadata
        == {
            "source_commit": (
                SOURCE_COMMIT
            ),
            "training_protocol_sha256": (
                "a" * 64
            ),
            "experiment_config_sha256": (
                "b" * 64
            ),
            "run_definition_sha256": (
                "c" * 64
            ),
        }
    )


def test_executor_closes_scheduler_files_on_simulation_failure(
    monkeypatch,
    tmp_path,
):
    state = _install_probe_runtime(
        monkeypatch,
        tmp_path,
        fail_run=True,
    )

    with pytest.raises(
        RuntimeError,
        match="simulation failure",
    ):
        executor_module.run_scientific_training(
            protocol=state["protocol"],
            config=_config(),
            run_directory=tmp_path / "run",
            source_commit=SOURCE_COMMIT,
        )

    assert state["close_calls"] == [
        state["scheduler"]
    ]

    assert state["summary_calls"] == []
