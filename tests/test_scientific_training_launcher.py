import os
from pathlib import Path
import subprocess
import sys

import pytest

from config.experiment_config import (
    ExperimentConfig,
)
from config.training_protocol import (
    TrainingProtocol,
)

import experiments.scientific_training_launcher as launcher_module


ROOT = Path(__file__).resolve().parents[1]


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


def _install_valid_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "PYTHONHASHSEED",
        "101",
    )

    monkeypatch.setenv(
        "TF_DETERMINISTIC_OPS",
        "1",
    )

    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "",
    )


def test_launcher_import_does_not_import_tensorflow():
    env = os.environ.copy()

    existing = env.get(
        "PYTHONPATH",
        "",
    )

    paths = [
        str(ROOT / "src"),
    ]

    if existing:
        paths.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(
        paths
    )

    probe = r'''
import sys

import experiments.scientific_training_launcher

if "tensorflow" in sys.modules:
    raise SystemExit(
        "launcher import loaded TensorFlow"
    )
'''

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "scientific launcher import must not "
        "load TensorFlow:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_process_environment_accepts_validated_cpu_path(
    monkeypatch,
):
    _install_valid_environment(
        monkeypatch
    )

    assert (
        launcher_module
        .validate_scientific_process_environment(
            protocol=_protocol(),
            config=_config(),
        )
        is True
    )


@pytest.mark.parametrize(
    "pythonhashseed",
    (
        None,
        "100",
        "0101",
    ),
)
def test_process_environment_requires_matching_pythonhashseed(
    monkeypatch,
    pythonhashseed,
):
    _install_valid_environment(
        monkeypatch
    )

    if pythonhashseed is None:
        monkeypatch.delenv(
            "PYTHONHASHSEED",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "PYTHONHASHSEED",
            pythonhashseed,
        )

    with pytest.raises(
        RuntimeError,
        match="PYTHONHASHSEED",
    ):
        launcher_module.validate_scientific_process_environment(
            protocol=_protocol(),
            config=_config(),
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        "0",
        "true",
    ),
)
def test_process_environment_requires_tensorflow_determinism(
    monkeypatch,
    value,
):
    _install_valid_environment(
        monkeypatch
    )

    if value is None:
        monkeypatch.delenv(
            "TF_DETERMINISTIC_OPS",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "TF_DETERMINISTIC_OPS",
            value,
        )

    with pytest.raises(
        RuntimeError,
        match="TF_DETERMINISTIC_OPS",
    ):
        launcher_module.validate_scientific_process_environment(
            protocol=_protocol(),
            config=_config(),
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        "-1",
        "0",
    ),
)
def test_process_environment_requires_validated_cpu_visibility(
    monkeypatch,
    value,
):
    _install_valid_environment(
        monkeypatch
    )

    if value is None:
        monkeypatch.delenv(
            "CUDA_VISIBLE_DEVICES",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "CUDA_VISIBLE_DEVICES",
            value,
        )

    with pytest.raises(
        RuntimeError,
        match="CUDA_VISIBLE_DEVICES",
    ):
        launcher_module.validate_scientific_process_environment(
            protocol=_protocol(),
            config=_config(),
        )


SOURCE_COMMIT = (
    "0123456789abcdef0123456789abcdef01234567"
)


def test_launch_derives_git_identity_and_delegates(
    monkeypatch,
    tmp_path,
):
    _install_valid_environment(
        monkeypatch
    )

    protocol = _protocol()
    config = _config()

    repo_root = ROOT
    run_directory = (
        tmp_path
        / "scientific-run"
    )

    events = []
    expected_result = object()

    monkeypatch.setattr(
        launcher_module,
        "_repository_root",
        lambda: repo_root,
        raising=False,
    )

    def require_clean(
        value,
    ):
        events.append(
            (
                "clean",
                Path(value),
            )
        )

    monkeypatch.setattr(
        launcher_module,
        "_require_clean_tracked_worktree",
        require_clean,
        raising=False,
    )

    def git_head(
        value,
    ):
        events.append(
            (
                "head",
                Path(value),
            )
        )

        return SOURCE_COMMIT

    monkeypatch.setattr(
        launcher_module,
        "_git_head",
        git_head,
        raising=False,
    )

    def set_global_seed(
        seed,
    ):
        events.append(
            (
                "seed",
                seed,
            )
        )

        return seed

    def run_training(
        protocol,
        config,
        run_directory,
        source_commit,
    ):
        events.append(
            (
                "execute",
                protocol,
                config,
                Path(run_directory),
                source_commit,
            )
        )

        return expected_result

    def load_runtime():
        events.append(
            (
                "load_runtime",
            )
        )

        return (
            set_global_seed,
            run_training,
        )

    monkeypatch.setattr(
        launcher_module,
        "_load_training_runtime",
        load_runtime,
        raising=False,
    )

    result = (
        launcher_module
        .launch_scientific_training(
            protocol=protocol,
            config=config,
            run_directory=run_directory,
        )
    )

    assert result is expected_result

    assert events == [
        (
            "clean",
            repo_root,
        ),
        (
            "head",
            repo_root,
        ),
        (
            "load_runtime",
        ),
        (
            "seed",
            config.seed,
        ),
        (
            "execute",
            protocol,
            config,
            run_directory,
            SOURCE_COMMIT,
        ),
    ]


def test_launch_validates_process_before_git_checks(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        "PYTHONHASHSEED",
        raising=False,
    )

    monkeypatch.setenv(
        "TF_DETERMINISTIC_OPS",
        "1",
    )

    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "",
    )

    monkeypatch.setattr(
        launcher_module,
        "_repository_root",
        lambda: pytest.fail(
            "repository inspection occurred "
            "before process validation"
        ),
        raising=False,
    )

    monkeypatch.setattr(
        launcher_module,
        "_load_training_runtime",
        lambda: pytest.fail(
            "training runtime loaded before "
            "process validation"
        ),
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="PYTHONHASHSEED",
    ):
        launcher_module.launch_scientific_training(
            protocol=_protocol(),
            config=_config(),
            run_directory=(
                tmp_path
                / "run"
            ),
        )


def test_launch_refuses_dirty_tracked_worktree_before_runtime(
    monkeypatch,
    tmp_path,
):
    _install_valid_environment(
        monkeypatch
    )

    repo_root = ROOT

    monkeypatch.setattr(
        launcher_module,
        "_repository_root",
        lambda: repo_root,
        raising=False,
    )

    def dirty_worktree(
        value,
    ):
        assert Path(value) == repo_root

        raise RuntimeError(
            "tracked worktree is not clean"
        )

    monkeypatch.setattr(
        launcher_module,
        "_require_clean_tracked_worktree",
        dirty_worktree,
        raising=False,
    )

    monkeypatch.setattr(
        launcher_module,
        "_git_head",
        lambda value: pytest.fail(
            "HEAD read despite dirty worktree"
        ),
        raising=False,
    )

    monkeypatch.setattr(
        launcher_module,
        "_load_training_runtime",
        lambda: pytest.fail(
            "training runtime loaded despite "
            "dirty worktree"
        ),
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="not clean",
    ):
        launcher_module.launch_scientific_training(
            protocol=_protocol(),
            config=_config(),
            run_directory=(
                tmp_path
                / "run"
            ),
        )


def test_launch_reads_head_before_loading_training_runtime(
    monkeypatch,
    tmp_path,
):
    _install_valid_environment(
        monkeypatch
    )

    state = {
        "head_read": False,
    }

    monkeypatch.setattr(
        launcher_module,
        "_repository_root",
        lambda: ROOT,
        raising=False,
    )

    monkeypatch.setattr(
        launcher_module,
        "_require_clean_tracked_worktree",
        lambda value: None,
        raising=False,
    )

    def git_head(
        value,
    ):
        state["head_read"] = True
        return SOURCE_COMMIT

    monkeypatch.setattr(
        launcher_module,
        "_git_head",
        git_head,
        raising=False,
    )

    def load_runtime():
        assert state["head_read"] is True

        return (
            lambda seed: seed,
            lambda **kwargs: object(),
        )

    monkeypatch.setattr(
        launcher_module,
        "_load_training_runtime",
        load_runtime,
        raising=False,
    )

    launcher_module.launch_scientific_training(
        protocol=_protocol(),
        config=_config(),
        run_directory=(
            tmp_path
            / "run"
        ),
    )
