import subprocess
import sys

import pytest

from config.experiment_config import ExperimentConfig


def check_environment(config):
    from experiments.scientific_evaluation_worker import (
        validate_evaluation_process_environment,
    )

    return validate_evaluation_process_environment(config)


@pytest.fixture
def cpu_environment(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "101")
    monkeypatch.setenv("TF_DETERMINISTIC_OPS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def make_config():
    return ExperimentConfig(
        seed=101,
        training_mode=False,
        checkpoint_path="/tmp/synthetic-checkpoint",
    )


def test_worker_import_does_not_load_tensorflow():
    code = (
        "import sys\n"
        "import experiments.scientific_evaluation_worker\n"
        "assert 'tensorflow' not in sys.modules\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_accepts_deterministic_cpu_environment(cpu_environment):
    assert check_environment(make_config()) is True


@pytest.mark.parametrize("value", [None, "102"])
def test_rejects_incorrect_pythonhashseed(
    cpu_environment,
    monkeypatch,
    value,
):
    if value is None:
        monkeypatch.delenv("PYTHONHASHSEED")
    else:
        monkeypatch.setenv("PYTHONHASHSEED", value)

    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        check_environment(make_config())


@pytest.mark.parametrize("value", [None, "0"])
def test_rejects_nondeterministic_tensorflow(
    cpu_environment,
    monkeypatch,
    value,
):
    if value is None:
        monkeypatch.delenv("TF_DETERMINISTIC_OPS")
    else:
        monkeypatch.setenv("TF_DETERMINISTIC_OPS", value)

    with pytest.raises(RuntimeError, match="TF_DETERMINISTIC_OPS"):
        check_environment(make_config())


@pytest.mark.parametrize("value", [None, "0"])
def test_rejects_unvalidated_gpu_visibility(
    cpu_environment,
    monkeypatch,
    value,
):
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        check_environment(make_config())


def test_rejects_training_configuration(cpu_environment):
    config = ExperimentConfig(
        seed=101,
        training_mode=True,
    )

    with pytest.raises(ValueError, match="training_mode"):
        check_environment(config)


def test_rejects_non_configuration(cpu_environment):
    with pytest.raises(TypeError, match="ExperimentConfig"):
        check_environment(object())
