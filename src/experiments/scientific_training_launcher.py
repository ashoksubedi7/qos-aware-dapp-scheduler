import os
from pathlib import Path


def validate_scientific_process_environment(
    protocol,
    config,
):
    """
    Validate process-level requirements for an
    AssuredQoS scientific training run.

    This module intentionally does not import TensorFlow
    or the scientific training executor. Environment
    requirements must be checked before either is loaded.
    """
    protocol.assert_ready_for_scientific_training()

    expected_pythonhashseed = str(
        config.seed
    )

    actual_pythonhashseed = os.environ.get(
        "PYTHONHASHSEED"
    )

    if (
        actual_pythonhashseed
        != expected_pythonhashseed
    ):
        raise RuntimeError(
            "PYTHONHASHSEED must be set before "
            "Python starts and must exactly match "
            f"the experiment seed; expected "
            f"{expected_pythonhashseed!r}, found "
            f"{actual_pythonhashseed!r}"
        )

    deterministic_ops = os.environ.get(
        "TF_DETERMINISTIC_OPS"
    )

    if deterministic_ops != "1":
        raise RuntimeError(
            "TF_DETERMINISTIC_OPS must be exactly "
            "'1' for scientific training; found "
            f"{deterministic_ops!r}"
        )

    cuda_visible_devices = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )

    if cuda_visible_devices != "":
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be exactly "
            "the empty string for the validated "
            "CPU-only scientific training path; "
            f"found {cuda_visible_devices!r}"
        )

    return True


def _repository_root():
    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def _require_clean_tracked_worktree(
    repo_root,
):
    from experiments.post_buffer_fix_drain_evidence import (
        require_clean_tracked_worktree,
    )

    return require_clean_tracked_worktree(
        repo_root
    )


def _git_head(
    repo_root,
):
    from experiments.post_buffer_fix_drain_evidence import (
        git_head,
    )

    return git_head(
        repo_root
    )


def _load_training_runtime():
    """
    Import TensorFlow-dependent training code only after
    process-level reproducibility requirements have been
    validated.
    """
    from common.reproducibility import (
        set_global_seed,
    )
    from experiments.scientific_training_executor import (
        run_scientific_training,
    )

    return (
        set_global_seed,
        run_scientific_training,
    )


def launch_scientific_training(
    protocol,
    config,
    run_directory,
):
    """
    Launch one provenance-bound scientific training run.

    The source commit is derived from the repository that
    contains this launcher rather than accepted from the
    caller.
    """
    validate_scientific_process_environment(
        protocol=protocol,
        config=config,
    )

    repo_root = _repository_root()

    _require_clean_tracked_worktree(
        repo_root
    )

    source_commit = _git_head(
        repo_root
    )

    (
        set_global_seed,
        run_scientific_training,
    ) = _load_training_runtime()

    set_global_seed(
        config.seed
    )

    return run_scientific_training(
        protocol=protocol,
        config=config,
        run_directory=run_directory,
        source_commit=source_commit,
    )
