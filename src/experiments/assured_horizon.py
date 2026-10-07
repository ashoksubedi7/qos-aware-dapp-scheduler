from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentHorizon:
    active_duration_ms: float
    drain_duration_ms: float
    total_duration_ms: float


def build_experiment_horizon(
    active_duration_ms,
    drain_duration_ms,
):
    active_duration_ms = float(
        active_duration_ms
    )

    drain_duration_ms = float(
        drain_duration_ms
    )

    if active_duration_ms <= 0:
        raise ValueError(
            "active_duration_ms must be "
            "greater than zero"
        )

    if drain_duration_ms < 0:
        raise ValueError(
            "drain_duration_ms must be "
            "non-negative"
        )

    return ExperimentHorizon(
        active_duration_ms=(
            active_duration_ms
        ),
        drain_duration_ms=(
            drain_duration_ms
        ),
        total_duration_ms=(
            active_duration_ms
            + drain_duration_ms
        ),
    )

