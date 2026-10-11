from dataclasses import (
    dataclass,
    replace,
)

from config.experiment_config import (
    ExperimentConfig,
)
from config.training_protocol import (
    TrainingProtocol,
)
from experiments.assured_horizon import (
    ExperimentHorizon,
    build_experiment_horizon,
)


@dataclass(frozen=True)
class ScientificTrainingSetup:
    config: ExperimentConfig
    horizon: ExperimentHorizon


def build_scientific_training_setup(
    protocol,
    config,
):
    """
    Derive the immutable configuration and simulator
    horizon for one scientific training run.

    The training protocol is authoritative for the
    successful-update budget, epsilon-floor update,
    and checkpoint interval. The supplied experiment
    configuration remains authoritative for DQN
    parameters that are not protocol-derived.
    """
    if not isinstance(
        protocol,
        TrainingProtocol,
    ):
        raise TypeError(
            "protocol must be a TrainingProtocol"
        )

    if not isinstance(
        config,
        ExperimentConfig,
    ):
        raise TypeError(
            "config must be an ExperimentConfig"
        )

    protocol.assert_ready_for_scientific_training()

    if not config.training_mode:
        raise ValueError(
            "scientific training requires "
            "training_mode=True"
        )

    if config.checkpoint_path is not None:
        raise ValueError(
            "scientific training must start fresh; "
            "checkpoint_path must be None"
        )

    base_hyperparameters = (
        config.dqn_hyperparameters
    )

    epsilon_decay = (
        protocol.derive_epsilon_decay(
            epsilon_start=(
                base_hyperparameters.epsilon_start
            ),
            epsilon_min=(
                base_hyperparameters.epsilon_min
            ),
        )
    )

    derived_hyperparameters = replace(
        base_hyperparameters,
        epsilon_decay=epsilon_decay,
    )

    derived_config = replace(
        config,
        dqn_hyperparameters=(
            derived_hyperparameters
        ),
    )

    active_duration_ms = (
        protocol.derive_active_training_duration_ms(
            min_replay_size=(
                derived_hyperparameters
                .min_replay_size
            ),
            control_interval_ms=(
                derived_config
                .control_interval_ms
            ),
        )
    )

    horizon = build_experiment_horizon(
        active_duration_ms=(
            active_duration_ms
        ),
        drain_duration_ms=(
            derived_config.drain_duration_ms
        ),
    )

    return ScientificTrainingSetup(
        config=derived_config,
        horizon=horizon,
    )
