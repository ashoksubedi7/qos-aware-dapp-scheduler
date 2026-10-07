from dataclasses import (
    asdict,
    dataclass,
)


@dataclass(frozen=True)
class DQNHyperparameters:
    learning_rate: float = 1e-3
    gamma: float = 0.99

    replay_capacity: int = 100_000
    batch_size: int = 64
    min_replay_size: int = 1_000

    target_update_interval: int = 250

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be "
                "greater than zero"
            )

        if not (
            0.0
            <= self.gamma
            <= 1.0
        ):
            raise ValueError(
                "gamma must be in [0, 1]"
            )

        if self.replay_capacity <= 0:
            raise ValueError(
                "replay_capacity must be "
                "greater than zero"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be "
                "greater than zero"
            )

        if (
            self.min_replay_size
            < self.batch_size
        ):
            raise ValueError(
                "min_replay_size must be "
                ">= batch_size"
            )

        if (
            self.replay_capacity
            < self.min_replay_size
        ):
            raise ValueError(
                "replay_capacity must be "
                ">= min_replay_size"
            )

        if self.target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must "
                "be greater than zero"
            )

        if not (
            0.0
            <= self.epsilon_min
            <= self.epsilon_start
            <= 1.0
        ):
            raise ValueError(
                "epsilon values must satisfy "
                "0 <= min <= start <= 1"
            )

        if not (
            0.0
            < self.epsilon_decay
            <= 1.0
        ):
            raise ValueError(
                "epsilon_decay must be in "
                "(0, 1]"
            )

    def to_dict(self):
        return asdict(
            self
        )

