from dataclasses import asdict, dataclass, field

from config.dqn_hyperparameters import (
    DQNHyperparameters,
)


VALID_MODEL_VARIANTS = (
    "M1",
    "M2",
    "M3",
)

VALID_RADIO_SCENARIOS = (
    "STATIC_POOR",
    "STATIC_MODERATE",
    "STATIC_GOOD",
    "STEP_DEGRADATION",
    "STEP_RECOVERY",
    "FLUCTUATING",
)


@dataclass(frozen=True)
class ExperimentConfig:
    model_variant: str = "M1"

    seed: int = 7

    control_interval_ms: float = 1.0
    radio_update_interval_ms: float = 10.0
    # Duration after traffic generation stops during which
	# scheduling, radio evolution, retransmissions, and packet
	# completion continue.
	#
	# Keep zero until the drain duration is calibrated and frozen
	# for the primary experiments.
    drain_duration_ms: float = 0.0

    urllc_scheduling_deadline_ms: float = 1.0

    backlog_cap: int = 500
    sinr_min_db: float = 0.0
    sinr_max_db: float = 35.0

    starvation_threshold_ms: float = 10.0

    training_mode: bool = True

    dqn_hyperparameters: DQNHyperparameters = field(
        default_factory=DQNHyperparameters
    )

    # Required when training_mode=False.
    #
    # This checkpoint is intended for frozen policy evaluation.
    # It is not yet a complete resumable-training snapshot because
    # replay-buffer and optimizer state are not serialized.
    checkpoint_path: str | None = None

    radio_scenario: str = "STATIC_GOOD"
    traffic_scenario: str = "BASELINE"

    reward_urllc_service: float = 0.30
    reward_embb_service: float = 0.25
    reward_mmtc_service: float = 0.15
    reward_utilization: float = 0.10
    reward_deadline: float = 0.15
    reward_starvation: float = 0.05

    def __post_init__(self):
        if (
            self.model_variant
            not in VALID_MODEL_VARIANTS
        ):
            raise ValueError(
                "Unknown model variant: "
                f"{self.model_variant}"
            )

        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
        ):
            raise TypeError(
                "seed must be a non-Boolean integer"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if self.control_interval_ms <= 0:
            raise ValueError(
                "control_interval_ms must be "
                "greater than zero"
            )
        if self.radio_update_interval_ms <= 0:
            raise ValueError(
                "radio_update_interval_ms must be "
                "greater than zero"
            )
        if self.drain_duration_ms < 0:
            raise ValueError(
                "drain_duration_ms must be "
                "non-negative"
            )
        if (
            self.urllc_scheduling_deadline_ms
            <= 0
        ):
            raise ValueError(
                "urllc_scheduling_deadline_ms "
                "must be greater than zero"
            )

        if self.backlog_cap <= 0:
            raise ValueError(
                "backlog_cap must be greater "
                "than zero"
            )

        if (
            self.sinr_max_db
            <= self.sinr_min_db
        ):
            raise ValueError(
                "sinr_max_db must be greater "
                "than sinr_min_db"
            )

        if (
            self.starvation_threshold_ms
            <= 0
        ):
            raise ValueError(
                "starvation_threshold_ms must "
                "be greater than zero"
            )

        if (
            self.radio_scenario
            not in VALID_RADIO_SCENARIOS
        ):
            raise ValueError(
                "Unknown radio scenario: "
                f"{self.radio_scenario}"
            )

        if (
            not self.training_mode
            and not self.checkpoint_path
        ):
            raise ValueError(
                "evaluation mode requires "
                "checkpoint_path"
            )

        reward_weights = (
            self.reward_urllc_service,
            self.reward_embb_service,
            self.reward_mmtc_service,
            self.reward_utilization,
            self.reward_deadline,
            self.reward_starvation,
        )

        for weight in reward_weights:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    "each reward weight must "
                    "be in [0, 1]"
                )

        reward_sum = sum(
            reward_weights
        )

        if abs(
            reward_sum - 1.0
        ) > 1e-9:
            raise ValueError(
                "reward weights must sum "
                "to 1.0"
            )

    def to_dict(self):
        return asdict(self)
