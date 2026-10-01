from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    model_variant: str = "M1"

    seed: int = 7

    control_interval_ms: float = 1.0
    urllc_scheduling_deadline_ms: float = 1.0

    backlog_cap: int = 500
    sinr_min_db: float = 0.0
    sinr_max_db: float = 35.0

    starvation_threshold_ms: float = 10.0

    training_mode: bool = True

    radio_scenario: str = "STATIC_GOOD"
    traffic_scenario: str = "BASELINE"

    reward_urllc_service: float = 0.30
    reward_embb_service: float = 0.25
    reward_mmtc_service: float = 0.15
    reward_utilization: float = 0.10
    reward_deadline: float = 0.15
    reward_starvation: float = 0.05

    def __post_init__(self):
        if self.model_variant not in (
            "M1",
            "M2",
            "M3",
        ):
            raise ValueError(
                f"Unknown model variant: "
                f"{self.model_variant}"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if self.control_interval_ms <= 0:
            raise ValueError(
                "control_interval_ms must be greater than zero"
            )

        if self.urllc_scheduling_deadline_ms <= 0:
            raise ValueError(
                "urllc_scheduling_deadline_ms must be greater than zero"
            )

        if self.backlog_cap <= 0:
            raise ValueError(
                "backlog_cap must be greater than zero"
            )

        if self.sinr_max_db <= self.sinr_min_db:
            raise ValueError(
                "sinr_max_db must be greater than sinr_min_db"
            )

        if self.starvation_threshold_ms <= 0:
            raise ValueError(
                "starvation_threshold_ms must be greater than zero"
            )

        reward_sum = (
            self.reward_urllc_service
            + self.reward_embb_service
            + self.reward_mmtc_service
            + self.reward_utilization
            + self.reward_deadline
            + self.reward_starvation
        )

        if abs(reward_sum - 1.0) > 1e-9:
            raise ValueError(
                "reward weights must sum to 1.0"
            )

    def to_dict(self):
        return asdict(self)
