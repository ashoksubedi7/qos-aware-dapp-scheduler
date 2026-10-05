from dataclasses import dataclass


@dataclass
class RewardComponents:
    urllc_service: float
    embb_service: float
    mmtc_service: float
    utilization: float
    deadline_penalty: float
    starvation_penalty: float
    total: float


DEFAULT_REWARD_WEIGHTS = {
    "urllc": 0.30,
    "embb": 0.25,
    "mmtc": 0.15,
    "utilization": 0.10,
    "deadline": 0.15,
    "starvation": 0.05,
}


def clamp01(value):
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def validate_reward_weights(
    weights,
):
    required = {
        "urllc",
        "embb",
        "mmtc",
        "utilization",
        "deadline",
        "starvation",
    }

    missing = (
        required
        - set(weights.keys())
    )

    extra = (
        set(weights.keys())
        - required
    )

    if missing:
        raise ValueError(
            "missing reward weights: "
            f"{sorted(missing)}"
        )

    if extra:
        raise ValueError(
            "unknown reward weights: "
            f"{sorted(extra)}"
        )

    for name in required:
        value = float(
            weights[name]
        )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"reward weight {name} "
                "must be in [0, 1]"
            )

    total = sum(
        float(
            weights[name]
        )
        for name in required
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "reward weights must sum "
            "to 1.0"
        )


def compute_assured_reward(
    urllc_service,
    embb_service,
    mmtc_service,
    utilization,
    deadline_miss_ratio,
    starvation_penalty,
    weights=None,
):
    """
    Compute AssuredQoS reward from observed
    network outcomes.

    All service/utilization/penalty inputs are
    normalized to [0, 1].

    Formal verification properties are intentionally
    not embedded in this reward.
    """

    if weights is None:
        weights = (
            DEFAULT_REWARD_WEIGHTS.copy()
        )

    validate_reward_weights(
        weights
    )

    urllc_service = clamp01(
        urllc_service
    )

    embb_service = clamp01(
        embb_service
    )

    mmtc_service = clamp01(
        mmtc_service
    )

    utilization = clamp01(
        utilization
    )

    deadline_miss_ratio = clamp01(
        deadline_miss_ratio
    )

    starvation_penalty = clamp01(
        starvation_penalty
    )

    total = (
        float(weights["urllc"])
        * urllc_service

        + float(weights["embb"])
        * embb_service

        + float(weights["mmtc"])
        * mmtc_service

        + float(weights["utilization"])
        * utilization

        - float(weights["deadline"])
        * deadline_miss_ratio

        - float(weights["starvation"])
        * starvation_penalty
    )

    return RewardComponents(
        urllc_service=urllc_service,
        embb_service=embb_service,
        mmtc_service=mmtc_service,
        utilization=utilization,
        deadline_penalty=(
            deadline_miss_ratio
        ),
        starvation_penalty=(
            starvation_penalty
        ),
        total=float(
            total
        ),
    )
