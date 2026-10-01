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


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def compute_assured_reward(
    urllc_service,
    embb_service,
    mmtc_service,
    utilization,
    deadline_miss_ratio,
    starvation_penalty,
    weights=None,
):
    """Compute reward from observed network outcomes."""

    if weights is None:
        weights = {
            "urllc": 0.30,
            "embb": 0.25,
            "mmtc": 0.15,
            "utilization": 0.10,
            "deadline": 0.15,
            "starvation": 0.05,
        }

    urllc_service = clamp01(urllc_service)
    embb_service = clamp01(embb_service)
    mmtc_service = clamp01(mmtc_service)
    utilization = clamp01(utilization)
    deadline_miss_ratio = clamp01(deadline_miss_ratio)
    starvation_penalty = clamp01(starvation_penalty)

    total = (
        weights["urllc"] * urllc_service
        + weights["embb"] * embb_service
        + weights["mmtc"] * mmtc_service
        + weights["utilization"] * utilization
        - weights["deadline"] * deadline_miss_ratio
        - weights["starvation"] * starvation_penalty
    )

    return RewardComponents(
        urllc_service=urllc_service,
        embb_service=embb_service,
        mmtc_service=mmtc_service,
        utilization=utilization,
        deadline_penalty=deadline_miss_ratio,
        starvation_penalty=starvation_penalty,
        total=float(total),
    )
