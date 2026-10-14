"""Predict next-TTI RR PRB assignments without changing scheduler state."""

from dataclasses import dataclass
from math import isfinite

from common.action_space import (
    SLICE_ORDER,
    get_action_weights,
)
from common.prb_allocation import (
    percentage_action_to_numerology_prbs,
)


@dataclass(frozen=True)
class BearerUE:
    ue_id: str
    bearer_packet_count: int
    bwp_size: float


@dataclass(frozen=True)
class RRDistribution:
    slice_prbs: int
    assignments: tuple[tuple[str, int], ...]
    unused_prbs: int

    @property
    def total_assigned(self):
        return sum(prbs for _, prbs in self.assignments)


@dataclass(frozen=True)
class ActionRRPrediction:
    action: int
    weights: tuple[int, int, int]
    slice_prbs: tuple[int, int, int]
    urllc_distribution: RRDistribution


def predict_rr_distribution(slice_prbs, ue_snapshots):
    """
    Reproduce the Rr branch of RR_Scheduler.resAlloc.

    Uses current bearer backlog and existing UE order.
    Predicts PRB assignment only, not TB insertion,
    packet service, or scheduling in future TTIs.

    UEs without bearer packets receive no new assignment.
    Their preexisting mutable 'prbs' fields are not modeled.
    """
    budget = int(slice_prbs)

    if budget != slice_prbs or budget < 0:
        raise ValueError("slice_prbs must be a nonnegative integer")

    ues = tuple(ue_snapshots)
    seen = set()

    for ue in ues:
        if not isinstance(ue, BearerUE):
            raise TypeError("expected BearerUE")

        if not ue.ue_id or ue.ue_id in seen:
            raise ValueError("invalid or duplicate UE ID")

        seen.add(ue.ue_id)

        count = ue.bearer_packet_count

        if (
            not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("invalid bearer packet count")

        if (
            not isfinite(float(ue.bwp_size))
            or float(ue.bwp_size) < 0.0
        ):
            raise ValueError("invalid BWP size")

    active = [
        ue for ue in ues
        if ue.bearer_packet_count > 0
    ]

    if not active:
        return RRDistribution(
            slice_prbs=budget,
            assignments=(),
            unused_prbs=budget,
        )

    per_ue = budget // len(active)
    remainder = budget % per_ue if per_ue else 0
    remaining = budget
    assignments = []

    for index, ue in enumerate(active):
        last = index == len(active) - 1

        if last:
            # Preserve the production implementation's
            # last-backlogged-UE remainder rule.
            assigned = max(remainder, remaining)
        else:
            if ue.bwp_size < 37:
                cap = 4
            elif ue.bwp_size < 73:
                cap = 8
            else:
                cap = budget

            assigned = min(
                per_ue,
                cap,
                remaining,
            )
            remaining -= assigned

        assignments.append((ue.ue_id, assigned))

    allocated = sum(value for _, value in assignments)

    if allocated > budget:
        raise AssertionError("RR allocation exceeded PRB budget")

    return RRDistribution(
        slice_prbs=budget,
        assignments=tuple(assignments),
        unused_prbs=budget - allocated,
    )


def predict_action_rr_distribution(
    action,
    total_reference_prbs,
    num_ref_factors,
    urllc_ues,
):
    """Evaluate one action using the existing PRB conversion."""
    weights = tuple(
        int(value)
        for value in get_action_weights(action)
    )

    factors = tuple(
        int(value)
        for value in num_ref_factors
    )

    allocation = tuple(
        int(value)
        for value in percentage_action_to_numerology_prbs(
            total_reference_prbs=total_reference_prbs,
            weights=weights,
            num_ref_factors=factors,
        )
    )

    urllc_index = SLICE_ORDER.index("URLLC")

    prediction = predict_rr_distribution(
        allocation[urllc_index],
        urllc_ues,
    )

    return ActionRRPrediction(
        action=int(action),
        weights=weights,
        slice_prbs=allocation,
        urllc_distribution=prediction,
    )
