"""Conservative assessment of learned MAC allocation candidates.

REJECT identifies a limited necessary-condition failure.
POTENTIALLY_ADMISSIBLE applies only to known current demand.
UNKNOWN makes no claim of feasibility or infeasibility.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

from assurance.known_deadline_demand import (
    collect_known_deadline_demand,
)
from assurance.scheduling_opportunities import (
    SchedulingClock,
    assess_known_deadline_demand,
)
from assurance.rr_prb_distribution import (
    BearerUE,
    predict_action_rr_distribution,
)
from assurance.rr_first_visit import predict_rr_first_visit
from assurance.tb_payload_projection import project_rr_tb_payload
from common.action_space import ACTION_SPACE_SIZE, get_action_weights
from common.prb_allocation import percentage_action_to_numerology_prbs


@dataclass(frozen=True)
class CandidateAssessment:
    action: int
    verdict: str
    reason: str
    urllc_prbs: int
    known_waiting: int
    imminent_deadlines: int
    potential_payload_bytes: int | None
    first_visit_status: str | None


def assess_candidate_actions(
    urllc_slice,
    now_ms,
    control_interval_ms,
    clock,
    total_reference_prbs,
    num_ref_factors,
    *,
    control_precedes_current_mac_event=False,
    include_diagnostics=False,
):
    """Evaluate the finite DQN action catalog without applying actions.

    Assumes each candidate would hold its converted allocation
    until the next control update. Does not model future arrivals,
    future queue transfers, or complete intra-slice service.
    """
    now = float(now_ms)
    interval = float(control_interval_ms)

    if not math.isfinite(now) or now < 0:
        raise ValueError("invalid observation time")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("invalid control interval")
    if not isinstance(clock, SchedulingClock):
        raise TypeError("expected SchedulingClock")

    if type(control_precedes_current_mac_event) is not bool:
        raise TypeError('control order must be boolean')
    if type(include_diagnostics) is not bool:
        raise TypeError('diagnostics flag must be boolean')

    factors = tuple(num_ref_factors)
    if len(factors) != 3:
        raise ValueError("expected three numerology factors")

    scheduler = urllc_slice.schedulerDL

    if scheduler.schType[:2] != "Rr":
        raise ValueError("RR scheduler required")

    observed_period = 1.0 / float(scheduler.ttiByms)
    if not math.isclose(
        observed_period, clock.period_ms,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("clock and scheduler TTI disagree")

    packets = collect_known_deadline_demand(
        urllc_slice, now
    )
    opportunities = assess_known_deadline_demand(
        packets, now, clock
    )

    end = now + interval
    tolerance = 1e-9

    imminent = [
        (packet, opportunity)
        for packet, opportunity in zip(
            packets, opportunities
        )
        if (
            not packet.deadline_already_evaluated
            and packet.absolute_deadline_ms >= now
            and packet.absolute_deadline_ms < end - tolerance
        )
    ]

    has_clock_gap = any(
        opportunity.future_events_before_deadline == 0
        and not opportunity.on_current_event_boundary
        for _, opportunity in imminent
    )

    has_boundary_ambiguity = (
        not control_precedes_current_mac_event
        and any(
            opportunity.on_current_event_boundary
            for _, opportunity in imminent
        )
    )

    already_overdue = any(
        packet.absolute_deadline_ms < now
        for packet in packets
    )

    ue_snapshots = None

    if include_diagnostics:
        ue_snapshots = tuple(
            BearerUE(
                ue_id=str(name),
                bearer_packet_count=len(
                    ue.bearers[0].buffer.pckts
                ),
                bwp_size=float(ue.BWPs),
            )
            for name, ue in scheduler.ues.items()
        )

    results = []

    for action in range(ACTION_SPACE_SIZE):
        weights = tuple(
            int(value) for value in get_action_weights(action)
        )

        converted = percentage_action_to_numerology_prbs(
            total_reference_prbs=total_reference_prbs,
            weights=weights,
            num_ref_factors=factors,
        )

        urllc_prbs = int(converted[1])
        potential_bytes = None
        first_visit_status = None

        # Optional diagnostic path. It is not necessary to
        # establish the zero-resource rejection condition.
        if include_diagnostics:
            allocation = predict_action_rr_distribution(
                action=action,
                total_reference_prbs=total_reference_prbs,
                num_ref_factors=factors,
                urllc_ues=ue_snapshots,
            )
            distribution = allocation.urllc_distribution

            assert distribution.slice_prbs == urllc_prbs

            view = SimpleNamespace(
                ues=scheduler.ues,
                schType=scheduler.schType,
                nrbUEmax=urllc_prbs,
                mimomd=scheduler.mimomd,
                nlayers=scheduler.nlayers,
                ind_u=scheduler.ind_u,
                sLoad=scheduler.sLoad,
                sbFrNum=scheduler.sbFrNum,
                setMod=scheduler.setMod,
            )

            first_visit_status = predict_rr_first_visit(
                view, distribution
            ).status

            payloads = project_rr_tb_payload(
                view, distribution
            )

            potential_bytes = sum(
                item.potential_payload_bytes
                for item in payloads
                if item.status == "potential_data_tb"
            )

        if (
            urllc_prbs == 0
            and imminent
            and not has_boundary_ambiguity
        ):
            verdict = "REJECT"
            reason = "zero_urllc_budget_before_known_deadline"
        elif has_clock_gap:
            verdict = "UNKNOWN"
            reason = "no_future_clock_event_for_known_packet"
        elif has_boundary_ambiguity:
            verdict = "UNKNOWN"
            reason = "same_time_event_order_not_established"
        elif imminent:
            verdict = "UNKNOWN"
            reason = "complete_packet_service_not_proven"
        elif already_overdue:
            verdict = "UNKNOWN"
            reason = "previously_missed_deadline_irrecoverable"
        else:
            verdict = "POTENTIALLY_ADMISSIBLE"
            reason = "no_imminent_known_deadline"

        results.append(
            CandidateAssessment(
                action=action,
                verdict=verdict,
                reason=reason,
                urllc_prbs=urllc_prbs,
                known_waiting=len(packets),
                imminent_deadlines=len(imminent),
                potential_payload_bytes=potential_bytes,
                first_visit_status=first_visit_status,
            )
        )

    return tuple(results)
