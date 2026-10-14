"""Read-only first-UE control-flow prediction for the RR MAC loop."""

from dataclasses import dataclass

from assurance.rr_prb_distribution import RRDistribution


@dataclass(frozen=True)
class RRFirstVisit:
    ue_id: str | None
    assigned_prbs: int
    rr_assigned_total: int
    rb_limit: int
    status: str
    will_call_signalling: bool
    will_call_retransmission: bool
    will_call_data: bool
    stale_nonbacklogged_prbs: bool


def predict_rr_first_visit(scheduler, distribution):
    """
    Predict only the first queueUpdate loop iteration.

    This models control-flow eligibility, not TB insertion,
    packet delivery, or visits to subsequent UEs.
    """
    if not isinstance(distribution, RRDistribution):
        raise TypeError("expected RRDistribution")

    if scheduler.schType[:2] != "Rr":
        raise ValueError("RR scheduler required")

    budget = int(scheduler.nrbUEmax)

    if budget != distribution.slice_prbs:
        raise ValueError("distribution does not match slice budget")

    limit = (
        budget * int(scheduler.nlayers)
        if scheduler.mimomd == "MU"
        else budget
    )

    names = tuple(scheduler.ues)
    assigned = dict(distribution.assignments)

    if not names:
        return RRFirstVisit(
            None, 0, distribution.total_assigned,
            limit, "no_ues", False, False, False, False,
        )

    index = int(scheduler.ind_u)

    if not 0 <= index < len(names):
        raise ValueError("invalid UE rotation index")

    name = names[index]
    ue = scheduler.ues[name]

    if not ue.bearers:
        raise ValueError("RR requires a bearer for each UE")

    bearer_count = len(ue.bearers[0].buffer.pckts)

    # RR updates the fields of backlogged UEs only.
    # Nonbacklogged UEs retain their previous PRB field.
    prbs = int(assigned.get(name, ue.prbs))

    if prbs < 0:
        raise ValueError("negative UE PRB assignment")

    stale = (
        bearer_count == 0
        and name not in assigned
        and prbs > 0
    )

    def outcome(
        status,
        signalling=False,
        retransmission=False,
        data=False,
    ):
        return RRFirstVisit(
            ue_id=name,
            assigned_prbs=prbs,
            rr_assigned_total=distribution.total_assigned,
            rb_limit=limit,
            status=status,
            will_call_signalling=signalling,
            will_call_retransmission=retransmission,
            will_call_data=data,
            stale_nonbacklogged_prbs=stale,
        )

    if limit <= 0:
        return outcome("zero_tti_budget")

    # Initial rb is zero and packts is initialized to one.
    if prbs > limit:
        return outcome(
            "stale_prbs_block_loop"
            if stale else "prbs_block_loop"
        )

    if prbs == 0:
        return outcome("zero_ue_prbs")

    # queueUpdate checks rb < rb_assigned.
    # Initially rb = 0.
    if distribution.total_assigned <= 0:
        return outcome("no_rr_assignment")

    if ue.pendingTB:
        return outcome(
            "retransmission_preempts_data",
            retransmission=True,
        )

    # Reproduce rrcUncstSigIn's scheduling condition.
    load = float(scheduler.sLoad)

    if load <= 0.0:
        raise ValueError("invalid signalling load")

    signalling_period = int(1.0 / load)

    if signalling_period <= 0:
        raise ValueError("invalid signalling period")

    ue_number = int(str(ue.id)[2:])

    signalling_due = (
        (int(scheduler.sbFrNum) - ue_number)
        % signalling_period == 0
    )

    signalling_cost = budget if signalling_due else 0

    if signalling_cost >= limit:
        return outcome(
            "signalling_blocks_data",
            signalling=True,
        )

    if bearer_count == 0:
        return outcome(
            "empty_bearer_data_call",
            signalling=True,
            data=True,
        )

    return outcome(
        "fresh_data_call_candidate",
        signalling=True,
        data=True,
    )
