from types import SimpleNamespace

import pytest

from assurance.rr_first_visit import predict_rr_first_visit
from assurance.rr_prb_distribution import (
    BearerUE,
    predict_rr_distribution,
)


def make_case(
    budget=8,
    counts=(1, 1),
    first_old_prbs=0,
    frame=2,
    pending=False,
    mimo="SU",
    index=0,
):
    ues = {}

    for position, count in enumerate(counts):
        name = f"ue{position + 1}"

        ues[name] = SimpleNamespace(
            id=name,
            prbs=first_old_prbs if position == 0 else 0,
            BWPs=20,
            bearers=[
                SimpleNamespace(
                    buffer=SimpleNamespace(
                        pckts=[object() for _ in range(count)]
                    )
                )
            ],
            pendingTB=[object()] if pending and position == index else [],
        )

    distribution = predict_rr_distribution(
        budget,
        [
            BearerUE(
                name,
                len(ue.bearers[0].buffer.pckts),
                ue.BWPs,
            )
            for name, ue in ues.items()
        ],
    )

    trace = []

    s = SimpleNamespace(
        ues=ues,
        schType="Rr",
        nrbUEmax=budget,
        mimomd=mimo,
        nlayers=2 if mimo == "MU" else 1,
        sbFrNum=frame,
        sLoad=0.003,
        ind_u=index,
        dbMd=False,
        printDebDataDM=lambda *args: None,
    )

    def res_alloc(n):
        assert n == budget

        for name, prbs in distribution.assignments:
            s.ues[name].prbs = prbs

        return distribution.total_assigned

    def rrc(ue_id):
        trace.append(("signalling", ue_id))

        period = int(1.0 / s.sLoad)
        number = int(s.ues[ue_id].id[2:])

        return (
            budget
            if (s.sbFrNum - number) % period == 0
            else 0
        )

    def data(ue_id):
        trace.append(("data", ue_id))
        return 0

    def retransmit(ue_id):
        trace.append(("retransmission", ue_id))
        return 0

    def advance():
        s.ind_u = (s.ind_u + 1) % len(s.ueLst)

    s.resAlloc = res_alloc
    s.rrcUncstSigIn = rrc
    s.dataPtoTB = data
    s.retransmitTB = retransmit
    s.updIndUE = advance

    # End after the first iteration so that the differential
    # test concerns exactly the model's declared scope.
    s.updSumPcks = lambda: 0

    return s, distribution, trace


@pytest.mark.parametrize(
    "kwargs,status,data,signalling,retransmission",
    [
        (
            {},
            "fresh_data_call_candidate",
            True, True, False,
        ),
        (
            {"frame": 1},
            "signalling_blocks_data",
            False, True, False,
        ),
        (
            {"pending": True},
            "retransmission_preempts_data",
            False, False, True,
        ),
        (
            {"counts": (0, 1), "first_old_prbs": 9},
            "stale_prbs_block_loop",
            False, False, False,
        ),
        (
            {"counts": (0, 1), "first_old_prbs": 2},
            "empty_bearer_data_call",
            True, True, False,
        ),
        (
            {"budget": 1},
            "zero_ue_prbs",
            False, False, False,
        ),
        (
            {"budget": 0},
            "zero_tti_budget",
            False, False, False,
        ),
        (
            {"counts": (0, 0), "first_old_prbs": 2},
            "no_rr_assignment",
            False, False, False,
        ),
        (
            {"frame": 1, "mimo": "MU"},
            "fresh_data_call_candidate",
            True, True, False,
        ),
        (
            {"index": 1, "frame": 3},
            "fresh_data_call_candidate",
            True, True, False,
        ),
    ],
)
def test_prediction_matches_first_production_iteration(
    kwargs, status, data, signalling, retransmission
):
    from IntraSliceSch import IntraSliceScheduler

    scheduler, distribution, trace = make_case(**kwargs)

    prediction = predict_rr_first_visit(
        scheduler, distribution
    )

    assert prediction.status == status
    assert prediction.will_call_data is data
    assert prediction.will_call_signalling is signalling
    assert (
        prediction.will_call_retransmission
        is retransmission
    )

    # The predictor must not change the UE PRB fields.
    before = tuple(
        ue.prbs for ue in scheduler.ues.values()
    )

    predict_rr_first_visit(scheduler, distribution)

    assert before == tuple(
        ue.prbs for ue in scheduler.ues.values()
    )

    # Execute the original production queueUpdate method
    # with controlled side-effect return values.
    IntraSliceScheduler.queueUpdate(scheduler)

    first_name = prediction.ue_id

    assert (
        ("data", first_name) in trace
    ) is data

    assert (
        ("signalling", first_name) in trace
    ) is signalling

    assert (
        ("retransmission", first_name) in trace
    ) is retransmission


def test_stale_allocation_is_identified():
    scheduler, distribution, _ = make_case(
        counts=(0, 1),
        first_old_prbs=9,
    )

    result = predict_rr_first_visit(
        scheduler, distribution
    )

    assert result.stale_nonbacklogged_prbs
    assert result.status == "stale_prbs_block_loop"


def test_next_ue_rotation_index_is_respected():
    scheduler, distribution, _ = make_case(
        index=1,
        frame=3,
    )

    result = predict_rr_first_visit(
        scheduler, distribution
    )

    assert result.ue_id == "ue2"


def test_invalid_rotation_index_is_rejected():
    scheduler, distribution, _ = make_case()
    scheduler.ind_u = 3

    with pytest.raises(ValueError):
        predict_rr_first_visit(
            scheduler, distribution
        )


def test_wrong_slice_budget_is_rejected():
    scheduler, distribution, _ = make_case()
    scheduler.nrbUEmax = 9

    with pytest.raises(ValueError):
        predict_rr_first_visit(
            scheduler, distribution
        )


def test_wrong_distribution_type_is_rejected():
    scheduler, _, _ = make_case()

    with pytest.raises(TypeError):
        predict_rr_first_visit(
            scheduler, object()
        )
