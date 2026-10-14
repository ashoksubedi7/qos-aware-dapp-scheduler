from types import SimpleNamespace

import pytest

from assurance.known_deadline_demand import (
    collect_known_deadline_demand,
)


def packet(number, arrival, deadline=1.0, scheduled=None):
    return SimpleNamespace(
        secNum=number,
        tIn=arrival,
        deadline=deadline,
        scheduled_at=scheduled,
    )


def make_ue(application=(), bearer=(), evaluated=()):
    flow = SimpleNamespace(
        appBuff=SimpleNamespace(pckts=list(application)),
        deadlineEvaluatedPacketIds=set(evaluated),
    )

    return SimpleNamespace(
        packetFlows=[flow],
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(pckts=list(bearer))
            )
        ],
    )


def make_slice(**users):
    return SimpleNamespace(
        schedulerDL=SimpleNamespace(ues=users)
    )


def test_duplicate_packet_in_two_queues_is_counted_once():
    waiting = packet(4, 2.0)
    slice_obj = make_slice(
        ue1=make_ue(
            application=[waiting],
            bearer=[waiting],
        )
    )

    result = collect_known_deadline_demand(
        slice_obj, 2.5
    )

    assert len(result) == 1
    assert result[0].packet_id == 4
    assert result[0].slack_ms == 0.5
    assert result[0].locations == (
        "application",
        "bearer",
    )


def test_exact_deadline_is_not_overdue():
    slice_obj = make_slice(
        ue1=make_ue(
            application=[packet(1, 0.0)]
        )
    )

    result = collect_known_deadline_demand(
        slice_obj, 1.0
    )

    assert len(result) == 1
    assert result[0].slack_ms == 0.0
    assert result[0].overdue is False


def test_exceeded_deadline_is_overdue():
    slice_obj = make_slice(
        ue1=make_ue(
            application=[packet(1, 0.0)]
        )
    )

    result = collect_known_deadline_demand(
        slice_obj, 1.001
    )

    assert result[0].overdue is True
    assert result[0].slack_ms < 0.0


def test_packet_ids_are_distinct_across_ues():
    slice_obj = make_slice(
        ue1=make_ue(application=[packet(1, 0.0)]),
        ue2=make_ue(application=[packet(1, 0.0)]),
    )

    result = collect_known_deadline_demand(
        slice_obj, 0.5
    )

    assert len(result) == 2
    assert {item.ue_id for item in result} == {
        "ue1",
        "ue2",
    }


def test_already_scheduled_packets_are_excluded():
    slice_obj = make_slice(
        ue1=make_ue(
            application=[
                packet(1, 0.0, scheduled=0.2),
                packet(2, 0.0, deadline=None),
                packet(3, 0.0),
            ]
        )
    )

    result = collect_known_deadline_demand(
        slice_obj, 0.5
    )

    assert [item.packet_id for item in result] == [3]


def test_deadline_accounting_status_is_preserved():
    slice_obj = make_slice(
        ue1=make_ue(
            application=[packet(7, 0.0)],
            evaluated={7},
        )
    )

    result = collect_known_deadline_demand(
        slice_obj, 2.0
    )

    assert len(result) == 1
    assert result[0].overdue is True
    assert result[0].deadline_already_evaluated is True


def test_conflicting_packet_copies_are_rejected():
    slice_obj = make_slice(
        ue1=make_ue(
            application=[packet(1, 0.0)],
            bearer=[packet(1, 0.2)],
        )
    )

    with pytest.raises(ValueError):
        collect_known_deadline_demand(
            slice_obj, 0.5
        )


@pytest.mark.parametrize(
    "now, arrival, deadline",
    [
        (float("nan"), 0.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.5, 0.0, 0.0),
    ],
)
def test_invalid_timing_is_rejected(
    now, arrival, deadline
):
    slice_obj = make_slice(
        ue1=make_ue(
            application=[
                packet(1, arrival, deadline)
            ]
        )
    )

    with pytest.raises(ValueError):
        collect_known_deadline_demand(
            slice_obj, now
        )
