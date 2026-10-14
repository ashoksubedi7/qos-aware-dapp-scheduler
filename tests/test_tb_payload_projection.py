from types import SimpleNamespace

import pytest

from assurance.rr_prb_distribution import RRDistribution
from assurance.tb_payload_projection import (
    project_rr_tb_payload,
)


class FakeScheduler:
    def __init__(self, bits_per_prb=80):
        self.calls = []
        self.bits_per_prb = bits_per_prb
        self.ues = {
            "ue1": SimpleNamespace(pendingTB=[]),
            "ue2": SimpleNamespace(pendingTB=[]),
        }

    def setMod(self, ue_id, prbs):
        self.calls.append((ue_id, prbs))
        return (
            self.bits_per_prb * prbs,
            "QPSK",
            2,
            1,
        )


def distribution(assignments, budget=20):
    assignments = tuple(assignments)

    return RRDistribution(
        slice_prbs=budget,
        assignments=assignments,
        unused_prbs=(
            budget
            - sum(prbs for _, prbs in assignments)
        ),
    )


def test_potential_payload_uses_tbs_bits():
    scheduler = FakeScheduler(bits_per_prb=80)

    result = project_rr_tb_payload(
        scheduler,
        distribution([("ue1", 3)]),
    )

    assert len(result) == 1
    assert result[0].tbs_bits == 240
    assert result[0].potential_payload_bytes == 30
    assert result[0].status == "potential_data_tb"
    assert scheduler.calls == [("ue1", 3)]


def test_fractional_byte_capacity_is_truncated():
    scheduler = FakeScheduler(bits_per_prb=17)

    result = project_rr_tb_payload(
        scheduler,
        distribution([("ue1", 1)]),
    )

    assert result[0].potential_payload_bytes == 2


def test_zero_prbs_do_not_call_setmod():
    scheduler = FakeScheduler()

    result = project_rr_tb_payload(
        scheduler,
        distribution([("ue1", 0)]),
    )

    assert result[0].status == "no_prbs"
    assert result[0].potential_payload_bytes == 0
    assert scheduler.calls == []


def test_pending_retransmission_blocks_fresh_projection():
    scheduler = FakeScheduler()
    scheduler.ues["ue1"].pendingTB.append(object())

    result = project_rr_tb_payload(
        scheduler,
        distribution([("ue1", 5)]),
    )

    assert result[0].status == "retransmission_pending"
    assert result[0].potential_payload_bytes == 0
    assert scheduler.calls == []


def test_zero_byte_tbs_is_not_sufficient():
    scheduler = FakeScheduler(bits_per_prb=4)

    result = project_rr_tb_payload(
        scheduler,
        distribution([("ue1", 1)]),
    )

    assert result[0].status == "no_payload_capacity"


def test_multiple_ues_preserve_assignment_order():
    scheduler = FakeScheduler()

    result = project_rr_tb_payload(
        scheduler,
        distribution([
            ("ue2", 4),
            ("ue1", 3),
        ]),
    )

    assert [item.ue_id for item in result] == [
        "ue2", "ue1"
    ]
    assert scheduler.calls == [
        ("ue2", 4),
        ("ue1", 3),
    ]


def test_empty_distribution():
    scheduler = FakeScheduler()

    result = project_rr_tb_payload(
        scheduler,
        distribution([], budget=10),
    )

    assert result == ()
    assert scheduler.calls == []


def test_scheduler_state_is_not_modified_by_stub():
    scheduler = FakeScheduler()
    original = tuple(
        (key, tuple(value.pendingTB))
        for key, value in scheduler.ues.items()
    )

    project_rr_tb_payload(
        scheduler,
        distribution([("ue1", 5)]),
    )

    after = tuple(
        (key, tuple(value.pendingTB))
        for key, value in scheduler.ues.items()
    )

    assert original == after


@pytest.mark.parametrize(
    "bits",
    [
        -8,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_tbs_values_are_rejected(bits):
    scheduler = FakeScheduler(bits_per_prb=bits)

    with pytest.raises(ValueError):
        project_rr_tb_payload(
            scheduler,
            distribution([("ue1", 1)]),
        )


def test_unknown_ue_is_rejected():
    scheduler = FakeScheduler()

    with pytest.raises(ValueError):
        project_rr_tb_payload(
            scheduler,
            distribution([("missing", 1)]),
        )


def test_invalid_distribution_type_is_rejected():
    with pytest.raises(TypeError):
        project_rr_tb_payload(
            FakeScheduler(),
            object(),
        )
