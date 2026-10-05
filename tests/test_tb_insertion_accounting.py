import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)


from IntraSliceSch import (
    IntraSliceScheduler,
)


class RejectingQueue:
    def insertTB(
        self,
        tb,
    ):
        return False


class AcceptingQueue:
    def insertTB(
        self,
        tb,
    ):
        return True


def make_scheduler():
    scheduler = object.__new__(
        IntraSliceScheduler
    )

    scheduler.ues = {
        "ue1": SimpleNamespace(
            TBid=4,
            pendingPckts={},
        )
    }

    return scheduler


def test_failed_tb_insertion_does_not_mutate_packet_state():
    scheduler = make_scheduler()
    scheduler.queue = (
        RejectingQueue()
    )

    result = scheduler.insertTB(
        4,
        "QPSK",
        "ue1",
        "data",
        [7],
        5,
        100,
    )

    assert result is False

    assert (
        scheduler.ues[
            "ue1"
        ].TBid
        == 4
    )

    assert (
        scheduler.ues[
            "ue1"
        ].pendingPckts
        == {}
    )


def test_successful_tb_insertion_updates_packet_state():
    scheduler = make_scheduler()
    scheduler.queue = (
        AcceptingQueue()
    )

    result = scheduler.insertTB(
        4,
        "QPSK",
        "ue1",
        "data",
        [7],
        5,
        100,
    )

    assert result is True

    assert (
        scheduler.ues[
            "ue1"
        ].TBid
        == 5
    )

    assert (
        scheduler.ues[
            "ue1"
        ].pendingPckts[
            7
        ]
        == 1
    )
