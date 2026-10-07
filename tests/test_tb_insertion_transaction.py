from types import SimpleNamespace

from IntraSliceSch import (
    IntraSliceScheduler,
)

from collections import deque
from types import SimpleNamespace

import pytest

class RecordingQueue:
    def __init__(
        self,
        succeed=True,
    ):
        self.succeed = succeed
        self.calls = 0
        self.inserted = []

    def insertTB(
        self,
        tb,
    ):
        self.calls += 1

        if not self.succeed:
            return False

        self.inserted.append(
            tb
        )
        return True


def make_scheduler(
    queue,
):
    scheduler = (
        IntraSliceScheduler.__new__(
            IntraSliceScheduler
        )
    )

    scheduler.queue = queue

    scheduler.ues = {
        "ue1": SimpleNamespace(
            TBid=10,
            pendingPckts={},
        )
    }

    return scheduler


def test_insert_tb_inserts_exactly_once():
    queue = RecordingQueue(
        succeed=True
    )

    scheduler = make_scheduler(
        queue
    )

    result = scheduler.insertTB(
        id=10,
        m="QPSK",
        uu="ue1",
        type="data",
        pack_lst=[
            1,
        ],
        n=2,
        s=100,
    )

    assert result is True
    assert queue.calls == 1
    assert len(queue.inserted) == 1

    assert (
        scheduler.ues[
            "ue1"
        ].TBid
        == 11
    )

    assert (
        scheduler.ues[
            "ue1"
        ].pendingPckts[1]
        == 1
    )


def test_failed_insert_does_not_create_pending_reference():
    queue = RecordingQueue(
        succeed=False
    )

    scheduler = make_scheduler(
        queue
    )

    result = scheduler.insertTB(
        id=10,
        m="QPSK",
        uu="ue1",
        type="data",
        pack_lst=[
            1,
        ],
        n=2,
        s=100,
    )

    assert result is False
    assert queue.calls == 1
    assert queue.inserted == []

    assert (
        scheduler.ues[
            "ue1"
        ].TBid
        == 10
    )

    assert (
        scheduler.ues[
            "ue1"
        ].pendingPckts
        == {}
    )
class TransactionPacketBuffer:
    def __init__(
        self,
        packets,
    ):
        self.pckts = deque(
            packets
        )

    def removePckt(self):
        if not self.pckts:
            return None

        return self.pckts.popleft()

    def insertPcktLeft(
        self,
        packet,
    ):
        self.pckts.appendleft(
            packet
        )


class TransactionFlow:
    def __init__(self):
        self.scheduled_ids = []

    def recordSchedulingOutcome(
        self,
        packet,
        now,
    ):
        if (
            getattr(
                packet,
                "scheduled_at",
                None,
            )
            is None
        ):
            packet.scheduled_at = float(
                now
            )

            self.scheduled_ids.append(
                int(
                    packet.secNum
                )
            )


class TransactionQueue:
    def getFreeSpace(self):
        return 10


def make_transaction_packet(
    packet_id,
    size,
    generation_time=0.0,
):
    return SimpleNamespace(
        secNum=int(
            packet_id
        ),
        size=int(
            size
        ),
        tIn=float(
            generation_time
        ),
        scheduled_at=None,
    )


def make_data_ptotb_scheduler(
    packets,
    insert_success,
):
    scheduler = (
        IntraSliceScheduler.__new__(
            IntraSliceScheduler
        )
    )

    flow = TransactionFlow()

    buffer = TransactionPacketBuffer(
        packets
    )

    ue = SimpleNamespace(
        prbs=2,
        pastTbsz=deque(),
        tbsz=0,
        MCS=0,
        delay=0.0,
        bearers=[
            SimpleNamespace(
                buffer=buffer
            )
        ],
        packetFlows=[
            flow
        ],
        TBid=1,
    )

    scheduler.ues = {
        "ue1": ue
    }

    scheduler.schType = "RR"

    scheduler.queue = (
        TransactionQueue()
    )

    scheduler.env = (
        SimpleNamespace(
            now=5.0
        )
    )

    scheduler.pks_s = 0
    scheduler.tbSize = 0

    scheduler.setMod = (
        lambda u, n: [
            800,
            "QPSK",
            2,
            1,
        ]
    )

    scheduler.setBLER = (
        lambda u: None
    )

    scheduler.printDebDataDM = (
        lambda message: None
    )

    scheduler.insertTB = (
        lambda *args, **kwargs:
        insert_success
    )

    return (
        scheduler,
        ue,
        flow,
        buffer,
    )
def test_data_ptotb_failed_insert_restores_packets_without_scheduling():
    packet_1 = (
        make_transaction_packet(
            packet_id=1,
            size=60,
        )
    )

    packet_2 = (
        make_transaction_packet(
            packet_id=2,
            size=50,
        )
    )

    (
        scheduler,
        ue,
        flow,
        buffer,
    ) = make_data_ptotb_scheduler(
        packets=[
            packet_1,
            packet_2,
        ],
        insert_success=False,
    )

    result = scheduler.dataPtoTB(
        "ue1"
    )

    assert result == 0

    assert list(
        buffer.pckts
    ) == [
        packet_1,
        packet_2,
    ]

    assert (
        packet_1.size
        == 60
    )

    assert (
        packet_2.size
        == 50
    )

    assert (
        packet_1.scheduled_at
        is None
    )

    assert (
        packet_2.scheduled_at
        is None
    )

    assert (
        flow.scheduled_ids
        == []
    )

    assert scheduler.pks_s == 0
    assert scheduler.tbSize == 0
def test_data_ptotb_success_records_scheduling_after_commit():
    packet_1 = (
        make_transaction_packet(
            packet_id=1,
            size=40,
        )
    )

    packet_2 = (
        make_transaction_packet(
            packet_id=2,
            size=50,
        )
    )

    (
        scheduler,
        ue,
        flow,
        buffer,
    ) = make_data_ptotb_scheduler(
        packets=[
            packet_1,
            packet_2,
        ],
        insert_success=True,
    )

    result = scheduler.dataPtoTB(
        "ue1"
    )

    assert result == 2

    assert list(
        buffer.pckts
    ) == []

    assert (
        flow.scheduled_ids
        == [
            1,
            2,
        ]
    )

    assert (
        packet_1.scheduled_at
        == pytest.approx(
            5.0
        )
    )

    assert (
        packet_2.scheduled_at
        == pytest.approx(
            5.0
        )
    )
def test_data_ptotb_success_reinserts_only_unsent_residual():
    packet_1 = (
        make_transaction_packet(
            packet_id=1,
            size=60,
        )
    )

    packet_2 = (
        make_transaction_packet(
            packet_id=2,
            size=50,
        )
    )

    (
        scheduler,
        ue,
        flow,
        buffer,
    ) = make_data_ptotb_scheduler(
        packets=[
            packet_1,
            packet_2,
        ],
        insert_success=True,
    )

    result = scheduler.dataPtoTB(
        "ue1"
    )

    assert result == 2

    assert len(
        buffer.pckts
    ) == 1

    residual = (
        buffer.pckts[0]
    )

    assert residual is packet_2

    assert (
        residual.size
        == 10
    )

    assert (
        flow.scheduled_ids
        == [
            1,
            2,
        ]
    )

    assert (
        packet_2.scheduled_at
        == pytest.approx(
            5.0
        )
    )
