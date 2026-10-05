from types import SimpleNamespace

from IntraSliceSch import (
    IntraSliceScheduler,
)

from UE import PacketFlow


class AcceptingQueue:
    def insertTB(
        self,
        tb,
    ):
        return True


class RejectingQueue:
    def insertTB(
        self,
        tb,
    ):
        return False


def make_flow():
    flow = PacketFlow(
        1,
        300,
        1.0,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Constant",
    )

    flow.generatedPacketIds.add(
        7
    )

    return flow


def make_scheduler(
    queue,
    re_tx_num=0,
):
    flow = make_flow()

    tb = SimpleNamespace(
        id=1,
        mod="QPSK",
        ue="ue1",
        type="data",
        pckt_l=[7],
        numRB=5,
        size=100,
        reTxNum=re_tx_num,
    )


    ue = SimpleNamespace(
        id="ue1",
        prbs=5,
        packetFlows=[
            flow
        ],
        pendingTB=[
            tb
        ],
        pendingPckts={
            7: 1
        },
    )

    scheduler = object.__new__(
        IntraSliceScheduler
    )

    scheduler.ues = {
        "ue1": ue
    }

    scheduler.queue = queue

    return (
        scheduler,
        ue,
        flow,
        tb,
    )


def test_successful_retransmission_requeues_tb():
    (
        scheduler,
        ue,
        flow,
        tb,
    ) = make_scheduler(
        AcceptingQueue(),
        re_tx_num=2,
    )

    result = (
        scheduler.retransmitTB(
            "ue1"
        )
    )

    assert result == 5

    assert ue.pendingTB == []

    assert tb.reTxNum == 3

    # Retransmission does not represent
    # another packet fragment.
    assert (
        ue.pendingPckts[7]
        == 1
    )

    assert (
        flow.droppedPacketIds
        == set()
    )


def test_failed_queue_reinsertion_preserves_pending_tb():
    (
        scheduler,
        ue,
        flow,
        tb,
    ) = make_scheduler(
        RejectingQueue(),
        re_tx_num=2,
    )

    result = (
        scheduler.retransmitTB(
            "ue1"
        )
    )

    assert result == 0

    assert len(
        ue.pendingTB
    ) == 1

    assert (
        ue.pendingTB[0]
        is tb
    )

    assert tb.reTxNum == 2

    assert (
        ue.pendingPckts[7]
        == 1
    )

    assert (
        flow.droppedPacketIds
        == set()
    )


def test_permanent_tb_failure_records_packet_drop():
    (
        scheduler,
        ue,
        flow,
        tb,
    ) = make_scheduler(
        RejectingQueue(),
        re_tx_num=3000,
    )

    result = (
        scheduler.retransmitTB(
            "ue1"
        )
    )

    assert result == 0

    assert ue.pendingTB == []

    assert (
        7
        not in ue.pendingPckts
    )

    assert (
        7
        in flow.droppedPacketIds
    )

    assert flow.lostPackets == 1


def test_permanent_drop_decrements_only_one_tb_reference():
    (
        scheduler,
        ue,
        flow,
        tb,
    ) = make_scheduler(
        RejectingQueue(),
        re_tx_num=3000,
    )

    # Packet 7 also has another outstanding TB.
    ue.pendingPckts[
        7
    ] = 2

    scheduler.retransmitTB(
        "ue1"
    )

    assert (
        ue.pendingPckts[
            7
        ]
        == 1
    )

    # The whole packet is nevertheless
    # classified as dropped because one
    # constituent TB was permanently lost.
    assert (
        7
        in flow.droppedPacketIds
    )

    assert flow.lostPackets == 1
