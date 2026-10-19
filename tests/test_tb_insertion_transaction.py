from collections import deque
from types import SimpleNamespace

import pytest

from IntraSliceSch import (
    IntraSliceScheduler,
)


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

    # This counter must not be modified by
    # generic insertTB(). It is updated only
    # by committed data scheduling paths.
    scheduler.assuredPrbsScheduled = 0

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

    assert len(
        queue.inserted
    ) == 1

    assert (
        scheduler.ues[
            "ue1"
        ].TBid
        == 11
    )

    assert (
        scheduler.ues[
            "ue1"
        ].pendingPckts[
            1
        ]
        == 1
    )

    # Generic insertion itself must not
    # classify MAC scheduling service.
    assert (
        scheduler.assuredPrbsScheduled
        == 0
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

    assert (
        queue.inserted
        == []
    )

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

    assert (
        scheduler.assuredPrbsScheduled
        == 0
    )


class TransactionPacketBuffer:
    def __init__(
        self,
        packets,
    ):
        self.pckts = deque(
            packets
        )

    def removePckt(
        self,
    ):
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
    def __init__(
        self,
    ):
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
    def getFreeSpace(
        self,
    ):
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

    # Cumulative committed data-scheduling
    # PRBs used by AssuredQoS starvation
    # accounting.
    scheduler.assuredPrbsScheduled = 0

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

    # Failed TB admission is not
    # successful scheduler service.
    assert (
        scheduler.assuredPrbsScheduled
        == 0
    )

    # Original packet order must be restored.
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

    assert (
        scheduler.pks_s
        == 0
    )

    assert (
        scheduler.tbSize
        == 0
    )


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

    # Successful committed fresh-data
    # scheduling contributes PRBs.
    assert (
        scheduler.assuredPrbsScheduled
        == scheduler.ues[
            "ue1"
        ].prbs
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
        buffer.pckts[
            0
        ]
    )

    assert (
        residual
        is packet_2
    )

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

    assert (
        scheduler.assuredPrbsScheduled
        == scheduler.ues[
            "ue1"
        ].prbs
    )


def test_data_ptotb_scheduled_prbs_counter_is_cumulative():
    packet_1 = (
        make_transaction_packet(
            packet_id=1,
            size=40,
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
        ],
        insert_success=True,
    )

    result_1 = scheduler.dataPtoTB(
        "ue1"
    )

    assert result_1 == 2

    assert (
        scheduler.assuredPrbsScheduled
        == 2
    )

    packet_2 = (
        make_transaction_packet(
            packet_id=2,
            size=40,
        )
    )

    buffer.pckts.append(
        packet_2
    )

    result_2 = scheduler.dataPtoTB(
        "ue1"
    )

    assert result_2 == 2

    # The counter is cumulative rather than
    # an instantaneous per-TTI value.
    assert (
        scheduler.assuredPrbsScheduled
        == 4
    )

    assert (
        flow.scheduled_ids
        == [
            1,
            2,
        ]
    )


@pytest.mark.parametrize(
    "action, expected_budget, expected_scheduled",
    [(15, 0, False), (17, 5, True)],
)
def test_action_budget_changes_first_scheduling(
    action, expected_budget, expected_scheduled
):
    from common.action_space import get_action_weights
    from common.prb_allocation import (
        percentage_action_to_numerology_prbs,
    )
    from IntraSliceSch import TBqueue

    pkt = make_transaction_packet(
        packet_id=77,
        size=50,
        generation_time=0.0,
    )

    scheduler, ue, flow, buffer = make_data_ptotb_scheduler(
        packets=[pkt],
        insert_success=True,
    )

    budget = percentage_action_to_numerology_prbs(
        total_reference_prbs=32,
        weights=get_action_weights(action),
        num_ref_factors=(1, 4, 1),
    )[1]

    assert budget == expected_budget

    scheduler.queue = TBqueue(budget)
    scheduler.insertTB = lambda *args, **kwargs: (
        IntraSliceScheduler.insertTB(scheduler, *args, **kwargs)
    )

    ue.id = "ue1"
    ue.pendingTB = []
    ue.pendingPckts = {}

    scheduler.nrbUEmax = budget
    scheduler.mimomd = "SU"
    scheduler.ind_u = 0
    scheduler.sbFrNum = 2
    scheduler.sLoad = 0.003
    scheduler.dbMd = False

    from Scheds_Intra import RR_Scheduler

    ue.BWPs = 20
    scheduler.schType = "Rr"
    scheduler.printResAlloc = lambda: None
    scheduler.resAlloc = lambda nrb: RR_Scheduler.resAlloc(
        scheduler, nrb
    )
    scheduler.updSumPcks = lambda: 0

    # The action remains in force throughout the 0.5 ms deadline.
    for time_ms in (0.25, 0.50, 0.75):
        scheduler.env.now = time_ms
        IntraSliceScheduler.queueUpdate(scheduler)

    assert (pkt.scheduled_at is not None) == expected_scheduled
    assert flow.scheduled_ids == ([77] if expected_scheduled else [])
    assert scheduler.assuredPrbsScheduled == (
        budget if expected_scheduled else 0
    )
    assert ue.pendingPckts == (
        {77: 1} if expected_scheduled else {}
    )

    if expected_scheduled:
        assert pkt.scheduled_at == pytest.approx(0.25)
        assert not buffer.pckts
    else:
        assert list(buffer.pckts) == [pkt]
        assert scheduler.env.now > 0.50


@pytest.mark.parametrize(
    "action, expected_budget, missed",
    [(15, 0, True), (17, 5, False)],
)
def test_action_budget_changes_production_deadline_outcome(
    action, expected_budget, missed
):
    import simpy

    from UE import Packet, PacketFlow
    from Scheds_Intra import RR_Scheduler
    from IntraSliceSch import TBqueue
    from common.action_space import get_action_weights
    from common.prb_allocation import (
        percentage_action_to_numerology_prbs,
    )

    budget = percentage_action_to_numerology_prbs(
        total_reference_prbs=32,
        weights=get_action_weights(action),
        num_ref_factors=(1, 4, 1),
    )[1]
    assert budget == expected_budget

    pkt = Packet(77, 50, 0.0, "ue1")
    pkt.deadline = 0.5

    scheduler, ue, _, buffer = make_data_ptotb_scheduler(
        packets=[pkt],
        insert_success=True,
    )

    flow = PacketFlow(
        1, 300, 0.6, "ue1", "DL", "URLLC",
        0, 1, "Constant", "Uniform", 0.5,
    )
    flow.generatedPacketIds.add(77)
    flow.packetGenerationTimes[77] = 0.0

    ue.packetFlows = [flow]
    ue.id = "ue1"
    ue.BWPs = 20
    ue.pendingTB = []
    ue.pendingPckts = {}
    ue.bler = 0.0
    ue.resUse = 0
    ue.TXedTB = 0
    ue.sndbytes = 0
    ue.lostTB = 0

    scheduler.queue = TBqueue(budget)
    scheduler.insertTB = lambda *args, **kwargs: (
        IntraSliceScheduler.insertTB(scheduler, *args, **kwargs)
    )
    scheduler.resAlloc = lambda n: RR_Scheduler.resAlloc(
        scheduler, n
    )
    scheduler.printResAlloc = lambda: None
    scheduler.updSumPcks = lambda: 0

    scheduler.schType = "Rr"
    scheduler.nrbUEmax = budget
    scheduler.mimomd = "SU"
    scheduler.nlayers = 1
    scheduler.ind_u = 0
    scheduler.sbFrNum = 2
    scheduler.sLoad = 0.003
    scheduler.dbMd = False
    scheduler.ttiByms = 4
    scheduler.rets = 0
    scheduler.assuredPrbsUsed = 0
    scheduler.assuredPrbsAvailable = 0

    env = simpy.Environment()

    def run_mac():
        yield env.timeout(0.25)
        yield from scheduler.queuesOut(env)

    env.process(run_mac())
    env.run(until=0.76)

    # Evaluate any packet still waiting after its deadline.
    flow.recordDeadlineCrossing(pkt, env.now)

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == int(missed)

    if missed:
        assert pkt.scheduled_at is None
        assert list(buffer.pckts) == [pkt]
        assert flow.deliveredPackets == 0
    else:
        assert pkt.scheduled_at == pytest.approx(0.25)
        assert pkt.scheduling_delay == pytest.approx(0.25)
        assert flow.deliveredPackets == 1
        assert flow.completedPacketIds == {77}
        assert flow.completionDelays == pytest.approx([0.5])
