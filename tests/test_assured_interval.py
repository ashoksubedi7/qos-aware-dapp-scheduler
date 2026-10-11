import sys
from pathlib import Path
from types import SimpleNamespace
from IntraSliceSch import IntraSliceScheduler
import simpy

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulator"))

from assured_interval import (
    snapshot_slice_counters,
    snapshot_urllc_deadline,
)

from assured_interval import (
    snapshot_radio_counters,
)


class FakeScheduler:
    def __init__(self):
        self.ues = {}
        self.assuredPrbsUsed = 0
        self.assuredPrbsScheduled = 0
        self.assuredPrbsAvailable = 0
        self._backlog = 0

    def updSumPcks(self):
        return self._backlog


def make_ue(
    sent,
    delivered,
    delivered_bytes,
    deadline_evaluated=0,
    deadline_misses=0,
):
    flow = SimpleNamespace(
        sentPackets=sent,
        deliveredPackets=delivered,
        deliveredBytes=delivered_bytes,
        deadlineEvaluated=deadline_evaluated,
        deadlineMisses=deadline_misses,
    )

    return SimpleNamespace(
        packetFlows=[flow],
    )


def test_snapshot_slice_counters():
    scheduler = FakeScheduler()
    scheduler._backlog = 12

    scheduler.assuredPrbsScheduled = 37
    scheduler.ues = {
        "ue1": make_ue(10, 7, 1000),
        "ue2": make_ue(20, 15, 2000),
    }

    slice_obj = SimpleNamespace(
        schedulerDL=scheduler,
    )

    counters = snapshot_slice_counters(
        slice_obj
    )

    assert counters.generated_packets == 30
    assert counters.delivered_packets == 22
    assert counters.delivered_bytes == 3000
    assert counters.backlog == 12
    assert counters.scheduled_prbs == 37

def test_snapshot_radio_counters():
    s1 = SimpleNamespace(
        schedulerDL=FakeScheduler()
    )
    s2 = SimpleNamespace(
        schedulerDL=FakeScheduler()
    )

    s1.schedulerDL.assuredPrbsUsed = 10
    s1.schedulerDL.assuredPrbsAvailable = 20

    s2.schedulerDL.assuredPrbsUsed = 30
    s2.schedulerDL.assuredPrbsAvailable = 40

    counters = snapshot_radio_counters(
        {
            "a": s1,
            "b": s2,
        }
    )

    assert counters.used_prbs == 40
    assert counters.available_prbs == 60


def test_snapshot_deadline_counts():
    scheduler = FakeScheduler()

    scheduler.ues = {
        "ue1": make_ue(
            10,
            5,
            1000,
            deadline_evaluated=8,
            deadline_misses=2,
        ),
        "ue2": make_ue(
            10,
            5,
            1000,
            deadline_evaluated=12,
            deadline_misses=3,
        ),
    }

    slice_obj = SimpleNamespace(
        schedulerDL=scheduler,
    )

    assert snapshot_urllc_deadline(
        slice_obj
    ) == (20, 5)
def test_snapshot_radio_counters_sums_cumulative_prb_slots():
    slices = {
        "eMBB": SimpleNamespace(
            schedulerDL=SimpleNamespace(
                assuredPrbsUsed=30,
                assuredPrbsAvailable=50,
            )
        ),
        "URLLC": SimpleNamespace(
            schedulerDL=SimpleNamespace(
                assuredPrbsUsed=10,
                assuredPrbsAvailable=20,
            )
        ),
        "mMTC": SimpleNamespace(
            schedulerDL=SimpleNamespace(
                assuredPrbsUsed=5,
                assuredPrbsAvailable=10,
            )
        ),
    }

    result = snapshot_radio_counters(
        slices
    )

    assert result.used_prbs == 45
    assert result.available_prbs == 80
class FakeEnv:
    def __init__(self):
        self.now = 0.0

    def timeout(self, value):
        return value


def test_scheduler_excludes_incomplete_tti_capacity():
    scheduler = IntraSliceScheduler.__new__(
        IntraSliceScheduler
    )

    scheduler.dbMd = False
    scheduler.nrbUEmax = 7
    scheduler.ttiByms = 1

    scheduler.assuredPrbsAvailable = 0

    scheduler.queueUpdate = lambda: None

    env = FakeEnv()

    process = scheduler.queuesOut(
        env
    )

    # Advance only until the first yield. The TTI is
    # incomplete, so its capacity is not yet counted.
    next(process)

    assert (
        scheduler.assuredPrbsAvailable
        == 0
    )


def test_scheduled_prbs_visible_in_same_control_interval():
    env = simpy.Environment()

    scheduler = (
        IntraSliceScheduler.__new__(
            IntraSliceScheduler
        )
    )

    scheduler.dbMd = False
    scheduler.ttiByms = 1
    scheduler.nrbUEmax = 10

    scheduler.assuredPrbsScheduled = 0
    scheduler.assuredPrbsUsed = 0
    scheduler.assuredPrbsAvailable = 0

    scheduler.sbFrNum = 0
    scheduler.ues = {}

    scheduler.queue = SimpleNamespace(
        res=[]
    )

    scheduler.printDebDataDM = (
        lambda *args, **kwargs: None
    )

    scheduler.updSumPcks = (
        lambda: 0
    )

    def queue_update():
        # Represent a successful committed
        # data-TB scheduling decision made
        # during this control interval.
        scheduler.assuredPrbsScheduled += 5

    scheduler.queueUpdate = queue_update

    slice_obj = SimpleNamespace(
        schedulerDL=scheduler
    )

    observed = {}

    def controller():
        before = snapshot_slice_counters(
            slice_obj
        )

        used_before = (
            scheduler.assuredPrbsUsed
        )

        yield env.timeout(1.0)

        after = snapshot_slice_counters(
            slice_obj
        )

        observed[
            "scheduled_delta"
        ] = (
            after.scheduled_prbs
            - before.scheduled_prbs
        )

        observed[
            "used_delta"
        ] = (
            scheduler.assuredPrbsUsed
            - used_before
        )

    # Important: controller is started first,
    # matching the ordering that exposed the
    # original used-PRB timing problem.
    env.process(
        controller()
    )

    env.process(
        scheduler.queuesOut(
            env
        )
    )

    env.run(
        until=1.01
    )

    assert (
        observed["scheduled_delta"]
        == 5
    )

    # Transmission consumption has not been
    # needed to recognize scheduler service.
    assert (
        observed["used_delta"]
        == 0
    )


def test_tti_capacity_is_attributed_to_completed_tti():
    scheduler = IntraSliceScheduler.__new__(
        IntraSliceScheduler
    )

    scheduler.dbMd = False
    scheduler.nrbUEmax = 30
    scheduler.ttiByms = 1

    scheduler.assuredPrbsAvailable = 0
    scheduler.queueUpdate = lambda: None

    env = FakeEnv()

    process = scheduler.queuesOut(env)

    # The TTI has started, but has not finished.
    assert next(process) == 1.0

    # Its capacity must not yet be included in
    # completed-TTI utilization accounting.
    assert scheduler.assuredPrbsAvailable == 0


def test_tti_reallocation_preserves_committed_capacity():
    import pytest
    from IntraSliceSch import TBqueue, TransportBlock
    from metrics.service_metrics import prb_utilization

    scheduler = IntraSliceScheduler.__new__(
        IntraSliceScheduler
    )

    scheduler.dbMd = False
    scheduler.ttiByms = 1
    scheduler.nrbUEmax = 30
    scheduler.sbFrNum = 0

    scheduler.assuredPrbsUsed = 0
    scheduler.assuredPrbsAvailable = 0

    scheduler.queue = TBqueue(30)
    scheduler.queueUpdate = lambda: None
    scheduler.printDebDataDM = (
        lambda *args, **kwargs: None
    )

    ue = SimpleNamespace(
        resUse=0,
        bler=0.0,
        TXedTB=0,
        sndbytes=0,
        packetFlows=[
            SimpleNamespace(rcvdBytes=0)
        ],
    )

    scheduler.ues = {"UE1": ue}

    # Six committed transport blocks, four PRBs each.
    # Signaling blocks isolate PRB accounting from
    # packet-level QoS bookkeeping.
    for index in range(6):
        block = TransportBlock(
            index,
            "4-QAM",
            "UE1",
            "Sig",
            [],
            4,
            20,
        )
        assert scheduler.queue.insertTB(block)

    env = FakeEnv()
    process = scheduler.queuesOut(env)

    # Begin a TTI under the 30-PRB allocation.
    assert next(process) == 1.0

    assert scheduler.assuredPrbsAvailable == 0
    assert len(scheduler.queue.res) == 6

    # Reallocate while these 24 PRBs are in flight.
    scheduler.nrbUEmax = 11
    scheduler.queue.updateSize(11)

    assert sum(
        block.numRB
        for block in scheduler.queue.res
    ) == 24

    # Complete the previous TTI and begin the next.
    assert next(process) == 1.0

    assert scheduler.assuredPrbsUsed == 24
    assert scheduler.assuredPrbsAvailable == 30
    assert ue.TXedTB == 6
    assert len(scheduler.queue.res) == 0

    assert prb_utilization(
        scheduler.assuredPrbsUsed,
        scheduler.assuredPrbsAvailable,
    ) == pytest.approx(0.8)
