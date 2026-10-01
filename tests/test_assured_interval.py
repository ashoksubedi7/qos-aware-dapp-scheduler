import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulator"))

from assured_interval import (
    snapshot_radio_counters,
    snapshot_slice_counters,
    snapshot_urllc_deadline,
)


class FakeScheduler:
    def __init__(self):
        self.ues = {}
        self.assuredPrbsUsed = 0
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
