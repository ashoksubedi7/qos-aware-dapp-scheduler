import json
from types import SimpleNamespace

from assurance.decision_snapshot import (
    capture_decision_snapshot,
)


def make_slice(ues, factor):
    return SimpleNamespace(
        numRefFactor=factor,
        schedulerDL=SimpleNamespace(
            ues=ues,
            ind_u=0,
            ttiByms=4,
        ),
    )


def test_capture_does_not_modify_packet_state():
    packet = SimpleNamespace(
        secNum=77,
        size=50,
        tIn=0.0,
        scheduled_at=None,
    )

    flow = SimpleNamespace(
        schedulingDeadline=0.45,
        appBuff=SimpleNamespace(pckts=[]),
    )

    ue = SimpleNamespace(
        packetFlows=[flow],
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(pckts=[packet])
            )
        ],
        radioLinks=SimpleNamespace(linkQuality=27.5),
        BWPs=20,
        pendingTB=[],
        pendingPckts={},
    )

    slices = {
        "eMBB": make_slice({}, 1),
        "URLLC": make_slice({"ue1": ue}, 4),
        "mMTC": make_slice({}, 1),
    }

    scheduler = SimpleNamespace(
        PRBs=32,
        _canonical_slices=lambda: slices,
    )

    snapshot = capture_decision_snapshot(
        scheduler=scheduler,
        now=0.25,
        observation=[0.0] * 6,
        action=20,
    )

    assert snapshot["selected_action"] == 20
    assert snapshot["proposed_slice_prbs"] == [12, 2, 12]

    captured = (
        snapshot["slices"]["URLLC"]
        ["ues"][0]["bearer_queues"][0]["packets"][0]
    )

    assert captured["packet_id"] == 77
    assert captured["generated_at_ms"] == 0.0
    assert captured["deadline_at_ms"] == 0.45
    assert captured["remaining_deadline_ms"] == 0.2
    assert captured["scheduled_at_ms"] is None

    json.dumps(snapshot, allow_nan=False)

    packet.size = 100
    packet.scheduled_at = 0.30

    assert captured["size_bytes"] == 50
    assert captured["scheduled_at_ms"] is None


def test_capture_handles_no_registered_users():
    slices = {
        "eMBB": make_slice({}, 1),
        "URLLC": make_slice({}, 4),
        "mMTC": make_slice({}, 1),
    }

    scheduler = SimpleNamespace(
        PRBs=32,
        _canonical_slices=lambda: slices,
    )

    snapshot = capture_decision_snapshot(
        scheduler=scheduler,
        now=0.0,
        observation=[0.0] * 7,
        action=17,
    )

    assert snapshot["proposed_slice_prbs"] == [6, 5, 6]

    for name in ("eMBB", "URLLC", "mMTC"):
        assert (
            snapshot["slices"][name]["registered_ue_count"]
            == 0
        )

    json.dumps(snapshot, allow_nan=False)
