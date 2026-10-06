import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)


from UE import (
    Bearer,
    Packet,
    PacketFlow,
)

from metrics.packet_accounting import (
    collect_packet_accounting,
    collect_slice_packet_accounting,
)


def make_flow():
    return PacketFlow(
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


def make_packet(
    packet_id,
):
    return Packet(
        packet_id,
        330,
        1,
        "ue1",
    )


def make_ue():
    bearer = Bearer(
        1,
        9,
        "DL",
    )

    return SimpleNamespace(
        id="ue1",
        bearers=[bearer],
        pendingPckts={},
        pendingTB=[],
    )


def test_generated_delivered_residual_conservation():
    flow = make_flow()
    ue = make_ue()

    flow.generatedPacketIds.update(
        {
            1,
            2,
            3,
        }
    )

    flow.completedPacketIds.add(
        1
    )

    flow.deliveredPackets = 1

    packet = make_packet(
        2
    )

    flow.appBuff.insertPckt(
        packet
    )

    flow.recordPacketDrop(
        3
    )

    result = (
        collect_packet_accounting(
            flow,
            ue,
        )
    )

    assert result.generated == 3
    assert result.delivered == 1
    assert result.dropped == 1
    assert result.residual == 1

    assert (
        result.residual_app
        == 1
    )

    assert (
        result.conservation_ok
        is True
    )


def test_fragmented_packet_is_not_double_counted():
    flow = make_flow()
    ue = make_ue()

    flow.generatedPacketIds.add(
        7
    )

    fragment = make_packet(
        7
    )

    ue.bearers[
        0
    ].buffer.insertPckt(
        fragment
    )

    ue.pendingPckts[
        7
    ] = 1

    result = (
        collect_packet_accounting(
            flow,
            ue,
        )
    )

    assert result.generated == 1
    assert result.residual == 1

    assert (
        result.observed_residual
        == 1
    )


def test_untracked_residual_is_detected():
    flow = make_flow()
    ue = make_ue()

    flow.generatedPacketIds.add(
        5
    )

    result = (
        collect_packet_accounting(
            flow,
            ue,
        )
    )

    assert result.residual == 1

    assert (
        result.untracked_residual
        == 1
    )


def test_packet_cannot_be_delivered_and_dropped():
    flow = make_flow()
    ue = make_ue()

    flow.generatedPacketIds.add(
        9
    )

    flow.completedPacketIds.add(
        9
    )

    flow.droppedPacketIds.add(
        9
    )

    with pytest.raises(
        ValueError,
        match="delivered and dropped",
    ):
        collect_packet_accounting(
            flow,
            ue,
        )


def test_drop_is_idempotent():
    flow = make_flow()

    flow.generatedPacketIds.add(
        11
    )

    assert (
        flow.recordPacketDrop(
            11
        )
        is True
    )

    assert (
        flow.recordPacketDrop(
            11
        )
        is False
    )

    assert flow.lostPackets == 1

from collections import deque
from types import SimpleNamespace


def make_slice_accounting_flow(
    generated,
    delivered,
    dropped,
    residual,
):
    """
    Build one minimal flow whose lifecycle state satisfies:

        generated
        = delivered + dropped + residual

    Residual packets are placed in the application buffer
    so they are observable at end of run.
    """

    expected = (
        int(delivered)
        + int(dropped)
        + int(residual)
    )

    if int(generated) != expected:
        raise ValueError(
            "test fixture must conserve packets"
        )

    generated_ids = set(
        range(
            1,
            int(generated) + 1,
        )
    )

    delivered_ids = set(
        range(
            1,
            int(delivered) + 1,
        )
    )

    dropped_start = (
        int(delivered)
        + 1
    )

    dropped_ids = set(
        range(
            dropped_start,
            dropped_start
            + int(dropped),
        )
    )

    residual_ids = (
        generated_ids
        - delivered_ids
        - dropped_ids
    )

    residual_packets = deque(
        SimpleNamespace(
            secNum=packet_id
        )
        for packet_id
        in residual_ids
    )

    return SimpleNamespace(
        generatedPacketIds=generated_ids,
        completedPacketIds=delivered_ids,
        droppedPacketIds=dropped_ids,

        appBuff=SimpleNamespace(
            pckts=residual_packets
        ),
    )


def make_slice_accounting_ue(
    ue_id,
    flow,
):
    return SimpleNamespace(
        id=ue_id,

        packetFlows=[
            flow
        ],

        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=deque()
                )
            )
        ],

        pendingPckts={},
        pendingTB=[],
    )


def make_slice_accounting_slice(
    ues,
):
    scheduler = SimpleNamespace(
        ues={
            ue.id: ue
            for ue in ues
        },

        queue=SimpleNamespace(
            res=[]
        ),
    )

    return SimpleNamespace(
        schedulerDL=scheduler
    )

def test_slice_packet_accounting_sums_ues():
    flow_1 = make_slice_accounting_flow(
        generated=10,
        delivered=8,
        dropped=1,
        residual=1,
    )

    flow_2 = make_slice_accounting_flow(
        generated=20,
        delivered=15,
        dropped=2,
        residual=3,
    )

    ue_1 = make_slice_accounting_ue(
        "ue1",
        flow_1,
    )

    ue_2 = make_slice_accounting_ue(
        "ue2",
        flow_2,
    )

    slice_obj = (
        make_slice_accounting_slice(
            [
                ue_1,
                ue_2,
            ]
        )
    )

    result = (
        collect_slice_packet_accounting(
            slice_obj
        )
    )

    assert result.generated == 30
    assert result.delivered == 23
    assert result.dropped == 3
    assert result.residual == 4

    assert (
        result.observed_residual
        == 4
    )

    assert (
        result.untracked_residual
        == 0
    )

    assert result.conservation_ok

def test_slice_packet_accounting_does_not_merge_ids_across_ues():
    flow_1 = make_slice_accounting_flow(
        generated=1,
        delivered=1,
        dropped=0,
        residual=0,
    )

    flow_2 = make_slice_accounting_flow(
        generated=1,
        delivered=1,
        dropped=0,
        residual=0,
    )

    ue_1 = make_slice_accounting_ue(
        "ue1",
        flow_1,
    )

    ue_2 = make_slice_accounting_ue(
        "ue2",
        flow_2,
    )

    slice_obj = (
        make_slice_accounting_slice(
            [
                ue_1,
                ue_2,
            ]
        )
    )

    result = (
        collect_slice_packet_accounting(
            slice_obj
        )
    )

    assert result.generated == 2
    assert result.delivered == 2
    assert result.dropped == 0
    assert result.residual == 0
    assert result.conservation_ok

def test_slice_packet_accounting_rejects_untracked_residual():
    flow = make_slice_accounting_flow(
        generated=3,
        delivered=1,
        dropped=1,
        residual=1,
    )

    # Deliberately remove the residual packet from every
    # observable simulator queue.
    flow.appBuff.pckts.clear()

    ue = make_slice_accounting_ue(
        "ue1",
        flow,
    )

    slice_obj = (
        make_slice_accounting_slice(
            [
                ue,
            ]
        )
    )

    with pytest.raises(
        RuntimeError,
        match="untracked residual",
    ):
        collect_slice_packet_accounting(
            slice_obj
        )
