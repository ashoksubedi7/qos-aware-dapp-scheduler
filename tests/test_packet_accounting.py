import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from UE import Packet, PcktQueue, UE

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
def test_multiple_true_drops_increment_lost_packets():
    flow = make_flow()

    flow.generatedPacketIds.update(
        {
            11,
            12,
        }
    )

    assert (
        flow.recordPacketDrop(
            11
        )
        is True
    )

    assert flow.lostPackets == 1

    assert (
        flow.recordPacketDrop(
            12
        )
        is True
    )

    assert flow.lostPackets == 2

    # Duplicate classification must remain idempotent.
    assert (
        flow.recordPacketDrop(
            12
        )
        is False
    )

    assert flow.lostPackets == 2
def test_queue_data_packet_multiple_overflows_preserve_drop_count():
    flow = make_flow()

    flow.sliceName = "URLLC"
    flow.type = "DL"

    packet_1 = Packet(
        101,
        300,
        0,
        "ue1",
    )

    packet_2 = Packet(
        102,
        300,
        0,
        "ue1",
    )

    packet_1.tIn = 1.0
    packet_2.tIn = 2.0

    flow.generatedPacketIds.update(
        {
            101,
            102,
        }
    )

    flow.packetGenerationTimes[
        101
    ] = 1.0

    flow.packetGenerationTimes[
        102
    ] = 2.0

    flow.appBuff.insertPckt(
        packet_1
    )

    flow.appBuff.insertPckt(
        packet_2
    )

    bearer_buffer = PcktQueue()

    # Force the bearer to appear full before
    # queueDataPckt() attempts admission.
    existing_packet = Packet(
        999,
        1000,
        0,
        "ue1",
    )

    bearer_buffer.insertPckt(
        existing_packet
    )

    fake_ue = SimpleNamespace(
        id="ue1",
        packetFlows=[
            flow
        ],
        bearers=[
            SimpleNamespace(
                buffer=bearer_buffer
            )
        ],
    )

    fake_scheduler = SimpleNamespace(
        ues={
            "ue1": fake_ue
        },
        printDebDataDM=(
            lambda *args, **kwargs: None
        ),
    )

    fake_slice = SimpleNamespace(
        schedulerDL=fake_scheduler
    )

    fake_cell = SimpleNamespace(
        maxBuffUE=100,
        interSliceSched=SimpleNamespace(
            slices={
                "URLLC": fake_slice
            }
        ),
    )

    UE.queueDataPckt(
        fake_ue,
        fake_cell,
    )

    assert flow.lostPackets == 1
    assert 101 in flow.droppedPacketIds

    UE.queueDataPckt(
        fake_ue,
        fake_cell,
    )

    assert flow.lostPackets == 2
    assert 102 in flow.droppedPacketIds

    assert flow.droppedPacketIds == {
        101,
        102,
    }


def _make_bearer_admission_fixture(
    current_bytes,
    incoming_bytes,
    max_buffer_bytes=100,
):
    flow = make_flow()

    flow.sliceName = "URLLC"
    flow.type = "DL"

    incoming = Packet(
        2001,
        incoming_bytes,
        0,
        "ue1",
    )

    incoming.tIn = 1.0

    flow.generatedPacketIds.add(
        incoming.secNum
    )

    flow.packetGenerationTimes[
        incoming.secNum
    ] = incoming.tIn

    flow.appBuff.insertPckt(
        incoming
    )

    bearer_buffer = PcktQueue()

    if current_bytes > 0:
        existing = Packet(
            1999,
            current_bytes,
            0,
            "ue1",
        )

        bearer_buffer.insertPckt(
            existing
        )

    fake_ue = SimpleNamespace(
        id="ue1",
        packetFlows=[
            flow
        ],
        bearers=[
            SimpleNamespace(
                buffer=bearer_buffer
            )
        ],
    )

    fake_scheduler = SimpleNamespace(
        ues={
            "ue1": fake_ue
        },
        printDebDataDM=(
            lambda *args, **kwargs: None
        ),
    )

    fake_slice = SimpleNamespace(
        schedulerDL=fake_scheduler
    )

    fake_cell = SimpleNamespace(
        maxBuffUE=max_buffer_bytes,
        interSliceSched=SimpleNamespace(
            slices={
                "URLLC": fake_slice
            }
        ),
    )

    return (
        flow,
        fake_ue,
        fake_cell,
        incoming,
        bearer_buffer,
    )


@pytest.mark.parametrize(
    (
        "current_bytes",
        "incoming_bytes",
        "should_admit",
    ),
    [
        # Strictly below capacity.
        (59, 40, True),

        # Exactly fills capacity.
        (60, 40, True),

        # Would exceed capacity by one byte.
        (60, 41, False),

        # Oversized packet into an empty bearer.
        (0, 101, False),
    ],
)
def test_bearer_admission_enforces_byte_capacity(
    current_bytes,
    incoming_bytes,
    should_admit,
):
    (
        flow,
        fake_ue,
        fake_cell,
        incoming,
        bearer_buffer,
    ) = _make_bearer_admission_fixture(
        current_bytes=current_bytes,
        incoming_bytes=incoming_bytes,
        max_buffer_bytes=100,
    )

    UE.queueDataPckt(
        fake_ue,
        fake_cell,
    )

    final_bytes = sum(
        packet.size
        for packet in bearer_buffer.pckts
    )

    if should_admit:
        assert final_bytes == (
            current_bytes
            + incoming_bytes
        )

        assert final_bytes <= 100

        assert (
            incoming.secNum
            not in flow.droppedPacketIds
        )

        assert flow.lostPackets == 0

    else:
        assert final_bytes == current_bytes

        assert (
            incoming.secNum
            in flow.droppedPacketIds
        )

        assert flow.lostPackets == 1
