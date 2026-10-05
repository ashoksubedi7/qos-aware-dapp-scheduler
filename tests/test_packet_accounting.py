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
