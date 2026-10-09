import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))

from UE import Packet, PacketFlow


def test_packet_deadline_defaults():
    packet = Packet(1, 100, 0, "ue1")

    assert packet.deadline is None
    assert packet.deadline_missed is False
    assert packet.scheduled_at is None
    assert packet.scheduling_delay is None

def make_urllc_flow():
    return PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

def test_packetflow_deadline_configuration():
    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    assert flow.schedulingDeadline == 1.0
    assert flow.deadlineMisses == 0
    assert flow.deadlineEvaluated == 0
    assert flow.schedulingDelays == []
    
def test_packet_is_evaluated_only_once():
    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    packet = Packet(1, 300, 0, "ue1")
    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.recordSchedulingOutcome(
        packet,
        12.0,
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1
    assert flow.schedulingDelays == [2.0]

    # Simulate the same partially served packet being scheduled again.
    flow.recordSchedulingOutcome(
        packet,
        15.0,
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1
    assert flow.schedulingDelays == [2.0]

    assert packet.scheduled_at == 12.0
    assert packet.scheduling_delay == 2.0
def test_packetflow_service_counters_start_zero():
    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    assert flow.deliveredPackets == 0
    assert flow.deliveredBytes == 0
def test_completion_delay_is_measured_from_generation():
    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0

    flow.recordSchedulingOutcome(
        packet,
        10.5,
    )

    delay = flow.recordDeliveryCompletion(
        packet.secNum,
        12.0,
    )

    assert delay == 2.0
    assert flow.deliveredPackets == 1
    assert flow.completionDelays == [2.0]


def test_completion_is_recorded_only_once():
    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 5.0

    flow.recordSchedulingOutcome(
        packet,
        5.2,
    )

    first = flow.recordDeliveryCompletion(
        packet.secNum,
        6.0,
    )

    second = flow.recordDeliveryCompletion(
        packet.secNum,
        7.0,
    )

    assert first == 1.0
    assert second is None

    assert flow.deliveredPackets == 1
    assert len(flow.completionDelays) == 1


def test_completion_requires_known_packet():
    flow = PacketFlow(
        1,
        300,
        0.6,
        "ue1",
        "DL",
        "URLLC",
        0,
        1,
        "Constant",
        "Uniform",
        1.0,
    )

    result = flow.recordDeliveryCompletion(
        999,
        5.0,
    )

    assert result is None
    assert flow.deliveredPackets == 0

def test_packet_before_deadline_passes():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.packetGenerationTimes[
        1
    ] = 10.0

    flow.recordSchedulingOutcome(
        packet,
        10.9,
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 0
    assert packet.deadline_missed is False


def test_packet_exactly_at_deadline_passes():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.packetGenerationTimes[
        1
    ] = 10.0

    flow.recordSchedulingOutcome(
        packet,
        11.0,
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 0


def test_unscheduled_packet_crossing_deadline_misses():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.packetGenerationTimes[
        1
    ] = 10.0

    recorded = (
        flow.recordDeadlineCrossing(
            packet,
            11.1,
        )
    )

    assert recorded is True
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1
    assert packet.deadline_missed is True
    assert packet.scheduled_at is None


def test_deadline_crossing_is_recorded_once():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.packetGenerationTimes[
        1
    ] = 10.0

    assert (
        flow.recordDeadlineCrossing(
            packet,
            11.1,
        )
        is True
    )

    assert (
        flow.recordDeadlineCrossing(
            packet,
            12.0,
        )
        is False
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1


def test_late_packet_scheduled_after_crossing_not_double_counted():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.packetGenerationTimes[
        1
    ] = 10.0

    flow.recordDeadlineCrossing(
        packet,
        11.1,
    )

    flow.recordSchedulingOutcome(
        packet,
        20.0,
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1
    assert flow.schedulingDelays == [
        10.0
    ]

    assert packet.scheduled_at == 20.0
    assert packet.scheduling_delay == 10.0


def test_unscheduled_permanent_drop_is_deadline_failure():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.generatedPacketIds.add(
        1
    )

    flow.packetGenerationTimes[
        1
    ] = 10.0

    result = flow.recordPacketDrop(
        1
    )

    assert result is True
    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 1


def test_scheduled_packet_drop_does_not_duplicate_deadline():
    flow = make_urllc_flow()

    packet = Packet(
        1,
        300,
        0,
        "ue1",
    )

    packet.tIn = 10.0
    packet.deadline = 1.0

    flow.generatedPacketIds.add(
        1
    )

    flow.packetGenerationTimes[
        1
    ] = 10.0

    flow.recordSchedulingOutcome(
        packet,
        10.5,
    )

    flow.recordPacketDrop(
        1
    )

    assert flow.deadlineEvaluated == 1
    assert flow.deadlineMisses == 0
