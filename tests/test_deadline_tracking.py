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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
