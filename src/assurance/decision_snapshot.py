"""Read-only evidence captured before an inter-slice action is applied."""

from common.action_space import get_action_weights
from common.prb_allocation import (
    percentage_action_to_numerology_prbs,
)


SLICE_NAMES = ("eMBB", "URLLC", "mMTC")


def _capture_packet(packet, now, deadline_ms):
    generated_at = float(packet.tIn)

    scheduled_at = getattr(packet, "scheduled_at", None)
    if scheduled_at is not None:
        scheduled_at = float(scheduled_at)

    if deadline_ms is not None:
        deadline_at = generated_at + deadline_ms
        remaining_ms = deadline_at - now
    else:
        deadline_at = None
        remaining_ms = None

    return {
        "packet_id": int(packet.secNum),
        "size_bytes": float(packet.size),
        "generated_at_ms": generated_at,
        "scheduled_at_ms": scheduled_at,
        "deadline_at_ms": deadline_at,
        "remaining_deadline_ms": remaining_ms,
    }


def _capture_ue(ue_id, ue, now, rr_position):
    flows = list(getattr(ue, "packetFlows", []))
    flow = flows[0] if flows else None

    deadline = None
    application_packets = []

    if flow is not None:
        value = getattr(flow, "schedulingDeadline", None)
        if value is not None:
            deadline = float(value)

        application_packets = [
            _capture_packet(packet, now, deadline)
            for packet in flow.appBuff.pckts
        ]

    bearer_queues = []

    for index, bearer in enumerate(
        getattr(ue, "bearers", [])
    ):
        bearer_queues.append({
            "bearer_index": index,
            "packets": [
                _capture_packet(packet, now, deadline)
                for packet in bearer.buffer.pckts
            ],
        })

    radio = getattr(ue, "radioLinks", None)
    sinr = (
        float(radio.linkQuality)
        if radio is not None
        else None
    )

    return {
        "ue_id": str(ue_id),
        "rr_position": rr_position,
        "sinr_db": sinr,
        "bwp_prbs": getattr(ue, "BWPs", None),
        "scheduling_deadline_ms": deadline,
        "application_packets": application_packets,
        "bearer_queues": bearer_queues,
        "pending_tb_count": len(
            getattr(ue, "pendingTB", [])
        ),
        "pending_packet_count": len(
            getattr(ue, "pendingPckts", {})
        ),
    }


def capture_decision_snapshot(
    scheduler,
    now,
    observation,
    action,
):
    """Capture serializable scheduling evidence without mutation."""
    now = float(now)
    action = int(action)

    slices = scheduler._canonical_slices()

    factors = tuple(
        int(slices[name].numRefFactor)
        for name in SLICE_NAMES
    )

    weights = tuple(
        int(value)
        for value in get_action_weights(action)
    )

    allocation = percentage_action_to_numerology_prbs(
        total_reference_prbs=int(scheduler.PRBs),
        weights=weights,
        num_ref_factors=factors,
    )

    captured_slices = {}

    for name in SLICE_NAMES:
        slice_obj = slices[name]
        mac = slice_obj.schedulerDL

        captured_slices[name] = {
            "num_ref_factor": int(slice_obj.numRefFactor),
            "registered_ue_count": len(mac.ues),
            "rr_index": getattr(mac, "ind_u", None),
            "tti_per_ms": getattr(mac, "ttiByms", None),
            "ues": [
                _capture_ue(ue_id, ue, now, position)
                for position, (ue_id, ue)
                in enumerate(mac.ues.items())
            ],
        }

    return {
        "schema_version": 1,
        "time_ms": now,
        "observation": [
            float(value) for value in observation
        ],
        "selected_action": action,
        "action_weights": list(weights),
        "reference_prbs_available": int(scheduler.PRBs),
        "proposed_slice_prbs": [
            int(value) for value in allocation
        ],
        "slices": captured_slices,
    }
