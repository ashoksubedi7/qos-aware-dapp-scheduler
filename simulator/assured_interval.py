from metrics.deadline_metrics import snapshot_deadline_counts
from metrics.transition_metrics import (
    RadioCounters,
    SliceCounters,
)


def snapshot_slice_counters(slice_obj):
    generated = 0
    delivered = 0
    delivered_bytes = 0

    for ue in slice_obj.schedulerDL.ues.values():
        flow = ue.packetFlows[0]

        generated += int(flow.sentPackets)
        delivered += int(flow.deliveredPackets)
        delivered_bytes += int(flow.deliveredBytes)

    backlog = int(
        slice_obj.schedulerDL.updSumPcks()
    )

    return SliceCounters(
        generated_packets=generated,
        delivered_packets=delivered,
        delivered_bytes=delivered_bytes,
        backlog=backlog,
    )


def snapshot_radio_counters(slices):
    used = 0
    available = 0

    for slice_obj in slices.values():
        scheduler = slice_obj.schedulerDL

        used += int(scheduler.assuredPrbsUsed)
        available += int(scheduler.assuredPrbsAvailable)

    return RadioCounters(
        used_prbs=used,
        available_prbs=available,
    )


def snapshot_urllc_deadline(slice_obj):
    flows = [
        ue.packetFlows[0]
        for ue in slice_obj.schedulerDL.ues.values()
    ]

    return snapshot_deadline_counts(flows)
