from dataclasses import dataclass


@dataclass
class SliceSnapshot:
    sent_packets: int
    lost_packets: int
    received_bytes: int
    backlog: int


@dataclass
class SliceInterval:
    generated_packets: int
    lost_packets: int
    received_bytes: int
    backlog_before: int
    backlog_after: int

    @property
    def served_backlog(self):
        return max(
            0,
            self.backlog_before
            + self.generated_packets
            - self.lost_packets
            - self.backlog_after,
        )


def snapshot_slice(slice_obj):
    sent_packets = 0
    lost_packets = 0
    received_bytes = 0

    for ue in slice_obj.schedulerDL.ues.values():
        flow = ue.packetFlows[0]

        sent_packets += int(flow.sentPackets)
        lost_packets += int(flow.lostPackets)
        received_bytes += int(flow.rcvdBytes)

    backlog = int(
        slice_obj.schedulerDL.updSumPcks()
    )

    return SliceSnapshot(
        sent_packets=sent_packets,
        lost_packets=lost_packets,
        received_bytes=received_bytes,
        backlog=backlog,
    )


def interval_from_snapshots(before, after):
    return SliceInterval(
        generated_packets=max(
            0,
            after.sent_packets - before.sent_packets,
        ),
        lost_packets=max(
            0,
            after.lost_packets - before.lost_packets,
        ),
        received_bytes=max(
            0,
            after.received_bytes - before.received_bytes,
        ),
        backlog_before=before.backlog,
        backlog_after=after.backlog,
    )
