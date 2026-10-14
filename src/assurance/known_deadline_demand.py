"""Observe known URLLC first-scheduling deadline demand."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class KnownDeadlinePacket:
    ue_id: str
    packet_id: int
    arrival_ms: float
    deadline_ms: float
    absolute_deadline_ms: float
    slack_ms: float
    deadline_already_evaluated: bool
    locations: tuple[str, ...]

    @property
    def overdue(self):
        # Scheduling exactly at the deadline is allowed.
        return self.slack_ms < 0.0


def collect_known_deadline_demand(slice_obj, now_ms):
    """
    Snapshot currently queued, unscheduled deadline packets.

    Reads application and bearer queues without mutation.
    Deduplicates packet IDs within each UE.

    This function observes demand. It does not predict
    scheduling capacity or establish deadline feasibility.
    """
    now = float(now_ms)

    if not isfinite(now):
        raise ValueError("observation time must be finite")

    found = {}

    for ue_key, ue in slice_obj.schedulerDL.ues.items():
        if not ue.packetFlows:
            continue

        flow = ue.packetFlows[0]
        ue_id = str(ue_key)

        evaluated = set(
            getattr(flow, "deadlineEvaluatedPacketIds", ())
        )

        queues = [
            ("application", flow.appBuff.pckts),
        ]

        if ue.bearers:
            queues.append(
                ("bearer", ue.bearers[0].buffer.pckts)
            )

        for location, packets in queues:
            for packet in packets:
                if packet.deadline is None:
                    continue

                if packet.scheduled_at is not None:
                    continue

                packet_id = int(packet.secNum)
                arrival = float(packet.tIn)
                deadline = float(packet.deadline)

                if (
                    not isfinite(arrival)
                    or not isfinite(deadline)
                    or deadline <= 0.0
                    or arrival > now
                ):
                    raise ValueError(
                        "invalid queued packet timing"
                    )

                key = (ue_id, packet_id)

                if key not in found:
                    found[key] = {
                        "arrival": arrival,
                        "deadline": deadline,
                        "evaluated": packet_id in evaluated,
                        "locations": set(),
                    }

                entry = found[key]

                if (
                    entry["arrival"] != arrival
                    or entry["deadline"] != deadline
                ):
                    raise ValueError(
                        "conflicting copies of queued packet"
                    )

                entry["locations"].add(location)

    result = []

    for (ue_id, packet_id), entry in found.items():
        arrival = entry["arrival"]
        deadline = entry["deadline"]
        absolute_deadline = arrival + deadline

        result.append(
            KnownDeadlinePacket(
                ue_id=ue_id,
                packet_id=packet_id,
                arrival_ms=arrival,
                deadline_ms=deadline,
                absolute_deadline_ms=absolute_deadline,
                slack_ms=absolute_deadline - now,
                deadline_already_evaluated=entry["evaluated"],
                locations=tuple(sorted(entry["locations"])),
            )
        )

    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.absolute_deadline_ms,
                item.ue_id,
                item.packet_id,
            ),
        )
    )
