"""Project potential fresh-data TB payload from predicted UE PRBs.

This projection uses the simulator's current setMod calculation.
It does not guarantee TB insertion or timely packet scheduling.
"""

import math
from dataclasses import dataclass

from assurance.rr_prb_distribution import RRDistribution


@dataclass(frozen=True)
class UEPayloadProjection:
    ue_id: str
    assigned_prbs: int
    tbs_bits: float
    potential_payload_bytes: int
    status: str


def project_rr_tb_payload(scheduler, distribution):
    """Project potential fresh-data TB payload for one RR TTI.

    The inspected setMod implementation computes MCS/TBS without
    intentionally changing scheduling state. Noninterference still
    requires validation against a live simulator run.

    Pending retransmissions, signalling, queue availability, and
    scheduler loop ordering are not fully modeled here.
    """
    if not isinstance(distribution, RRDistribution):
        raise TypeError("expected RRDistribution")

    results = []

    for ue_id, assigned_prbs in distribution.assignments:
        prbs = int(assigned_prbs)

        if prbs < 0 or prbs != assigned_prbs:
            raise ValueError("invalid predicted UE PRB assignment")

        if ue_id not in scheduler.ues:
            raise ValueError(f"unknown scheduler UE: {ue_id}")

        ue = scheduler.ues[ue_id]

        if prbs == 0:
            results.append(
                UEPayloadProjection(
                    ue_id=ue_id,
                    assigned_prbs=0,
                    tbs_bits=0.0,
                    potential_payload_bytes=0,
                    status="no_prbs",
                )
            )
            continue

        if ue.pendingTB:
            results.append(
                UEPayloadProjection(
                    ue_id=ue_id,
                    assigned_prbs=prbs,
                    tbs_bits=0.0,
                    potential_payload_bytes=0,
                    status="retransmission_pending",
                )
            )
            continue

        tb = scheduler.setMod(ue_id, prbs)

        if len(tb) != 4:
            raise ValueError("unexpected setMod result")

        tbs_bits = float(tb[0])

        if not math.isfinite(tbs_bits) or tbs_bits < 0:
            raise ValueError("invalid transport-block size")

        # Match IntraSliceScheduler.dataPtoTB conversion.
        payload_bytes = int(tbs_bits / 8.0)

        results.append(
            UEPayloadProjection(
                ue_id=ue_id,
                assigned_prbs=prbs,
                tbs_bits=tbs_bits,
                potential_payload_bytes=payload_bytes,
                status=(
                    "potential_data_tb"
                    if payload_bytes > 0
                    else "no_payload_capacity"
                ),
            )
        )

    return tuple(results)
