import numpy as np


BACKLOG_CAP = 500.0

# The current AssuredQoS experiments use FDD.
# The simulator's FDD MCS table has meaningful thresholds up to 35 dB.
SINR_MIN_DB = 0.0
SINR_MAX_DB = 35.0


def slice_backlog(slice_obj):
    """Return total DL packet backlog for a slice."""
    return float(
        slice_obj.schedulerDL.updSumPcks()
    )

def slice_backlog_bytes(slice_obj):
    """
    Return scheduler-visible DL bearer backlog in bytes.

    Only bytes currently waiting in each UE's first
    DL bearer queue are counted.

    Application-buffer bytes and already committed /
    in-flight transport-block bytes are not included.
    """

    total_bytes = 0.0

    for ue in (
        slice_obj
        .schedulerDL
        .ues
        .values()
    ):
        if not ue.bearers:
            continue

        for packet in (
            ue.bearers[
                0
            ].buffer.pckts
        ):
            size = float(
                packet.size
            )

            if size < 0:
                raise ValueError(
                    "packet size cannot be negative"
                )

            total_bytes += size

    return float(
        total_bytes
    )

def slice_mean_sinr(slice_obj):
    """Return mean DL SINR across UEs in a slice."""
    ues = slice_obj.schedulerDL.ues

    if not ues:
        return 0.0

    values = [
        float(ue.radioLinks.linkQuality)
        for ue in ues.values()
    ]

    return float(np.mean(values))
def slice_max_hol_delay(
    slice_obj,
    now,
):
    """
    Return the maximum waiting time among unique packets
    that have not yet received their first scheduling
    decision.

    Both the application buffer and bearer buffer are
    inspected because scheduling delay is measured from
    packet generation time.

    Already-scheduled packets are excluded, including
    residual fragments of packets previously selected.
    """

    now = float(now)

    max_hol = 0.0

    for ue in slice_obj.schedulerDL.ues.values():
        candidate_packets = []

        flow = ue.packetFlows[0]

        candidate_packets.extend(
            flow.appBuff.pckts
        )

        if ue.bearers:
            candidate_packets.extend(
                ue.bearers[0].buffer.pckts
            )

        seen_packet_ids = set()

        for packet in candidate_packets:
            packet_id = int(
                packet.secNum
            )

            if packet_id in seen_packet_ids:
                continue

            seen_packet_ids.add(
                packet_id
            )

            if packet.scheduled_at is not None:
                continue

            hol = (
                now
                - float(packet.tIn)
            )

            if hol < 0:
                raise ValueError(
                    "packet generation time "
                    "cannot be in the future"
                )

            max_hol = max(
                max_hol,
                hol,
            )

    return float(max_hol)


def normalize_backlog(
    backlog,
    cap=BACKLOG_CAP,
):
    if cap <= 0:
        raise ValueError(
            "backlog cap must be positive"
        )

    return min(
        max(float(backlog), 0.0),
        float(cap),
    ) / float(cap)


def normalize_sinr(
    sinr,
    minimum=SINR_MIN_DB,
    maximum=SINR_MAX_DB,
):
    if maximum <= minimum:
        raise ValueError(
            "SINR maximum must exceed minimum"
        )

    clipped = min(
        max(float(sinr), float(minimum)),
        float(maximum),
    )

    return (
        clipped - float(minimum)
    ) / (
        float(maximum)
        - float(minimum)
    )


def normalize_urgency(
    hol_delay,
    deadline,
):
    if deadline <= 0:
        raise ValueError(
            "deadline must be positive"
        )

    raw = (
        float(hol_delay)
        / float(deadline)
    )

    return min(
        max(raw, 0.0),
        1.0,
    )
def normalize_urllc_urgency(
    hol_delay,
    deadline,
):
    return normalize_urgency(
        hol_delay,
        deadline,
    )    
def build_m1_state(
    slices,
    backlog_cap=BACKLOG_CAP,
    sinr_min_db=SINR_MIN_DB,
    sinr_max_db=SINR_MAX_DB,
):
    return np.array(
        [
            normalize_backlog(
                slice_backlog(slices["eMBB"]),
                backlog_cap,
            ),
            normalize_backlog(
                slice_backlog(slices["URLLC"]),
                backlog_cap,
            ),
            normalize_backlog(
                slice_backlog(slices["mMTC"]),
                backlog_cap,
            ),
            normalize_sinr(
                slice_mean_sinr(slices["eMBB"]),
                sinr_min_db,
                sinr_max_db,
            ),
            normalize_sinr(
                slice_mean_sinr(slices["URLLC"]),
                sinr_min_db,
                sinr_max_db,
            ),
            normalize_sinr(
                slice_mean_sinr(slices["mMTC"]),
                sinr_min_db,
                sinr_max_db,
            ),
        ],
        dtype=np.float32,
    )


def build_m2_state(
    slices,
    now,
    urllc_deadline,
    backlog_cap=BACKLOG_CAP,
    sinr_min_db=SINR_MIN_DB,
    sinr_max_db=SINR_MAX_DB,
):
    m1_state = build_m1_state(
        slices,
        backlog_cap=backlog_cap,
        sinr_min_db=sinr_min_db,
        sinr_max_db=sinr_max_db,
    )

    hol_delay = slice_max_hol_delay(
        slices["URLLC"],
        now,
    )

    urgency = normalize_urllc_urgency(
        hol_delay,
        urllc_deadline,
    )

    return np.concatenate(
        [
            m1_state,
            np.array(
                [urgency],
                dtype=np.float32,
            ),
        ]
    )
