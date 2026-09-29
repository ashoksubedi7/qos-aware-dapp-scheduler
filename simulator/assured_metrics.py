import numpy as np


BACKLOG_CAP = 500.0

# The current AssuredQoS experiments use FDD.
# The simulator's FDD MCS table has meaningful thresholds up to 35 dB.
SINR_MIN_DB = 0.0
SINR_MAX_DB = 35.0


def slice_backlog(slice_obj):
    """Return the number of packets currently waiting in bearer queues."""
    return slice_obj.schedulerDL.updSumPcks()


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


def slice_max_hol_delay(slice_obj, now):
    """Return the largest head-of-line waiting time across UEs."""
    max_hol = 0.0

    for ue in slice_obj.schedulerDL.ues.values():
        queue = ue.bearers[0].buffer.pckts

        if not queue:
            continue

        oldest_packet = queue[0]
        hol = float(now - oldest_packet.tIn)

        if hol > max_hol:
            max_hol = hol

    return max_hol


def normalize_backlog(backlog):
    """Map queue backlog to [0, 1] using the original 500-packet cap."""
    return float(
        np.clip(
            float(backlog) / BACKLOG_CAP,
            0.0,
            1.0,
        )
    )


def normalize_sinr(sinr_db):
    """Normalize FDD SINR to [0, 1] using the simulator MCS range."""
    clipped = np.clip(
        float(sinr_db),
        SINR_MIN_DB,
        SINR_MAX_DB,
    )

    return float(
        (clipped - SINR_MIN_DB)
        / (SINR_MAX_DB - SINR_MIN_DB)
    )


def normalize_urllc_urgency(hol_delay, deadline):
    """Represent URLLC HoL as fraction of the scheduling deadline."""
    if deadline <= 0:
        raise ValueError("URLLC deadline must be greater than zero")

    return float(
        np.clip(
            float(hol_delay) / float(deadline),
            0.0,
            1.0,
        )
    )


def build_m1_state(slices):
    """Build normalized context-aware state."""
    embb = slices["eMBB"]
    urllc = slices["URLLC"]
    mmtc = slices["mMTC"]

    return np.array(
        [
            normalize_backlog(slice_backlog(embb)),
            normalize_backlog(slice_backlog(urllc)),
            normalize_backlog(slice_backlog(mmtc)),
            normalize_sinr(slice_mean_sinr(embb)),
            normalize_sinr(slice_mean_sinr(urllc)),
            normalize_sinr(slice_mean_sinr(mmtc)),
        ],
        dtype=np.float32,
    )


def build_m2_state(slices, now, urllc_deadline):
    """Build normalized QoS-aware state with URLLC urgency."""
    m1_state = build_m1_state(slices)

    urllc_hol = slice_max_hol_delay(
        slices["URLLC"],
        now,
    )

    urgency = normalize_urllc_urgency(
        urllc_hol,
        urllc_deadline,
    )

    return np.append(
        m1_state,
        np.float32(urgency),
    )
