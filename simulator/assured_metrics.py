import numpy as np


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


def build_m1_state(slices):
    """Context-aware state: backlog plus mean SINR for each slice."""
    embb = slices["eMBB"]
    urllc = slices["URLLC"]
    mmtc = slices["mMTC"]

    return np.array(
        [
            slice_backlog(embb),
            slice_backlog(urllc),
            slice_backlog(mmtc),
            slice_mean_sinr(embb),
            slice_mean_sinr(urllc),
            slice_mean_sinr(mmtc),
        ],
        dtype=np.float32,
    )


def build_m2_state(slices, now):
    """QoS-aware state adds URLLC deadline urgency through max HoL."""
    m1_state = build_m1_state(slices)

    urllc_hol = slice_max_hol_delay(
        slices["URLLC"],
        now,
    )

    return np.append(
        m1_state,
        np.float32(urllc_hol),
    )
