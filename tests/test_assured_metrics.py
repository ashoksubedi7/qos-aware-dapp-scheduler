import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))

from assured_metrics import (
    build_m1_state,
    build_m2_state,
    normalize_backlog,
    normalize_sinr,
    normalize_urllc_urgency,
    slice_backlog,
    slice_mean_sinr,
    slice_max_hol_delay,
)
from types import SimpleNamespace


def make_test_slices(
    embb_backlog=0,
    urllc_backlog=0,
    mmtc_backlog=0,
    embb_sinr=20.0,
    urllc_sinr=20.0,
    mmtc_sinr=20.0,
):
    def make_slice(backlog, sinr):
        ue = SimpleNamespace(
            radioLinks=SimpleNamespace(
                linkQuality=sinr
            ),
            bearers=[
                SimpleNamespace(
                    buffer=SimpleNamespace(
                        pckts=[
                            object()
                            for _ in range(backlog)
                        ]
                    )
                )
            ],
        )

        scheduler = SimpleNamespace(
            ues={
                "ue1": ue
            }
        )

        return SimpleNamespace(
            schedulerDL=scheduler
        )

    return {
        "eMBB": make_slice(
            embb_backlog,
            embb_sinr,
        ),
        "URLLC": make_slice(
            urllc_backlog,
            urllc_sinr,
        ),
        "mMTC": make_slice(
            mmtc_backlog,
            mmtc_sinr,
        ),
    }

def make_ue(sinr, packet_times):
    packets = deque(
        SimpleNamespace(tIn=t)
        for t in packet_times
    )

    bearer = SimpleNamespace(
        buffer=SimpleNamespace(pckts=packets)
    )

    return SimpleNamespace(
        radioLinks=SimpleNamespace(linkQuality=sinr),
        bearers=[bearer],
    )


def make_slice(ues):
    scheduler = SimpleNamespace(ues=ues)
    return SimpleNamespace(schedulerDL=scheduler)


def test_mean_sinr():
    slice_obj = make_slice(
        {
            "ue1": make_ue(10.0, []),
            "ue2": make_ue(20.0, []),
            "ue3": make_ue(30.0, []),
        }
    )

    assert np.isclose(
        slice_mean_sinr(slice_obj),
        20.0,
    )


def test_empty_slice_mean_sinr():
    slice_obj = make_slice({})

    assert slice_mean_sinr(slice_obj) == 0.0


def test_max_hol_delay():
    slice_obj = make_slice(
        {
            "ue1": make_ue(10.0, [7.0, 8.0]),
            "ue2": make_ue(15.0, [5.0, 9.0]),
        }
    )

    # At t=10, the oldest waiting packet arrived at t=5.
    assert np.isclose(
        slice_max_hol_delay(slice_obj, 10.0),
        5.0,
    )


def test_empty_queues_have_zero_hol():
    slice_obj = make_slice(
        {
            "ue1": make_ue(10.0, []),
            "ue2": make_ue(15.0, []),
        }
    )

    assert slice_max_hol_delay(slice_obj, 10.0) == 0.0


from assured_metrics import (
    normalize_backlog,
    normalize_sinr,
    normalize_urllc_urgency,
)


def test_backlog_normalization():
    assert np.isclose(normalize_backlog(0), 0.0)
    assert np.isclose(normalize_backlog(250), 0.5)
    assert np.isclose(normalize_backlog(500), 1.0)
    assert np.isclose(normalize_backlog(900), 1.0)


def test_sinr_normalization():
    assert np.isclose(normalize_sinr(0.0), 0.0)
    assert np.isclose(normalize_sinr(17.5), 0.5)
    assert np.isclose(normalize_sinr(35.0), 1.0)
    assert np.isclose(normalize_sinr(37.0), 1.0)


def test_sinr_normalization_clips_low_values():
    assert np.isclose(normalize_sinr(-5.0), 0.0)


def test_urllc_urgency_normalization():
    assert np.isclose(
        normalize_urllc_urgency(0.0, 1.0),
        0.0,
    )

    assert np.isclose(
        normalize_urllc_urgency(0.5, 1.0),
        0.5,
    )

    assert np.isclose(
        normalize_urllc_urgency(1.0, 1.0),
        1.0,
    )

    assert np.isclose(
        normalize_urllc_urgency(2.0, 1.0),
        1.0,
    )


def test_invalid_deadline():
    try:
        normalize_urllc_urgency(0.5, 0.0)
        assert False
    except ValueError:
        assert True
def test_m1_uses_configurable_backlog_cap():
    slices = make_test_slices(
        embb_backlog=250,
        urllc_backlog=250,
        mmtc_backlog=250,
    )

    state = build_m1_state(
        slices,
        backlog_cap=1000,
    )

    assert np.isclose(
        state[0],
        0.25,
    )


def test_m1_uses_configurable_sinr_bounds():
    slices = make_test_slices(
        embb_sinr=20,
        urllc_sinr=20,
        mmtc_sinr=20,
    )

    state = build_m1_state(
        slices,
        sinr_min_db=0,
        sinr_max_db=40,
    )

    assert np.isclose(
        state[3],
        0.5,
    )
def make_test_slices(
    embb_backlog=0,
    urllc_backlog=0,
    mmtc_backlog=0,
    embb_sinr=20.0,
    urllc_sinr=20.0,
    mmtc_sinr=20.0,
):
    def make_slice(backlog, sinr):
        scheduler = SimpleNamespace()

        scheduler.updSumPcks = (
            lambda value=backlog: value
        )

        scheduler.ues = {
            "ue1": SimpleNamespace(
                radioLinks=SimpleNamespace(
                    linkQuality=sinr
                )
            )
        }

        return SimpleNamespace(
            schedulerDL=scheduler
        )

    return {
        "eMBB": make_slice(
            embb_backlog,
            embb_sinr,
        ),
        "URLLC": make_slice(
            urllc_backlog,
            urllc_sinr,
        ),
        "mMTC": make_slice(
            mmtc_backlog,
            mmtc_sinr,
        ),
    }
