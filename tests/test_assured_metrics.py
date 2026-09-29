import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))

from assured_metrics import (
    slice_mean_sinr,
    slice_max_hol_delay,
)


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
