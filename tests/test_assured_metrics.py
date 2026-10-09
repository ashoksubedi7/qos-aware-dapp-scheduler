import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from assured_metrics import (
    build_m1_state,
    normalize_backlog,
    normalize_sinr,
    normalize_urllc_urgency,
    slice_backlog_bytes,
    slice_max_hol_delay,
    slice_mean_sinr,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))

from assured_metrics import (
    build_m1_state,
    normalize_backlog,
    normalize_sinr,
    normalize_urllc_urgency,
    slice_max_hol_delay,
    slice_mean_sinr,
)
from UE import Packet


def make_packet(
    packet_id,
    generation_time,
    scheduled_at=None,
):
    packet = Packet(
        packet_id,
        100,
        0,
        "ue1",
    )
    packet.tIn = float(generation_time)
    packet.timestamp = float(generation_time)
    packet.scheduled_at = scheduled_at
    return packet


def make_ue(
    sinr,
    bearer_arrivals=None,
    app_arrivals=None,
):
    bearer_arrivals = list(
        bearer_arrivals or []
    )
    app_arrivals = list(
        app_arrivals or []
    )

    bearer_packets = deque(
        make_packet(
            packet_id=2000 + index,
            generation_time=t_in,
        )
        for index, t_in
        in enumerate(bearer_arrivals)
    )

    app_packets = deque(
        make_packet(
            packet_id=1000 + index,
            generation_time=t_in,
        )
        for index, t_in
        in enumerate(app_arrivals)
    )

    flow = SimpleNamespace(
        appBuff=SimpleNamespace(
            pckts=app_packets
        )
    )

    bearer = SimpleNamespace(
        buffer=SimpleNamespace(
            pckts=bearer_packets
        )
    )

    return SimpleNamespace(
        radioLinks=SimpleNamespace(
            linkQuality=sinr
        ),
        packetFlows=[flow],
        bearers=[bearer],
    )


def make_slice(
    ues,
    backlog=None,
):
    scheduler = SimpleNamespace(
        ues=ues
    )

    if backlog is None:
        backlog = sum(
            len(ue.packetFlows[0].appBuff.pckts)
            + len(ue.bearers[0].buffer.pckts)
            for ue in ues.values()
        )

    scheduler.updSumPcks = (
        lambda value=backlog: value
    )

    return SimpleNamespace(
        schedulerDL=scheduler
    )


def make_test_slices(
    embb_backlog=0,
    urllc_backlog=0,
    mmtc_backlog=0,
    embb_sinr=20.0,
    urllc_sinr=20.0,
    mmtc_sinr=20.0,
):
    def make_named_slice(
        backlog,
        sinr,
    ):
        ue = make_ue(
            sinr=sinr,
            bearer_arrivals=[],
            app_arrivals=[],
        )

        return make_slice(
            {
                "ue1": ue
            },
            backlog=backlog,
        )

    return {
        "eMBB": make_named_slice(
            embb_backlog,
            embb_sinr,
        ),
        "URLLC": make_named_slice(
            urllc_backlog,
            urllc_sinr,
        ),
        "mMTC": make_named_slice(
            mmtc_backlog,
            mmtc_sinr,
        ),
    }


def test_mean_sinr():
    slice_obj = make_slice(
        {
            "ue1": make_ue(10.0),
            "ue2": make_ue(20.0),
            "ue3": make_ue(30.0),
        }
    )

    assert np.isclose(
        slice_mean_sinr(slice_obj),
        20.0,
    )


def test_empty_slice_mean_sinr():
    slice_obj = make_slice({})

    assert (
        slice_mean_sinr(slice_obj)
        == 0.0
    )


def test_max_hol_delay():
    slice_obj = make_slice(
        {
            "ue1": make_ue(
                10.0,
                bearer_arrivals=[
                    7.0,
                    8.0,
                ],
            ),
            "ue2": make_ue(
                15.0,
                bearer_arrivals=[
                    5.0,
                    9.0,
                ],
            ),
        }
    )

    # At t=10, the oldest unscheduled packet arrived at t=5.
    assert np.isclose(
        slice_max_hol_delay(
            slice_obj,
            10.0,
        ),
        5.0,
    )


def test_empty_queues_have_zero_hol():
    slice_obj = make_slice(
        {
            "ue1": make_ue(10.0),
            "ue2": make_ue(15.0),
        }
    )

    assert (
        slice_max_hol_delay(
            slice_obj,
            10.0,
        )
        == 0.0
    )


def test_backlog_normalization():
    assert np.isclose(
        normalize_backlog(0),
        0.0,
    )
    assert np.isclose(
        normalize_backlog(250),
        0.5,
    )
    assert np.isclose(
        normalize_backlog(500),
        1.0,
    )
    assert np.isclose(
        normalize_backlog(900),
        1.0,
    )


def test_sinr_normalization():
    assert np.isclose(
        normalize_sinr(0.0),
        0.0,
    )
    assert np.isclose(
        normalize_sinr(17.5),
        0.5,
    )
    assert np.isclose(
        normalize_sinr(35.0),
        1.0,
    )
    assert np.isclose(
        normalize_sinr(37.0),
        1.0,
    )


def test_sinr_normalization_clips_low_values():
    assert np.isclose(
        normalize_sinr(-5.0),
        0.0,
    )


def test_urllc_urgency_normalization():
    assert np.isclose(
        normalize_urllc_urgency(
            0.0,
            1.0,
        ),
        0.0,
    )

    assert np.isclose(
        normalize_urllc_urgency(
            0.5,
            1.0,
        ),
        0.5,
    )

    assert np.isclose(
        normalize_urllc_urgency(
            1.0,
            1.0,
        ),
        1.0,
    )

    assert np.isclose(
        normalize_urllc_urgency(
            2.0,
            1.0,
        ),
        1.0,
    )


def test_invalid_deadline():
    with pytest.raises(ValueError):
        normalize_urllc_urgency(
            0.5,
            0.0,
        )


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


class DummyQueue:
    def __init__(
        self,
        packets=None,
    ):
        self.pckts = list(
            packets or []
        )


class DummyBearer:
    def __init__(
        self,
        packets=None,
    ):
        self.buffer = DummyQueue(
            packets
        )


class DummyFlow:
    def __init__(
        self,
        packets=None,
    ):
        self.appBuff = DummyQueue(
            packets
        )


class DummyUE:
    def __init__(
        self,
        app_packets=None,
        bearer_packets=None,
    ):
        self.packetFlows = [
            DummyFlow(
                app_packets
            )
        ]

        self.bearers = [
            DummyBearer(
                bearer_packets
            )
        ]


class DummyScheduler:
    def __init__(
        self,
        ues,
    ):
        self.ues = ues


class DummySlice:
    def __init__(
        self,
        ues,
    ):
        self.schedulerDL = (
            DummyScheduler(
                ues
            )
        )


def test_hol_includes_unscheduled_app_buffer_packet():
    packet = make_packet(
        packet_id=1,
        generation_time=2.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                app_packets=[
                    packet,
                ],
            )
        }
    )

    result = slice_max_hol_delay(
        slice_obj,
        now=7.0,
    )

    assert result == pytest.approx(
        5.0
    )


def test_hol_includes_unscheduled_bearer_packet():
    packet = make_packet(
        packet_id=1,
        generation_time=3.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                bearer_packets=[
                    packet,
                ],
            )
        }
    )

    result = slice_max_hol_delay(
        slice_obj,
        now=8.0,
    )

    assert result == pytest.approx(
        5.0
    )


def test_hol_ignores_already_scheduled_packet():
    old_scheduled = make_packet(
        packet_id=1,
        generation_time=1.0,
        scheduled_at=2.0,
    )

    younger_unscheduled = make_packet(
        packet_id=2,
        generation_time=6.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                app_packets=[
                    younger_unscheduled,
                ],
                bearer_packets=[
                    old_scheduled,
                ],
            )
        }
    )

    result = slice_max_hol_delay(
        slice_obj,
        now=10.0,
    )

    assert result == pytest.approx(
        4.0
    )


def test_hol_deduplicates_same_packet_across_buffers():
    packet = make_packet(
        packet_id=1,
        generation_time=3.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                app_packets=[
                    packet,
                ],
                bearer_packets=[
                    packet,
                ],
            )
        }
    )

    result = slice_max_hol_delay(
        slice_obj,
        now=10.0,
    )

    assert result == pytest.approx(
        7.0
    )


def test_hol_returns_oldest_unscheduled_packet():
    packet_1 = make_packet(
        packet_id=1,
        generation_time=8.0,
    )

    packet_2 = make_packet(
        packet_id=2,
        generation_time=2.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                app_packets=[
                    packet_1,
                ],
                bearer_packets=[
                    packet_2,
                ],
            )
        }
    )

    result = slice_max_hol_delay(
        slice_obj,
        now=10.0,
    )

    assert result == pytest.approx(
        8.0
    )


def test_hol_returns_zero_when_no_unscheduled_packets():
    scheduled = make_packet(
        packet_id=1,
        generation_time=1.0,
        scheduled_at=2.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                bearer_packets=[
                    scheduled,
                ],
            )
        }
    )

    result = slice_max_hol_delay(
        slice_obj,
        now=10.0,
    )

    assert result == pytest.approx(
        0.0
    )


def test_hol_rejects_future_generation_time():
    packet = make_packet(
        packet_id=1,
        generation_time=11.0,
    )

    slice_obj = DummySlice(
        {
            "ue1": DummyUE(
                app_packets=[
                    packet,
                ],
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="future",
    ):
        slice_max_hol_delay(
            slice_obj,
            now=10.0,
        )

def test_slice_backlog_bytes_sums_bearer_bytes():
    ue1 = SimpleNamespace(
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=deque(
                        [
                            SimpleNamespace(
                                size=100
                            ),
                            SimpleNamespace(
                                size=1200
                            ),
                        ]
                    )
                )
            )
        ]
    )

    ue2 = SimpleNamespace(
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=deque(
                        [
                            SimpleNamespace(
                                size=300
                            ),
                        ]
                    )
                )
            )
        ]
    )

    slice_obj = SimpleNamespace(
        schedulerDL=SimpleNamespace(
            ues={
                "ue1": ue1,
                "ue2": ue2,
            }
        )
    )

    assert (
        slice_backlog_bytes(
            slice_obj
        )
        == 1600.0
    )

def test_slice_backlog_bytes_excludes_app_buffer():
    ue = SimpleNamespace(
        packetFlows=[
            SimpleNamespace(
                appBuff=SimpleNamespace(
                    pckts=deque(
                        [
                            SimpleNamespace(
                                size=9000
                            ),
                        ]
                    )
                )
            )
        ],
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=deque(
                        [
                            SimpleNamespace(
                                size=400
                            ),
                        ]
                    )
                )
            )
        ],
    )

    slice_obj = SimpleNamespace(
        schedulerDL=SimpleNamespace(
            ues={
                "ue1": ue
            }
        )
    )

    assert (
        slice_backlog_bytes(
            slice_obj
        )
        == 400.0
    )

def test_slice_backlog_bytes_handles_ue_without_bearer():
    slice_obj = SimpleNamespace(
        schedulerDL=SimpleNamespace(
            ues={
                "ue1": SimpleNamespace(
                    bearers=[]
                )
            }
        )
    )

    assert (
        slice_backlog_bytes(
            slice_obj
        )
        == 0.0
    )

def test_slice_backlog_bytes_rejects_negative_size():
    slice_obj = SimpleNamespace(
        schedulerDL=SimpleNamespace(
            ues={
                "ue1": SimpleNamespace(
                    bearers=[
                        SimpleNamespace(
                            buffer=SimpleNamespace(
                                pckts=[
                                    SimpleNamespace(
                                        size=-1
                                    )
                                ]
                            )
                        )
                    ]
                )
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "packet size cannot be negative"
        ),
    ):
        slice_backlog_bytes(
            slice_obj
        )
