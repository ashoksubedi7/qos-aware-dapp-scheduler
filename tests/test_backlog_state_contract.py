from types import SimpleNamespace

import numpy as np
import pytest

from assured_metrics import (
    build_m1_state,
    build_m2_state,
)
from assured_scheduler import AssuredScheduler
from config.experiment_config import ExperimentConfig


EMBB_CAP_BYTES = 818_176
URLLC_CAP_BYTES = 1_884_160
MMTC_CAP_BYTES = 6_244_352


def _packet(
    size,
    packet_id,
):
    # Mark the packet as already scheduled so that it
    # contributes to bearer backlog bytes but not to
    # unscheduled URLLC HOL urgency.
    return SimpleNamespace(
        size=float(size),
        secNum=int(packet_id),
        scheduled_at=0.0,
        tIn=0.0,
    )


def _slice(
    backlog_bytes,
    packet_count,
    sinr,
    packet_id,
):
    packet = _packet(
        backlog_bytes,
        packet_id,
    )

    ue = SimpleNamespace(
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=[
                        packet
                    ]
                )
            )
        ],
        packetFlows=[
            SimpleNamespace(
                appBuff=SimpleNamespace(
                    pckts=[]
                )
            )
        ],
        radioLinks=SimpleNamespace(
            linkQuality=float(
                sinr
            )
        ),
    )

    scheduler_dl = SimpleNamespace(
        ues={
            "ue": ue
        },
        # Deliberately disagree with the byte value.
        # The new state contract must ignore this
        # packet-count backlog.
        updSumPcks=lambda: float(
            packet_count
        ),
    )

    return SimpleNamespace(
        schedulerDL=scheduler_dl
    )


def _slices():
    return {
        "eMBB": _slice(
            backlog_bytes=(
                EMBB_CAP_BYTES / 2
            ),
            packet_count=499,
            sinr=17.5,
            packet_id=1,
        ),
        "URLLC": _slice(
            backlog_bytes=(
                URLLC_CAP_BYTES / 2
            ),
            packet_count=498,
            sinr=17.5,
            packet_id=2,
        ),
        "mMTC": _slice(
            backlog_bytes=(
                MMTC_CAP_BYTES / 2
            ),
            packet_count=497,
            sinr=17.5,
            packet_id=3,
        ),
    }


def _state_kwargs():
    return {
        "embb_backlog_cap_bytes":
            EMBB_CAP_BYTES,
        "urllc_backlog_cap_bytes":
            URLLC_CAP_BYTES,
        "mmtc_backlog_cap_bytes":
            MMTC_CAP_BYTES,
        "sinr_min_db": 0.0,
        "sinr_max_db": 35.0,
    }


def test_experiment_config_freezes_calibrated_byte_caps():
    config = ExperimentConfig()

    assert (
        config.embb_backlog_cap_bytes
        == EMBB_CAP_BYTES
    )

    assert (
        config.urllc_backlog_cap_bytes
        == URLLC_CAP_BYTES
    )

    assert (
        config.mmtc_backlog_cap_bytes
        == MMTC_CAP_BYTES
    )

    # The ambiguous legacy single-cap contract must
    # disappear once the byte-state migration is done.
    assert not hasattr(
        config,
        "backlog_cap",
    )



@pytest.mark.parametrize(
    "field_name",
    (
        "embb_backlog_cap_bytes",
        "urllc_backlog_cap_bytes",
        "mmtc_backlog_cap_bytes",
    ),
)
@pytest.mark.parametrize(
    "invalid_value",
    (
        True,
        1.5,
    ),
)
def test_backlog_caps_reject_non_integer_types(
    field_name,
    invalid_value,
):
    kwargs = {
        field_name: invalid_value
    }

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} must be a "
            "non-Boolean integer"
        ),
    ):
        ExperimentConfig(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "embb_backlog_cap_bytes",
        "urllc_backlog_cap_bytes",
        "mmtc_backlog_cap_bytes",
    ),
)
@pytest.mark.parametrize(
    "invalid_value",
    (
        0,
        -1,
    ),
)
def test_backlog_caps_reject_nonpositive_values(
    field_name,
    invalid_value,
):
    kwargs = {
        field_name: invalid_value
    }

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must be greater "
            "than zero"
        ),
    ):
        ExperimentConfig(
            **kwargs
        )


def test_m1_uses_bearer_bytes_and_slice_specific_caps():
    state = build_m1_state(
        _slices(),
        **_state_kwargs(),
    )

    assert state.shape == (
        6,
    )

    np.testing.assert_allclose(
        state[:3],
        np.array(
            [
                0.5,
                0.5,
                0.5,
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_m2_preserves_seven_feature_state_contract():
    state = build_m2_state(
        _slices(),
        now=10.0,
        urllc_deadline=1.0,
        **_state_kwargs(),
    )

    assert state.shape == (
        7,
    )

    np.testing.assert_allclose(
        state[:3],
        np.array(
            [
                0.5,
                0.5,
                0.5,
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )

    # All bearer packets in this fixture are already
    # scheduled, so no unscheduled URLLC urgency exists.
    assert np.isclose(
        state[6],
        0.0,
    )


class _DiagnosticsRecorder:
    def __init__(self):
        self.backlog_calls = []
        self.sinr_calls = []
        self.urgency_calls = []
        self.backlog_slice_names = []
        self.sinr_slice_names = []

    def record_backlog(
        self,
        backlog,
        cap,
        slice_name=None,
    ):
        self.backlog_slice_names.append(
            slice_name
        )
        self.backlog_calls.append(
            (
                float(backlog),
                float(cap),
            )
        )

    def record_sinr(
        self,
        sinr,
        minimum,
        maximum,
        slice_name=None,
    ):
        self.sinr_slice_names.append(
            slice_name
        )
        self.sinr_calls.append(
            (
                float(sinr),
                float(minimum),
                float(maximum),
            )
        )

    def record_urgency(
        self,
        urgency,
    ):
        self.urgency_calls.append(
            float(urgency)
        )


def test_normalization_diagnostics_use_same_byte_contract():
    slices = _slices()

    scheduler = AssuredScheduler.__new__(
        AssuredScheduler
    )

    scheduler._canonical_slices = (
        lambda: slices
    )

    scheduler.config = SimpleNamespace(
        embb_backlog_cap_bytes=(
            EMBB_CAP_BYTES
        ),
        urllc_backlog_cap_bytes=(
            URLLC_CAP_BYTES
        ),
        mmtc_backlog_cap_bytes=(
            MMTC_CAP_BYTES
        ),
        sinr_min_db=0.0,
        sinr_max_db=35.0,
        urllc_scheduling_deadline_ms=1.0,
    )

    scheduler.normalization_diagnostics = (
        _DiagnosticsRecorder()
    )

    scheduler.record_normalization_diagnostics(
        now=10.0
    )

    assert (
        scheduler
        .normalization_diagnostics
        .backlog_slice_names
        == [
            "eMBB",
            "URLLC",
            "mMTC",
        ]
    )

    assert (
        scheduler
        .normalization_diagnostics
        .sinr_slice_names
        == [
            "eMBB",
            "URLLC",
            "mMTC",
        ]
    )

    assert (
        scheduler
        .normalization_diagnostics
        .backlog_calls
        == [
            (
                EMBB_CAP_BYTES / 2,
                float(
                    EMBB_CAP_BYTES
                ),
            ),
            (
                URLLC_CAP_BYTES / 2,
                float(
                    URLLC_CAP_BYTES
                ),
            ),
            (
                MMTC_CAP_BYTES / 2,
                float(
                    MMTC_CAP_BYTES
                ),
            ),
        ]
    )
