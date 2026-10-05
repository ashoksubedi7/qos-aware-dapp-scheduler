import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reward.assured_reward import (
    clamp01,
    compute_assured_reward,
)


def test_clamp01():
    assert clamp01(-1.0) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(2.0) == 1.0


def test_reward_without_penalties():
    result = compute_assured_reward(
        urllc_service=1.0,
        embb_service=1.0,
        mmtc_service=1.0,
        utilization=1.0,
        deadline_miss_ratio=0.0,
        starvation_penalty=0.0,
    )

    assert np.isclose(result.total, 0.80)


def test_deadline_penalty_reduces_reward():
    good = compute_assured_reward(
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0,
    )

    bad = compute_assured_reward(
        1.0, 1.0, 1.0, 1.0,
        1.0, 0.0,
    )

    assert bad.total < good.total


def test_starvation_penalty_reduces_reward():
    good = compute_assured_reward(
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0,
    )

    bad = compute_assured_reward(
        1.0, 1.0, 1.0, 1.0,
        0.0, 1.0,
    )

    assert bad.total < good.total

def test_custom_weights_change_reward():
    result = compute_assured_reward(
        urllc_service=1.0,
        embb_service=0.0,
        mmtc_service=0.0,
        utilization=0.0,
        deadline_miss_ratio=0.0,
        starvation_penalty=0.0,
        weights={
            "urllc": 1.0,
            "embb": 0.0,
            "mmtc": 0.0,
            "utilization": 0.0,
            "deadline": 0.0,
            "starvation": 0.0,
        },
    )

    assert result.total == 1.0
import pytest


def test_invalid_reward_weights_rejected():
    with pytest.raises(
        ValueError,
        match="sum",
    ):
        compute_assured_reward(
            urllc_service=1.0,
            embb_service=1.0,
            mmtc_service=1.0,
            utilization=1.0,
            deadline_miss_ratio=0.0,
            starvation_penalty=0.0,
            weights={
                "urllc": 0.5,
                "embb": 0.5,
                "mmtc": 0.5,
                "utilization": 0.0,
                "deadline": 0.0,
                "starvation": 0.0,
            },
        )
