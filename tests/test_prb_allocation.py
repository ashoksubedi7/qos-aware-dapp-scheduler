import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.action_space import (
    ACTION_WEIGHTS,
)
from common.prb_allocation import (
    percentage_action_to_prbs,
)


def test_idle_action_allocates_zero():
    assert percentage_action_to_prbs(
        52,
        (0, 0, 0),
    ) == (0, 0, 0)


def test_full_budget_is_preserved():
    allocation = percentage_action_to_prbs(
        52,
        (60, 20, 20),
    )
    assert sum(allocation) == 52


def test_exact_allocation():
    allocation = percentage_action_to_prbs(
        100,
        (60, 20, 20),
    )
    assert allocation == (
        60,
        20,
        20,
    )


def test_rounding_preserves_budget():
    allocation = percentage_action_to_prbs(
        11,
        (40, 40, 20),
    )
    assert allocation == (
        5,
        4,
        2,
    )
    assert sum(allocation) == 11


def test_every_non_idle_action_preserves_budget():
    for weights in ACTION_WEIGHTS[1:]:
        allocation = percentage_action_to_prbs(
            52,
            weights,
        )
        assert sum(allocation) == 52


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        percentage_action_to_prbs(
            -1,
            (60, 20, 20),
        )


def test_invalid_weight_sum_rejected():
    with pytest.raises(ValueError):
        percentage_action_to_prbs(
            52,
            (50, 20, 20),
        )
from common.prb_allocation import (
    percentage_action_to_numerology_prbs,
    reference_prbs_used,
)


def test_equal_numerology_matches_common_allocation():
    allocation = percentage_action_to_numerology_prbs(
        52,
        (60, 20, 20),
        (1, 1, 1),
    )

    assert allocation == (
        31,
        10,
        11,
    )

    assert reference_prbs_used(
        allocation,
        (1, 1, 1),
    ) == 52


def test_mixed_numerology_preserves_reference_budget():
    allocation = percentage_action_to_numerology_prbs(
        52,
        (60, 20, 20),
        (1, 4, 1),
    )

    assert reference_prbs_used(
        allocation,
        (1, 4, 1),
    ) == 52


def test_mixed_numerology_returns_integer_slice_prbs():
    allocation = percentage_action_to_numerology_prbs(
        52,
        (60, 20, 20),
        (1, 4, 1),
    )

    assert allocation == (
        30,
        3,
        10,
    )


def test_numerology_idle_action():
    allocation = percentage_action_to_numerology_prbs(
        52,
        (0, 0, 0),
        (1, 4, 1),
    )

    assert allocation == (
        0,
        0,
        0,
    )


def test_invalid_numerology_factor_rejected():
    with pytest.raises(ValueError):
        percentage_action_to_numerology_prbs(
            52,
            (60, 20, 20),
            (1, 0, 1),
        )
        
