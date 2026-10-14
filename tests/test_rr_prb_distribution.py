from types import SimpleNamespace

import pytest

from assurance.rr_prb_distribution import (
    BearerUE,
    predict_rr_distribution,
    predict_action_rr_distribution,
)
from common.prb_allocation import reference_prbs_used


def ue(name, count, bwp):
    return BearerUE(name, count, bwp)


def test_order_dependent_remainder_allocation():
    result = predict_rr_distribution(
        20,
        [
            ue("ue1", 1, 20),
            ue("ue2", 1, 50),
            ue("ue3", 1, 20),
        ],
    )

    assert result.assignments == (
        ("ue1", 4),
        ("ue2", 6),
        ("ue3", 10),
    )
    assert result.total_assigned == 20
    assert result.unused_prbs == 0


def test_application_only_ue_not_counted():
    result = predict_rr_distribution(
        12,
        [
            ue("application_only", 0, 20),
            ue("bearer_ready", 1, 20),
        ],
    )

    assert result.assignments == (
        ("bearer_ready", 12),
    )


def test_zero_bearer_backlog():
    result = predict_rr_distribution(
        20,
        [ue("ue1", 0, 20)],
    )

    assert result.assignments == ()
    assert result.unused_prbs == 20


def test_too_few_prbs_favors_last_backlogged_ue():
    result = predict_rr_distribution(
        3,
        [
            ue(f"ue{i}", 1, 20)
            for i in range(8)
        ],
    )

    assert [n for _, n in result.assignments] == (
        [0] * 7 + [3]
    )


def test_zero_slice_budget():
    result = predict_rr_distribution(
        0,
        [ue("ue1", 1, 20)],
    )

    assert result.assignments == (("ue1", 0),)
    assert result.total_assigned == 0


def test_action_converts_reference_resources():
    ues = [
        ue("ue1", 1, 20),
        ue("ue2", 1, 50),
        ue("ue3", 1, 80),
    ]

    result = predict_action_rr_distribution(
        17,
        total_reference_prbs=100,
        num_ref_factors=(1, 4, 1),
        urllc_ues=ues,
    )

    assert result.weights == (20, 60, 20)
    assert result.slice_prbs == (20, 15, 20)
    assert result.urllc_distribution.assignments == (
        ("ue1", 4),
        ("ue2", 5),
        ("ue3", 6),
    )


def test_zero_urllc_action():
    result = predict_action_rr_distribution(
        15,
        total_reference_prbs=100,
        num_ref_factors=(1, 4, 1),
        urllc_ues=[ue("ue1", 5, 20)],
    )

    assert result.weights == (20, 0, 80)
    assert result.slice_prbs[1] == 0
    assert result.urllc_distribution.total_assigned == 0


def test_all_22_actions_respect_reference_budget():
    for action in range(22):
        result = predict_action_rr_distribution(
            action,
            total_reference_prbs=100,
            num_ref_factors=(1, 4, 1),
            urllc_ues=[ue("ue1", 5, 20)],
        )

        assert reference_prbs_used(
            result.slice_prbs,
            (1, 4, 1),
        ) <= 100

        assert (
            result.urllc_distribution.total_assigned
            <= result.slice_prbs[1]
        )


@pytest.mark.parametrize(
    "snapshots",
    [
        [ue("ue1", 1, 20), ue("ue1", 1, 50)],
        [ue("", 1, 20)],
        [ue("ue1", -1, 20)],
        [ue("ue1", 1, -1)],
        [ue("ue1", 1, float("nan"))],
    ],
)
def test_invalid_ue_snapshots(snapshots):
    with pytest.raises(ValueError):
        predict_rr_distribution(20, snapshots)


@pytest.mark.parametrize(
    "budget",
    [-1, 1.5],
)
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        predict_rr_distribution(
            budget,
            [ue("ue1", 1, 20)],
        )


@pytest.mark.parametrize(
    "budget",
    [0, 1, 2, 3, 8, 12, 20, 40, 100],
)
@pytest.mark.parametrize(
    "counts",
    [
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, 4),
        (0, 2, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ],
)
def test_predictor_matches_production_rr(budget, counts):
    """Differential check against the simulator's actual RR method."""
    from Scheds_Intra import RR_Scheduler

    scheduler = object.__new__(RR_Scheduler)
    scheduler.schType = "Rr"
    scheduler.ues = {}
    scheduler.printResAlloc = lambda: None

    snapshots = []

    for i, count in enumerate(counts):
        name = f"ue{i}"
        bwp = (20, 50, 80)[i % 3]

        scheduler.ues[name] = SimpleNamespace(
            BWPs=bwp,
            prbs=0,
            bearers=[
                SimpleNamespace(
                    buffer=SimpleNamespace(
                        pckts=[object() for _ in range(count)]
                    )
                )
            ],
        )

        snapshots.append(ue(name, count, bwp))

    prediction = predict_rr_distribution(
        budget, snapshots
    )

    actual_total = scheduler.resAlloc(budget)

    assert prediction.total_assigned == actual_total

    for name, expected in prediction.assignments:
        assert scheduler.ues[name].prbs == expected

    for item in snapshots:
        if item.bearer_packet_count == 0:
            assert scheduler.ues[item.ue_id].prbs == 0
