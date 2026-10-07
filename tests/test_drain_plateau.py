import pytest

from experiments.drain_plateau import (
    evaluate_drain_candidate,
    p99_tolerance_ms,
    select_smallest_stable_drain,
)


def make_slice(
    residual_ratio,
    p99,
    count=100,
):
    return {
        "residual_ratio": residual_ratio,
        "completion_delay_count": count,
        "completion_delay_p99_ms": p99,
    }


def make_result(
    scenario,
    seed,
    drain,
    residual,
    p99,
):
    slice_summary = make_slice(
        residual_ratio=residual,
        p99=p99,
    )

    return {
        "traffic_scenario": scenario,
        "seed": seed,
        "drain_duration_ms": drain,
        "embb": dict(
            slice_summary
        ),
        "urllc": dict(
            slice_summary
        ),
        "mmtc": dict(
            slice_summary
        ),
    }


def test_p99_tolerance_uses_absolute_floor():
    assert (
        p99_tolerance_ms(
            1.0
        )
        == pytest.approx(
            0.1
        )
    )


def test_p99_tolerance_uses_relative_rule_when_larger():
    assert (
        p99_tolerance_ms(
            100.0
        )
        == pytest.approx(
            2.0
        )
    )


def test_candidate_qualifies_when_all_later_results_are_stable():
    results = [
        make_result(
            "BASELINE",
            7,
            0.0,
            0.10,
            5.0,
        ),
        make_result(
            "BASELINE",
            7,
            10.0,
            0.050,
            4.0,
        ),
        make_result(
            "BASELINE",
            7,
            25.0,
            0.048,
            4.05,
        ),
        make_result(
            "BASELINE",
            7,
            50.0,
            0.047,
            4.08,
        ),
    ]

    check = evaluate_drain_candidate(
        results,
        10.0,
    )

    assert check.qualifies is True
    assert (
        check.failure_reason
        is None
    )


def test_candidate_fails_when_residual_change_exceeds_tolerance():
    results = [
        make_result(
            "BASELINE",
            7,
            10.0,
            0.050,
            4.0,
        ),
        make_result(
            "BASELINE",
            7,
            25.0,
            0.0601,
            4.0,
        ),
    ]

    check = evaluate_drain_candidate(
        results,
        10.0,
    )

    assert check.qualifies is False

    assert (
        "residual_ratio"
        in check.failure_reason
    )


def test_candidate_fails_when_p99_change_exceeds_tolerance():
    results = [
        make_result(
            "BASELINE",
            7,
            10.0,
            0.05,
            4.0,
        ),
        make_result(
            "BASELINE",
            7,
            25.0,
            0.05,
            4.2,
        ),
    ]

    check = evaluate_drain_candidate(
        results,
        10.0,
    )

    assert check.qualifies is False

    assert (
        "completion_delay_p99_ms"
        in check.failure_reason
    )


def test_zero_completion_count_fails_closed():
    result_a = make_result(
        "BASELINE",
        7,
        10.0,
        0.05,
        4.0,
    )

    result_b = make_result(
        "BASELINE",
        7,
        25.0,
        0.05,
        4.0,
    )

    result_a[
        "urllc"
    ][
        "completion_delay_count"
    ] = 0

    check = evaluate_drain_candidate(
        [
            result_a,
            result_b,
        ],
        10.0,
    )

    assert check.qualifies is False

    assert (
        "completion_delay_count"
        in check.failure_reason
    )


def test_selects_smallest_stable_candidate():
    results = [
        make_result(
            "BASELINE",
            7,
            0.0,
            0.10,
            5.0,
        ),
        make_result(
            "BASELINE",
            7,
            10.0,
            0.050,
            4.0,
        ),
        make_result(
            "BASELINE",
            7,
            25.0,
            0.048,
            4.05,
        ),
        make_result(
            "BASELINE",
            7,
            50.0,
            0.047,
            4.08,
        ),
    ]

    selected = (
        select_smallest_stable_drain(
            results
        )
    )

    assert selected.qualifies is True

    assert (
        selected.candidate_drain_ms
        == pytest.approx(
            10.0
        )
    )


def test_largest_candidate_does_not_qualify_vacuously():
    results = [
        make_result(
            "BASELINE",
            7,
            0.0,
            0.10,
            5.0,
        ),
        make_result(
            "BASELINE",
            7,
            10.0,
            0.05,
            4.0,
        ),
    ]

    selected = (
        select_smallest_stable_drain(
            results
        )
    )

    assert (
        selected.qualifies
        is False
    )
