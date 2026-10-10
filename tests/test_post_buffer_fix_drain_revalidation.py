import pytest


from experiments.post_buffer_fix_drain_revalidation import (
    EXPECTED_RUN_COUNT,
    POST_FIX_ACTIVE_DURATION_MS,
    POST_FIX_DRAIN_CANDIDATES_MS,
    POST_FIX_SEEDS,
    POST_FIX_TRAFFIC_SCENARIOS,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    SIMULATOR_SEMANTICS_ANCHOR,
    classify_revalidation,
    expected_run_keys,
    validate_complete_grid,
)


def make_slice(
    residual_ratio,
    p99,
):
    return {
        "residual_ratio": residual_ratio,
        "completion_delay_count": 100,
        "completion_delay_p99_ms": p99,
    }


def make_complete_results(
    residual_by_drain,
    p99_by_drain=None,
):
    if p99_by_drain is None:
        p99_by_drain = {
            500.0: 5.0,
            1000.0: 5.0,
            2000.0: 5.0,
        }

    results = []

    for scenario in POST_FIX_TRAFFIC_SCENARIOS:
        for seed in POST_FIX_SEEDS:
            for drain in POST_FIX_DRAIN_CANDIDATES_MS:
                summary = make_slice(
                    residual_ratio=(
                        residual_by_drain[
                            drain
                        ]
                    ),
                    p99=(
                        p99_by_drain[
                            drain
                        ]
                    ),
                )

                results.append(
                    {
                        "traffic_scenario": scenario,
                        "seed": seed,
                        "drain_duration_ms": drain,
                        "embb": dict(summary),
                        "urllc": dict(summary),
                        "mmtc": dict(summary),
                    }
                )

    return results


def test_frozen_post_fix_matrix():
    assert POST_FIX_DRAIN_CANDIDATES_MS == (
        500.0,
        1000.0,
        2000.0,
    )

    assert POST_FIX_SEEDS == (
        7,
        17,
        27,
    )

    assert POST_FIX_TRAFFIC_SCENARIOS == (
        "BASELINE",
        "CONGESTED",
        "URLLC_HIGH",
        "SIMULTANEOUS_HIGH",
    )

    assert POST_FIX_ACTIVE_DURATION_MS == (
        10_000.0
    )

    assert EXPECTED_RUN_COUNT == 36

    assert len(
        expected_run_keys()
    ) == 36


def test_frozen_provenance_constants():
    assert SIMULATOR_SEMANTICS_ANCHOR == (
        "84b731a85732d10dbb30f5374b2e11c3715d86e4"
    )

    assert PROTOCOL_COMMIT == (
        "cbb030d81b19dd69e3b1b4dad988e637f330880b"
    )

    assert PROTOCOL_SHA256 == (
        "d2043cb7cae3536b4873fa3712e478db"
        "5debbca1e660374d82f17adc326a161d"
    )


def test_complete_grid_validation_passes():
    results = make_complete_results(
        {
            500.0: 0.01,
            1000.0: 0.01,
            2000.0: 0.01,
        }
    )

    validate_complete_grid(
        results
    )


def test_incomplete_grid_fails_closed():
    results = make_complete_results(
        {
            500.0: 0.01,
            1000.0: 0.01,
            2000.0: 0.01,
        }
    )

    results.pop()

    with pytest.raises(
        ValueError,
        match="requires 36 results",
    ):
        validate_complete_grid(
            results
        )


def test_duplicate_grid_entry_fails_closed():
    results = make_complete_results(
        {
            500.0: 0.01,
            1000.0: 0.01,
            2000.0: 0.01,
        }
    )

    results[-1] = dict(
        results[0]
    )

    with pytest.raises(
        ValueError
    ):
        validate_complete_grid(
            results
        )


def test_case_a_retains_1000_ms():
    results = make_complete_results(
        {
            500.0: 0.020,
            1000.0: 0.001,
            2000.0: 0.001,
        }
    )

    decision = classify_revalidation(
        results
    )

    assert decision.case == "A"
    assert (
        decision.check_500_qualifies
        is False
    )
    assert (
        decision.check_1000_qualifies
        is True
    )
    assert (
        decision.retain_1000_ms
        is True
    )
    assert (
        decision.reopen_search
        is False
    )


def test_case_b_reopens_lower_search():
    results = make_complete_results(
        {
            500.0: 0.001,
            1000.0: 0.001,
            2000.0: 0.001,
        }
    )

    decision = classify_revalidation(
        results
    )

    assert decision.case == "B"
    assert (
        decision.check_500_qualifies
        is True
    )
    assert (
        decision.check_1000_qualifies
        is True
    )
    assert (
        decision.retain_1000_ms
        is False
    )
    assert (
        decision.reopen_search
        is True
    )


def test_case_c_reopens_expanded_search():
    results = make_complete_results(
        {
            500.0: 0.020,
            1000.0: 0.020,
            2000.0: 0.001,
        }
    )

    decision = classify_revalidation(
        results
    )

    assert decision.case == "C"
    assert (
        decision.check_1000_qualifies
        is False
    )
    assert (
        decision.retain_1000_ms
        is False
    )
    assert (
        decision.reopen_search
        is True
    )
