from dataclasses import dataclass


from experiments.drain_plateau import (
    evaluate_drain_candidate,
)


POST_FIX_DRAIN_CANDIDATES_MS = (
    500.0,
    1000.0,
    2000.0,
)

POST_FIX_SEEDS = (
    7,
    17,
    27,
)

POST_FIX_TRAFFIC_SCENARIOS = (
    "BASELINE",
    "CONGESTED",
    "URLLC_HIGH",
    "SIMULTANEOUS_HIGH",
)

POST_FIX_ACTIVE_DURATION_MS = 10_000.0

EXPECTED_RUN_COUNT = (
    len(POST_FIX_DRAIN_CANDIDATES_MS)
    * len(POST_FIX_SEEDS)
    * len(POST_FIX_TRAFFIC_SCENARIOS)
)

SIMULATOR_SEMANTICS_ANCHOR = (
    "84b731a85732d10dbb30f5374b2e11c3715d86e4"
)

PROTOCOL_COMMIT = (
    "cbb030d81b19dd69e3b1b4dad988e637f330880b"
)

PROTOCOL_SHA256 = (
    "d2043cb7cae3536b4873fa3712e478db"
    "5debbca1e660374d82f17adc326a161d"
)


@dataclass(frozen=True)
class RevalidationDecision:
    case: str
    retain_1000_ms: bool
    reopen_search: bool
    check_500_qualifies: bool
    check_500_failure_reason: str | None
    check_1000_qualifies: bool
    check_1000_failure_reason: str | None


def expected_run_keys():
    return {
        (
            scenario,
            int(seed),
            float(drain_ms),
        )
        for scenario
        in POST_FIX_TRAFFIC_SCENARIOS
        for seed
        in POST_FIX_SEEDS
        for drain_ms
        in POST_FIX_DRAIN_CANDIDATES_MS
    }


def _result_key(result):
    return (
        str(
            result["traffic_scenario"]
        ),
        int(
            result["seed"]
        ),
        float(
            result["drain_duration_ms"]
        ),
    )


def validate_complete_grid(results):
    if len(results) != EXPECTED_RUN_COUNT:
        raise ValueError(
            "post-fix drain revalidation requires "
            f"{EXPECTED_RUN_COUNT} results; "
            f"found {len(results)}"
        )

    actual_keys = [
        _result_key(result)
        for result
        in results
    ]

    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError(
            "duplicate post-fix drain "
            "revalidation result"
        )

    expected = expected_run_keys()
    actual = set(actual_keys)

    if actual != expected:
        missing = sorted(
            expected - actual
        )
        unexpected = sorted(
            actual - expected
        )

        raise ValueError(
            "post-fix drain revalidation "
            "grid mismatch; "
            f"missing={missing}; "
            f"unexpected={unexpected}"
        )


def classify_revalidation(results):
    validate_complete_grid(
        results
    )

    check_500 = evaluate_drain_candidate(
        results,
        500.0,
    )

    check_1000 = evaluate_drain_candidate(
        results,
        1000.0,
    )

    if check_1000.qualifies:
        if check_500.qualifies:
            case = "B"
            retain_1000 = False
            reopen_search = True
        else:
            case = "A"
            retain_1000 = True
            reopen_search = False
    else:
        case = "C"
        retain_1000 = False
        reopen_search = True

    return RevalidationDecision(
        case=case,
        retain_1000_ms=retain_1000,
        reopen_search=reopen_search,
        check_500_qualifies=(
            check_500.qualifies
        ),
        check_500_failure_reason=(
            check_500.failure_reason
        ),
        check_1000_qualifies=(
            check_1000.qualifies
        ),
        check_1000_failure_reason=(
            check_1000.failure_reason
        ),
    )
