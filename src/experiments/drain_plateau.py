from dataclasses import dataclass


RESIDUAL_RATIO_TOLERANCE = 0.005
P99_ABSOLUTE_TOLERANCE_MS = 0.1
P99_RELATIVE_TOLERANCE = 0.02

SLICE_NAMES = (
    "embb",
    "urllc",
    "mmtc",
)


@dataclass(frozen=True)
class PlateauCheck:
    candidate_drain_ms: float
    qualifies: bool
    failure_reason: str | None


def p99_tolerance_ms(
    reference_p99_ms,
):
    reference = float(
        reference_p99_ms
    )

    if reference < 0:
        raise ValueError(
            "reference p99 must be nonnegative"
        )

    return max(
        P99_ABSOLUTE_TOLERANCE_MS,
        P99_RELATIVE_TOLERANCE
        * reference,
    )


def _slice_is_stable(
    reference_slice,
    later_slice,
):
    residual_change = abs(
        float(
            later_slice[
                "residual_ratio"
            ]
        )
        - float(
            reference_slice[
                "residual_ratio"
            ]
        )
    )

    if (
        residual_change
        > RESIDUAL_RATIO_TOLERANCE
    ):
        return (
            False,
            "residual_ratio",
        )

    reference_count = int(
        reference_slice[
            "completion_delay_count"
        ]
    )

    later_count = int(
        later_slice[
            "completion_delay_count"
        ]
    )

    if (
        reference_count <= 0
        or later_count <= 0
    ):
        return (
            False,
            "completion_delay_count",
        )

    reference_p99 = float(
        reference_slice[
            "completion_delay_p99_ms"
        ]
    )

    later_p99 = float(
        later_slice[
            "completion_delay_p99_ms"
        ]
    )

    tolerance = p99_tolerance_ms(
        reference_p99
    )

    if (
        abs(
            later_p99
            - reference_p99
        )
        > tolerance
    ):
        return (
            False,
            "completion_delay_p99_ms",
        )

    return (
        True,
        None,
    )


def _index_results(
    results,
):
    index = {}

    for result in results:
        key = (
            str(
                result[
                    "traffic_scenario"
                ]
            ),
            int(
                result[
                    "seed"
                ]
            ),
            float(
                result[
                    "drain_duration_ms"
                ]
            ),
        )

        if key in index:
            raise ValueError(
                "duplicate drain calibration result "
                f"for {key}"
            )

        index[
            key
        ] = result

    return index


def evaluate_drain_candidate(
    results,
    candidate_drain_ms,
):
    index = _index_results(
        results
    )

    candidate = float(
        candidate_drain_ms
    )

    scenarios = sorted(
        {
            key[0]
            for key in index
        }
    )

    seeds = sorted(
        {
            key[1]
            for key in index
        }
    )

    drains = sorted(
        {
            key[2]
            for key in index
        }
    )

    if candidate not in drains:
        raise ValueError(
            "candidate drain is not present "
            "in calibration results"
        )

    later_drains = [
        drain
        for drain in drains
        if drain > candidate
    ]

    for scenario in scenarios:
        for seed in seeds:
            reference_key = (
                scenario,
                seed,
                candidate,
            )

            if reference_key not in index:
                return PlateauCheck(
                    candidate_drain_ms=candidate,
                    qualifies=False,
                    failure_reason=(
                        "missing reference result: "
                        f"{reference_key}"
                    ),
                )

            reference = index[
                reference_key
            ]

            for later_drain in later_drains:
                later_key = (
                    scenario,
                    seed,
                    later_drain,
                )

                if later_key not in index:
                    return PlateauCheck(
                        candidate_drain_ms=candidate,
                        qualifies=False,
                        failure_reason=(
                            "missing later result: "
                            f"{later_key}"
                        ),
                    )

                later = index[
                    later_key
                ]

                for slice_name in SLICE_NAMES:
                    stable, metric = (
                        _slice_is_stable(
                            reference[
                                slice_name
                            ],
                            later[
                                slice_name
                            ],
                        )
                    )

                    if not stable:
                        return PlateauCheck(
                            candidate_drain_ms=candidate,
                            qualifies=False,
                            failure_reason=(
                                f"{scenario}/"
                                f"seed={seed}/"
                                f"{slice_name}/"
                                f"{metric}/"
                                f"{candidate}"
                                "->"
                                f"{later_drain}"
                            ),
                        )

    return PlateauCheck(
        candidate_drain_ms=candidate,
        qualifies=True,
        failure_reason=None,
    )


def select_smallest_stable_drain(
    results,
):
    if not results:
        raise ValueError(
            "drain calibration results are empty"
        )

    drains = sorted(
        {
            float(
                result[
                    "drain_duration_ms"
                ]
            )
            for result in results
        }
    )

    # The largest candidate cannot demonstrate
    # stability relative to any larger tested drain.
    for candidate in drains[
        :-1
    ]:
        check = (
            evaluate_drain_candidate(
                results,
                candidate,
            )
        )

        if check.qualifies:
            return check

    return PlateauCheck(
        candidate_drain_ms=float(
            drains[-1]
        ),
        qualifies=False,
        failure_reason=(
            "no tested drain demonstrated "
            "stability against all larger "
            "candidates"
        ),
    )

