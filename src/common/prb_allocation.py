import math


def percentage_action_to_prbs(
    total_prbs,
    weights,
):
    """
    Convert slice percentage weights to integer PRBs while preserving
    the complete PRB budget.

    The largest-remainder method is used so integer rounding does not
    create or destroy radio resources.

    The all-zero action intentionally allocates no PRBs.
    """
    total_prbs = int(total_prbs)

    if total_prbs < 0:
        raise ValueError(
            "total_prbs must be non-negative"
        )

    weights = tuple(
        float(weight)
        for weight in weights
    )

    if len(weights) != 3:
        raise ValueError(
            "expected exactly three slice weights"
        )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            "weights must be non-negative"
        )

    weight_sum = sum(weights)

    if weight_sum == 0:
        return (0, 0, 0)

    if not math.isclose(
        weight_sum,
        100.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "non-idle action weights must sum to 100"
        )

    exact = [
        total_prbs * weight / 100.0
        for weight in weights
    ]

    allocation = [
        math.floor(value)
        for value in exact
    ]

    remaining = (
        total_prbs
        - sum(allocation)
    )

    remainders = [
        exact[index] - allocation[index]
        for index in range(3)
    ]

    order = sorted(
        range(3),
        key=lambda index: (
            -remainders[index],
            index,
        ),
    )

    for index in order[:remaining]:
        allocation[index] += 1

    return tuple(allocation)
from functools import lru_cache


@lru_cache(maxsize=None)
def _numerology_allocation_cached(
    total_reference_prbs,
    weights,
    factors,
):
    targets = tuple(
        total_reference_prbs * weight / 100.0
        for weight in weights
    )

    best_allocation = None
    best_score = None

    f0, f1, f2 = factors

    for n0 in range(total_reference_prbs // f0 + 1):
        used0 = n0 * f0

        for n1 in range(
            (total_reference_prbs - used0) // f1 + 1
        ):
            used01 = used0 + n1 * f1

            n2 = (
                total_reference_prbs - used01
            ) // f2

            allocation = (
                n0,
                n1,
                n2,
            )

            actual_reference = (
                n0 * f0,
                n1 * f1,
                n2 * f2,
            )

            used = sum(actual_reference)

            errors = tuple(
                actual_reference[i] - targets[i]
                for i in range(3)
            )

            squared_error = sum(
                error * error
                for error in errors
            )

            max_error = max(
                abs(error)
                for error in errors
            )

            # Primary objective:
            # use as much common radio capacity as possible.
            #
            # Secondary objective:
            # approximate the requested percentage split.
            score = (
                -used,
                squared_error,
                max_error,
                allocation,
            )

            if (
                best_score is None
                or score < best_score
            ):
                best_score = score
                best_allocation = allocation

    return best_allocation


def percentage_action_to_numerology_prbs(
    total_reference_prbs,
    weights,
    num_ref_factors,
):
    """
    Map a percentage action to integer slice PRBs while accounting
    for slice numerology.

    num_ref_factors express each slice PRB in common reference-resource
    units. For FR1 in this simulator:

        15 kHz -> 1
        30 kHz -> 2
        60 kHz -> 4
    """
    total_reference_prbs = int(
        total_reference_prbs
    )

    if total_reference_prbs < 0:
        raise ValueError(
            "total_reference_prbs must be non-negative"
        )

    weights = tuple(
        float(value)
        for value in weights
    )

    factors = tuple(
        int(value)
        for value in num_ref_factors
    )

    if len(weights) != 3:
        raise ValueError(
            "expected exactly three slice weights"
        )

    if len(factors) != 3:
        raise ValueError(
            "expected exactly three numerology factors"
        )

    if any(value <= 0 for value in factors):
        raise ValueError(
            "numerology factors must be positive"
        )

    if any(value < 0 for value in weights):
        raise ValueError(
            "weights must be non-negative"
        )

    weight_sum = sum(weights)

    if weight_sum == 0:
        return (0, 0, 0)

    if abs(weight_sum - 100.0) > 1e-9:
        raise ValueError(
            "non-idle action weights must sum to 100"
        )

    return _numerology_allocation_cached(
        total_reference_prbs,
        weights,
        factors,
    )


def reference_prbs_used(
    allocation,
    num_ref_factors,
):
    return sum(
        int(allocation[i])
        * int(num_ref_factors[i])
        for i in range(3)
    )
