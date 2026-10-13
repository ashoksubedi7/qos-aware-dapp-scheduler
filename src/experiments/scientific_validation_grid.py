"""Validate completeness against an explicitly declared evaluation grid."""


def _validate_axis(values, name, expected_type):
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be a collection, not a string"
        )

    values = tuple(values)

    if not values:
        raise ValueError(
            f"empty declared {name}"
        )

    for value in values:
        if type(value) is not expected_type:
            raise TypeError(
                f"{name} contains an invalid value type"
            )

        if expected_type is str and not value.strip():
            raise ValueError(
                f"{name} contains an empty name"
            )

        if expected_type is int and value < 0:
            raise ValueError(
                f"{name} contains a negative seed"
            )

    if len(set(values)) != len(values):
        raise ValueError(
            f"duplicate declared {name}"
        )

    return values


def validate_declared_validation_grid(
    summaries,
    *,
    traffic_scenarios,
    radio_scenarios,
    seeds,
):
    """
    Require exactly one result for every declared
    traffic x radio x seed combination.

    This verifies grid identity and completeness only.
    Scientific metric eligibility is checked separately
    by the existing checkpoint aggregation module.
    """
    traffic_scenarios = _validate_axis(
        traffic_scenarios,
        "traffic_scenarios",
        str,
    )

    radio_scenarios = _validate_axis(
        radio_scenarios,
        "radio_scenarios",
        str,
    )

    seeds = _validate_axis(
        seeds,
        "seeds",
        int,
    )

    summaries = tuple(summaries)

    expected = {
        (traffic, radio, seed)
        for traffic in traffic_scenarios
        for radio in radio_scenarios
        for seed in seeds
    }

    observed = set()
    model_variants = set()

    for summary in summaries:
        key = (
            summary.traffic_scenario,
            summary.radio_scenario,
            summary.seed,
        )

        if (
            type(key[0]) is not str
            or type(key[1]) is not str
            or type(key[2]) is not int
        ):
            raise ValueError(
                "validation result has invalid grid identity"
            )

        if key in observed:
            raise ValueError(
                f"duplicate validation result: {key}"
            )

        observed.add(key)
        model_variants.add(summary.model_variant)

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    if missing or unexpected:
        raise ValueError(
            "declared validation grid mismatch; "
            f"missing={missing}; "
            f"unexpected={unexpected}"
        )

    if len(model_variants) != 1:
        raise ValueError(
            "validation results must belong "
            "to one model variant"
        )

    conditions = {
        (traffic, radio)
        for traffic in traffic_scenarios
        for radio in radio_scenarios
    }

    return {
        "run_count": len(summaries),
        "conditions": tuple(sorted(conditions)),
        "seeds": tuple(sorted(seeds)),
        "model_variant": next(iter(model_variants)),
    }
