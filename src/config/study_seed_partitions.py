from dataclasses import dataclass


# These seeds have already been used by the
# calibration methodology. They are reserved from
# the prospective scientific-study namespaces.
CALIBRATION_SEEDS = (
    7,
    17,
    27,
)


@dataclass(frozen=True)
class StudySeedPartitions:
    """
    Prospective AssuredQoS seed namespaces.

    Numerical seed membership is deliberately unset
    until the study-partition protocol is frozen.

    The four namespaces below have different roles:

      pilot:
        select training-budget / epsilon /
        checkpoint-interval methodology.

      training:
        independent stochastic training runs.

      checkpoint_validation:
        frozen-policy checkpoint selection only.

      final_test:
        held out until all model-selection and
        training decisions are complete.
    """

    pilot_seeds: tuple[int, ...] = ()

    training_seeds: tuple[int, ...] = ()

    checkpoint_validation_seeds: tuple[
        int,
        ...
    ] = ()

    final_test_seeds: tuple[int, ...] = ()

    def __post_init__(self):
        namespaces = self.as_dict()

        for name, seeds in namespaces.items():
            if not isinstance(
                seeds,
                tuple,
            ):
                raise TypeError(
                    f"{name} must be a tuple"
                )

            if any(
                not isinstance(seed, int)
                or isinstance(seed, bool)
                for seed in seeds
            ):
                raise TypeError(
                    f"{name} must contain "
                    "integer seeds"
                )

            if any(
                seed < 0
                for seed in seeds
            ):
                raise ValueError(
                    f"{name} contains a "
                    "negative seed"
                )

            if len(set(seeds)) != len(seeds):
                raise ValueError(
                    f"{name} contains duplicate "
                    "seeds"
                )

            calibration_overlap = (
                set(seeds)
                &
                set(CALIBRATION_SEEDS)
            )

            if calibration_overlap:
                raise ValueError(
                    f"{name} reuses reserved "
                    "calibration seeds: "
                    f"{sorted(calibration_overlap)}"
                )

        names = tuple(
            namespaces
        )

        for index, first_name in enumerate(
            names
        ):
            first = set(
                namespaces[first_name]
            )

            for second_name in names[
                index + 1:
            ]:
                overlap = (
                    first
                    &
                    set(
                        namespaces[
                            second_name
                        ]
                    )
                )

                if overlap:
                    raise ValueError(
                        "scientific seed namespaces "
                        "must be disjoint: "
                        f"{first_name} and "
                        f"{second_name} share "
                        f"{sorted(overlap)}"
                    )

    def as_dict(self):
        return {
            "pilot_seeds":
                self.pilot_seeds,
            "training_seeds":
                self.training_seeds,
            "checkpoint_validation_seeds":
                self.checkpoint_validation_seeds,
            "final_test_seeds":
                self.final_test_seeds,
        }

    def unresolved_items(self):
        unresolved = []

        for name, seeds in (
            self.as_dict().items()
        ):
            if not seeds:
                unresolved.append(
                    name
                )

        return tuple(
            unresolved
        )

    def assert_ready(self):
        unresolved = (
            self.unresolved_items()
        )

        if unresolved:
            raise RuntimeError(
                "study seed partitions are "
                "not frozen; unresolved: "
                + ", ".join(
                    unresolved
                )
            )

        return True
