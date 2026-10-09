from dataclasses import dataclass
import math

import numpy as np


KIB = 1024
TAIL_QUANTILE = 0.999
MAX_ALLOWED_CLIP_FRACTION = 0.001


@dataclass(frozen=True)
class BacklogDistributionSummary:
    count: int
    minimum: float
    median: float
    p90: float
    p95: float
    p99: float
    p99_9: float
    maximum: float


@dataclass(frozen=True)
class BacklogCapValidation:
    count: int
    cap_bytes: int
    above_cap: int
    fraction_above_cap: float
    passes: bool


def _as_valid_samples(values):
    samples = np.asarray(
        tuple(values),
        dtype=np.float64,
    )

    if samples.ndim != 1:
        raise ValueError(
            "backlog samples must be one-dimensional"
        )

    if samples.size == 0:
        raise ValueError(
            "backlog samples cannot be empty"
        )

    if not np.all(
        np.isfinite(samples)
    ):
        raise ValueError(
            "backlog samples must be finite"
        )

    if np.any(
        samples < 0
    ):
        raise ValueError(
            "backlog samples cannot be negative"
        )

    return samples


def summarize_backlog_samples(values):
    samples = _as_valid_samples(
        values
    )

    # p99.9 is the statistic used for cap
    # selection by the preregistered protocol.
    p99_9 = np.quantile(
        samples,
        TAIL_QUANTILE,
        method="higher",
    )

    return BacklogDistributionSummary(
        count=int(samples.size),
        minimum=float(
            np.min(samples)
        ),
        median=float(
            np.quantile(
                samples,
                0.50,
            )
        ),
        p90=float(
            np.quantile(
                samples,
                0.90,
            )
        ),
        p95=float(
            np.quantile(
                samples,
                0.95,
            )
        ),
        p99=float(
            np.quantile(
                samples,
                0.99,
            )
        ),
        p99_9=float(
            p99_9
        ),
        maximum=float(
            np.max(samples)
        ),
    )


def select_backlog_cap(
    run_summaries,
):
    summaries = tuple(
        run_summaries
    )

    if not summaries:
        raise ValueError(
            "at least one run summary is required"
        )

    run_q999_max = max(
        float(summary.p99_9)
        for summary in summaries
    )

    cap_bytes = max(
        KIB,
        int(
            math.ceil(
                run_q999_max
                / KIB
            )
            * KIB
        ),
    )

    return cap_bytes


def validate_backlog_cap(
    values,
    cap_bytes,
):
    samples = _as_valid_samples(
        values
    )

    cap_bytes = int(
        cap_bytes
    )

    if cap_bytes <= 0:
        raise ValueError(
            "cap_bytes must be positive"
        )

    above_cap = int(
        np.count_nonzero(
            samples
            > float(cap_bytes)
        )
    )

    fraction = (
        above_cap
        / int(samples.size)
    )

    return BacklogCapValidation(
        count=int(samples.size),
        cap_bytes=cap_bytes,
        above_cap=above_cap,
        fraction_above_cap=float(
            fraction
        ),
        passes=(
            fraction
            <= MAX_ALLOWED_CLIP_FRACTION
        ),
    )
