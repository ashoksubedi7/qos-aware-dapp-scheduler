import csv

import numpy as np
import pytest

from experiments.backlog_calibration import (
    BacklogDistributionSummary,
    select_backlog_cap,
    summarize_backlog_samples,
    validate_backlog_cap,
)

from backlog_calibration_runner import (
    BacklogCalibrationRun,
    write_backlog_run_csv,
)


def test_p999_uses_higher_quantile():
    values = np.arange(
        1000,
        dtype=float,
    )

    summary = (
        summarize_backlog_samples(
            values
        )
    )

    expected = np.quantile(
        values,
        0.999,
        method="higher",
    )

    assert (
        summary.p99_9
        == expected
    )


def test_select_cap_uses_worst_run_q999():
    summaries = [
        BacklogDistributionSummary(
            count=10000,
            minimum=0,
            median=100,
            p90=200,
            p95=300,
            p99=800,
            p99_9=1500,
            maximum=2000,
        ),
        BacklogDistributionSummary(
            count=10000,
            minimum=0,
            median=100,
            p90=200,
            p95=300,
            p99=900,
            p99_9=2500,
            maximum=4000,
        ),
    ]

    assert (
        select_backlog_cap(
            summaries
        )
        == 3072
    )


def test_select_cap_has_one_kib_minimum():
    summary = BacklogDistributionSummary(
        count=10,
        minimum=0,
        median=0,
        p90=0,
        p95=0,
        p99=0,
        p99_9=0,
        maximum=0,
    )

    assert (
        select_backlog_cap(
            [summary]
        )
        == 1024
    )


def test_cap_validation_uses_strictly_greater():
    values = [
        1024,
        1024,
        2048,
    ]

    validation = (
        validate_backlog_cap(
            values,
            cap_bytes=1024,
        )
    )

    assert (
        validation.above_cap
        == 1
    )

    assert validation.fraction_above_cap == (
        pytest.approx(
            1 / 3
        )
    )

    assert not validation.passes


def test_cap_validation_accepts_exact_protocol_limit():
    values = (
        [0.0] * 9990
        + [2048.0] * 10
    )

    validation = (
        validate_backlog_cap(
            values,
            cap_bytes=1024,
        )
    )

    assert (
        validation.above_cap
        == 10
    )

    assert (
        validation.fraction_above_cap
        == pytest.approx(
            0.001
        )
    )

    assert validation.passes


def test_raw_csv_writer(tmp_path):
    run = BacklogCalibrationRun(
        seed=7,
        traffic_scenario="BASELINE",
        active_duration_ms=2.0,
        sample_interval_ms=1.0,
        times_ms=(0.0, 1.0),
        embb_bytes=(100.0, 200.0),
        urllc_bytes=(300.0, 400.0),
        mmtc_bytes=(500.0, 600.0),
    )

    output = (
        write_backlog_run_csv(
            run,
            tmp_path
            / "run.csv",
        )
    )

    with output.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.reader(handle)
        )

    assert rows == [
        [
            "time_ms",
            "embb_bytes",
            "urllc_bytes",
            "mmtc_bytes",
        ],
        [
            "0.0",
            "100.0",
            "300.0",
            "500.0",
        ],
        [
            "1.0",
            "200.0",
            "400.0",
            "600.0",
        ],
    ]
