from types import SimpleNamespace

import pytest

from drain_calibration_runner import (
    DrainSliceSummary,
)
from drain_calibration_runner import (
    run_drain_calibration_case,
)


def test_drain_calibration_smoke_run():
    result = run_drain_calibration_case(
        active_duration_ms=20.0,
        drain_duration_ms=5.0,
        seed=7,
    )

    assert (
        result.active_duration_ms
        == pytest.approx(
            20.0
        )
    )

    assert (
        result.drain_duration_ms
        == pytest.approx(
            5.0
        )
    )

    assert (
        result.total_duration_ms
        == pytest.approx(
            25.0
        )
    )

    for summary in (
        result.embb,
        result.urllc,
        result.mmtc,
    ):
        assert (
            summary.generated
            ==
            summary.delivered
            + summary.dropped
            + summary.residual
        )

        assert (
            0.0
            <= summary.residual_ratio
            <= 1.0
        )
def test_drain_calibration_accepts_nonbaseline_scenario():
    result = run_drain_calibration_case(
        active_duration_ms=20.0,
        drain_duration_ms=5.0,
        seed=7,
        traffic_scenario="URLLC_HIGH",
    )

    assert (
        result.traffic_scenario
        == "URLLC_HIGH"
    )
