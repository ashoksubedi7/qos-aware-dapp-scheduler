import pytest

from config.experiment_config import (
    ExperimentConfig,
)

from UE import PacketFlow


def make_flow(
    experiment_config=None,
    deactivation_time=1.0,
):
    return PacketFlow(
        1,
        1200,
        1.2,
        "ue1",
        "DL",
        "eMBB",
        0,
        deactivation_time,
        "Constant",
        "Constant",
        experiment_config=(
            experiment_config
        ),
    )


def test_legacy_traffic_end_preserves_083_behavior():
    flow = make_flow(
        experiment_config=None,
    )

    assert (
        flow.getTrafficEndTime(
            1000.0
        )
        == pytest.approx(
            830.0
        )
    )


def test_assured_zero_drain_uses_full_duration():
    config = ExperimentConfig(
        drain_duration_ms=0.0,
    )

    flow = make_flow(
        experiment_config=config,
    )

    assert (
        flow.getTrafficEndTime(
            1000.0
        )
        == pytest.approx(
            1000.0
        )
    )


def test_assured_drain_stops_traffic_before_simulation_end():
    config = ExperimentConfig(
        drain_duration_ms=100.0,
    )

    flow = make_flow(
        experiment_config=config,
    )

    assert (
        flow.getTrafficEndTime(
            1000.0
        )
        == pytest.approx(
            900.0
        )
    )


def test_flow_deactivation_can_end_before_global_drain_boundary():
    config = ExperimentConfig(
        drain_duration_ms=100.0,
    )

    flow = make_flow(
        experiment_config=config,
        deactivation_time=0.5,
    )

    assert (
        flow.getTrafficEndTime(
            1000.0
        )
        == pytest.approx(
            500.0
        )
    )


def test_drain_duration_equal_to_simulation_is_rejected():
    config = ExperimentConfig(
        drain_duration_ms=1000.0,
    )

    flow = make_flow(
        experiment_config=config,
    )

    with pytest.raises(
        ValueError,
        match="smaller than tSim",
    ):
        flow.getTrafficEndTime(
            1000.0
        )


def test_drain_duration_greater_than_simulation_is_rejected():
    config = ExperimentConfig(
        drain_duration_ms=1001.0,
    )

    flow = make_flow(
        experiment_config=config,
    )

    with pytest.raises(
        ValueError,
        match="smaller than tSim",
    ):
        flow.getTrafficEndTime(
            1000.0
        )
def test_assured_active_measurement_boundary():
    config = ExperimentConfig(
        drain_duration_ms=100.0,
    )

    flow = make_flow(
        experiment_config=config,
    )

    assert (
        flow.getActiveMeasurementEndTime(
            1000.0
        )
        == pytest.approx(
            900.0
        )
    )


def test_early_flow_deactivation_does_not_change_active_boundary():
    config = ExperimentConfig(
        drain_duration_ms=100.0,
    )

    flow = make_flow(
        experiment_config=config,
        deactivation_time=0.5,
    )

    assert (
        flow.getTrafficEndTime(
            1000.0
        )
        == pytest.approx(
            500.0
        )
    )

    assert (
        flow.getActiveMeasurementEndTime(
            1000.0
        )
        == pytest.approx(
            900.0
        )
    )
def test_legacy_traffic_end_preserves_exact_083_formula():
    flow = make_flow(
        experiment_config=None,
        deactivation_time=0.5,
    )

    assert (
        flow.getTrafficEndTime(
            1000.0
        )
        == pytest.approx(
            415.0
        )
    )
