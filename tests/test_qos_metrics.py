import pytest

from metrics.qos_metrics import (
    active_throughput_mbps,
    deadline_summary,
    delay_summary,
    jains_fairness_index,
    lifecycle_ratios,
    radio_prb_utilization,
)


def test_lifecycle_ratios_conserve_packets():
    result = lifecycle_ratios(
        generated=100,
        delivered=80,
        dropped=5,
        residual=15,
    )

    assert result.conservation_ok
    assert result.completion_ratio == pytest.approx(
        0.80
    )
    assert result.true_drop_ratio == pytest.approx(
        0.05
    )
    assert result.residual_ratio == pytest.approx(
        0.15
    )


def test_lifecycle_detects_nonconservation():
    result = lifecycle_ratios(
        generated=100,
        delivered=80,
        dropped=5,
        residual=10,
    )

    assert not result.conservation_ok


def test_lifecycle_rejects_negative_counts():
    with pytest.raises(ValueError):
        lifecycle_ratios(
            generated=10,
            delivered=9,
            dropped=-1,
            residual=2,
        )


def test_empty_lifecycle_ratios_are_zero():
    result = lifecycle_ratios(
        generated=0,
        delivered=0,
        dropped=0,
        residual=0,
    )

    assert result.completion_ratio == 0.0
    assert result.true_drop_ratio == 0.0
    assert result.residual_ratio == 0.0
    assert result.conservation_ok


def test_delay_summary():
    result = delay_summary(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    assert result.count == 4
    assert result.mean_ms == pytest.approx(
        2.5
    )
    assert result.max_ms == pytest.approx(
        4.0
    )


def test_empty_delay_summary():
    result = delay_summary([])

    assert result.count == 0
    assert result.mean_ms == 0.0
    assert result.p99_ms == 0.0


def test_delay_summary_rejects_negative_values():
    with pytest.raises(ValueError):
        delay_summary(
            [
                1.0,
                -0.1,
            ]
        )


def test_deadline_summary():
    result = deadline_summary(
        evaluated=100,
        misses=7,
    )

    assert result.miss_ratio == pytest.approx(
        0.07
    )


def test_deadline_summary_rejects_more_misses_than_evaluated():
    with pytest.raises(ValueError):
        deadline_summary(
            evaluated=5,
            misses=6,
        )


def test_active_throughput_uses_si_mbps():
    value = active_throughput_mbps(
        delivered_bytes=125_000,
        active_duration_ms=1000.0,
    )

    assert value == pytest.approx(
        1.0
    )


def test_prb_utilization():
    value = radio_prb_utilization(
        used_prbs=75,
        available_prbs=100,
    )

    assert value == pytest.approx(
        0.75
    )


def test_prb_utilization_rejects_impossible_usage():
    with pytest.raises(ValueError):
        radio_prb_utilization(
            used_prbs=101,
            available_prbs=100,
        )


def test_jains_fairness_equal_service():
    value = jains_fairness_index(
        [
            10,
            10,
            10,
        ]
    )

    assert value == pytest.approx(
        1.0
    )


def test_jains_fairness_unequal_service():
    value = jains_fairness_index(
        [
            10,
            0,
        ]
    )

    assert value == pytest.approx(
        0.5
    )


def test_jains_fairness_zero_service_is_undefined():
    assert (
        jains_fairness_index(
            [
                0,
                0,
                0,
            ]
        )
        is None
    )
