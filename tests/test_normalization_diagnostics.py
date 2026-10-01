import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.normalization_diagnostics import (
    FeatureClipStats,
    NormalizationDiagnostics,
)


def test_feature_stats_no_clipping():
    stats = FeatureClipStats()

    stats.record(
        raw_value=250,
        lower_bound=0,
        upper_bound=500,
    )

    assert stats.observations == 1
    assert stats.clipped_total == 0
    assert stats.clip_rate == 0.0


def test_feature_stats_high_clipping():
    stats = FeatureClipStats()

    stats.record(
        raw_value=700,
        lower_bound=0,
        upper_bound=500,
    )

    assert stats.clipped_high == 1
    assert stats.clip_rate == 1.0


def test_feature_stats_low_clipping():
    stats = FeatureClipStats()

    stats.record(
        raw_value=-5,
        lower_bound=0,
        upper_bound=500,
    )

    assert stats.clipped_low == 1


def test_backlog_diagnostics():
    diagnostics = NormalizationDiagnostics()

    diagnostics.record_backlog(
        600,
        500,
    )

    diagnostics.record_backlog(
        100,
        500,
    )

    assert (
        diagnostics.backlog.observations
        == 2
    )

    assert (
        diagnostics.backlog.clipped_high
        == 1
    )

    assert (
        diagnostics.backlog.clip_rate
        == 0.5
    )


def test_sinr_diagnostics():
    diagnostics = NormalizationDiagnostics()

    diagnostics.record_sinr(
        -3,
        0,
        35,
    )

    diagnostics.record_sinr(
        40,
        0,
        35,
    )

    assert diagnostics.sinr.clipped_low == 1
    assert diagnostics.sinr.clipped_high == 1


def test_urgency_above_deadline_is_detected():
    diagnostics = NormalizationDiagnostics()

    diagnostics.record_urgency(
        1.7
    )

    assert (
        diagnostics.urgency.clipped_high
        == 1
    )


def test_to_dict_contains_clip_rate():
    diagnostics = NormalizationDiagnostics()

    diagnostics.record_backlog(
        700,
        500,
    )

    result = diagnostics.to_dict()

    assert (
        result["backlog"]["clip_rate"]
        == 1.0
    )
