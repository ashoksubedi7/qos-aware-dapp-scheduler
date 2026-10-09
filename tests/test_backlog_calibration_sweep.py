import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.backlog_calibration_sweep import (
    build_calibration_analysis,
    sha256_file,
    verify_protocol_hash,
    write_analysis_artifacts,
)


def make_run(
    scenario,
    seed,
    offset=0,
):
    values = (
        0.0 + offset,
        100.0 + offset,
        200.0 + offset,
        300.0 + offset,
    )

    return SimpleNamespace(
        traffic_scenario=scenario,
        seed=seed,
        sample_count=4,
        embb_bytes=values,
        urllc_bytes=values,
        mmtc_bytes=values,
    )


def test_protocol_hash_verification(
    tmp_path,
):
    path = (
        tmp_path
        / "protocol.txt"
    )

    path.write_text(
        "fixed protocol\n",
        encoding="utf-8",
    )

    expected = sha256_file(
        path
    )

    assert (
        verify_protocol_hash(
            path,
            expected,
        )
        == expected
    )


def test_protocol_hash_mismatch_fails(
    tmp_path,
):
    path = (
        tmp_path
        / "protocol.txt"
    )

    path.write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="hash mismatch",
    ):
        verify_protocol_hash(
            path,
            "0" * 64,
        )


def test_complete_grid_analysis():
    runs = [
        make_run(
            "A",
            7,
            0,
        ),
        make_run(
            "A",
            17,
            1000,
        ),
        make_run(
            "B",
            7,
            2000,
        ),
        make_run(
            "B",
            17,
            3000,
        ),
    ]

    analysis = (
        build_calibration_analysis(
            runs=runs,
            scenarios=(
                "A",
                "B",
            ),
            seeds=(
                7,
                17,
            ),
            expected_samples=4,
        )
    )

    assert set(
        analysis["caps"]
    ) == {
        "eMBB",
        "URLLC",
        "mMTC",
    }

    assert len(
        analysis[
            "per_run_rows"
        ]
    ) == 12

    assert len(
        analysis[
            "pooled_rows"
        ]
    ) == 3

    assert analysis[
        "all_valid"
    ]


def test_incomplete_grid_rejected():
    runs = [
        make_run(
            "A",
            7,
        )
    ]

    with pytest.raises(
        ValueError,
        match="incomplete calibration grid",
    ):
        build_calibration_analysis(
            runs=runs,
            scenarios=(
                "A",
                "B",
            ),
            seeds=(
                7,
            ),
            expected_samples=4,
        )


def test_manifest_records_raw_hashes(
    tmp_path,
):
    output_dir = (
        tmp_path
        / "collection"
    )

    raw_dir = (
        output_dir
        / "raw"
    )

    raw_dir.mkdir(
        parents=True
    )

    raw_a = (
        raw_dir
        / "A_seed7.csv"
    )

    raw_a.write_text(
        "time_ms,embb_bytes,"
        "urllc_bytes,mmtc_bytes\n"
        "0,1,2,3\n",
        encoding="utf-8",
    )

    protocol = (
        tmp_path
        / "protocol.txt"
    )

    protocol.write_text(
        "protocol\n",
        encoding="utf-8",
    )

    analysis = (
        build_calibration_analysis(
            runs=[
                make_run(
                    "A",
                    7,
                )
            ],
            scenarios=(
                "A",
            ),
            seeds=(
                7,
            ),
            expected_samples=4,
        )
    )

    protocol_sha = (
        sha256_file(
            protocol
        )
    )

    manifest_path = (
        write_analysis_artifacts(
            analysis=analysis,
            output_dir=output_dir,
            protocol_path=protocol,
            protocol_sha256=(
                protocol_sha
            ),
            raw_files={
                (
                    "A",
                    7,
                ): raw_a,
            },
            scenarios=(
                "A",
            ),
            seeds=(
                7,
            ),
            active_duration_ms=4,
            sample_interval_ms=1,
        )
    )

    manifest = json.loads(
        Path(
            manifest_path
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "protocol_sha256"
        ]
        == protocol_sha
    )

    assert (
        manifest[
            "raw_files"
        ][0]["sha256"]
        == sha256_file(
            raw_a
        )
    )

    assert (
        manifest[
            "run_count"
        ]
        == 1
    )
