from types import SimpleNamespace

import pytest


from UE import UE

from run_post_buffer_fix_drain_revalidation import (
    EXPECTED_BUFFER_BYTES,
    bearer_capacity_guard,
    finalize_capacity_evidence,
    prepare_output_directory,
)


def make_fake_ue():
    return SimpleNamespace(
        id="ue-test",
        packetFlows=[
            SimpleNamespace(
                sliceName="eMBB"
            )
        ],
        bearers=[
            SimpleNamespace(
                buffer=SimpleNamespace(
                    pckts=[]
                )
            )
        ],
    )


def test_capacity_guard_records_valid_admission(
    monkeypatch,
):
    def fake_queue(
        self,
        cell,
    ):
        self.bearers[
            0
        ].buffer.pckts.append(
            SimpleNamespace(
                size=81_919
            )
        )

    monkeypatch.setattr(
        UE,
        "queueDataPckt",
        fake_queue,
    )

    ue = make_fake_ue()

    cell = SimpleNamespace(
        maxBuffUE=(
            EXPECTED_BUFFER_BYTES
        )
    )

    with (
        bearer_capacity_guard()
        as monitor
    ):
        UE.queueDataPckt(
            ue,
            cell,
        )

    evidence = (
        finalize_capacity_evidence(
            monitor
        )
    )

    assert (
        evidence["passes"]
        is True
    )

    assert (
        evidence[
            "admission_calls"
        ]
        == 1
    )

    assert (
        evidence[
            "max_per_slice_bytes"
        ]["eMBB"]
        == 81_919
    )


def test_capacity_guard_fails_on_overflow(
    monkeypatch,
):
    def fake_queue(
        self,
        cell,
    ):
        self.bearers[
            0
        ].buffer.pckts.append(
            SimpleNamespace(
                size=81_921
            )
        )

    monkeypatch.setattr(
        UE,
        "queueDataPckt",
        fake_queue,
    )

    ue = make_fake_ue()

    cell = SimpleNamespace(
        maxBuffUE=(
            EXPECTED_BUFFER_BYTES
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "capacity invariant "
            "violated"
        ),
    ):
        with bearer_capacity_guard():
            UE.queueDataPckt(
                ue,
                cell,
            )


def test_capacity_guard_rejects_wrong_limit(
    monkeypatch,
):
    called = False

    def fake_queue(
        self,
        cell,
    ):
        nonlocal called
        called = True

    monkeypatch.setattr(
        UE,
        "queueDataPckt",
        fake_queue,
    )

    ue = make_fake_ue()

    cell = SimpleNamespace(
        maxBuffUE=100
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "unexpected per-UE bearer "
            "capacity"
        ),
    ):
        with bearer_capacity_guard():
            UE.queueDataPckt(
                ue,
                cell,
            )

    assert called is False


def test_output_directory_must_be_empty(
    tmp_path,
):
    output = (
        tmp_path
        / "collection"
    )

    output.mkdir(
        parents=True
    )

    (
        output
        / "existing.txt"
    ).write_text(
        "evidence\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="not empty",
    ):
        prepare_output_directory(
            output
        )


def test_output_directory_creates_raw_dir(
    tmp_path,
):
    output = (
        tmp_path
        / "collection"
    )

    raw = (
        prepare_output_directory(
            output
        )
    )

    assert raw == (
        output / "raw"
    )

    assert raw.is_dir()
