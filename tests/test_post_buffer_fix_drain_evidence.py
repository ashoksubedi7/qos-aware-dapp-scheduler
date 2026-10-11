import json
from pathlib import Path

import pytest


from experiments.post_buffer_fix_drain_evidence import (
    load_run_result,
    require_clean_tracked_worktree,
    sha256_file,
    verify_protocol_hash,
    write_revalidation_artifacts,
    write_run_result,
)

from experiments.post_buffer_fix_drain_revalidation import (
    POST_FIX_DRAIN_CANDIDATES_MS,
    POST_FIX_SEEDS,
    POST_FIX_TRAFFIC_SCENARIOS,
    PROTOCOL_SHA256,
)


def make_slice(
    residual,
    p99=5.0,
):
    return {
        "generated": 100,
        "delivered": 100,
        "dropped": 0,
        "residual": 0,
        "completion_ratio": 1.0,
        "true_drop_ratio": 0.0,
        "residual_ratio": residual,
        "completion_delay_count": 100,
        "completion_delay_mean_ms": 1.0,
        "completion_delay_p95_ms": 4.0,
        "completion_delay_p99_ms": p99,
        "completion_delay_p99_9_ms": 6.0,
        "completion_delay_max_ms": 7.0,
    }


def make_results():
    results = []

    for scenario in (
        POST_FIX_TRAFFIC_SCENARIOS
    ):
        for seed in POST_FIX_SEEDS:
            for drain in (
                POST_FIX_DRAIN_CANDIDATES_MS
            ):
                residual = (
                    0.02
                    if drain == 500.0
                    else 0.001
                )

                results.append(
                    {
                        "seed": seed,
                        "traffic_scenario":
                            scenario,
                        "active_duration_ms":
                            10_000.0,
                        "drain_duration_ms":
                            drain,
                        "total_duration_ms":
                            10_000.0 + drain,
                        "embb":
                            make_slice(
                                residual
                            ),
                        "urllc":
                            make_slice(
                                residual
                            ),
                        "mmtc":
                            make_slice(
                                residual
                            ),
                    }
                )

    return results


def test_sha256_file(tmp_path):
    path = tmp_path / "x.txt"

    path.write_text(
        "abc\n",
        encoding="utf-8",
    )

    assert sha256_file(path) == (
        "edeaaff3f1774ad2888673770c6d6409"
        "7e391bc362d7d6fb34982ddf0efd18cb"
    )


def test_protocol_hash_verification(
    tmp_path,
):
    protocol = tmp_path / "protocol.txt"

    protocol.write_text(
        "not the frozen protocol\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="protocol hash mismatch",
    ):
        verify_protocol_hash(
            protocol
        )


def test_write_and_load_run_result(
    tmp_path,
):
    result = make_results()[0]

    path = (
        tmp_path
        / "run.json"
    )

    written = write_run_result(
        result,
        path,
    )

    assert written == result
    assert (
        load_run_result(path)
        == result
    )


def test_manifest_and_decision_hashes(
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

    results = make_results()
    raw_files = {}

    for result in results:
        key = (
            result[
                "traffic_scenario"
            ],
            result["seed"],
            result[
                "drain_duration_ms"
            ],
        )

        path = (
            raw_dir
            / (
                f"{key[0]}_"
                f"seed{key[1]}_"
                f"drain{int(key[2])}.json"
            )
        )

        write_run_result(
            result,
            path,
        )

        raw_files[key] = path

    protocol = (
        tmp_path
        / "protocol.txt"
    )

    protocol.write_text(
        "protocol placeholder\n",
        encoding="utf-8",
    )

    decision, manifest_path = (
        write_revalidation_artifacts(
            results=results,
            raw_files=raw_files,
            output_dir=output_dir,
            protocol_path=protocol,
            protocol_sha256=(
                PROTOCOL_SHA256
            ),
            collection_source_commit=(
                "a" * 40
            ),
        )
    )

    assert decision.case == "A"
    assert (
        decision.retain_1000_ms
        is True
    )

    manifest = json.loads(
        Path(
            manifest_path
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["run_count"]
        == 36
    )

    assert (
        manifest[
            "collection_source_commit"
        ]
        == "a" * 40
    )

    assert (
        manifest[
            "protocol_sha256"
        ]
        == PROTOCOL_SHA256
    )

    assert len(
        manifest["raw_files"]
    ) == 36

    for entry in manifest[
        "raw_files"
    ]:
        path = (
            output_dir
            / entry["path"]
        )

        assert entry[
            "sha256"
        ] == sha256_file(path)

    decision_path = (
        output_dir
        / manifest["decision"]["path"]
    )

    assert manifest[
        "decision"
    ]["sha256"] == sha256_file(
        decision_path
    )


def test_dirty_tracked_worktree_fails(
    tmp_path,
):
    repo = tmp_path / "repo"

    repo.mkdir()

    import subprocess

    subprocess.run(
        ["git", "init"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "test@example.com",
        ],
        cwd=repo,
        check=True,
    )

    subprocess.run(
        [
            "git",
            "config",
            "user.name",
            "Test",
        ],
        cwd=repo,
        check=True,
    )

    tracked = repo / "tracked.txt"

    tracked.write_text(
        "clean\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "git",
            "add",
            "tracked.txt",
        ],
        cwd=repo,
        check=True,
    )

    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    require_clean_tracked_worktree(
        repo
    )

    tracked.write_text(
        "dirty\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="not clean",
    ):
        require_clean_tracked_worktree(
            repo
        )
