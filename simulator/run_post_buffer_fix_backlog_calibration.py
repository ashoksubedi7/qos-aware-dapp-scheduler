from pathlib import Path
import json
import subprocess
import time

from backlog_calibration_runner import (
    CALIBRATION_SCENARIOS,
    SAMPLE_INTERVAL_MS,
    run_backlog_calibration_case,
    write_backlog_run_csv,
)

from experiments.backlog_calibration_sweep import (
    build_calibration_analysis,
    verify_protocol_hash,
    write_analysis_artifacts,
)


EXPECTED_PROTOCOL_SHA256 = (
    "6dcd4d2d7b2e91d549253e95c087c116"
    "5a84989440075af95512a1546cdddc50"
)

SIMULATOR_SEMANTICS_ANCHOR = (
    "84b731a85732d10dbb30f5374b2e11c3715d86e4"
)

HISTORICAL_COLLECTION_SOURCE_COMMIT = (
    "8f205487a5aa7803496fd93b06d39ecca81c5733"
)

SEEDS = (
    7,
    17,
    27,
)

ACTIVE_DURATION_MS = 10_000.0
EXPECTED_SAMPLES = 10_000

EXPECTED_RUN_COUNT = (
    len(CALIBRATION_SCENARIOS)
    * len(SEEDS)
)

PROTOCOL_PATH = Path(
    "results/backlog_calibration/"
    "post_buffer_fix_backlog_calibration_protocol.txt"
)

OUTPUT_DIR = Path(
    "results/backlog_calibration/"
    "post_buffer_fix_collection"
)

HISTORICAL_OUTPUT_DIR = Path(
    "results/backlog_calibration/"
    "collection"
)


def _git_output(*args):
    return subprocess.check_output(
        ("git", *args),
        text=True,
    ).strip()


def _git_head():
    return _git_output(
        "rev-parse",
        "HEAD",
    )


def _remote_main_head():
    output = subprocess.check_output(
        (
            "git",
            "ls-remote",
            "origin",
            "refs/heads/main",
        ),
        text=True,
    ).strip()

    fields = output.split()

    if (
        len(fields) != 2
        or fields[1] != "refs/heads/main"
    ):
        raise RuntimeError(
            "could not resolve exactly one "
            "origin main branch from the "
            "remote repository"
        )

    return fields[0]


def _assert_clean_worktree():
    status = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )

    if status:
        raise RuntimeError(
            "tracked working tree is not clean; "
            "refusing scientific collection:\n"
            f"{status}"
        )


def _assert_semantics_ancestry(
    collection_source_commit,
):
    result = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            SIMULATOR_SEMANTICS_ANCHOR,
            collection_source_commit,
        ),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "collection source commit does not "
            "descend from corrected simulator "
            "semantics anchor"
        )


def _assert_remote_main_matches(
    collection_source_commit,
):
    remote_main = _remote_main_head()

    if remote_main != collection_source_commit:
        raise RuntimeError(
            "origin/main does not match the "
            "collection source commit: "
            f"HEAD={collection_source_commit}, "
            f"origin/main={remote_main}"
        )

    return remote_main


def _assert_output_locations():
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "post-fix backlog calibration "
            "output directory already exists; "
            "refusing to overwrite evidence: "
            f"{OUTPUT_DIR}"
        )

    if not HISTORICAL_OUTPUT_DIR.is_dir():
        raise RuntimeError(
            "historical pre-fix backlog "
            "collection is missing: "
            f"{HISTORICAL_OUTPUT_DIR}"
        )

    if (
        OUTPUT_DIR.resolve()
        == HISTORICAL_OUTPUT_DIR.resolve()
    ):
        raise RuntimeError(
            "post-fix and historical collection "
            "directories must be distinct"
        )


def _augment_manifest(
    manifest_path,
    collection_source_commit,
    remote_main_commit,
):
    manifest_path = Path(
        manifest_path
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    observed_run_count = int(
        manifest["run_count"]
    )

    if observed_run_count != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "manifest run count mismatch: "
            f"expected {EXPECTED_RUN_COUNT}, "
            f"found {observed_run_count}"
        )

    manifest.update(
        {
            "experiment":
                "post_buffer_fix_backlog_calibration",
            "simulator_semantics_anchor":
                SIMULATOR_SEMANTICS_ANCHOR,
            "historical_collection_source_commit":
                HISTORICAL_COLLECTION_SOURCE_COMMIT,
            "collection_source_commit":
                collection_source_commit,
            "remote_main_commit":
                remote_main_commit,
            "expected_run_count":
                EXPECTED_RUN_COUNT,
            "observed_run_count":
                observed_run_count,
        }
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def main():
    protocol_sha = (
        verify_protocol_hash(
            PROTOCOL_PATH,
            EXPECTED_PROTOCOL_SHA256,
        )
    )

    collection_source_commit = (
        _git_head()
    )

    _assert_clean_worktree()

    _assert_semantics_ancestry(
        collection_source_commit
    )

    remote_main_commit = (
        _assert_remote_main_matches(
            collection_source_commit
        )
    )

    _assert_output_locations()

    print(
        "===== POST-BUFFER-FIX "
        "BACKLOG CALIBRATION ====="
    )

    print(
        "collection_source_commit =",
        collection_source_commit,
    )

    print(
        "remote_main_commit =",
        remote_main_commit,
    )

    print(
        "simulator_semantics_anchor =",
        SIMULATOR_SEMANTICS_ANCHOR,
    )

    print(
        "historical_collection_source_commit =",
        HISTORICAL_COLLECTION_SOURCE_COMMIT,
    )

    print(
        "protocol_sha256 =",
        protocol_sha,
    )

    print(
        "expected_run_count =",
        EXPECTED_RUN_COUNT,
    )

    raw_dir = (
        OUTPUT_DIR
        / "raw"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    runs = []
    raw_files = {}

    run_number = 0
    start_all = time.time()

    for scenario in (
        CALIBRATION_SCENARIOS
    ):
        for seed in SEEDS:
            run_number += 1

            print(
                f"[{run_number}/"
                f"{EXPECTED_RUN_COUNT}] "
                f"{scenario} seed={seed}",
                flush=True,
            )

            started = time.time()

            run = (
                run_backlog_calibration_case(
                    active_duration_ms=(
                        ACTIVE_DURATION_MS
                    ),
                    seed=seed,
                    traffic_scenario=(
                        scenario
                    ),
                    sample_interval_ms=(
                        SAMPLE_INTERVAL_MS
                    ),
                )
            )

            if (
                run.sample_count
                != EXPECTED_SAMPLES
            ):
                raise RuntimeError(
                    "unexpected run sample "
                    f"count: {run.sample_count}; "
                    f"expected "
                    f"{EXPECTED_SAMPLES}"
                )

            raw_path = (
                raw_dir
                / (
                    f"{scenario}_"
                    f"seed{seed}.csv"
                )
            )

            write_backlog_run_csv(
                run,
                raw_path,
            )

            runs.append(
                run
            )

            raw_files[
                (
                    scenario,
                    seed,
                )
            ] = raw_path

            elapsed = (
                time.time()
                - started
            )

            print(
                f"    samples="
                f"{run.sample_count} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if len(runs) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "incomplete post-fix backlog "
            "collection: "
            f"expected {EXPECTED_RUN_COUNT}, "
            f"found {len(runs)}"
        )

    if len(raw_files) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "raw-file grid is incomplete: "
            f"expected {EXPECTED_RUN_COUNT}, "
            f"found {len(raw_files)}"
        )

    analysis = (
        build_calibration_analysis(
            runs=runs,
            scenarios=(
                CALIBRATION_SCENARIOS
            ),
            seeds=SEEDS,
            expected_samples=(
                EXPECTED_SAMPLES
            ),
        )
    )

    manifest_path = (
        write_analysis_artifacts(
            analysis=analysis,
            output_dir=OUTPUT_DIR,
            protocol_path=(
                PROTOCOL_PATH
            ),
            protocol_sha256=(
                protocol_sha
            ),
            raw_files=raw_files,
            scenarios=(
                CALIBRATION_SCENARIOS
            ),
            seeds=SEEDS,
            active_duration_ms=(
                ACTIVE_DURATION_MS
            ),
            sample_interval_ms=(
                SAMPLE_INTERVAL_MS
            ),
        )
    )

    manifest = _augment_manifest(
        manifest_path,
        collection_source_commit,
        remote_main_commit,
    )

    print()
    print(
        "===== POST-FIX BACKLOG "
        "CALIBRATION COMPLETE ====="
    )

    print(
        "collection_source_commit =",
        manifest[
            "collection_source_commit"
        ],
    )

    print(
        "protocol_sha256 =",
        manifest[
            "protocol_sha256"
        ],
    )

    print(
        "run_count =",
        manifest[
            "run_count"
        ],
    )

    print(
        "Selected backlog caps:"
    )

    for slice_name in (
        "eMBB",
        "URLLC",
        "mMTC",
    ):
        print(
            f"  {slice_name}: "
            f"{analysis['caps'][slice_name]} "
            "bytes"
        )

    print(
        "All per-run validation "
        f"passed: "
        f"{analysis['all_valid']}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print(
        "Total elapsed: "
        f"{time.time() - start_all:.1f}s"
    )

    if not analysis[
        "all_valid"
    ]:
        raise RuntimeError(
            "one or more backlog cap "
            "validation checks failed"
        )


if __name__ == "__main__":
    main()
