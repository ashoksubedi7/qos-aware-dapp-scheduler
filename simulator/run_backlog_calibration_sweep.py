from pathlib import Path
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
    "d8184bb5871f9c687aba8b2ad29d17b7c53f31cdf3ea2ddcba3b6cc93f8a8540"
)

SEEDS = (
    7,
    17,
    27,
)

ACTIVE_DURATION_MS = 10_000.0

EXPECTED_SAMPLES = 10_000

PROTOCOL_PATH = Path(
    "results/backlog_calibration/"
    "backlog_calibration_protocol.txt"
)

OUTPUT_DIR = Path(
    "results/backlog_calibration/"
    "collection"
)


def main():
    protocol_sha = (
        verify_protocol_hash(
            PROTOCOL_PATH,
            EXPECTED_PROTOCOL_SHA256,
        )
    )

    if (
        OUTPUT_DIR.exists()
        and any(
            OUTPUT_DIR.iterdir()
        )
    ):
        raise RuntimeError(
            "calibration output directory "
            "is not empty; refusing to "
            "overwrite collected evidence: "
            f"{OUTPUT_DIR}"
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

    total_runs = (
        len(CALIBRATION_SCENARIOS)
        * len(SEEDS)
    )

    run_number = 0

    start_all = time.time()

    for scenario in (
        CALIBRATION_SCENARIOS
    ):
        for seed in SEEDS:
            run_number += 1

            print(
                f"[{run_number}/{total_runs}] "
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
                    f"count: {run.sample_count}"
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

    print()
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
