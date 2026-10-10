from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import time


from UE import UE

from drain_calibration_runner import (
    run_drain_calibration_case,
)

from experiments.post_buffer_fix_drain_evidence import (
    git_head,
    require_clean_tracked_worktree,
    verify_protocol_hash,
    write_revalidation_artifacts,
    write_run_result,
)

from experiments.post_buffer_fix_drain_revalidation import (
    EXPECTED_RUN_COUNT,
    POST_FIX_ACTIVE_DURATION_MS,
    POST_FIX_DRAIN_CANDIDATES_MS,
    POST_FIX_SEEDS,
    POST_FIX_TRAFFIC_SCENARIOS,
    PROTOCOL_COMMIT,
    SIMULATOR_SEMANTICS_ANCHOR,
)


EXPECTED_BUFFER_BYTES = 81_920

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROTOCOL_RELATIVE_PATH = Path(
    "results/drain_calibration/"
    "post_buffer_fix_revalidation_protocol.txt"
)

PROTOCOL_PATH = (
    REPO_ROOT
    / PROTOCOL_RELATIVE_PATH
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results/drain_calibration/"
    "post_buffer_fix_revalidation_collection"
)


def require_commit_ancestor(
    repo_root,
    ancestor,
    descendant,
):
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "required provenance commit is "
            "not an ancestor of collection "
            "source commit: "
            f"ancestor={ancestor}, "
            f"descendant={descendant}, "
            f"stderr={completed.stderr.strip()}"
        )


def prepare_output_directory(
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    if (
        output_dir.exists()
        and any(
            output_dir.iterdir()
        )
    ):
        raise RuntimeError(
            "post-fix drain revalidation "
            "output directory is not empty; "
            "refusing to overwrite evidence: "
            f"{output_dir}"
        )

    raw_dir = (
        output_dir
        / "raw"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return raw_dir


@contextmanager
def bearer_capacity_guard():
    original = UE.queueDataPckt

    monitor = {
        "admission_calls": 0,
        "limits_seen_bytes": set(),
        "max_per_slice_bytes": {},
    }

    def checked_queue_data_packet(
        self,
        cell,
    ):
        limit = int(
            cell.maxBuffUE
        )

        if limit != EXPECTED_BUFFER_BYTES:
            raise RuntimeError(
                "unexpected per-UE bearer "
                "capacity during post-fix "
                "revalidation: "
                f"expected="
                f"{EXPECTED_BUFFER_BYTES}, "
                f"found={limit}"
            )

        result = original(
            self,
            cell,
        )

        monitor[
            "admission_calls"
        ] += 1

        monitor[
            "limits_seen_bytes"
        ].add(
            limit
        )

        if self.packetFlows:
            slice_name = str(
                self.packetFlows[
                    0
                ].sliceName
            )
        else:
            slice_name = "UNKNOWN"

        for bearer in self.bearers:
            occupancy = sum(
                int(packet.size)
                for packet
                in bearer.buffer.pckts
            )

            previous = monitor[
                "max_per_slice_bytes"
            ].get(
                slice_name,
                0,
            )

            monitor[
                "max_per_slice_bytes"
            ][slice_name] = max(
                previous,
                occupancy,
            )

            if occupancy > limit:
                raise RuntimeError(
                    "bearer capacity invariant "
                    "violated: "
                    f"slice={slice_name}, "
                    f"ue={self.id}, "
                    f"occupancy={occupancy}, "
                    f"limit={limit}"
                )

        return result

    UE.queueDataPckt = (
        checked_queue_data_packet
    )

    try:
        yield monitor
    finally:
        UE.queueDataPckt = original


def finalize_capacity_evidence(
    monitor,
):
    admission_calls = int(
        monitor["admission_calls"]
    )

    if admission_calls <= 0:
        raise RuntimeError(
            "capacity invariant monitor "
            "observed no packet admissions"
        )

    limits = sorted(
        int(value)
        for value
        in monitor[
            "limits_seen_bytes"
        ]
    )

    if limits != [
        EXPECTED_BUFFER_BYTES
    ]:
        raise RuntimeError(
            "unexpected observed bearer "
            "capacity set: "
            f"{limits}"
        )

    max_per_slice = {
        str(name): int(value)
        for name, value
        in sorted(
            monitor[
                "max_per_slice_bytes"
            ].items()
        )
    }

    if not max_per_slice:
        raise RuntimeError(
            "capacity invariant monitor "
            "observed no bearer occupancy"
        )

    for (
        slice_name,
        occupancy,
    ) in max_per_slice.items():
        if (
            occupancy
            > EXPECTED_BUFFER_BYTES
        ):
            raise RuntimeError(
                "bearer capacity invariant "
                "failed after collection: "
                f"slice={slice_name}, "
                f"occupancy={occupancy}"
            )

    return {
        "passes": True,
        "configured_limit_bytes":
            EXPECTED_BUFFER_BYTES,
        "admission_calls":
            admission_calls,
        "limits_seen_bytes":
            limits,
        "max_per_slice_bytes":
            max_per_slice,
    }


def _require_finite_nonnegative(
    value,
    name,
):
    numeric = float(value)

    if (
        not math.isfinite(
            numeric
        )
        or numeric < 0.0
    ):
        raise RuntimeError(
            "invalid drain metric: "
            f"{name}={value}"
        )


def validate_run_summary(
    summary,
    expected_scenario,
    expected_seed,
    expected_drain_ms,
):
    result = asdict(
        summary
    )

    if (
        result[
            "traffic_scenario"
        ]
        != expected_scenario
    ):
        raise RuntimeError(
            "traffic scenario mismatch"
        )

    if (
        int(result["seed"])
        != int(expected_seed)
    ):
        raise RuntimeError(
            "seed mismatch"
        )

    if (
        float(
            result[
                "active_duration_ms"
            ]
        )
        != POST_FIX_ACTIVE_DURATION_MS
    ):
        raise RuntimeError(
            "active duration mismatch"
        )

    if (
        float(
            result[
                "drain_duration_ms"
            ]
        )
        != float(
            expected_drain_ms
        )
    ):
        raise RuntimeError(
            "drain duration mismatch"
        )

    expected_total = (
        POST_FIX_ACTIVE_DURATION_MS
        + float(
            expected_drain_ms
        )
    )

    if (
        float(
            result[
                "total_duration_ms"
            ]
        )
        != expected_total
    ):
        raise RuntimeError(
            "total duration mismatch"
        )

    for slice_name in (
        "embb",
        "urllc",
        "mmtc",
    ):
        data = result[
            slice_name
        ]

        generated = int(
            data["generated"]
        )

        delivered = int(
            data["delivered"]
        )

        dropped = int(
            data["dropped"]
        )

        residual = int(
            data["residual"]
        )

        for name, value in (
            ("generated", generated),
            ("delivered", delivered),
            ("dropped", dropped),
            ("residual", residual),
        ):
            if value < 0:
                raise RuntimeError(
                    "negative packet "
                    "accounting metric: "
                    f"{slice_name}/"
                    f"{name}={value}"
                )

        if (
            generated
            != delivered
            + dropped
            + residual
        ):
            raise RuntimeError(
                "packet accounting "
                "conservation failed: "
                f"{slice_name}"
            )

        for ratio_name in (
            "completion_ratio",
            "true_drop_ratio",
            "residual_ratio",
        ):
            ratio = float(
                data[
                    ratio_name
                ]
            )

            if (
                not math.isfinite(
                    ratio
                )
                or ratio < 0.0
                or ratio > 1.0
            ):
                raise RuntimeError(
                    "invalid lifecycle ratio: "
                    f"{slice_name}/"
                    f"{ratio_name}={ratio}"
                )

        count = int(
            data[
                "completion_delay_count"
            ]
        )

        if count < 0:
            raise RuntimeError(
                "negative completion "
                "delay count: "
                f"{slice_name}"
            )

        for metric_name in (
            "completion_delay_mean_ms",
            "completion_delay_p95_ms",
            "completion_delay_p99_ms",
            "completion_delay_p99_9_ms",
            "completion_delay_max_ms",
        ):
            _require_finite_nonnegative(
                data[
                    metric_name
                ],
                (
                    f"{slice_name}/"
                    f"{metric_name}"
                ),
            )

    return result


def write_failure_record(
    output_dir,
    collection_source_commit,
    protocol_sha256,
    scenario,
    seed,
    drain_ms,
    error,
):
    failure_path = (
        Path(output_dir)
        / "failure.json"
    )

    record = {
        "experiment":
            "post_buffer_fix_drain_revalidation",
        "collection_source_commit":
            collection_source_commit,
        "protocol_sha256":
            protocol_sha256,
        "traffic_scenario":
            scenario,
        "seed":
            int(seed),
        "drain_duration_ms":
            float(drain_ms),
        "error_type":
            type(error).__name__,
        "error_message":
            str(error),
    }

    with failure_path.open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            record,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    return failure_path


def main():
    require_clean_tracked_worktree(
        REPO_ROOT
    )

    protocol_sha = (
        verify_protocol_hash(
            PROTOCOL_PATH
        )
    )

    source_commit = git_head(
        REPO_ROOT
    )

    require_commit_ancestor(
        REPO_ROOT,
        SIMULATOR_SEMANTICS_ANCHOR,
        source_commit,
    )

    require_commit_ancestor(
        REPO_ROOT,
        PROTOCOL_COMMIT,
        source_commit,
    )

    raw_dir = (
        prepare_output_directory(
            OUTPUT_DIR
        )
    )

    results = []
    raw_files = {}

    run_number = 0

    started_all = time.time()

    for scenario in (
        POST_FIX_TRAFFIC_SCENARIOS
    ):
        for seed in POST_FIX_SEEDS:
            for drain_ms in (
                POST_FIX_DRAIN_CANDIDATES_MS
            ):
                run_number += 1

                print(
                    f"[{run_number}/"
                    f"{EXPECTED_RUN_COUNT}] "
                    f"{scenario} "
                    f"seed={seed} "
                    f"drain="
                    f"{int(drain_ms)}ms",
                    flush=True,
                )

                started = time.time()

                try:
                    with (
                        bearer_capacity_guard()
                        as monitor
                    ):
                        summary = (
                            run_drain_calibration_case(
                                active_duration_ms=(
                                    POST_FIX_ACTIVE_DURATION_MS
                                ),
                                drain_duration_ms=(
                                    drain_ms
                                ),
                                seed=seed,
                                traffic_scenario=(
                                    scenario
                                ),
                            )
                        )

                    capacity = (
                        finalize_capacity_evidence(
                            monitor
                        )
                    )

                    result = (
                        validate_run_summary(
                            summary,
                            expected_scenario=(
                                scenario
                            ),
                            expected_seed=seed,
                            expected_drain_ms=(
                                drain_ms
                            ),
                        )
                    )

                    result[
                        "capacity_invariant"
                    ] = capacity

                    raw_path = (
                        raw_dir
                        / (
                            f"{scenario}_"
                            f"seed{seed}_"
                            f"drain"
                            f"{int(drain_ms)}ms"
                            ".json"
                        )
                    )

                    write_run_result(
                        result,
                        raw_path,
                    )

                    key = (
                        scenario,
                        int(seed),
                        float(drain_ms),
                    )

                    results.append(
                        result
                    )

                    raw_files[
                        key
                    ] = raw_path

                except Exception as error:
                    failure_path = (
                        write_failure_record(
                            OUTPUT_DIR,
                            source_commit,
                            protocol_sha,
                            scenario,
                            seed,
                            drain_ms,
                            error,
                        )
                    )

                    print(
                        "CASE D / COLLECTION "
                        "FAILURE: "
                        f"{failure_path}",
                        flush=True,
                    )

                    raise

                elapsed = (
                    time.time()
                    - started
                )

                print(
                    "    admissions="
                    f"{capacity['admission_calls']} "
                    "elapsed="
                    f"{elapsed:.1f}s",
                    flush=True,
                )

    decision, manifest_path = (
        write_revalidation_artifacts(
            results=results,
            raw_files=raw_files,
            output_dir=OUTPUT_DIR,
            protocol_path=(
                PROTOCOL_RELATIVE_PATH
            ),
            protocol_sha256=(
                protocol_sha
            ),
            collection_source_commit=(
                source_commit
            ),
        )
    )

    print()
    print(
        "===== POST-FIX DRAIN "
        "REVALIDATION COMPLETE ====="
    )

    print(
        "collection_source_commit =",
        source_commit,
    )

    print(
        "protocol_sha256 =",
        protocol_sha,
    )

    print(
        "run_count =",
        len(results),
    )

    print(
        "decision_case =",
        decision.case,
    )

    print(
        "retain_1000_ms =",
        decision.retain_1000_ms,
    )

    print(
        "reopen_search =",
        decision.reopen_search,
    )

    print(
        "500_ms_qualifies =",
        decision.check_500_qualifies,
    )

    print(
        "1000_ms_qualifies =",
        decision.check_1000_qualifies,
    )

    print(
        "manifest =",
        manifest_path,
    )

    print(
        "total_elapsed_s =",
        f"{time.time() - started_all:.1f}",
    )


if __name__ == "__main__":
    main()
