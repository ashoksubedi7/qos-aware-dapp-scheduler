from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

from experiments.backlog_calibration import (
    select_backlog_cap,
    summarize_backlog_samples,
    validate_backlog_cap,
)


SLICE_NAMES = (
    "eMBB",
    "URLLC",
    "mMTC",
)

SLICE_ATTRIBUTES = {
    "eMBB": "embb_bytes",
    "URLLC": "urllc_bytes",
    "mMTC": "mmtc_bytes",
}


def sha256_file(path):
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_protocol_hash(
    protocol_path,
    expected_sha256,
):
    actual = sha256_file(
        protocol_path
    )

    if actual != expected_sha256:
        raise RuntimeError(
            "backlog calibration protocol "
            "hash mismatch: "
            f"expected {expected_sha256}, "
            f"found {actual}"
        )

    return actual


def _slice_values(
    run,
    slice_name,
):
    return tuple(
        getattr(
            run,
            SLICE_ATTRIBUTES[
                slice_name
            ],
        )
    )


def build_calibration_analysis(
    runs,
    scenarios,
    seeds,
    expected_samples,
):
    runs = tuple(runs)
    scenarios = tuple(scenarios)
    seeds = tuple(
        int(seed)
        for seed in seeds
    )

    expected_keys = {
        (
            scenario,
            seed,
        )
        for scenario in scenarios
        for seed in seeds
    }

    runs_by_key = {}

    for run in runs:
        key = (
            run.traffic_scenario,
            int(run.seed),
        )

        if key in runs_by_key:
            raise ValueError(
                "duplicate calibration run: "
                f"{key}"
            )

        runs_by_key[key] = run

    actual_keys = set(
        runs_by_key
    )

    if actual_keys != expected_keys:
        missing = sorted(
            expected_keys
            - actual_keys
        )

        unexpected = sorted(
            actual_keys
            - expected_keys
        )

        raise ValueError(
            "incomplete calibration grid; "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )

    summaries = {}

    for scenario in scenarios:
        for seed in seeds:
            run = runs_by_key[
                (
                    scenario,
                    seed,
                )
            ]

            if (
                run.sample_count
                != expected_samples
            ):
                raise ValueError(
                    "unexpected sample count "
                    f"for {scenario}/seed{seed}: "
                    f"{run.sample_count}; "
                    f"expected {expected_samples}"
                )

            for slice_name in (
                SLICE_NAMES
            ):
                values = _slice_values(
                    run,
                    slice_name,
                )

                if (
                    len(values)
                    != expected_samples
                ):
                    raise ValueError(
                        "slice sample count "
                        "mismatch for "
                        f"{scenario}/seed{seed}/"
                        f"{slice_name}"
                    )

                summaries[
                    (
                        scenario,
                        seed,
                        slice_name,
                    )
                ] = (
                    summarize_backlog_samples(
                        values
                    )
                )

    caps = {}

    for slice_name in SLICE_NAMES:
        caps[slice_name] = (
            select_backlog_cap(
                summaries[
                    (
                        scenario,
                        seed,
                        slice_name,
                    )
                ]
                for scenario
                in scenarios
                for seed
                in seeds
            )
        )

    per_run_rows = []
    all_valid = True

    for scenario in scenarios:
        for seed in seeds:
            run = runs_by_key[
                (
                    scenario,
                    seed,
                )
            ]

            for slice_name in (
                SLICE_NAMES
            ):
                summary = summaries[
                    (
                        scenario,
                        seed,
                        slice_name,
                    )
                ]

                values = _slice_values(
                    run,
                    slice_name,
                )

                validation = (
                    validate_backlog_cap(
                        values,
                        caps[
                            slice_name
                        ],
                    )
                )

                if not validation.passes:
                    all_valid = False

                row = {
                    "traffic_scenario":
                        scenario,
                    "seed":
                        seed,
                    "slice":
                        slice_name,
                }

                row.update(
                    asdict(summary)
                )

                row.update(
                    {
                        "selected_cap_bytes":
                            validation
                            .cap_bytes,
                        "above_cap":
                            validation
                            .above_cap,
                        "fraction_above_cap":
                            validation
                            .fraction_above_cap,
                        "passes":
                            validation
                            .passes,
                    }
                )

                per_run_rows.append(
                    row
                )

    pooled_rows = []

    for slice_name in SLICE_NAMES:
        pooled = []

        for scenario in scenarios:
            for seed in seeds:
                pooled.extend(
                    _slice_values(
                        runs_by_key[
                            (
                                scenario,
                                seed,
                            )
                        ],
                        slice_name,
                    )
                )

        summary = (
            summarize_backlog_samples(
                pooled
            )
        )

        row = {
            "slice":
                slice_name,
            "selected_cap_bytes":
                caps[
                    slice_name
                ],
        }

        row.update(
            asdict(summary)
        )

        pooled_rows.append(
            row
        )

    return {
        "caps": caps,
        "per_run_rows":
            tuple(per_run_rows),
        "pooled_rows":
            tuple(pooled_rows),
        "all_valid":
            all_valid,
    }


def _write_rows_csv(
    path,
    rows,
):
    rows = tuple(rows)

    if not rows:
        raise ValueError(
            "cannot write empty CSV"
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_analysis_artifacts(
    analysis,
    output_dir,
    protocol_path,
    protocol_sha256,
    raw_files,
    scenarios,
    seeds,
    active_duration_ms,
    sample_interval_ms,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_run_path = (
        output_dir
        / "per_run_summary.csv"
    )

    pooled_path = (
        output_dir
        / "pooled_summary.csv"
    )

    caps_path = (
        output_dir
        / "selected_caps.json"
    )

    _write_rows_csv(
        per_run_path,
        analysis[
            "per_run_rows"
        ],
    )

    _write_rows_csv(
        pooled_path,
        analysis[
            "pooled_rows"
        ],
    )

    with caps_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "unit": "bytes",
                "eMBB":
                    analysis[
                        "caps"
                    ]["eMBB"],
                "URLLC":
                    analysis[
                        "caps"
                    ]["URLLC"],
                "mMTC":
                    analysis[
                        "caps"
                    ]["mMTC"],
                "all_runs_pass":
                    analysis[
                        "all_valid"
                    ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    raw_entries = []

    for key in sorted(
        raw_files
    ):
        raw_path = Path(
            raw_files[key]
        )

        raw_entries.append(
            {
                "traffic_scenario":
                    key[0],
                "seed":
                    int(key[1]),
                "path":
                    str(
                        raw_path.relative_to(
                            output_dir
                        )
                    ),
                "sha256":
                    sha256_file(
                        raw_path
                    ),
            }
        )

    combined_digest = (
        hashlib.sha256()
    )

    for entry in raw_entries:
        combined_digest.update(
            (
                entry["path"]
                + "\t"
                + entry["sha256"]
                + "\n"
            ).encode(
                "utf-8"
            )
        )

    manifest = {
        "protocol_path":
            str(protocol_path),
        "protocol_sha256":
            protocol_sha256,
        "active_duration_ms":
            float(
                active_duration_ms
            ),
        "sample_interval_ms":
            float(
                sample_interval_ms
            ),
        "scenarios":
            list(scenarios),
        "seeds":
            [
                int(seed)
                for seed in seeds
            ],
        "run_count":
            len(raw_entries),
        "raw_files":
            raw_entries,
        "combined_raw_sha256":
            combined_digest.hexdigest(),
        "analysis_files": {
            "per_run_summary.csv":
                sha256_file(
                    per_run_path
                ),
            "pooled_summary.csv":
                sha256_file(
                    pooled_path
                ),
            "selected_caps.json":
                sha256_file(
                    caps_path
                ),
        },
    }

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    return manifest_path
