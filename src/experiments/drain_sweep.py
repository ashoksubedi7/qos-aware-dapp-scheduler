from dataclasses import asdict
import json
from pathlib import Path

from drain_calibration_runner import (
    run_drain_calibration_case,
)


DRAIN_CANDIDATES_MS = (
    0.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
)

CALIBRATION_SEEDS = (
    7,
    17,
    27,
)

TRAFFIC_SCENARIOS = (
    "BASELINE",
    "CONGESTED",
    "URLLC_HIGH",
    "SIMULTANEOUS_HIGH",
)

ACTIVE_DURATION_MS = 10_000.0


def run_drain_sweep(
    output_path,
):
    results = load_drain_sweep(
        output_path
    )

    completed = {
        _run_key(
            result
        )
        for result in results
    }

    total_expected = (
        len(
            TRAFFIC_SCENARIOS
        )
        * len(
            CALIBRATION_SEEDS
        )
        * len(
            DRAIN_CANDIDATES_MS
        )
    )

    for traffic_scenario in TRAFFIC_SCENARIOS:
        for seed in CALIBRATION_SEEDS:
            for drain_ms in DRAIN_CANDIDATES_MS:
                key = (
                    traffic_scenario,
                    int(
                        seed
                    ),
                    float(
                        drain_ms
                    ),
                )

                if key in completed:
                    print(
                        "Skipping completed:",
                        key,
                    )
                    continue

                print(
                    "Running:",
                    key,
                    f"({len(results) + 1}/"
                    f"{total_expected})",
                    flush=True,
                )

                summary = (
                    run_drain_calibration_case(
                        active_duration_ms=(
                            ACTIVE_DURATION_MS
                        ),
                        drain_duration_ms=(
                            drain_ms
                        ),
                        seed=seed,
                        traffic_scenario=(
                            traffic_scenario
                        ),
                    )
                )

                result = asdict(
                    summary
                )

                results.append(
                    result
                )

                completed.add(
                    key
                )

                save_drain_sweep(
                    results,
                    output_path,
                )

    return results


def save_drain_sweep(
    results,
    output_path,
):
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            sort_keys=True,
        )

    temporary.replace(
        path
    )


def _run_key(
    result,
):
    return (
        str(
            result[
                "traffic_scenario"
            ]
        ),
        int(
            result[
                "seed"
            ]
        ),
        float(
            result[
                "drain_duration_ms"
            ]
        ),
    )
def load_drain_sweep(
    output_path,
):
    path = Path(
        output_path
    )

    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        results = json.load(
            handle
        )

    seen = set()

    for result in results:
        key = _run_key(
            result
        )

        if key in seen:
            raise ValueError(
                "duplicate saved calibration "
                f"result: {key}"
            )

        seen.add(
            key
        )

    return results
    
if __name__ == "__main__":
    output_path = (
        "results/"
        "drain_calibration/"
        "drain_sweep.json"
    )

    results = run_drain_sweep(
        output_path
    )

    print(
        "Completed "
        f"{len(results)} "
        "drain-calibration runs."
    )

