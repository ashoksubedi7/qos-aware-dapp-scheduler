from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess


from experiments.post_buffer_fix_drain_revalidation import (
    EXPECTED_RUN_COUNT,
    POST_FIX_ACTIVE_DURATION_MS,
    POST_FIX_DRAIN_CANDIDATES_MS,
    POST_FIX_SEEDS,
    POST_FIX_TRAFFIC_SCENARIOS,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    SIMULATOR_SEMANTICS_ANCHOR,
    classify_revalidation,
    validate_complete_grid,
)


def sha256_file(path):
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_protocol_hash(
    protocol_path,
):
    actual = sha256_file(
        protocol_path
    )

    if actual != PROTOCOL_SHA256:
        raise RuntimeError(
            "post-buffer-fix drain "
            "revalidation protocol hash "
            "mismatch: "
            f"expected {PROTOCOL_SHA256}, "
            f"found {actual}"
        )

    return actual


def git_head(repo_root):
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_root,
        text=True,
    ).strip()


def tracked_worktree_status(repo_root):
    return subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=repo_root,
        text=True,
    ).strip()


def require_clean_tracked_worktree(
    repo_root,
):
    status = tracked_worktree_status(
        repo_root
    )

    if status:
        raise RuntimeError(
            "tracked worktree is not clean; "
            "refusing scientific collection:\n"
            f"{status}"
        )


def write_run_result(
    summary,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(summary, "__dataclass_fields__"):
        result = asdict(summary)
    else:
        result = dict(summary)

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    return result


def load_run_result(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _raw_entry(
    output_dir,
    key,
    path,
):
    scenario, seed, drain_ms = key

    path = Path(path)

    return {
        "traffic_scenario":
            str(scenario),
        "seed":
            int(seed),
        "drain_duration_ms":
            float(drain_ms),
        "path":
            str(
                path.relative_to(
                    output_dir
                )
            ),
        "sha256":
            sha256_file(path),
    }


def write_revalidation_artifacts(
    results,
    raw_files,
    output_dir,
    protocol_path,
    protocol_sha256,
    collection_source_commit,
):
    results = list(results)

    validate_complete_grid(
        results
    )

    if len(raw_files) != EXPECTED_RUN_COUNT:
        raise ValueError(
            "expected "
            f"{EXPECTED_RUN_COUNT} raw files; "
            f"found {len(raw_files)}"
        )

    output_dir = Path(
        output_dir
    )

    decision = classify_revalidation(
        results
    )

    decision_path = (
        output_dir
        / "decision.json"
    )

    with decision_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            asdict(decision),
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    raw_entries = []

    for key in sorted(
        raw_files
    ):
        raw_entries.append(
            _raw_entry(
                output_dir,
                key,
                raw_files[key],
            )
        )

    combined_digest = hashlib.sha256()

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
        "experiment":
            "post_buffer_fix_drain_revalidation",
        "simulator_semantics_anchor":
            SIMULATOR_SEMANTICS_ANCHOR,
        "protocol_commit":
            PROTOCOL_COMMIT,
        "protocol_path":
            str(protocol_path),
        "protocol_sha256":
            protocol_sha256,
        "collection_source_commit":
            str(collection_source_commit),
        "active_duration_ms":
            POST_FIX_ACTIVE_DURATION_MS,
        "drain_candidates_ms":
            list(
                POST_FIX_DRAIN_CANDIDATES_MS
            ),
        "traffic_scenarios":
            list(
                POST_FIX_TRAFFIC_SCENARIOS
            ),
        "seeds":
            list(
                POST_FIX_SEEDS
            ),
        "run_count":
            len(raw_entries),
        "raw_files":
            raw_entries,
        "combined_raw_sha256":
            combined_digest.hexdigest(),
        "decision": {
            "path":
                str(
                    decision_path.relative_to(
                        output_dir
                    )
                ),
            "sha256":
                sha256_file(
                    decision_path
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

    return (
        decision,
        manifest_path,
    )
