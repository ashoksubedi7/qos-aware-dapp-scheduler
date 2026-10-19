"""Validate the recorded M1 packet-level counterfactual."""

import json
from pathlib import Path


ROOT = Path(
    "evidence/witnesses/m1-update50-seed202-t1"
)


def read(mode):
    return json.loads(
        (ROOT / f"{mode}.json").read_text()
    )


baseline = read("baseline")
alternative = read("alternative")

assert baseline["seed"] == alternative["seed"] == 202
assert baseline["greedy_action_at_1ms"] == 13
assert alternative["greedy_action_at_1ms"] == 13
assert baseline["conservation_ok"]
assert alternative["conservation_ok"]

before_b = baseline["snapshots"]["1.0"]
before_a = alternative["snapshots"]["1.0"]

for field in (
    "time_ms",
    "observation",
    "reference_prbs_available",
    "slices",
):
    assert before_b[field] == before_a[field], field

assert before_b["selected_action"] == 13
assert before_a["selected_action"] == 17

assert before_b["proposed_slice_prbs"][1] == 0
assert before_a["proposed_slice_prbs"][1] == 8

original = baseline["tracked_packets"]
modified = alternative["tracked_packets"]

assert set(original) == set(modified)
assert len(original) == 3


def queued_packets(snapshot):
    return {
        f'{ue["ue_id"]}:{packet["packet_id"]}': packet
        for ue in snapshot["slices"]["URLLC"]["ues"]
        for bearer in ue["bearer_queues"]
        for packet in bearer["packets"]
    }


waiting = queued_packets(before_b)
avoided = 0

for key in sorted(original):
    b = original[key]
    a = modified[key]

    assert key in waiting
    assert waiting[key]["scheduled_at_ms"] is None

    for field in (
        "generated_at_ms",
        "size_bytes",
        "deadline_at_ms",
    ):
        assert b[field] == a[field], (key, field)

    deadline = b["deadline_at_ms"]
    baseline_time = b["scheduled_at_ms"]
    alternative_time = a["scheduled_at_ms"]

    assert b["generated_at_ms"] <= 1.0 < deadline

    baseline_missed = (
        baseline_time is None
        or baseline_time > deadline + 1e-9
    )

    alternative_met = (
        alternative_time is not None
        and 1.0 - 1e-9 <= alternative_time
        <= deadline + 1e-9
    )

    assert baseline_missed, key
    assert alternative_met, key
    assert not b["met_first_scheduling_deadline"]
    assert a["met_first_scheduling_deadline"]

    avoided += 1

    print(
        f"{key}: baseline={baseline_time}, "
        f"alternative={alternative_time}, "
        f"deadline={deadline:.6f}"
    )

print("\nCaptured pre-action states match.")
print("Both branches passed packet conservation.")
print(f"Verified packet-specific misses avoided: {avoided}")
print("Status: empirical witness; not a formal proof.")
