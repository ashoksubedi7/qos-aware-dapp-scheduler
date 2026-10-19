import json
import sys
from pathlib import Path

from common.reproducibility import set_global_seed
from config.experiment_config import ExperimentConfig
from experiments import scientific_evaluation_executor as ev


mode = sys.argv[1]
if mode not in ("baseline", "alternative"):
    raise SystemExit("Use baseline or alternative")

seed = 202
set_global_seed(seed)

root = Path.home() / "assuredqos-pilots"

checkpoint = (
    root / "m1-200updates-seed101"
    / "checkpoints/update_00000050"
)

output = (
    root / "evaluation-debug"
    / f"m1-branch-t1-{mode}-seed202-v2-packets"
)

if output.exists():
    raise SystemExit(f"Output already exists: {output}")

evidence = {
    "mode": mode,
    "seed": seed,
    "greedy_action_at_1ms": None,
    "snapshots": {},
    "deadline_counts": {},
}

tracked_packets = {}

original_cell = ev.Cell


def instrumented_cell(*args, **kwargs):
    cell = original_cell(*args, **kwargs)
    scheduler = cell.interSliceSched

    original_select = scheduler.select_policy_action
    original_apply = scheduler.apply_action

    decision_number = [0]
    target_flows = {}

    def select_with_one_intervention(state):
        index = decision_number[0]
        decision_number[0] += 1

        greedy = int(original_select(state))

        if index == 1:
            evidence["greedy_action_at_1ms"] = greedy

            if greedy != 13:
                raise AssertionError(
                    f"Expected Action 13 at 1 ms, got {greedy}"
                )

            if mode == "alternative":
                return 17

        return greedy

    def deadline_counts():
        return {
            ue_id: {
                "evaluated": int(flow.deadlineEvaluated),
                "misses": int(flow.deadlineMisses),
            }
            for ue_id, flow in target_flows.items()
        }

    def observe(snapshot):
        time_ms = float(snapshot["time_ms"])

        if time_ms == 1.0:
            users = (
                scheduler._canonical_slices()["URLLC"]
                .schedulerDL.ues
            )

            for ue_id, ue in users.items():
                if not ue.packetFlows:
                    continue

                flow = ue.packetFlows[0]
                target_flows[str(ue_id)] = flow

                deadline = flow.schedulingDeadline
                if deadline is None:
                    continue

                deadline = float(deadline)

                for bearer in ue.bearers:
                    for packet in bearer.buffer.pckts:
                        arrival = float(packet.tIn)

                        if not (
                            arrival <= time_ms
                            < arrival + deadline
                        ):
                            continue

                        if packet.scheduled_at is not None:
                            continue

                        key = f"{ue_id}:{int(packet.secNum)}"

                        tracked_packets[key] = (
                            packet,
                            arrival,
                            float(packet.size),
                            deadline,
                        )

        if time_ms in (1.0, 2.0):
            key = str(time_ms)

            evidence["snapshots"][key] = snapshot
            evidence["deadline_counts"][key] = (
                deadline_counts()
            )

    def checked_apply(action):
        result = original_apply(action)

        # Checks the allocation actually used by the
        # production scheduler, not just the proposal.
        time_ms = float(
            getattr(scheduler, "_branch_capture_time", -1)
        )

        if time_ms in (1.0, 2.0):
            snapshot = evidence["snapshots"][str(time_ms)]

            assert (
                result["action"]
                == snapshot["selected_action"]
            )

            assert (
                list(result["slice_prbs"])
                == snapshot["proposed_slice_prbs"]
            )

        return result

    def observe_with_time(snapshot):
        scheduler._branch_capture_time = (
            snapshot["time_ms"]
        )
        observe(snapshot)

    scheduler.select_policy_action = (
        select_with_one_intervention
    )

    scheduler.decision_snapshot_observer = (
        observe_with_time
    )

    scheduler.apply_action = checked_apply

    return cell


ev.Cell = instrumented_cell

config = ExperimentConfig(
    model_variant="M1",
    seed=seed,
    training_mode=False,
    checkpoint_path=str(checkpoint),
    traffic_scenario="BASELINE",
    radio_scenario="STATIC_GOOD",
    drain_duration_ms=0.0,
)

result = ev.run_checkpoint_evaluation(
    config=config,
    active_duration_ms=30.0,
    run_directory=output,
)

assert "1.0" in evidence["snapshots"]
assert "2.0" in evidence["snapshots"]
assert result.conservation_ok

expected = 13 if mode == "baseline" else 17

assert (
    evidence["snapshots"]["1.0"]["selected_action"]
    == expected
)

evidence["tracked_packets"] = {}

for key, item in sorted(tracked_packets.items()):
    packet, arrival, size, deadline = item

    scheduled = packet.scheduled_at
    if scheduled is not None:
        scheduled = float(scheduled)

    due = arrival + deadline

    evidence["tracked_packets"][key] = {
        "generated_at_ms": arrival,
        "size_bytes": size,
        "deadline_at_ms": due,
        "scheduled_at_ms": scheduled,
        "met_first_scheduling_deadline": (
            scheduled is not None
            and scheduled <= due + 1e-9
        ),
    }

print("Tracked original packets:",
      len(evidence["tracked_packets"]))

evidence["conservation_ok"] = bool(
    result.conservation_ok
)

output.mkdir(parents=True, exist_ok=True)

with (output / "branch_evidence.json").open("w") as f:
    json.dump(evidence, f, indent=2, allow_nan=False)

print(f"\n===== {mode.upper()} =====")
print("Frozen DQN action:", evidence["greedy_action_at_1ms"])
print("Applied action at 1 ms:", expected)

print(
    "Physical PRBs:",
    evidence["snapshots"]["1.0"]["proposed_slice_prbs"]
)

print("Deadline counters at 1 ms:")
print(evidence["deadline_counts"]["1.0"])

print("Deadline counters at 2 ms:")
print(evidence["deadline_counts"]["2.0"])

print("Packet conservation:", result.conservation_ok)
print("Evidence:", output / "branch_evidence.json")
