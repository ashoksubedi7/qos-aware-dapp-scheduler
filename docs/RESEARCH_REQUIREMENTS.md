# AssuredQoS Research Requirements

## 1. Project Goal

AssuredQoS develops a QoS-aware deep reinforcement learning scheduler for O-RAN and evaluates whether formal neural-network verification can uncover rare QoS-critical scheduling decisions that conventional simulation may not expose.

The project will use formal counterexamples to reproduce network-level failures, improve the learned policy, and re-verify the repaired policy.

The final implementation target is a scheduling dApp integrated with an O-RAN/OAI environment.

---

## 2. Core Research Question

Can formal verification discover QoS-critical failures in a DRL-based RAN scheduler, and can those counterexamples be used to repair the learned policy while preserving network performance?

---

## 3. Research Questions

### RQ1
Can formal neural-network verification identify QoS-critical scheduling states that conventional simulation and stress testing fail to expose?

### RQ2
Can formal counterexamples be reproduced in the RAN simulation environment and shown to produce measurable QoS consequences?

### RQ3
Can counterexample-guided retraining improve the learned scheduler until selected QoS properties are formally satisfied over specified input regions?

### RQ4
What performance tradeoff results from counterexample-guided policy repair?

### RQ5
Can the assured policy operate within practical O-RAN dApp scheduling requirements?

---

## 4. Scheduler Models

### M0 — Original Queue-Only DQN

Inputs:

- eMBB backlog
- URLLC backlog
- mMTC backlog

Architecture:

- 3 inputs
- 32 ReLU
- 32 ReLU
- 22 outputs

Purpose:

Preserve the previously developed DQN as the original baseline.

---

### M1 — Context-Aware DRL Baseline

Inputs:

- eMBB backlog
- URLLC backlog
- mMTC backlog
- eMBB mean SINR
- URLLC mean SINR
- mMTC mean SINR

Purpose:

Measure the value of radio-context information without exposing deadline urgency to the policy.

---

### M2 — QoS/Deadline-Aware DRL Scheduler

Inputs:

- eMBB backlog
- URLLC backlog
- mMTC backlog
- eMBB mean SINR
- URLLC mean SINR
- mMTC mean SINR
- URLLC maximum head-of-line delay

Purpose:

Provide the scheduler with traffic demand, radio condition, and explicit URLLC urgency information.

---

### M3 — Counterexample-Guided Repaired Scheduler

M3 uses the same state representation, action space, and neural-network architecture as M2.

The difference is the training process:

1. Train M2.
2. Formally verify selected properties.
3. Extract counterexamples when properties are SAT.
4. Reproduce counterexamples in the simulator.
5. Add counterexample states and nearby states to the training distribution.
6. Retrain the policy.
7. Re-run formal verification.

This design keeps the M2-to-M3 comparison controlled.

---

## 5. Action Space

The initial action space remains the existing 22 discrete inter-slice allocation configurations.

Each neural-network output corresponds to one deterministic allocation across:

- eMBB
- URLLC
- mMTC

The discrete action space is retained because it:

- enables direct mapping between neural-network outputs and PRB allocation decisions,
- supports DQN/DDQN learning,
- simplifies formal argmax verification,
- preserves comparability with the original baseline.

Action-space granularity may later be evaluated as a separate experiment.

---

## 6. URLLC Scheduling Budgets

The primary experimental URLLC scheduling target will be:

- 1 ms

Sensitivity cases:

- 2 ms
- 5 ms
- 10 ms

These values represent experimental scheduling budgets.

Matching a standardized packet delay budget value does not establish compliance with the broader 3GPP end-to-end or UE-to-UPF packet delay budget.

---

## 7. QoS Metrics

### URLLC

Collect:

- mean scheduling delay
- p95 scheduling delay
- p99 scheduling delay
- p99.9 scheduling delay
- deadline miss ratio
- maximum head-of-line delay
- packet loss ratio

### eMBB

Collect:

- aggregate throughput
- queueing delay
- starvation event count
- mean starvation duration
- maximum starvation duration

### mMTC

Collect:

- generated packets
- served packets
- dropped packets
- queueing delay
- starvation event count
- mean starvation duration
- maximum starvation duration

### System

Collect:

- actual PRB utilization
- Jain fairness
- total reward
- mean reward

### DRL

Collect:

- training reward
- convergence behavior
- inference latency

### Formal Verification

For every property report:

- property definition
- input domain
- SAT / UNSAT / timeout
- solver runtime
- counterexample when SAT
- network-level reproduction result
- result after repair
- verification runtime after repair

---

## 8. Simulator Requirements

The simulator must expose or derive:

- per-slice backlog
- per-slice radio quality
- URLLC head-of-line delay
- actual PRB allocation
- actual PRB usage
- packet arrival timestamps
- scheduling/service timestamps
- packet loss
- throughput
- starvation duration
- configurable URLLC scheduling deadline

Raw packet- and scheduling-level observations should be logged so percentile and fairness metrics can be calculated offline.

---

## 9. Formal Properties

### P1 — URLLC Deadline-Risk Safety

When URLLC backlog is nonzero, the oldest URLLC packet is close to its scheduling deadline, radio conditions permit useful transmission, and resources are available, the learned policy must not select an action classified as deadline-risky.

The deadline-risk classification must be based on the relationship among:

- URLLC backlog
- remaining scheduling budget
- radio condition
- resource allocation
- achievable service capacity

The property must not be reduced to a fixed arbitrary allocation percentage unless that percentage is justified by the modeled service requirement.

---

### P2 — Slice Starvation Protection

A persistently backlogged slice should not continue receiving zero or insufficient service when usable resources are available.

Temporal starvation information may be represented through explicit state variables such as time since last service.

---

### P3 — Urgency Monotonicity

Holding other relevant state variables fixed, increasing URLLC urgency should not cause the learned policy to substantially reduce URLLC resource allocation.

---

### P4 — Robustness to Measurement Perturbation

Small perturbations in measured state variables, such as SINR or queue estimates, should not cause a safe scheduling decision to transition to a clearly unsafe allocation.

---

### P5 — No Idle Allocation Under Demand

When traffic is waiting and resources are available, the policy should not select the all-zero resource-allocation action.

---

## 10. Formal Verification Scope

Formal claims must be stated precisely.

The project must not claim that the entire O-RAN system or scheduler is universally formally verified.

Claims should use language such as:

"The trained neural policy satisfies property P over input region R under the encoded state, action-mapping, and model assumptions."

---

## 11. Counterexample-Guided Workflow

The required workflow is:

Training
→ conventional evaluation
→ ONNX export
→ formal verification
→ counterexample discovery
→ simulator reproduction
→ QoS consequence measurement
→ counterexample-guided retraining
→ re-evaluation
→ formal re-verification

A strong result is a property transition such as:

M2: SAT
→ counterexample reproduction
→ repair
→ M3: UNSAT

while maintaining competitive QoS performance.

---

## 12. Experimental Traffic Conditions

All comparable learning models must be evaluated under the same traffic and radio conditions.

Include:

- low load
- moderate load
- high load
- URLLC burst traffic
- eMBB-heavy traffic
- mMTC-heavy traffic
- simultaneous congestion
- radio-quality variation
- overload where demand exceeds available capacity

Multiple random seeds must be used.

---

## 13. Control-Timescale Experiments

The relationship between DRL control-loop timing and QoS assurance should be evaluated.

Candidate control periods include:

- 0.25 ms
- 0.5 ms
- 1 ms
- 5 ms
- 10 ms
- 100 ms

The original baseline configuration remains unchanged.

The proposed scheduler may use a faster control interval where required by the experimental scheduling deadline.

For each tested control interval measure:

- URLLC deadline miss ratio
- tail scheduling delay
- throughput
- PRB utilization
- formal verification result
- number of discovered counterexamples
- control/inference overhead

---

## 14. dApp Target

The final runtime architecture is:

RAN measurements
→ dApp state builder
→ assured DRL policy
→ selected allocation action
→ deterministic PRB allocation mapping
→ OAI scheduling/control interface

The dApp implementation should separate:

- measurement collection
- state construction
- model inference
- action interpretation
- control delivery
- fallback behavior
- logging

The simulator-trained state definition must map to measurements that can realistically be obtained in the O-RAN/OAI implementation.

---

## 15. Reproducibility Requirements

The repository must preserve:

- configuration files
- random seeds
- model architecture
- action definitions
- training parameters
- verification domains
- verification properties
- trained model identifiers
- experiment outputs
- counterexamples
- repair datasets or generation procedures

Major experimental conclusions must be reproducible from version-controlled code and configuration.

---

## 16. Scientific Constraints

The project will not weaken the research design merely to simplify implementation.

In particular:

- the original model remains unchanged as a baseline,
- formal verification is independent from ordinary simulation testing,
- formal properties must correspond to meaningful network behavior,
- counterexamples must be reproduced in the network simulator,
- policy repair must be evaluated for performance cost,
- comparisons must keep architecture and environment controlled where required,
- QoS metrics must be derived from traceable simulator events,
- dApp feasibility must be demonstrated rather than assumed.
