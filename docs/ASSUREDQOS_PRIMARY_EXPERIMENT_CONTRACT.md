# AssuredQoS Primary Experiment Contract

## Purpose

This document prospectively records the primary AssuredQoS
experimental assumptions that are already derived, calibrated,
or established by the simulator implementation.

Its purpose is to prevent post-hoc adjustment of important
experimental quantities after primary model results are observed.

This contract does not freeze training-budget parameters that
belong to the separate training-protocol work.

## Source baseline

Primary configuration baseline commit:

`45157b08fcf611c19b8b8cde17aad1df04cc9980`

This commit contains the frozen calibrated byte-backlog state
contract and passed the complete repository regression:

- 320 tests passed
- 0 tests failed
- 12 dependency deprecation warnings

## Cell and controller configuration

Primary simulated radio configuration:

- Frequency range: FR1
- Bandwidth: 10 MHz
- Common reference-resource budget: 52 15-kHz reference-PRB units
- AssuredQoS control interval: 1 ms
- Radio update interval: 10 ms
- Per-UE bearer-buffer capacity: 81,920 bytes
- Post-traffic drain duration: 1,000 ms

The 1-ms value is the scheduler/controller period used by this
experiment. It must not be described as a 3GPP end-to-end packet
delay budget.

The 1,000-ms drain duration was selected by the frozen
post-buffer-fix drain calibration. Under the frozen comparison
criterion, 1,000 ms was stable relative to the 2,000-ms reference
drain. It provides a post-traffic period for scheduling,
retransmission, and packet completion before final lifecycle
metrics are computed; it is not a guarantee that every generated
packet completes.

## Slice numerology

The simulator derives FR1 numerology from the configured slice
delay requirement.

Primary slice requirements are:

| Slice | Delay requirement | SCS | Reference factor |
| --- | ---: | ---: | ---: |
| eMBB | 20 ms | 15 kHz | 1 |
| URLLC | 5 ms | 60 kHz | 4 |
| mMTC | 20 ms | 15 kHz | 1 |

Therefore the primary mixed-numerology reference-factor vector is:

`(eMBB, URLLC, mMTC) = (1, 4, 1)`

The resource mapper interprets DQN actions in common
reference-resource units and deterministically converts them to
integer per-slice PRBs.

For example, with 52 reference PRBs and requested weights
`(60, 20, 20)`, the current deterministic mapper produces:

- slice PRBs: `(30, 3, 10)`
- reference-resource use: `(30, 12, 10)`
- total reference-resource use: `52`

This mixed-numerology representation is a simulator abstraction
and must be described as such.

## Action space

The DQN action space contains 22 discrete actions:

- one explicit idle action `(0, 0, 0)`;
- all 21 non-idle three-slice percentage allocations on the
  20-percentage-point simplex.

Every non-idle action sums to 100 percent before deterministic
numerology-aware integer conversion.

The 20-percent discretization is the primary action resolution.
It is not claimed to be optimal. If action-resolution sensitivity
is studied, that analysis must be declared separately.

## Backlog state domain

Scheduler-visible backlog is measured in bytes waiting in the
first DL bearer queue of each UE.

It excludes:

- application-buffer bytes;
- already committed or in-flight transport blocks;
- retransmission transport blocks;
- unrelated auxiliary queues.

Frozen empirical normalization caps are:

| Slice | Backlog cap |
| --- | ---: |
| eMBB | 818,176 bytes |
| URLLC | 1,884,160 bytes |
| mMTC | 6,244,352 bytes |

Values above the corresponding cap are clipped by normalization.

These are empirical calibration caps, not universal physical
limits.

The calibration conformance check was performed on the same runs
used to select the caps and therefore must not be described as
held-out validation.

## SINR state domain

Primary normalization bounds are:

- minimum: 0 dB
- maximum: 35 dB

Defined radio scenarios include:

- STATIC_POOR: 6.0 dB
- STATIC_MODERATE: 18.5 dB
- STATIC_GOOD: 27.5 dB
- STEP_DEGRADATION
- STEP_RECOVERY
- FLUCTUATING

Step scenarios change at the midpoint of a simulation.

The fluctuating process is seeded deterministically for each
experiment/slice/UE and clipped to the 0--35 dB domain.

## Traffic-model semantics

The field currently named `arrival_rate` in the traffic-profile
code must not be interpreted as packets per second.

For the Uniform arrival model used by the frozen traffic
scenarios, its value `a` is passed to the packet generator as an
inter-arrival-time upper bound:

`T_interarrival ~ Uniform(0, a)`

Therefore:

`E[T_interarrival] = a / 2`

in simulator time units (milliseconds in these experiments).

A smaller configured value therefore represents more frequent
packet generation.

The existing field name is retained for compatibility with the
executed calibration code, but publications and experiment
documentation must describe it as the Uniform inter-arrival upper
bound.

### Frozen traffic scenarios

| Scenario | Slice | DL UEs | Packet bytes | Uniform inter-arrival upper bound |
| --- | --- | ---: | ---: | ---: |
| BASELINE | eMBB | 4 | 1200 | 1.2 ms |
| BASELINE | URLLC | 8 | 300 | 0.6 ms |
| BASELINE | mMTC | 30 | 100 | 10.0 ms |
| CONGESTED | eMBB | 8 | 1200 | 0.8 ms |
| CONGESTED | URLLC | 16 | 300 | 0.4 ms |
| CONGESTED | mMTC | 60 | 100 | 5.0 ms |
| URLLC_HIGH | eMBB | 4 | 1200 | 1.2 ms |
| URLLC_HIGH | URLLC | 24 | 300 | 0.2 ms |
| URLLC_HIGH | mMTC | 30 | 100 | 10.0 ms |
| SIMULTANEOUS_HIGH | eMBB | 10 | 1200 | 0.6 ms |
| SIMULTANEOUS_HIGH | URLLC | 20 | 300 | 0.3 ms |
| SIMULTANEOUS_HIGH | mMTC | 80 | 100 | 4.0 ms |

The configured packet-size distribution is Pareto for eMBB and
Constant for URLLC and mMTC.

## QoS and reward contract

The scheduler uses a 1-ms URLLC scheduling deadline.

This is an experimental scheduler-level deadline and must not be
presented as an end-to-end 5G URLLC latency guarantee.

The primary reward is:

`R = 0.30*S_URLLC + 0.25*S_eMBB + 0.15*S_mMTC
     + 0.10*U - 0.15*D - 0.05*Z`

where:

- `S_URLLC` is URLLC packet-demand service ratio;
- `S_eMBB` is achieved eMBB throughput normalized by the
  configured eMBB throughput target;
- `S_mMTC` is mMTC packet-demand service ratio;
- `U` is PRB utilization;
- `D` is URLLC deadline-miss ratio;
- `Z` is the normalized starvation penalty.

All reward components are bounded to `[0, 1]`.

The primary starvation threshold is 10 ms.

These reward weights are prospectively fixed for the primary
comparison. They must not be retuned after primary evaluation
results are observed.

A separately declared reward-sensitivity study may be performed,
but it must remain distinct from the primary result.

## Model comparison contract

M1:

- input dimension: 6
- hidden layers: 64, 64
- output dimension: 22
- state: backlog plus slice SINR

M2:

- input dimension: 7
- hidden layers: 64, 64
- output dimension: 22
- state: M1 state plus URLLC deadline urgency

M3:

- input dimension: 7
- hidden layers: 64, 64
- output dimension: 22
- architecture matched to M2
- represents counterexample-guided repair of M2

M1 and M2 use the same action space and hidden-layer capacity.
M3 must remain architecturally matched to M2 so that the repair
experiment does not obtain an architectural advantage.

## Calibration-seed exclusion

Seeds:

`7, 17, 27`

were used for calibration and are reserved from primary
training/model-selection/final-evaluation seed sets.

They must not be reused as primary study seeds.

Training, model-selection, and final-evaluation seeds will be
frozen separately in the training protocol before primary
training begins.

## Not yet frozen by this contract

The following remain intentionally deferred to the training
protocol:

- training seeds;
- checkpoint/model-selection seeds;
- final evaluation seeds;
- total training update budget;
- checkpoint interval;
- checkpoint-selection schedule;
- exact training scenario schedule;
- DQN training hyperparameter freeze;
- formal-verification property domains;
- reward/action-resolution sensitivity configurations.

These quantities must be fixed before their corresponding
primary results are inspected.

## Anti-tuning rule

No parameter in this primary contract may be changed because a
later result is unfavorable.

Any later modification must be:

1. motivated independently of the observed final result;
2. recorded as a new protocol/version;
3. evaluated separately from the prospectively frozen primary
   experiment.
