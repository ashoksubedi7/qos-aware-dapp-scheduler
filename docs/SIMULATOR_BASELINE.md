# Simulator Baseline

## Source

The initial simulator code in this repository was copied from:

~/formally-verified-dqn-oran-scheduler/formal-verification/py5chesim

The copied simulator is preserved initially without intentional behavioral changes.

## Purpose

This baseline provides the reference implementation from which the AssuredQoS simulator instrumentation will be developed.

The baseline will be validated before adding:

- URLLC head-of-line delay measurement
- configurable scheduling deadlines
- deadline-miss tracking
- starvation tracking
- structured packet-level logging
- structured scheduler-decision logging
- revised PRB-utilization measurement
- QoS-aware DRL state variables

## Baseline DRL State

The original DQN uses three queue/backlog inputs:

1. eMBB backlog
2. URLLC backlog
3. mMTC backlog

## Baseline Action Space

The original scheduler uses 22 discrete inter-slice allocation actions.

Each action maps to a deterministic resource-allocation weight vector across:

- eMBB
- URLLC
- mMTC

## Baseline Preservation Rule

The original behavior will not be silently modified.

Any change that affects traffic generation, scheduling, timing, reward, action mapping, packet service, or metric calculation must be introduced in a separate commit and documented.
