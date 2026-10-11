# AssuredQoS State-Domain Provenance

## Purpose

This note records the provenance, units, ordering, clipping rules,
and interpretation of the AssuredQoS policy observation space.

It documents the state contract after the post-buffer-fix backlog
calibration and the per-slice normalization-diagnostics update.

## Relevant source commits

Backlog calibration source freeze:

`8679ce868d6ca3dcb4a773d0d837e113ed0d5f81`

Calibrated byte-backlog state implementation:

`45157b08fcf611c19b8b8cde17aad1df04cc9980`

Primary experiment contract:

`a64c25b6de1bc8a7db1788f8a3c7244c8ba91fde`

Per-slice normalization diagnostics:

`526a5509f9e154d4d80c4d222d790f518d8e30cc`

## Backlog feature

Backlog is measured in scheduler-visible DL bearer bytes.

For each slice, the feature is the sum of packet bytes currently
waiting in the first DL bearer queue of each UE.

It excludes:

- application-buffer bytes;
- transport-block bytes already committed or in flight;
- retransmission transport blocks;
- unrelated auxiliary queues.

The frozen empirical normalization caps are:

| Slice | Cap |
| --- | ---: |
| eMBB | 818,176 bytes |
| URLLC | 1,884,160 bytes |
| mMTC | 6,244,352 bytes |

For raw backlog `b` and slice-specific cap `C`, the normalized
feature is:

`clip(b, 0, C) / C`

The result is therefore bounded to `[0, 1]`.

These values are empirical normalization caps derived under the
frozen simulator calibration protocol. They are not universal
physical limits on a 5G or O-RAN system.

## Backlog-calibration provenance

Executed post-buffer-fix calibration protocol SHA-256:

`6dcd4d2d7b2e91d549253e95c087c1165a84989440075af95512a1546cdddc50`

Calibration launcher SHA-256:

`ce0ab811d78908bf4bc5cbadf74683b5c72dbd51c46d1553774d679b454ee365`

Combined raw calibration-data SHA-256 recorded by the manifest:

`3500aa81fb3efad23078764d36ceede54050b7b022887af7801c871b4570870d`

Calibration scenarios:

- BASELINE
- CONGESTED
- URLLC_HIGH
- SIMULTANEOUS_HIGH

Calibration seeds:

- 7
- 17
- 27

Each scenario/seed run contributed 10,000 active-period samples
per slice at 1-ms controller boundaries before the inter-slice
scheduling action.

For each run, the 99.9th percentile was calculated with:

`numpy.quantile(..., 0.999, method="higher")`

The slice cap was selected as the maximum run-level 99.9th
percentile for that slice and rounded upward to the next KiB.

The protocol's cap-conformance criterion was evaluated on the
same runs used to select the caps. It is therefore a same-data
calibration conformance check and not independent held-out
validation.

## SINR feature

The primary SINR normalization domain is:

- minimum: 0 dB
- maximum: 35 dB

For raw SINR `s`, the normalized feature is:

`(clip(s, 0, 35) - 0) / (35 - 0)`

The result is bounded to `[0, 1]`.

The 35-dB upper bound is tied to the simulator's current FDD MCS
table range used by the AssuredQoS experiments. It is a simulator
state-domain choice, not a universal radio-physics bound.

Values below 0 dB or above 35 dB are clipped for policy input and
are counted by normalization diagnostics.

## URLLC urgency feature

URLLC urgency is based on the maximum head-of-line waiting time
among unique URLLC packets that have not yet received their first
scheduling decision.

Candidate unscheduled packets are inspected in both:

- the application buffer;
- the first DL bearer buffer.

Already-scheduled packets, including residual fragments of a
packet that has already received its first scheduling decision,
are excluded.

For maximum unscheduled URLLC HOL delay `h` and scheduling
deadline `d`, raw urgency is:

`h / d`

The policy feature is:

`clip(h / d, 0, 1)`

The primary scheduling deadline is 1 ms.

Therefore:

- urgency 0 means no current unscheduled HOL waiting time;
- urgency between 0 and 1 represents progress toward the
  scheduler-level deadline;
- urgency 1 represents the deadline or any larger raw ratio after
  clipping.

The 1-ms deadline is an experimental scheduler-level requirement.
It must not be presented as a 3GPP end-to-end URLLC packet-delay
budget.

## State ordering and dimensions

M1 input dimension: 6

Feature order:

1. normalized eMBB backlog
2. normalized URLLC backlog
3. normalized mMTC backlog
4. normalized mean eMBB SINR
5. normalized mean URLLC SINR
6. normalized mean mMTC SINR

M2 input dimension: 7

M2 contains the six M1 features in the same order and appends:

7. normalized URLLC urgency

M3 input dimension: 7

M3 uses the same observation structure and ordering as M2.

## Normalization diagnostics

The scheduler records raw backlog and SINR values using the same
underlying measurements and configured domains used to build the
policy state.

Diagnostics retain aggregate counters for backward compatibility
and additionally maintain per-slice backlog and SINR clipping
statistics for:

- eMBB
- URLLC
- mMTC

Urgency has one diagnostic counter because urgency is a
URLLC-specific state feature.

These diagnostics are observational instrumentation. They do not
modify the state supplied to the policy or alter action selection.

## Interpretation rule

Normalization clipping defines the bounded numerical input domain
seen by the neural policy.

It must not be interpreted as proof that the physical system
cannot exceed the raw configured domains.

Formal verification performed later in AssuredQoS will therefore
apply to the explicitly stated neural-network input domain and
property assumptions, not to the entire simulator, scheduler
implementation, or O-RAN system.
