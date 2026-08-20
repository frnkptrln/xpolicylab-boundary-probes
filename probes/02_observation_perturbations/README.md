# 02 — Observation perturbations

## Question

Which changes at the decoded-observation boundary should preserve an action,
and which changes can the deterministic instrument actually detect?

The probe declares the relation before comparing outputs. It does not treat
every difference as a robustness defect.

| Case | Declared relation | Required control behavior |
| --- | --- | --- |
| Camera dictionary order reversed | equivalent | invariant |
| Identical observation delivered twice | equivalent | invariant |
| Each configured camera removed in turn | non-equivalent | detectable |
| Each configured camera replaced by a fixed stale-source fixture | non-equivalent | detectable |

The missing- and stale-camera cases are sensitivity controls for the identity
instrument. A learned task policy is not automatically wrong if it produces a
similar action for either case; its result would need task-specific analysis.
The stale source is a deterministic contrast fixture, not a sample from a real
temporal stream.

## Known-blind canary

The suite includes a deliberately vision-blind policy. It must pass the two
equivalence controls and fail both vision-sensitivity controls. This establishes
that a passing identity probe is not caused by an observation-insensitive
comparison.

## Why latency is not emulated here

The pinned policy lifecycle is synchronous and receives decoded arrays. It
contains no capture timestamp, arrival timestamp, queue, or asynchronous
delivery behavior. Sleeping locally or attaching an invented timestamp would
not test transport latency. The run record therefore reports latency as
`not_run` and requires timing evidence from the official client/server debug
integration.

## Run

```bash
python -m boundary_probes observation
```

## Interpretation

A pass supports fixture-level invariance and detector sensitivity at this
policy boundary. It does not establish end-to-end timing robustness, tolerance
of camera loss, semantic robustness, simulator performance, task competence,
sim-to-real transfer, or physical safety.
