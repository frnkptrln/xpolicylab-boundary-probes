# 03 — Action transforms

Status: executable offline coordinate-contract probe. No learned policy or
simulator was run. The pinned upstream revision remains unchanged.

```bash
python -m boundary_probes action --output artifacts/action-transforms.json
python -m boundary_probes all --output artifacts/all.json
pytest
```

Two environments each supply three ordered six-joint and seven-value pose
fixtures. Joints are canonical radians; the alternate representation uses a
signed permutation and degrees. Pose fixtures explicitly use metres, xyz plus
unit-quaternion xyzw, active rotations; the alternate frame applies a fixed
90-degree z rotation and translation `(0.3,-0.2,0.5)`. These are probe
conventions, not a claim about every upstream controller.

The 12 required controls cover round trips, single/batch consistency, quaternion
sign equivalence, nontrivial encodings and five deliberately wrong decoders
(missing joint order, missing units, missing translation, swapped environments,
reversed action time). Shape, finiteness and unit-quaternion checks reject
invalid witnesses. Comparison uses coordinate error and sign-invariant quaternion
chord error at `1e-10`; raw vector equality is deliberately not the criterion.

The independent analytic unit fixtures include a quarter-turn of the unit
x-axis and a known signed joint permutation. A substituted broken decoder
must make the complete probe fail.

On 2026-09-08 all 12 Stage-03 controls passed; the full repository suite passed
26 tests and the CLI reported pass for Stages 00–03. Results are deterministic
except for the run-record timestamp and environment metadata.

Official controller-specific action conventions and an official client/server
trace remain necessary for integration. Stage 04 still needs a pinned learned
policy and explicit compute authorization. Coordinate equivalence alone says
nothing about task competence, timing, physical safety or sim-to-real transfer.
