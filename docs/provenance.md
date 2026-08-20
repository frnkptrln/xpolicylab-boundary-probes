# Provenance and reconstruction notes

## Upstream reference

Inspected 2026-08-20 at XPolicyLab revision
`c07a09614dd44cc4a67483bcb9a82e7439d99926`:

- `model_template.py` for the lifecycle surface;
- `policy/demo_policy/model.py` for action keys and chunk shape;
- `policy/demo_policy/deploy.py` for call order and batch alignment;
- `utils/robot/_robot_info.json` for the pinned `arx_x5` dimensions.

The current contract says that runtime camera colors reach the model as
decoded RGB arrays, `reset()` takes no arguments, and
`get_action_batch(env_idx_list)` aligns output chunks with active environment
indices.

## Prior control experiment

The first private control experiment established the intended measurement
shape: nonzero deterministic actions derived from instruction, RGB statistics,
numeric state statistics, and `env_idx`; single/batch parity; ordering; reset;
and plain-versus-encoded debug runs.

This repository is a clean reconstruction from that experiment's recorded
behavior, not a claim of byte-for-byte source identity. Exact code and raw logs
from that run were not available during reconstruction.

## Deliberate changes

- Runtime model code accepts only decoded arrays; it does not reproduce image
  decoding inside `model.py` because that would violate the current upstream
  boundary.
- The local test suite separates model-contract controls from the official
  server transport integration check.
- The reset detector includes a known-bad stateful canary to establish that a
  passing target is not caused by an insensitive comparison.
