# 00 — Contract probe

## Hypothesis

For a deterministic observation-sensitive adapter, single and batched calls
produce the same action chunk for the same observation and environment
identity. Reordering `env_idx_list` reorders outputs without changing which
action belongs to which environment.

## Controls

- ten observations with `env_idx` 0–9;
- nonzero signatures so equality is not vacuous;
- single-versus-batch parity at absolute tolerance `1e-7`;
- reversed index request to test identity alignment;
- duplicate `env_idx` rejection.

Runtime image decoding is intentionally outside this local unit probe. The
XPolicyLab server guarantees decoded RGB arrays at the model boundary; the
official encoded debug-client run is the corresponding integration check.

## Interpretation

A pass supports only contract-level parity for these fixtures. It says
nothing about task performance or physical safety.
