# XPolicyLab boundary probes

Independent, falsifiable experiments at the boundary between an XPolicyLab
policy adapter and its evaluation environment.

This repository is not affiliated with or maintained by the XPolicyLab
project. It contains no learned policy, simulator result, or physical-robot
claim.

## What exists now

| Stage | Question | Status |
| --- | --- | --- |
| `00_contract` | Does a deterministic, observation-sensitive adapter preserve single/batch semantics and `env_idx` alignment? | runnable |
| `01_reset_leakage` | Does an episode reset remove history that could alter the next episode? | runnable |
| `02_observation_perturbations` | What changes under camera order, availability, latency, or stale frames? | planned |
| `03_action_transforms` | Are equivalent action-coordinate transforms equivalent after canonicalization? | planned |
| `04_policy_divergence` | Does a small learned policy turn boundary changes into behavioral divergence? | planned |

The deterministic identity probe is deliberately nonzero and
observation-sensitive. A zero-action demo can prove that plumbing runs, but
not that observations or batch identities arrive intact.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
python -m boundary_probes all --output artifacts/latest.json
```

The command exits nonzero if a required control fails. Generated run records
go under `artifacts/`, which is intentionally ignored by Git.

## XPolicyLab compatibility target

The adapter mirrors the current XPolicyLab lifecycle:

- `Model(model_cfg)`
- `update_obs(obs)` / `update_obs_batch(obs_list)`
- `get_action()` / `get_action_batch(env_idx_list=None)`
- argument-free `reset()`

Runtime camera colors are treated as already-decoded RGB arrays. Transport
decoding belongs to the XPolicyLab policy server, not to `model.py`.

The pinned reference is XPolicyLab commit
[`c07a09614dd44cc4a67483bcb9a82e7439d99926`](https://github.com/XPolicyLab/XPolicyLab/commit/c07a09614dd44cc4a67483bcb9a82e7439d99926),
inspected on 2026-08-20. See [`upstream.lock.json`](upstream.lock.json) and
[`docs/provenance.md`](docs/provenance.md).

## Claim boundary

A passing control supports only contract-level claims about the supplied
fixtures and adapter calls. It does **not** establish task competence,
semantic robustness, simulator performance, sim-to-real transfer, or
physical safety.

## Repository map

```text
adapters/identity_probe/  drop-in model entry point
probes/                   staged hypotheses and protocols
src/boundary_probes/      deterministic adapter and runners
tests/                    executable controls, including a known-bad canary
artifacts/                ignored local run records
```

## Relationship to systems-and-intelligence

The conceptual interpretation belongs in `systems-and-intelligence`. This
repository is the executable empirical apparatus: pinned upstream contracts,
fixtures, probes, and machine-readable results.

## License

MIT. See [`LICENSE`](LICENSE).
