"""Stage 00: deterministic lifecycle, parity, and alignment controls."""

from __future__ import annotations

from typing import Any

from .compare import action_delta, within
from .fixtures import observation_batch
from .identity import IdentityProbeModel, observation_scalar

DEFAULT_CONFIG = {"env_cfg_type": "arx_x5", "action_type": "joint", "chunk_size": 2}


def run_contract_probe(batch_size: int = 10, atol: float = 1e-7) -> dict[str, Any]:
    observations = observation_batch(batch_size)

    singles = []
    for obs in observations:
        model = IdentityProbeModel(DEFAULT_CONFIG)
        model.reset()
        model.update_obs(obs)
        singles.append(model.get_action())

    batch_model = IdentityProbeModel(DEFAULT_CONFIG)
    batch_model.reset()
    batch_model.update_obs_batch(observations)
    indices = [int(obs["env_idx"]) for obs in observations]
    batched = batch_model.get_action_batch(indices)
    parity_deltas = [
        action_delta(single, batch)
        for single, batch in zip(singles, batched, strict=True)
    ]

    reversed_indices = list(reversed(indices))
    reordered = batch_model.get_action_batch(reversed_indices)
    expected = {env_idx: singles[position] for position, env_idx in enumerate(indices)}
    ordering_deltas = [
        action_delta(expected[env_idx], action)
        for env_idx, action in zip(reversed_indices, reordered, strict=True)
    ]

    first_joint_values = [float(observation_scalar(obs)) for obs in observations]
    distinct = len(set(first_joint_values)) == len(first_joint_values)
    parity_pass = all(within(delta, atol) for delta in parity_deltas)
    ordering_pass = all(within(delta, atol) for delta in ordering_deltas)

    checks = {
        "single_batch_parity": {
            "status": "pass" if parity_pass else "fail",
            "max_abs": max(delta["max_abs"] or 0.0 for delta in parity_deltas),
        },
        "batch_ordering": {
            "status": "pass" if ordering_pass else "fail",
            "requested_env_idx": reversed_indices,
            "max_abs": max(delta["max_abs"] or 0.0 for delta in ordering_deltas),
        },
        "nontrivial_identity": {
            "status": "pass" if distinct else "fail",
            "distinct_signatures": len(set(first_joint_values)),
            "signature_range": [min(first_joint_values), max(first_joint_values)],
        },
    }
    passed = all(check["status"] == "pass" for check in checks.values())
    return {
        "probe": "00_contract",
        "status": "pass" if passed else "fail",
        "atol": atol,
        "batch_size": batch_size,
        "checks": checks,
        "claim_boundary": "contract-level parity and env_idx alignment for deterministic fixtures only",
    }
