"""Stage 01: compare post-reset output with a fresh-model baseline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .compare import action_delta, within
from .fixtures import observation
from .identity import IdentityProbeModel, observation_scalar


class Policy(Protocol):
    def reset(self) -> None: ...
    def update_obs(self, obs: Mapping[str, Any]) -> None: ...
    def get_action(self) -> Sequence[Mapping[str, Any]]: ...


class StatefulCanaryModel:
    """Known-stateful control; `clear_on_reset=False` injects a reset defect."""

    def __init__(self, clear_on_reset: bool):
        self.clear_on_reset = clear_on_reset
        self.hidden = np.float32(0.0)
        self.current = np.float32(0.0)

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self.current = observation_scalar(obs)
        self.hidden = np.float32(self.hidden * np.float32(0.5) + self.current)

    def get_action(self) -> list[dict[str, np.ndarray]]:
        return [{"arm_joint_state": np.array([self.hidden], dtype=np.float32)}]

    def reset(self) -> None:
        self.current = np.float32(0.0)
        if self.clear_on_reset:
            self.hidden = np.float32(0.0)


def probe_factory(
    factory: Callable[[], Policy],
    *,
    priming_obs: Mapping[str, Any] | None = None,
    target_obs: Mapping[str, Any] | None = None,
    atol: float = 1e-7,
) -> dict[str, Any]:
    prime = observation(env_idx=3, variant=1) if priming_obs is None else priming_obs
    target = observation(env_idx=7, variant=2) if target_obs is None else target_obs

    reused = factory()
    reused.reset()
    reused.update_obs(prime)
    reused.get_action()
    reused.reset()
    reused.update_obs(target)
    post_reset = reused.get_action()

    fresh = factory()
    fresh.reset()
    fresh.update_obs(target)
    fresh_action = fresh.get_action()

    delta = action_delta(post_reset, fresh_action)
    passed = within(delta, atol)
    return {
        "status": "pass" if passed else "fail",
        "atol": atol,
        "max_abs": delta["max_abs"],
        "same_shape": delta["same_shape"],
        "criterion": "post-reset target action equals fresh-model target action",
    }


def run_reset_probe(atol: float = 1e-7) -> dict[str, Any]:
    identity_target = probe_factory(
        lambda: IdentityProbeModel(
            {"env_cfg_type": "arx_x5", "action_type": "joint", "chunk_size": 2}
        ),
        atol=atol,
    )
    correct = probe_factory(lambda: StatefulCanaryModel(clear_on_reset=True), atol=atol)
    injected_defect = probe_factory(
        lambda: StatefulCanaryModel(clear_on_reset=False), atol=atol
    )
    passed = (
        identity_target["status"] == "pass"
        and correct["status"] == "pass"
        and injected_defect["status"] == "fail"
    )
    return {
        "probe": "01_reset_leakage",
        "status": "pass" if passed else "fail",
        "atol": atol,
        "checks": {
            "identity_probe": identity_target,
            "correct_reset": correct,
            "known_bad_reset": {
                **injected_defect,
                "status": "pass" if injected_defect["status"] == "fail" else "fail",
                "detector_observed": injected_defect["status"],
            },
        },
        "claim_boundary": "sensitivity of the reset protocol to history-dependent action leakage",
    }
