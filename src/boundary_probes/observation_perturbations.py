"""Stage 02: observation-boundary invariance and sensitivity controls.

The executable cases distinguish declared semantic equivalence from detector
sensitivity. A missing or stale camera is not labelled a robustness failure:
it is a deliberately non-equivalent input that the deterministic identity
instrument must be able to notice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

import numpy as np

from .compare import action_delta, within
from .fixtures import observation
from .identity import IdentityProbeModel, observation_scalar


class Policy(Protocol):
    def reset(self) -> None: ...
    def update_obs(self, obs: Mapping[str, Any]) -> None: ...
    def get_action(self) -> Sequence[Mapping[str, Any]]: ...


PolicyFactory = Callable[[], Policy]
DEFAULT_CONFIG = {"env_cfg_type": "arx_x5", "action_type": "joint", "chunk_size": 2}
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SENSITIVITY_CASES = tuple(
    f"{factor}.{camera}"
    for factor in ("camera_unavailable", "stale_camera_frame")
    for camera in CAMERAS
)


class VisionBlindCanaryModel:
    """Known-blind control that deliberately discards the complete vision map."""

    def __init__(self) -> None:
        self._obs: Mapping[str, Any] | None = None

    def reset(self) -> None:
        self._obs = None

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self._obs = obs

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._obs is None:
            raise RuntimeError("get_action() called before update_obs()")
        without_vision = {**self._obs, "vision": {}}
        value = observation_scalar(without_vision)
        return [{"arm_joint_state": np.array([value], dtype=np.float32)}]


def _fresh_action(factory: PolicyFactory, obs: Mapping[str, Any]) -> Any:
    model = factory()
    model.reset()
    model.update_obs(obs)
    return model.get_action()


def _fresh_pair(
    factory: PolicyFactory,
    baseline: Mapping[str, Any],
    perturbed: Mapping[str, Any],
) -> tuple[Any, Any]:
    return _fresh_action(factory, baseline), _fresh_action(factory, perturbed)


def _repeated_pair(
    factory: PolicyFactory, baseline: Mapping[str, Any]
) -> tuple[Any, Any]:
    model = factory()
    model.reset()
    model.update_obs(baseline)
    first = model.get_action()
    model.update_obs(deepcopy(baseline))
    repeated = model.get_action()
    return first, repeated


def _reorder_cameras(obs: Mapping[str, Any]) -> dict[str, Any]:
    changed = deepcopy(obs)
    changed["vision"] = dict(reversed(list(changed["vision"].items())))
    return changed


def _remove_camera(obs: Mapping[str, Any], camera: str) -> dict[str, Any]:
    changed = deepcopy(obs)
    if camera not in changed["vision"]:
        raise KeyError(f"fixture has no camera named {camera!r}")
    del changed["vision"][camera]
    return changed


def _inject_stale_camera(
    current: Mapping[str, Any], stale_source: Mapping[str, Any], camera: str
) -> dict[str, Any]:
    changed = deepcopy(current)
    if camera not in changed["vision"] or camera not in stale_source["vision"]:
        raise KeyError(f"fixture has no camera named {camera!r}")
    changed["vision"][camera]["color"] = np.array(
        stale_source["vision"][camera]["color"], copy=True
    )
    return changed


def _evaluate_case(
    *,
    relation: str,
    requirement: str,
    left: Any,
    right: Any,
    atol: float,
    execution: str,
) -> dict[str, Any]:
    delta = action_delta(left, right)
    invariant = within(delta, atol)
    if requirement == "invariant":
        passed = invariant
    elif requirement == "detectable":
        passed = not invariant
    else:
        raise ValueError(f"unknown requirement: {requirement!r}")
    return {
        "status": "pass" if passed else "fail",
        "declared_relation": relation,
        "requirement": requirement,
        "observed": "invariant" if invariant else "detectable",
        "execution": execution,
        "atol": atol,
        "same_shape": delta["same_shape"],
        "max_abs": delta["max_abs"],
    }


def run_observation_matrix(
    factory: PolicyFactory, *, atol: float = 1e-7
) -> dict[str, Any]:
    """Run the executable Stage-02 cases for one policy factory."""

    baseline = observation(env_idx=4, variant=0)
    # Fixed detector-control contrast: under the pinned identity instrument,
    # replacing any one camera with variant 9 clears the declared tolerance.
    # It is not a sample from a real temporal process.
    stale_source = observation(env_idx=4, variant=9)
    reordered = _reorder_cameras(baseline)

    base_action, reordered_action = _fresh_pair(factory, baseline, reordered)
    first_delivery, repeated_delivery = _repeated_pair(factory, baseline)

    checks = {
        "camera_mapping_order": _evaluate_case(
            relation="equivalent",
            requirement="invariant",
            left=base_action,
            right=reordered_action,
            atol=atol,
            execution="fresh_instance_pair",
        ),
        "identical_frame_redelivery": _evaluate_case(
            relation="equivalent",
            requirement="invariant",
            left=first_delivery,
            right=repeated_delivery,
            atol=atol,
            execution="same_instance_sequential",
        ),
    }
    for camera in CAMERAS:
        unavailable = _remove_camera(baseline, camera)
        missing_base, unavailable_action = _fresh_pair(factory, baseline, unavailable)
        checks[f"camera_unavailable.{camera}"] = _evaluate_case(
            relation="non_equivalent",
            requirement="detectable",
            left=missing_base,
            right=unavailable_action,
            atol=atol,
            execution="fresh_instance_pair",
        )

        stale = _inject_stale_camera(baseline, stale_source, camera)
        stale_base, stale_action = _fresh_pair(factory, baseline, stale)
        checks[f"stale_camera_frame.{camera}"] = _evaluate_case(
            relation="non_equivalent",
            requirement="detectable",
            left=stale_base,
            right=stale_action,
            atol=atol,
            execution="fresh_instance_pair",
        )
    return {
        "status": "pass"
        if all(check["status"] == "pass" for check in checks.values())
        else "fail",
        "checks": checks,
        "deferred": {
            "delivery_latency": {
                "status": "not_run",
                "reason": (
                    "the synchronous decoded-array policy boundary carries no "
                    "capture time, arrival time, or asynchronous delivery behavior; "
                    "measure latency with the official client/server integration"
                ),
                "required_evidence": "official_debug_client_timing",
            }
        },
    }


def run_observation_probe(atol: float = 1e-7) -> dict[str, Any]:
    """Run Stage 02 with the identity target and a deliberately blind canary."""

    target = run_observation_matrix(
        lambda: IdentityProbeModel(DEFAULT_CONFIG), atol=atol
    )
    blind = run_observation_matrix(VisionBlindCanaryModel, atol=atol)
    observed_blind_failures = sorted(
        name for name, check in blind["checks"].items() if check["status"] == "fail"
    )
    blind_detected = observed_blind_failures == sorted(SENSITIVITY_CASES)
    canary = {
        "status": "pass" if blind_detected else "fail",
        "expected_failed_checks": sorted(SENSITIVITY_CASES),
        "observed_failed_checks": observed_blind_failures,
        "criterion": "the matrix rejects a policy that discards vision",
    }
    passed = target["status"] == "pass" and canary["status"] == "pass"
    return {
        "probe": "02_observation_perturbations",
        "status": "pass" if passed else "fail",
        "atol": atol,
        "checks": {**target["checks"], "known_blind_canary": canary},
        "deferred": target["deferred"],
        "claim_boundary": (
            "fixture-level invariance and detector sensitivity at the synchronous "
            "decoded-array policy boundary; not end-to-end timing robustness, task "
            "competence, simulator performance, or physical safety"
        ),
    }
