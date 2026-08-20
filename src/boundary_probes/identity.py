"""Deterministic XPolicyLab-compatible identity policy.

The policy converts observable semantics into small, nonzero actions. It is a
contract instrument, not a controller for a simulator or physical robot.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PINNED_ROBOT_INFO: dict[str, dict[str, list[int]]] = {
    # XPolicyLab utils/robot/_robot_info.json at the pinned upstream revision.
    "arx_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
}


def _robot_info(model_cfg: Mapping[str, Any]) -> dict[str, list[int]]:
    override = model_cfg.get("robot_action_dim_info")
    if override is not None:
        return {
            "arm_dim": [int(value) for value in override["arm_dim"]],
            "ee_dim": [int(value) for value in override["ee_dim"]],
        }

    env_cfg_type = str(model_cfg.get("env_cfg_type", "arx_x5"))
    if env_cfg_type in PINNED_ROBOT_INFO:
        return {
            key: list(values) for key, values in PINNED_ROBOT_INFO[env_cfg_type].items()
        }

    try:
        from XPolicyLab.utils.process_data import get_robot_action_dim_info
    except ImportError as exc:
        raise ValueError(
            f"Unknown env_cfg_type {env_cfg_type!r}; install XPolicyLab or provide "
            "model_cfg['robot_action_dim_info']."
        ) from exc
    return get_robot_action_dim_info(env_cfg_type)


def _instruction(obs: Mapping[str, Any]) -> Any:
    if "instruction" in obs:
        return obs["instruction"]
    return obs.get("instructions", "")


def _numeric_summary(value: Any) -> list[Any]:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or array.size == 0:
        return []
    real = np.real(array.astype(np.complex128, copy=False)).astype(np.float64)
    return [
        list(array.shape),
        float(real.mean()),
        float(real.std()),
        float(real.min()),
        float(real.max()),
    ]


def semantic_fingerprint(obs: Mapping[str, Any]) -> str:
    """Hash stable instruction, RGB-summary, state-summary, and identity fields."""

    cameras: list[list[Any]] = []
    for name, camera in sorted(obs.get("vision", {}).items()):
        if isinstance(camera, Mapping) and "color" in camera:
            cameras.append([str(name), _numeric_summary(camera["color"])])

    state: list[list[Any]] = []
    for name, value in sorted(obs.get("state", {}).items()):
        summary = _numeric_summary(value)
        if summary:
            state.append([str(name), summary])

    payload = {
        "instruction": _instruction(obs),
        "vision": cameras,
        "state": state,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def observation_scalar(
    obs: Mapping[str, Any], env_idx: int | None = None
) -> np.float32:
    """Return a compact signature whose first six decimals expose env identity."""

    digest = semantic_fingerprint(obs)
    semantic_jitter = (int(digest[:8], 16) % 1000) * 1e-9
    identity = int(obs.get("env_idx", 0) if env_idx is None else env_idx)
    return np.float32(0.043957 + identity * 0.001 + semantic_jitter)


class IdentityProbeModel:
    """Minimal model implementing XPolicyLab's policy-side lifecycle."""

    def __init__(self, model_cfg: Mapping[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg.get("action_type", "joint"))
        if self.action_type not in {"joint", "ee"}:
            raise ValueError("action_type must be 'joint' or 'ee'")
        self.robot_action_dim_info = _robot_info(model_cfg)
        if len(self.robot_action_dim_info["arm_dim"]) != len(
            self.robot_action_dim_info["ee_dim"]
        ):
            raise ValueError("arm_dim and ee_dim must describe the same number of arms")
        self.chunk_size = int(model_cfg.get("chunk_size", 2))
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._single_obs: Mapping[str, Any] | None = None
        self._batch_obs: dict[int, Mapping[str, Any]] = {}
        self._batch_order: list[int] = []
        self._batch_has_explicit_ids = False

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self._single_obs = obs

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> None:
        explicit = ["env_idx" in obs for obs in obs_list]
        if any(explicit) and not all(explicit):
            raise ValueError(
                "a batch must either include env_idx for every observation or for none"
            )
        indexed: dict[int, Mapping[str, Any]] = {}
        order: list[int] = []
        for position, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", position))
            if env_idx in indexed:
                raise ValueError(f"duplicate env_idx in batch: {env_idx}")
            indexed[env_idx] = obs
            order.append(env_idx)
        self._batch_obs = indexed
        self._batch_order = order
        self._batch_has_explicit_ids = all(explicit)

    def _action_chunk(
        self, obs: Mapping[str, Any], env_idx: int | None = None
    ) -> list[dict[str, np.ndarray]]:
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms == 1:
            arm_keys = ["arm_joint_state" if self.action_type == "joint" else "ee_pose"]
            ee_keys = ["ee_joint_state"]
        elif num_arms == 2:
            arm_keys = (
                ["left_arm_joint_state", "right_arm_joint_state"]
                if self.action_type == "joint"
                else ["left_ee_pose", "right_ee_pose"]
            )
            ee_keys = ["left_ee_joint_state", "right_ee_joint_state"]
        else:
            raise NotImplementedError(f"unsupported number of arms: {num_arms}")

        base = observation_scalar(obs, env_idx)
        actions: list[dict[str, np.ndarray]] = []
        for step in range(self.chunk_size):
            action: dict[str, np.ndarray] = {}
            for arm_number, (arm_key, ee_key) in enumerate(
                zip(arm_keys, ee_keys, strict=True)
            ):
                arm_dim = (
                    self.robot_action_dim_info["arm_dim"][arm_number]
                    if self.action_type == "joint"
                    else 7
                )
                offsets = np.arange(arm_dim, dtype=np.float32) * np.float32(1e-5)
                action[arm_key] = (
                    np.full(arm_dim, base + np.float32(step * 1e-4), dtype=np.float32)
                    + offsets
                )
                action[ee_key] = np.full(
                    self.robot_action_dim_info["ee_dim"][arm_number],
                    base * np.float32(0.1) + np.float32(step * 1e-5),
                    dtype=np.float32,
                )
            actions.append(action)
        return actions

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._single_obs is None:
            raise RuntimeError("get_action() called before update_obs()")
        return self._action_chunk(self._single_obs)

    def get_action_batch(
        self, env_idx_list: Sequence[int] | None = None
    ) -> list[list[dict[str, np.ndarray]]]:
        indices = (
            self._batch_order
            if env_idx_list is None
            else [int(value) for value in env_idx_list]
        )
        if not self._batch_has_explicit_ids:
            if len(indices) != len(self._batch_order):
                raise ValueError("env_idx_list length must match the observation batch")
            observations = [self._batch_obs[position] for position in self._batch_order]
            return [
                self._action_chunk(obs, env_idx)
                for obs, env_idx in zip(observations, indices, strict=True)
            ]
        missing = [env_idx for env_idx in indices if env_idx not in self._batch_obs]
        if missing:
            raise KeyError(f"no observation for env_idx values: {missing}")
        return [
            self._action_chunk(self._batch_obs[env_idx], env_idx) for env_idx in indices
        ]

    def reset(self) -> None:
        self._single_obs = None
        self._batch_obs.clear()
        self._batch_order.clear()
        self._batch_has_explicit_ids = False


# XPolicyLab discovers a class named Model in policy/<POLICY>/model.py.
Model = IdentityProbeModel
